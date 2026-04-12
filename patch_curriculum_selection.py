"""
Patch-Level K-Medoids Curriculum Selection for CIFAR10
======================================================
Replaces single-image L2 distance with patch-based representativeness:

  1. Each 32x32 image is split into a 4x4 grid -> 16 patches of 8x8x3 = 192 dims each
  2. For each of the 16 spatial positions, KMeans finds K_patch cluster centers
     across all 50k patches at that position (the "patch vocabulary" per location)
  3. Each image gets a 16-dim feature: dist(patch_p, nearest_center_p) for p in 0..15
  4. Image patch-centrality score = mean of those 16 normalized distances
     -> low score  = patches are near the centroid cloud  = representative / important
     -> high score = patches are outliers                 = atypical / hard
  5. Per-class K-Medoids runs on the 16-dim patch-distance vectors
     -> medoids are the images whose PATCHES are most central to their positions

Usage:
    python patch_curriculum_selection.py
    python patch_curriculum_selection.py --k 5 --K_patch 20 --n_samples 5000
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# ── constants ────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

PATCH_GRID  = 4          # 4x4 grid
PATCH_SIZE  = 8          # 32 / 4 = 8 pixels per side
N_PATCHES   = PATCH_GRID * PATCH_GRID   # 16
PATCH_DIM   = 3 * PATCH_SIZE * PATCH_SIZE  # 192


# ─────────────────────────────────────────────────────────────────────────────
# 0.  PURE-NUMPY KMEANS  (avoids sklearn threadpoolctl bug on Windows)
# ─────────────────────────────────────────────────────────────────────────────

def numpy_kmeans(X, k, max_iter=150, n_init=3, random_state=42):
    """
    Simple Lloyd's algorithm, fully in numpy.
    Returns: centroids (k, d), labels (n,), min_dists (n,)
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    best_inertia = np.inf
    best_centroids = None
    best_labels = None

    for trial in range(n_init):
        # KMeans++ init
        idx = [int(rng.integers(n))]
        for _ in range(k - 1):
            dists = np.min(cdist(X, X[idx], metric="sqeuclidean"), axis=1)
            probs = dists / dists.sum()
            idx.append(int(rng.choice(n, p=probs)))
        centroids = X[idx].copy()

        for _ in range(max_iter):
            dists_all = cdist(X, centroids, metric="sqeuclidean")   # (n, k)
            labels = dists_all.argmin(axis=1)
            new_centroids = np.array([
                X[labels == c].mean(axis=0) if (labels == c).any() else centroids[c]
                for c in range(k)
            ])
            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        inertia = dists_all[np.arange(n), labels].sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.copy()
            best_labels = labels.copy()

    min_dists = np.sqrt(
        np.min(cdist(X, best_centroids, metric="sqeuclidean"), axis=1)
    )
    return best_centroids, best_labels, min_dists


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATCH EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_patches(data_dir, batch_size=512):
    """
    Returns:
        patches : (N, 16, 192)  float32 raw pixel values in [0,1]
        labels  : (N,)          int
    """
    # No normalisation — raw pixels for patch features
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    N = len(dataset)
    patches = np.zeros((N, N_PATCHES, PATCH_DIM), dtype=np.float32)
    labels  = np.zeros(N, dtype=np.int64)

    idx = 0
    for imgs, lbls in loader:
        B = imgs.shape[0]
        imgs_np = imgs.numpy()   # (B, 3, 32, 32)
        for b in range(B):
            p_num = 0
            for row in range(PATCH_GRID):
                for col in range(PATCH_GRID):
                    patch = imgs_np[b, :,
                                    row * PATCH_SIZE:(row + 1) * PATCH_SIZE,
                                    col * PATCH_SIZE:(col + 1) * PATCH_SIZE]  # (3,8,8)
                    patches[idx + b, p_num] = patch.flatten()
                    p_num += 1
            labels[idx + b] = lbls[b].item()
        idx += B
        if idx % 10000 == 0 or idx == N:
            print(f"  Patches extracted: {idx:,} / {N:,}")

    return patches, labels


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FLAT FEATURES  (raw L2 — identical to curriculum_selection.py for t-SNE)
# ─────────────────────────────────────────────────────────────────────────────

def extract_flat_features(data_dir, batch_size=512):
    """
    Flatten each 32x32x3 image to 3072-d.
    Identical to the same function in curriculum_selection.py →
    both experiments use the same t-SNE feature space.
    """
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True,
                          transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats = []
    for imgs, _ in loader:
        feats.append(imgs.view(len(imgs), -1).numpy())
    return np.concatenate(feats)   # (N, 3072)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PER-POSITION KMEANS → PATCH-DISTANCE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_patch_distances(patches, K_patch, random_state=42):
    """
    For each of the 16 spatial positions, run KMeans(K_patch) on all N patches
    at that position and compute each patch's distance to its nearest centroid.

    Returns:
        dist_matrix  : (N, 16)  normalised distances  in [0, 1]
        raw_dists    : (N, 16)  raw Euclidean distances (for diagnostics)
        centroids    : list[16] of (K_patch, PATCH_DIM)  cluster centers
        assignments  : (N, 16) cluster id per patch
    """
    N = patches.shape[0]
    dist_matrix  = np.zeros((N, N_PATCHES), dtype=np.float32)
    raw_dists    = np.zeros((N, N_PATCHES), dtype=np.float32)
    centroids    = []
    assignments  = np.zeros((N, N_PATCHES), dtype=np.int32)

    for p in range(N_PATCHES):
        row, col = divmod(p, PATCH_GRID)
        pos_patches = patches[:, p, :]   # (N, 192)

        # L2-normalise patches so features are scale-invariant
        norms = np.linalg.norm(pos_patches, axis=1, keepdims=True) + 1e-8
        pos_norm = pos_patches / norms

        print(f"  Position ({row},{col})  KMeans k={K_patch} … ", end="", flush=True)
        ctrs, lbls, dists = numpy_kmeans(pos_norm, k=K_patch,
                                         random_state=random_state + p)

        raw_dists[:, p]   = dists
        d_max = dists.max() + 1e-8
        dist_matrix[:, p] = dists / d_max   # normalised to [0,1]
        assignments[:, p] = lbls
        centroids.append(ctrs)
        print(f"done  (mean dist={dists.mean():.3f})")

    return dist_matrix, raw_dists, centroids, assignments


# ─────────────────────────────────────────────────────────────────────────────
# 3.  K-MEDOIDS  (PAM style, on small feature vectors)
# ─────────────────────────────────────────────────────────────────────────────

def kmedoids(X, k, max_iter=300, random_state=42):
    """K-Medoids on the 16-dim patch-distance vectors."""
    rng = np.random.default_rng(random_state)
    n   = len(X)
    D   = cdist(X, X, metric="euclidean")

    # K-Means++ initialisation
    medoid_idx = [int(rng.integers(n))]
    for _ in range(k - 1):
        dist_to_closest = D[:, medoid_idx].min(axis=1)
        probs = dist_to_closest ** 2
        probs /= probs.sum()
        medoid_idx.append(int(rng.choice(n, p=probs)))
    medoid_idx = np.array(medoid_idx)

    for it in range(max_iter):
        assignment = D[:, medoid_idx].argmin(axis=1)
        new_medoids = medoid_idx.copy()
        for m in range(k):
            cluster_pts = np.where(assignment == m)[0]
            if len(cluster_pts) == 0:
                continue
            intra = D[np.ix_(cluster_pts, cluster_pts)]
            new_medoids[m] = cluster_pts[intra.sum(axis=1).argmin()]
        if np.all(new_medoids == medoid_idx):
            print(f"  converged after {it + 1} iters.")
            break
        medoid_idx = new_medoids

    assignment      = D[:, medoid_idx].argmin(axis=1)
    dist_to_medoid  = D[np.arange(n), medoid_idx[assignment]]
    return medoid_idx, assignment, dist_to_medoid


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BUILD CURRICULUM
# ─────────────────────────────────────────────────────────────────────────────

def build_curriculum(dist_matrix, labels, k, out_dir):
    """
    Per-class K-Medoids on 16-dim patch-distance feature vectors.
    Curriculum score = distance to class medoid in patch-distance space.
    """
    N = len(dist_matrix)
    patch_centrality  = dist_matrix.mean(axis=1)   # (N,) raw patch score
    curriculum_score  = np.zeros(N)
    medoid_mask       = np.zeros(N, dtype=bool)
    cluster_label     = np.full(N, -1, dtype=int)

    for cls in range(10):
        mask    = labels == cls
        cls_idx = np.where(mask)[0]
        cls_X   = dist_matrix[cls_idx]    # (5000, 16)

        print(f"  Class {cls:2d} ({CIFAR10_CLASSES[cls]:10s}): k={k}", end="  ")
        med_local, assign, dist = kmedoids(cls_X, k=k, random_state=cls * 7)

        d_max = dist.max() + 1e-8
        curriculum_score[cls_idx] = dist / d_max
        cluster_label[cls_idx]    = assign + cls * k

        for m in med_local:
            medoid_mask[cls_idx[m]] = True

    curriculum_order = np.argsort(curriculum_score)

    result = dict(
        patches=None,              # not stored (large)
        labels=labels,
        dist_matrix=dist_matrix,
        patch_centrality=patch_centrality,
        curriculum_score=curriculum_score,
        curriculum_order=curriculum_order,
        medoid_mask=medoid_mask,
        cluster_label=cluster_label,
        k=k,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "patch_curriculum_order.npy",  np.arange(N)[curriculum_order])
    np.save(out_dir / "patch_curriculum_score.npy",  curriculum_score)
    np.save(out_dir / "patch_medoid_mask.npy",       medoid_mask)
    np.save(out_dir / "patch_centrality.npy",        patch_centrality)
    print(f"\n  Saved curriculum to {out_dir}/")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  t-SNE  (on per-image patch features)
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(feats, perplexity=40, random_state=42):
    """PCA(50) → t-SNE(2)  on patch feature matrix."""
    n_pca = min(50, feats.shape[1], feats.shape[0] - 1)
    print(f"[t-SNE] PCA → {n_pca} dims …")
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(feats)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    print(f"[t-SNE] Fitting {len(feats)} samples …")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000,
                random_state=random_state, verbose=1, method="exact")
    return tsne.fit_transform(X_pca)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_tsne_overview(emb, labels, medoid_mask, out_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    for cls in range(10):
        m = (labels == cls) & ~medoid_mask
        ax.scatter(emb[m, 0], emb[m, 1], c=[CLASS_COLORS[cls]],
                   s=8, alpha=0.35, linewidths=0, label=CIFAR10_CLASSES[cls])
    for cls in range(10):
        m = (labels == cls) & medoid_mask
        ax.scatter(emb[m, 0], emb[m, 1], c=[CLASS_COLORS[cls]],
                   s=200, marker="*", edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_title("t-SNE of CIFAR10 Patch Features (4×4 grid, 8×8 patches)\n"
                 "Stars = Patch-Level K-Medoid centres (most representative images)", fontsize=13)
    ax.legend(loc="upper right", markerscale=2, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_tsne_difficulty(emb, curriculum_score, medoid_mask, out_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    sc = ax.scatter(emb[~medoid_mask, 0], emb[~medoid_mask, 1],
                    c=curriculum_score[~medoid_mask], cmap="RdYlBu_r",
                    s=8, alpha=0.5, linewidths=0, vmin=0, vmax=1)
    ax.scatter(emb[medoid_mask, 0], emb[medoid_mask, 1],
               c="black", s=160, marker="*", zorder=5, label="Medoids")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Patch Curriculum Difficulty\n(0 = patches near centroid,  1 = patches atypical)",
                   fontsize=10)
    ax.set_title("t-SNE — Patch-Level Curriculum Difficulty\n"
                 "Blue = easy (central patches)    Red = hard (atypical patches)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_medoid_images_with_heatmap(raw_dataset, medoid_mask, labels,
                                    dist_matrix, curriculum_score, k, out_path):
    """
    Grid of medoid images (k rows × 10 columns).
    Each image has a 4×4 patch heatmap overlay showing per-patch distance to centroid
    (green = central patch, red = atypical patch).
    """
    n_cls = 10
    medoid_pos = np.where(medoid_mask)[0]

    fig = plt.figure(figsize=(n_cls * 2.2, k * 2.8 + 0.8))
    outer = fig.add_gridspec(k, n_cls, hspace=0.55, wspace=0.15)

    for cls in range(n_cls):
        cls_meds = [p for p in medoid_pos if labels[p] == cls]
        cls_meds = sorted(cls_meds, key=lambda p: curriculum_score[p])

        for row, pos in enumerate(cls_meds[:k]):
            inner = outer[row, cls].subgridspec(1, 2, wspace=0.05, width_ratios=[1, 0.08])
            ax_img  = fig.add_subplot(inner[0])
            ax_cbar = fig.add_subplot(inner[1])

            # Raw image
            img_tensor, _ = raw_dataset[pos]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            ax_img.imshow(img_np, aspect="equal")

            # Patch heatmap overlay
            patch_dists = dist_matrix[pos]  # (16,) normalised distances
            heat = patch_dists.reshape(PATCH_GRID, PATCH_GRID)
            # Use alpha blending: red channel where atypical, green where central
            heat_img = plt.cm.RdYlGn_r(heat)   # RGBA
            heat_img[..., 3] = 0.45             # semi-transparent
            ax_img.imshow(heat_img, extent=[0, 32, 32, 0], aspect="equal")

            # 4×4 grid lines
            for g in range(1, PATCH_GRID):
                ax_img.axhline(g * PATCH_SIZE, color="white", lw=0.5, alpha=0.6)
                ax_img.axvline(g * PATCH_SIZE, color="white", lw=0.5, alpha=0.6)

            ax_img.axis("off")
            score = curriculum_score[pos]
            if row == 0:
                ax_img.set_title(CIFAR10_CLASSES[cls], fontsize=9, fontweight="bold", pad=3)
            ax_img.text(0.5, -0.06, f"d={score:.2f}",
                        transform=ax_img.transAxes, ha="center", fontsize=7, color="dimgray")

            # Tiny colour bar (only first column)
            if cls == 0:
                sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                                           norm=mcolors.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
                ax_cbar.tick_params(labelsize=6)
                ax_cbar.yaxis.set_label_position("left")
                ax_cbar.set_ylabel("patch\ndifficulty", fontsize=6, labelpad=2)
            else:
                ax_cbar.axis("off")

    fig.suptitle(
        f"Most Important Images — Patch-Level K-Medoids  (k={k} per class)\n"
        "Overlay: green patch = near centroid cloud,  red patch = atypical",
        fontsize=11, y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_medoid_images_clean(raw_dataset, medoid_mask, labels,
                             curriculum_score, k, out_path):
    """
    Clean grid identical in style to most_important_samples.png from curriculum_selection.py.
    No heatmap — pure image + difficulty score label, for direct comparison.
    """
    n_cls = 10
    medoid_pos = np.where(medoid_mask)[0]
    fig, axes = plt.subplots(k, n_cls, figsize=(n_cls * 1.5, k * 1.7 + 1))
    if k == 1:
        axes = axes[np.newaxis, :]

    for cls in range(n_cls):
        cls_meds = [p for p in medoid_pos if labels[p] == cls]
        cls_meds = sorted(cls_meds, key=lambda p: curriculum_score[p])
        for row, pos in enumerate(cls_meds[:k]):
            img_tensor, _ = raw_dataset[pos]
            ax = axes[row, cls]
            ax.imshow(img_tensor.permute(1, 2, 0).numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(CIFAR10_CLASSES[cls], fontsize=9, fontweight="bold")
            ax.text(0.5, -0.06, f"d={curriculum_score[pos]:.2f}",
                    transform=ax.transAxes, ha="center", fontsize=7, color="dimgray")

    fig.suptitle(
        f"Most Important Samples — Patch-Level K-Medoids  (k={k} per class)\n"
        "d = patch curriculum score  (lower = patches more central/representative)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_difficulty_distribution(curriculum_score, labels, out_path):
    fig, ax = plt.subplots(figsize=(13, 5))
    data = [curriculum_score[labels == c] for c in range(10)]
    parts = ax.violinplot(data, positions=range(10), showmedians=True)
    for pc, color in zip(parts["bodies"], CLASS_COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=20, ha="right")
    ax.set_ylabel("Patch Curriculum Difficulty Score")
    ax.set_title("Per-Class Patch Difficulty Distribution\n"
                 "(score = mean normalised distance across 16 patch positions to nearest cluster center)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_patch_centrality_overview(patch_centrality, labels, out_path):
    """
    Bar chart: mean patch centrality per class (how central the class is on average).
    Classes with low centrality have more prototypical patch appearances.
    """
    means = [patch_centrality[labels == c].mean() for c in range(10)]
    stds  = [patch_centrality[labels == c].std()  for c in range(10)]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(10), means, yerr=stds, capsize=4,
                  color=CLASS_COLORS, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=20, ha="right")
    ax.set_ylabel("Mean Patch Centrality Score")
    ax.set_title("Per-Class Average Patch Centrality\n"
                 "Lower = class has more uniform/prototypical patches    "
                 "Higher = class has more diverse patch appearances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="./data")
    p.add_argument("--out_dir",   default="./patch_curriculum")
    p.add_argument("--k",         type=int, default=5,
                   help="K-Medoids clusters per class (medoids to select)")
    p.add_argument("--K_patch",   type=int, default=20,
                   help="KMeans clusters per patch position")
    p.add_argument("--n_samples", type=int, default=5000,
                   help="Samples for t-SNE visualisation (0 = all 50k)")
    p.add_argument("--batch_size",type=int, default=512)
    p.add_argument("--skip_tsne", action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Experiment 2: Patch-Level L2 K-Medoids")

    # ── 1. Extract patches ────────────────────────────────────────────────────
    print("\n[1/5] Extracting 4x4 patches from all CIFAR10 images …")
    patches, labels = extract_all_patches(args.data_dir, args.batch_size)
    print(f"  patches shape: {patches.shape}   labels: {labels.shape}")

    # ── 2. Per-position KMeans → patch distance matrix ───────────────────────
    print(f"\n[2/5] Per-position KMeans  (K_patch={args.K_patch}) …")
    dist_matrix, raw_dists, centroids, assignments = compute_patch_distances(
        patches, K_patch=args.K_patch
    )
    print(f"  dist_matrix shape: {dist_matrix.shape}  (images x positions)")
    np.save(out_dir / "dist_matrix.npy", dist_matrix)

    # ── 3. K-Medoids curriculum ───────────────────────────────────────────────
    print(f"\n[3/5] K-Medoids per class  (k={args.k}) …")
    result = build_curriculum(dist_matrix, labels, args.k, out_dir)

    # ── 4. t-SNE — same 3072-d raw pixel features as curriculum_selection.py ─
    if not args.skip_tsne:
        n_vis = args.n_samples if args.n_samples > 0 else len(patches)
        print(f"\n[4/5] t-SNE on {n_vis} samples (3072-d raw pixels, same as Exp 1) …")
        flat_feats = extract_flat_features(args.data_dir, args.batch_size)

        medoid_pos = np.where(result["medoid_mask"])[0]
        non_med    = np.where(~result["medoid_mask"])[0]
        rng  = np.random.default_rng(0)   # same seed → same subsample as Exp 1
        fill = rng.choice(non_med, size=min(n_vis - len(medoid_pos), len(non_med)),
                          replace=False)
        vis_idx = np.concatenate([medoid_pos, fill])

        vis_feats  = flat_feats[vis_idx]   # 3072-d, same feature space as Exp 1
        vis_labels = labels[vis_idx]
        vis_scores = result["curriculum_score"][vis_idx]
        vis_medoid = result["medoid_mask"][vis_idx]

        emb = run_tsne(vis_feats, perplexity=40)

        print("\n[4/5] Plotting t-SNE …")
        plot_tsne_overview(emb, vis_labels, vis_medoid,
                           out_dir / "patch_tsne_class_overview.png")
        plot_tsne_difficulty(emb, vis_scores, vis_medoid,
                             out_dir / "patch_tsne_curriculum_difficulty.png")
    else:
        print("\n[4/5] Skipping t-SNE.")

    # ── 5. Visualise most important images ────────────────────────────────────
    print("\n[5/5] Plotting most important images …")
    raw_ds = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True,
        transform=transforms.ToTensor()
    )
    # With heatmap overlay
    plot_medoid_images_with_heatmap(
        raw_ds, result["medoid_mask"], result["labels"],
        result["dist_matrix"], result["curriculum_score"], args.k,
        out_dir / "patch_most_important_samples.png",
    )
    # Clean version (same style as most_important_samples.png) for comparison
    plot_medoid_images_clean(
        raw_ds, result["medoid_mask"], result["labels"],
        result["curriculum_score"], args.k,
        out_dir / "patch_most_important_samples_clean.png",
    )
    plot_difficulty_distribution(
        result["curriculum_score"], result["labels"],
        out_dir / "patch_difficulty_distribution.png",
    )
    plot_patch_centrality_overview(
        result["patch_centrality"], result["labels"],
        out_dir / "patch_centrality_per_class.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    scores = result["curriculum_score"]
    order  = result["curriculum_order"]
    n10    = len(order) // 10
    print("\n═══ Patch Curriculum Summary ═══")
    print(f"  Patch grid          : {PATCH_GRID}×{PATCH_GRID}  ({N_PATCHES} patches per image)")
    print(f"  Patch size          : {PATCH_SIZE}×{PATCH_SIZE}×3 = {PATCH_DIM} dims")
    print(f"  KMeans clusters/pos : {args.K_patch}")
    print(f"  K-Medoids / class   : {args.k}  →  {args.k * 10} total medoid images")
    print(f"  Easy 10% score      : {scores[order[:n10]].max():.3f}  (below this = central patches)")
    print(f"  Hard 10% score      : {scores[order[-n10:]].min():.3f}  (above this = atypical patches)")
    print(f"\n  Output: {out_dir.resolve()}")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
