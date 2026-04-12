"""
K-Medoids Curriculum Selection for CIFAR10  [Experiment 1: Whole-Image L2]
==========================================================================
Selects the most representative (important) training samples per class
using K-Medoids clustering on raw flattened pixel features (L2 distance).

  Feature: flatten 32x32x3 image -> 3072-d vector, L2 distance
  K-Medoids: per class, finds k most central images
  t-SNE: same 3072-d -> PCA(50) -> t-SNE  (identical layout to patch experiment)

Curriculum ordering: sorted by distance to class medoid.
  - Near medoid  => "easy" / most representative
  - Far from medoid => "hard" / outlier

Usage:
    python curriculum_selection.py
    python curriculum_selection.py --k 5 --n_samples 5000
"""

# Must be set before any numpy/sklearn import to avoid threadpoolctl crash on Windows
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# ── CIFAR10 class names ──────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ── Palette: one colour per class ────────────────────────────────────────────
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FEATURE EXTRACTION  (raw L2 — identical to patch_curriculum_selection.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_flat_features(data_dir, batch_size=512):
    """
    Load CIFAR10, flatten each 32x32x3 image to 3072-d float32 vector.
    Returns: feats (N, 3072), labels (N,)
    Same function used in patch_curriculum_selection.py → t-SNE layouts are comparable.
    """
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True,
                          transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for imgs, lbls in loader:
        feats.append(imgs.view(len(imgs), -1).numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  K-MEDOIDS
# ─────────────────────────────────────────────────────────────────────────────

def kmedoids(X, k, max_iter=300, random_state=42):
    """
    Simple PAM-style K-Medoids on a distance matrix.
    Returns: medoid_indices (local to X), labels, distances_to_medoid.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)

    # Pre-compute pairwise L2 distances
    D = cdist(X, X, metric="euclidean")

    # Init: k-means++ style seed
    medoid_idx = [int(rng.integers(n))]
    for _ in range(k - 1):
        dist_to_closest = D[:, medoid_idx].min(axis=1)
        probs = dist_to_closest ** 2
        probs /= probs.sum()
        medoid_idx.append(int(rng.choice(n, p=probs)))

    medoid_idx = np.array(medoid_idx)

    for iteration in range(max_iter):
        # Assignment step
        assignment = D[:, medoid_idx].argmin(axis=1)   # which medoid each point belongs to

        # Update step – swap each medoid with the point that minimises total cost
        new_medoids = medoid_idx.copy()
        for m in range(k):
            cluster_mask = assignment == m
            if not cluster_mask.any():
                continue
            cluster_pts = np.where(cluster_mask)[0]
            # Total distance within cluster for each candidate medoid
            intra_D = D[np.ix_(cluster_pts, cluster_pts)]
            best_local = intra_D.sum(axis=1).argmin()
            new_medoids[m] = cluster_pts[best_local]

        if np.all(new_medoids == medoid_idx):
            print(f"  K-Medoids converged after {iteration + 1} iterations.")
            break
        medoid_idx = new_medoids

    # Final assignment & distances
    assignment = D[:, medoid_idx].argmin(axis=1)
    dist_to_medoid = D[np.arange(n), medoid_idx[assignment]]
    return medoid_idx, assignment, dist_to_medoid


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CURRICULUM SCORING
# ─────────────────────────────────────────────────────────────────────────────

def build_curriculum(feats, labels, global_indices, k_per_class, out_dir):
    """
    Run K-Medoids per class and compute a curriculum difficulty score
    (= distance to the nearest class medoid, normalised to [0,1]).

    Returns a dict with all results.
    """
    n_total = len(feats)
    curriculum_score = np.zeros(n_total)
    medoid_mask = np.zeros(n_total, dtype=bool)   # True if sample is a medoid
    cluster_label = np.full(n_total, -1, dtype=int)

    all_medoid_local = {}   # class → local indices of medoids within class subset

    for cls in range(10):
        mask = labels == cls
        cls_idx = np.where(mask)[0]        # positions in feats/labels arrays
        cls_feats = feats[cls_idx]

        print(f"  Class {cls:2d} ({CIFAR10_CLASSES[cls]:10s}): "
              f"{len(cls_idx):5d} samples, k={k_per_class}")

        med_local, assign, dist = kmedoids(cls_feats, k=k_per_class, random_state=cls)

        # Normalise distances within class to [0, 1]
        d_max = dist.max() + 1e-8
        norm_dist = dist / d_max

        curriculum_score[cls_idx] = norm_dist
        cluster_label[cls_idx] = assign + cls * k_per_class   # globally unique cluster id

        for m in med_local:
            medoid_mask[cls_idx[m]] = True

        all_medoid_local[cls] = med_local

    # Curriculum order: sort ascending by score → easy first
    curriculum_order = np.argsort(curriculum_score)

    result = dict(
        feats=feats,
        labels=labels,
        global_indices=global_indices,
        curriculum_score=curriculum_score,
        curriculum_order=curriculum_order,
        medoid_mask=medoid_mask,
        cluster_label=cluster_label,
        all_medoid_local=all_medoid_local,
        k_per_class=k_per_class,
    )

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "curriculum_order.npy", global_indices[curriculum_order])
    np.save(out_dir / "curriculum_score.npy", curriculum_score)
    np.save(out_dir / "medoid_mask.npy", medoid_mask)
    print(f"\n  Saved curriculum to {out_dir}/")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  t-SNE VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def run_tsne(feats, n_components=2, perplexity=40, random_state=42):
    """PCA → 50 dims first (standard practice), then t-SNE."""
    print("\n[t-SNE] PCA pre-reduction to 50 dims …")
    n_pca = min(50, feats.shape[1], feats.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=random_state)
    feats_pca = pca.fit_transform(feats)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    print(f"[t-SNE] Fitting on {len(feats)} samples … (this takes ~1-2 min)")
    # Use method='exact' to avoid threadpoolctl crash on Windows (sklearn KNN bug)
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=1000,
        random_state=random_state,
        verbose=1,
        method="exact",
    )
    return tsne.fit_transform(feats_pca)


def plot_tsne_overview(embedding, labels, medoid_mask, out_path):
    """t-SNE coloured by class, medoids highlighted as large stars."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: all points, small, semi-transparent
    for cls in range(10):
        mask = (labels == cls) & ~medoid_mask
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[CLASS_COLORS[cls]],
            s=8, alpha=0.35, linewidths=0,
            label=CIFAR10_CLASSES[cls],
        )

    # Medoids: large stars with black edge
    for cls in range(10):
        mask = (labels == cls) & medoid_mask
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[CLASS_COLORS[cls]],
            s=200, marker="*", edgecolors="black", linewidths=0.8,
            zorder=5,
        )

    ax.set_title("t-SNE of CIFAR10 ResNet18 Features\n"
                 "Stars = K-Medoid centres (most representative samples)",
                 fontsize=14)
    ax.legend(loc="upper right", markerscale=2, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_tsne_curriculum(embedding, curriculum_score, medoid_mask, out_path):
    """t-SNE coloured by curriculum difficulty (easy=blue, hard=red)."""
    fig, ax = plt.subplots(figsize=(11, 9))

    sc = ax.scatter(
        embedding[~medoid_mask, 0], embedding[~medoid_mask, 1],
        c=curriculum_score[~medoid_mask],
        cmap="RdYlBu_r", s=8, alpha=0.5, linewidths=0,
        vmin=0, vmax=1,
    )
    # Medoids on top
    ax.scatter(
        embedding[medoid_mask, 0], embedding[medoid_mask, 1],
        c="black", s=160, marker="*", zorder=5, label="Medoids",
    )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Curriculum Difficulty\n(0 = easy/representative,  1 = hard/outlier)",
                   fontsize=10)
    ax.set_title("t-SNE — Curriculum Difficulty Score\n"
                 "Blue = easy (close to medoid)    Red = hard (far from medoid)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SHOW MOST IMPORTANT IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def get_raw_dataset(data_dir):
    """CIFAR10 without normalisation, for display."""
    transform = transforms.Compose([transforms.ToTensor()])
    return datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)


def plot_medoid_images(raw_dataset, global_indices, medoid_mask, labels,
                       curriculum_score, k_per_class, out_path):
    """
    Grid: one column per class, k_per_class rows.
    Shows the actual medoid images with their difficulty score.
    """
    n_cls = 10
    fig, axes = plt.subplots(
        k_per_class, n_cls,
        figsize=(n_cls * 1.5, k_per_class * 1.7 + 1),
    )
    if k_per_class == 1:
        axes = axes[np.newaxis, :]

    # Gather medoid samples per class
    medoid_pos = np.where(medoid_mask)[0]   # positions in our feats array

    for cls in range(n_cls):
        cls_medoids = [p for p in medoid_pos if labels[p] == cls]
        # Sort by curriculum score (ascending → most representative first)
        cls_medoids = sorted(cls_medoids, key=lambda p: curriculum_score[p])

        for row, pos in enumerate(cls_medoids[:k_per_class]):
            global_idx = global_indices[pos]
            img, _ = raw_dataset[global_idx]
            img_np = img.permute(1, 2, 0).numpy()

            ax = axes[row, cls]
            ax.imshow(img_np)
            ax.axis("off")
            score = curriculum_score[pos]
            if row == 0:
                ax.set_title(CIFAR10_CLASSES[cls], fontsize=9, fontweight="bold")
            ax.text(0.5, -0.06, f"d={score:.2f}",
                    transform=ax.transAxes, ha="center", fontsize=7, color="dimgray")

    fig.suptitle(
        f"Most Important (Representative) Samples — K-Medoids  (k={k_per_class} per class)\n"
        "d = curriculum difficulty score (lower = more central/representative)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_class_distribution(curriculum_score, labels, out_path):
    """Per-class distribution of difficulty scores (violin plot)."""
    fig, ax = plt.subplots(figsize=(13, 5))

    data_per_class = [curriculum_score[labels == c] for c in range(10)]
    parts = ax.violinplot(data_per_class, positions=range(10), showmedians=True)

    for pc, color in zip(parts["bodies"], CLASS_COLORS):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=20, ha="right")
    ax.set_ylabel("Curriculum Difficulty Score")
    ax.set_title("Per-Class Difficulty Distribution\n"
                 "(0 = close to medoid / easy,   1 = outlier / hard)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="./data",        help="CIFAR10 data root")
    p.add_argument("--out_dir",   default="./curriculum",  help="Output directory")
    p.add_argument("--k",         type=int, default=5,     help="Medoids per class")
    p.add_argument("--n_samples", type=int, default=5000,
                   help="Max samples for t-SNE visualisation (use 0 for all 50k)")
    p.add_argument("--batch_size",type=int, default=256)
    p.add_argument("--tsne_perplexity", type=int, default=40)
    p.add_argument("--skip_tsne", action="store_true",
                   help="Skip t-SNE (fast mode, just show medoid images)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Experiment 1: Whole-Image L2 K-Medoids")

    # ── Extract flat features (3072-d L2) ────────────────────────────────────
    print("\n[1/4] Loading CIFAR10 + flattening to 3072-d vectors …")
    feats, labels = extract_flat_features(args.data_dir, args.batch_size)
    global_indices = np.arange(len(feats))
    print(f"  feats: {feats.shape},  labels: {labels.shape}")

    # ── K-Medoids curriculum ─────────────────────────────────────────────────
    print(f"\n[2/4] K-Medoids per class  (k={args.k}) …")
    result = build_curriculum(feats, labels, global_indices, args.k, out_dir)

    # ── t-SNE (same 3072-d features, same seed → comparable with patch exp) ──
    if not args.skip_tsne:
        n_vis = args.n_samples if args.n_samples > 0 else len(feats)
        print(f"\n[3/4] t-SNE on {n_vis} samples (3072-d raw pixels, PCA->50) …")
        medoid_pos = np.where(result["medoid_mask"])[0]
        non_med    = np.where(~result["medoid_mask"])[0]
        rng  = np.random.default_rng(0)   # same seed as patch experiment
        fill = rng.choice(non_med, size=min(n_vis - len(medoid_pos), len(non_med)),
                          replace=False)
        vis_idx = np.concatenate([medoid_pos, fill])

        vis_feats  = feats[vis_idx]
        vis_labels = labels[vis_idx]
        vis_scores = result["curriculum_score"][vis_idx]
        vis_medoid = result["medoid_mask"][vis_idx]

        embedding = run_tsne(vis_feats, perplexity=args.tsne_perplexity)

        print("\n[3/4] Plotting t-SNE …")
        plot_tsne_overview(embedding, vis_labels, vis_medoid,
                           out_dir / "tsne_class_overview.png")
        plot_tsne_curriculum(embedding, vis_scores, vis_medoid,
                             out_dir / "tsne_curriculum_difficulty.png")
    else:
        print("\n[3/4] Skipping t-SNE.")

    # ── Most important sample images ─────────────────────────────────────────
    print("\n[4/4] Plotting medoid images …")
    raw_dataset = get_raw_dataset(args.data_dir)
    plot_medoid_images(
        raw_dataset,
        result["global_indices"],
        result["medoid_mask"],
        result["labels"],
        result["curriculum_score"],
        args.k,
        out_dir / "most_important_samples.png",
    )
    plot_class_distribution(
        result["curriculum_score"],
        result["labels"],
        out_dir / "difficulty_distribution.png",
    )

    # ── Summary stats ──────────────────────────────────────────────────────────
    print("\n═══ Curriculum Summary ═══")
    scores = result["curriculum_score"]
    order  = result["curriculum_order"]
    print(f"  Total samples          : {len(scores):,}")
    print(f"  Total medoids selected : {result['medoid_mask'].sum():,}  "
          f"({args.k} per class × 10 classes)")
    print(f"  Easy 10% score range   : {scores[order[:len(order)//10]].min():.3f} – "
          f"{scores[order[:len(order)//10]].max():.3f}")
    print(f"  Hard 10% score range   : {scores[order[-len(order)//10:]].min():.3f} – "
          f"{scores[order[-len(order)//10:]].max():.3f}")
    print(f"\n  Output directory       : {out_dir.resolve()}")
    print("  Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
