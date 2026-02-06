"""
Experiment: Heatmap comparing EFM variants vs t and sigma.

Methods:
- EFM: random pairing, standard closed-form weights
- LSEFM: random pairing, local spatially-aware EFM (3x3 kernel)
- s_EFM: random pairing, score uses xt + sigma*epsilon - t*x1 (noise-injected)

X-axis: t (0.05, 0.25, 0.5, 0.75, 0.95)
Y-axis: sigma (0.1, 0.3, 0.5, 0.7, 0.9)
Value: mean(method cos) - mean(EFM cos)  [positive = method better than EFM]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import create_model
from flow_matcher import create_flow_matcher
from dataset import get_dataset
from utils.efm import efm_closed_form_weights_and_u, efm_closed_form_weights_and_u_sigma, lsefm_closed_form_u


def compute_cosine_similarity(vec1, vec2):
    vec1_flat = vec1.reshape(vec1.shape[0], -1)
    vec2_flat = vec2.reshape(vec2.shape[0], -1)
    dot_product = (vec1_flat * vec2_flat).sum(dim=1)
    norm1 = torch.norm(vec1_flat, dim=1)
    norm2 = torch.norm(vec2_flat, dim=1)
    cos_sim = dot_product / (norm1 * norm2 + 1e-8)
    return cos_sim


def compute_pairing_error_efm(model, flow_matcher, x0_small, x1_small, x1_full, epsilon, t_value, device, method='efm', sigma=0.0, seed=None):
    """Compute mean cosine similarity for EFM / LSEFM / s_EFM.
    
    Args:
        x0_small: [pairing_batch_size, C, H, W] - noise for pairing
        x1_small: [pairing_batch_size, C, H, W] - subset of x1 for pairing
        x1_full: [efm_batch_size, C, H, W] - full x1 for EFM weights
        epsilon: [pairing_batch_size, C, H, W] - noise for xt
        method: 'efm', 'lsefm', or 's_efm'
        sigma: noise strength for s_efm
    """
    model.eval()
    batch_size = x0_small.shape[0]
    t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Pair within small batch
        x0_paired, x1_paired = get_paired_samples_cfm(x0_small, x1_small)
        t_expanded = t.reshape(-1, *([1] * (x0_paired.dim() - 1)))
        mu_t = t_expanded * x1_paired + (1 - t_expanded) * x0_paired
        sigma_t = flow_matcher.compute_sigma_t(t)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t_expanded = sigma_t.reshape(-1, *([1] * (mu_t.dim() - 1)))
        else:
            sigma_t_expanded = sigma_t
        xt = mu_t + sigma_t_expanded * epsilon

        # Compute theoretical direction based on method
        if method == 'efm':
            xt_flat = xt.reshape(batch_size, -1)
            x1_full_flat = x1_full.reshape(x1_full.shape[0], -1)
            _, ut_theoretical, _ = efm_closed_form_weights_and_u(xt_flat, x1_full_flat, t, eps=1e-6)
            ut_theoretical = ut_theoretical.reshape_as(xt)
        elif method == 'lsefm':
            ut_theoretical = lsefm_closed_form_u(xt, x1_full, t, kernel_size=3, eps=1e-6)
        elif method == 's_efm':
            xt_flat = xt.reshape(batch_size, -1)
            x1_full_flat = x1_full.reshape(x1_full.shape[0], -1)
            gen = torch.Generator(device=device)
            if seed is not None:
                gen.manual_seed(seed)
            _, ut_theoretical, _ = efm_closed_form_weights_and_u_sigma(
                xt_flat, x1_full_flat, t, sigma=sigma, generator=gen, eps=1e-6
            )
            ut_theoretical = ut_theoretical.reshape_as(xt)
        else:
            raise ValueError(f"Unknown method: {method}")

        vt_predicted = model(xt, t)
        errors = compute_cosine_similarity(ut_theoretical, vt_predicted)

    return errors.cpu().numpy()


def get_paired_samples_cfm(x0, x1):
    indices = torch.randperm(x1.shape[0], device=x1.device)
    x1_paired = x1[indices]
    return x0, x1_paired


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    dataset_name = train_args.get("dataset", "cifar10")
    sigma_fm = 0.0
    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Load large batch for EFM, small batch for pairing
    dataloader_large, _ = get_dataset(dataset_name, args.efm_batch_size, args.data_dir)
    flow_matcher = create_flow_matcher("cfm", sigma_fm)

    t_values = [0.05, 0.25, 0.5, 0.75, 0.95]
    sigma_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # [sigma_idx, t_idx] -> mean cos for each method
    results_efm = np.zeros((len(sigma_values), len(t_values)))
    results_lsefm = np.zeros((len(sigma_values), len(t_values)))
    results_s_efm = np.zeros((len(sigma_values), len(t_values)))

    for batch_idx in range(args.num_batches):
        print(f"\nBatch {batch_idx + 1}/{args.num_batches}")
        # Load full EFM batch
        x1_full = next(iter(dataloader_large))[0].to(device)
        # Sample small pairing batch
        x0_small = torch.randn(args.pairing_batch_size, *x1_full.shape[1:], device=device)
        x1_small = x1_full[: args.pairing_batch_size]
        epsilon = torch.randn_like(x0_small)

        for ti, t_val in enumerate(t_values):
            # EFM: compute once per (batch, t)
            efm_cos = compute_pairing_error_efm(
                model, flow_matcher, x0_small, x1_small, x1_full, epsilon, t_val, device, method='efm'
            )
            efm_mean = float(efm_cos.mean())
            results_efm[:, ti] += efm_mean  # same EFM for all sigma rows

            # LSEFM: compute once per (batch, t)
            lsefm_cos = compute_pairing_error_efm(
                model, flow_matcher, x0_small, x1_small, x1_full, epsilon, t_val, device, method='lsefm'
            )
            lsefm_mean = float(lsefm_cos.mean())
            results_lsefm[:, ti] += lsefm_mean  # same LSEFM for all sigma rows

            # s_EFM: varies with sigma
            for si, sigma_val in enumerate(sigma_values):
                s_efm_cos = compute_pairing_error_efm(
                    model, flow_matcher, x0_small, x1_small, x1_full, epsilon, t_val, device,
                    method='s_efm', sigma=sigma_val, seed=args.seed + batch_idx * 1000 + ti * 100 + si
                )
                results_s_efm[si, ti] += float(s_efm_cos.mean())

    # Average over batches
    n_batches = args.num_batches
    results_efm /= n_batches
    results_lsefm /= n_batches
    results_s_efm /= n_batches

    # Compute differences
    diff_lsefm = results_lsefm - results_efm
    diff_s_efm = results_s_efm - results_efm

    # Visualization: 2 heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: LSEFM - EFM
    ax1 = axes[0]
    im1 = ax1.imshow(diff_lsefm, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax1.set_xticks(range(len(t_values)))
    ax1.set_xticklabels([f"{t:.2f}" for t in t_values])
    ax1.set_yticks(range(len(sigma_values)))
    ax1.set_yticklabels([f"{s:.1f}" for s in sigma_values])
    ax1.set_xlabel("t")
    ax1.set_ylabel("sigma")
    ax1.set_title("LSEFM cos - EFM cos (positive = LSEFM better)")
    for si in range(len(sigma_values)):
        for ti in range(len(t_values)):
            val = diff_lsefm[si, ti]
            ax1.text(ti, si, f"{val:.3f}", ha="center", va="center", fontsize=9)
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("cos(LSEFM) - cos(EFM)")

    # Right: s_EFM - EFM
    ax2 = axes[1]
    im2 = ax2.imshow(diff_s_efm, aspect="auto", cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    ax2.set_xticks(range(len(t_values)))
    ax2.set_xticklabels([f"{t:.2f}" for t in t_values])
    ax2.set_yticks(range(len(sigma_values)))
    ax2.set_yticklabels([f"{s:.1f}" for s in sigma_values])
    ax2.set_xlabel("t")
    ax2.set_ylabel("sigma")
    ax2.set_title("s_EFM cos - EFM cos (positive = s_EFM better)")
    for si in range(len(sigma_values)):
        for ti in range(len(t_values)):
            val = diff_s_efm[si, ti]
            ax2.text(ti, si, f"{val:.3f}", ha="center", va="center", fontsize=9)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label("cos(s_EFM) - cos(EFM)")

    fig.suptitle("Comparison of EFM variants", fontsize=12)
    plt.tight_layout()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "efm_variants_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved heatmap to {out_path}")

    # Save raw data
    np.savez(
        output_dir / "efm_variants_heatmap_data.npz",
        t_values=t_values,
        sigma_values=sigma_values,
        results_efm=results_efm,
        results_lsefm=results_lsefm,
        results_s_efm=results_s_efm,
        diff_lsefm=diff_lsefm,
        diff_s_efm=diff_s_efm,
    )
    print("Saved raw data to efm_variants_heatmap_data.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--efm_batch_size", type=int, default=8192, help="Large batch for EFM weights (all x1 candidates)")
    parser.add_argument("--pairing_batch_size", type=int, default=128, help="Small batch for pairing (subset of efm_batch)")
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./exp/experiment_results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    run_experiment(args)
