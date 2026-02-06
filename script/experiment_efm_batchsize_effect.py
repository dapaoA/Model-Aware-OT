"""
Experiment: Effect of EFM batch size on cosine similarity (EFM direction vs model prediction).

- Pairing batch: fixed (small) - selects x0, x1 pairs and computes xt
- EFM batch: varied - number of x1 candidates used for closed-form softmax weights

Vary efm_batch_size, fix pairing_batch_size. Compute cos(EFM direction, model prediction)
at different t values. Visualize how cos changes with efm_batch_size.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import create_model
from flow_matcher import create_flow_matcher
from dataset import get_dataset
from utils.efm import efm_closed_form_weights_and_u, lsefm_closed_form_u


def compute_cosine_similarity(vec1, vec2):
    vec1_flat = vec1.reshape(vec1.shape[0], -1)
    vec2_flat = vec2.reshape(vec2.shape[0], -1)
    dot_product = (vec1_flat * vec2_flat).sum(dim=1)
    norm1 = torch.norm(vec1_flat, dim=1)
    norm2 = torch.norm(vec2_flat, dim=1)
    cos_sim = dot_product / (norm1 * norm2 + 1e-8)
    return cos_sim


def compute_efm_cos(model, flow_matcher, x0_small, x1_small, x1_full, epsilon, t_value, device, method='efm'):
    """Compute mean cosine similarity between EFM/LSEFM direction and model prediction.
    
    Args:
        x0_small: [pairing_batch_size, C, H, W] - noise for pairing
        x1_small: [pairing_batch_size, C, H, W] - subset of x1 for pairing (x1_full[:pairing_batch_size])
        x1_full: [efm_batch_size, C, H, W] - full x1 for EFM closed-form weights
        epsilon: [pairing_batch_size, C, H, W] - noise for xt
        method: 'efm' or 'lsefm'
    """
    model.eval()
    batch_size = x0_small.shape[0]
    t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)

    with torch.no_grad():
        indices = torch.randperm(x1_small.shape[0], device=x1_small.device)
        x1_paired = x1_small[indices]
        t_expanded = t.reshape(-1, *([1] * (x0_small.dim() - 1)))
        mu_t = t_expanded * x1_paired + (1 - t_expanded) * x0_small
        sigma_t = flow_matcher.compute_sigma_t(t)
        if isinstance(sigma_t, torch.Tensor):
            sigma_t_expanded = sigma_t.reshape(-1, *([1] * (mu_t.dim() - 1)))
        else:
            sigma_t_expanded = sigma_t
        xt = mu_t + sigma_t_expanded * epsilon

        if method == 'efm':
            xt_flat = xt.reshape(batch_size, -1)
            x1_full_flat = x1_full.reshape(x1_full.shape[0], -1)
            _, ut_theoretical, _ = efm_closed_form_weights_and_u(xt_flat, x1_full_flat, t, eps=1e-6)
            ut_theoretical = ut_theoretical.reshape_as(xt)
        elif method == 'lsefm':
            ut_theoretical = lsefm_closed_form_u(xt, x1_full, t, kernel_size=3, eps=1e-6)
        else:
            raise ValueError(f"Unknown method: {method}")

        vt_predicted = model(xt, t)
        cos_sim = compute_cosine_similarity(ut_theoretical, vt_predicted)

    return cos_sim.cpu().numpy()


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

    flow_matcher = create_flow_matcher("cfm", sigma_fm)

    t_values = args.t_values if args.t_values else [0.05, 0.25, 0.5, 0.75, 0.95]
    efm_batch_sizes = sorted(args.efm_batch_sizes)
    pairing_batch_size = args.pairing_batch_size
    methods = ['efm', 'lsefm']

    # Ensure all efm_batch_sizes >= pairing_batch_size
    efm_batch_sizes = [m for m in efm_batch_sizes if m >= pairing_batch_size]
    if not efm_batch_sizes:
        raise ValueError(f"All efm_batch_sizes must be >= pairing_batch_size ({pairing_batch_size})")

    # [method_idx, efm_batch_idx, t_idx] -> mean cos
    results = np.zeros((len(methods), len(efm_batch_sizes), len(t_values)))
    results_std = np.zeros((len(methods), len(efm_batch_sizes), len(t_values)))

    max_efm_batch = max(efm_batch_sizes)
    dataloader, _ = get_dataset(dataset_name, max_efm_batch, args.data_dir)

    total_steps = args.num_batches * len(efm_batch_sizes) * len(t_values) * len(methods)
    pbar = tqdm(total=total_steps, desc="Computing EFM/LSEFM")
    
    for batch_idx in range(args.num_batches):
        x1_full_batch = next(iter(dataloader))[0].to(device)
        x0_small = torch.randn(pairing_batch_size, *x1_full_batch.shape[1:], device=device)
        epsilon = torch.randn(pairing_batch_size, *x1_full_batch.shape[1:], device=device)

        for ei, efm_size in enumerate(efm_batch_sizes):
            x1_full = x1_full_batch[:efm_size]
            x1_small = x1_full[:pairing_batch_size]

            for ti, t_val in enumerate(t_values):
                for mi, method in enumerate(methods):
                    pbar.set_postfix({
                        'batch': f'{batch_idx+1}/{args.num_batches}',
                        'efm_size': efm_size,
                        't': f'{t_val:.2f}',
                        'method': method
                    })
                    cos_arr = compute_efm_cos(
                        model, flow_matcher,
                        x0_small, x1_small, x1_full, epsilon,
                        t_val, device, method=method,
                    )
                    results[mi, ei, ti] += cos_arr.mean()
                    results_std[mi, ei, ti] += cos_arr.std()
                    pbar.update(1)
    
    pbar.close()

    results /= args.num_batches
    results_std /= args.num_batches

    # Save data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "efm_batchsize_effect.npz",
        efm_batch_sizes=efm_batch_sizes,
        t_values=t_values,
        methods=methods,
        results=results,
        results_std=results_std,
    )
    print(f"Saved data to {output_dir / 'efm_batchsize_effect.npz'}")

    # Visualization: 2 rows x 2 cols (EFM and LSEFM, each with line plot and heatmap)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for mi, method in enumerate(methods):
        # Left: line plot
        ax1 = axes[mi, 0]
        for ti, t_val in enumerate(t_values):
            ax1.plot(
                efm_batch_sizes,
                results[mi, :, ti],
                "o-",
                label=f"t={t_val}",
                linewidth=2,
            )
            ax1.fill_between(
                efm_batch_sizes,
                results[mi, :, ti] - results_std[mi, :, ti],
                results[mi, :, ti] + results_std[mi, :, ti],
                alpha=0.2,
            )
        ax1.set_xscale("log")
        ax1.set_xlabel("EFM batch size (x1 candidates)")
        ax1.set_ylabel("Mean cosine similarity")
        ax1.set_title(f"{method.upper()}: cos vs batch size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: heatmap
        ax2 = axes[mi, 1]
        im = ax2.imshow(results[mi].T, aspect="auto", cmap="viridis")
        ax2.set_xticks(range(len(efm_batch_sizes)))
        ax2.set_xticklabels([str(s) for s in efm_batch_sizes])
        ax2.set_yticks(range(len(t_values)))
        ax2.set_yticklabels([f"{t:.2f}" for t in t_values])
        ax2.set_xlabel("EFM batch size")
        ax2.set_ylabel("t")
        ax2.set_title(f"{method.upper()}: Mean cosine similarity")
        plt.colorbar(im, ax=ax2, label="cos")

    fig.suptitle(f"Effect of batch size on accuracy (pairing_batch={pairing_batch_size})", fontsize=12)
    plt.tight_layout()
    out_path = output_dir / "efm_batchsize_effect.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pairing_batch_size", type=int, default=128, help="Fixed small batch for pairing")
    parser.add_argument("--efm_batch_sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--t_values", type=float, nargs="+", default=None, help="t values to test (default: [0.05, 0.25, 0.5, 0.75, 0.95])")
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./exp/experiment_results")
    args = parser.parse_args()
    torch.manual_seed(42)
    run_experiment(args)
