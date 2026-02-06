"""
Experiment: At what t does the closest x1 to xt change from the paired image to another?

Each run: randomly pick ONE image from the batch, fix one random noise x0,
vary t: xt = t*x1 + (1-t)*x0. At each t, find which image in batch is closest to xt.
When closest != self, crossover ("切了").
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import get_dataset


def denormalize_cifar10(x):
    """Reverse CIFAR-10 normalization for display."""
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)
    return x * std + mean


def run_experiment(args):
    """One batch, pick num_runs random (x0,x1) pairs within batch, process all in parallel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, _ = get_dataset("cifar10", args.batch_size, args.data_dir)

    x1 = next(iter(dataloader))[0].to(device)  # [B, C, H, W]
    B = x1.shape[0]
    N = min(args.num_runs, 20)

    # Pick N random indices, N random x0
    i_idx = torch.randint(0, B, (N,), device=device)  # [N]
    x0 = torch.randn(N, *x1.shape[1:], device=device)  # [N, C, H, W]
    x1_self = x1[i_idx]  # [N, C, H, W]

    t_grid = np.linspace(1.0, 1e-4, args.t_steps)
    x1_flat = x1.reshape(B, -1)

    # Track per trial: crossover t, diverged j (unset = -1)
    t_cross = torch.full((N,), float("nan"), device=device)
    j_div = torch.full((N,), -1, dtype=torch.long, device=device)
    done = torch.zeros(N, dtype=torch.bool, device=device)

    for t in t_grid:
        if t < 1e-6:
            break
        t_t = torch.tensor(t, device=device, dtype=x1.dtype)
        xt = t_t * x1_self + (1 - t_t) * x0  # [N, C, H, W]
        dists = torch.cdist(xt.reshape(N, -1), x1_flat)  # [N, B]
        closest = dists.argmin(dim=1)  # [N]
        crossed = (closest != i_idx) & (~done)
        if crossed.any():
            t_cross[crossed] = t
            j_div[crossed] = closest[crossed]
            done = done | crossed
        if done.all():
            break

    # Build results
    all_results = []
    for k in range(N):
        if j_div[k].item() >= 0:
            all_results.append({
                "x1_self": x1_self[k].cpu().clone(),
                "x1_diverged": x1[j_div[k]].cpu().clone(),
                "t": float(t_cross[k].item()),
                "i": i_idx[k].item(),
                "j": j_div[k].item(),
            })
    return all_results


def plot_results(results, output_path):
    """Left: fixed 5x2 grid for 10 pairs [self|div]. Right: t distribution. Fixed fig size."""
    n = len(results)
    if n == 0:
        print("No crossover events found.")
        return

    n_display = min(n, 10)
    # Fixed grid: 5 rows x 2 pair-cols = 10 slots. Each pair = self|div. So 5x4 image grid.
    n_rows, n_pair_cols = 5, 2
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    gs_outer = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.55], wspace=0.1)
    gs_left = gs_outer[0].subgridspec(n_rows, 4, hspace=0.08, wspace=0.03)  # 5 rows, 4 imgs/row

    for idx in range(n_display):
        r = results[idx]
        x1_self = r["x1_self"]
        x1_div = r["x1_diverged"]
        t_val = r["t"]

        x1_self_np = denormalize_cifar10(x1_self.numpy()).squeeze(0)
        x1_div_np = denormalize_cifar10(x1_div.numpy()).squeeze(0)
        x1_self_np = np.clip(x1_self_np.transpose(1, 2, 0), 0, 1)
        x1_div_np = np.clip(x1_div_np.transpose(1, 2, 0), 0, 1)

        row, col = idx // n_pair_cols, (idx % n_pair_cols) * 2
        ax1 = fig.add_subplot(gs_left[row, col])
        ax1.imshow(x1_self_np)
        ax1.axis("off")
        ax1.text(0.5, -0.02, f"t={t_val:.3f}", transform=ax1.transAxes, ha="center", fontsize=7)
        if row == 0 and col == 0:
            ax1.set_title("Self")

        ax2 = fig.add_subplot(gs_left[row, col + 1])
        ax2.imshow(x1_div_np)
        ax2.axis("off")
        if row == 0 and col == 0:
            ax2.set_title("Diverged")

    t_vals = [r["t"] for r in results]
    ax_hist = fig.add_subplot(gs_outer[1])
    ax_hist.hist(t_vals, bins=min(30, len(t_vals)), range=(0, 1), color="steelblue", edgecolor="white", alpha=0.8)
    ax_hist.axvline(np.mean(t_vals), color="red", linestyle="--", label=f"Mean={np.mean(t_vals):.3f}")
    ax_hist.axvline(np.median(t_vals), color="orange", linestyle="-.", label=f"Median={np.median(t_vals):.3f}")
    ax_hist.set_xlabel("Crossover t")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(0, 1)
    ax_hist.set_title(f"Distribution (n={len(t_vals)})")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Find t where closest x1 to xt changes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (images to search in)")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of trials (pick 1 img per trial, max 20 shown)")
    parser.add_argument("--t_steps", type=int, default=200, help="Grid resolution for t")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="./exp/experiment_xt_crossover_results.png")
    args = parser.parse_args()

    results = run_experiment(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(results, output_path)

    if results:
        t_vals = [r["t"] for r in results]
        print(f"\nCrossover t: mean={np.mean(t_vals):.4f}, median={np.median(t_vals):.4f}, "
              f"std={np.std(t_vals):.4f}, min={np.min(t_vals):.4f}, max={np.max(t_vals):.4f}")


if __name__ == "__main__":
    main()
