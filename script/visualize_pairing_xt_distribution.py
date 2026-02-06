"""
Visualize xt distribution at different t for 2D flow matching.

Setup:
- Source (t=0): Gaussian N(0, I)
- Target (t=1): Double moon (from torchcfm)

Pairing methods: CFM (random) vs OTCFM (OT)

For each t in [0.05, 0.25, 0.6, 0.75, 0.95], we compute:
  xt = t * x1_paired + (1-t) * x0_paired

Then plot xt distributions for both pairing methods with different colors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch

from torchcfm.utils import sample_moons
from utils import set_seed


def get_paired_samples_cfm(x0: torch.Tensor, x1: torch.Tensor):
    """CFM: random pairing"""
    indices = torch.randperm(x1.shape[0], device=x1.device)
    x1_paired = x1[indices]
    return x0, x1_paired


def get_paired_samples_otcfm(x0: torch.Tensor, x1: torch.Tensor):
    """OTCFM: OT pairing via Hungarian algorithm"""
    x0_flat = x0.reshape(x0.shape[0], -1)
    x1_flat = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0_flat, x1_flat) ** 2
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(x0.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(x1.device)
    return x0[row_ind], x1[col_ind]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize xt distribution: CFM vs OT pairing (2D)")
    parser.add_argument("--n", type=int, default=500, help="Number of samples")
    parser.add_argument(
        "--t_values",
        type=str,
        default="0.05,0.25,0.6,0.75,0.95",
        help="Comma-separated t values",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./exp/experiment_results", help="Output directory")
    parser.add_argument("--output_name", type=str, default="pairing_xt_distribution_2d.png", help="Output filename")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    t_values = [float(x.strip()) for x in args.t_values.split(",") if x.strip() != ""]
    n = args.n

    # x0: Gaussian N(0, I), 2D
    x0 = torch.randn(n, 2, device=device, dtype=torch.float32)
    # x1: Double moon
    x1 = sample_moons(n).float().to(device)

    # Pairing: CFM (random) and OTCFM
    x0_cfm, x1_cfm = get_paired_samples_cfm(x0, x1)
    x0_ot, x1_ot = get_paired_samples_otcfm(x0, x1)

    # Compute xt at each t for both pairings
    xt_cfm_per_t = []
    xt_ot_per_t = []
    for t in t_values:
        t_val = float(t)
        xt_cfm = t_val * x1_cfm + (1 - t_val) * x0_cfm
        xt_ot = t_val * x1_ot + (1 - t_val) * x0_ot
        xt_cfm_per_t.append(xt_cfm.cpu().numpy())
        xt_ot_per_t.append(xt_ot.cpu().numpy())

    # Plot: one subplot per t
    n_t = len(t_values)
    n_cols = min(5, n_t)
    n_rows = (n_t + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, t in enumerate(t_values):
        ax = axes[idx]
        xt_cfm = xt_cfm_per_t[idx]
        xt_ot = xt_ot_per_t[idx]
        ax.scatter(xt_cfm[:, 0], xt_cfm[:, 1], s=8, alpha=0.6, c="C0", label="CFM (random pair)")
        ax.scatter(xt_ot[:, 0], xt_ot[:, 1], s=8, alpha=0.6, c="C1", label="OT pair")
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    for idx in range(n_t, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("xt distribution: CFM (blue) vs OT (orange) pairing, 2D Gaussian→Moons", fontsize=12, y=1.02)
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
