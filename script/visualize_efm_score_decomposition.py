"""
Visualize EFM score decomposition: ||xt - t*x1_j||^2

xt = t*x1* + (1-t)*x0  (x1* = paired target, x0 = noise)
diff_j = xt - t*x1_j = t*(x1* - x1_j) + (1-t)*x0

||diff_j||^2 = ||t*(x1*-x1_j) + (1-t)*x0||^2
             = t^2||x1*-x1_j||^2 + (1-t)^2||x0||^2 + 2t(1-t)<x1*-x1_j, x0>

The j-dependent part (x1* and x0 fixed):
  term_A = t^2 * ||x1* - x1_j||^2
  term_B = 2t(1-t) * <x1* - x1_j, x0> = 2t(1-t)*(<x1*,x0> - <x1_j,x0>)

So: ||diff_j||^2 = term_A + term_B + const
    score_j = -||diff_j||^2 / (2(1-t)^2)

We visualize how term_A and term_B vary with t for top/bottom scoring x1_j.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="exp/experiment_results/efm_score_decomposition.png")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tfm)
    idxs = torch.randperm(len(ds))[: args.n].tolist()
    X1 = torch.stack([ds[i][0] for i in idxs], dim=0).to(device).reshape(args.n, -1)

    g = torch.Generator(device=device).manual_seed(args.seed + 12345)
    x0 = torch.randn(1, 3, 32, 32, device=device, generator=g).reshape(1, -1)
    x1_star = X1[0:1]  # paired target

    t_values = np.linspace(0.05, 0.95, 19)
    eps = 1e-6

    # For each t, compute for all j:
    # diff_j = xt - t*x1_j, xt = t*x1* + (1-t)*x0
    # term_A = t^2 * ||x1* - x1_j||^2
    # term_B = 2t(1-t) * <x1* - x1_j, x0>
    # ||diff_j||^2 (full) for verification

    results = []
    for t in t_values:
        t = float(t)
        one_minus_t = 1 - t
        xt = t * x1_star + one_minus_t * x0  # [1, D]

        # diff_j = xt - t*x1_j  [1, N, D] then [N, D]
        diff = xt - t * X1  # [N, D] (broadcast)

        dist_sq_full = (diff ** 2).sum(dim=1).cpu().numpy()  # [N]

        # term_A = t^2 * ||x1* - x1_j||^2
        delta = x1_star - X1  # [N, D]
        term_A = (t ** 2) * (delta ** 2).sum(dim=1).cpu().numpy()  # [N]

        # term_B = 2t(1-t) * <x1* - x1_j, x0>
        # <delta_j, x0>  [N]
        dot_delta_x0 = (delta * x0).sum(dim=1).cpu().numpy()
        term_B = 2 * t * one_minus_t * dot_delta_x0  # [N]

        # constant: (1-t)^2 ||x0||^2
        const = (one_minus_t ** 2) * (x0 ** 2).sum().item()

        # Verify: ||diff||^2 = term_A + term_B + const?
        # diff = t*delta + (1-t)*x0, so ||diff||^2 = t^2||delta||^2 + (1-t)^2||x0||^2 + 2t(1-t)<delta,x0>
        # = term_A + const + term_B  (since const = (1-t)^2||x0||^2)
        recon = term_A + term_B + const
        err = np.abs(dist_sq_full - recon).max()
        if err > 1e-2:
            print(f"Warning: decomposition max error {err:.2e} at t={t}")

        # scores = -||diff||^2 / (2(1-t)^2)
        scores = -dist_sq_full / (2 * (one_minus_t ** 2))
        w = np.exp(scores - scores.max())
        w = w / w.sum()

        results.append({
            "t": t,
            "term_A": term_A,
            "term_B": term_B,
            "dist_sq": dist_sq_full,
            "scores": scores,
            "w": w,
        })

    # Plot: pick a few j (e.g. j=0 paired, j=1,2,3,4 others), show term_A, term_B, w over t
    n_show = min(8, args.n)
    j_indices = list(range(n_show))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    t_arr = [r["t"] for r in results]

    # Row 0: term_A for each j over t
    ax = axes[0, 0]
    for j in j_indices:
        vals = [r["term_A"][j] for r in results]
        ax.plot(t_arr, vals, "-o", markersize=3, label=f"j={j}")
    ax.set_xlabel("t")
    ax.set_ylabel("term_A = t^2 ||x1*-x1_j||^2")
    ax.set_title("Term A (squared distance to paired x1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 0 right: term_B for each j over t
    ax = axes[0, 1]
    for j in j_indices:
        vals = [r["term_B"][j] for r in results]
        ax.plot(t_arr, vals, "-o", markersize=3, label=f"j={j}")
    ax.set_xlabel("t")
    ax.set_ylabel("term_B = 2t(1-t) <x1*-x1_j, x0>")
    ax.set_title("Term B (cross term with x0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1: ||diff||^2 = term_A + term_B + const for each j
    ax = axes[1, 0]
    for j in j_indices:
        vals = [r["dist_sq"][j] for r in results]
        ax.plot(t_arr, vals, "-o", markersize=3, label=f"j={j}")
    ax.set_xlabel("t")
    ax.set_ylabel("||xt - t*x1_j||^2")
    ax.set_title("Full squared distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1 right: weights w over t
    ax = axes[1, 1]
    for j in j_indices:
        vals = [r["w"][j] for r in results]
        ax.plot(t_arr, vals, "-o", markersize=3, label=f"j={j}")
    ax.set_xlabel("t")
    ax.set_ylabel("w_j (EFM weight)")
    ax.set_title("EFM weights over t")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: -term_A and -term_B (to show what drives score up)
    # score = -||diff||^2 / (2(1-t)^2), so lower ||diff||^2 => higher score
    ax = axes[2, 0]
    for j in j_indices:
        vals_A = [r["term_A"][j] for r in results]
        vals_B = [r["term_B"][j] for r in results]
        ax.plot(t_arr, vals_A, "-", markersize=2, label=f"j={j} term_A")
        ax.plot(t_arr, vals_B, "--", markersize=2, label=f"j={j} term_B")
    ax.set_xlabel("t")
    ax.set_ylabel("term value")
    ax.set_title("Term A (solid) vs Term B (dashed)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Row 2 right: ratio or contribution
    ax = axes[2, 1]
    # Show at t=0.5: term_A and term_B for all j, bar chart
    mid_idx = len(results) // 2
    r = results[mid_idx]
    t_mid = r["t"]
    x_j = np.arange(args.n)
    w = 0.35
    ax.bar(x_j - w/2, r["term_A"], w, label="term_A")
    ax.bar(x_j + w/2, r["term_B"], w, label="term_B")
    ax.set_xlabel("j (x1 index)")
    ax.set_ylabel("value")
    ax.set_title(f"Term A vs B at t={t_mid:.2f} (all {args.n} samples)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("EFM score decomposition: ||xt-t*x1_j||^2 = t^2||x1*-x1_j||^2 + 2t(1-t)<x1*-x1_j,x0> + (1-t)^2||x0||^2", fontsize=10)
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
