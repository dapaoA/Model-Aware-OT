"""
Visualize how closed-form EFM weights change over time t.

xt = t * x1 + (1-t) * x0 (linear interpolation between x0 and one x1).
x0: fixed noise, x1: one CIFAR-10 image (default: first of N).

At each t, we compute EFM weights at xt over all N CIFAR-10 images:
  w_j(t) = softmax( -||xt - t * x1_j||^2 / (2 (1-t)^2) )

We plot:
- an "alluvial/river" style stack plot showing w_j as a function of t
- a table-like heatmap for readability
- per-t display (max-weight img, xt, Tweedie x1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from torchvision import datasets, transforms

from model import create_model
from utils import set_seed
from utils.efm import efm_closed_form_weights_and_u, lsefm_block_weights_and_u


def denorm_cifar10(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze closed-form EFM weights over t (fixed x)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset dir")
    parser.add_argument("--n", type=int, default=4, help="Number of CIFAR-10 samples (X1) to compare")
    parser.add_argument("--x1_idx", type=int, default=0, help="Index of x1 to use for linear interpolation xt = t*x1 + (1-t)*x0 (default 0)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--t_values",
        type=str,
        default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1",
        help="Comma-separated t values",
    )
    parser.add_argument("--output_dir", type=str, default="./exp/experiment_results", help="Output directory")
    parser.add_argument("--output_name", type=str, default="efm_weights_over_t_n4.png", help="Output PNG filename")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to CFM model checkpoint (required for Tweedie x1 row: one-step from t to 1 using model v)",
    )
    parser.add_argument(
        "--x0_path",
        type=str,
        default=None,
        help="Path to locked x0 tensor (.pt). If provided, load x0 from file instead of sampling.",
    )
    parser.add_argument(
        "--save_x0",
        type=str,
        default=None,
        help="If provided, save the x0 used to this path (.pt) for future --x0_path.",
    )
    parser.add_argument(
        "--save_topk",
        action="store_true",
        help="Save top-k similar images at each t to output_dir/n/ (stop when max weight > 95%%).",
    )
    parser.add_argument("--topk_k", type=int, default=5, help="Max number of top images to save (default 5).")
    parser.add_argument("--topk_stop_weight", type=float, default=0.95, help="Stop when max weight exceeds this (default 0.95).")
    parser.add_argument(
        "--lsefm_blocks",
        type=int,
        default=None,
        help="If set (e.g. 2 or 4), use block LSEFM: n_blocks x n_blocks blocks, same x0/X1/t. Plot avg weights; bottom row shows stitched image (per-block winner).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    t_values: List[float] = [float(x.strip()) for x in args.t_values.split(",") if x.strip() != ""]
    if len(t_values) < 2:
        raise ValueError("Need at least 2 t values to visualize transitions")

    # x0: load from file or sample (use separate RNG so x0 is locked regardless of n)
    if args.x0_path:
        x_img = torch.load(args.x0_path, map_location=device, weights_only=True)
        if x_img.dim() == 3:
            x_img = x_img.unsqueeze(0)
        x_img = x_img.to(device=device, dtype=torch.float32)
        print(f"Loaded locked x0 from {args.x0_path}")
    else:
        g = torch.Generator(device=device).manual_seed(args.seed + 12345)
        x_img = torch.randn((1, 3, 32, 32), device=device, dtype=torch.float32, generator=g)

    if args.save_x0:
        torch.save(x_img.cpu(), args.save_x0)
        print(f"Saved x0 to {args.save_x0}")

    # CIFAR-10 transform consistent with training/infer (no random flip for stability)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tfm)

    # Pick N fixed samples deterministically (seeded)
    idxs = torch.randperm(len(ds))[: args.n].tolist()
    X1_imgs = torch.stack([ds[i][0] for i in idxs], dim=0).to(device)  # (N,3,32,32)

    # Flatten to [B,D] and [M,D]
    x = x_img.reshape(1, -1)
    X1 = X1_imgs.reshape(args.n, -1)

    # xt = t * x1_paired + (1-t) * x0 (linear interpolation)
    x1_idx = min(args.x1_idx, args.n - 1)
    x1_paired = X1[x1_idx : x1_idx + 1]  # [1, D]
    x1_paired_img = X1_imgs[x1_idx : x1_idx + 1]  # [1, 3, 32, 32]
    eps_t = 1e-5

    use_lsefm = args.lsefm_blocks is not None
    n_blocks = args.lsefm_blocks if use_lsefm else 1
    W_blocks_list = []  # per t: [n_blocks^2, N] when LSEFM
    W = []
    xt_list = []
    for t in t_values:
        t_val = max(t, eps_t)
        t_tensor = torch.tensor([t], device=device, dtype=torch.float32)
        xt = t_val * x1_paired + (1 - t_val) * x  # [1, D]
        xt_list.append(xt.squeeze(0))
        if use_lsefm:
            xt_4d = t_val * x1_paired_img + (1 - t_val) * x_img  # [1, 3, 32, 32]
            w_blocks, _ = lsefm_block_weights_and_u(xt_4d, X1_imgs, t_tensor, n_blocks=n_blocks, eps=1e-6)
            w_blocks_np = w_blocks[0].detach().cpu().numpy()  # [n_blocks^2, N]
            W_blocks_list.append(w_blocks_np)
            W.append(w_blocks_np.mean(axis=0))  # [N] average over blocks
        else:
            w, _, _ = efm_closed_form_weights_and_u(xt, X1, t_tensor, eps=1e-6)
            W.append(w[0].detach().cpu().numpy())
    W = np.stack(W, axis=0)  # [T, N]

    argmax_per_t = W.argmax(axis=1)  # [T]
    # Stitched image per t (when LSEFM: each block from its winner)
    stitched_imgs = None
    if use_lsefm and W_blocks_list:
        h_b, w_b = 32 // n_blocks, 32 // n_blocks
        stitched_imgs = []
        for ti in range(len(t_values)):
            stitched = torch.zeros(1, 3, 32, 32, device=device, dtype=X1_imgs.dtype)
            wb = W_blocks_list[ti]  # [n_blocks^2, N]
            for block_idx in range(n_blocks * n_blocks):
                bi, bj = block_idx // n_blocks, block_idx % n_blocks
                i0, i1 = bi * h_b, (bi + 1) * h_b
                j0, j1 = bj * w_b, (bj + 1) * w_b
                winner = int(wb[block_idx].argmax())
                stitched[:, :, i0:i1, j0:j1] = X1_imgs[winner : winner + 1, :, i0:i1, j0:j1]
            stitched_imgs.append(stitched.squeeze(0))
        stitched_imgs = torch.stack(stitched_imgs, dim=0)  # [T, 3, 32, 32]
    xt_all = torch.stack(xt_list, dim=0)  # [T, D]
    xt_imgs = xt_all.reshape(-1, 3, 32, 32)  # [T, 3, 32, 32]

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        train_args = ckpt.get("args", {})
        model_config = ckpt.get("model_config", {})
        dataset_name = train_args.get("dataset", "cifar10")
        model = create_model(dataset_name, model_config, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        x1_tweedie_list = []
        with torch.no_grad():
            for ti, t in enumerate(t_values):
                xt_img = xt_imgs[ti : ti + 1]  # [1, 3, 32, 32]
                t_t = torch.full((1,), t, device=device, dtype=torch.float32)
                v_pred = model(xt_img, t_t)  # [1, 3, 32, 32]
                x1_tweedie = xt_img + (1.0 - t) * v_pred
                x1_tweedie_list.append(x1_tweedie.squeeze(0))
        x1_tweedie_imgs = torch.stack(x1_tweedie_list, dim=0)
        has_tweedie = True
    else:
        x1_tweedie_imgs = xt_imgs
        has_tweedie = False

    # Save a figure: images + stackplot + heatmap + per-t display
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # When save_topk, main figure also goes to output_dir/n/
    fig_dir = (out_dir / str(args.n)) if args.save_topk else out_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    if args.save_topk:
        out_name = "efm_weights_over_t.png"
        if use_lsefm:
            out_name = f"efm_weights_over_t_blocks{n_blocks}.png"
    else:
        out_name = args.output_name
        if use_lsefm:
            base, ext = args.output_name.rsplit(".", 1) if "." in args.output_name else (args.output_name, "png")
            out_name = f"{base}_blocks{n_blocks}.{ext}"
    out_path = fig_dir / out_name

    n = args.n
    labels = [f"img_{k}" for k in range(n)]
    n_t = len(t_values)
    n_t_display = min(n_t, 10)
    if n_t > n_t_display:
        t_display_idx = np.linspace(0, n_t - 1, n_t_display, dtype=int)
    else:
        t_display_idx = np.arange(n_t)

    n_cols_bottom = len(t_display_idx)
    show_legend = n <= 16
    legend_fontsize = max(6, 10 - n // 4) if show_legend else 9
    legend_ncol = min(n, 8) if show_legend else 1
    n_rows = 3 if has_tweedie else 2
    row0_label = "stitched (per-block winner)" if use_lsefm else "max-weight img"
    row_labels = (
        [row0_label, "xt (denoised at t)", "Tweedie x1 (1-step via model v)"]
        if has_tweedie
        else [row0_label, "xt (denoised at t)"]
    )

    fig = plt.figure(figsize=(max(14, n_cols_bottom * 1.5), 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 2.0], hspace=0.4)

    # 1) Stack plot (was section 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.stackplot(t_values, W.T, labels=labels, alpha=0.85)
    ax1.set_xlim(min(t_values), max(t_values))
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("t")
    ax1.set_ylabel("weight w_j(t)")
    ax1.set_title("EFM weights over t (fixed noise x)" + (f" [block LSEFM {n_blocks}x{n_blocks}]" if use_lsefm else ""), fontsize=12)
    ax1.grid(True, alpha=0.25)
    if show_legend:
        ax1.legend(loc="upper left", ncol=legend_ncol, fontsize=legend_fontsize)

    # 2) Heatmap (was section 3)
    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(W.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    y_step = max(1, n // 20)
    y_ticks = list(range(0, n, y_step))
    if y_ticks[-1] != n - 1:
        y_ticks.append(n - 1)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels([labels[i] for i in y_ticks], fontsize=max(6, 10 - n // 16))
    ax2.set_xticks(range(len(t_values)))
    ax2.set_xticklabels([f"{t:.2f}" for t in t_values], rotation=0)
    ax2.set_xlabel("t")
    ax2.set_title("Weights heatmap (rows: samples, cols: t)", fontsize=12)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cbar.set_label("w")

    # 3) Per-t display (was section 4)
    ax3 = fig.add_subplot(gs[2, 0])
    n_cols = len(t_display_idx)
    imgs_for_grid = []
    for row_idx in range(n_rows):
        for c, ti in enumerate(t_display_idx):
            if row_idx == 0:
                if stitched_imgs is not None:
                    img = stitched_imgs[ti]  # stitched per-block winner
                else:
                    img = X1_imgs[argmax_per_t[ti]]
            elif row_idx == 1:
                img = xt_imgs[ti]
            else:
                img = x1_tweedie_imgs[ti]
            imgs_for_grid.append(denorm_cifar10(img.unsqueeze(0)).squeeze(0))
    imgs_stack = torch.stack(imgs_for_grid, dim=0)
    grid_bottom = vutils.make_grid(imgs_stack, nrow=n_cols, padding=2, pad_value=1.0)
    grid_bottom_np = grid_bottom.permute(1, 2, 0).cpu().numpy()
    ax3.imshow(grid_bottom_np)
    title = "Per t: max-weight image | xt (denoised at t) | Tweedie x1 (1-step t→1 via model v)"
    if not has_tweedie:
        title += " [Tweedie omitted: use --checkpoint]"
    ax3.set_title(title, fontsize=11)
    ax3.axis("off")
    for c, ti in enumerate(t_display_idx):
        ax3.text(
            c / n_cols + 0.5 / n_cols,
            -0.02,
            f"t={t_values[ti]:.2f}",
            transform=ax3.transAxes,
            ha="center",
            fontsize=9,
        )
    y_offsets = [0.25, 0.75] if n_rows == 2 else [0.17, 0.5, 0.83]
    for r, lbl in enumerate(row_labels):
        ax3.text(-0.02, 1 - y_offsets[r], lbl, transform=ax3.transAxes, va="center", ha="right", fontsize=9, rotation=90)

    fig.suptitle("Closed-form EFM weight dynamics" + (f" (block LSEFM {n_blocks}x{n_blocks})" if use_lsefm else ""), fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Save top-k images at each t to output_dir/n/ (t>0, stop when max>95%)
    if args.save_topk:
        # Precompute <x0, x1_j> and ||x1_paired - x1_j|| for all N
        dot_x0_x1 = (x @ X1.T).squeeze(0).cpu().numpy()  # [N]
        diff_paired = x1_paired - X1  # [N, D]
        norm_x1_diff = torch.norm(diff_paired, dim=1).cpu().numpy()  # [N]
        mean_dot = float(np.mean(dot_x0_x1))
        mean_norm = float(np.mean(norm_x1_diff))

        topk_dir = out_dir / str(args.n)
        topk_dir.mkdir(parents=True, exist_ok=True)
        k = args.topk_k
        for ti, t in enumerate(t_values):
            if t < 0.05 or t > 0.95:
                continue
            w_row = W[ti]
            max_w = float(np.max(w_row))
            order = np.argsort(w_row)[::-1]
            indices = order[:k].tolist()
            imgs = [denorm_cifar10(X1_imgs[i].unsqueeze(0)).squeeze(0) for i in indices]
            weights = [w_row[i] for i in indices]
            # When LSEFM blocks: add stitched (mixed) image row above top-k
            if use_lsefm and stitched_imgs is not None:
                stitched_one = denorm_cifar10(stitched_imgs[ti].unsqueeze(0)).squeeze(0)
                xt_one = denorm_cifar10(xt_imgs[ti].unsqueeze(0)).squeeze(0)
                row_stitched_xt = torch.stack([stitched_one, xt_one], dim=0)
                grid_upper = vutils.make_grid(row_stitched_xt, nrow=2, padding=4, pad_value=1.0)
                grid_lower = vutils.make_grid(torch.stack(imgs, dim=0), nrow=len(imgs), padding=4, pad_value=1.0)
                fig_t, ax_t = plt.subplots(2, 1, figsize=(2 * max(2, len(imgs)), 5.0), gridspec_kw={"height_ratios": [1, 1.2]})
                ax_t[0].imshow(grid_upper.permute(1, 2, 0).cpu().numpy())
                ax_t[0].axis("off")
                ax_t[0].set_title(f"t={t:.2f} — stitched (per-block winner) | xt")
                ax_t[1].imshow(grid_lower.permute(1, 2, 0).cpu().numpy())
                ax_t[1].axis("off")
                n_img = len(imgs)
                for ii, (idx, ww) in enumerate(zip(indices, weights)):
                    cx = ii / n_img + 0.5 / n_img
                    dot_val = dot_x0_x1[idx]
                    norm_val = norm_x1_diff[idx]
                    ax_t[1].text(cx, -0.06, f"img_{idx} w={ww:.2%}", transform=ax_t[1].transAxes, ha="center", fontsize=9)
                    ax_t[1].text(cx, -0.12, f"<x0,x1>={dot_val:.1f}", transform=ax_t[1].transAxes, ha="center", fontsize=8)
                    ax_t[1].text(cx, -0.18, f"||x1*-x1||={norm_val:.1f}", transform=ax_t[1].transAxes, ha="center", fontsize=8)
                ax_t[1].set_title(f"Top-{k} by weight (max={max_w:.2%}) | mean <x0,x1>={mean_dot:.1f} mean ||x1*-x1||={mean_norm:.1f}")
            else:
                grid = vutils.make_grid(torch.stack(imgs, dim=0), nrow=len(imgs), padding=4, pad_value=1.0)
                grid_np = grid.permute(1, 2, 0).cpu().numpy()
                fig_t, ax_t = plt.subplots(1, 1, figsize=(2 * len(imgs), 3.0))
                ax_t = [ax_t]
                ax_t[0].imshow(grid_np)
                ax_t[0].axis("off")
                n_img = len(imgs)
                for ii, (idx, ww) in enumerate(zip(indices, weights)):
                    cx = ii / n_img + 0.5 / n_img
                    dot_val = dot_x0_x1[idx]
                    norm_val = norm_x1_diff[idx]
                    ax_t[0].text(cx, -0.06, f"img_{idx} w={ww:.2%}", transform=ax_t[0].transAxes, ha="center", fontsize=9)
                    ax_t[0].text(cx, -0.12, f"<x0,x1>={dot_val:.1f}", transform=ax_t[0].transAxes, ha="center", fontsize=8)
                    ax_t[0].text(cx, -0.18, f"||x1*-x1||={norm_val:.1f}", transform=ax_t[0].transAxes, ha="center", fontsize=8)
                ax_t[0].set_title(f"t={t:.2f} (max={max_w:.2%}) | mean <x0,x1>={mean_dot:.1f} mean ||x1*-x1||={mean_norm:.1f}")
            t_name = f"t_{int(round(t * 100)):03d}"
            fig_t.savefig(topk_dir / f"{t_name}.png", dpi=120, bbox_inches="tight")
            plt.close(fig_t)
        print(f"Saved top-k images to {topk_dir}")

    print(f"Saved figure to: {out_path}")
    print("Weights (rows=t, cols=img_k):")
    for ti, t in enumerate(t_values):
        row = " ".join([f"{W[ti, k]:.3f}" for k in range(args.n)])
        print(f"t={t:>5.2f}: {row}")


if __name__ == "__main__":
    main()

