"""
Visualize direction comparison at different t:
- v_model: 8g->2moons trained model's predicted direction
- u_efm: EFM direction (softmax over ~1000 x1 from 2moons)
- u_A: frozen 8g->1moon direction
- u_B: (x1 - xt)/(1-t) toward right moon (continue target)
- u_composite: w_A*u_A + w_B*u_B (EFM-style weights)

x1 is from right moon (continue-training target). t in [0, 0.1, 0.2, 0.5, 0.7, 0.9].
"""
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import create_model, load_model_config
from torchcfm.utils import sample_moons, sample_right_moon, sample_8gaussians


def load_model(ckpt_path, device, dataset_key="8g_to_2moons"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config") or load_model_config(str(_root / "config" / "model_config.yaml"), dataset_key)
    model = create_model(dataset_key, cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


def compute_efm_direction(xt, t, X1_2moons, eps=1e-6):
    """u_efm = sum_j w_j * (X1_j - xt)/(1-t), w = softmax(-||xt - t*X1_j||^2 / (2(1-t)^2))"""
    B, D = xt.shape
    M = X1_2moons.shape[0]
    t = t if t.dim() >= 2 else t[:, None]
    one_minus_t = (1.0 - t).clamp_min(eps)
    # xt [B,D], X1 [M,D] -> diff [B,M,D]
    diff = xt[:, None, :] - t[:, None, :] * X1_2moons[None, :, :]
    scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t.pow(2))
    w = torch.softmax(scores, dim=1)  # [B, M]
    v = (X1_2moons[None, :, :] - xt[:, None, :]) / one_minus_t[:, None, :]  # [B, M, D]
    u_efm = (w[:, :, None] * v).sum(dim=1)  # [B, D]
    return u_efm


def main():
    parser = argparse.ArgumentParser(description="Visualize direction comparison at different t")
    parser.add_argument("--checkpoint_2moons", type=str, default="models/cfm_8g_to_2moons/checkpoint_iter_20000.pt")
    parser.add_argument("--checkpoint_1moon", type=str, default="models/cfm_8g_to_1moon/checkpoint_iter_20000.pt")
    parser.add_argument("--num_pairs", type=int, default=80, help="Number of (x0,x1) pairs per t")
    parser.add_argument("--num_efm", type=int, default=1000, help="Number of 2moon samples for EFM")
    parser.add_argument("--t_values", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.5, 0.7, 0.9])
    parser.add_argument("--subsample", type=int, default=4, help="Show every Nth arrow to reduce clutter")
    parser.add_argument("--arrow_scale", type=float, default=0.15, help="Scale for quiver arrows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="exp/visualize_direction_comparison/directions.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_2moons = load_model(args.checkpoint_2moons, device, "8g_to_2moons")
    model_1moon = load_model(args.checkpoint_1moon, device, "8g_to_1moon")

    # EFM: ~1000 samples from 2moons
    X1_efm = sample_moons(args.num_efm).to(device)  # [M, 2]

    eps = 1e-6
    t_values = args.t_values
    n_t = len(t_values)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    for ti, t_val in enumerate(t_values):
        ax = axes[ti]
        t_scalar = float(t_val)
        t_tensor = torch.full((args.num_pairs,), t_scalar, device=device, dtype=torch.float32)

        x0 = sample_8gaussians(args.num_pairs).to(device)
        x1 = sample_right_moon(args.num_pairs).to(device)  # continue-training target
        xt = (1 - t_scalar) * x0 + t_scalar * x1

        one_minus_t = max(1 - t_scalar, eps)

        with torch.no_grad():
            xt_t = torch.cat([xt, t_tensor[:, None]], dim=-1)

            # 1) v_model: 8g->2moons
            v_model = model_2moons(xt_t)

            # 2) u_efm: EFM over 2moons
            u_efm = compute_efm_direction(xt, t_tensor[:, None], X1_efm, eps)

            # 3) u_A: frozen 8g->1moon
            u_A = model_1moon(xt_t)

            # 4) u_B: (x1 - xt)/(1-t)
            u_B = (x1 - xt) / one_minus_t

            # 5) u_composite: w_A*u_A + w_B*u_B
            x1_star = xt + u_A * one_minus_t
            X1_both = torch.stack([x1_star, x1], dim=1)  # [B, 2, D]
            diff = xt[:, None, :] - t_tensor[:, None, None] * X1_both
            scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t ** 2)
            w = torch.softmax(scores, dim=1)
            u_composite = w[:, 0:1] * u_A + w[:, 1:2] * u_B

        xt_np = xt.cpu().numpy()
        step = max(1, args.subsample)
        idx = slice(None, None, step)

        def quiver(u, color, label):
            u_np = u.cpu().numpy()
            ax.quiver(
                xt_np[idx, 0], xt_np[idx, 1],
                u_np[idx, 0], u_np[idx, 1],
                color=color, alpha=0.7, scale=1.0 / args.arrow_scale,
                scale_units="xy", width=0.003, headwidth=4, label=label
            )

        ax.scatter(xt_np[:, 0], xt_np[:, 1], s=3, alpha=0.4, c="gray")
        quiver(v_model, "blue", "model (8g->2m)")
        quiver(u_efm, "green", "EFM (2moons)")
        quiver(u_A, "red", "A (frozen 1m)")
        quiver(u_B, "orange", "B (→right moon)")
        quiver(u_composite, "purple", "composite")

        ax.set_title(f"t = {t_val}")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)

    plt.suptitle("Direction comparison: model vs EFM vs A vs B vs composite (x1 from right moon)", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {args.output}")

    # Optional: cosine similarity heatmap per t
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
    axes2 = axes2.flatten()
    dir_names = ["model", "EFM", "A", "B", "composite"]

    for ti, t_val in enumerate(t_values):
        ax = axes2[ti]
        t_scalar = float(t_val)
        t_tensor = torch.full((args.num_pairs,), t_scalar, device=device, dtype=torch.float32)
        x0 = sample_8gaussians(args.num_pairs).to(device)
        x1 = sample_right_moon(args.num_pairs).to(device)
        xt = (1 - t_scalar) * x0 + t_scalar * x1
        one_minus_t = max(1 - t_scalar, eps)

        with torch.no_grad():
            xt_t = torch.cat([xt, t_tensor[:, None]], dim=-1)
            v_model = model_2moons(xt_t)
            u_efm = compute_efm_direction(xt, t_tensor[:, None], X1_efm, eps)
            u_A = model_1moon(xt_t)
            u_B = (x1 - xt) / one_minus_t
            x1_star = xt + u_A * one_minus_t
            X1_both = torch.stack([x1_star, x1], dim=1)
            diff = xt[:, None, :] - t_tensor[:, None, None] * X1_both
            scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t ** 2)
            w = torch.softmax(scores, dim=1)
            u_composite = w[:, 0:1] * u_A + w[:, 1:2] * u_B

        dirs = [v_model, u_efm, u_A, u_B, u_composite]
        n_d = len(dirs)
        cos_mat = np.zeros((n_d, n_d))
        for i in range(n_d):
            for j in range(n_d):
                di = dirs[i].cpu().numpy()
                dj = dirs[j].cpu().numpy()
                n_i = np.linalg.norm(di, axis=1, keepdims=True).clip(1e-8, None)
                n_j = np.linalg.norm(dj, axis=1, keepdims=True).clip(1e-8, None)
                cos_mat[i, j] = np.mean((di / n_i) * (dj / n_j))

        im = ax.imshow(cos_mat, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(n_d))
        ax.set_yticks(range(n_d))
        ax.set_xticklabels(dir_names)
        ax.set_yticklabels(dir_names)
        for i in range(n_d):
            for j in range(n_d):
                ax.text(j, i, f"{cos_mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"t = {t_val} (cosine)")
    plt.suptitle("Mean cosine similarity between directions")
    plt.tight_layout()
    out_cos = str(Path(args.output).parent / "directions_cosine.png")
    plt.savefig(out_cos, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cosine heatmap to {out_cos}")

    # Figure 3: single (x0, x1) pair - 5 arrows at each t for visual comparison
    x0_single = sample_8gaussians(1).to(device).squeeze(0)  # [2]
    x1_single = sample_right_moon(1).to(device).squeeze(0)  # [2]
    fig3, axes3 = plt.subplots(2, 3, figsize=(14, 10))
    axes3 = axes3.flatten()
    out_dir = Path(args.output).parent
    arrow_scale_single = 0.25  # for single-point arrows

    for ti, t_val in enumerate(t_values):
        ax = axes3[ti]
        t_scalar = float(t_val)
        xt_single = (1 - t_scalar) * x0_single + t_scalar * x1_single
        one_minus_t = max(1 - t_scalar, eps)

        xt_batch = xt_single.unsqueeze(0)
        t_batch = torch.full((1,), t_scalar, device=device, dtype=torch.float32)

        with torch.no_grad():
            xt_t = torch.cat([xt_batch, t_batch[:, None]], dim=-1)
            v_model = model_2moons(xt_t).squeeze(0)
            u_efm = compute_efm_direction(xt_batch, t_batch[:, None], X1_efm, eps).squeeze(0)
            u_A = model_1moon(xt_t).squeeze(0)
            u_B = (x1_single - xt_single) / one_minus_t
            x1_star = xt_single + u_A * one_minus_t
            X1_both = torch.stack([x1_star, x1_single], dim=0).unsqueeze(0)  # [1, 2, 2]
            diff = xt_batch[:, None, :] - t_batch[:, None, None] * X1_both
            scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t ** 2)
            w = torch.softmax(scores, dim=1)
            u_composite = (w[:, 0:1] * u_A.unsqueeze(0) + w[:, 1:2] * u_B.unsqueeze(0)).squeeze(0)

        px, py = xt_single.cpu().numpy()
        arrows = [
            (v_model, "blue", "model"),
            (u_efm, "green", "EFM"),
            (u_A, "red", "A"),
            (u_B, "orange", "B"),
            (u_composite, "purple", "composite"),
        ]
        for u_vec, color, label in arrows:
            u = u_vec.cpu().numpy()
            ax.quiver(px, py, u[0], u[1], color=color, alpha=0.9, scale=1.0 / arrow_scale_single,
                      scale_units="xy", width=0.008, headwidth=5, headlength=6, label=label)

        # Background: 2moons + right moon (light)
        moon2 = sample_moons(300).numpy()
        moon1 = sample_right_moon(150).numpy()
        ax.scatter(moon2[:, 0], moon2[:, 1], s=8, alpha=0.15, c="gray", label="_nolegend_")
        ax.scatter(moon1[:, 0], moon1[:, 1], s=8, alpha=0.2, c="orangered", label="_nolegend_")
        # x0, x1, xt
        ax.scatter(*x0_single.cpu().numpy(), s=80, c="black", marker="o", zorder=5, label="x0")
        ax.scatter(*x1_single.cpu().numpy(), s=80, c="orangered", marker="*", zorder=5, label="x1")
        ax.scatter(px, py, s=120, c="lime", marker="s", zorder=5, edgecolors="black", linewidths=2, label="xt")
        ax.plot([x0_single[0].item(), x1_single[0].item()], [x0_single[1].item(), x1_single[1].item()],
                "k--", alpha=0.5, linewidth=1)
        ax.set_title(f"t = {t_val}")
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)

    plt.suptitle("Single (x0,x1) pair: 5 direction arrows at each t", fontsize=11)
    plt.tight_layout()
    out_single = str(out_dir / "directions_single_x1.png")
    plt.savefig(out_single, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved single-x1 figure to {out_single}")


if __name__ == "__main__":
    main()
