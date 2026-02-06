"""
Visualize at several t: denoising direction (model) vs CFM direction for one (x0, x1) pair.

- Pick one x1 (CIFAR-10) and one x0 (noise), compute xt = t*x1 + (1-t)*x0.
- Loop over key t values (0.05, 0.25, 0.5, 0.75, 0.95), save same figure per t.
- CFM direction: v_cfm = x1 - x0. Model direction: v_model = model(xt, t).
- From t, integrate one step to t=1: x1_cfm = xt + (1-t)*v_cfm, x1_model = xt + (1-t)*v_model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import create_model
from flow_matcher import create_flow_matcher
from dataset import get_dataset


def velocity_to_display(v, percentile=99):
    """Scale velocity to [0,1] for display (per-channel, symmetric around 0)."""
    v = v.cpu().numpy()
    if v.ndim == 3:
        v = v[np.newaxis, ...]  # [1,C,H,W]
    # v: [B,C,H,W], scale each channel so that percentile maps to 0.5 ± 0.5
    out = np.zeros_like(v, dtype=np.float32)
    for b in range(v.shape[0]):
        for c in range(v.shape[1]):
            vc = v[b, c]
            abs_max = np.percentile(np.abs(vc), percentile)
            if abs_max > 1e-8:
                out[b, c] = np.clip(vc / (2 * abs_max) + 0.5, 0, 1)
            else:
                out[b, c] = 0.5
    return out.squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--x1_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--t_values", type=float, nargs="+", default=[0.05, 0.25, 0.5, 0.75, 0.95])
    parser.add_argument("--output_dir", type=str, default="exp/experiment_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    dataset_name = train_args.get("dataset", "cifar10")
    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Data: one x1, one x0 (fixed for all t)
    dataloader, _ = get_dataset(dataset_name, 1, args.data_dir)
    x1 = next(iter(dataloader))[0].to(device)  # [1, C, H, W]
    x0 = torch.randn_like(x1, device=device)

    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)

    def denorm(x):
        return (x * std + mean).clamp(0, 1).cpu().numpy().squeeze(0)  # [C,H,W]

    def cos_sim(a, b):
        a_flat = a.reshape(1, -1)
        b_flat = b.reshape(1, -1)
        return (a_flat * b_flat).sum() / (a_flat.norm() * b_flat.norm() + 1e-8)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for t_val in args.t_values:
        t = torch.full((1,), t_val, device=device, dtype=torch.float32)
        xt = t_val * x1 + (1 - t_val) * x0

        v_cfm = (x1 - x0).detach()
        with torch.no_grad():
            v_model = model(xt, t).detach()

        dt = 1.0 - t_val
        x1_cfm = (xt + dt * v_cfm).detach()
        x1_model = (xt + dt * v_model).detach()

        im_x0 = denorm(x0)
        im_x1 = denorm(x1)
        im_xt = denorm(xt)
        im_x1_cfm = denorm(x1_cfm)
        im_x1_model = denorm(x1_model)

        disp_v_cfm = velocity_to_display(v_cfm)
        disp_v_model = velocity_to_display(v_model)
        if disp_v_cfm.ndim == 3:
            disp_v_cfm = disp_v_cfm.transpose(1, 2, 0)
            disp_v_model = disp_v_model.transpose(1, 2, 0)

        err_map = (x1 - x1_model).abs().cpu().numpy().squeeze(0).mean(axis=0)
        err_map = np.clip(err_map / (err_map.max() + 1e-8), 0, 1)

        d_cfm_x1 = (v_cfm - x1).detach()
        d_model_x1 = (v_model - x1).detach()
        d_cfm_x0 = (v_cfm + x0).detach()
        d_model_x0 = (v_model + x0).detach()

        cos_x1 = cos_sim(d_cfm_x1, d_model_x1).item()
        cos_x0 = cos_sim(d_cfm_x0, d_model_x0).item()

        disp_d_cfm_x1 = velocity_to_display(d_cfm_x1)
        disp_d_model_x1 = velocity_to_display(d_model_x1)
        disp_d_cfm_x0 = velocity_to_display(d_cfm_x0)
        disp_d_model_x0 = velocity_to_display(d_model_x0)
        if disp_d_cfm_x1.ndim == 3:
            disp_d_cfm_x1 = disp_d_cfm_x1.transpose(1, 2, 0)
            disp_d_model_x1 = disp_d_model_x1.transpose(1, 2, 0)
            disp_d_cfm_x0 = disp_d_cfm_x0.transpose(1, 2, 0)
            disp_d_model_x0 = disp_d_model_x0.transpose(1, 2, 0)

        fig, axes = plt.subplots(5, 3, figsize=(10, 14))

        axes[0, 0].imshow(im_x0.transpose(1, 2, 0))
        axes[0, 0].set_title("x0 (noise)")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(im_x1.transpose(1, 2, 0))
        axes[0, 1].set_title("x1 (GT)")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(im_xt.transpose(1, 2, 0))
        axes[0, 2].set_title(f"xt (t={t_val:.2f})")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(disp_v_cfm)
        axes[1, 0].set_title("v_cfm (CFM direction)")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(disp_v_model)
        axes[1, 1].set_title("v_model (model direction)")
        axes[1, 1].axis("off")
        diff_v = np.abs(disp_v_cfm.astype(np.float32) - disp_v_model.astype(np.float32))
        axes[1, 2].imshow(np.clip(diff_v, 0, 1))
        axes[1, 2].set_title("|v_cfm - v_model|")
        axes[1, 2].axis("off")

        axes[2, 0].imshow(im_x1_cfm.transpose(1, 2, 0))
        axes[2, 0].set_title("x1_cfm (xt + dt*v_cfm)")
        axes[2, 0].axis("off")
        axes[2, 1].imshow(im_x1_model.transpose(1, 2, 0))
        axes[2, 1].set_title("x1_model (xt + dt*v_model)")
        axes[2, 1].axis("off")
        axes[2, 2].imshow(err_map, cmap="hot")
        axes[2, 2].set_title("|x1 - x1_model|")
        axes[2, 2].axis("off")

        axes[3, 0].imshow(disp_d_cfm_x1)
        axes[3, 0].set_title("v_cfm - x1")
        axes[3, 0].axis("off")
        axes[3, 1].imshow(disp_d_model_x1)
        axes[3, 1].set_title("v_model - x1")
        axes[3, 1].axis("off")
        axes[3, 2].axis("off")
        axes[3, 2].text(0.5, 0.5, f"cos(v_cfm-x1,\nv_model-x1)\n= {cos_x1:.4f}", ha="center", va="center", fontsize=12, transform=axes[3, 2].transAxes)

        axes[4, 0].imshow(disp_d_cfm_x0)
        axes[4, 0].set_title("v_cfm + x0")
        axes[4, 0].axis("off")
        axes[4, 1].imshow(disp_d_model_x0)
        axes[4, 1].set_title("v_model + x0")
        axes[4, 1].axis("off")
        axes[4, 2].axis("off")
        axes[4, 2].text(0.5, 0.5, f"cos(v_cfm+x0,\nv_model+x0)\n= {cos_x0:.4f}", ha="center", va="center", fontsize=12, transform=axes[4, 2].transAxes)

        fig.suptitle(f"t={t_val:.2f}: CFM vs model denoising direction and one-step integration to t=1", fontsize=11)
        plt.tight_layout()
        t_tag = f"{int(t_val*100):03d}"
        out_path = out_dir / f"direction_t{t_tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        cos_v = cos_sim(v_cfm, v_model).item()
        print(f"Saved {out_path}  cos(v_cfm,v_model)={cos_v:.4f} cos(v_cfm-x1,v_model-x1)={cos_x1:.4f} cos(v_cfm+x0,v_model+x0)={cos_x0:.4f}")


if __name__ == "__main__":
    main()
