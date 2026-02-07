"""
Continue learning: load a model trained on 8g_to_1moon and continue training
using ONLY the other moon (the one 1-moon didn't have). So we only train on
the missing part, not on full 2 moons.
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchdyn.core import NeuralODE
from tqdm import tqdm
from PIL import Image

from flow_matcher import create_flow_matcher
from model import create_model, load_model_config, create_default_config
from utils import set_seed
from utils.common import get_rng_states
from torchcfm.utils import sample_8gaussians, sample_right_moon, sample_moons


def _visualize_inference_distribution(model, device, save_path, num_samples=2000, num_steps=50, seed=42):
    """Generate samples from model (8g->integrate) and plot 2D distribution with true 2moons overlay."""
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0 = sample_8gaussians(num_samples).to(device)

    def model_wrapper(t, x, **kwargs):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return model(torch.cat([x, t[:, None]], 1))

    node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        t_span = torch.linspace(0, 1, num_steps + 1).to(device)
        traj = node.trajectory(x0, t_span=t_span)
    gen = traj[-1].cpu().numpy()
    model.train()

    true_2moons = sample_moons(num_samples).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(true_2moons[:, 0], true_2moons[:, 1], s=4, alpha=0.3, c="gray", label="true 2moons")
    ax.scatter(gen[:, 0], gen[:, 1], s=4, alpha=0.6, c="blue", label="generated")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title(f"Inference distribution (n={num_samples})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def train_continue_2moons(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    source_dataset = checkpoint_args.get("dataset", "8g_to_1moon")
    method = checkpoint_args.get("method", args.method)
    assert source_dataset == "8g_to_1moon", (
        f"Continue learning expects a 8g_to_1moon checkpoint, got dataset={source_dataset}"
    )

    set_seed(args.seed)
    iteration = 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    loss_type = (args.loss or "mse").lower()
    model_name = args.output_name or (
        f"{method}_8g_to_1moon_continue_2moons_distill" if loss_type == "distill"
        else f"{method}_8g_to_1moon_continue_2moons_prior" if loss_type == "prior"
        else f"{method}_8g_to_1moon_continue_2moons_efm" if loss_type == "efm"
        else f"{method}_8g_to_1moon_continue_2moons"
    )
    log_dir = save_dir / model_name / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs: {log_dir}")

    config_path = args.model_config
    if not Path(config_path).exists():
        create_default_config(config_path)
    model_config = load_model_config(config_path, "8g_to_2moons")

    model = create_model("8g_to_2moons", model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")

    flow_matcher = create_flow_matcher(
        method,
        checkpoint_args.get("sigma", 0.1),
        ma_method=checkpoint_args.get("ma_method", "downsample_2x"),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 8g_to_1moon 训的是左边月牙；continue 只用右边月牙（the missing moon）
    print(f"Continue training: ONLY the right moon (右边月牙), iterations={args.iterations}, loss={loss_type}")

    model_old = None
    if loss_type == "distill":
        model_old = create_model("8g_to_2moons", model_config, device)
        model_old.load_state_dict(checkpoint["model_state_dict"])
        model_old.eval()
        for p in model_old.parameters():
            p.requires_grad = False
        print(f"Distill: L = L_B + {args.lambda_distill} * L_distill, model_old frozen")
    elif loss_type == "prior":
        model_old = create_model("8g_to_1moon", load_model_config(config_path, "8g_to_1moon"), device)
        model_old.load_state_dict(checkpoint["model_state_dict"])
        model_old.eval()
        for p in model_old.parameters():
            p.requires_grad = False
        print(f"Prior-preservation: L = L_B + {args.lambda_prior} * L_prior, x_pr from frozen 8g->1moon")
    elif loss_type == "efm":
        model_old = create_model("8g_to_1moon", load_model_config(config_path, "8g_to_1moon"), device)
        model_old.load_state_dict(checkpoint["model_state_dict"])
        model_old.eval()
        for p in model_old.parameters():
            p.requires_grad = False
        print("EFM composite: target u = w_A*u_freeze + w_B*u_B, w_i = exp(-||xt-t*x_i||^2/(2(1-t)^2))")

    start_time = time.time()
    pbar = tqdm(total=args.iterations, desc="Continue (right moon only)", unit="iter")

    while iteration < args.iterations:
        optimizer.zero_grad()
        x0 = sample_8gaussians(args.batch_size).to(device)
        x1 = sample_right_moon(args.batch_size).to(device)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
        xt_t = torch.cat([xt, t[:, None]], dim=-1)
        vt = model(xt_t)

        if loss_type == "mse":
            loss = torch.mean((vt - ut) ** 2)
        elif loss_type == "distill":
            with torch.no_grad():
                v_old = model_old(xt_t)
            l_b = torch.mean((vt - ut) ** 2)
            l_distill = torch.mean((vt - v_old) ** 2)
            loss = l_b + args.lambda_distill * l_distill
            writer.add_scalar("train/l_b", l_b.item(), iteration)
            writer.add_scalar("train/l_distill", l_distill.item(), iteration)
        elif loss_type == "prior":
            l_b = torch.mean((vt - ut) ** 2)
            # Prior-preservation: x_pr = ancestral sample from frozen model (8g -> 1moon)
            x0_pr = sample_8gaussians(args.batch_size).to(device)
            with torch.no_grad():
                def model_wrapper(t, x, **kwargs):
                    if t.dim() == 0:
                        t = t.expand(x.shape[0])
                    return model_old(torch.cat([x, t[:, None]], dim=-1))
                node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
                t_span = torch.linspace(0, 1, 51).to(device)
                traj = node.trajectory(x0_pr, t_span=t_span)
                x_pr = traj[-1]
            t_pr, xt_pr, ut_pr = flow_matcher.sample_location_and_conditional_flow(x0_pr, x_pr)
            xt_pr_t = torch.cat([xt_pr, t_pr[:, None]], dim=-1)
            vt_pr = model(xt_pr_t)
            l_prior = torch.mean((vt_pr - ut_pr) ** 2)
            loss = l_b + args.lambda_prior * l_prior
            writer.add_scalar("train/l_b", l_b.item(), iteration)
            writer.add_scalar("train/l_prior", l_prior.item(), iteration)
        elif loss_type == "efm":
            # EFM composite: u_target = w_A*u_A + w_B*u_B
            # u_A = v_freeze(xt,t) (direction toward frozen 1moon endpoint)
            # u_B = (x1 - xt)/(1-t) (direction toward current right moon)
            # x1* = xt + v_freeze(xt,t)*(1-t)  (frozen endpoint, one step to t=1)
            # w_i(xt) = exp(-||xt - t*x_i||^2 / (2*(1-t)^2)), softmax -> S_A/(S_A+S_B), S_B/(S_A+S_B)
            with torch.no_grad():
                v_freeze = model_old(xt_t)
                eps = 1e-6
                one_minus_t = (1.0 - t).clamp_min(eps)
                x1_star = xt + v_freeze * one_minus_t[:, None]  # [B, D] frozen endpoint
                X1_both = torch.stack([x1_star, x1], dim=1)  # [B, 2, D]
                t_2d = t[:, None] if t.dim() == 1 else t
                diff = xt[:, None, :] - t_2d[:, None, :] * X1_both  # [B, 2, D]
                scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t[:, None].pow(2))  # [B, 2]
                w_efm = torch.softmax(scores, dim=1)  # [B, 2]
                u_A = v_freeze
                u_B = (x1 - xt) / one_minus_t[:, None]
                u_target = w_efm[:, 0:1] * u_A + w_efm[:, 1:2] * u_B  # [B, D]
            loss = torch.mean((vt - u_target) ** 2)
            writer.add_scalar("train/w_A_mean", w_efm[:, 0].mean().item(), iteration)
            writer.add_scalar("train/w_B_mean", w_efm[:, 1].mean().item(), iteration)
        else:
            loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()
        iteration += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        writer.add_scalar("train/loss", loss.item(), iteration)

        if iteration % max(1, args.log_iter) == 0:
            elapsed = time.time() - start_time
            tqdm.write(f"Iteration {iteration}: loss {loss.item():.4f}, time {elapsed:.2f}s")

        if args.sample_vis_iter > 0 and iteration > 0 and (
            iteration % args.sample_vis_iter == 0 or iteration == args.iterations
        ):
            sample_vis_path = save_dir / model_name / f"inference_iter_{iteration}.png"
            _visualize_inference_distribution(
                model, device, str(sample_vis_path),
                num_samples=args.sample_vis_num, num_steps=args.vis_steps, seed=args.seed,
            )
            tqdm.write(f"Saved inference distribution to {sample_vis_path}")
            try:
                img = Image.open(sample_vis_path)
                arr = np.array(img)
                ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0 if len(arr.shape) == 3 else torch.from_numpy(arr).unsqueeze(0).float() / 255.0
                writer.add_image("visualization/inference", ten, iteration)
            except Exception as e:
                tqdm.write(f"Warning: Could not add inference image to TensorBoard: {e}")

        if (args.save_iter > 0 and iteration % args.save_iter == 0) or iteration == args.iterations:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
                "args": {
                    **checkpoint_args,
                    "dataset": "8g_to_2moons",
                    "continue_from": str(checkpoint_path),
                    "loss": loss_type,
                    "lambda_distill": getattr(args, "lambda_distill", 0.0) if loss_type == "distill" else None,
                    "lambda_prior": getattr(args, "lambda_prior", 0.0) if loss_type == "prior" else None,
                    "loss_efm": loss_type == "efm",
                },
                "model_config": model_config,
                "rng_states": get_rng_states(),
            }
            out_path = save_dir / model_name / f"checkpoint_iter_{iteration}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out_path)
            tqdm.write(f"Saved checkpoint to {out_path}")

    pbar.close()
    writer.close()
    print("Continue training completed!")
    print(f"Checkpoint: {save_dir / model_name / f'checkpoint_iter_{iteration}.pt'}")
    return save_dir / model_name / f"checkpoint_iter_{iteration}.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue learning: 8g->1moon -> 8g->2moons")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to 8g_to_1moon checkpoint (.pt)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Subdir name under save_dir (default: {method}_8g_to_1moon_continue_2moons)")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--method", type=str, default="cfm", choices=["cfm", "otcfm", "sbcfm", "ma_otcfm", "ma_tcfm", "ma3_tcfm"])
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "distill", "prior", "efm"],
                        help="mse: FM loss; distill: L_B+λ*L_distill; prior: L_B+λ*L_prior; efm: EFM composite direction (frozen+current)")
    parser.add_argument("--lambda_distill", type=float, default=1.0,
                        help="Weight for L_distill when loss=distill. L = L_B + λ*||v_θ - v_θ_old||^2")
    parser.add_argument("--lambda_prior", type=float, default=1.0,
                        help="Weight for L_prior when loss=prior. L = L_B + λ*||v_θ - u_pr||^2 on prior samples")
    parser.add_argument("--save_iter", type=int, default=0,
                        help="Save checkpoint every N iters (0=only at end)")
    parser.add_argument("--log_iter", type=int, default=25)
    parser.add_argument("--vis_steps", type=int, default=50)
    parser.add_argument("--vis_step_interval", type=int, default=5)
    parser.add_argument("--sample_vis_iter", type=int, default=1000,
                        help="Every N iterations, draw inference distribution (0=disable)")
    parser.add_argument("--sample_vis_num", type=int, default=2000,
                        help="Number of samples for inference distribution plot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_continue_2moons(args)
