"""
Continue learning: load a model trained on moons_to_7gaussians and continue training
with target = 8 gaussians (or other target/loss). For comparing post-training methods
with different losses later.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from dataset import sample_source_distribution
from flow_matcher import create_flow_matcher
from model import create_model, load_model_config, create_default_config
from utils import set_seed
from utils.common import get_rng_states
from utils.visualization import visualize_denoising_process
from torchcfm.utils import sample_moons, sample_8gaussians


def train_continue(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    # Source dataset (what the model was trained on)
    source_dataset = checkpoint_args.get("dataset", "moons_to_7gaussians")
    method = checkpoint_args.get("method", args.method)
    assert source_dataset == "moons_to_7gaussians", (
        f"Continue learning expects a moons_to_7gaussians checkpoint, got dataset={source_dataset}"
    )

    set_seed(args.seed)
    iteration = 0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.output_name or f"{method}_moons_to_7gaussians_continue_8g"
    log_dir = save_dir / model_name / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs: {log_dir}")

    config_path = args.model_config
    if not Path(config_path).exists():
        create_default_config(config_path)
    # Use same 2D config (moons_to_8gaussians has same arch as moons_to_7gaussians)
    model_config = load_model_config(config_path, "moons_to_8gaussians")

    model = create_model("moons_to_8gaussians", model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {checkpoint_path}")

    flow_matcher = create_flow_matcher(
        method,
        checkpoint_args.get("sigma", 0.1),
        ma_method=checkpoint_args.get("ma_method", "downsample_2x"),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_type = (args.loss or "mse").lower()
    print(f"Continue training: target=8 gaussians, iterations={args.iterations}, loss={loss_type}")

    start_time = time.time()
    pbar = tqdm(total=args.iterations, desc="Continue training", unit="iter")

    while iteration < args.iterations:
        optimizer.zero_grad()
        x0 = sample_moons(args.batch_size).to(device)
        x1 = sample_8gaussians(args.batch_size).to(device)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = model(torch.cat([xt, t[:, None]], dim=-1))

        if loss_type == "mse":
            loss = torch.mean((vt - ut) ** 2)
        else:
            # Placeholder for other losses (e.g. KL, contrastive) - extend here
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

        if iteration % max(1, args.save_iter) == 0 or iteration == args.iterations:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iteration": iteration,
                "args": {
                    **checkpoint_args,
                    "dataset": "moons_to_8gaussians",
                    "continue_from": str(checkpoint_path),
                },
                "model_config": model_config,
                "rng_states": get_rng_states(),
            }
            out_path = save_dir / model_name / f"checkpoint_iter_{iteration}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out_path)
            tqdm.write(f"Saved checkpoint to {out_path}")

            vis_path = save_dir / model_name / f"denoising_iter_{iteration}.png"
            visualize_denoising_process(
                model, "moons_to_8gaussians", device, str(vis_path),
                num_steps=args.vis_steps, step_interval=args.vis_step_interval,
            )
            try:
                img = Image.open(vis_path)
                import numpy as np
                arr = np.array(img)
                if len(arr.shape) == 3:
                    ten = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                else:
                    ten = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
                writer.add_image("visualization/denoising", ten, iteration)
            except Exception as e:
                tqdm.write(f"Warning: Could not add image to TensorBoard: {e}")

    pbar.close()
    writer.close()
    print("Continue training completed!")
    print(f"Checkpoint: {save_dir / model_name / f'checkpoint_iter_{iteration}.pt'}")
    return save_dir / model_name / f"checkpoint_iter_{iteration}.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue learning: 7g -> 8g")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to moons_to_7gaussians checkpoint (.pt)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Subdir name under save_dir (default: {method}_moons_to_7gaussians_continue_8g)")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml")
    parser.add_argument("--method", type=str, default="cfm", choices=["cfm", "otcfm", "sbcfm", "ma_otcfm", "ma_tcfm", "ma3_tcfm"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mse", choices=["mse"],
                        help="Loss type (mse default; extend for future post-training losses)")
    parser.add_argument("--save_iter", type=int, default=50)
    parser.add_argument("--log_iter", type=int, default=25)
    parser.add_argument("--vis_steps", type=int, default=50)
    parser.add_argument("--vis_step_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_continue(args)
