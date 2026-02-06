"""
Generate (noise, generated_image) pairs from a trained CIFAR-10 CFM-style model.

This script:
- loads a checkpoint (e.g. checkpoint_iter_400000.pt)
- samples x0 ~ N(0, I)
- generates x1 by integrating dx/dt = v_theta(x,t) from t=0 -> 1 with Euler steps
- saves a single .pt file containing:
    - x0: (N, 3, 32, 32) noise (float16/float32 on CPU)
    - x1: (N, 3, 32, 32) generated images in *model space* (CIFAR-10 normalized) (float16/float32 on CPU)
    - meta: dict with checkpoint path, steps, dtype, etc.

Example:
  python generate_noise_image_pairs.py ^
    --checkpoint models/cifar10_otcfm/otcfm_cifar10/checkpoint_iter_400000.pt ^
    --num_pairs 10000 --batch_size 128 --num_steps 50 ^
    --out pairs/cifar10_ckpt400000_pairs.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchvision.utils as vutils
from tqdm import tqdm

from model import create_model
from utils import set_seed


def _infer_cifar10_shape_from_checkpoint(train_args: Dict) -> Tuple[int, int, int]:
    # Default CIFAR-10
    dataset_name = train_args.get("dataset", "cifar10")
    if dataset_name != "cifar10":
        raise ValueError(f"This script currently supports only cifar10 checkpoints, got dataset={dataset_name!r}")
    return (3, 32, 32)


@torch.no_grad()
def euler_generate(model: torch.nn.Module, x0: torch.Tensor, num_steps: int) -> torch.Tensor:
    """
    Euler integrator for dx/dt = v_theta(x,t), t in [0,1].

    Matches the intent of infer.py (NeuralODE solver="euler") but avoids storing trajectories.
    """
    model.eval()
    x = x0
    dt = 1.0 / float(num_steps)
    for i in range(num_steps):
        # Keep t in float32 to match training/inference conventions.
        t = torch.full((x.shape[0],), float(i) / float(num_steps), device=x.device, dtype=torch.float32)
        v = model(x, t)
        x = x + dt * v
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save (noise, generated_image) pairs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_iter_*.pt")
    parser.add_argument("--num_pairs", type=int, default=10_000, help="Number of pairs to generate")
    parser.add_argument("--batch_size", type=int, default=128, help="Generation batch size")
    parser.add_argument("--num_steps", type=int, default=50, help="Euler steps from t=0 to t=1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Storage dtype for x0/x1 in the saved .pt (generation runs in float32 unless --amp)",
    )
    parser.add_argument("--amp", action="store_true", help="Use autocast (fp16/bf16 depending on CUDA) for generation")
    parser.add_argument("--out", type=str, required=True, help="Output .pt path for pairs")
    parser.add_argument("--preview_png", type=str, default=None, help="Optional path to save a preview grid PNG")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    train_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    dataset_name = train_args.get("dataset", "cifar10")
    if dataset_name != "cifar10":
        raise ValueError(f"Expected cifar10 checkpoint, got dataset={dataset_name!r}")

    c, h, w = _infer_cifar10_shape_from_checkpoint(train_args)

    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    store_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    x0_all = torch.empty((args.num_pairs, c, h, w), dtype=store_dtype, device="cpu")
    x1_all = torch.empty((args.num_pairs, c, h, w), dtype=store_dtype, device="cpu")

    # Generate in batches
    pbar = tqdm(total=args.num_pairs, desc="Generating pairs", unit="img")
    start = 0
    while start < args.num_pairs:
        b = min(args.batch_size, args.num_pairs - start)
        x0 = torch.randn((b, c, h, w), device=device, dtype=torch.float32)

        if args.amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x1 = euler_generate(model, x0, args.num_steps)
        else:
            x1 = euler_generate(model, x0, args.num_steps)

        x0_all[start : start + b].copy_(x0.detach().cpu().to(store_dtype))
        x1_all[start : start + b].copy_(x1.detach().cpu().to(store_dtype))

        start += b
        pbar.update(b)
    pbar.close()

    meta = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset_name,
        "num_pairs": int(args.num_pairs),
        "batch_size": int(args.batch_size),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "storage_dtype": args.dtype,
        "note": "x1 is in model space (CIFAR-10 normalized by mean/std), not denormalized to [0,1].",
    }

    payload = {"x0": x0_all, "x1": x1_all, "meta": meta}
    torch.save(payload, out_path)
    print(f"Saved pairs to: {out_path}")
    print("Meta:")
    print(json.dumps(meta, indent=2))

    # Optional preview grid (denormalize to [0,1] for viewing)
    if args.preview_png:
        preview_path = Path(args.preview_png)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        # CIFAR-10 denorm consistent with infer.py
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        imgs = x1_all[:100].float()
        imgs = (imgs * std + mean).clamp(0, 1)
        vutils.save_image(imgs, preview_path, nrow=10, padding=2)
        print(f"Saved preview grid to: {preview_path}")


if __name__ == "__main__":
    main()

