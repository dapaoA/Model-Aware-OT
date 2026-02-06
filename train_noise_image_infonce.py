"""
Train a simple dual-tower (noise encoder, image encoder) with InfoNCE on saved (noise, generated_image) pairs.

You should first generate a pairs file with:
  python generate_noise_image_pairs.py --checkpoint <...checkpoint_iter_400000.pt> --num_pairs 10000 --out pairs.pt

Then train:
  python train_noise_image_infonce.py --pairs pairs.pt --log_dir runs/noise_image_infonce --epochs 50

Notes:
- The saved x1 is in *model space* for CIFAR-10: normalized by mean/std (same as dataset/utils.py).
- We feed x1 directly to the image encoder; for TensorBoard visualization we denormalize to [0,1].
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import set_seed


class NoiseImagePairsDataset(Dataset):
    def __init__(self, pairs_path: str, augment_hflip: bool = True):
        payload = torch.load(pairs_path, map_location="cpu")
        self.x0 = payload["x0"]  # (N,3,32,32)
        self.x1 = payload["x1"]  # (N,3,32,32) normalized (model space)
        self.meta: Dict = payload.get("meta", {})

        if self.x0.shape != self.x1.shape:
            raise ValueError(f"x0 shape {tuple(self.x0.shape)} != x1 shape {tuple(self.x1.shape)}")
        if self.x0.dim() != 4 or self.x0.shape[1:] != (3, 32, 32):
            raise ValueError(f"Expected (N,3,32,32), got {tuple(self.x0.shape)}")

        self.augment_hflip = augment_hflip

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.x0[idx].float()
        x1 = self.x1[idx].float()

        if self.augment_hflip and torch.rand(()) < 0.5:
            x1 = torch.flip(x1, dims=[2])  # horizontal flip: width dim for CHW is dim=2? Actually CHW: (C,H,W) => W=2
        return x0, x1


class SmallImageEncoder(nn.Module):
    """A tiny CLIP-style image tower for 32x32 images."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 16x16
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 8x8
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        z = self.proj(h)
        z = self.ln(z)
        return z


class SmallNoiseEncoder(nn.Module):
    """A tiny MLP tower for noise tensors shaped like images."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = self.ln(z)
        return z


class DualTowerInfoNCE(nn.Module):
    def __init__(self, embed_dim: int, init_temperature: float = 0.07):
        super().__init__()
        self.image_encoder = SmallImageEncoder(embed_dim)
        self.noise_encoder = SmallNoiseEncoder(embed_dim)
        # CLIP-style learnable logit scale
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z0 = F.normalize(self.noise_encoder(x0), dim=-1)
        z1 = F.normalize(self.image_encoder(x1), dim=-1)

        # clamp like CLIP to avoid exploding scale
        logit_scale = self.logit_scale.clamp(min=math.log(1.0), max=math.log(100.0)).exp()
        logits = logit_scale * (z0 @ z1.t())
        return logits, z0, z1


def clip_accuracy(logits: torch.Tensor) -> torch.Tensor:
    # retrieval top-1 (noise -> image)
    labels = torch.arange(logits.shape[0], device=logits.device)
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean()


def denorm_cifar10(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,32,32) normalized -> [0,1] for visualization."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual-tower InfoNCE on (noise, generated_image) pairs")
    parser.add_argument("--pairs", type=str, required=True, help="Path to .pt produced by generate_noise_image_pairs.py")
    parser.add_argument("--log_dir", type=str, default="runs/noise_image_infonce", help="TensorBoard log dir")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs over the pairs file")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--temperature", type=float, default=0.07, help="Initial temperature for InfoNCE")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--no_hflip", action="store_true", help="Disable random horizontal flip augmentation on images")
    parser.add_argument("--amp", action="store_true", help="Use AMP for training (CUDA only)")
    parser.add_argument("--save_ckpt", type=str, default=None, help="Optional path to save trained encoder checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = NoiseImagePairsDataset(args.pairs, augment_hflip=(not args.no_hflip))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = DualTowerInfoNCE(embed_dim=args.embed_dim, init_temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("meta/pairs_meta", str(dataset.meta))
    writer.add_text("meta/pairs_path", str(Path(args.pairs).resolve()))

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        for x0, x1 in pbar:
            x0 = x0.to(device, non_blocking=True)
            x1 = x1.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if args.amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, _, _ = model(x0, x1)
                    labels = torch.arange(logits.shape[0], device=device)
                    loss_i = F.cross_entropy(logits, labels)
                    loss_t = F.cross_entropy(logits.t(), labels)
                    loss = 0.5 * (loss_i + loss_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _, _ = model(x0, x1)
                labels = torch.arange(logits.shape[0], device=device)
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.t(), labels)
                loss = 0.5 * (loss_i + loss_t)
                loss.backward()
                optimizer.step()

            acc = clip_accuracy(logits).item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/acc_top1_noise_to_img", acc, global_step)
            writer.add_scalar("train/logit_scale", model.logit_scale.detach().exp().item(), global_step)
            global_step += 1

            # lightweight visualization every ~200 steps
            if global_step % 200 == 0:
                with torch.no_grad():
                    imgs = denorm_cifar10(x1[:64])
                    grid = vutils.make_grid(imgs, nrow=8, padding=2)
                    writer.add_image("train/generated_images", grid, global_step)

        pbar.close()

    writer.close()

    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "embed_dim": args.embed_dim,
                "pairs_meta": dataset.meta,
                "pairs_path": str(Path(args.pairs).resolve()),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved encoder checkpoint to: {ckpt_path}")

    print(f"Done. TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()

