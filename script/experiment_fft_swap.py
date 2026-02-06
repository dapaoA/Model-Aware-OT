"""
Experiment: Swap low/high frequency between two CIFAR-10 images via FFT.

- Load two images, FFT each (per channel).
- Low-frequency region: center circle with radius = min(H,W)//2.
- Synthesize:
  1) low_A + high_B
  2) low_B + high_A
- Inverse FFT and visualize.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


def get_low_freq_mask(H, W, radius_frac=1.0):
    """Binary mask: 1 inside center circle (low freq), 0 outside (high freq).
    radius = min(H,W)/2 * radius_frac, so radius_frac=1.0 => radius = min/2 (half of image).
    """
    cy, cx = H // 2, W // 2
    radius = (min(H, W) / 2.0) * radius_frac
    y = np.arange(H, dtype=np.float32) - cy
    x = np.arange(W, dtype=np.float32) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r = np.sqrt(yy ** 2 + xx ** 2)
    mask = (r <= radius).astype(np.float32)
    return mask


def fft_swap_and_reconstruct(img_a, img_b, low_radius_frac=0.5):
    """
    img_a, img_b: [C, H, W] numpy or tensor, float [0,1] or normalized.
    Returns: (recon_lowA_highB, recon_lowB_highA) as numpy [C,H,W].
    """
    if torch.is_tensor(img_a):
        img_a = img_a.cpu().numpy()
    if torch.is_tensor(img_b):
        img_b = img_b.cpu().numpy()
    C, H, W = img_a.shape
    mask_low = get_low_freq_mask(H, W, low_radius_frac)  # [H, W]
    mask_high = 1.0 - mask_low  # [H, W]

    recon_lowA_highB = np.zeros_like(img_a, dtype=np.complex64)
    recon_lowB_highA = np.zeros_like(img_a, dtype=np.complex64)

    for c in range(C):
        fa = np.fft.fft2(img_a[c])
        fb = np.fft.fft2(img_b[c])
        fa = np.fft.fftshift(fa)
        fb = np.fft.fftshift(fb)

        # low_A + high_B
        low_a = fa * mask_low
        high_b = fb * mask_high
        combined1 = low_a + high_b
        combined1 = np.fft.ifftshift(combined1)
        recon_lowA_highB[c] = np.fft.ifft2(combined1).real

        # low_B + high_A
        low_b = fb * mask_low
        high_a = fa * mask_high
        combined2 = low_b + high_a
        combined2 = np.fft.ifftshift(combined2)
        recon_lowB_highA[c] = np.fft.ifft2(combined2).real

    return recon_lowA_highB, recon_lowB_highA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--idx_a", type=int, default=0, help="First image index")
    parser.add_argument("--idx_b", type=int, default=100, help="Second image index")
    parser.add_argument("--radius_frac", type=float, default=1.0, help="Low-freq circle radius = (min(H,W)/2)*radius_frac; 1.0 = half image")
    parser.add_argument("--output", type=str, default="exp/experiment_results/fft_swap.png")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)

    img_a = ds[args.idx_a][0]   # [C, H, W] normalized
    img_b = ds[args.idx_b][0]

    # Denormalize for visualization (we'll show both original and synth)
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)

    def to_display(x):
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        if x.dtype == np.complex64:
            x = x.real
        if x.max() <= 1.0 and x.min() >= -0.5:
            return np.clip(x, 0, 1)
        return np.clip((x * std + mean), 0, 1)

    img_a_np = img_a.cpu().numpy()
    img_b_np = img_b.cpu().numpy()

    recon_AB, recon_BA = fft_swap_and_reconstruct(img_a_np, img_b_np, low_radius_frac=args.radius_frac)
    recon_AB = np.real(recon_AB).astype(np.float32)
    recon_BA = np.real(recon_BA).astype(np.float32)

    # For display: denormalize originals; reconstructions are in normalized space so denorm too
    disp_a = to_display(img_a_np * std + mean)
    disp_b = to_display(img_b_np * std + mean)
    disp_AB = np.clip(recon_AB * std + mean, 0, 1).astype(np.float32)
    disp_BA = np.clip(recon_BA * std + mean, 0, 1).astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(disp_a.transpose(1, 2, 0))
    axes[0, 0].set_title("Image A (original)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(disp_b.transpose(1, 2, 0))
    axes[0, 1].set_title("Image B (original)")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(disp_AB.transpose(1, 2, 0))
    axes[1, 0].set_title("Low A + High B")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(disp_BA.transpose(1, 2, 0))
    axes[1, 1].set_title("Low B + High A")
    axes[1, 1].axis("off")
    r_px = (32 / 2.0) * args.radius_frac
    fig.suptitle(f"FFT swap (low-freq circle radius = {r_px:.0f} px)")
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
