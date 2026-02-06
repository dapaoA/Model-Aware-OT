"""
Debug script: Show step-by-step how EFM DCT 4x4 computes scores, weights, and u_efm.

Run: python debug_efm_dct_4x4.py

Compares:
1. Original EFM (Euclidean): scores = -||diff||^2 / (2(1-t)^2)
2. EFM DCT 4x4:            scores = -||DCT_4x4_low(diff)||^2 / (2(1-t)^2)
"""

import torch
import numpy as np
from scipy.fft import dctn
from torchvision import datasets, transforms

from utils.efm import (
    efm_closed_form_weights_and_u,
    efm_closed_form_weights_and_u_dct_4x4,
    _dct_4x4_low_flat,
)
from utils import set_seed


def _get_zigzag_indices_cached(h: int, w: int):
    cache_key = (h, w)
    if not hasattr(_get_zigzag_indices_cached, "_cache"):
        _get_zigzag_indices_cached._cache = {}
    if cache_key not in _get_zigzag_indices_cached._cache:
        total = h * w
        idx_list = []
        i, j, direction = 0, 0, 1
        while len(idx_list) < total and i < h and j < w:
            idx_list.append((i, j))
            if direction == 1:
                if i == 0 or j == w - 1:
                    if j == w - 1:
                        i += 1
                    else:
                        j += 1
                    direction = -1
                else:
                    i -= 1
                    j += 1
            else:
                if j == 0 or i == h - 1:
                    if i == h - 1:
                        j += 1
                    else:
                        i += 1
                    direction = 1
                else:
                    i += 1
                    j -= 1
        _get_zigzag_indices_cached._cache[cache_key] = idx_list[:total]
    return _get_zigzag_indices_cached._cache[cache_key]


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, M = 4, 4
    t_val = 0.5

    # Tiny example: 4 noise, 4 images
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    x1_imgs = torch.stack([ds[i][0] for i in range(M)], dim=0).to(device)
    x0 = torch.randn(B, 3, 32, 32, device=device)
    x1 = x1_imgs[:B]

    # xt = t*x1 + (1-t)*x0 (use first B for simplicity)
    xt = t_val * x1 + (1 - t_val) * x0
    t = torch.full((B,), t_val, device=device)

    # Flatten for Euclidean EFM
    xt_flat = xt.reshape(B, -1)
    x1_flat = x1.reshape(M, -1)

    print("=" * 60)
    print("EFM DCT 4x4 step-by-step")
    print("=" * 60)
    print(f"xt shape: {xt.shape}  x1 shape: {x1.shape}")
    print(f"t = {t_val}")
    print()

    # Step 1: diff = xt - t * x1_j
    diff = xt[:, None, :, :, :] - t[:, None, None, None, None] * x1[None, :, :, :, :]
    print(f"1. diff = xt - t*x1_j  =>  shape {tuple(diff.shape)}")
    print(f"   diff range: [{diff.min().item():.4f}, {diff.max().item():.4f}]")
    print()

    # Step 2: Euclidean distance (for comparison)
    diff_flat_euc = xt_flat[:, None, :] - t[:, None, None] * x1_flat[None, :, :]
    dist_sq_euc = (diff_flat_euc ** 2).sum(dim=-1)
    one_minus_t = (1 - t_val)
    scores_euc = -dist_sq_euc / (2 * one_minus_t ** 2)
    print(f"2a. Euclidean: ||diff||^2 per (i,j)")
    print(f"    dist_sq_euc shape: {tuple(dist_sq_euc.shape)}")
    print(f"    dist_sq_euc sample (first row): {dist_sq_euc[0].cpu().numpy()}")
    print(f"    scores_euc = -dist_sq / (2(1-t)^2), (1-t)^2 = {one_minus_t**2:.4f}")
    print(f"    scores_euc sample: {scores_euc[0].cpu().numpy()}")
    print()

    # Step 2b: DCT 4x4 low distance (manual)
    BM, C, H, W = diff.shape[0] * diff.shape[1], 3, 32, 32
    diff_flat = diff.reshape(BM, C, H, W).clone()
    indices = _get_zigzag_indices_cached(H, W)[:16]
    dct_low_list = []
    for n in range(BM):
        row = []
        for c in range(C):
            dct_2d = dctn(diff_flat[n, c].cpu().numpy(), norm="ortho")
            dct_low = np.array([dct_2d[i, j] for (i, j) in indices])
            row.append(dct_low)
        dct_low_list.append(np.concatenate(row))
    dct_low_manual = torch.from_numpy(np.stack(dct_low_list)).to(device).float()
    dct_low_manual = dct_low_manual.reshape(B, M, -1)

    # Compare with utils _dct_4x4_low_flat
    dct_low_util = _dct_4x4_low_flat(diff_flat)
    dct_low_util = dct_low_util.reshape(B, M, -1)
    dct_diff = (dct_low_manual - dct_low_util).abs().max().item()

    dct_low = dct_low_manual
    dist_sq_dct = (dct_low ** 2).sum(dim=-1)
    scores_dct = -dist_sq_dct / (2 * one_minus_t ** 2)
    print(f"2b. DCT 4x4 low: ||DCT_low(diff)||^2 per (i,j)")
    print(f"    DCT extracts first 16 zigzag coeffs per channel -> 48 dims")
    print(f"    dist_sq_dct shape: {tuple(dist_sq_dct.shape)}")
    print(f"    dist_sq_dct sample (first row): {dist_sq_dct[0].cpu().numpy()}")
    print(f"    dist_sq_dct range: [{dist_sq_dct.min().item():.2f}, {dist_sq_dct.max().item():.2f}]")
    print(f"    DCT manual vs utils max diff: {dct_diff:.6f}")
    print(f"    scores_dct sample: {scores_dct[0].cpu().numpy()}")
    print()

    # Step 3: Softmax weights
    w_euc = torch.softmax(scores_euc, dim=1)
    w_dct = torch.softmax(scores_dct, dim=1)
    print(f"3. Weights w = softmax(scores)")
    print(f"   w_euc (Euclidean) sample: {w_euc[0].cpu().numpy()}")
    print(f"   w_dct (DCT 4x4)   sample: {w_dct[0].cpu().numpy()}")
    print()

    # Step 4: u_efm
    v = (x1[None, :, :, :, :] - xt[:, None, :, :, :]) / one_minus_t
    u_euc = (w_euc[:, :, None, None, None] * v).sum(dim=1)
    u_dct = (w_dct[:, :, None, None, None] * v).sum(dim=1)
    print(f"4. u_efm = sum_j w_j * (x1_j - xt) / (1-t)")
    print(f"   u_euc norm: {u_euc.reshape(B,-1).norm(dim=1).cpu().numpy()}")
    print(f"   u_dct norm: {u_dct.reshape(B,-1).norm(dim=1).cpu().numpy()}")
    print()

    # Compare with utils - use same diff to trace
    _, u_efm_euc, _ = efm_closed_form_weights_and_u(xt_flat, x1_flat, t)
    # Trace utils: compute dist_sq inside utils with same inputs
    from utils.efm import efm_closed_form_weights_and_u_dct_4x4
    B2, C2, H2, W2 = xt.shape
    M2 = x1.shape[0]
    diff_util = xt[:, None, :, :, :] - t[:, None, None, None, None] * x1[None, :, :, :, :]
    BM2 = diff_util.numel() // (C2 * H2 * W2)
    dct_low_util_full = _dct_4x4_low_flat(diff_util.reshape(BM2, C2, H2, W2))
    dct_low_util_full = dct_low_util_full.reshape(B2, M2, -1)
    dist_sq_util = (dct_low_util_full ** 2).sum(dim=-1)
    print(f"   Utils internal dist_sq[0]: {dist_sq_util[0].cpu().numpy()}")
    print(f"   Manual dist_sq[0]:         {dist_sq_dct[0].cpu().numpy()}")
    print(f"   dist_sq match: {(dist_sq_util - dist_sq_dct).abs().max().item():.6f}")

    w_util, u_efm_dct, scores_util = efm_closed_form_weights_and_u_dct_4x4(xt, x1, t)
    print("5. Utils efm_closed_form_weights_and_u_dct_4x4 output:")
    print(f"   w shape: {w_util.shape}, scores shape: {scores_util.shape}, u_efm shape: {u_efm_dct.shape}")
    print(f"   w[0]: {w_util[0].cpu().numpy()}")
    print(f"   scores[0]: {scores_util[0].cpu().numpy()}")
    if u_efm_dct.dim() == 5:
        print(f"   WARNING: u_efm_dct has 5 dims {tuple(u_efm_dct.shape)}, summing dim=1")
        u_efm_dct = u_efm_dct.sum(dim=1)
    print()
    print("6. Verify against utils.efm:")
    print(f"   u_efm (Euclidean) from utils - norm: {u_efm_euc.reshape(B,-1).norm(dim=1).cpu().numpy()}")
    print(f"   u_efm (DCT 4x4)   from utils - norm: {u_efm_dct.reshape(B,-1).norm(dim=1).cpu().numpy()}")
    print(f"   Manually u_euc vs utils u_efm_euc diff: {(u_euc - u_efm_euc.reshape(B,3,32,32)).abs().max().item():.6f}")
    print(f"   Manually u_dct vs utils u_efm_dct diff: {(u_dct - u_efm_dct).abs().max().item():.6f}")
    print()

    # Key comparison: scale of scores
    print("7. Scale comparison (why DCT might fail):")
    print(f"   Euclidean ||diff||^2: mean {dist_sq_euc.mean().item():.1f}, std {dist_sq_euc.std().item():.1f}")
    print(f"   DCT      ||dct_low||^2: mean {dist_sq_dct.mean().item():.1f}, std {dist_sq_dct.std().item():.1f}")
    print(f"   Ratio (Euc/DCT): {dist_sq_euc.mean().item() / (dist_sq_dct.mean().item() + 1e-8):.1f}x")
    print()
    print("   -> DCT uses only 48 dims vs 3072 for Euclidean. Scores are less discriminative.")
    print("   -> Softmax over M with small score differences -> more uniform weights.")
    print("   -> u_efm becomes average of many directions -> may not match model's OT pairing.")


if __name__ == "__main__":
    main()
