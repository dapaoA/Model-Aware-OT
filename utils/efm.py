"""
Closed-form EFM (Exponential Flow Matching) utilities.

Implements the closed-form vector field / weights:

Given:
  x:   [B, D]
  X1:  [M, D]
  t:   [B] or [B, 1] with t in (0, 1)

Compute:
  scores_j = -||x - t * x1_j||^2 / (2 (1 - t)^2)
  w = softmax(scores, dim=1)
  v_j = (x1_j - x) / (1 - t)
  u_efm = sum_j w_j * v_j
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dctn


def _get_zigzag_indices_cached(h: int, w: int):
    """Zigzag scan indices (cached)."""
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


def _dct_zigzag_flat(
    x: torch.Tensor,
    num_coeffs: int,
    mode: str = "low",
) -> torch.Tensor:
    """Extract DCT zigzag coefficients per channel, flatten. x: [..., C, H, W].
    mode: 'low' = first num_coeffs, 'high' = last num_coeffs, 'mid' = middle num_coeffs.
    """
    x_np = x.cpu().numpy()
    orig_shape = x_np.shape
    if x_np.ndim == 4:
        x_np = x_np.reshape(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
    N, C, H, W = x_np.shape
    total = H * W
    indices_full = _get_zigzag_indices_cached(H, W)
    if mode == "low":
        indices = indices_full[:num_coeffs]
    elif mode == "high":
        indices = indices_full[-num_coeffs:]
    elif mode == "mid":
        skip = total // 4
        indices = indices_full[skip : skip + num_coeffs]
    else:
        raise ValueError(f"mode must be 'low','high','mid', got {mode}")
    out_list = []
    for n in range(N):
        row = []
        for c in range(C):
            dct_2d = dctn(x_np[n, c], norm="ortho")
            coeffs = np.array([dct_2d[i, j] for (i, j) in indices])
            row.append(coeffs)
        out_list.append(np.concatenate(row))
    out = np.stack(out_list, axis=0)
    return torch.from_numpy(out).to(x.device).float()


def _dct_4x4_low_flat(x: torch.Tensor) -> torch.Tensor:
    """Extract DCT 4x4 low-freq (first 16 zigzag) per channel, flatten. x: [..., C, H, W]."""
    return _dct_zigzag_flat(x, 16, mode="low")


def efm_closed_form_weights_and_u(
    x: torch.Tensor,
    X1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        x: [B, D]
        X1: [M, D]
        t: [B] or [B, 1]
        eps: clamp floor for (1 - t)

    Returns:
        w: [B, M] softmax weights
        u_efm: [B, D] closed-form vector field
        scores: [B, M] pre-softmax scores
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [B,D], got shape {tuple(x.shape)}")
    if X1.dim() != 2:
        raise ValueError(f"X1 must be [M,D], got shape {tuple(X1.shape)}")
    if x.shape[1] != X1.shape[1]:
        raise ValueError(f"Feature dim mismatch: x has D={x.shape[1]}, X1 has D={X1.shape[1]}")

    if t.dim() == 1:
        t = t[:, None]
    if t.dim() != 2 or t.shape[0] != x.shape[0] or t.shape[1] != 1:
        raise ValueError(f"t must be [B] or [B,1], got shape {tuple(t.shape)} for B={x.shape[0]}")

    B, D = x.shape
    M = X1.shape[0]
    one_minus_t = (1.0 - t).clamp_min(eps)  # [B,1]

    # Memory-efficient: process each x[i] separately when M is large
    if B * M * D > 1e9:  # ~4GB threshold
        w_list, u_list, scores_list = [], [], []
        for i in range(B):
            xi = x[i : i + 1]  # [1, D]
            ti = t[i : i + 1]  # [1, 1]
            one_minus_ti = one_minus_t[i : i + 1]  # [1, 1]
            diff_i = xi - ti * X1  # [M, D]
            scores_i = -(diff_i.pow(2).sum(-1)) / (2.0 * one_minus_ti.pow(2))  # [M]
            w_i = torch.softmax(scores_i, dim=0)  # [M]
            v_i = (X1 - xi) / one_minus_ti  # [M, D]
            u_i = w_i @ v_i  # [M] @ [M, D] = [D]
            w_list.append(w_i)
            u_list.append(u_i)
            scores_list.append(scores_i)
        w = torch.stack(w_list, dim=0)  # [B, M]
        u_efm = torch.stack(u_list, dim=0)  # [B, D]
        scores = torch.stack(scores_list, dim=0)  # [B, M]
    else:
        # diff = x - t * X1_j  => [B, M, D]
        diff = x[:, None, :] - t[:, None, :] * X1[None, :, :]
        scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t.pow(2))
        w = torch.softmax(scores, dim=1)
        v = (X1[None, :, :] - x[:, None, :]) / one_minus_t[:, None, :]
        u_efm = (w[:, :, None] * v).sum(dim=1)

    return w, u_efm, scores


def lsefm_block_weights_and_u(
    x: torch.Tensor,
    X1: torch.Tensor,
    t: torch.Tensor,
    n_blocks: int = 2,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block LSEFM: split image into n_blocks x n_blocks blocks; within each block
    compute EFM weights from the block patch, then velocity for each pixel in the block
    using those weights. Fully vectorized (no Python for-loops).

    Args:
        x: [B, C, H, W]
        X1: [M, C, H, W]
        t: [B] or [B, 1]
        n_blocks: grid size (e.g. 2 -> 2x2=4 blocks, 4 -> 4x4=16 blocks). H and W must be divisible by n_blocks.
        eps: clamp floor for (1 - t)

    Returns:
        w_blocks: [B, n_blocks*n_blocks, M] - softmax weights per block
        u: [B, C, H, W] - vector field
    """
    if x.dim() != 4 or X1.dim() != 4:
        raise ValueError(f"x and X1 must be [B,C,H,W] and [M,C,H,W], got {x.shape}, {X1.shape}")
    if x.shape[1:] != X1.shape[1:]:
        raise ValueError(f"x and X1 must have same (C,H,W)")

    if t.dim() == 1:
        t = t[:, None]
    B, C, H, W = x.shape
    M = X1.shape[0]
    if H % n_blocks != 0 or W % n_blocks != 0:
        raise ValueError(f"H and W must be divisible by n_blocks={n_blocks}, got H={H} W={W}")

    one_minus_t = (1.0 - t).clamp_min(eps)  # [B, 1]
    h_b, w_b = H // n_blocks, W // n_blocks
    n_b2 = n_blocks * n_blocks
    D_block = C * h_b * w_b

    # Reshape to [B, n_b2, D_block] and [M, n_b2, D_block]: each block flattened
    # x: [B, C, H, W] -> [B, C, n_blocks, h_b, n_blocks, w_b] -> [B, n_b2, D_block]
    x_blocks = (
        x.reshape(B, C, n_blocks, h_b, n_blocks, w_b)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B, n_b2, D_block)
    )
    X1_blocks = (
        X1.reshape(M, C, n_blocks, h_b, n_blocks, w_b)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(M, n_b2, D_block)
    )

    # Scores: diff[b, k, m, :] = x_blocks[b, k, :] - t[b]*X1_blocks[m, k, :]
    # x_blocks [B, n_b2, D] -> [B, n_b2, 1, D]; X1_blocks [M, n_b2, D] -> [1, n_b2, M, D] so block index k aligns
    t_b = t.squeeze(1)  # [B]
    X1_blocks_k = X1_blocks.permute(1, 0, 2)  # [n_b2, M, D_block]
    diff = x_blocks[:, :, None, :] - t_b[:, None, None, None] * X1_blocks_k[None, :, :, :]  # [B, n_b2, M, D_block]
    denom = (2.0 * one_minus_t.pow(2)).view(B, 1, 1)  # [B, 1, 1] for broadcast over (n_b2, M)
    scores = -(diff.pow(2).sum(dim=-1)) / denom  # [B, n_b2, M]
    w_blocks = torch.softmax(scores, dim=-1)  # [B, n_b2, M]

    # Expand block weights to per-pixel weights by repeating each block over its spatial region.
    # w_blocks: [B, n_b2, M] -> [B, n_blocks, n_blocks, M] -> repeat to [B, H, W, M]
    w_grid = w_blocks.view(B, n_blocks, n_blocks, M)  # [B, nb, nb, M]
    w_map = w_grid.repeat_interleave(h_b, dim=1).repeat_interleave(w_b, dim=2)  # [B, H, W, M]

    # v[b, m, c, i, j] = (X1[m, c, i, j] - x[b, c, i, j]) / (1-t[b])
    one_minus_t_b = one_minus_t.squeeze(1)[:, None, None, None, None]  # [B, 1, 1, 1, 1]
    v = (X1[None, :, :, :, :] - x[:, None, :, :, :]) / one_minus_t_b  # [B, M, C, H, W]

    # u[b, c, i, j] = sum_m w_map[b, i, j, m] * v[b, m, c, i, j]
    # Use einsum to avoid broadcasting shape pitfalls.
    u = torch.einsum("bhwm,bmchw->bchw", w_map, v)  # [B, C, H, W]

    return w_blocks, u


def lsefm_closed_form_u(
    x: torch.Tensor,
    X1: torch.Tensor,
    t: torch.Tensor,
    kernel_size: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Local Spatially-aware EFM (LSEFM) - optimized version using unfold.
    
    For each pixel (i,j), compute EFM weights using local kernel_size x kernel_size patches
    from all M images, then compute velocity for the center pixel.
    
    Args:
        x: [B, C, H, W]
        X1: [M, C, H, W]
        t: [B] or [B, 1]
        kernel_size: size of local neighborhood (default 3)
        eps: clamp floor for (1 - t)
    
    Returns:
        u_lsefm: [B, C, H, W] local EFM vector field
    """
    if x.dim() != 4:
        raise ValueError(f"x must be [B,C,H,W], got shape {tuple(x.shape)}")
    if X1.dim() != 4:
        raise ValueError(f"X1 must be [M,C,H,W], got shape {tuple(X1.shape)}")
    if x.shape[1:] != X1.shape[1:]:
        raise ValueError(f"x and X1 must have same (C,H,W), got {x.shape[1:]} vs {X1.shape[1:]}")
    
    if t.dim() == 1:
        t = t[:, None]
    
    B, C, H, W = x.shape
    M = X1.shape[0]
    one_minus_t = (1.0 - t).clamp_min(eps)  # [B, 1]
    
    # Pad for boundary handling
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')  # [B, C, H+2*pad, W+2*pad]
    X1_padded = F.pad(X1, (pad, pad, pad, pad), mode='reflect')  # [M, C, H+2*pad, W+2*pad]
    
    # Extract all patches using unfold
    # unfold: [B, C, H+2p, W+2p] -> [B, C*k*k, H*W]
    x_patches = F.unfold(x_padded, kernel_size=kernel_size, stride=1)  # [B, C*k*k, H*W]
    X1_patches = F.unfold(X1_padded, kernel_size=kernel_size, stride=1)  # [M, C*k*k, H*W]
    
    # Reshape: [B, C*k*k, H*W] -> [B, H*W, C*k*k]
    x_patches = x_patches.transpose(1, 2)  # [B, H*W, C*k*k]
    X1_patches = X1_patches.transpose(1, 2)  # [M, H*W, C*k*k]
    
    # Flatten x and X1 for center pixel velocities
    x_flat = x.reshape(B, C, -1).transpose(1, 2)  # [B, H*W, C]
    X1_flat = X1.reshape(M, C, -1).transpose(1, 2)  # [M, H*W, C]
    
    u_lsefm_flat = torch.zeros(B, H * W, C, device=x.device, dtype=x.dtype)
    
    # Process each spatial position
    for pos in range(H * W):
        x_patch_pos = x_patches[:, pos, :]  # [B, C*k*k]
        X1_patch_pos = X1_patches[:, pos, :]  # [M, C*k*k]
        
        x_center_pos = x_flat[:, pos, :]  # [B, C]
        X1_center_pos = X1_flat[:, pos, :]  # [M, C]
        
        # Compute for each sample in batch
        for b in range(B):
            x_b = x_patch_pos[b:b+1]  # [1, C*k*k]
            t_b = t[b:b+1]  # [1, 1]
            one_minus_t_b = one_minus_t[b:b+1]  # [1, 1]
            
            # Compute scores
            diff = x_b - t_b * X1_patch_pos  # [M, C*k*k]
            scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t_b.pow(2))  # [M]
            w = torch.softmax(scores, dim=0)  # [M]
            
            # Compute velocity for center pixel
            x_center_b = x_center_pos[b]  # [C]
            v = (X1_center_pos - x_center_b[None, :]) / one_minus_t_b  # [M, C]
            
            u_lsefm_flat[b, pos, :] = w @ v  # [M] @ [M, C] = [C]
    
    # Reshape back to image
    u_lsefm = u_lsefm_flat.transpose(1, 2).reshape(B, C, H, W)
    
    return u_lsefm


def efm_closed_form_weights_and_u_sigma(
    x: torch.Tensor,
    X1: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    s_EFM: EFM with noise-injected score. Score uses (x + sigma*epsilon) - t*x1 instead of x - t*x1.
    epsilon ~ N(0, I), same shape as x.

    Args:
        x: [B, D]
        X1: [M, D]
        t: [B] or [B, 1]
        sigma: noise strength
        generator: optional RNG for reproducibility
        eps: clamp floor for (1 - t)

    Returns:
        w: [B, M] softmax weights
        u_efm: [B, D] closed-form vector field
        scores: [B, M] pre-softmax scores
    """
    if x.dim() != 2:
        raise ValueError(f"x must be [B,D], got shape {tuple(x.shape)}")
    if X1.dim() != 2:
        raise ValueError(f"X1 must be [M,D], got shape {tuple(X1.shape)}")

    if t.dim() == 1:
        t = t[:, None]
    
    B, D = x.shape
    M = X1.shape[0]
    one_minus_t = (1.0 - t).clamp_min(eps)  # [B,1]

    # Sample epsilon ~ N(0, I)
    epsilon = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    x_noisy = x + sigma * epsilon  # [B, D]

    # Memory-efficient: process each x[i] separately when M is large
    if B * M * D > 1e9:
        w_list, u_list, scores_list = [], [], []
        for i in range(B):
            xi_noisy = x_noisy[i : i + 1]  # [1, D]
            xi = x[i : i + 1]  # [1, D]
            ti = t[i : i + 1]  # [1, 1]
            one_minus_ti = one_minus_t[i : i + 1]  # [1, 1]
            diff_i = xi_noisy - ti * X1  # [M, D]
            scores_i = -(diff_i.pow(2).sum(-1)) / (2.0 * one_minus_ti.pow(2))  # [M]
            w_i = torch.softmax(scores_i, dim=0)  # [M]
            v_i = (X1 - xi) / one_minus_ti  # [M, D] (use original x, not noisy)
            u_i = w_i @ v_i  # [M] @ [M, D] = [D]
            w_list.append(w_i)
            u_list.append(u_i)
            scores_list.append(scores_i)
        w = torch.stack(w_list, dim=0)  # [B, M]
        u_efm = torch.stack(u_list, dim=0)  # [B, D]
        scores = torch.stack(scores_list, dim=0)  # [B, M]
    else:
        # diff = x_noisy - t * X1_j  => [B, M, D]
        diff = x_noisy[:, None, :] - t[:, None, :] * X1[None, :, :]
        scores = -(diff.pow(2).sum(-1)) / (2.0 * one_minus_t.pow(2))
        w = torch.softmax(scores, dim=1)
        v = (X1[None, :, :] - x[:, None, :]) / one_minus_t[:, None, :]
        u_efm = (w[:, :, None] * v).sum(dim=1)

    return w, u_efm, scores


def efm_closed_form_weights_and_u_dct_4x4(
    x: torch.Tensor,
    X1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EFM closed-form but scores use DCT 4x4 low-frequency distance instead of Euclidean.
    
    CORRECT: Compute DCT on x and t*X1 separately, then compute distance in DCT space.

    Args:
        x: [B, 3, H, W] image batch
        X1: [M, 3, H, W] target image batch
        t: [B] or [B, 1]
        eps: clamp floor for (1 - t)

    Returns:
        w: [B, M], u_efm: [B, 3, H, W], scores: [B, M]
    """
    if x.dim() != 4 or X1.dim() != 4:
        raise ValueError(f"Expect 4D images, got x {tuple(x.shape)}, X1 {tuple(X1.shape)}")
    B, C, H, W = x.shape
    M = X1.shape[0]
    if t.dim() == 1:
        t = t[:, None]
    t = t.to(x.device).float()
    one_minus_t = (1.0 - t).clamp_min(eps)

    # Compute DCT on x: [B, 3, H, W] -> [B, 48]
    dct_x = _dct_4x4_low_flat(x)  # [B, 48]
    
    # Compute t * X1: [B, M, 3, H, W]
    tX1 = t[:, None, None, None, None] * X1[None, :, :, :, :]  # [B, M, 3, H, W]
    
    # Compute DCT on t*X1: reshape to [B*M, 3, H, W], extract DCT, reshape to [B, M, 48]
    BM = B * M
    tX1_flat = tX1.reshape(BM, C, H, W)
    dct_tX1 = _dct_4x4_low_flat(tX1_flat)  # [B*M, 48]
    dct_tX1 = dct_tX1.reshape(B, M, -1)  # [B, M, 48]
    
    # Compute distance in DCT space: dct_x - dct_tX1
    diff_dct = dct_x[:, None, :] - dct_tX1  # [B, 1, 48] - [B, M, 48] = [B, M, 48]
    
    # scores = -||diff_dct||^2 / (2(1-t)^2)
    dist_sq = (diff_dct ** 2).sum(dim=-1)  # [B, M]
    scores = -dist_sq / (2.0 * one_minus_t ** 2)

    # weights
    w = torch.softmax(scores, dim=1)

    # v_j = (X1_j - x) / (1-t) => [B, M, 3, H, W]
    v = (X1[None, :, :, :, :] - x[:, None, :, :, :]) / one_minus_t[:, None, None, None, None]

    # u_efm => [B, 3, H, W]: sum over m of w[b,m] * v[b,m,c,h,w]
    # w: [B, M], v: [B, M, C, H, W]
    w_bc = w.reshape(B, M, 1, 1, 1)
    prod = w_bc * v  # [B, M, C, H, W]
    u_efm = prod.sum(dim=1)  # [B, C, H, W]

    return w, u_efm, scores


def _efm_closed_form_weights_and_u_dct_zigzag(
    x: torch.Tensor,
    X1: torch.Tensor,
    t: torch.Tensor,
    num_coeffs: int,
    mode: str,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EFM with DCT zigzag distance. mode: 'low','high','mid'."""
    if x.dim() != 4 or X1.dim() != 4:
        raise ValueError(f"Expect 4D images, got x {tuple(x.shape)}, X1 {tuple(X1.shape)}")
    B, C, H, W = x.shape
    M = X1.shape[0]
    if t.dim() == 1:
        t = t[:, None]
    t = t.to(x.device).float()
    one_minus_t = (1.0 - t).clamp_min(eps)

    dct_x = _dct_zigzag_flat(x, num_coeffs, mode)  # [B, C*num_coeffs]
    t_bc = t.reshape(B, 1, 1, 1, 1)
    tX1 = t_bc * X1[None, :, :, :, :]  # [B, M, 3, H, W]
    tX1_flat = tX1.reshape(B * M, C, H, W)
    dct_tX1 = _dct_zigzag_flat(tX1_flat, num_coeffs, mode)  # [B*M, C*num_coeffs]
    dct_tX1 = dct_tX1.reshape(B, M, -1)

    diff_dct = dct_x[:, None, :] - dct_tX1
    dist_sq = (diff_dct ** 2).sum(dim=-1)
    scores = -dist_sq / (2.0 * one_minus_t ** 2)
    w = torch.softmax(scores, dim=1)

    omt_bc = one_minus_t.reshape(B, 1, 1, 1, 1)
    v = (X1[None, :, :, :, :] - x[:, None, :, :, :]) / omt_bc
    u_efm = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
    for m in range(M):
        u_efm = u_efm + w[:, m : m + 1].view(B, 1, 1, 1) * v[:, m]
    if u_efm.dim() == 5:
        u_efm = u_efm.sum(dim=1)
    return w, u_efm, scores


def efm_closed_form_weights_and_u_dct_8x8(
    x: torch.Tensor, X1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EFM with DCT 8x8 low-frequency distance (first 64 zigzag coeffs)."""
    return _efm_closed_form_weights_and_u_dct_zigzag(x, X1, t, 64, "low", eps)


def efm_closed_form_weights_and_u_dct_hf_8x8(
    x: torch.Tensor, X1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EFM with DCT 8x8 high-frequency distance (last 64 zigzag coeffs)."""
    return _efm_closed_form_weights_and_u_dct_zigzag(x, X1, t, 64, "high", eps)


def efm_closed_form_weights_and_u_dct_mf_8x8(
    x: torch.Tensor, X1: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """EFM with DCT 8x8 mid-frequency distance (middle 64 zigzag coeffs)."""
    return _efm_closed_form_weights_and_u_dct_zigzag(x, X1, t, 64, "mid", eps)


def efm_closed_form_weights_and_u_downsample_2x(
    x: torch.Tensor,
    X1: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EFM closed-form but scores use 2x downsampled (avg pool) distance instead of full-resolution Euclidean.
    Distance: ||downsample(x) - downsample(t*X1_j)||^2 in pixel space after 2x avg pool.

    Args:
        x: [B, 3, H, W] image batch
        X1: [M, 3, H, W] target image batch
        t: [B] or [B, 1]
        eps: clamp floor for (1 - t)

    Returns:
        w: [B, M], u_efm: [B, 3, H, W], scores: [B, M]
    """
    # x and X1 MUST be 4D [B, 3, H, W] and [M, 3, H, W]. If caller passes 5D, fix at call site.
    if x.dim() != 4 or X1.dim() != 4:
        raise ValueError(
            f"efm_downsample_2x requires 4D images [B,C,H,W] and [M,C,H,W]. Got x {tuple(x.shape)}, X1 {tuple(X1.shape)}"
        )
    B, C, H, W = x.shape
    M = X1.shape[0]
    if t.dim() == 1:
        t = t[:, None]  # [B] -> [B, 1]
    t = t.to(x.device).float()
    one_minus_t = (1.0 - t).clamp_min(eps)  # [B, 1]

    # 2x downsample x and X1: [B, 3, H/2, W/2] and [M, 3, H/2, W/2]
    down_x = F.adaptive_avg_pool2d(x, (H // 2, W // 2))  # [B, 3, H/2, W/2]
    down_X1 = F.adaptive_avg_pool2d(X1, (H // 2, W // 2))  # [M, 3, H/2, W/2]

    # diff = down_x - t * down_X1 in downsampled space. t is now [B, 1], need [B, 1, 1, 1] for broadcast.
    t_bc = t.reshape(B, 1, 1, 1)  # [B, 1, 1, 1]
    diff_down = down_x[:, None, :, :, :] - t_bc[:, None, :, :, :] * down_X1[None, :, :, :, :]  # [B, M, 3, H/2, W/2]
    dist_sq = (diff_down ** 2).sum(dim=(-3, -2, -1))  # [B, M]

    scores = -dist_sq / (2.0 * one_minus_t ** 2)
    w = torch.softmax(scores, dim=1)

    # v = (X1 - x) / (1-t), shape [B, M, C, H, W]. one_minus_t [B,1] -> [B,1,1,1,1] for broadcast
    omt_bc = one_minus_t.reshape(B, 1, 1, 1, 1)
    v = (X1[None, :, :, :, :] - x[:, None, :, :, :]) / omt_bc
    # v is [B, M, C, H, W]. Sum over M -> u_efm [B, C, H, W]. Force 4D via loop (sum(dim=1) can be wrong in some envs).
    u_efm = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
    for m in range(M):
        u_efm = u_efm + w[:, m : m + 1].view(B, 1, 1, 1) * v[:, m]
    if u_efm.dim() == 5:
        u_efm = u_efm.sum(dim=1)  # [B, M, C, H, W] -> [B, C, H, W]

    return w, u_efm, scores

