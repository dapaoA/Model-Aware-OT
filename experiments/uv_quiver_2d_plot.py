"""
UV quiver diagnostic for OT-CFM vs Sinkformer 2D (same batch, shared t, theory v vs pred u).

Used by visualize_uv_quiver_2d.py and ot_cfm_vs_sinkformer_2d.py (--viz-uv-quiver).
Sinkformer uses P_fm = Hungarian STE when hard_st else soft P (must match training).
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import linear_sum_assignment

from torchcfm.utils import sample_8gaussians, sample_moons


def hungarian_hard_plan_from_P(P: torch.Tensor, batch_size: int) -> torch.Tensor:
    P_np = P.detach().cpu().numpy()
    r, c = linear_sum_assignment(-P_np)
    P_hard = torch.zeros_like(P)
    P_hard[r, c] = 1.0 / float(batch_size)
    return P_hard.to(device=P.device, dtype=P.dtype)


def ste_hard_P(P: torch.Tensor, batch_size: int) -> torch.Tensor:
    p_star = hungarian_hard_plan_from_P(P, batch_size)
    return P + (p_star - P).detach()


def ot_pairs(x0: torch.Tensor, x1: torch.Tensor, batch_size: int):
    M = torch.cdist(x0, x1).pow(2).detach().cpu().numpy()
    r, c = linear_sum_assignment(M)
    perm = np.empty(batch_size, dtype=np.int64)
    perm[r] = c
    return x0, x1[torch.as_tensor(perm, device=x1.device, dtype=torch.long)]


@torch.no_grad()
def save_uv_quiver_figure(
    out_path: str,
    net_ot: nn.Module,
    net_sf: nn.Module,
    pairer: nn.Module,
    device: torch.device,
    batch: int,
    vis_seed: int,
    hard_st: bool,
    per_pair_t: bool = False,
) -> None:
    """
    One sample batch; shared scalar t unless per_pair_t. Saves 2x3 quiver figure to out_path.
    """
    dn = os.path.dirname(out_path)
    if dn:
        os.makedirs(dn, exist_ok=True)

    B = batch
    x0 = sample_8gaussians(B).to(device)
    x1 = sample_moons(B, random_state=int(vis_seed) + 10_007).to(device)
    if per_pair_t:
        t = torch.rand(B, device=device)
        tv = t.view(B, 1)
        t_note = 'per-pair t'
    else:
        t_scalar = torch.rand(1, device=device)
        t = t_scalar.expand(B)
        tv = t_scalar.expand(B, 1)
        t_note = 'shared scalar t (OT & SF identical)'

    net_ot.eval()
    net_sf.eval()
    pairer.eval()

    x0p, x1p = ot_pairs(x0, x1, B)
    xt_ot = (1 - tv) * x0p + tv * x1p
    v_ot = (x1p - x0p).cpu().numpy()
    u_ot = net_ot(xt_ot, t).cpu().numpy()
    x_ot = xt_ot.cpu().numpy()
    err_ot = np.linalg.norm(u_ot - v_ot, axis=1)

    P = pairer(x0, x1)
    if hard_st:
        P_fm = ste_hard_P(P, B)
    else:
        P_fm = P
    x0_soft = torch.mm(P_fm.t() * B, x0)
    xt_sf = (1 - tv) * x0_soft + tv * x1
    v_sf = (x1 - x0_soft).cpu().numpy()
    u_sf = net_sf(xt_sf, t).cpu().numpy()
    x_sf = xt_sf.cpu().numpy()
    err_sf = np.linalg.norm(u_sf - v_sf, axis=1)

    x0p_np = x0p.cpu().numpy()
    x1p_np = x1p.cpu().numpy()
    x0_raw_np = x0.cpu().numpy()
    x1_np = x1.cpu().numpy()
    x0_soft_np = x0_soft.cpu().numpy()

    t_val_mean = float(t.mean().item())
    t_val_std = float(t.std().item()) if B > 1 else 0.0

    def vecs_scaled(V, span, frac=0.22):
        m = float(np.linalg.norm(V, axis=1).max()) + 1e-12
        U = V * (frac * span / m)
        return U[:, 0], U[:, 1]

    span_ot = max(np.ptp(x_ot[:, 0]), np.ptp(x_ot[:, 1]), 1e-6)
    span_sf = max(np.ptp(x_sf[:, 0]), np.ptp(x_sf[:, 1]), 1e-6)
    vmax = max(float(err_ot.max()), float(err_sf.max()), 1e-8)
    norm = Normalize(vmin=0.0, vmax=vmax)

    def bg_ot(ax):
        for i in range(B):
            ax.plot(
                [x0p_np[i, 0], x1p_np[i, 0]],
                [x0p_np[i, 1], x1p_np[i, 1]],
                color='0.5',
                alpha=0.2,
                lw=0.9,
                zorder=1,
            )
        ax.scatter(x0p_np[:, 0], x0p_np[:, 1], s=14, c='steelblue', alpha=0.45, zorder=2, label='$x_0$ (paired)')
        ax.scatter(x1p_np[:, 0], x1p_np[:, 1], s=14, c='indianred', alpha=0.45, zorder=2, label='$x_1$ (paired)')
        ax.scatter(
            x_ot[:, 0],
            x_ot[:, 1],
            s=22,
            facecolors='white',
            edgecolors='k',
            linewidths=0.6,
            alpha=0.9,
            zorder=3,
            label='$x_t$',
        )

    def bg_sf(ax):
        for i in range(B):
            ax.plot(
                [x0_soft_np[i, 0], x1_np[i, 0]],
                [x0_soft_np[i, 1], x1_np[i, 1]],
                color='0.5',
                alpha=0.2,
                lw=0.9,
                zorder=1,
            )
        ax.scatter(x0_raw_np[:, 0], x0_raw_np[:, 1], s=14, c='steelblue', alpha=0.45, zorder=2, label='$x_0$ (samples)')
        ax.scatter(
            x0_soft_np[:, 0],
            x0_soft_np[:, 1],
            s=36,
            c='darkorange',
            alpha=0.75,
            marker='s',
            edgecolors='saddlebrown',
            linewidths=0.4,
            zorder=2,
            label=r'$\tilde x_0$ (soft/hard STE)',
        )
        ax.scatter(x1_np[:, 0], x1_np[:, 1], s=14, c='indianred', alpha=0.45, zorder=2, label='$x_1$')
        ax.scatter(
            x_sf[:, 0],
            x_sf[:, 1],
            s=22,
            facecolors='white',
            edgecolors='k',
            linewidths=0.6,
            alpha=0.9,
            zorder=3,
            label='$x_t$',
        )

    def panel(ax, xt_xy, vx, vy, err, title, span, bg_fn):
        bg_fn(ax)
        V = np.stack([vx, vy], axis=1)
        qx, qy = vecs_scaled(V, span)
        ax.quiver(
            xt_xy[:, 0],
            xt_xy[:, 1],
            qx,
            qy,
            err,
            cmap='magma',
            norm=norm,
            angles='xy',
            scale_units='xy',
            scale=1.0,
            width=0.006,
            headwidth=4,
            headlength=5,
            zorder=5,
        )
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.22)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    ste_tag = 'Hungarian STE' if hard_st else 'soft P'
    fig.suptitle(
        f'u = net(xt,t), v = target  |  {t_note}  t≈{t_val_mean:.3f}  B={B}  ({ste_tag})\n'
        r'Arrow length $\propto$ |shown vector|; color $=$ $||u-v||$.  '
        r'Sinkformer: $\tilde x_0 = P_{\mathrm{fm}}^{\top} B\,x_0$.',
        fontsize=9,
    )

    res_ot = u_ot - v_ot
    res_sf = u_sf - v_sf

    panel(axes[0, 0], x_ot, v_ot[:, 0], v_ot[:, 1], err_ot, 'OT-CFM: theory v', span_ot, bg_ot)
    panel(axes[0, 1], x_ot, u_ot[:, 0], u_ot[:, 1], err_ot, 'OT-CFM: pred u', span_ot, bg_ot)
    panel(axes[0, 2], x_ot, res_ot[:, 0], res_ot[:, 1], err_ot, 'OT-CFM: residual u−v', span_ot, bg_ot)
    axes[0, 0].legend(fontsize=7, loc='upper right')

    panel(axes[1, 0], x_sf, v_sf[:, 0], v_sf[:, 1], err_sf, 'Sinkformer: theory v', span_sf, bg_sf)
    panel(axes[1, 1], x_sf, u_sf[:, 0], u_sf[:, 1], err_sf, 'Sinkformer: pred u', span_sf, bg_sf)
    panel(axes[1, 2], x_sf, res_sf[:, 0], res_sf[:, 1], err_sf, 'Sinkformer: residual u−v', span_sf, bg_sf)
    axes[1, 0].legend(fontsize=7, loc='upper right')

    sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.04, pad=0.02)
    cbar.set_label(r'$||u-v||$')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
