"""
visualize_uv_quiver_2d.py
=========================
Load trained OT-CFM + Sinkformer 2d checkpoints; same x0,x1 batch; **one scalar t** shared by
all pairs and by **both** OT and Sinkformer rows (fair cross-method comparison). Training may
use per-pair t — this script defaults to shared t for visualization only.

Faint pairing segments, x0/x1/x_t markers, quiver at x_t (arrow color = ||u-v||); one shared colorbar.

Run:
  python experiments/visualize_uv_quiver_2d.py --ckpt-dir experiments/results/sinkformer_2d_b64_v2
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _EXP)
from torchcfm.utils import sample_8gaussians, sample_moons
from scipy.optimize import linear_sum_assignment
from sinkformer_2d_pairer import SinkPairer


class FlowNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t.view(-1, 1).expand(x.shape[0], 1)], 1))


def ot_pairs(x0, x1, B):
    M = torch.cdist(x0, x1).pow(2).detach().cpu().numpy()
    r, c = linear_sum_assignment(M)
    perm = np.empty(B, dtype=np.int64)
    perm[r] = c
    return x0, x1[torch.as_tensor(perm, device=x1.device, dtype=torch.long)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', type=str, default='experiments/results/sinkformer_2d_b64_v2')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--d-sink', type=int, default=16)
    ap.add_argument('--sink-iters', type=int, default=20, help='Must match training SinkPairer.n_sink_iters')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument(
        '--per-pair-t', action='store_true',
        help='Sample t independently per pair (matches per-pair training); default is one shared t',
    )
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    B = args.batch
    ckpt = args.ckpt_dir

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    net_ot = FlowNet(args.hidden).to(device)
    net_sf = FlowNet(args.hidden).to(device)
    pairer = SinkPairer(args.d_sink, n_sink_iters=args.sink_iters).to(device)

    net_ot.load_state_dict(torch.load(os.path.join(ckpt, 'net_ot.pt'), map_location=device))
    net_sf.load_state_dict(torch.load(os.path.join(ckpt, 'net_sf.pt'), map_location=device))
    pairer.load_state_dict(torch.load(os.path.join(ckpt, 'pairer.pt'), map_location=device))
    net_ot.eval()
    net_sf.eval()
    pairer.eval()

    x0 = sample_8gaussians(B).to(device)
    x1 = sample_moons(B).to(device)
    if args.per_pair_t:
        t = torch.rand(B, device=device)
        tv = t.view(B, 1)
        t_note = 'per-pair t'
    else:
        t_scalar = torch.rand(1, device=device)
        t = t_scalar.expand(B)
        tv = t_scalar.expand(B, 1)
        t_note = 'shared scalar t (OT & SF identical)'

    # OT-CFM
    x0p, x1p = ot_pairs(x0, x1, B)
    xt_ot = (1 - tv) * x0p + tv * x1p
    v_ot = (x1p - x0p).cpu().numpy()
    u_ot = net_ot(xt_ot, t).cpu().numpy()
    x_ot = xt_ot.cpu().numpy()
    err_ot = np.linalg.norm(u_ot - v_ot, axis=1)

    # Sinkformer
    P = pairer(x0, x1)
    x0_soft = torch.mm(P.t() * B, x0)
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
    print(f'{t_note}  |  mean(t)={t_val_mean:.4f}  std(t)={t_val_std:.4f}')
    print(f'OT-CFM: mean ||u-v|| = {err_ot.mean():.6f}  max = {err_ot.max():.6f}')
    print(f'Sinkformer: mean ||u-v|| = {err_sf.mean():.6f}  max = {err_sf.max():.6f}')

    def vecs_scaled(V, span, frac=0.22):
        """
        V: [B,2] raw vectors. Scale so max length = frac*span; smaller ||V_i|| → shorter arrow.
        """
        m = float(np.linalg.norm(V, axis=1).max()) + 1e-12
        U = V * (frac * span / m)
        return U[:, 0], U[:, 1]

    span_ot = max(np.ptp(x_ot[:, 0]), np.ptp(x_ot[:, 1]), 1e-6)
    span_sf = max(np.ptp(x_sf[:, 0]), np.ptp(x_sf[:, 1]), 1e-6)
    vmax = max(float(err_ot.max()), float(err_sf.max()), 1e-8)
    norm = Normalize(vmin=0.0, vmax=vmax)

    def bg_ot(ax):
        """Paired OT: faint segment x0p[i]–x1p[i]; xt on segment; scatter x0/x1."""
        for i in range(B):
            ax.plot(
                [x0p_np[i, 0], x1p_np[i, 0]], [x0p_np[i, 1], x1p_np[i, 1]],
                color='0.5', alpha=0.2, lw=0.9, zorder=1,
            )
        ax.scatter(x0p_np[:, 0], x0p_np[:, 1], s=14, c='steelblue', alpha=0.45, zorder=2, label='$x_0$ (paired)')
        ax.scatter(x1p_np[:, 0], x1p_np[:, 1], s=14, c='indianred', alpha=0.45, zorder=2, label='$x_1$ (paired)')
        ax.scatter(
            x_ot[:, 0], x_ot[:, 1], s=22, facecolors='white', edgecolors='k',
            linewidths=0.6, alpha=0.9, zorder=3, label='$x_t$',
        )

    def bg_sf(ax):
        """
        SF path: x_t = (1-t) * x0_soft[i] + t * x1[i], with
        x0_soft[i] = sum_j (B*P[j,i]) * x0[j]  — convex combo of **all** batch x0, not x0[i] alone.
        Gray lines connect x0_soft[i]→x1[i]; blue = sampled x0; orange = soft source per index i.
        """
        for i in range(B):
            ax.plot(
                [x0_soft_np[i, 0], x1_np[i, 0]], [x0_soft_np[i, 1], x1_np[i, 1]],
                color='0.5', alpha=0.2, lw=0.9, zorder=1,
            )
        ax.scatter(x0_raw_np[:, 0], x0_raw_np[:, 1], s=14, c='steelblue', alpha=0.45, zorder=2, label='$x_0$ (samples)')
        ax.scatter(
            x0_soft_np[:, 0], x0_soft_np[:, 1], s=36, c='darkorange', alpha=0.75,
            marker='s', edgecolors='saddlebrown', linewidths=0.4, zorder=2,
            label=r'$\tilde x_0$ (soft)',
        )
        ax.scatter(x1_np[:, 0], x1_np[:, 1], s=14, c='indianred', alpha=0.45, zorder=2, label='$x_1$')
        ax.scatter(
            x_sf[:, 0], x_sf[:, 1], s=22, facecolors='white', edgecolors='k',
            linewidths=0.6, alpha=0.9, zorder=3, label='$x_t$',
        )

    def panel(ax, xt_xy, vx, vy, err, title, span, bg_fn):
        bg_fn(ax)
        V = np.stack([vx, vy], axis=1)
        qx, qy = vecs_scaled(V, span)
        ax.quiver(
            xt_xy[:, 0], xt_xy[:, 1], qx, qy, err, cmap='magma', norm=norm,
            angles='xy', scale_units='xy', scale=1.0,
            width=0.006, headwidth=4, headlength=5, zorder=5,
        )
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.22)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    fig.suptitle(
        f'u = net(xt,t), v = target  |  {t_note}  t≈{t_val_mean:.3f}  B={B}\n'
        r'Arrow length $\propto$ |shown vector|; color $=$ $||u-v||$.  '
        r'Sinkformer: $\tilde x_0=(P^\top B)x_0$ mixes all blue samples — segment starts at orange $\tilde x_0$, not $x_0^{(i)}$.',
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
    out = os.path.join(ckpt, 'uv_quiver_ot_vs_sf.png')
    os.makedirs(ckpt, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
