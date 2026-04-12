"""
analyze_sinkformer_2d.py
========================
Post-hoc analysis for ot_cfm_vs_sinkformer_2d trained checkpoints.

1) Fixed anchor + many random batches: each trial resamples the other (B-1) x0
   and all B x1; collect where the anchor row/column is “sent” (soft expectation,
   Hungarian 1-1 on P) vs OT (EMD) partner — empirical cloud in R^2.
2) Trajectory ribbons: Euler integrate both FlowNets, color by time t or |v|.

Run (match training hyperparams: --batch, --outdir, --sink-iters):
  python experiments/analyze_sinkformer_2d.py --outdir experiments/results/sinkformer_2d_b64_v2 --batch 64

Old pairer.pt checkpoints (cdist head or self-attn on batch) do not match current SinkPairer weights;
retrain before analysing, or load_state_dict will error / mismatch shapes.
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _EXP)
from torchcfm.utils import sample_8gaussians, sample_moons
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


@torch.no_grad()
def integrate_trajectory(net, x0, steps=200, device=None):
    """
    x0: [n, 2]. Returns positions [steps+1, n, 2], t_grid [steps+1], speeds [steps, n].
    """
    device = device or x0.device
    x = x0.clone()
    dt = 1.0 / steps
    positions = [x.cpu().numpy()]
    speeds = []
    t_grid = [0.0]
    for k in range(steps):
        t = torch.full((1,), k * dt, device=device)
        v = net(x, t.expand(x.shape[0]))
        speeds.append(v.norm(dim=-1).cpu().numpy())
        x = x + v * dt
        positions.append(x.cpu().numpy())
        t_grid.append((k + 1) * dt)
    return np.stack(positions), np.array(t_grid), np.stack(speeds)


@torch.no_grad()
def plot_fixed_x0_marginal(pairer, B, device, outpath, n_trials=500, seed=0, anchor_idx=0):
    """
    Fix one anchor x0; repeat n_trials times with fresh random (B-1) x0 + B x1.
    For each trial record where row anchor_idx sends mass in x1-space:
      - soft:  y = sum_j (B*P[i,j]) x1[j]
      - hard:  x1[j] from Hungarian matching on P (max-weight perfect matching)
      - OT:    x1[j] from **Hungarian(M)** (same as discrete OT), not argmax on P_ot
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    x0_anchor = sample_8gaussians(1).to(device)
    anchor_xy = x0_anchor.cpu().numpy()[0]

    soft_y, hung_y, ot_y = [], [], []

    for tr in range(n_trials):
        torch.manual_seed(seed + 1 + tr)
        np.random.seed(seed + 1 + tr)
        x0_rest = sample_8gaussians(B - 1).to(device)
        x0 = torch.cat([x0_anchor, x0_rest], dim=0)
        x1 = sample_moons(B).to(device)

        P = pairer(x0, x1).detach().cpu().numpy()
        x1n = x1.cpu().numpy()
        x0n = x0.cpu().numpy()
        M_cost = np.sum((x0n[:, None] - x1n[None]) ** 2, axis=-1)
        r_ot, c_ot = linear_sum_assignment(M_cost)
        perm_ot = np.empty(B, dtype=np.int64)
        perm_ot[r_ot] = c_ot

        i = anchor_idx
        row = P[i]
        mass = row * B
        y_soft = (mass[:, None] * x1n).sum(axis=0)
        soft_y.append(y_soft)

        ri, ci = linear_sum_assignment(-P)
        j_h = ci[ri == i][0]
        hung_y.append(x1n[j_h].copy())

        j_ot = int(perm_ot[i])
        ot_y.append(x1n[j_ot].copy())

    soft_y = np.stack(soft_y)
    hung_y = np.stack(hung_y)
    ot_y = np.stack(ot_y)

    ref = sample_moons(4000).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7))
    ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.08, c='gray', label='moon ref')
    ax.scatter(soft_y[:, 0], soft_y[:, 1], s=14, alpha=0.45, c='darkorange', label=f'Sink soft E[x1|row {i}]', edgecolors='none')
    ax.scatter(hung_y[:, 0], hung_y[:, 1], s=10, alpha=0.35, c='crimson', marker='x', label='Sink Hungarian x1')
    ax.scatter(ot_y[:, 0], ot_y[:, 1], s=10, alpha=0.35, c='steelblue', marker='+', label='OT (EMD) x1')
    ax.scatter(anchor_xy[0], anchor_xy[1], s=280, c='navy', marker='*', zorder=10, edgecolors='k', linewidths=0.5, label='fixed anchor x0')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(
        f'Fixed x0 anchor — {n_trials} random batches (other {B-1} x0 + {B} x1 resampled)\n'
        f'clouds = empirical distribution of paired target in R² (compare to OT)',
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {outpath}')


@torch.no_grad()
def plot_fixed_x1_marginal(pairer, B, device, outpath, n_trials=500, seed=1, anchor_idx=0):
    """
    Fix one anchor x1; repeat n_trials with fresh random B x0 and (B-1) other x1.
    Record source in x0-space for column anchor_idx:
      - soft:  E[x0 | col j] = sum_i (B*P[i,j]) x0[i]
      - hard:  x0[i] from Hungarian row matched to column j
      - OT:    x0[i] from **Hungarian(M)** (permutation inverse), not argmax on P_ot
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    x1_anchor = sample_moons(1).to(device)
    anchor_xy = x1_anchor.cpu().numpy()[0]

    soft_x, hung_x, ot_x = [], [], []

    for tr in range(n_trials):
        torch.manual_seed(seed + 1 + tr)
        np.random.seed(seed + 1 + tr)
        x1_rest = sample_moons(B - 1).to(device)
        x1 = torch.cat([x1_anchor, x1_rest], dim=0)
        x0 = sample_8gaussians(B).to(device)

        P = pairer(x0, x1).detach().cpu().numpy()
        x0n = x0.cpu().numpy()
        x1n = x1.cpu().numpy()
        M_cost = np.sum((x0n[:, None] - x1n[None]) ** 2, axis=-1)
        r_ot, c_ot = linear_sum_assignment(M_cost)
        perm_ot = np.empty(B, dtype=np.int64)
        perm_ot[r_ot] = c_ot

        j = anchor_idx
        col = P[:, j]
        mass = col * B
        x_soft = (mass[:, None] * x0n).sum(axis=0)
        soft_x.append(x_soft)

        ri, ci = linear_sum_assignment(-P)
        i_h = ri[ci == j][0]
        hung_x.append(x0n[i_h].copy())

        i_ot = int(np.where(perm_ot == j)[0][0])
        ot_x.append(x0n[i_ot].copy())

    soft_x = np.stack(soft_x)
    hung_x = np.stack(hung_x)
    ot_x = np.stack(ot_x)

    ref = sample_8gaussians(4000).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7))
    ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.08, c='gray', label='8-Gauss ref')
    ax.scatter(soft_x[:, 0], soft_x[:, 1], s=14, alpha=0.45, c='darkorange', label=f'Sink soft E[x0|col {j}]', edgecolors='none')
    ax.scatter(hung_x[:, 0], hung_x[:, 1], s=10, alpha=0.35, c='crimson', marker='x', label='Sink Hungarian x0')
    ax.scatter(ot_x[:, 0], ot_x[:, 1], s=10, alpha=0.35, c='steelblue', marker='+', label='OT (EMD) x0')
    ax.scatter(anchor_xy[0], anchor_xy[1], s=280, c='darkred', marker='*', zorder=10, edgecolors='k', linewidths=0.5, label='fixed anchor x1')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(
        f'Fixed x1 anchor — {n_trials} random batches ({B} x0 + other {B-1} x1 resampled)\n'
        f'clouds = empirical distribution of paired source in R² (compare to OT)',
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {outpath}')


@torch.no_grad()
def plot_trajectories_colored(net_ot, net_sf, device, outpath, n_paths=80, steps=200, seed=2):
    """Few particles, both models; color by time t along path."""
    torch.manual_seed(seed)
    x0 = sample_8gaussians(n_paths).to(device)

    pos_ot, t_grid, spd_ot = integrate_trajectory(net_ot, x0, steps=steps, device=device)
    pos_sf, _, spd_sf = integrate_trajectory(net_sf, x0, steps=steps, device=device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ref = sample_moons(3000).numpy()

    for ax, pos, title in [(axes[0], pos_ot, 'OT-CFM'), (axes[1], pos_sf, 'Sinkformer-CFM')]:
        ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.12, c='gray')
        for p in range(n_paths):
            path = pos[:, p, :]
            for k in range(len(t_grid) - 1):
                c = plt.cm.plasma(t_grid[k])
                ax.plot(path[k:k + 2, 0], path[k:k + 2, 1], color=c, lw=1.2, alpha=0.85)
        ax.scatter(pos[0, :, 0], pos[0, :, 1], c='royalblue', s=25, zorder=4, label='t=0')
        ax.scatter(pos[-1, :, 0], pos[-1, :, 1], c='darkorange', s=25, zorder=4, label='t=1')
        ax.set_title(title + '  (line color = time t)')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, label='t')

    plt.suptitle('Same initial x0 trajectories: OT vs Sinkformer velocity fields', fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {outpath}')

    # Second figure: color by |v| at segment midpoint
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, pos, spd, title in [
            (axes[0], pos_ot, spd_ot, 'OT-CFM  (|v|)'),
            (axes[1], pos_sf, spd_sf, 'Sinkformer-CFM  (|v|)')]:
        ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.12, c='gray')
        vmax = max(spd_ot.max(), spd_sf.max())
        for p in range(n_paths):
            for k in range(steps):
                vm = 0.5 * (spd[k, p] + spd[k + 1, p]) if k + 1 < len(pos) - 1 else spd[k, p]
                c = plt.cm.viridis(np.clip(vm / (vmax + 1e-8), 0, 1))
                seg = pos[k:k + 2, p, :]
                ax.plot(seg[:, 0], seg[:, 1], color=c, lw=1.2, alpha=0.9)
        ax.set_title(title)
        ax.set_aspect('equal')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, label='|v|')

    plt.suptitle('Same trajectories colored by speed |v(x,t)|', fontsize=11)
    plt.tight_layout()
    out_v = outpath.replace('.png', '_speed.png')
    plt.savefig(out_v, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_v}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, required=True, help='Folder with net_ot.pt, net_sf.pt, pairer.pt')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--d_sink', type=int, default=16)
    ap.add_argument('--sink-iters', type=int, default=20, help='Must match training SinkPairer.n_sink_iters')
    ap.add_argument('--cpu', action='store_true', help='Force CPU (avoids some GPU/driver issues)')
    ap.add_argument('--n-trials', type=int, default=500, help='Monte Carlo trials for fixed-anchor marginal plots')
    ap.add_argument('--n-paths', type=int, default=80, help='Number of particles for trajectory figures')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    B = args.batch

    pairer = SinkPairer(args.d_sink, n_sink_iters=args.sink_iters).to(device)
    net_ot = FlowNet(args.hidden).to(device)
    net_sf = FlowNet(args.hidden).to(device)

    pairer.load_state_dict(torch.load(os.path.join(args.outdir, 'pairer.pt'), map_location=device))
    net_ot.load_state_dict(torch.load(os.path.join(args.outdir, 'net_ot.pt'), map_location=device))
    net_sf.load_state_dict(torch.load(os.path.join(args.outdir, 'net_sf.pt'), map_location=device))
    pairer.eval()
    net_ot.eval()
    net_sf.eval()

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    print(f'Device: {device}  B={B}  n_trials={args.n_trials}  n_paths={args.n_paths}')
    plot_fixed_x0_marginal(
        pairer, B, device, os.path.join(out_dir, 'analysis_fixed_x0_marginal.png'),
        n_trials=args.n_trials,
    )
    plot_fixed_x1_marginal(
        pairer, B, device, os.path.join(out_dir, 'analysis_fixed_x1_marginal.png'),
        n_trials=args.n_trials,
    )
    plot_trajectories_colored(
        net_ot, net_sf, device, os.path.join(out_dir, 'analysis_trajectories_t.png'),
        n_paths=args.n_paths,
    )
    print('Done.')


if __name__ == '__main__':
    main()
