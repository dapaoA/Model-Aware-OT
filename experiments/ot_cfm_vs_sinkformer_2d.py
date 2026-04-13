"""
ot_cfm_vs_sinkformer_2d.py
8Gaussians -> 2Moons, batch=32 (default)

OT-CFM   : exact minibatch EMD pairing  -> standard FM loss
Sinkformer (default = **hard** Hungarian + STE):
  - Bipartite **pre-softmax** scores S_ij = (Q enc0(x0)_i)·(K enc1(x1)_j)/sqrt(d); log K_ij = S_ij/tau
  - Sinkhorn on K -> soft plan P (marginals 1/B); no x0↔x0 or x1↔x1 self-attention in the pairer
  - Hungarian: P* = hard 1-1 plan from P (max-weight matching on P)
  - Straight-through: P_fm = P + stop_grad(P* − P)  (forward uses P*; ∂L/∂θ flows through P)
  - FM: x0_soft = (B P_fm^T) @ x0, same CFM loss as below

Soft training only if: --soft-pair  (then P_fm = P, full grad, no STE)

--hungarian-ste is a no-op (hard STE is default); kept so old shell lines still parse.

Time sampling: **one shared** t ~ Uniform(0,1) per step (same for all pairs in the batch).
  Use --per-pair-t for independent t ~ U(0,1) per pair.

Outputs under --outdir (created automatically):
  viz_panels/step{N}.png          — 6-panel training figure
  viz_trajectories/step{N}.png  — if --viz-trajectories
  viz_quiver/step{N}.png        — if --viz-uv-quiver (theory v vs pred u)
  checkpoints/step{N}/          — if --save-ckpt-every > 0 (default 10000): net_*.pt + losses.npz
  net_*.pt, losses.npz          — final weights at run root (unchanged)

Run:
  python experiments/ot_cfm_vs_sinkformer_2d.py
  python experiments/ot_cfm_vs_sinkformer_2d.py --quick
  python experiments/ot_cfm_vs_sinkformer_2d.py ... --viz-trajectories --viz-uv-quiver

Frozen pairer ablation (train flow nets from scratch; pairer fixed as selection only):
  python experiments/ot_cfm_vs_sinkformer_2d.py --frozen-pairer experiments/results/sinkformer_2d_b64_v2/pairer.pt \\
      --outdir experiments/results/sinkformer_2d_frozen_pairer_50k --steps 50000 --batch 64

Use --soft-pair only if you want to train without Hungarian STE.
"""
import os, sys, time, warnings, argparse, gc
import numpy as np
import torch, torch.nn as nn
import ot as pot
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore', category=UserWarning)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _EXP)
from torchcfm.utils import sample_8gaussians, sample_moons
from sinkformer_2d_pairer import SinkPairer
from uv_quiver_2d_plot import save_uv_quiver_figure

parser = argparse.ArgumentParser()
parser.add_argument('--steps',     type=int,   default=200_000)
parser.add_argument('--batch',     type=int,   default=32)
parser.add_argument('--lr',        type=float, default=1e-3)
parser.add_argument('--hidden',    type=int,   default=64)
parser.add_argument('--d_sink',    type=int,   default=16)
parser.add_argument('--log_every', type=int,   default=200)
parser.add_argument('--vis_every', type=int,   default=40_000)
parser.add_argument('--seed',      type=int,   default=42)
parser.add_argument(
    '--vis-seed', type=int, default=None,
    help='Fixed RNG for visualize() only (ref moons, gen init x0, bottom-row batch). '
         'Default: same as --seed so plots are reproducible across training steps.',
)
parser.add_argument('--quick',     action='store_true')
parser.add_argument('--outdir',    type=str,   default='experiments/results/sinkformer_2d')
parser.add_argument(
    '--frozen-pairer', type=str, default=None,
    help='Path to pretrained pairer.pt: load, freeze, train both FlowNets from scratch (SF uses fixed P for selection only)',
)
parser.add_argument(
    '--sink-iters', type=int, default=20,
    help='Sinkhorn iterations on K=exp(S/tau) (S bipartite pre-softmax scores) before FM (default 20).',
)
parser.add_argument(
    '--hungarian-ste', action='store_true',
    help='No-op: Hungarian STE is already the default (see --soft-pair to disable).',
)
parser.add_argument(
    '--soft-pair', action='store_true',
    help='Sinkformer: train with soft P only (no Hungarian / no STE). Default: hard P* with STE.',
)
parser.add_argument(
    '--per-pair-t', action='store_true',
    help='Sample t independently per pair (shape [B]). Default: one shared t per step.',
)
parser.add_argument(
    '--vis-light', action='store_true',
    help='Faster viz: use smaller subsample for W2/EMD in scatter titles (still shows W2).',
)
parser.add_argument(
    '--fig-save-every', '--ckpt-every', type=int, default=0, dest='fig_save_every',
    help='If >0, save viz_panels/step{N:07d}.png (same 6-panel viz as vis_every) every N steps. '
         'Merged with vis_every so each matching step is saved once. Does not save .pt.',
)
parser.add_argument(
    '--viz-trajectories', action='store_true',
    help='Whenever panel viz is saved, also save viz_trajectories/step{N}.png: same initial '
         'x0 for OT vs Sinkformer, segment color = time t (like analyze_trajectories_t).',
)
parser.add_argument('--trajectory-paths', type=int, default=80, help='Number of particles for trajectory viz')
parser.add_argument('--trajectory-steps', type=int, default=200, help='Euler steps 0..1 for trajectory viz')
parser.add_argument(
    '--trajectory-seed', type=int, default=None,
    help='RNG seed for initial x0 in trajectory plot (default: vis_seed + 90210).',
)
parser.add_argument(
    '--save-ckpt-every', type=int, default=10_000,
    help='Save net_ot/net_sf/pairer + losses.npz under outdir/checkpoints/step{N}/ every N steps; 0 disables.',
)
parser.add_argument(
    '--viz-uv-quiver', action='store_true',
    help='When saving panel viz, also save viz_quiver/step{N}.png (theory v, pred u, residual u−v).',
)
args = parser.parse_args()

if args.quick:
    args.steps = 5_000; args.vis_every = 2_500; args.log_every = 50

torch.manual_seed(args.seed); np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.outdir, exist_ok=True)
B = args.batch
VIS_SEED = args.vis_seed if args.vis_seed is not None else args.seed

PANEL_DIR = os.path.join(args.outdir, 'viz_panels')
TRAJ_DIR = os.path.join(args.outdir, 'viz_trajectories')
QUIVER_DIR = os.path.join(args.outdir, 'viz_quiver')
CKPT_ROOT = os.path.join(args.outdir, 'checkpoints')
for _sub in (PANEL_DIR, TRAJ_DIR, QUIVER_DIR, CKPT_ROOT):
    os.makedirs(_sub, exist_ok=True)

# ── Flow network ──────────────────────────────────────────────────
class FlowNet(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 2))
    def forward(self, x, t):
        return self.net(torch.cat([x, t.view(-1,1).expand(x.shape[0],1)], 1))

# ── OT pairing ────────────────────────────────────────────────────
def ot_pairs(x0, x1):
    """
    Exact **permutation** pairing for uniform batch OT: minimize sum_i ||x0[i]-x1[j_i]||^2.
    Uses Hungarian on cost M (same optimum as pot.emd here, B=B).

    **Do not** use row-wise argmax on pot.emd’s P: the plan is doubly stochastic but
    argmax per row is **not** a bijection — multiple rows can pick the same column, so
    some x1 never appear (what you saw on the plot). Training used to share that bug.
    """
    M = torch.cdist(x0, x1).pow(2).detach().cpu().numpy()
    r, c = linear_sum_assignment(M)
    perm = np.empty(B, dtype=np.int64)
    perm[r] = c
    return x0, x1[torch.as_tensor(perm, device=x1.device, dtype=torch.long)]

def hungarian_hard_plan_from_P(P: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Max-weight perfect matching on P (same as viz: linear_sum_assignment(-P)).
    Returns P_hard with P_hard[i,j]=1/B on chosen edges, 0 elsewhere (same marginals as soft P).
    """
    P_np = P.detach().cpu().numpy()
    r, c = linear_sum_assignment(-P_np)
    P_hard = torch.zeros_like(P)
    P_hard[r, c] = 1.0 / float(batch_size)
    return P_hard.to(device=P.device, dtype=P.dtype)

def ste_hard_P(P: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Straight-through hard transport plan.

    Let P_soft = P (Sinkhorn output), P* = Hungarian hard plan from P_soft.
    Forward uses P_fm = P*  equivalently  P_fm = P_soft + (P* − P_soft) with the
    difference detached from the graph:
        P_fm = P_soft + stop_grad(P* − P_soft)
    so autograd treats P_fm as P_soft for derivatives (gradient “through the soft branch”).
    """
    p_star = hungarian_hard_plan_from_P(P, batch_size)
    return P + (p_star - P).detach()

# ── ODE generation ────────────────────────────────────────────────
@torch.no_grad()
def generate(net, n=2000, steps=200, x0=None):
    """If x0 is given, integrate from that batch (same noise for OT vs SF panels)."""
    if x0 is None:
        x = sample_8gaussians(n).to(DEVICE)
    else:
        x = x0.to(DEVICE).clone()
    dt = 1.0 / steps
    for k in range(steps):
        x = x + net(x, torch.full((1,), k * dt, device=DEVICE)) * dt
    return x.cpu().numpy()


@torch.no_grad()
def integrate_trajectory(net, x0, steps=200):
    """Euler integrate v_theta from t=0..1. x0: [n,2]. Returns pos [steps+1,n,2], t_grid, speeds [steps,n]."""
    device = x0.device
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


def save_trajectory_viz(step, net_ot, net_sf):
    """Same as analyze_sinkformer_2d plot_trajectories_colored (t-color only); own RNG save/restore."""
    if not args.viz_trajectories:
        return
    r_cpu = torch.get_rng_state()
    r_cuda = torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else None
    np_st = np.random.get_state()
    try:
        ts = int(args.trajectory_seed if args.trajectory_seed is not None else int(VIS_SEED) + 90210)
        torch.manual_seed(ts)
        np.random.seed(ts % (2**32 - 1))
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(ts % (2**32 - 1))

        n_paths = args.trajectory_paths
        steps = args.trajectory_steps
        x0 = sample_8gaussians(n_paths).to(DEVICE)
        net_ot.eval()
        net_sf.eval()
        pos_ot, t_grid, _ = integrate_trajectory(net_ot, x0, steps=steps)
        pos_sf, _, _ = integrate_trajectory(net_sf, x0, steps=steps)

        ref = sample_moons(3000, random_state=int(VIS_SEED)).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        for ax, pos, title in [(axes[0], pos_ot, 'OT-CFM'), (axes[1], pos_sf, 'Sinkformer-CFM')]:
            ax.scatter(ref[:, 0], ref[:, 1], s=2, alpha=0.12, c='gray')
            for p in range(n_paths):
                path = pos[:, p, :]
                for k in range(len(t_grid) - 1):
                    c = plt.cm.plasma(float(t_grid[k]))
                    ax.plot(path[k : k + 2, 0], path[k : k + 2, 1], color=c, lw=1.2, alpha=0.85)
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
        out = os.path.join(TRAJ_DIR, f'step{step:07d}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  traj viz -> {out}', flush=True)
    finally:
        torch.set_rng_state(r_cpu)
        if r_cuda is not None:
            torch.cuda.set_rng_state_all(r_cuda)
        np.random.set_state(np_st)
        net_ot.train()
        net_sf.train()
        if not FROZEN:
            pairer.train()


def save_uv_quiver_viz(step, net_ot, net_sf, pairer):
    if not args.viz_uv_quiver:
        return
    r_cpu = torch.get_rng_state()
    r_cuda = torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else None
    np_st = np.random.get_state()
    try:
        torch.manual_seed(int(VIS_SEED))
        np.random.seed(int(VIS_SEED) % (2**32 - 1))
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(int(VIS_SEED) % (2**32 - 1))
        out = os.path.join(QUIVER_DIR, f'step{step:07d}.png')
        save_uv_quiver_figure(
            out, net_ot, net_sf, pairer, DEVICE, B, int(VIS_SEED), HARD_ST, args.per_pair_t,
        )
        print(f'  uv quiver -> {out}', flush=True)
    finally:
        torch.set_rng_state(r_cpu)
        if r_cuda is not None:
            torch.cuda.set_rng_state_all(r_cuda)
        np.random.set_state(np_st)
        net_ot.train()
        net_sf.train()
        if not FROZEN:
            pairer.train()


# ── Visualize ─────────────────────────────────────────────────────
def visualize(step, net_ot, net_sf, pairer, losses_ot, losses_sf, steps_log, frozen_pairer=False, hard_st=True):
    """All stochastic panels use VIS_SEED (torch/np/cuda); training RNG is saved and restored."""
    r_cpu = torch.get_rng_state()
    r_cuda = torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else None
    np_st = np.random.get_state()
    try:
        torch.manual_seed(VIS_SEED)
        np.random.seed(VIS_SEED)
        if DEVICE.type == 'cuda':
            torch.cuda.manual_seed_all(VIS_SEED)
        _visualize_body(
            step, net_ot, net_sf, pairer, losses_ot, losses_sf, steps_log,
            frozen_pairer=frozen_pairer, hard_st=hard_st,
        )
    finally:
        torch.set_rng_state(r_cpu)
        if r_cuda is not None:
            torch.cuda.set_rng_state_all(r_cuda)
        np.random.set_state(np_st)


@torch.no_grad()
def _visualize_body(step, net_ot, net_sf, pairer, losses_ot, losses_sf, steps_log, frozen_pairer=False, hard_st=True):
    mode = ' (frozen pairer + fresh flow)' if frozen_pairer else ''
    sf_mode = '  SF: Hungarian STE (P_fm=P+stop(P*−P))' if hard_st else '  SF: soft P (Sinkhorn on K)'
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f'OT-CFM vs Sinkformer{mode}{sf_mode}  step={step:,}  B={B}  vis_seed={VIS_SEED}',
        fontsize=12,
    )

    ax = axes[0,0]
    if len(steps_log) >= 2:
        if len(steps_log) > 10:
            w = max(1, len(losses_ot) // 30)
            sm = lambda x: np.convolve(x, np.ones(w) / w, 'valid')
            s = steps_log[w - 1 :]
            ax.plot(s, sm(losses_ot), label='OT-CFM', color='steelblue')
            ax.plot(s, sm(losses_sf), label='Sinkformer-CFM', color='darkorange')
        else:
            ax.plot(steps_log, losses_ot, label='OT-CFM', color='steelblue')
            ax.plot(steps_log, losses_sf, label='Sinkformer-CFM', color='darkorange')
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            '(FM loss will appear after a few log_every steps)',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=10,
        )
    ax.set_title('FM Loss'); ax.set_yscale('log')

    ref = sample_moons(3000, random_state=VIS_SEED).numpy()
    x0_gen = sample_8gaussians(2000).to(DEVICE)
    gen_ot = generate(net_ot, x0=x0_gen)
    gen_sf = generate(net_sf, x0=x0_gen)
    # W2 between *empirical* measures: sqrt( OT loss ) with M = ||x-y||^2  => 2-Wasserstein W2.
    # Do NOT use ref[:n], gen[:n]: sklearn moons order can cluster early indices on one lobe,
    # so the OT subset would not match what you see in the full scatter → misleading numbers.
    n_cap = 2000 if not args.vis_light else 800
    n_emd = int(min(n_cap, len(gen_ot), len(gen_sf), len(ref)))
    rng_w2 = np.random.default_rng(int(VIS_SEED) + 513_297)
    ir = rng_w2.choice(len(ref), size=n_emd, replace=False)
    ig = rng_w2.choice(len(gen_ot), size=n_emd, replace=False)
    ref_w2 = ref[ir]
    w = np.ones(n_emd) / n_emd
    for ax, label, c, gen in [
        (axes[0, 1], 'OT-CFM', 'steelblue', gen_ot),
        (axes[0, 2], 'Sinkformer', 'darkorange', gen_sf),
    ]:
        gen_w2 = gen[ig]
        M_emd = pot.dist(gen_w2, ref_w2, metric='sqeuclidean')
        w2_sq = float(pot.emd2(w, w, M_emd))
        emd_val = float(np.sqrt(max(0.0, w2_sq)))
        w2_txt = f'W2≈{emd_val:.4f} (n={n_emd}, rand subset)'
        ax.scatter(ref[:, 0], ref[:, 1], s=3, alpha=0.2, c='gray')
        ax.scatter(gen[:, 0], gen[:, 1], s=3, alpha=0.4, c=c, label=label)
        ax.set_title(label)
        ax.legend(markerscale=3)
        ax.text(
            0.02,
            0.02,
            w2_txt,
            transform=ax.transAxes,
            fontsize=9,
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7),
        )

    x0v = sample_8gaussians(B).to(DEVICE)
    x1v = sample_moons(B, random_state=VIS_SEED + 10_007).to(DEVICE)
    P = pairer(x0v, x1v).cpu().numpy()
    sharp = (P.max(1) > 0.5/B).mean()

    ax = axes[1,0]
    # Heatmap = B·P (doubly-stochastic scale: each row/col should sum to 1)
    PB = P * B
    row_sums = PB.sum(axis=1)
    col_sums = PB.sum(axis=0)
    err_row = float(np.max(np.abs(row_sums - 1.0)))
    err_col = float(np.max(np.abs(col_sums - 1.0)))
    err_marg = max(err_row, err_col)
    sum_pb = float(PB.sum())

    M_cost = torch.cdist(x0v, x1v).pow(2).detach().cpu().numpy()
    r_ot, c_ot = linear_sum_assignment(M_cost)
    im = ax.imshow(
        PB, cmap='Blues', vmin=0, vmax=1, aspect='equal',
        extent=[0, B, B, 0], origin='upper',
    )
    ax.scatter(
        c_ot + 0.5, r_ot + 0.5, marker='*', s=220, c='#ffd700',
        edgecolors='k', linewidths=0.6, zorder=10, label='OT optimum',
    )
    tau = pairer.log_tau.exp().item()
    ax.set_title(
        f'B·P  τ={tau:.1f}  sharp={sharp*100:.0f}%  '
        f'max|Σrow−1|={err_row:.2e}  max|Σcol−1|={err_col:.2e}  Σ(BP)={sum_pb:.4f}\n'
        f'(★ = coord OT)  dashed line at 1 on marginals',
        fontsize=9,
    )
    ax.set_xlabel('x1 index j'); ax.set_ylabel('x0 index i')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes('top', size='12%', pad=0.07)
    ax_left = divider.append_axes('left', size='11%', pad=0.12)
    cax = divider.append_axes('right', size='3.5%', pad=0.02)
    fig.colorbar(im, cax=cax)

    off = max(0.015, min(0.15, 5.0 * err_marg + 1e-9))
    jj = np.arange(B)
    ax_top.bar(jj + 0.5, col_sums, width=0.85, color='steelblue', edgecolor='k', linewidth=0.25)
    ax_top.axhline(1.0, color='coral', linestyle='--', linewidth=1.0)
    ax_top.set_ylim(1.0 - off, 1.0 + off)
    ax_top.set_xlim(0, B)
    ax_top.set_xticks([])
    ax_top.set_ylabel('col\nΣ', fontsize=7)
    ax_top.tick_params(axis='y', labelsize=6)

    ii = np.arange(B)
    ax_left.barh(ii + 0.5, row_sums, height=0.85, color='steelblue', edgecolor='k', linewidth=0.25)
    ax_left.axvline(1.0, color='coral', linestyle='--', linewidth=1.0)
    ax_left.set_xlim(1.0 - off, 1.0 + off)
    ax_left.set_ylim(B, 0)
    ax_left.set_yticks([])
    ax_left.set_xlabel('row\nΣ', fontsize=7)
    ax_left.tick_params(axis='x', labelsize=6)

    x0n = x0v.cpu().numpy(); x1n = x1v.cpu().numpy()
    _, x1_ot = ot_pairs(x0v, x1v)
    x1_ot = x1_ot.cpu().numpy()
    i_sf, j_sf = linear_sum_assignment(-P)

    ax = axes[1,1]
    ax.scatter(x0n[:,0], x0n[:,1], c='royalblue', s=40, zorder=3)
    ax.scatter(x1_ot[:,0], x1_ot[:,1], c='crimson', s=40, zorder=3)
    for i in range(B):
        ax.plot([x0n[i,0], x1_ot[i,0]], [x0n[i,1], x1_ot[i,1]], 'steelblue', alpha=0.5)
    ax.set_title('OT pairs')

    ax = axes[1,2]
    ax.scatter(x0n[:,0], x0n[:,1], c='royalblue', s=40, zorder=3)
    ax.scatter(x1n[:,0], x1n[:,1], c='crimson', s=40, zorder=3)
    for ii,jj in zip(i_sf, j_sf):
        ax.plot([x0n[ii,0], x1n[jj,0]], [x0n[ii,1], x1n[jj,1]], 'darkorange', alpha=0.5)
    if hard_st:
        sf_pair_title = 'SF 1-1 (Hungarian on P, same as training STE)'
    else:
        sf_pair_title = 'SF 1-1 lines (Hungarian max-weight on soft P; train uses full P)'
    ax.set_title(sf_pair_title)

    plt.tight_layout()
    path = os.path.join(PANEL_DIR, f'step{step:07d}.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    del fig, axes
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print(f'  viz -> {path}')

# ── Init ──────────────────────────────────────────────────────────
net_ot = FlowNet(args.hidden).to(DEVICE)
net_sf = FlowNet(args.hidden).to(DEVICE)
pairer = SinkPairer(args.d_sink, n_sink_iters=args.sink_iters).to(DEVICE)

FROZEN = args.frozen_pairer is not None
if FROZEN:
    ckpt = args.frozen_pairer
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f'--frozen-pairer not found: {ckpt}')
    pairer.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    for p in pairer.parameters():
        p.requires_grad = False
    pairer.eval()
    print(f'Frozen pairer loaded from {ckpt} (selection only; FlowNets train from scratch)', flush=True)
    opt_sf = torch.optim.Adam(net_sf.parameters(), lr=args.lr)
else:
    opt_sf = torch.optim.Adam(list(net_sf.parameters()) + list(pairer.parameters()), lr=args.lr)

opt_ot = torch.optim.Adam(net_ot.parameters(), lr=args.lr)

print(f'Device: {DEVICE}', flush=True)
print(f'FlowNet: {sum(p.numel() for p in net_ot.parameters()):,} params', flush=True)
print(f'Pairer:  {sum(p.numel() for p in pairer.parameters()):,} params  (frozen={FROZEN})', flush=True)
# Default: Hungarian hard P* with STE. --soft-pair disables (train on full soft P).
HARD_ST = not args.soft_pair
print(
    f'Sinkformer: log K=S/tau (bipartite QK^T), Sinkhorn iters={args.sink_iters}, '
    f'train={"Hungarian STE  P_fm=P+stop(P*−P)" if HARD_ST else "soft P (full grad, no STE)"}',
    flush=True,
)
print(f't sampling: {"per-pair U(0,1)" if args.per_pair_t else "shared scalar per step"}', flush=True)
if args.vis_light:
    print('viz: --vis-light (W2 uses n<=800 random ref/gen subsets)', flush=True)
if args.viz_trajectories:
    print(
        f'viz-trajectories: paths={args.trajectory_paths} steps={args.trajectory_steps} '
        f'-> {TRAJ_DIR}/step{{N}}.png',
        flush=True,
    )
if args.viz_uv_quiver:
    print(f'viz-uv-quiver: -> {QUIVER_DIR}/step{{N}}.png', flush=True)
if args.save_ckpt_every > 0:
    print(f'save-ckpt-every: {args.save_ckpt_every} -> {CKPT_ROOT}/step{{N}}/', flush=True)
print(f'Panels -> {PANEL_DIR}/  |  vis_seed={VIS_SEED}', flush=True)
print(f'Steps: {args.steps:,}  B={B}', flush=True)
print('='*50, flush=True)

# ── Train ─────────────────────────────────────────────────────────
losses_ot, losses_sf, steps_log = [], [], []
t0 = time.time()

for step in range(args.steps + 1):
    x0 = sample_8gaussians(B).to(DEVICE)
    x1 = sample_moons(B).to(DEVICE)
    if args.per_pair_t:
        t = torch.rand(B, device=DEVICE)
    else:
        t = torch.rand(1, device=DEVICE).expand(B)
    tv = t.view(B, 1)

    # OT-CFM
    x0p, x1p = ot_pairs(x0, x1)
    xt = (1 - tv) * x0p + tv * x1p
    L_ot = ((net_ot(xt, t) - (x1p - x0p))**2).mean()
    opt_ot.zero_grad(); L_ot.backward(); opt_ot.step()

    # Sinkformer-CFM: P from pairer (trainable or frozen selection)
    if FROZEN:
        with torch.no_grad():
            P = pairer(x0, x1)
    else:
        P = pairer(x0, x1)
    if HARD_ST:
        P_fm = ste_hard_P(P, B)
    else:
        P_fm = P
    x0_soft = torch.mm(P_fm.t() * B, x0)
    xt      = (1 - tv) * x0_soft + tv * x1
    L_sf    = ((net_sf(xt, t) - (x1 - x0_soft))**2).mean()
    opt_sf.zero_grad(); L_sf.backward(); opt_sf.step()

    if step % args.log_every == 0:
        losses_ot.append(L_ot.item())
        losses_sf.append(L_sf.item())
        steps_log.append(step)

    do_log = (step % 10_000 == 0) or (args.quick and step % 1000 == 0)
    if args.fig_save_every > 0 and step % args.fig_save_every == 0:
        do_log = True
    if do_log:
        tau = pairer.log_tau.exp().item()
        with torch.no_grad():
            Pv = pairer(x0, x1)
            sharp = (Pv.max(1).values > 0.5/B).float().mean().item()
        print(f'[{step:7d}|{(time.time()-t0)/60:5.1f}min] '
              f'OT={L_ot.item():.4f}  SF={L_sf.item():.4f}  '
              f'tau={tau:.1f}  sharp={sharp*100:.0f}%', flush=True)

    do_viz = (step % args.vis_every == 0) or (
        args.fig_save_every > 0 and step % args.fig_save_every == 0
    )
    if do_viz:
        visualize(step, net_ot, net_sf, pairer, losses_ot, losses_sf, steps_log,
                  frozen_pairer=FROZEN, hard_st=HARD_ST)
        if args.viz_trajectories:
            save_trajectory_viz(step, net_ot, net_sf)
        if args.viz_uv_quiver:
            save_uv_quiver_viz(step, net_ot, net_sf, pairer)

    if args.save_ckpt_every > 0 and step > 0 and step % args.save_ckpt_every == 0:
        ck = os.path.join(CKPT_ROOT, f'step{step:07d}')
        os.makedirs(ck, exist_ok=True)
        torch.save(net_ot.state_dict(), os.path.join(ck, 'net_ot.pt'))
        torch.save(net_sf.state_dict(), os.path.join(ck, 'net_sf.pt'))
        torch.save(pairer.state_dict(), os.path.join(ck, 'pairer.pt'))
        np.savez(
            os.path.join(ck, 'losses.npz'),
            steps=np.asarray(steps_log, dtype=np.int64),
            ot=np.asarray(losses_ot, dtype=np.float64),
            sf=np.asarray(losses_sf, dtype=np.float64),
        )
        print(f'  checkpoint -> {ck}', flush=True)

# ── Summary ───────────────────────────────────────────────────────
n = min(50, len(losses_ot)//5)
m_ot = np.mean(losses_ot[-n:])
m_sf = np.mean(losses_sf[-n:])
print(f'\n{"="*50}')
print(f'Final FM loss (last {n} pts):')
print(f'  OT-CFM:     {m_ot:.5f}')
print(f'  Sinkformer: {m_sf:.5f}')
if m_sf < m_ot:
    print(f'  Sinkformer wins by {(m_ot-m_sf)/m_ot*100:.1f}%')
else:
    print(f'  OT-CFM wins by {(m_sf-m_ot)/m_ot*100:.1f}%')

np.savez(f'{args.outdir}/losses.npz', steps=steps_log, ot=losses_ot, sf=losses_sf)
torch.save(net_ot.state_dict(), f'{args.outdir}/net_ot.pt')
torch.save(net_sf.state_dict(), f'{args.outdir}/net_sf.pt')
torch.save(pairer.state_dict(), f'{args.outdir}/pairer.pt')
print('Models saved.')
