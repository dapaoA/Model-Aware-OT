"""
sinkformer_cfm_cifar.py
=======================
Compare Sinkformer-CFM vs OT-CFM on CIFAR-10.

KEY INSIGHT (Round 3):
  To *beat* OT-CFM we need model-aware pairings — pairings that minimise
  FM loss directly, not pixel-space OT cost.

  Round 1 failure: CNNEncoder breaks x0·x1 inner-product → OT_pred ≈ OT_rand
  Round 2 failure: LinearEncoder + score normalisation — task killed before
                   Sinkformer-CFM ran
  Round 3 fix: SOFT FM training — differentiable x0_soft = P^T @ x0
               Backprop flows FM_loss → UNet → x0_soft → P → encoder/pairer
               Pairer learns to minimise FM loss, not just pixel OT

Architecture:
  LinearEncoder : [B, 3, 32, 32] → [B, D=128]  (preserves x0·x1 inner product)
  TimePairer    : (x0_feat [B,D], x1_feat [B,D], t scalar)
                  → prepend t_token → self-attention on x0/x1 seqs
                  → Q @ K^T → score normalisation → log_sinkhorn → P [B, B]
  x0_soft[j]   = B * Σ_i P[i,j] * x0[i]          (differentiable pairing)
  xt            = (1-t)*x0_soft + t*x1             (soft interpolant)
  L_FM          = ||UNet(xt, t) - (x1 - x0_soft)||²  (trains everything)
  L_OT          = ΣΣ B*P[i,j]*M[i,j] / B          (OT regulariser, λ=0.1)
  L             = L_FM + λ * L_OT

Training:
  Single optimizer for encoder + pairer + UNet (joint model-aware training)
  OT-CFM baseline: pot.emd exact pairing, same FM loss

Run:
  python experiments/sinkformer_cfm_cifar.py [--method {both,otcfm,sinkformer}]
                                              [--minutes FLOAT]
                                              [--smoke]

Outputs in experiments/results/sinkformer_cfm_cifar/
"""

import os, sys, io, time, copy, warnings, argparse
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ot as pot
from scipy.optimize import linear_sum_assignment

# ── project UNet ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchcfm.models.unet.unet import UNetModelWrapper

OUTDIR = 'experiments/results/sinkformer_cfm_cifar'
os.makedirs(OUTDIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BATCH        = 128
NUM_CHANNEL  = 64        # UNet base channels (~8M params)
LR           = 2e-4
WARMUP       = 2000
EMA_DECAY    = 0.9999
GRAD_CLIP    = 1.0
MINUTES_OT   = 5         # shorter OT-CFM baseline (just need reference)
MINUTES_SF   = 12        # Sinkformer-CFM budget
D_SF         = 128       # Sinkformer feature dimension
N_HEADS_SF   = 4
N_LAYERS_SF  = 2
N_SINK       = 50        # Sinkhorn iters (50 handles score explosion reliably)
LAMBDA_OT    = 0.5       # OT regulariser weight in combined loss (stronger signal for pairer)
LOG_EVERY    = 100
VIS_EVERY    = 2000
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# Part 1 – Sinkhorn (log-domain, marginals = 1/N)
# ═════════════════════════════════════════════════════════════════════════════

def log_sinkhorn(log_alpha: torch.Tensor, n_iters: int = N_SINK) -> torch.Tensor:
    """Log-domain Sinkhorn → doubly stochastic, marginals = 1/N."""
    N = log_alpha.shape[-1]
    log_N = torch.tensor(float(np.log(N)), dtype=log_alpha.dtype,
                         device=log_alpha.device)
    log_alpha = log_alpha - log_alpha.amax(dim=(-2, -1), keepdim=True)
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True) - log_N
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True) - log_N
    return torch.exp(log_alpha)


# ═════════════════════════════════════════════════════════════════════════════
# Part 2 – Encoder + Time-conditioned Pairer
# ═════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbed(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(1) if t.dim() == 0 else t
        half = self.dim // 2
        freq = torch.exp(-torch.arange(half, device=t.device) *
                         (np.log(10000) / (half - 1)))
        emb = t.float() * freq[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb.squeeze(0)


class LinearEncoder(nn.Module):
    """
    Linear projection: [B, 3, 32, 32] → [B, D_SF]

    WHY LINEAR: OT pairing minimises Σ ||x0[i]-x1[j]||² ≡ maximises x0[i]·x1[j].
    A linear W preserves this: (Wx0)·(Wx1) ∝ x0^T W^T W x1 (Mahalanobis metric).
    CNNs break this nonlinearity, destroying the OT signal.

    SHARED encoder for x0 and x1 so features live in the same space.
    W can learn a metric that makes OT pairing easier for the UNet.
    """
    def __init__(self, d_out: int = D_SF):
        super().__init__()
        d_in = 3 * 32 * 32
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_out, bias=False)
        nn.init.normal_(self.proj.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.view(x.shape[0], -1)
        return self.proj(self.norm(h))


class TimePairer(nn.Module):
    """
    Time-conditioned OT Pairer.

    Given a batch of x0 and x1 features and a time t, outputs P [B,B]:
    doubly stochastic transport plan (marginals = 1/B).

    Architecture:
      t_token = SinEmbed(t) → MLP → [1, 1, D]
      x0_seq  = [t_token; x0_feat]  [1, B+1, D]
      x1_seq  = [t_token; x1_feat]  [1, B+1, D]
      H0      = SA(x0_seq)          [1, B+1, D]  (cross-sample context)
      H1      = SA(x1_seq)          [1, B+1, D]
      Q       = proj_q(H0[:,1:])    [1, B, D]
      K       = proj_k(H1[:,1:])    [1, B, D]
      S       = Q @ K^T / √D        [1, B, B]
      S       = S / std(S) * τ      (prevent score explosion)
      P       = log_sinkhorn(S)     [B, B]
    """
    def __init__(self, d_model: int = D_SF, n_heads: int = N_HEADS_SF,
                 n_layers: int = N_LAYERS_SF, n_sinkhorn: int = N_SINK):
        super().__init__()
        self.d_model    = d_model
        self.n_sinkhorn = n_sinkhorn
        # Learnable log-temperature: controls sharpness of transport plan P.
        # tau=1 → near-uniform P (degenerate: x0_soft ≈ mean(x0) ≈ 0)
        # tau=5 → sharp P (near-permutation → proper FM task from step 0)
        # Start at tau=5 and let gradient fine-tune.
        self.log_tau = nn.Parameter(torch.full((1,), np.log(5.0)))

        # Time embedding
        self.t_sin = SinusoidalEmbed(d_model)
        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Self-attention for context aggregation within each set
        def make_sa():
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=0.0, batch_first=True, norm_first=True,
                ),
                num_layers=n_layers,
            )
        self.sa0 = make_sa()
        self.sa1 = make_sa()

        # Cross-set projection
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x0_feat: torch.Tensor, x1_feat: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        x0_feat : [B, D]   x1_feat : [B, D]   t : scalar ∈ [0,1]
        Returns P : [B, B]  transport plan, marginals = 1/B
        """
        t_token = self.t_mlp(self.t_sin(t)).unsqueeze(0).unsqueeze(0)  # [1,1,D]
        x0_seq  = torch.cat([t_token, x0_feat.unsqueeze(0)], dim=1)    # [1,B+1,D]
        x1_seq  = torch.cat([t_token, x1_feat.unsqueeze(0)], dim=1)    # [1,B+1,D]

        H0 = self.sa0(x0_seq)   # [1, B+1, D]
        H1 = self.sa1(x1_seq)   # [1, B+1, D]

        Q = self.proj_q(H0[:, 1:])    # [1, B, D]  (skip t_token output)
        K = self.proj_k(H1[:, 1:])    # [1, B, D]

        S = torch.bmm(Q, K.transpose(-2, -1)) * (self.d_model ** -0.5)  # [1, B, B]

        # Score normalisation: keep std ≈ τ to prevent score explosion
        tau   = self.log_tau.exp().clamp(min=0.1)
        S_std = S.std(dim=(-2, -1), keepdim=True).clamp(min=0.1)
        S     = (S / S_std) * tau                                         # [1, B, B]

        P = log_sinkhorn(S, self.n_sinkhorn)   # [1, B, B]
        return P.squeeze(0)                    # [B, B]

    def score_stats(self, x0_feat, x1_feat, t):
        with torch.no_grad():
            t_token = self.t_mlp(self.t_sin(t)).unsqueeze(0).unsqueeze(0)
            x0_seq  = torch.cat([t_token, x0_feat.unsqueeze(0)], dim=1)
            x1_seq  = torch.cat([t_token, x1_feat.unsqueeze(0)], dim=1)
            H0 = self.sa0(x0_seq)
            H1 = self.sa1(x1_seq)
            Q  = self.proj_q(H0[:, 1:])
            K  = self.proj_k(H1[:, 1:])
            S  = torch.bmm(Q, K.transpose(-2, -1)) * (self.d_model ** -0.5)
            return S.std().item(), S.max().item(), self.log_tau.exp().item()

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# Part 3 – Utilities
# ═════════════════════════════════════════════════════════════════════════════

def make_unet():
    return UNetModelWrapper(
        dim=(3, 32, 32), num_res_blocks=2, num_channels=NUM_CHANNEL,
        channel_mult=[1, 2, 2, 2], num_heads=4, num_head_channels=64,
        attention_resolutions='16', dropout=0.1,
    ).to(DEVICE)


def ema_update(source, target, decay=EMA_DECAY):
    for s, t_ in zip(source.parameters(), target.parameters()):
        t_.data.mul_(decay).add_(s.data, alpha=1 - decay)


def warmup_schedule(step):
    return min(step + 1, WARMUP) / WARMUP


def pixel_dist_matrix(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Normalised squared L2 distance matrix, [B, B]."""
    B = x0.shape[0]
    x0f = x0.view(B, -1).float()
    x1f = x1.view(B, -1).float()
    return torch.cdist(x0f, x1f) ** 2 / x0f.shape[1]


@torch.no_grad()
def exact_ot_pairs(x0: torch.Tensor, x1: torch.Tensor):
    """Exact EMD pairing. Returns (i_idx, j_idx)."""
    M = pixel_dist_matrix(x0, x1).cpu().numpy().astype(np.float64)
    a = np.ones(BATCH) / BATCH
    b = np.ones(BATCH) / BATCH
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        P = pot.emd(a, b, M)
    return linear_sum_assignment(-P)


def hungarian_pairs(P: torch.Tensor):
    """Hard assignment from soft plan P [B, B]."""
    return linear_sum_assignment(-(BATCH * P).detach().cpu().numpy())


def ot_cost(P: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Expected transport cost under plan P with marginals 1/B."""
    return (BATCH * P * M).sum() / BATCH


def make_dataloader():
    ds = datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True,
    )


def infinite(loader):
    while True:
        for x, _ in loader:
            yield x.to(DEVICE)


def check_pairing_quality(P: torch.Tensor) -> dict:
    """Verify doubly stochastic and 1-1 pairing quality."""
    B = P.shape[0]
    target = 1.0 / B
    row_err = (P.sum(-1) - target).abs().max().item()
    col_err = (P.sum(-2) - target).abs().max().item()
    P_clamp = P.clamp(1e-12)
    row_entropy   = -(P_clamp * P_clamp.log()).sum(-1).mean().item()
    max_vals = P.max(-1).values
    sharp_frac = (max_vals > 0.5 / B).float().mean().item()
    return {'row_err': row_err, 'col_err': col_err,
            'row_entropy': row_entropy, 'sharp_frac': sharp_frac}


# ═════════════════════════════════════════════════════════════════════════════
# Part 4 – Training
# ═════════════════════════════════════════════════════════════════════════════

def train_otcfm(minutes: float = MINUTES_OT, tag: str = 'otcfm'):
    """OT-CFM baseline (exact EMD pairing each step)."""
    print(f'\n{"="*60}')
    print(f'Training OT-CFM  ({minutes:.0f} min budget)')
    print(f'{"="*60}')

    loader    = make_dataloader()
    data_iter = infinite(loader)

    unet     = make_unet()
    ema_unet = copy.deepcopy(unet)
    opt      = torch.optim.Adam(unet.parameters(), lr=LR)
    sched    = torch.optim.lr_scheduler.LambdaLR(opt, warmup_schedule)
    print(f'UNet params: {sum(p.numel() for p in unet.parameters()):,}')

    losses, steps_log = [], []
    ot_costs_random, ot_costs_exact = [], []

    t_start = time.time()
    step    = 0

    while (time.time() - t_start) / 60 < minutes:
        unet.train()
        x1 = next(data_iter)
        x0 = torch.randn_like(x1)

        i_idx, j_idx = exact_ot_pairs(x0, x1)
        x0_p = x0[i_idx]
        x1_p = x1[j_idx]

        t_scalar = torch.rand(1, device=DEVICE)
        t        = t_scalar.expand(BATCH)
        t_v      = t.view(BATCH, 1, 1, 1)
        xt       = (1 - t_v) * x0_p + t_v * x1_p
        ut       = x1_p - x0_p

        vt   = unet(t, xt)
        loss = ((vt - ut) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        ema_update(unet, ema_unet)

        if step % LOG_EVERY == 0:
            elapsed = (time.time() - t_start) / 60
            with torch.no_grad():
                M     = pixel_dist_matrix(x0, x1)
                i_ex, j_ex = exact_ot_pairs(x0, x1)
                c_exact = M[i_ex, j_ex].mean().item()
                c_rand  = M.mean().item()
            losses.append(loss.item())
            steps_log.append(step)
            ot_costs_exact.append(c_exact)
            ot_costs_random.append(c_rand)
            print(f'  [{tag}] step={step:6d} | t={elapsed:.1f}min '
                  f'| FM_loss={loss.item():.5f}'
                  f'| OT_cost={c_exact:.4f} (exact) {c_rand:.4f} (rand)')

        if step % VIS_EVERY == 0 and step > 0:
            _save_checkpoint(unet, ema_unet, opt, sched, step, tag)

        step += 1

    elapsed = (time.time() - t_start) / 60
    print(f'  [{tag}] Done: {step} steps in {elapsed:.1f} min')
    _save_checkpoint(unet, ema_unet, opt, sched, step, tag)

    res = {
        'steps': steps_log, 'fm_loss': losses,
        'ot_costs_exact': ot_costs_exact,
        'ot_costs_random': ot_costs_random,
        'unet': unet, 'ema_unet': ema_unet,
        'tag': tag,
    }
    # Save for later reuse
    np.savez(f'{OUTDIR}/otcfm_results.npz',
             steps=np.array(steps_log), fm_loss=np.array(losses),
             ot_costs_exact=np.array(ot_costs_exact),
             ot_costs_random=np.array(ot_costs_random))
    print(f'  OT-CFM results saved to {OUTDIR}/otcfm_results.npz')
    return res


def load_otcfm_results():
    """Load cached OT-CFM results if available (avoids rerunning)."""
    path = f'{OUTDIR}/otcfm_results.npz'
    if not os.path.exists(path):
        return None
    d = np.load(path)
    print(f'  Loaded OT-CFM results from {path} '
          f'({len(d["steps"])} log points, '
          f'final FM loss = {d["fm_loss"][-5:].mean():.5f})')
    return {
        'steps': d['steps'].tolist(),
        'fm_loss': d['fm_loss'].tolist(),
        'ot_costs_exact': d['ot_costs_exact'].tolist(),
        'ot_costs_random': d['ot_costs_random'].tolist(),
        'unet': None, 'ema_unet': None,
        'tag': 'otcfm',
    }


def train_sinkformer_cfm(minutes: float = MINUTES_SF, tag: str = 'sink_cfm'):
    """
    Sinkformer-CFM – Round 5: Decoupled training with soft FM.

    DIAGNOSIS of Rounds 3 & 4:
      Joint optimizer: FM gradient flows through P → encoder, pushing P toward
      uniform (x0_soft → 0 = trivially easy FM task). FM loss drops to 0.01
      but the model learns the wrong thing (predict x1 from t*x1).

    ROUND 5 FIX:
      DECOUPLE: FM gradient stays in UNet only (P.detach() for soft x0_soft).
      Pairer trained on L_OT alone → learns OT-quality matching.
      UNet trained on L_FM with soft pairs from detached P → smooth training.

      opt_sf   trains: encoder + pairer  (signal: L_OT)
      opt_unet trains: UNet              (signal: L_FM with soft detached pairs)

    Why soft (not hard) pairs for UNet?
      Soft x0_soft = Σ_i (B*P[i,j]) * x0[i]: weighted avg of B noise samples.
      When P is 20% sharp (tau=5 init): x0_soft is between mean(x0)=0 and x0[k].
      UNet trains on intermediate difficulty targets → smoother than hard pairs.
      As pairer converges to sharp OT permutation: x0_soft → x0[k] → proper FM.

    Why tau=5 init?
      tau=1 (Round 2): P near-uniform from start → pairer gradient is tiny
      tau=5 (Round 5): P 20% sharp from start → pairer gets meaningful OT gradient
    """
    print(f'\n{"="*60}')
    print(f'Training Sinkformer-CFM Round 5 (decoupled)  ({minutes:.0f} min budget)')
    print(f'{"="*60}')

    loader    = make_dataloader()
    data_iter = infinite(loader)

    encoder  = LinearEncoder(d_out=D_SF).to(DEVICE)
    pairer   = TimePairer(d_model=D_SF, n_heads=N_HEADS_SF,
                          n_layers=N_LAYERS_SF, n_sinkhorn=N_SINK).to(DEVICE)
    unet     = make_unet()
    ema_unet = copy.deepcopy(unet)

    # SEPARATE optimizers: no gradient conflict between OT and FM
    sf_params   = list(encoder.parameters()) + list(pairer.parameters())
    opt_sf      = torch.optim.Adam(sf_params,           lr=LR)
    opt_unet    = torch.optim.Adam(unet.parameters(),   lr=LR)
    sched_sf    = torch.optim.lr_scheduler.LambdaLR(opt_sf,   warmup_schedule)
    sched_unet  = torch.optim.lr_scheduler.LambdaLR(opt_unet, warmup_schedule)

    n_sf = sum(p.numel() for p in sf_params)
    n_un = sum(p.numel() for p in unet.parameters())
    print(f'UNet params: {n_un:,}  |  Pairer+Encoder params: {n_sf:,}')
    print(f'  tau_init={pairer.log_tau.exp().item():.1f}  '
          f'(sharp P from start)')

    losses, steps_log = [], []
    ot_costs_pred, ot_costs_exact, ot_costs_random = [], [], []
    pair_quality_log = []
    plan_samples     = []

    t_start = time.time()
    step    = 0

    while (time.time() - t_start) / 60 < minutes:
        encoder.train()
        pairer.train()
        unet.train()

        x1 = next(data_iter)
        x0 = torch.randn_like(x1)

        t_scalar = torch.rand(1, device=DEVICE)
        t        = t_scalar.expand(BATCH)
        t_v      = t.view(BATCH, 1, 1, 1)

        # ── 1. Encode ─────────────────────────────────────────────────────────
        x0_feat = encoder(x0)   # [B, D_SF]
        x1_feat = encoder(x1)   # [B, D_SF]

        # ── 2. Transport plan ─────────────────────────────────────────────────
        P = pairer(x0_feat, x1_feat, t_scalar.squeeze())   # [B, B]

        # ── 3. Train pairer on L_OT (pixel-space transport cost) ─────────────
        M      = pixel_dist_matrix(x0, x1)   # [B, B]  (no grad needed)
        L_OT   = ot_cost(P, M.detach())      # scalar

        opt_sf.zero_grad()
        L_OT.backward()
        torch.nn.utils.clip_grad_norm_(sf_params, GRAD_CLIP)
        opt_sf.step()
        sched_sf.step()

        # ── 4. Soft x0 partner (DETACHED P → FM grad stays in UNet) ──────────
        #   P_det has no grad → FM loss doesn't push P toward uniform
        P_det    = P.detach()
        P_scaled = P_det * BATCH                      # rows/cols sum to 1
        x0_flat  = x0.view(BATCH, -1)
        x0_soft  = torch.mm(P_scaled.t(), x0_flat).view(BATCH, 3, 32, 32)

        # ── 5. Train UNet on L_FM with soft pairings ──────────────────────────
        xt   = (1 - t_v) * x0_soft + t_v * x1
        ut   = x1 - x0_soft
        vt   = unet(t, xt)
        L_FM = ((vt - ut) ** 2).mean()

        opt_unet.zero_grad()
        L_FM.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), GRAD_CLIP)
        opt_unet.step()
        sched_unet.step()
        ema_update(unet, ema_unet)

        # ── 6. Logging ────────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            elapsed = (time.time() - t_start) / 60
            with torch.no_grad():
                i_idx, j_idx = hungarian_pairs(P)
                c_pred = M[i_idx, j_idx].mean().item()
                c_rand = M.mean().item()
                pq     = check_pairing_quality(P)
                s_std, s_max, tau_val = pairer.score_stats(
                    x0_feat.detach(), x1_feat.detach(), t_scalar.squeeze().detach())
            losses.append(L_FM.item())
            steps_log.append(step)
            ot_costs_pred.append(c_pred)
            ot_costs_random.append(c_rand)
            pair_quality_log.append(pq)

            if step % 500 == 0:
                with torch.no_grad():
                    i_ex, j_ex = exact_ot_pairs(x0, x1)
                    c_exact = M[i_ex, j_ex].mean().item()
                ot_costs_exact.append(c_exact)
            else:
                ot_costs_exact.append(ot_costs_exact[-1] if ot_costs_exact else c_rand)

            print(f'  [{tag}] step={step:6d} | t={elapsed:.1f}min '
                  f'| FM={L_FM.item():.5f} OT={L_OT.item():.4f} '
                  f'| OT_hard={c_pred:.4f} OT_rand={c_rand:.4f} '
                  f'| row_err={pq["row_err"]:.1e} ent={pq["row_entropy"]:.3f} '
                  f'sharpness={pq["sharp_frac"]:.2f} '
                  f'| S_std={s_std:.2f} tau={tau_val:.2f}')

            if len(plan_samples) < 8:
                plan_samples.append(P.detach().cpu().numpy())

        if step % VIS_EVERY == 0 and step > 0:
            _save_checkpoint(unet, ema_unet, opt_unet, sched_unet, step, tag)
            _visualise_plan(plan_samples[-1] if plan_samples else None,
                            step, tag, pair_quality_log)

        step += 1

    elapsed = (time.time() - t_start) / 60
    print(f'  [{tag}] Done: {step} steps in {elapsed:.1f} min')
    _save_checkpoint(unet, ema_unet, opt_unet, sched_unet, step, tag)

    res = {
        'steps': steps_log, 'fm_loss': losses,
        'ot_costs_pred':   ot_costs_pred,
        'ot_costs_exact':  ot_costs_exact,
        'ot_costs_random': ot_costs_random,
        'pair_quality':    pair_quality_log,
        'plan_samples':    plan_samples,
        'unet': unet, 'ema_unet': ema_unet,
        'encoder': encoder, 'pairer': pairer,
        'tag': tag,
    }
    np.savez(f'{OUTDIR}/sink_results.npz',
             steps=np.array(steps_log), fm_loss=np.array(losses),
             ot_costs_pred=np.array(ot_costs_pred),
             ot_costs_exact=np.array(ot_costs_exact),
             ot_costs_random=np.array(ot_costs_random))
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Part 5 – Visualisation
# ═════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(unet, ema_unet, opt, sched, step, tag):
    torch.save({'unet': unet.state_dict(), 'ema': ema_unet.state_dict(),
                'opt': opt.state_dict(), 'sched': sched.state_dict(),
                'step': step},
               f'{OUTDIR}/{tag}_step{step}.pt')


def _visualise_plan(P_np, step, tag, pq_log):
    if P_np is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    B   = P_np.shape[0]
    sub = min(32, B)
    im  = axes[0].imshow(P_np[:sub, :sub] * B, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title(f'P×B (top {sub}×{sub}) – step {step}')
    axes[0].set_xlabel('x1 index')
    axes[0].set_ylabel('x0 index')
    plt.colorbar(im, ax=axes[0])

    row_sums = P_np.sum(axis=1)
    axes[1].hist(row_sums, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    axes[1].axvline(1.0 / B, color='red', linestyle='--', label=f'target=1/{B}')
    axes[1].set_title('Row sum distribution')
    axes[1].legend()

    if pq_log:
        entropies = [pq['row_entropy'] for pq in pq_log]
        axes[2].plot(entropies, color='darkorange')
        axes[2].set_title('Row entropy (lower = sharper pairing)')
        axes[2].set_xlabel('log step (×100)')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/{tag}_plan_step{step}.png', dpi=120)
    plt.close()


def plot_comparison(otcfm_res, sink_res):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sinkformer-CFM (Round 3: Soft FM) vs OT-CFM on CIFAR-10',
                 fontsize=13)

    # 1. FM loss
    ax = axes[0, 0]
    ax.plot(otcfm_res['steps'], otcfm_res['fm_loss'],
            label='OT-CFM', color='steelblue', alpha=0.8)
    ax.plot(sink_res['steps'], sink_res['fm_loss'],
            label='Sinkformer-CFM (soft)', color='darkorange', alpha=0.8)
    ax.set_title('FM Loss (lower is better)')
    ax.set_xlabel('Training step')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.set_yscale('log')

    # 2. OT cost (Sinkformer)
    ax = axes[0, 1]
    ax.plot(sink_res['steps'], sink_res['ot_costs_random'],
            label='Random', color='gray', alpha=0.5)
    ax.plot(sink_res['steps'], sink_res['ot_costs_exact'],
            label='Exact OT', color='green', linestyle='--')
    ax.plot(sink_res['steps'], sink_res['ot_costs_pred'],
            label='Sinkformer', color='darkorange')
    ax.set_title('OT Cost – Sinkformer-CFM')
    ax.set_xlabel('Training step')
    ax.legend()

    # 3. OT cost (OT-CFM)
    ax = axes[0, 2]
    ax.plot(otcfm_res['steps'], otcfm_res['ot_costs_random'],
            label='Random', color='gray', alpha=0.5)
    ax.plot(otcfm_res['steps'], otcfm_res['ot_costs_exact'],
            label='Exact OT', color='steelblue')
    ax.set_title('OT Cost – OT-CFM baseline')
    ax.set_xlabel('Training step')
    ax.legend()

    # 4-6. Pairing quality
    if sink_res['pair_quality']:
        pq      = sink_res['pair_quality']
        steps_pq = sink_res['steps'][:len(pq)]

        ax = axes[1, 0]
        ax.plot(steps_pq, [p['row_entropy'] for p in pq], color='darkorange')
        ax.set_title('Row entropy (lower = sharper 1-1 pairing)')
        ax.set_xlabel('Training step')

        ax = axes[1, 1]
        ax.plot(steps_pq, [p['row_err'] for p in pq], label='row error', color='blue')
        ax.plot(steps_pq, [p['col_err'] for p in pq], label='col error',
                color='red', linestyle='--')
        ax.set_title('DS Error (should be < 1e-3)')
        ax.set_yscale('log')
        ax.legend()

        ax = axes[1, 2]
        ax.plot(steps_pq, [p['sharp_frac'] for p in pq], color='purple')
        ax.set_title('Fraction of sharp pairings (max > 0.5/B)')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Training step')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/comparison.png', dpi=150)
    plt.close()
    print(f'Saved comparison → {OUTDIR}/comparison.png')


def plot_plan_grid(sink_res):
    plans = sink_res.get('plan_samples', [])
    if not plans:
        return
    n    = min(len(plans), 8)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for idx, P_np in enumerate(plans[:n]):
        B   = P_np.shape[0]
        sub = min(32, B)
        axes[idx].imshow(P_np[:sub, :sub] * B, cmap='Blues', vmin=0, vmax=1,
                         aspect='auto')
        axes[idx].set_title(f'Snapshot {idx+1}')
    plt.suptitle('Transport plan snapshots (P×B, top 32×32)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/plan_grid.png', dpi=120)
    plt.close()
    print(f'Saved plan grid → {OUTDIR}/plan_grid.png')


def analyse_results(otcfm_res, sink_res):
    print('\n' + '='*60)
    print('ANALYSIS – Round 5 (Decoupled: OT→pairer, FM→UNet)')
    print('='*60)

    last_n = 10
    loss_ot = np.mean(otcfm_res['fm_loss'][-last_n:]) if otcfm_res['fm_loss'] else float('nan')
    loss_sf = np.mean(sink_res['fm_loss'][-last_n:])  if sink_res['fm_loss']  else float('nan')

    print(f'\nFinal FM loss (last {last_n} log points):')
    print(f'  OT-CFM:                   {loss_ot:.5f}')
    print(f'  Sinkformer-CFM (soft FM): {loss_sf:.5f}')
    if loss_sf < loss_ot:
        print(f'  → Sinkformer-CFM BETTER by {(loss_ot - loss_sf)/loss_ot*100:.1f}%  ✓')
    else:
        print(f'  → OT-CFM better (SF-CFM is {(loss_sf - loss_ot)/loss_ot*100:.1f}% worse)')

    if sink_res['ot_costs_pred']:
        last_n2  = min(10, len(sink_res['ot_costs_pred']))
        c_pred   = np.mean(sink_res['ot_costs_pred'][-last_n2:])
        c_exact  = np.mean(sink_res['ot_costs_exact'][-last_n2:])
        c_rand   = np.mean(sink_res['ot_costs_random'][-last_n2:])
        reduction = (c_rand - c_pred) / (c_rand - c_exact + 1e-8) * 100
        print(f'\nSinkformer OT cost:')
        print(f'  Random: {c_rand:.4f}  |  Exact OT: {c_exact:.4f}  |  Sinkformer: {c_pred:.4f}')
        print(f'  → Achieves {reduction:.0f}% of possible cost reduction')

    if sink_res['pair_quality']:
        last_pq = sink_res['pair_quality'][-1]
        print(f'\nFinal pairing quality:')
        print(f'  DS row err:  {last_pq["row_err"]:.2e}  '
              f'{"OK" if last_pq["row_err"] < 1e-3 else "FAIL"}')
        print(f'  DS col err:  {last_pq["col_err"]:.2e}  '
              f'{"OK" if last_pq["col_err"] < 1e-3 else "FAIL"}')
        print(f'  Row entropy: {last_pq["row_entropy"]:.3f}  '
              f'(uniform: {np.log(BATCH):.2f})')
        print(f'  Sharp pairs: {last_pq["sharp_frac"]*100:.0f}%')

    print('\nRound 5 design:')
    print('  - P.detach() for soft FM: FM gradient stays in UNet only')
    print('  - Pairer trained on L_OT (pixel-space OT cost)')
    print('  - tau=5 init: P ~20% sharp from start → meaningful OT gradient')
    if loss_sf < loss_ot:
        print('  SUCCESS: Sinkformer-CFM converges faster/better than OT-CFM!')
    else:
        print('  NEXT STEPS if still underperforming:')
        print('    1. Add frequency (DCT) features for richer image representation')
        print('    2. Use larger tau (tau=10) for sharper initial P')
        print('    3. Pre-train pairer longer on L_OT before FM training starts')
        print('    4. Use separate encoders for x0 (noise) and x1 (images)')


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke',  action='store_true',
                        help='Quick smoke test (~30 sec per method)')
    parser.add_argument('--method', choices=['both', 'otcfm', 'sinkformer'],
                        default='both',
                        help='Which method(s) to run (default: both)')
    parser.add_argument('--minutes-ot', type=float, default=MINUTES_OT,
                        help=f'OT-CFM budget (default: {MINUTES_OT})')
    parser.add_argument('--minutes-sf', type=float, default=MINUTES_SF,
                        help=f'Sinkformer-CFM budget (default: {MINUTES_SF})')
    args = parser.parse_args()

    if args.smoke:
        MINUTES_OT_run = 0.4
        MINUTES_SF_run = 0.4
        LOG_EVERY = 1   # module-level, already accessible
    else:
        MINUTES_OT_run = args.minutes_ot
        MINUTES_SF_run = args.minutes_sf

    print(f'Device: {DEVICE}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)},  '
              f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    print('\n' + '='*60)
    print('Sinkformer-CFM (Round 3: Soft FM) vs OT-CFM on CIFAR-10')
    print(f'Batch={BATCH}, D_SF={D_SF}, N_SINK={N_SINK}, λ_OT={LAMBDA_OT}')
    print(f'Method={args.method}  OT={MINUTES_OT_run}min  SF={MINUTES_SF_run}min')
    print('='*60)

    # ── OT-CFM ────────────────────────────────────────────────────────────────
    if args.method in ('both', 'otcfm'):
        otcfm_res = train_otcfm(minutes=MINUTES_OT_run, tag='otcfm')
    else:
        # Try to load cached results; if not available, run OT-CFM
        otcfm_res = load_otcfm_results()
        if otcfm_res is None:
            print('  No cached OT-CFM results found; running OT-CFM first...')
            otcfm_res = train_otcfm(minutes=MINUTES_OT_run, tag='otcfm')

    # ── Sinkformer-CFM ────────────────────────────────────────────────────────
    if args.method in ('both', 'sinkformer'):
        sink_res = train_sinkformer_cfm(minutes=MINUTES_SF_run, tag='sink_cfm')

        # ── Visualise & Analyse ───────────────────────────────────────────────
        plot_comparison(otcfm_res, sink_res)
        plot_plan_grid(sink_res)
        analyse_results(otcfm_res, sink_res)

        np.save(f'{OUTDIR}/otcfm_losses.npy',  np.array(otcfm_res['fm_loss']))
        np.save(f'{OUTDIR}/sink_losses.npy',   np.array(sink_res['fm_loss']))
        np.save(f'{OUTDIR}/sink_ot_pred.npy',  np.array(sink_res['ot_costs_pred']))
        np.save(f'{OUTDIR}/sink_ot_exact.npy', np.array(sink_res['ot_costs_exact']))
        print(f'\nResults saved to {OUTDIR}/')
