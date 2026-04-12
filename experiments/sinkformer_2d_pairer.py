"""
Shared Sinkformer 2D pairer for toy OT-CFM experiments.

Bipartite **cross-attention scores** (no softmax here): each x0_i only "sees" all x1_j
via dot-products Q_i·K_j / sqrt(d). Those scores are used as log-domain Gibbs logits
scaled by temperature tau, then fixed-step Sinkhorn yields doubly-stochastic P.

There is **no** explicit Euclidean cost matrix C_ij = ||q_i - k_j||^2 and no self-attention
within the x0 batch or within the x1 batch — only x0 ↔ x1 coupling in the score matrix.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def sinkhorn_uniform_marginals(log_kernel: torch.Tensor, n_iters: int) -> torch.Tensor:
    """
    Turn a positive kernel K = exp(log_kernel) into a doubly stochastic coupling P
    with uniform marginals u_i = v_j = 1/B (so B*P has row/col sums 1).

    Same log-domain scaling as the toy scripts: each sweep subtracts logsumexp then
    log(B), so exp(S) row/col sums become 1/B.

    log_kernel: [..., B, B]  (typically log K_ij, K_ij > 0)
    """
    B = log_kernel.shape[-1]
    if log_kernel.shape[-2] != B:
        raise ValueError("log_kernel must be square in the last two dims")
    logB = log_kernel.new_tensor(math.log(float(B)))

    S = log_kernel - log_kernel.amax(dim=(-2, -1), keepdim=True)
    for _ in range(n_iters):
        S = S - torch.logsumexp(S, dim=-1, keepdim=True) - logB
        S = S - torch.logsumexp(S, dim=-2, keepdim=True) - logB
    return S.exp()


class SinkPairer(nn.Module):
    """
    Bipartite cross-side scores -> Gibbs kernel -> Sinkhorn -> P.

    - enc0, enc1: map x0 / x1 in R^2 to d (separate heads for source vs target).
    - Q, K: linear maps; S_ij = (Q h0_i)·(K h1_j) / sqrt(d)  (pre-softmax attention).
    - K_ij = exp(S_ij / tau); log-domain Sinkhorn on log K gives P.
    """

    def __init__(self, d: int = 16, n_sink_iters: int = 20):
        super().__init__()
        self.d = int(d)
        self.n_sink_iters = int(n_sink_iters)
        self.log_tau = nn.Parameter(torch.tensor(float(math.log(5.0))))
        self.enc0 = nn.Linear(2, self.d, bias=False)
        self.enc1 = nn.Linear(2, self.d, bias=False)
        self.ln0 = nn.LayerNorm(self.d)
        self.ln1 = nn.LayerNorm(self.d)
        self.q = nn.Linear(self.d, self.d, bias=False)
        self.k = nn.Linear(self.d, self.d, bias=False)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        x0, x1: [B, 2]
        Returns P: [B, B] with sum_j P_ij == sum_i P_ij == 1/B (soft Sinkhorn coupling).
        """
        h0 = self.ln0(self.enc0(x0))
        h1 = self.ln1(self.enc1(x1))
        Q = self.q(h0)
        K = self.k(h1)
        scale = self.d ** -0.5
        S = (Q @ K.transpose(-1, -2)) * scale  # [B, B], i = x0 index, j = x1 index
        tau = self.log_tau.exp().clamp(1e-4, 50.0)
        log_k = (S / tau).clamp(min=-30.0, max=30.0)
        return sinkhorn_uniform_marginals(log_k, self.n_sink_iters)
