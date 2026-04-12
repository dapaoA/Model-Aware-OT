"""
sinkformer.py
=============
Sinkformer: Transformers with Doubly Stochastic Attention
原文: Sander et al., AISTATS 2022  arXiv:2110.11773

唯一改动：把 self-attention 里的 softmax 替换成 Sinkhorn 归一化。
其余结构（V projection、FFN、残差、LayerNorm）与标准 Transformer 完全相同。
"""

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# 核心：Sinkhorn 归一化（替代 softmax）
# ══════════════════════════════════════════════════════════════════════════════

def sinkhorn_attn(S: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    对数域 Sinkhorn 归一化，替代 softmax。

    原文迭代规则（从 K^0 = exp(S) 出发）：
      l=0 (偶数): 行归一化  → K^1 = softmax(S)    ← n_iters=1 时等价普通 attention
      l=1 (奇数): 列归一化  → K^2
      l=2 (偶数): 行归一化  → K^3
      ... 交替 n_iters 步

    收敛性质：
      n_iters = 1  →  行和 = 1（等价 softmax，row-stochastic only）
      n_iters = 3  →  行和 ≈ 1，列和 ≈ 1（doubly stochastic，推荐）
      n_iters = 5  →  更精确的 doubly stochastic（原文默认）

    Args:
        S       : [..., N, N]  原始得分矩阵 = QK^T / sqrt(d_head)
        n_iters : Sinkhorn 迭代次数（原文推荐 3~5）
    Returns:
        A       : [..., N, N]  attention 权重，行和 ≈ 1，列和 ≈ 1
    """
    # 中心化防止极端值导致 exp 溢出（不改变归一化结果）
    log_K = S - S.amax(dim=(-2, -1), keepdim=True)
    for l in range(n_iters):
        if l % 2 == 0:                                              # 行归一化
            log_K = log_K - log_K.logsumexp(dim=-1, keepdim=True)
        else:                                                        # 列归一化
            log_K = log_K - log_K.logsumexp(dim=-2, keepdim=True)
    return log_K.exp()


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Head Sinkhorn Attention
# ══════════════════════════════════════════════════════════════════════════════

class SinkAttention(nn.Module):
    """
    Multi-Head Self-Attention，softmax 替换为 Sinkhorn。

    shape 流程（B=batch, N=seq_len, D=d_model, H=n_heads, d=D//H）：
      输入  x        : [B, N, D]
      Q = x W_Q      : [B, N, D]  →  reshape  →  [B, H, N, d]
      K = x W_K      : [B, N, D]  →  reshape  →  [B, H, N, d]
      V = x W_V      : [B, N, D]  →  reshape  →  [B, H, N, d]
      S = QK^T/√d    : [B, H, N, N]
      A = sinkhorn(S): [B, H, N, N]  ← 行和≈1，列和≈1（不同于 softmax 仅行和=1）
      out = A @ V    : [B, H, N, d]  →  reshape  →  [B, N, D]
      输出  x W_O    : [B, N, D]
    """
    def __init__(self, d_model: int, n_heads: int, n_sink: int = 20):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.n_sink  = n_sink

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D]  →  [B, N, D]"""
        B, N, D = x.shape
        H, d    = self.n_heads, self.d_head

        # 线性投影 + reshape 成多头
        Q = self.W_Q(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        K = self.W_K(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        V = self.W_V(x).view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]

        # 得分矩阵
        S = torch.matmul(Q, K.transpose(-2, -1)) * (d ** -0.5)  # [B, H, N, N]

        # Sinkhorn 归一化（替代 softmax）
        A = sinkhorn_attn(S, self.n_sink)                        # [B, H, N, N]

        # 聚合 Value
        out = torch.matmul(A, V)                                 # [B, H, N, d]
        out = out.transpose(1, 2).reshape(B, N, D)               # [B, N, D]
        return self.W_O(out)                                      # [B, N, D]


# ══════════════════════════════════════════════════════════════════════════════
# Sinkformer Block（Pre-Norm）
# ══════════════════════════════════════════════════════════════════════════════

class SinkformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block。唯一改动：attention 换成 SinkAttention。

    shape：[B, N, D] → [B, N, D]
    """
    def __init__(self, d_model: int, n_heads: int, n_sink: int = 20,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = SinkAttention(d_model, n_heads, n_sink)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim    = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, N, D]"""
        x = x + self.attn(self.norm1(x))   # 残差 + Sinkhorn attention
        x = x + self.mlp(self.norm2(x))    # 残差 + FFN
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Sinkformer（完整模型）
# ══════════════════════════════════════════════════════════════════════════════

class Sinkformer(nn.Module):
    """
    Sinkformer：标准 Transformer，softmax attention → Sinkhorn attention。

    n_sink 控制迭代次数：
      n_sink = 1  → 退化为普通 softmax Transformer
      n_sink = 3  → doubly stochastic（推荐最小值）
      n_sink = 5  → 原文默认

    输入  : [B, N, D]  token 序列（已含位置编码或 embedding）
    输出  : [B, N, D]  更新后的 token 序列
    """
    def __init__(
        self,
        d_model  : int   = 64,
        n_heads  : int   = 4,
        n_layers : int   = 4,
        n_sink   : int   = 20,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SinkformerBlock(d_model, n_heads, n_sink, mlp_ratio)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, N, D]"""
        for block in self.blocks:
            x = block(x)
        return x

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# 快速验证
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      line_buffering=True)

    B, N, D = 2, 8, 64
    x = torch.randn(B, N, D)

    print("=== n_sink 对 attention 矩阵的影响 ===")
    # 模拟真实 transformer 的得分尺度：QK^T/sqrt(d_head)，d_head=16
    d_head = D // 4
    Q_t = torch.randn(B, 4, N, d_head)
    K_t = torch.randn(B, 4, N, d_head)
    S_real = torch.matmul(Q_t, K_t.transpose(-2, -1)) * (d_head ** -0.5)  # [B,H,N,N]
    for n_sink in [1, 5, 10, 20]:
        A = sinkhorn_attn(S_real, n_sink)
        row_err = (A.sum(dim=-1) - 1.0).abs().max().item()
        col_err = (A.sum(dim=-2) - 1.0).abs().max().item()
        print(f"  n_sink={n_sink}: row_err={row_err:.2e}, col_err={col_err:.2e}")

    print("\n=== Sinkformer 前向 shape 验证 ===")
    model = Sinkformer(d_model=D, n_heads=4, n_layers=2, n_sink=5)
    with torch.no_grad():
        out = model(x)
    print(f"  input : {list(x.shape)}")
    print(f"  output: {list(out.shape)}")
    print(f"  params: {model.n_params:,}")
    assert out.shape == x.shape
    print("  shape check passed")
