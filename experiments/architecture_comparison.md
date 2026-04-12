# 架构对比：Standard Transformer / Sinkformer / OTTransformer

> 参考：Sander et al., "Sinkformers: Transformers with Doubly Stochastic Attention", AISTATS 2022 ([arXiv:2110.11773](https://arxiv.org/abs/2110.11773))

---

## 1. 标准 Transformer Self-Attention

**目的**：对一个序列做特征聚合（token 更新）

```
输入 x          [B, N, D]

Q = x @ W_Q     [B, N, D]
K = x @ W_K     [B, N, D]
V = x @ W_V     [B, N, D]

  ↓ reshape 成多头

Q               [B, H, N, d]      d = D / H
K               [B, H, N, d]
V               [B, H, N, d]

S = Q @ K^T / √d   [B, H, N, N]  ← 原始得分矩阵

A = softmax(S, dim=-1)  [B, H, N, N]
              ↑ 每行独立做 softmax
              行和 = 1  ✓
              列和 = 任意  ✗  ← row-stochastic only

out = A @ V         [B, H, N, d]
    → reshape       [B, N, D]
    → @ W_O         [B, N, D]

输出 x'             [B, N, D]     与输入同 shape
```

---

## 2. Sinkformer Self-Attention

**目的**：与标准 Transformer 相同，但 attention 矩阵满足 doubly stochastic

**唯一改动**：把 `softmax` 换成 `sinkhorn_attn(n_iters)`

```
输入 x          [B, N, D]

Q = x @ W_Q     [B, N, D]
K = x @ W_K     [B, N, D]
V = x @ W_V     [B, N, D]

  ↓ reshape 成多头

Q               [B, H, N, d]
K               [B, H, N, d]
V               [B, H, N, d]

S = Q @ K^T / √d   [B, H, N, N]

A = sinkhorn_attn(S, n_iters)   [B, H, N, N]
              ↑ 交替行/列归一化
              行和 = 1  ✓
              列和 = 1  ✓  ← doubly stochastic！

  Sinkhorn 迭代内部（对数域，从 K^0 = exp(S) 出发）：
    l=0 (偶数): log_K -= logsumexp(dim=-1)  → 行归一化 = softmax
    l=1 (奇数): log_K -= logsumexp(dim=-2)  → 列归一化
    l=2 (偶数): log_K -= logsumexp(dim=-1)  → 行归一化
    ...交替 n_iters 步

    n_iters=1  → 退化为普通 softmax（row-stochastic only）
    n_iters=5  → 原文推荐（适用于已训练的模型，scores 较温和）
    n_iters=20 → 本实现默认（覆盖随机初始化时 scores 方差较大的情况）

out = A @ V         [B, H, N, d]
    → reshape       [B, N, D]
    → @ W_O         [B, N, D]

输出 x'             [B, N, D]     与输入同 shape
```

---

## 3. OTTransformer（本项目，ot_transformer_verify.py）

**目的完全不同**：给定两个点集 X0, X1，输出它们之间的 OT transport plan

```
输入 x0         [B, N, d]    d=2（点坐标）
输入 x1         [B, N, d]

  ↓ 分别 MLP 编码

H0 = MLP(x0)    [B, N, D]    D=64（隐藏维）
H1 = MLP(x1)    [B, N, D]

  ↓ 分别做 Self-Attention（建立点集内部上下文）

H0 = SA(H0)     [B, N, D]
H1 = SA(H1)     [B, N, D]

  ↓ 跨点集线性投影（Cross-Attention 的 Q/K 部分）

Q = H0 @ W_Q    [B, N, D]    ← Q 来自 x0
K = H1 @ W_K    [B, N, D]    ← K 来自 x1

  ↓ 计算 x0 vs x1 的得分矩阵（单矩阵，无多头）

S[b,i,j] = Q[b,i] · K[b,j] / √D    [B, N, N]

  ↓ log_sinkhorn（目标边际 = 1/N，OT 约定）

P = log_sinkhorn(S, n_iters=100)    [B, N, N]
              行和 = 1/N  ✓
              列和 = 1/N  ✓
              P[b,i,j] = x0_i 传输到 x1_j 的质量

输出 P              [B, N, N]   transport plan（不是 token 序列！）
```

---

## 4. 关键区别对比表

| 特性 | Standard Transformer | Sinkformer | OTTransformer（本项目） |
|------|---------------------|------------|------------------------|
| **输入** | `[B, N, D]` 一个序列 | `[B, N, D]` 一个序列 | `[B, N, d]` × 2，两个点集 |
| **输出** | `[B, N, D]` 更新序列 | `[B, N, D]` 更新序列 | `[B, N, N]` transport plan |
| **Sinkhorn 位置** | 无 | attention 内部，替代 softmax | 最终输出层 |
| **Sinkhorn 作用在** | — | `[B, H, N, N]` per-head | `[B, N, N]` 单矩阵 |
| **边际约束** | 行和=1（softmax） | 行和=1，列和=1 | 行和=1/N，列和=1/N |
| **迭代次数** | — | 3~5（原文） | 100（保证收敛） |
| **Sinkhorn 是否可学** | — | 否（固定操作） | 否（固定操作） |
| **可学参数影响什么** | attention 权重分布 | attention 权重分布 | 得分矩阵 S 的形状 |
| **doubly stochastic 保证** | ✗ | ✓（只要 n_iters≥3） | ✓（只要 n_iters 足够） |
| **任务类型** | 序列建模 | 序列建模（改进版） | OT 配对计算 |
| **多头** | ✓ | ✓ | ✗（单矩阵） |

---

## 5. 直觉理解

### Sinkformer：改进 attention 的表达能力
```
标准 attention：token i 的权重分配（行 i）是独立的
              → 某个 token 可能被所有其他 token 忽视（列和≠1）

Sinkformer：   每个 token 既"平等分配注意力"又"被平等关注"
              → 信息流更均匀，防止注意力塌缩
```

### OTTransformer：用 transformer 计算 OT
```
标准 attention：S[i,j] = 相关性得分，归一化后聚合 V
              → 中间产物 A[b,h] 有意义但不是目标

OTTransformer：S[i,j] = x0_i 和 x1_j 的"亲和度"
              → P = sinkhorn(S) 就是最终目标 transport plan
              → V 投影和残差连接都不存在
```

### 边际约束的差异
```
Sinkformer:     行和=1，列和=1    → N×N matrix, total mass = N
OTTransformer:  行和=1/N，列和=1/N → N×N matrix, total mass = 1
（本质相同：都是 doubly stochastic，只差一个 N 的缩放）
```
