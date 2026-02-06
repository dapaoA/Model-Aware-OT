# EFM DCT 4x4 计算说明

## 1. 原始 EFM（欧氏距离）

给定：
- `x` (xt): [B, 3, H, W] 当前点
- `X1`: [M, 3, H, W] 目标图像 batch
- `t`: [B] 时间步

公式：
```
diff[i,j] = xt[i] - t[i] * X1[j]           # [B, M, 3, H, W]
scores[i,j] = -||diff[i,j]||² / (2(1-t)²)  # 欧氏距离的平方
w = softmax(scores, dim=1)                  # [B, M]
v[j] = (X1[j] - xt) / (1-t)                # 方向向量
u_efm = Σ_j w[j] * v[j]                    # 加权和
```

## 2. EFM DCT 4x4（DCT 低频距离）

**关键**：先对 xt 和 t*X1 **分别做 DCT**，然后在 **DCT 空间计算距离**：

```
dct_xt = DCT_4x4_low(xt)                              # 对 xt 做 DCT → [B, 48]
dct_tX1[j] = DCT_4x4_low(t * X1[j])                   # 对 t*X1[j] 做 DCT → [M, 48]
diff_dct[i,j] = dct_xt[i] - dct_tX1[j]                # DCT 空间的差
dist_sq[i,j] = ||diff_dct[i,j]||²                     # DCT 空间的距离平方
scores[i,j] = -dist_sq[i,j] / (2(1-t)²)
w = softmax(scores, dim=1)
v[j] = (X1[j] - xt) / (1-t)                           # 方向仍在像素空间
u_efm = Σ_j w[j] * v[j]
```

### DCT 4x4 低频提取
- 对图像的每个通道做 2D DCT
- 按 zigzag 顺序取前 16 个系数
- 3 通道 × 16 系数 = 48 维

**之前的错误实现**（已修复）：
```
diff = xt - t*X1                    # 像素空间的差
dct_low = DCT_4x4_low(diff)         # 对差做 DCT（错误！）
```

正确的应该是**先 DCT，再算差**，而不是先算差再 DCT！

## 3. 为什么 EFM DCT 4x4 效果差

1. **维度太少**：只用 48 维 vs 欧氏 3072 维，区分度下降，softmax 更接近均匀，u_efm 是多方向的平均。

2. **距离含义不同**：模型在像素空间训练，方向 `v = (x1 - x0)/(1-t)` 在像素空间；DCT 低频距离可能把不同目标排错序，导致权重偏向错误目标。

3. **尺度差异**：`||DCT_low||²` 和 `||diff||²` 量级不同，即使公式形式相同，softmax 的分布也会不同。

## 4. 代码位置

- `utils/efm.py`: `efm_closed_form_weights_and_u_dct_4x4()`
- `_dct_4x4_low_flat()`: 提取 DCT 4x4 低频系数
- 运行 `python debug_efm_dct_4x4.py` 可查看逐步计算与欧氏 EFM 对比
