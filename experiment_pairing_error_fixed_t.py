"""
实验脚本：在固定时间t下比较不同配对方式的理论去噪方向与模型预测方向的误差

输入变量：
- 模型checkpoint路径
- batch_size
- 配对方式（cfm, otcfm, ma_tcfm）

计算：
- 对于任意采样的noise和image batch对
- 用不同配对方式计算配对组合
- 在固定时间t下（0.05, 0.25, 0.5, 0.75, 0.95）
- 计算理论去噪方向（配对后的图片-噪声点）和模型预测的去噪方向v(xt, t)的余弦相似度
- 对每个t值分别绘制三种方法的cos分布对比图
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import scipy.optimize

from model import create_model, load_model_config
from flow_matcher import create_flow_matcher
from dataset import get_dataset
from torchcfm.optimal_transport import OTPlanSampler


# 固定的时间t值
FIXED_T_VALUES = [0.00, 0.05, 0.25, 0.5, 0.75, 0.95]


def compute_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度（按样本计算）"""
    # vec1, vec2: (batch_size, *dims)
    # 展平为 (batch_size, -1)
    vec1_flat = vec1.reshape(vec1.shape[0], -1)
    vec2_flat = vec2.reshape(vec2.shape[0], -1)
    
    # 计算余弦相似度
    dot_product = (vec1_flat * vec2_flat).sum(dim=1)
    norm1 = torch.norm(vec1_flat, dim=1)
    norm2 = torch.norm(vec2_flat, dim=1)
    
    # 避免除零
    cos_sim = dot_product / (norm1 * norm2 + 1e-8)
    return cos_sim


def compute_l2_distance(vec1, vec2):
    """计算两个向量的L2距离（按样本计算）"""
    # vec1, vec2: (batch_size, *dims)
    # 展平为 (batch_size, -1)
    vec1_flat = vec1.reshape(vec1.shape[0], -1)
    vec2_flat = vec2.reshape(vec2.shape[0], -1)
    
    # 计算L2距离
    l2_dist = torch.norm(vec1_flat - vec2_flat, dim=1)
    return l2_dist


def get_paired_samples_cfm(x0, x1):
    """CFM: 随机配对"""
    # 随机打乱x1的顺序
    indices = torch.randperm(x1.shape[0], device=x1.device)
    x1_paired = x1[indices]
    return x0, x1_paired


def get_paired_samples_otcfm(x0, x1):
    """OTCFM: 使用OT配对 - 使用匈牙利算法获取确定性最优配对"""
    # Compute cost matrix
    x0_flat = x0.reshape(x0.shape[0], -1)
    x1_flat = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0_flat, x1_flat) ** 2
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
    
    # Convert to torch tensor
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(x0.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(x1.device)
    
    return x0[row_ind], x1[col_ind]


def get_paired_samples_ma_tcfm(x0, x1, ma_method='downsample_2x'):
    """MA_TCFM: 使用模型感知的OT配对"""
    import torch.nn.functional as F
    import numpy as np
    
    # 应用模型感知变换（与MA_ExactOT中的实现一致）
    if ma_method == 'downsample_2x':
        def M(x):
            # 2x下采样：使用avg_pool2d
            if x.dim() == 4:  # (B, C, H, W)
                return F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
            else:
                return x
    else:
        raise ValueError(f"Unknown ma_method: {ma_method}. Supported: downsample_2x")
    
    # 在变换后的空间计算OT plan
    x0_transformed = M(x0)
    x1_transformed = M(x1)
    
    # Compute cost matrix in transformed space
    x0_flat = x0_transformed.reshape(x0_transformed.shape[0], -1)
    x1_flat = x1_transformed.reshape(x1_transformed.shape[0], -1)
    M_cost = torch.cdist(x0_flat, x1_flat) ** 2
    
    # Use Hungarian algorithm to find optimal assignment in transformed space
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cost.cpu().numpy())
    
    # Convert to torch tensor
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(x0.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(x1.device)
    
    # 使用原始（未变换）的x0和x1进行配对
    x0_paired = x0[row_ind]
    x1_paired = x1[col_ind]
    
    return x0_paired, x1_paired


def get_paired_samples_ma_incepcfm(x0, x1):
    """MA_IncepCFM: 使用Inception v3特征提取的模型感知OT配对"""
    import numpy as np
    
    # 使用MA_ExactOT的Inception特征提取方法
    # 创建一个临时的MA_ExactOT实例来使用其Inception特征提取器
    from torchcfm.conditional_flow_matching import MA_ExactOT
    
    # 创建临时实例（只用于特征提取，不需要完整的flow matcher）
    # 注意：这会加载Inception模型，但只初始化一次（使用函数属性缓存）
    # 使用设备作为key来支持多设备
    device_key = str(x0.device)
    cache_key = f'_inception_extractor_{device_key}'
    
    if not hasattr(get_paired_samples_ma_incepcfm, cache_key):
        # 使用lazy initialization，只在第一次调用时创建
        temp_ma = MA_ExactOT(sigma=0.0, ma_method='inception')
        # 确保Inception模型在正确的设备上
        if hasattr(temp_ma, 'inception_model'):
            temp_ma.inception_model = temp_ma.inception_model.to(x0.device)
            temp_ma._inception_device = x0.device
        setattr(get_paired_samples_ma_incepcfm, cache_key, temp_ma.M)
    
    M = getattr(get_paired_samples_ma_incepcfm, cache_key)
    
    # 在变换后的空间（Inception特征空间）计算OT plan
    x0_transformed = M(x0)
    x1_transformed = M(x1)
    
    # Compute cost matrix in transformed space (Inception features are already flattened)
    M_cost = torch.cdist(x0_transformed, x1_transformed) ** 2
    
    # Use Hungarian algorithm to find optimal assignment in transformed space
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cost.cpu().numpy())
    
    # Convert to torch tensor
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(x0.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(x1.device)
    
    # 使用原始（未变换）的x0和x1进行配对
    x0_paired = x0[row_ind]
    x1_paired = x1[col_ind]
    
    return x0_paired, x1_paired


def _get_zigzag_indices(h, w):
    """获取zigzag扫描的索引（带缓存）
    
    Args:
        h, w: 矩阵的高度和宽度
    
    Returns:
        indices: zigzag扫描的索引列表 [(h0, w0), (h1, w1), ...]
    """
    cache_key = (h, w)
    if not hasattr(_get_zigzag_indices, '_cache'):
        _get_zigzag_indices._cache = {}
    
    if cache_key not in _get_zigzag_indices._cache:
        total_coeffs = h * w
        indices = []
        i, j = 0, 0
        direction = 1  # 1: 向上，-1: 向下
        
        while len(indices) < total_coeffs and (i < h and j < w):
            indices.append((i, j))
            
            if direction == 1:  # 向上移动
                if i == 0 or j == w - 1:
                    if j == w - 1:
                        i += 1
                    else:
                        j += 1
                    direction = -1
                else:
                    i -= 1
                    j += 1
            else:  # 向下移动
                if j == 0 or i == h - 1:
                    if i == h - 1:
                        j += 1
                    else:
                        i += 1
                    direction = 1
                else:
                    i += 1
                    j -= 1
        
        _get_zigzag_indices._cache[cache_key] = indices[:total_coeffs]
    
    return _get_zigzag_indices._cache[cache_key]


def _zigzag_to_1d(dct_2d):
    """将2D DCT矩阵按zigzag扫描顺序转换为1D数组
    
    Args:
        dct_2d: 2D DCT系数矩阵 (H, W)
    
    Returns:
        dct_1d: 1D数组，按zigzag顺序排列 (H*W,)
    """
    h, w = dct_2d.shape
    indices = _get_zigzag_indices(h, w)
    # 使用缓存的索引提取并重排为1D
    dct_1d = np.array([dct_2d[h_idx, w_idx] for (h_idx, w_idx) in indices])
    return dct_1d


def get_paired_samples_dct_otcfm(x0, x1, low_freq_size=8):
    """DCT_OTCFM: 使用DCT变换提取低频信息，然后在低频空间进行OT配对
    
    对x0和x1进行DCT变换，按照zigzag扫描顺序提取低频部分（前N个系数），然后在低频空间计算OT配对。
    
    Args:
        x0: 噪声batch (N, C, H, W)
        x1: 图像batch (M, C, H, W)
        low_freq_size: 提取的低频系数数量（zigzag顺序中的前N个），默认8x8=64个系数
    """
    from scipy.fft import dctn
    
    B0, C, H, W = x0.shape
    B1, C, H, W = x1.shape
    
    # 计算要提取的低频系数数量
    low_freq_num = low_freq_size * low_freq_size
    
    # 对每个通道和每个样本分别做2D DCT变换并转换为1D zigzag顺序
    x0_low_coeffs = []
    for i in range(B0):
        sample_coeffs = []
        for c in range(C):
            # DCT变换：对2D图像做DCT
            x_channel = x0[i, c, :, :].cpu().numpy()  # (H, W)
            x_channel_dct = dctn(x_channel, norm='ortho')  # (H, W)
            # 按zigzag顺序转换为1D，然后取前low_freq_num个（低频）
            dct_1d = _zigzag_to_1d(x_channel_dct)  # (H*W,)
            dct_low = dct_1d[:low_freq_num]  # 取前N个
            sample_coeffs.append(torch.from_numpy(dct_low).to(x0.device))
        x0_low_coeffs.append(torch.cat(sample_coeffs, dim=0))  # (C * low_freq_num,)
    x0_low_flat = torch.stack(x0_low_coeffs, dim=0)  # (B0, C * low_freq_num)
    
    x1_low_coeffs = []
    for i in range(B1):
        sample_coeffs = []
        for c in range(C):
            x_channel = x1[i, c, :, :].cpu().numpy()  # (H, W)
            x_channel_dct = dctn(x_channel, norm='ortho')  # (H, W)
            # 按zigzag顺序转换为1D，然后取前low_freq_num个（低频）
            dct_1d = _zigzag_to_1d(x_channel_dct)  # (H*W,)
            dct_low = dct_1d[:low_freq_num]  # 取前N个
            sample_coeffs.append(torch.from_numpy(dct_low).to(x1.device))
        x1_low_coeffs.append(torch.cat(sample_coeffs, dim=0))  # (C * low_freq_num,)
    x1_low_flat = torch.stack(x1_low_coeffs, dim=0)  # (B1, C * low_freq_num)
    
    # 在低频空间计算cost matrix并使用匈牙利算法配对
    M_cost = torch.cdist(x0_low_flat, x1_low_flat) ** 2
    
    # Use Hungarian algorithm to find optimal assignment in DCT low-frequency space
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cost.cpu().numpy())
    
    # Convert to torch tensor
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(x0.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(x1.device)
    
    # 使用原始（未变换）的x0和x1进行配对
    x0_paired = x0[row_ind]
    x1_paired = x1[col_ind]
    
    return x0_paired, x1_paired


def get_paired_samples_dct_hf_otcfm(x0, x1, high_freq_size=8):
    """DCT_HF_OTCFM: 使用DCT变换提取高频信息，然后在高频空间进行OT配对
    
    对x0和x1进行DCT变换，按照zigzag扫描顺序提取高频部分（最后N个系数），然后在高频空间计算OT配对。
    
    Args:
        x0: 噪声batch (N, C, H, W)
        x1: 图像batch (M, C, H, W)
        high_freq_size: 提取的高频系数数量（zigzag顺序中的最后N个），默认8x8=64个系数
    """
    from scipy.fft import dctn
    
    B0, C, H, W = x0.shape
    B1, C, H, W = x1.shape
    
    # 计算要提取的高频系数数量
    total_coeffs = H * W
    high_freq_num = high_freq_size * high_freq_size
    
    # 对每个通道和每个样本分别做2D DCT变换并转换为1D zigzag顺序
    x0_high_coeffs = []
    for i in range(B0):
        sample_coeffs = []
        for c in range(C):
            # DCT变换：对2D图像做DCT
            x_channel = x0[i, c, :, :].cpu().numpy()  # (H, W)
            x_channel_dct = dctn(x_channel, norm='ortho')  # (H, W)
            # 按zigzag顺序转换为1D，然后取最后high_freq_num个（高频）
            dct_1d = _zigzag_to_1d(x_channel_dct)  # (H*W,)
            dct_high = dct_1d[-high_freq_num:] if high_freq_num <= total_coeffs else dct_1d  # 取最后N个
            sample_coeffs.append(torch.from_numpy(dct_high).to(x0.device))
        x0_high_coeffs.append(torch.cat(sample_coeffs, dim=0))  # (C * high_freq_num,)
    x0_high_flat = torch.stack(x0_high_coeffs, dim=0)  # (B0, C * high_freq_num)
    
    x1_high_coeffs = []
    for i in range(B1):
        sample_coeffs = []
        for c in range(C):
            x_channel = x1[i, c, :, :].cpu().numpy()  # (H, W)
            x_channel_dct = dctn(x_channel, norm='ortho')  # (H, W)
            # 按zigzag顺序转换为1D，然后取最后high_freq_num个（高频）
            dct_1d = _zigzag_to_1d(x_channel_dct)  # (H*W,)
            dct_high = dct_1d[-high_freq_num:] if high_freq_num <= total_coeffs else dct_1d  # 取最后N个
            sample_coeffs.append(torch.from_numpy(dct_high).to(x1.device))
        x1_high_coeffs.append(torch.cat(sample_coeffs, dim=0))  # (C * high_freq_num,)
    x1_high_flat = torch.stack(x1_high_coeffs, dim=0)  # (B1, C * high_freq_num)
    
    # 在高频空间计算cost matrix并使用匈牙利算法配对
    M_cost = torch.cdist(x0_high_flat, x1_high_flat) ** 2
    
    # Use Hungarian algorithm to find optimal assignment in DCT high-frequency space
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cost.cpu().numpy())
    
    # Convert to torch tensor
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(x0.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(x1.device)
    
    # 使用原始（未变换）的x0和x1进行配对
    x0_paired = x0[row_ind]
    x1_paired = x1[col_ind]
    
    return x0_paired, x1_paired


def get_paired_samples_dct_mf_otcfm(x0, x1, mid_freq_size=8):
    """DCT_MF_OTCFM: 使用DCT变换提取中间频率信息，然后在中间频率空间进行OT配对
    
    对x0和x1进行DCT变换，按照zigzag扫描顺序提取中间频率部分，然后在中间频率空间计算OT配对。
    
    Args:
        x0: 噪声batch (N, C, H, W)
        x1: 图像batch (M, C, H, W)
        mid_freq_size: 提取的中间频率系数数量（zigzag顺序中的中间部分），默认8x8=64个系数
    """
    from scipy.fft import dctn
    
    B0, C, H, W = x0.shape
    B1, C, H, W = x1.shape
    
    # 计算zigzag顺序的中间频率索引
    # 总系数数
    total_coeffs = H * W
    # 跳过前low_freq_num个（低频）和最后high_freq_num个（高频），取中间的
    # 假设低频占25%，高频占25%，中间频率占50%
    low_freq_num = total_coeffs // 4
    high_freq_num = total_coeffs // 4
    mid_freq_num = min(mid_freq_size * mid_freq_size, total_coeffs - low_freq_num - high_freq_num)
    
    # 对每个通道和每个样本分别做2D DCT变换并转换为1D zigzag顺序
    x0_mid_coeffs = []
    for i in range(B0):
        sample_coeffs = []
        for c in range(C):
            # DCT变换：对2D图像做DCT
            x_channel = x0[i, c, :, :].cpu().numpy()  # (H, W)
            x_channel_dct = dctn(x_channel, norm='ortho')  # (H, W)
            # 按zigzag顺序转换为1D，然后取中间mid_freq_num个（中间频率）
            dct_1d = _zigzag_to_1d(x_channel_dct)  # (H*W,)
            dct_mid = dct_1d[low_freq_num:low_freq_num + mid_freq_num]  # 取中间N个
            sample_coeffs.append(torch.from_numpy(dct_mid).to(x0.device))
        x0_mid_coeffs.append(torch.cat(sample_coeffs, dim=0))  # (C * mid_freq_num,)
    x0_mid_flat = torch.stack(x0_mid_coeffs, dim=0)  # (B0, C * mid_freq_num)
    
    x1_mid_coeffs = []
    for i in range(B1):
        sample_coeffs = []
        for c in range(C):
            x_channel = x1[i, c, :, :].cpu().numpy()  # (H, W)
            x_channel_dct = dctn(x_channel, norm='ortho')  # (H, W)
            # 按zigzag顺序转换为1D，然后取中间mid_freq_num个（中间频率）
            dct_1d = _zigzag_to_1d(x_channel_dct)  # (H*W,)
            dct_mid = dct_1d[low_freq_num:low_freq_num + mid_freq_num]  # 取中间N个
            sample_coeffs.append(torch.from_numpy(dct_mid).to(x1.device))
        x1_mid_coeffs.append(torch.cat(sample_coeffs, dim=0))  # (C * mid_freq_num,)
    x1_mid_flat = torch.stack(x1_mid_coeffs, dim=0)  # (B1, C * mid_freq_num)
    
    # 在中间频率空间计算cost matrix并使用匈牙利算法配对
    M_cost = torch.cdist(x0_mid_flat, x1_mid_flat) ** 2
    
    # Use Hungarian algorithm to find optimal assignment in DCT mid-frequency space
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cost.cpu().numpy())
    
    # Convert to torch tensor
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(x0.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(x1.device)
    
    # 使用原始（未变换）的x0和x1进行配对
    x0_paired = x0[row_ind]
    x1_paired = x1[col_ind]
    
    return x0_paired, x1_paired


def get_paired_samples_bary_otcfm(x0, x1, reg=0.1):
    """Bary_OTCFM: 使用熵正则化OT计算配对，然后计算x0的加权平均（barycenter）替代x0
    
    使用Sinkhorn算法（熵正则化OT）计算OT plan，这样OT plan不是稀疏的，
    一个点可以匹配到多个点。根据OT plan，对于每个x1_j，找到所有匹配到它的x0_i，
    计算这些x0_i的加权平均，权重满足 sum(w_i^2) = 1 以保持噪声统计特性。
    
    Args:
        x0: 噪声batch (N, *dims)
        x1: 图像batch (M, *dims)
        reg: Sinkhorn正则化参数，默认0.1（增大以减少数值问题）
    """
    from torchcfm.optimal_transport import OTPlanSampler
    import numpy as np
    
    # Get OT plan using Sinkhorn (entropy-regularized OT) to get non-sparse plan
    # This allows one point to be transported to multiple points
    # Use normalize_cost=True to improve numerical stability
    ot_sampler = OTPlanSampler(method="sinkhorn", reg=reg, normalize_cost=True)
    pi = ot_sampler.get_map(x0.cpu(), x1.cpu())  # Shape: (N, M)
    
    # Convert to torch tensor
    if isinstance(pi, np.ndarray):
        pi = torch.from_numpy(pi).to(x0.device).float()
    
    # For each x1_j, compute weighted average of x0_i
    x0_bary = []
    
    for j in range(x1.shape[0]):
        # Get OT plan weights for x1_j: pi[:, j]
        weights = pi[:, j]  # Shape: (N,)
        
        # Normalize weights so that sum(w_i^2) = 1 to preserve noise statistics
        weight_sum_sq = torch.sum(weights ** 2)
        if weight_sum_sq > 1e-8:
            weights_normalized = weights / torch.sqrt(weight_sum_sq)
        else:
            # Fallback: uniform weights
            weights_normalized = torch.ones_like(weights) / torch.sqrt(torch.tensor(float(weights.shape[0]), device=weights.device))
        
        # Compute weighted average: x0_bary[j] = sum_i(w_i * x0[i])
        weights_expanded = weights_normalized.view(-1, *([1] * (x0.dim() - 1)))  # (N, 1, ..., 1)
        x0_bary_j = torch.sum(weights_expanded * x0, dim=0)  # (*dim,)
        x0_bary.append(x0_bary_j)
    
    x0_bary = torch.stack(x0_bary, dim=0)  # (M, *dim)
    
    # x0_bary 和 x1 现在配对（每个 x0_bary[j] 对应 x1[j]）
    return x0_bary, x1


def get_paired_samples_mac_cfm(x0, x1, model):
    """MAC_CFM: 使用模型预测计算配对损失，通过匈牙利算法找到最小损失配对
    
    对于每个配对(x0_i, x1_j)，计算：
        loss = ||v(x0_i, 0) - (x1_j - x0_i)||^2 + ||v(x1_j, 1) - (x1_j - x0_i)||^2
    
    使用匈牙利算法找到使总损失最小的配对。
    """
    import numpy as np
    import scipy.optimize
    
    model.eval()
    with torch.no_grad():
        N = x0.shape[0]
        M = x1.shape[0]
        
        if N != M:
            raise ValueError("MAC_CFM currently requires x0 and x1 to have the same batch size")
        
        # Theoretical direction: x1_j - x0_i for each pair (i, j)
        # Shape: (N, M, *dim)
        x0_expanded = x0.unsqueeze(1).expand(-1, M, *[-1] * (x0.dim() - 1))
        x1_expanded = x1.unsqueeze(0).expand(N, -1, *[-1] * (x1.dim() - 1))
        theoretical_direction = x1_expanded - x0_expanded  # (N, M, *dim)
        
        # Compute v(x0_i, 0) for all x0_i
        t0 = torch.zeros(N, device=x0.device)
        if x0.dim() == 4:  # Image data: model(x, t)
            v_x0_0 = model(x0, t0)  # (N, *dim)
        else:  # 2D data: model(torch.cat([x, t], dim=-1))
            v_x0_0 = model(torch.cat([x0, t0.unsqueeze(-1)], dim=-1))
        
        # Compute v(x1_j, 1) for all x1_j
        t1 = torch.ones(M, device=x1.device)
        if x1.dim() == 4:  # Image data: model(x, t)
            v_x1_1 = model(x1, t1)  # (M, *dim)
        else:  # 2D data: model(torch.cat([x, t], dim=-1))
            v_x1_1 = model(torch.cat([x1, t1.unsqueeze(-1)], dim=-1))
        
        # Expand predictions to match theoretical_direction shape
        v_x0_0_expanded = v_x0_0.unsqueeze(1).expand(-1, M, *[-1] * (v_x0_0.dim() - 1))
        v_x1_1_expanded = v_x1_1.unsqueeze(0).expand(N, -1, *[-1] * (v_x1_1.dim() - 1))
        
        # Flatten spatial dimensions for L2 loss
        v_x0_0_flat = v_x0_0_expanded.reshape(N, M, -1)
        v_x1_1_flat = v_x1_1_expanded.reshape(N, M, -1)
        theoretical_direction_flat = theoretical_direction.reshape(N, M, -1)
        
        # Compute losses for each pair
        loss_v0 = torch.sum((v_x0_0_flat - theoretical_direction_flat) ** 2, dim=-1)  # (N, M)
        loss_v1 = torch.sum((v_x1_1_flat - theoretical_direction_flat) ** 2, dim=-1)  # (N, M)
        loss_matrix = loss_v0 + loss_v1  # (N, M)
        
        # Use Hungarian algorithm to find optimal pairing
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(loss_matrix.cpu().numpy())
        
        # Convert to torch tensors
        if isinstance(row_ind, np.ndarray):
            row_ind = torch.from_numpy(row_ind).to(x0.device)
        if isinstance(col_ind, np.ndarray):
            col_ind = torch.from_numpy(col_ind).to(x1.device)
        
        # Return paired samples
        x0_paired = x0[row_ind]
        x1_paired = x1[col_ind]
        
    return x0_paired, x1_paired


def compute_pairing_error_at_t(model, flow_matcher, x0, x1, pairing_method, device, t_value, epsilon, metric='cos'):
    """
    在固定时间t下计算配对误差
    
    Args:
        model: 训练好的模型（应该用CFM训练的模型）
        flow_matcher: flow matcher实例（用于计算xt和ut）
        x0: 噪声batch (batch_size, *dims)
        x1: 图像batch (batch_size, *dims)
        pairing_method: 配对方式 ('cfm', 'otcfm', 'ma_tcfm')
        device: 设备
        t_value: 固定的时间t值（标量）
        epsilon: 单个epsilon样本 (batch_size, *dims)的tensor
        metric: 度量方式 ('cos' 或 'l2')
        
    Returns:
        errors: 每个配对的误差值 (batch_size,)
    """
    model.eval()
    
    with torch.no_grad():
        # 根据配对方式获取配对后的x0和x1
        if pairing_method == 'cfm':
            x0_paired, x1_paired = get_paired_samples_cfm(x0, x1)
        elif pairing_method == 'otcfm':
            x0_paired, x1_paired = get_paired_samples_otcfm(x0, x1)
        elif pairing_method == 'ma_tcfm':
            x0_paired, x1_paired = get_paired_samples_ma_tcfm(x0, x1, ma_method='downsample_2x')
        elif pairing_method == 'ma_incepcfm':
            x0_paired, x1_paired = get_paired_samples_ma_incepcfm(x0, x1)
        elif pairing_method == 'dct_otcfm_4x4':
            x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=4)
        elif pairing_method == 'dct_otcfm_8x8':
            x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=8)
        elif pairing_method == 'dct_otcfm_16x16':
            x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=16)
        elif pairing_method == 'dct_otcfm':  # 保持向后兼容，默认8x8
            x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=8)
        elif pairing_method == 'dct_hf_otcfm_4x4':
            x0_paired, x1_paired = get_paired_samples_dct_hf_otcfm(x0, x1, high_freq_size=4)
        elif pairing_method == 'dct_hf_otcfm_8x8':
            x0_paired, x1_paired = get_paired_samples_dct_hf_otcfm(x0, x1, high_freq_size=8)
        elif pairing_method == 'dct_hf_otcfm_16x16':
            x0_paired, x1_paired = get_paired_samples_dct_hf_otcfm(x0, x1, high_freq_size=16)
        elif pairing_method == 'dct_mf_otcfm_4x4':
            x0_paired, x1_paired = get_paired_samples_dct_mf_otcfm(x0, x1, mid_freq_size=4)
        elif pairing_method == 'dct_mf_otcfm_8x8':
            x0_paired, x1_paired = get_paired_samples_dct_mf_otcfm(x0, x1, mid_freq_size=8)
        else:
            raise ValueError(f"Unknown pairing method: {pairing_method}")
        
        # 计算理论去噪方向 ut = x1_paired - x0_paired（配对后的方向）
        ut_theoretical = x1_paired - x0_paired
        
        # 创建固定的t tensor
        batch_size = x0_paired.shape[0]  # 使用 x0_paired 的 batch size
        t = torch.full((batch_size,), t_value, device=device, dtype=torch.float32)
        
        # 使用手动配对的x0_paired和x1_paired计算xt
        # 计算xt = t * x1_paired + (1-t) * x0_paired + sigma * epsilon
        t_expanded = t.reshape(-1, *([1] * (x0_paired.dim() - 1)))
        mu_t = t_expanded * x1_paired + (1 - t_expanded) * x0_paired
        sigma_t = flow_matcher.compute_sigma_t(t)
        sigma_t_expanded = sigma_t.reshape(-1, *([1] * (mu_t.dim() - 1))) if isinstance(sigma_t, torch.Tensor) else sigma_t
        if isinstance(sigma_t_expanded, (int, float)):
            xt = mu_t + sigma_t_expanded * epsilon
        else:
            xt = mu_t + sigma_t_expanded * epsilon
        
        # 模型预测的去噪方向
        if x1_paired.dim() == 4:  # 图像数据
            vt_predicted = model(xt, t)
        else:  # 2D数据
            vt_predicted = model(torch.cat([xt, t[:, None]], dim=-1))
        
        # 计算误差（每个配对一个值）
        if metric == 'cos':
            errors = compute_cosine_similarity(ut_theoretical, vt_predicted)
        elif metric == 'l2':
            errors = compute_l2_distance(ut_theoretical, vt_predicted)
        else:
            raise ValueError(f"Unknown metric: {metric}. Must be 'cos' or 'l2'")
    
    return errors.cpu().numpy()


def plot_results_for_t(t_value, results, output_dir, model_info, pairing_methods, method_labels, colors, metric='cos'):
    """为单个t值绘制对比图"""
    plt.figure(figsize=(18, 5))
    
    # 准备绘图数据
    data_to_plot = [results[method] for method in pairing_methods if method in results]
    labels_to_plot = [method_labels[method] for method in pairing_methods if method in results]
    
    # 确定标签
    if metric == 'cos':
        metric_label = 'Cosine Similarity'
        metric_title = 'Cosine Similarities'
    elif metric == 'l2':
        metric_label = 'L2 Distance'
        metric_title = 'L2 Distances'
    else:
        metric_label = 'Error'
        metric_title = 'Errors'
    
    # 1. KDE曲线对比（使用scipy的gaussian_kde）
    plt.subplot(1, 3, 1)
    x_min = min([data.min() for data in data_to_plot])
    x_max = max([data.max() for data in data_to_plot])
    x_range = np.linspace(x_min, x_max, 200)
    
    for idx, (method, data) in enumerate(zip(pairing_methods, data_to_plot)):
        if method in results:
            kde = gaussian_kde(data)
            kde_values = kde(x_range)
            plt.plot(x_range, kde_values, label=method_labels[method], 
                    color=colors[idx % len(colors)], linewidth=2)
    
    plt.xlabel(metric_label, fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title(f'Distribution of {metric_title} (KDE) - t={t_value}', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # 2. Violin plot对比（显示密度分布）
    plt.subplot(1, 3, 2)
    parts = plt.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                          showmeans=True, showmedians=True, widths=0.7)
    
    # 设置颜色
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idx % len(colors)])
        pc.set_alpha(0.7)
    
    plt.ylabel(metric_label, fontsize=11)
    plt.title(f'Violin Plot Comparison - t={t_value}', fontsize=12)
    plt.xticks(range(len(labels_to_plot)), labels_to_plot, rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 3. 统计摘要（调整y轴范围以突出差异）
    plt.subplot(1, 3, 3)
    methods_plot = [method_labels[method] for method in pairing_methods if method in results]
    means = [results[method].mean() for method in pairing_methods if method in results]
    stds = [results[method].std() for method in pairing_methods if method in results]
    
    x_pos = np.arange(len(methods_plot))
    
    # 计算y轴范围（只显示有差异的部分）
    y_min = min(means) - max(stds) * 1.5
    y_max = max(means) + max(stds) * 1.5
    y_range = y_max - y_min
    # 添加一些边距
    y_min = y_min - y_range * 0.1
    y_max = y_max + y_range * 0.1
    
    bars = plt.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, 
                   color=[colors[idx % len(colors)] for idx in range(len(methods_plot))])
    plt.xlabel('Method', fontsize=11)
    plt.ylabel(f'Mean {metric_label}', fontsize=11)
    plt.title(f'Mean ± Std Comparison - t={t_value}', fontsize=12)
    plt.xticks(x_pos, methods_plot, rotation=45, ha='right', fontsize=10)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + y_range * 0.02, f'{mean:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片（文件名包含t值和metric）
    png_filename = f"pairing_error_comparison_{model_info}_t{t_value:.2f}_{metric}.png"
    plt.savefig(output_dir / png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return png_filename


def experiment(args):
    """主实验函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型（应该使用CFM训练的模型）
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    print("Note: Model should be trained with CFM method")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    train_args = checkpoint.get('args', {})
    model_config = checkpoint.get('model_config', {})
    dataset_name = train_args.get('dataset', 'cifar10')
    sigma = 0.0  # Set sigma to 0 for experiment
    method = train_args.get('method', 'cfm')
    
    if method != 'cfm':
        print(f"Warning: Model was trained with method '{method}', but this experiment expects CFM model.")
        print("Proceeding anyway, but results may not be meaningful.")
    
    # 创建模型
    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # 获取数据集
    if dataset_name in ['cifar10', 'mnist']:
        dataloader, _ = get_dataset(dataset_name, args.batch_size, args.data_dir)
    else:
        raise ValueError("This experiment currently only supports image datasets (cifar10, mnist)")
    
    print(f"Dataset: {dataset_name}, Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    
    # 创建flow matcher（用于计算xt）
    flow_matcher = create_flow_matcher('cfm', sigma)  # 使用CFM的flow matcher来计算xt
    
    # 定义配对方法（包括ma_incepcfm和mac_cfm）
    # 暂时注释掉 ma_incepcfm，因为需要 pytorch_ood
    pairing_methods = ['otcfm', 'ma_tcfm', 'dct_otcfm_4x4', 'dct_otcfm_8x8', 'dct_otcfm_16x16', 'dct_hf_otcfm_4x4', 'dct_hf_otcfm_8x8', 'dct_hf_otcfm_16x16', 'dct_mf_otcfm_4x4', 'dct_mf_otcfm_8x8', 'cfm']  # OTCFM在前，CFM在最后（最右边）
    # pairing_methods = ['otcfm', 'ma_tcfm', 'ma_incepcfm', 'bary_otcfm', 'mac_cfm', 'cfm']  # 如果需要 ma_incepcfm，需要安装 pytorch_ood
    method_labels = {
        'cfm': 'CFM (Random)',
        'otcfm': 'OTCFM (OT)',
        'ma_tcfm': 'MA_TCFM (2x)',
        'ma_incepcfm': 'MA_IncepCFM',
        'bary_otcfm': 'Bary_OTCFM',
        'mac_cfm': 'MAC_CFM',
        'dct_otcfm': 'DCT_OTCFM',
        'dct_otcfm_4x4': 'DCT_OTCFM (4x4)',
        'dct_otcfm_8x8': 'DCT_OTCFM (8x8)',
        'dct_otcfm_16x16': 'DCT_OTCFM (16x16)',
        'dct_hf_otcfm_4x4': 'DCT_HF_OTCFM (4x4)',
        'dct_hf_otcfm_8x8': 'DCT_HF_OTCFM (8x8)',
        'dct_hf_otcfm_16x16': 'DCT_HF_OTCFM (16x16)',
        'dct_mf_otcfm_4x4': 'DCT_MF_OTCFM (4x4)',
        'dct_mf_otcfm_8x8': 'DCT_MF_OTCFM (8x8)'
    }
    colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'orange', 'brown', 'pink', 'purple', 'yellow', 'gray']  # 添加了中间频率DCT的颜色
    # colors = ['green', 'red', 'blue', 'purple', 'orange', 'gray']  # 如果使用 ma_incepcfm，使用这个
    
    # 构建文件名前缀
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = f"{method}_{dataset_name}_bs{args.batch_size}"
    
    # 保存所有结果
    all_results = {}  # {t_value: {method: cos_similarities}}
    
    # 对每个固定的t值进行计算
    for t_value in FIXED_T_VALUES:
        print(f"\n{'='*60}")
        print(f"Processing t = {t_value}")
        print(f"{'='*60}")
        
        # 收集所有batch的结果
        all_cos_similarities = {method: [] for method in pairing_methods}
        
        # 多次采样新的(x0, x1) batch
        for batch_idx in range(args.num_batches):
            print(f"\nBatch {batch_idx + 1}/{args.num_batches}:")
            # 获取新的batch数据
            # Note: x1 已经通过 dataset.get_dataset() 进行了归一化处理
            # CIFAR-10: Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            # MNIST: Normalize((0.5,), (0.5,))
            # 这与训练和推理时的归一化一致
            x1 = next(iter(dataloader))[0].to(device)
            x0 = torch.randn_like(x1)  # 标准高斯噪声，与训练时一致
            
            # 预采样一个epsilon，确保所有配对方法使用相同的数据
            epsilon = torch.randn_like(x0)
            
            for pairing_method in pairing_methods:
                print(f"  Processing {pairing_method}...", end=' ', flush=True)
                errors = compute_pairing_error_at_t(
                    model, flow_matcher, x0, x1, pairing_method, device, 
                    t_value, epsilon, metric=args.metric
                )
                all_cos_similarities[pairing_method].extend(errors.tolist())
                print(f"Done (mean={errors.mean():.4f})")
        
        # 合并所有batch的结果
        results = {method: np.array(all_cos_similarities[method]) for method in pairing_methods}
        
        metric_label = 'Cosine Similarity' if args.metric == 'cos' else 'L2 Distance'
        for pairing_method in pairing_methods:
            print(f"\n{pairing_method} (all batches):")
            print(f"  Total samples: {len(results[pairing_method])}")
            print(f"  Mean {metric_label.lower()}: {results[pairing_method].mean():.4f}")
            print(f"  Std {metric_label.lower()}: {results[pairing_method].std():.4f}")
            print(f"  Min {metric_label.lower()}: {results[pairing_method].min():.4f}")
            print(f"  Max {metric_label.lower()}: {results[pairing_method].max():.4f}")
        
        all_results[t_value] = results
        
        # 为当前t值绘制对比图
        png_filename = plot_results_for_t(
            t_value, results, output_dir, model_info, 
            pairing_methods, method_labels, colors, metric=args.metric
        )
        print(f"\nVisualization saved: {png_filename}")
    
    # 保存所有数值结果
    npz_filename = f"pairing_error_results_{model_info}_fixed_t_{args.metric}.npz"
    save_dict = {}
    for t_value in FIXED_T_VALUES:
        for method in pairing_methods:
            key = f"t{t_value:.2f}_{method}"
            save_dict[key] = all_results[t_value][method]
    
    np.savez(output_dir / npz_filename, **save_dict)
    print(f"\nNumerical results saved: {npz_filename}")
    
    # 打印统计摘要
    print("\n" + "="*60)
    print("Statistical Summary for all t values:")
    print("="*60)
    for t_value in FIXED_T_VALUES:
        print(f"\nt = {t_value}:")
        for method, values in all_results[t_value].items():
            print(f"  {method.upper()}:")
            print(f"    Mean:   {values.mean():.6f}")
            print(f"    Std:    {values.std():.6f}")
            print(f"    Median: {np.median(values):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment: Compare pairing error at fixed time t")
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for sampling (number of noise-image pairs)')
    
    # Number of batches to sample (for repeated experiments)
    parser.add_argument('--num_batches', type=int, default=1,
                       help='Number of batches to sample (each batch uses new images and noise, allowing repeated experiments)')
    
    # Metric type
    parser.add_argument('--metric', type=str, default='cos', choices=['cos', 'l2'],
                       help='Metric type: "cos" for cosine similarity, "l2" for L2 distance')
    
    # Data directory
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    experiment(args)
