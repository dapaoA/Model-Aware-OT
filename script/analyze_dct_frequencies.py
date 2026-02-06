"""
分析DCT变换后不同频率分量的统计信息（范围、均值、方差）
比较噪声和CIFAR-10图片的DCT频率分布
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.fft import dctn
from pathlib import Path
from dataset import get_dataset


def _get_zigzag_indices(h, w):
    """获取zigzag扫描索引（缓存）"""
    cache_key = (h, w)
    if not hasattr(_get_zigzag_indices, '_cache'):
        _get_zigzag_indices._cache = {}
    
    if cache_key not in _get_zigzag_indices._cache:
        total_coeffs = h * w
        indices = []
        i, j = 0, 0
        direction = 1  # 1: 向右上, -1: 向左下
        
        while len(indices) < total_coeffs and (i < h and j < w):
            indices.append((i, j))
            
            if direction == 1:  # 向右上移动
                if i == 0 or j == w - 1:
                    if j == w - 1:
                        i += 1
                    else:
                        j += 1
                    direction = -1
                else:
                    i -= 1
                    j += 1
            else:  # 向左下移动
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
    """将2D DCT矩阵按zigzag扫描顺序转换为1D数组"""
    h, w = dct_2d.shape
    indices = _get_zigzag_indices(h, w)
    dct_1d = np.array([dct_2d[h_idx, w_idx] for (h_idx, w_idx) in indices])
    return dct_1d


def extract_dct_frequencies(x, h=32, w=32):
    """
    对图像进行DCT变换，并按zigzag顺序提取不同频率分量
    同时保存完整的2D DCT矩阵用于可视化
    
    Args:
        x: 图像tensor (B, C, H, W) 或单个图像 (C, H, W)
        h, w: 图像高度和宽度
    
    Returns:
        dict: 包含低频、中频、高频系数的字典
            - 'low': 低频系数 (前16个，4x4) - 1D array
            - 'mid': 中频系数 (中间部分) - 1D array
            - 'high': 高频系数 (最后16个) - 1D array
            - 'all': 所有系数（按zigzag顺序）- 1D array
            - 'dct_2d_all': 所有2D DCT矩阵的列表，用于计算mean和var
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)  # (1, C, H, W)
    
    B, C, H, W = x.shape
    assert H == h and W == w, f"Image size mismatch: expected ({h}, {w}), got ({H}, {W})"
    
    total_coeffs = H * W
    low_freq_num = 16  # 4x4 = 16
    high_freq_num = 16  # 最后16个
    mid_freq_num = min(64, total_coeffs - low_freq_num - high_freq_num)  # 中间64个
    
    # 定义频率范围
    low_start = 0
    low_end = low_freq_num
    mid_start = total_coeffs // 4  # 从25%开始
    mid_end = mid_start + mid_freq_num
    high_start = total_coeffs - high_freq_num
    high_end = total_coeffs
    
    all_low = []
    all_mid = []
    all_high = []
    all_all = []
    dct_2d_all = []  # 保存所有2D DCT矩阵
    
    x_np = x.cpu().numpy()
    
    for b in range(B):
        for c in range(C):
            # 提取单个通道
            x_channel = x_np[b, c, :, :]  # (H, W)
            
            # DCT变换
            dct_2d = dctn(x_channel, norm='ortho')  # (H, W)
            dct_2d_all.append(dct_2d)  # 保存2D矩阵
            
            # 转换为zigzag顺序
            dct_1d = _zigzag_to_1d(dct_2d)  # (H*W,)
            
            # 提取不同频率分量
            dct_low = dct_1d[low_start:low_end]
            dct_mid = dct_1d[mid_start:mid_end]
            dct_high = dct_1d[high_start:high_end]
            
            all_low.append(dct_low)
            all_mid.append(dct_mid)
            all_high.append(dct_high)
            all_all.append(dct_1d)
    
    return {
        'low': np.concatenate(all_low) if all_low else np.array([]),
        'mid': np.concatenate(all_mid) if all_mid else np.array([]),
        'high': np.concatenate(all_high) if all_high else np.array([]),
        'all': np.concatenate(all_all) if all_all else np.array([]),
        'dct_2d_all': dct_2d_all,  # List of 2D DCT matrices
    }


def compute_statistics(data_dict):
    """计算统计信息"""
    stats = {}
    for freq_name, values in data_dict.items():
        # Skip 'dct_2d_all' as it's a list of 2D matrices, not a 1D array
        if freq_name == 'dct_2d_all':
            continue
        if isinstance(values, np.ndarray) and len(values) > 0:
            stats[freq_name] = {
                'min': np.min(values),
                'max': np.max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'var': np.var(values),
                'abs_mean': np.mean(np.abs(values)),
                'abs_max': np.max(np.abs(values)),
            }
        else:
            stats[freq_name] = None
    return stats


def plot_frequency_comparison(noise_stats, cifar_stats, output_path):
    """绘制频率分量对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    freq_names = ['low', 'mid', 'high']
    freq_labels = ['Low Freq (4x4)', 'Mid Freq', 'High Freq (Last 16)']
    
    # First row: Noise
    for idx, (freq_name, freq_label) in enumerate(zip(freq_names, freq_labels)):
        ax = axes[0, idx]
        if noise_stats[freq_name] is not None:
            stats = noise_stats[freq_name]
            ax.text(0.5, 0.9, f'Noise - {freq_label}', 
                   transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
            ax.text(0.1, 0.8, f'Range: [{stats["min"]:.2f}, {stats["max"]:.2f}]', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.7, f'Mean: {stats["mean"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.6, f'Std: {stats["std"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.5, f'Var: {stats["var"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.4, f'Abs Mean: {stats["abs_mean"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.3, f'Abs Max: {stats["abs_max"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
        ax.axis('off')
    
    # Second row: CIFAR-10
    for idx, (freq_name, freq_label) in enumerate(zip(freq_names, freq_labels)):
        ax = axes[1, idx]
        if cifar_stats[freq_name] is not None:
            stats = cifar_stats[freq_name]
            ax.text(0.5, 0.9, f'CIFAR-10 - {freq_label}', 
                   transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
            ax.text(0.1, 0.8, f'Range: [{stats["min"]:.2f}, {stats["max"]:.2f}]', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.7, f'Mean: {stats["mean"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.6, f'Std: {stats["std"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.5, f'Var: {stats["var"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.4, f'Abs Mean: {stats["abs_mean"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.3, f'Abs Max: {stats["abs_max"]:.4f}', 
                   transform=ax.transAxes, fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_frequency_distribution(noise_data, cifar_data, output_path):
    """绘制频率分量的分布对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    freq_names = ['low', 'mid', 'high']
    freq_labels = ['Low Freq (4x4)', 'Mid Freq', 'High Freq (Last 16)']
    
    for idx, (freq_name, freq_label) in enumerate(zip(freq_names, freq_labels)):
        ax = axes[idx]
        
        if len(noise_data[freq_name]) > 0 and len(cifar_data[freq_name]) > 0:
            # Plot histograms
            ax.hist(noise_data[freq_name], bins=50, alpha=0.5, label='Noise', 
                   color='blue', density=True)
            ax.hist(cifar_data[freq_name], bins=50, alpha=0.5, label='CIFAR-10', 
                   color='red', density=True)
            
            ax.set_xlabel('DCT Coefficient Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'{freq_label} Distribution Comparison', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {output_path}")
    plt.close()


def plot_frequency_magnitude_comparison(noise_stats, cifar_stats, output_path):
    """绘制频率分量幅值对比图（柱状图）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    freq_names = ['low', 'mid', 'high']
    freq_labels = ['Low Freq', 'Mid Freq', 'High Freq']
    
    # Absolute mean comparison
    ax = axes[0]
    noise_abs_means = [noise_stats[f]['abs_mean'] if noise_stats[f] is not None else 0 
                       for f in freq_names]
    cifar_abs_means = [cifar_stats[f]['abs_mean'] if cifar_stats[f] is not None else 0 
                       for f in freq_names]
    
    x = np.arange(len(freq_labels))
    width = 0.35
    ax.bar(x - width/2, noise_abs_means, width, label='Noise', color='blue', alpha=0.7)
    ax.bar(x + width/2, cifar_abs_means, width, label='CIFAR-10', color='red', alpha=0.7)
    ax.set_xlabel('Frequency Component', fontsize=12)
    ax.set_ylabel('Absolute Mean', fontsize=12)
    ax.set_title('Absolute Mean Comparison by Frequency', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(freq_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Absolute max comparison
    ax = axes[1]
    noise_abs_maxs = [noise_stats[f]['abs_max'] if noise_stats[f] is not None else 0 
                      for f in freq_names]
    cifar_abs_maxs = [cifar_stats[f]['abs_max'] if cifar_stats[f] is not None else 0 
                      for f in freq_names]
    
    ax.bar(x - width/2, noise_abs_maxs, width, label='Noise', color='blue', alpha=0.7)
    ax.bar(x + width/2, cifar_abs_maxs, width, label='CIFAR-10', color='red', alpha=0.7)
    ax.set_xlabel('Frequency Component', fontsize=12)
    ax.set_ylabel('Absolute Max', fontsize=12)
    ax.set_title('Absolute Max Comparison by Frequency', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(freq_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved magnitude comparison plot to {output_path}")
    plt.close()


def plot_dct_matrix_mean_var(noise_data, cifar_data, output_path, h=32, w=32):
    """将mean和var可视化为2D DCT矩阵热图"""
    # 计算每个位置的mean和var
    noise_dct_list = noise_data.get('dct_2d_all', [])
    cifar_dct_list = cifar_data.get('dct_2d_all', [])
    
    if len(noise_dct_list) == 0 or len(cifar_dct_list) == 0:
        print("Warning: No 2D DCT matrices found, skipping matrix visualization")
        return
    
    # 堆叠所有2D DCT矩阵
    noise_dct_stack = np.stack(noise_dct_list, axis=0)  # (N, H, W)
    cifar_dct_stack = np.stack(cifar_dct_list, axis=0)  # (N, H, W)
    
    # 计算mean和var
    noise_mean = np.mean(noise_dct_stack, axis=0)  # (H, W)
    noise_var = np.var(noise_dct_stack, axis=0)   # (H, W)
    cifar_mean = np.mean(cifar_dct_stack, axis=0)  # (H, W)
    cifar_var = np.var(cifar_dct_stack, axis=0)    # (H, W)
    
    # 创建4个子图：noise mean, noise var, cifar mean, cifar var
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Noise Mean
    im1 = axes[0, 0].imshow(noise_mean, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Noise - Mean of DCT Coefficients', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Frequency (W)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency (H)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Noise Variance
    im2 = axes[0, 1].imshow(noise_var, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Noise - Variance of DCT Coefficients', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Frequency (W)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency (H)', fontsize=12)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # CIFAR-10 Mean
    im3 = axes[1, 0].imshow(cifar_mean, cmap='viridis', aspect='auto')
    axes[1, 0].set_title('CIFAR-10 - Mean of DCT Coefficients', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Frequency (W)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency (H)', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0])
    
    # CIFAR-10 Variance
    im4 = axes[1, 1].imshow(cifar_var, cmap='plasma', aspect='auto')
    axes[1, 1].set_title('CIFAR-10 - Variance of DCT Coefficients', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Frequency (W)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency (H)', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved DCT matrix mean/var visualization to {output_path}")
    plt.close()
    
    # 打印一些统计信息
    print(f"\nNoise Mean - Min: {noise_mean.min():.4f}, Max: {noise_mean.max():.4f}, "
          f"Mean: {noise_mean.mean():.4f}, Std: {noise_mean.std():.4f}")
    print(f"Noise Var - Min: {noise_var.min():.4f}, Max: {noise_var.max():.4f}, "
          f"Mean: {noise_var.mean():.4f}, Std: {noise_var.std():.4f}")
    print(f"CIFAR-10 Mean - Min: {cifar_mean.min():.4f}, Max: {cifar_mean.max():.4f}, "
          f"Mean: {cifar_mean.mean():.4f}, Std: {cifar_mean.std():.4f}")
    print(f"CIFAR-10 Var - Min: {cifar_var.min():.4f}, Max: {cifar_var.max():.4f}, "
          f"Mean: {cifar_var.mean():.4f}, Std: {cifar_var.std():.4f}")


def main():
    print("="*60)
    print("DCT频率分量统计分析")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 生成100个噪声图片（使用与CIFAR-10相同的归一化）
    print("\n生成100个噪声图片...")
    # CIFAR-10归一化: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
    # 归一化后的噪声应该是标准正态分布
    noise = torch.randn(100, 3, 32, 32).to(device)
    print(f"噪声形状: {noise.shape}")
    
    # 2. 加载100个CIFAR-10图片
    print("\n加载100个CIFAR-10图片...")
    dataloader, _ = get_dataset('cifar10', batch_size=100, data_dir='./data')
    cifar_batch = next(iter(dataloader))[0].to(device)  # 取第一个batch
    print(f"CIFAR-10形状: {cifar_batch.shape}")
    
    # 3. 提取DCT频率分量
    print("\n提取噪声的DCT频率分量...")
    noise_freqs = extract_dct_frequencies(noise, h=32, w=32)
    
    print("\n提取CIFAR-10的DCT频率分量...")
    cifar_freqs = extract_dct_frequencies(cifar_batch, h=32, w=32)
    
    # 4. 计算统计信息
    print("\n计算统计信息...")
    noise_stats = compute_statistics(noise_freqs)
    cifar_stats = compute_statistics(cifar_freqs)
    
    # 5. 打印统计信息
    print("\n" + "="*60)
    print("噪声统计信息:")
    print("="*60)
    for freq_name in ['low', 'mid', 'high']:
        if noise_stats[freq_name] is not None:
            stats = noise_stats[freq_name]
            print(f"\n{freq_name.upper()} (低频/中频/高频):")
            print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  均值: {stats['mean']:.4f}")
            print(f"  标准差: {stats['std']:.4f}")
            print(f"  方差: {stats['var']:.4f}")
            print(f"  绝对均值: {stats['abs_mean']:.4f}")
            print(f"  绝对最大值: {stats['abs_max']:.4f}")
    
    print("\n" + "="*60)
    print("CIFAR-10统计信息:")
    print("="*60)
    for freq_name in ['low', 'mid', 'high']:
        if cifar_stats[freq_name] is not None:
            stats = cifar_stats[freq_name]
            print(f"\n{freq_name.upper()} (低频/中频/高频):")
            print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  均值: {stats['mean']:.4f}")
            print(f"  标准差: {stats['std']:.4f}")
            print(f"  方差: {stats['var']:.4f}")
            print(f"  绝对均值: {stats['abs_mean']:.4f}")
            print(f"  绝对最大值: {stats['abs_max']:.4f}")
    
    # 6. 绘制对比图
    output_dir = Path('./exp/dct_frequency_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n绘制对比图...")
    plot_frequency_comparison(noise_stats, cifar_stats, 
                             output_dir / 'frequency_statistics_comparison.png')
    plot_frequency_distribution(noise_freqs, cifar_freqs, 
                               output_dir / 'frequency_distribution_comparison.png')
    plot_frequency_magnitude_comparison(noise_stats, cifar_stats, 
                                       output_dir / 'frequency_magnitude_comparison.png')
    
    # 7. 绘制DCT矩阵的mean和var热图
    print("\n绘制DCT矩阵mean和var热图...")
    plot_dct_matrix_mean_var(noise_freqs, cifar_freqs, 
                             output_dir / 'dct_matrix_mean_var.png', h=32, w=32)
    
    print("\n" + "="*60)
    print("分析完成！结果保存在:", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()
