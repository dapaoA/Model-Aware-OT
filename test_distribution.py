"""
测试脚本：检查CIFAR10图片数据和随机噪音的分布统计特性
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import get_dataset


def compute_statistics(data, name="Data"):
    """计算数据的统计特性"""
    data_np = data.cpu().numpy()
    
    # 整体统计
    mean = np.mean(data_np)
    std = np.std(data_np)
    var = np.var(data_np)
    min_val = np.min(data_np)
    max_val = np.max(data_np)
    
    # 每个通道的统计（如果是图像数据）
    if len(data.shape) == 4:  # [B, C, H, W]
        channel_means = []
        channel_stds = []
        for c in range(data.shape[1]):
            channel_data = data_np[:, c, :, :]
            channel_means.append(np.mean(channel_data))
            channel_stds.append(np.std(channel_data))
    else:
        channel_means = None
        channel_stds = None
    
    print(f"\n{'='*60}")
    print(f"{name} 统计特性:")
    print(f"{'='*60}")
    print(f"整体均值 (mean): {mean:.6f}")
    print(f"整体标准差 (std): {std:.6f}")
    print(f"整体方差 (var): {var:.6f}")
    print(f"最小值: {min_val:.6f}")
    print(f"最大值: {max_val:.6f}")
    
    if channel_means is not None:
        print(f"\n每个通道的统计:")
        for c in range(len(channel_means)):
            print(f"  通道 {c}: mean={channel_means[c]:.6f}, std={channel_stds[c]:.6f}")
    
    return {
        'mean': mean,
        'std': std,
        'var': var,
        'min': min_val,
        'max': max_val,
        'channel_means': channel_means,
        'channel_stds': channel_stds,
        'data': data_np
    }


def plot_histogram(data_stats, noise_stats, save_path='distribution_comparison.png'):
    """绘制分布直方图对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 整体分布对比
    ax = axes[0, 0]
    ax.hist(data_stats['data'].flatten(), bins=100, alpha=0.6, label='CIFAR10图片', density=True)
    ax.hist(noise_stats['data'].flatten(), bins=100, alpha=0.6, label='随机噪音', density=True)
    ax.set_xlabel('值')
    ax.set_ylabel('密度')
    ax.set_title('整体分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 如果有多通道，显示每个通道的分布
    if data_stats['channel_means'] is not None:
        for c in range(len(data_stats['channel_means'])):
            ax = axes[0, 1] if c == 0 else (axes[1, 0] if c == 1 else axes[1, 1])
            data_channel = data_stats['data'][:, c, :, :].flatten()
            noise_channel = noise_stats['data'][:, c, :, :].flatten()
            ax.hist(data_channel, bins=100, alpha=0.6, label=f'CIFAR10 通道{c}', density=True)
            ax.hist(noise_channel, bins=100, alpha=0.6, label=f'噪音 通道{c}', density=True)
            ax.set_xlabel('值')
            ax.set_ylabel('密度')
            ax.set_title(f'通道 {c} 分布对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n分布对比图已保存到: {save_path}")
    plt.close()


def test_distribution():
    """主测试函数"""
    print("开始测试CIFAR10数据和随机噪音的分布...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据集（使用和训练脚本相同的参数）
    batch_size = 256
    num_samples = 10000  # 采样数量
    
    print(f"\n从数据集中采样 {num_samples} 张图片...")
    dataloader, _ = get_dataset('cifar10', batch_size=batch_size, data_dir='./data')
    
    # 收集图片数据
    images_list = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        x1 = batch[0].to(device)
        images_list.append(x1)
    
    images = torch.cat(images_list, dim=0)[:num_samples]
    print(f"已收集 {images.shape[0]} 张图片，形状: {images.shape}")
    
    # 生成对应的随机噪音
    print(f"\n生成相同形状的随机噪音...")
    noise = torch.randn_like(images[:num_samples])
    print(f"噪音形状: {noise.shape}")
    
    # 计算统计特性
    images_stats = compute_statistics(images, "CIFAR10图片 (归一化后)")
    noise_stats = compute_statistics(noise, "随机噪音 (标准正态分布)")
    
    # 对比分析
    print(f"\n{'='*60}")
    print("对比分析:")
    print(f"{'='*60}")
    print(f"图片均值 vs 噪音均值: {images_stats['mean']:.6f} vs {noise_stats['mean']:.6f} (期望接近0)")
    print(f"图片标准差 vs 噪音标准差: {images_stats['std']:.6f} vs {noise_stats['std']:.6f} (噪音期望接近1)")
    print(f"图片方差 vs 噪音方差: {images_stats['var']:.6f} vs {noise_stats['var']:.6f} (噪音期望接近1)")
    
    if images_stats['channel_means'] is not None:
        print(f"\n各通道对比:")
        for c in range(len(images_stats['channel_means'])):
            img_mean = images_stats['channel_means'][c]
            noise_mean = noise_stats['channel_means'][c]
            img_std = images_stats['channel_stds'][c]
            noise_std = noise_stats['channel_stds'][c]
            print(f"  通道 {c}:")
            print(f"    均值: {img_mean:.6f} vs {noise_mean:.6f}")
            print(f"    标准差: {img_std:.6f} vs {noise_std:.6f}")
    
    # 检查噪音是否接近标准正态分布
    print(f"\n{'='*60}")
    print("随机噪音分布验证:")
    print(f"{'='*60}")
    noise_mean_diff = abs(noise_stats['mean'] - 0.0)
    noise_std_diff = abs(noise_stats['std'] - 1.0)
    print(f"噪音均值与0的差异: {noise_mean_diff:.6f} (应该很小)")
    print(f"噪音标准差与1的差异: {noise_std_diff:.6f} (应该很小)")
    
    if noise_mean_diff < 0.1 and noise_std_diff < 0.1:
        print("✓ 随机噪音符合标准正态分布 (N(0,1))")
    else:
        print("✗ 警告: 随机噪音可能不符合标准正态分布")
    
    # 绘制分布对比图
    print(f"\n生成分布对比图...")
    plot_histogram(images_stats, noise_stats)
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_distribution()
