"""
比较 CIFAR-10 图像和标准高斯噪声的 L2 范数
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import get_dataset

def main():
    # 加载 CIFAR-10 数据集
    print("Loading CIFAR-10 dataset...")
    train_loader, _ = get_dataset(
        dataset_name='cifar10',
        batch_size=50,
        data_dir='./data'
    )
    
    # 获取一个 batch 的图像（50张）
    x1_batch = next(iter(train_loader))[0]  # (50, 3, 32, 32)
    
    # CIFAR-10 图像已经归一化到 [-1, 1]
    print(f"Image batch shape: {x1_batch.shape}")
    print(f"Image value range: [{x1_batch.min().item():.3f}, {x1_batch.max().item():.3f}]")
    
    # 计算每张图像的 L2 范数（展平后）
    x1_flat = x1_batch.reshape(x1_batch.shape[0], -1)  # (50, 3*32*32)
    image_norms = torch.norm(x1_flat, dim=1)  # (50,)
    
    # 生成对应数量的标准高斯噪声
    x0_batch = torch.randn_like(x1_batch)  # (50, 3, 32, 32)
    x0_flat = x0_batch.reshape(x0_batch.shape[0], -1)  # (50, 3*32*32)
    noise_norms = torch.norm(x0_flat, dim=1)  # (50,)
    
    # 转换为 numpy 用于统计分析
    image_norms_np = image_norms.cpu().numpy()
    noise_norms_np = noise_norms.cpu().numpy()
    
    # 统计信息
    print("\n" + "="*60)
    print("L2 Norm Statistics (50 samples)")
    print("="*60)
    print(f"\nImage norms:")
    print(f"  Mean: {image_norms_np.mean():.4f}")
    print(f"  Std:  {image_norms_np.std():.4f}")
    print(f"  Min:  {image_norms_np.min():.4f}")
    print(f"  Max:  {image_norms_np.max():.4f}")
    
    print(f"\nGaussian noise norms (standard normal):")
    print(f"  Mean: {noise_norms_np.mean():.4f}")
    print(f"  Std:  {noise_norms_np.std():.4f}")
    print(f"  Min:  {noise_norms_np.min():.4f}")
    print(f"  Max:  {noise_norms_np.max():.4f}")
    
    # 理论值：对于 d 维标准高斯噪声，L2范数的期望
    d = 3 * 32 * 32  # 3072 维
    theoretical_noise_mean = np.sqrt(d)  # 对于标准高斯，E[||x||] ≈ sqrt(d)
    theoretical_noise_std = np.sqrt(d / 2)  # 对于标准高斯，Std[||x||] ≈ sqrt(d/2)
    
    print(f"\nTheoretical Gaussian noise norm (d={d}):")
    print(f"  E[||x||] ≈ sqrt(d) = {theoretical_noise_mean:.4f}")
    print(f"  Std[||x||] ≈ sqrt(d/2) = {theoretical_noise_std:.4f}")
    
    # 比较
    ratio = image_norms_np / noise_norms_np
    print(f"\nRatio (Image norm / Noise norm):")
    print(f"  Mean: {ratio.mean():.4f}")
    print(f"  Std:  {ratio.std():.4f}")
    print(f"  Image norm is {ratio.mean():.2f}x of noise norm on average")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 直方图对比
    axes[0, 0].hist(image_norms_np, bins=20, alpha=0.7, label='CIFAR-10 Images', color='blue', edgecolor='black')
    axes[0, 0].hist(noise_norms_np, bins=20, alpha=0.7, label='Gaussian Noise', color='red', edgecolor='black')
    axes[0, 0].axvline(image_norms_np.mean(), color='blue', linestyle='--', linewidth=2, label=f'Image mean: {image_norms_np.mean():.2f}')
    axes[0, 0].axvline(noise_norms_np.mean(), color='red', linestyle='--', linewidth=2, label=f'Noise mean: {noise_norms_np.mean():.2f}')
    axes[0, 0].axvline(theoretical_noise_mean, color='green', linestyle=':', linewidth=2, label=f'Theoretical: {theoretical_noise_mean:.2f}')
    axes[0, 0].set_xlabel('L2 Norm', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of L2 Norms', fontsize=13)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 散点图：图像范数 vs 噪声范数
    axes[0, 1].scatter(noise_norms_np, image_norms_np, alpha=0.6, s=50)
    # 添加对角线 y=x
    min_val = min(noise_norms_np.min(), image_norms_np.min())
    max_val = max(noise_norms_np.max(), image_norms_np.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (equal norm)')
    axes[0, 1].set_xlabel('Gaussian Noise Norm', fontsize=12)
    axes[0, 1].set_ylabel('CIFAR-10 Image Norm', fontsize=12)
    axes[0, 1].set_title('Image Norm vs Noise Norm', fontsize=13)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 箱线图对比
    data_to_plot = [image_norms_np, noise_norms_np]
    bp = axes[1, 0].boxplot(data_to_plot, labels=['CIFAR-10\nImages', 'Gaussian\nNoise'], 
                            patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1, 0].set_ylabel('L2 Norm', fontsize=12)
    axes[1, 0].set_title('Boxplot Comparison', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 比率分布
    axes[1, 1].hist(ratio, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(ratio.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {ratio.mean():.3f}')
    axes[1, 1].axvline(1.0, color='green', linestyle=':', linewidth=2, label='Ratio = 1.0')
    axes[1, 1].set_xlabel('Ratio (Image Norm / Noise Norm)', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Norm Ratio', fontsize=13)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path("exp").mkdir(parents=True, exist_ok=True)
    plt.savefig('exp/image_noise_norm_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: exp/image_noise_norm_comparison.png")
    plt.close()
    
    # 额外的统计：按通道分析
    print("\n" + "="*60)
    print("Per-channel norm analysis (for images)")
    print("="*60)
    for c in range(3):
        channel_data = x1_batch[:, c, :, :]  # (50, 32, 32)
        channel_flat = channel_data.reshape(channel_data.shape[0], -1)  # (50, 1024)
        channel_norms = torch.norm(channel_flat, dim=1)  # (50,)
        print(f"Channel {c}: Mean norm = {channel_norms.mean().item():.4f}, Std = {channel_norms.std().item():.4f}")


if __name__ == '__main__':
    main()
