"""
测试脚本：可视化 DCT 压缩前后的 CIFAR-10 图像差异
展示不同低频保留大小（4x4, 8x8, 16x16）的重建效果
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import dctn, idctn
from dataset import get_dataset

def dct_compress_channel(x_channel, low_freq_size):
    """对单个通道进行 DCT 压缩和重建"""
    # x_channel: (H, W) numpy array
    # DCT 变换
    dct = dctn(x_channel, norm='ortho')
    
    # 保留低频部分（左上角）
    dct_compressed = np.zeros_like(dct)
    dct_compressed[:low_freq_size, :low_freq_size] = dct[:low_freq_size, :low_freq_size]
    
    # 反 DCT 变换重建
    reconstructed = idctn(dct_compressed, norm='ortho')
    
    return reconstructed, dct, dct_compressed

def dct_compress_image(x, low_freq_size):
    """对整张图像（3通道）进行 DCT 压缩"""
    # x: (C, H, W) numpy array or torch tensor
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    
    C, H, W = x.shape
    reconstructed = np.zeros_like(x)
    
    for c in range(C):
        x_channel = x[c, :, :]
        recon_channel, _, _ = dct_compress_channel(x_channel, low_freq_size)
        reconstructed[c, :, :] = recon_channel
    
    return reconstructed

def main():
    Path("exp").mkdir(parents=True, exist_ok=True)
    # 加载 CIFAR-10 数据集
    print("Loading CIFAR-10 dataset...")
    train_loader, _ = get_dataset(
        dataset_name='cifar10',
        batch_size=16,
        data_dir='./data'
    )
    
    # 获取一个 batch 的图像
    x1 = next(iter(train_loader))[0]  # (B, C, H, W)
    
    # CIFAR-10 图像范围是 [-1, 1]，需要转换到 [0, 1] 用于显示
    def denormalize(x):
        """将 [-1, 1] 范围的图像转换到 [0, 1]"""
        return (x + 1) / 2
    
    def normalize(x):
        """将 [0, 1] 范围的图像转换到 [-1, 1]"""
        return x * 2 - 1
    
    # 选择几张图像展示
    num_images = 8
    low_freq_sizes = [4, 8, 16]
    
    # 创建大图
    fig, axes = plt.subplots(len(low_freq_sizes) + 1, num_images, figsize=(20, 10))
    
    # 对每张图像进行处理
    for img_idx in range(num_images):
        img = x1[img_idx]  # (C, H, W)
        
        # 第一行：原始图像
        img_display = denormalize(img).permute(1, 2, 0).numpy()
        img_display = np.clip(img_display, 0, 1)
        axes[0, img_idx].imshow(img_display)
        axes[0, img_idx].axis('off')
        if img_idx == 0:
            axes[0, img_idx].set_ylabel('Original', fontsize=12, rotation=0, ha='right')
        
        # 转换到 numpy 进行处理
        img_np = img.numpy()  # (C, H, W)
        
        # 对每个 low_freq_size 进行压缩重建
        for row_idx, low_freq_size in enumerate(low_freq_sizes):
            # DCT 压缩重建
            reconstructed = dct_compress_image(img_np, low_freq_size)
            
            # 转换回 torch tensor 并 denormalize
            reconstructed_tensor = torch.from_numpy(reconstructed)
            recon_display = denormalize(reconstructed_tensor).permute(1, 2, 0).numpy()
            recon_display = np.clip(recon_display, 0, 1)
            
            # 显示重建图像
            axes[row_idx + 1, img_idx].imshow(recon_display)
            axes[row_idx + 1, img_idx].axis('off')
            
            # 计算重建误差（MSE）
            mse = np.mean((img_np - reconstructed) ** 2)
            
            if img_idx == 0:
                axes[row_idx + 1, img_idx].set_ylabel(f'DCT {low_freq_size}x{low_freq_size}\nMSE={mse:.4f}', 
                                                      fontsize=10, rotation=0, ha='right')
            else:
                axes[row_idx + 1, img_idx].set_title(f'MSE={mse:.4f}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('exp/dct_compression_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to: exp/dct_compression_visualization.png")
    plt.close()
    
    # 另外创建一个更详细的对比图，展示一张图像的细节
    print("\nCreating detailed comparison for one image...")
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    
    test_img = x1[0]  # 第一张图像
    test_img_np = test_img.numpy()
    test_img_display = denormalize(test_img).permute(1, 2, 0).numpy()
    test_img_display = np.clip(test_img_display, 0, 1)
    
    # 原始图像
    axes2[0, 0].imshow(test_img_display)
    axes2[0, 0].set_title('Original Image', fontsize=12)
    axes2[0, 0].axis('off')
    
    axes2[1, 0].imshow(test_img_display)
    axes2[1, 0].set_title('Original Image (Zoom)', fontsize=12)
    axes2[1, 0].axis('off')
    axes2[1, 0].set_xlim([8, 24])
    axes2[1, 0].set_ylim([8, 24])
    
    # 不同压缩级别的对比
    for col_idx, low_freq_size in enumerate([4, 8, 16]):
        reconstructed = dct_compress_image(test_img_np, low_freq_size)
        reconstructed_tensor = torch.from_numpy(reconstructed)
        recon_display = denormalize(reconstructed_tensor).permute(1, 2, 0).numpy()
        recon_display = np.clip(recon_display, 0, 1)
        
        mse = np.mean((test_img_np - reconstructed) ** 2)
        
        # 全图
        axes2[0, col_idx + 1].imshow(recon_display)
        axes2[0, col_idx + 1].set_title(f'DCT {low_freq_size}x{low_freq_size}\nMSE={mse:.4f}', fontsize=12)
        axes2[0, col_idx + 1].axis('off')
        
        # 局部放大
        axes2[1, col_idx + 1].imshow(recon_display)
        axes2[1, col_idx + 1].set_title(f'DCT {low_freq_size}x{low_freq_size} (Zoom)', fontsize=12)
        axes2[1, col_idx + 1].axis('off')
        axes2[1, col_idx + 1].set_xlim([8, 24])
        axes2[1, col_idx + 1].set_ylim([8, 24])
    
    plt.tight_layout()
    plt.savefig('exp/dct_compression_detailed.png', dpi=150, bbox_inches='tight')
    print("Detailed comparison saved to: exp/dct_compression_detailed.png")
    plt.close()
    
    print("\nCompression statistics:")
    print(f"Original image shape: {test_img.shape} (C, H, W)")
    print(f"Original pixel count: {np.prod(test_img.shape)}")
    
    for low_freq_size in [4, 8, 16]:
        preserved_coeffs = 3 * low_freq_size * low_freq_size  # 3 channels
        compression_ratio = preserved_coeffs / np.prod(test_img.shape) * 100
        reconstructed = dct_compress_image(test_img_np, low_freq_size)
        mse = np.mean((test_img_np - reconstructed) ** 2)
        print(f"\nDCT {low_freq_size}x{low_freq_size}:")
        print(f"  Preserved coefficients: {preserved_coeffs} / {np.prod(test_img.shape)} ({compression_ratio:.2f}%)")
        print(f"  Reconstruction MSE: {mse:.6f}")
    
    # 创建DCT系数的热力图可视化
    print("\nCreating DCT coefficient heatmaps...")
    fig3, axes3 = plt.subplots(3, 4, figsize=(16, 12))
    
    # 对三个通道分别可视化
    channel_names = ['Red', 'Green', 'Blue']
    for ch_idx in range(3):
        x_channel = test_img_np[ch_idx, :, :]  # (H, W)
        dct_full = dctn(x_channel, norm='ortho')
        
        # 计算DCT系数的统计信息
        abs_dct = np.abs(dct_full)
        log_abs_dct = np.log10(abs_dct + 1e-10)  # 使用对数刻度以便可视化
        
        # 第一列：原始DCT系数（绝对值，对数刻度）
        im1 = axes3[ch_idx, 0].imshow(log_abs_dct, cmap='hot', aspect='auto')
        axes3[ch_idx, 0].set_title(f'{channel_names[ch_idx]} Channel: Full DCT (log scale)', fontsize=10)
        axes3[ch_idx, 0].set_xlabel('Frequency (W)')
        axes3[ch_idx, 0].set_ylabel('Frequency (H)')
        plt.colorbar(im1, ax=axes3[ch_idx, 0], label='log10(|DCT|)')
        
        # 在左上角标注保留区域
        for low_freq_idx, low_freq_size in enumerate([4, 8, 16]):
            # 标注保留区域
            axes3[ch_idx, low_freq_idx + 1].imshow(log_abs_dct, cmap='hot', aspect='auto')
            
            # 绘制保留区域的边界
            axes3[ch_idx, low_freq_idx + 1].axhline(y=low_freq_size - 0.5, color='cyan', linewidth=2, linestyle='--', label='Preserved boundary')
            axes3[ch_idx, low_freq_idx + 1].axvline(x=low_freq_size - 0.5, color='cyan', linewidth=2, linestyle='--')
            
            # 用矩形框标出保留区域
            from matplotlib.patches import Rectangle
            rect = Rectangle((-0.5, -0.5), low_freq_size, low_freq_size, 
                           linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
            axes3[ch_idx, low_freq_idx + 1].add_patch(rect)
            
            # 计算保留区域的能量占比
            total_energy = np.sum(dct_full ** 2)
            preserved_energy = np.sum(dct_full[:low_freq_size, :low_freq_size] ** 2)
            energy_ratio = preserved_energy / total_energy * 100
            
            # 计算保留区域的最大/最小/平均系数值
            preserved_coeffs = dct_full[:low_freq_size, :low_freq_size]
            max_coeff = np.max(np.abs(preserved_coeffs))
            min_coeff = np.min(np.abs(preserved_coeffs))
            mean_coeff = np.mean(np.abs(preserved_coeffs))
            
            title = f'{channel_names[ch_idx]}: DCT {low_freq_size}x{low_freq_size}\n'
            title += f'Energy: {energy_ratio:.1f}% | Max: {max_coeff:.2f} | Mean: {mean_coeff:.2f}'
            axes3[ch_idx, low_freq_idx + 1].set_title(title, fontsize=9)
            axes3[ch_idx, low_freq_idx + 1].set_xlabel('Frequency (W)')
            axes3[ch_idx, low_freq_idx + 1].set_ylabel('Frequency (H)')
            
            # 添加颜色条
            im = axes3[ch_idx, low_freq_idx + 1].images[0]
            plt.colorbar(im, ax=axes3[ch_idx, low_freq_idx + 1], label='log10(|DCT|)')
    
    plt.tight_layout()
    plt.savefig('exp/dct_coefficient_heatmaps.png', dpi=150, bbox_inches='tight')
    print("DCT coefficient heatmaps saved to: exp/dct_coefficient_heatmaps.png")
    plt.close()
    
    # 创建单个通道的详细热力图，显示所有32x32的系数
    print("\nCreating detailed single-channel DCT visualization...")
    fig4, axes4 = plt.subplots(1, 4, figsize=(20, 5))
    
    # 使用绿色通道作为示例
    x_channel = test_img_np[1, :, :]  # Green channel
    dct_full = dctn(x_channel, norm='ortho')
    abs_dct = np.abs(dct_full)
    log_abs_dct = np.log10(abs_dct + 1e-10)
    
    # 原始DCT系数（对数刻度）
    im1 = axes4[0].imshow(log_abs_dct, cmap='hot', aspect='auto', interpolation='nearest')
    axes4[0].set_title('Full DCT Coefficients (log10 scale)', fontsize=12)
    axes4[0].set_xlabel('Frequency (W)')
    axes4[0].set_ylabel('Frequency (H)')
    plt.colorbar(im1, ax=axes4[0], label='log10(|DCT|)')
    
    # 标注不同保留区域
    for idx, low_freq_size in enumerate([4, 8, 16]):
        im = axes4[idx + 1].imshow(log_abs_dct, cmap='hot', aspect='auto', interpolation='nearest')
        
        # 绘制保留区域边界
        axes4[idx + 1].axhline(y=low_freq_size - 0.5, color='cyan', linewidth=3, linestyle='--')
        axes4[idx + 1].axvline(x=low_freq_size - 0.5, color='cyan', linewidth=3, linestyle='--')
        
        # 添加文本标注
        axes4[idx + 1].text(low_freq_size/2 - 0.5, low_freq_size/2 - 0.5, 
                           f'{low_freq_size}x{low_freq_size}\nPreserved', 
                           ha='center', va='center', 
                           bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3),
                           fontsize=14, fontweight='bold')
        
        # 计算统计信息
        preserved_coeffs = dct_full[:low_freq_size, :low_freq_size]
        total_energy = np.sum(dct_full ** 2)
        preserved_energy = np.sum(preserved_coeffs ** 2)
        energy_ratio = preserved_energy / total_energy * 100
        
        max_val = np.max(np.abs(dct_full))
        preserved_max = np.max(np.abs(preserved_coeffs))
        
        title = f'DCT {low_freq_size}x{low_freq_size} Preserved\n'
        title += f'Energy: {energy_ratio:.1f}% | Coeffs: {low_freq_size*low_freq_size}/{32*32} ({low_freq_size*low_freq_size/(32*32)*100:.1f}%)\n'
        title += f'Max preserved: {preserved_max:.2f} / {max_val:.2f}'
        axes4[idx + 1].set_title(title, fontsize=11)
        axes4[idx + 1].set_xlabel('Frequency (W)')
        axes4[idx + 1].set_ylabel('Frequency (H)')
        plt.colorbar(im, ax=axes4[idx + 1], label='log10(|DCT|)')
    
    plt.tight_layout()
    plt.savefig('exp/dct_coefficient_detailed_heatmap.png', dpi=150, bbox_inches='tight')
    print("Detailed DCT heatmap saved to: exp/dct_coefficient_detailed_heatmap.png")
    plt.close()
    
    # 打印详细的数值统计
    print("\nDetailed DCT coefficient statistics:")
    print("=" * 60)
    x_channel = test_img_np[1, :, :]  # Green channel
    dct_full = dctn(x_channel, norm='ortho')
    
    print(f"\nFull DCT coefficient range:")
    print(f"  Min: {np.min(dct_full):.6f}")
    print(f"  Max: {np.max(dct_full):.6f}")
    print(f"  Mean: {np.mean(dct_full):.6f}")
    print(f"  Std: {np.std(dct_full):.6f}")
    print(f"  Energy (sum of squares): {np.sum(dct_full ** 2):.6f}")
    
    for low_freq_size in [4, 8, 16]:
        preserved = dct_full[:low_freq_size, :low_freq_size]
        total_energy = np.sum(dct_full ** 2)
        preserved_energy = np.sum(preserved ** 2)
        
        print(f"\nDCT {low_freq_size}x{low_freq_size} preserved region:")
        print(f"  Min: {np.min(preserved):.6f}")
        print(f"  Max: {np.max(preserved):.6f}")
        print(f"  Mean: {np.mean(preserved):.6f}")
        print(f"  Std: {np.std(preserved):.6f}")
        print(f"  Energy: {preserved_energy:.6f} / {total_energy:.6f} ({preserved_energy/total_energy*100:.2f}%)")
        print(f"  DC component (0,0): {dct_full[0, 0]:.6f} (magnitude: {abs(dct_full[0, 0]):.6f})")


if __name__ == '__main__':
    main()
