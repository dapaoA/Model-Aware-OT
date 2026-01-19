"""
实验脚本：比较不同配对方式下理论去噪方向与模型预测方向的误差

输入变量：
- 模型checkpoint路径
- batch_size
- 配对方式（cfm, otcfm, ma_tcfm）

计算：
- 对于任意采样的noise和image batch对
- 用不同配对方式计算配对组合
- 计算理论去噪方向（配对后的图片-噪声点）和模型预测的去噪方向v(xt, t)的余弦相似度
- 比较三种方法的cos分布和均值
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
    elif ma_method == 'downsample_3x':
        def M(x):
            # 3x下采样
            if x.dim() == 4:
                return F.avg_pool2d(x, kernel_size=3, stride=3, padding=0)
            else:
                return x
    elif ma_method == 'downsample_4x':
        def M(x):
            # 4x下采样
            if x.dim() == 4:
                return F.avg_pool2d(x, kernel_size=4, stride=4, padding=0)
            else:
                return x
    elif ma_method == 'downsample_8x':
        def M(x):
            # 8x下采样
            if x.dim() == 4:
                return F.avg_pool2d(x, kernel_size=8, stride=8, padding=0)
            else:
                return x
    else:
        raise ValueError(f"Unknown ma_method: {ma_method}. Supported: downsample_2x, downsample_3x, downsample_4x, downsample_8x")
    
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


def get_paired_samples_ma_pcfm(x0, x1, ma_method='downsample_2x'):
    """MA_PCFM (Partial): 只对噪声下采样，图片不变，计算OT时还原噪声到原尺寸"""
    import torch.nn.functional as F
    import numpy as np
    
    # 确定下采样因子
    if ma_method == 'downsample_2x':
        factor = 2
    elif ma_method == 'downsample_3x':
        factor = 3
    elif ma_method == 'downsample_4x':
        factor = 4
    elif ma_method == 'downsample_8x':
        factor = 8
    else:
        raise ValueError(f"Unknown ma_method: {ma_method}. Supported: downsample_2x, downsample_3x, downsample_4x, downsample_8x")
    
    if x0.dim() == 4:  # 图像数据 (B, C, H, W)
        # 对噪声x0进行下采样
        x0_downsampled = F.avg_pool2d(x0, kernel_size=factor, stride=factor, padding=0)
        
        # 将下采样的x0还原（上采样）到原尺寸，用于计算OT plan
        # 使用双线性插值上采样
        x0_restored = F.interpolate(x0_downsampled, size=(x0.shape[2], x0.shape[3]), 
                                    mode='bilinear', align_corners=False)
        
        # x1保持不变
        x1_unchanged = x1
        
        # 在还原后的空间计算OT plan（x0_restored和x1都是原尺寸）
        # Compute cost matrix
        x0_flat = x0_restored.reshape(x0_restored.shape[0], -1)
        x1_flat = x1_unchanged.reshape(x1_unchanged.shape[0], -1)
        M_cost = torch.cdist(x0_flat, x1_flat) ** 2
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cost.cpu().numpy())
        
        # Convert to torch tensor
        if isinstance(row_ind, np.ndarray):
            row_ind = torch.from_numpy(row_ind).to(x0.device)
        if isinstance(col_ind, np.ndarray):
            col_ind = torch.from_numpy(col_ind).to(x1.device)
        
        # 使用原始（未变换）的x0和x1进行配对
        x0_paired = x0[row_ind]
        x1_paired = x1[col_ind]
        
    else:
        # 对于非图像数据，直接返回原始配对（不支持下采样）
        x0_paired = x0
        x1_paired = x1
    
    return x0_paired, x1_paired


def compute_pairing_error(model, flow_matcher, x0, x1, pairing_method, device, t_samples, epsilon_samples):
    """
    计算配对误差
    
    Args:
        model: 训练好的模型（应该用CFM训练的模型）
        flow_matcher: flow matcher实例（用于计算xt和ut）
        x0: 噪声batch (batch_size, *dims)
        x1: 图像batch (batch_size, *dims)
        pairing_method: 配对方式 ('cfm', 'otcfm', 'ma_tcfm', etc.)
        device: 设备
        t_samples: 预采样的t值列表，每个元素是(batch_size,)的tensor
        epsilon_samples: 预采样的epsilon值列表，每个元素是(batch_size, *dims)的tensor
        
    Returns:
        cos_similarities: 所有计算的余弦相似度值
    """
    model.eval()
    cos_similarities = []
    
    with torch.no_grad():
        # 根据配对方式获取配对后的x0和x1
        if pairing_method == 'cfm':
            x0_paired, x1_paired = get_paired_samples_cfm(x0, x1)
        elif pairing_method == 'otcfm':
            x0_paired, x1_paired = get_paired_samples_otcfm(x0, x1)
        elif pairing_method == 'ma_tcfm':
            x0_paired, x1_paired = get_paired_samples_ma_tcfm(x0, x1, ma_method='downsample_2x')
        elif pairing_method == 'ma_tcfm_3x':
            x0_paired, x1_paired = get_paired_samples_ma_tcfm(x0, x1, ma_method='downsample_3x')
        elif pairing_method == 'ma_tcfm_4x':
            x0_paired, x1_paired = get_paired_samples_ma_tcfm(x0, x1, ma_method='downsample_4x')
        elif pairing_method == 'ma_tcfm_8x':
            x0_paired, x1_paired = get_paired_samples_ma_tcfm(x0, x1, ma_method='downsample_8x')
        elif pairing_method == 'ma_pcfm':
            x0_paired, x1_paired = get_paired_samples_ma_pcfm(x0, x1, ma_method='downsample_2x')
        elif pairing_method == 'ma_pcfm_3x':
            x0_paired, x1_paired = get_paired_samples_ma_pcfm(x0, x1, ma_method='downsample_3x')
        elif pairing_method == 'ma_pcfm_4x':
            x0_paired, x1_paired = get_paired_samples_ma_pcfm(x0, x1, ma_method='downsample_4x')
        elif pairing_method == 'ma_pcfm_8x':
            x0_paired, x1_paired = get_paired_samples_ma_pcfm(x0, x1, ma_method='downsample_8x')
        else:
            raise ValueError(f"Unknown pairing method: {pairing_method}")
        
        # 计算理论去噪方向 ut = x1_paired - x0_paired（配对后的方向）
        ut_theoretical = x1_paired - x0_paired
        
        # 使用预采样的t和epsilon（确保所有方法使用相同的数据）
        for t, epsilon in zip(t_samples, epsilon_samples):
            # 使用手动配对的x0_paired和x1_paired计算xt
            # 计算xt = t * x1_paired + (1-t) * x0_paired + sigma * epsilon
            t_expanded = t.reshape(-1, *([1] * (x0_paired.dim() - 1)))
            mu_t = t_expanded * x1_paired + (1 - t_expanded) * x0_paired
            sigma_t = flow_matcher.compute_sigma_t(t)
            sigma_t_expanded = sigma_t.reshape(-1, *([1] * (x0_paired.dim() - 1))) if isinstance(sigma_t, torch.Tensor) else sigma_t
            if isinstance(sigma_t_expanded, (int, float)):
                xt = mu_t + sigma_t_expanded * epsilon
            else:
                xt = mu_t + sigma_t_expanded * epsilon
            
            # 模型预测的去噪方向
            if x0_paired.dim() == 4:  # 图像数据
                vt_predicted = model(xt, t)
            else:  # 2D数据
                vt_predicted = model(torch.cat([xt, t[:, None]], dim=-1))
            
            # 计算余弦相似度（每个配对一个值）
            cos_sim = compute_cosine_similarity(ut_theoretical, vt_predicted)
            cos_similarities.extend(cos_sim.cpu().numpy().tolist())
    
    return np.array(cos_similarities)


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
    sigma = train_args.get('sigma', 0.1)
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
        # 获取一个batch的数据
        x1 = next(iter(dataloader))[0].to(device)
        x0 = torch.randn_like(x1)
    else:
        raise ValueError("This experiment currently only supports image datasets (cifar10, mnist)")
    
    print(f"Dataset: {dataset_name}, Batch size: {args.batch_size}")
    print(f"Image shape: {x1.shape}")
    
    # 创建flow matcher（用于计算xt）
    flow_matcher = create_flow_matcher('cfm', sigma)  # 使用CFM的flow matcher来计算xt
    
    # 预采样t和epsilon，确保所有配对方法使用相同的数据
    print("\nPre-sampling t and epsilon values for all methods...")
    t_samples = []
    epsilon_samples = []
    for _ in range(args.num_t_samples):
        t = torch.rand(x0.shape[0], device=device)
        epsilon = torch.randn_like(x0)
        t_samples.append(t)
        epsilon_samples.append(epsilon)
    print(f"Pre-sampled {len(t_samples)} sets of t and epsilon values")
    
    # 计算多种配对方式的误差
    # 注意：这里使用CFM训练的模型，但用不同的配对方式来计算误差
    results = {}
    
    pairing_methods = ['otcfm', 'ma_tcfm', 'ma_tcfm_3x', 'ma_tcfm_4x', 'ma_tcfm_8x', 
                       'ma_pcfm', 'ma_pcfm_3x', 'ma_pcfm_4x', 'ma_pcfm_8x', 'cfm']
    
    for pairing_method in pairing_methods:
        print(f"\nComputing pairing error for {pairing_method}...")
        cos_similarities = compute_pairing_error(
            model, flow_matcher, x0, x1, pairing_method, device, 
            t_samples, epsilon_samples
        )
        results[pairing_method] = cos_similarities
        print(f"  Total samples: {len(cos_similarities)}")
        print(f"  Mean cosine similarity: {cos_similarities.mean():.4f}")
        print(f"  Std cosine similarity: {cos_similarities.std():.4f}")
        print(f"  Min cosine similarity: {cos_similarities.min():.4f}")
        print(f"  Max cosine similarity: {cos_similarities.max():.4f}")
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建文件名（包含模型信息和batch_size）
    model_info = f"{method}_{dataset_name}_bs{args.batch_size}"
    npz_filename = f"pairing_error_results_{model_info}.npz"
    png_filename = f"pairing_error_comparison_{model_info}.png"
    
    # 保存数值结果（包含所有配对方法）
    save_dict = {method: results[method] for method in pairing_methods if method in results}
    np.savez(
        output_dir / npz_filename,
        **save_dict
    )
    
    # 绘制对比图
    plt.figure(figsize=(18, 5))
    
    # 准备绘图数据
    method_labels = {
        'cfm': 'CFM (Random)',
        'otcfm': 'OTCFM (OT)',
        'ma_tcfm': 'MA_TCFM (2x)',
        'ma_tcfm_3x': 'MA_TCFM (3x)',
        'ma_tcfm_4x': 'MA_TCFM (4x)',
        'ma_tcfm_8x': 'MA_TCFM (8x)',
        'ma_pcfm': 'MA_PCFM (2x)',
        'ma_pcfm_3x': 'MA_PCFM (3x)',
        'ma_pcfm_4x': 'MA_PCFM (4x)',
        'ma_pcfm_8x': 'MA_PCFM (8x)'
    }
    
    # 准备数据
    data_to_plot = [results[method] for method in pairing_methods if method in results]
    labels_to_plot = [method_labels[method] for method in pairing_methods if method in results]
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'gray']
    
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
    
    plt.xlabel('Cosine Similarity', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.title('Distribution of Cosine Similarities (KDE)', fontsize=12)
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    
    # 2. Violin plot对比（显示密度分布）
    plt.subplot(1, 3, 2)
    parts = plt.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                          showmeans=True, showmedians=True, widths=0.7)
    
    # 设置颜色
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[idx % len(colors)])
        pc.set_alpha(0.7)
    
    plt.ylabel('Cosine Similarity', fontsize=11)
    plt.title('Violin Plot Comparison', fontsize=12)
    plt.xticks(range(len(labels_to_plot)), labels_to_plot, rotation=45, ha='right', fontsize=9)
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
    plt.ylabel('Mean Cosine Similarity', fontsize=11)
    plt.title('Mean ± Std Comparison', fontsize=12)
    plt.xticks(x_pos, methods_plot, rotation=45, ha='right', fontsize=9)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + y_range * 0.02, f'{mean:.4f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nResults saved to {output_dir}")
    print(f"  - Numerical results: {npz_filename}")
    print(f"  - Visualization: {png_filename}")
    
    # 打印统计摘要
    print("\n" + "="*60)
    print("Statistical Summary:")
    print("="*60)
    for method, values in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Mean:   {values.mean():.6f}")
        print(f"  Std:    {values.std():.6f}")
        print(f"  Median: {np.median(values):.6f}")
        print(f"  Min:    {values.min():.6f}")
        print(f"  Max:    {values.max():.6f}")
        print(f"  Q25:    {np.percentile(values, 25):.6f}")
        print(f"  Q75:    {np.percentile(values, 75):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment: Compare pairing error across different methods")
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    
    # Batch size
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for sampling (number of noise-image pairs)')
    
    # Number of t samples per pairing
    parser.add_argument('--num_t_samples', type=int, default=10,
                       help='Number of t samples per pairing (each pairing will be evaluated num_t_samples times with different t values)')
    
    # Data directory
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    experiment(args)
