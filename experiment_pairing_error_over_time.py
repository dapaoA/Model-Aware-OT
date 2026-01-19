"""
实验脚本：在固定配对下，计算不同时间t的理论去噪方向与模型预测方向的误差

与 experiment_pairing_error_fixed_t.py 的区别：
- 原脚本：对每个固定t，用不同配对方法计算误差
- 本脚本：对同一个配对结果，在不同t下计算误差，观察随时间的变化

功能：
1. 使用不同配对方法进行一次配对
2. 对配对结果，在多个时间t下计算误差
3. 绘制随时间t变化的曲线图（每个方法一条线）
4. 绘制随时间t变化的箱体图
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.optimize

from model import create_model, load_model_config
from flow_matcher import create_flow_matcher
from dataset import get_dataset
from torchcfm.optimal_transport import OTPlanSampler

# 导入配对函数
from experiment_pairing_error_fixed_t import (
    get_paired_samples_cfm,
    get_paired_samples_otcfm,
    get_paired_samples_ma_tcfm,
    get_paired_samples_dct_otcfm,
    compute_cosine_similarity,
    compute_l2_distance
)

# 时间t的采样点
T_VALUES = np.linspace(0.0, 1.0, 21)  # 从0到1，21个点（包括0和1）


def compute_snr_directional(vec, ref_direction):
    """计算方向性SNR (Signal-to-Noise Ratio)
    
    SNR_dir = ||u_parallel||^2 / ||u_perp||^2 = cos^2(θ) / sin^2(θ) = cot^2(θ)
    
    其中：
    - u_parallel: vec 在 ref_direction 方向上的投影
    - u_perp: vec 垂直于 ref_direction 的分量
    - θ: vec 和 ref_direction 之间的夹角
    
    Args:
        vec: 向量 (batch_size, *dims)
        ref_direction: 参考方向 (batch_size, *dims)
    
    Returns:
        snr: SNR值 (batch_size,)
    """
    # 展平为 (batch_size, -1)
    vec_flat = vec.reshape(vec.shape[0], -1)
    ref_flat = ref_direction.reshape(ref_direction.shape[0], -1)
    
    # 归一化参考方向
    ref_norm = torch.norm(ref_flat, dim=1, keepdim=True)  # (batch_size, 1)
    ref_normalized = ref_flat / (ref_norm + 1e-8)  # (batch_size, dim)
    
    # 计算 u_parallel: vec 在 ref_direction 方向上的投影
    # proj = (vec · ref) * ref / ||ref||^2 = (vec · ref_normalized) * ref_normalized
    dot_product = (vec_flat * ref_normalized).sum(dim=1, keepdim=True)  # (batch_size, 1)
    u_parallel = dot_product * ref_normalized  # (batch_size, dim)
    
    # 计算 u_perp: vec 垂直于 ref_direction 的分量
    u_perp = vec_flat - u_parallel  # (batch_size, dim)
    
    # 计算 ||u_parallel||^2 和 ||u_perp||^2
    u_parallel_norm_sq = torch.sum(u_parallel ** 2, dim=1)  # (batch_size,)
    u_perp_norm_sq = torch.sum(u_perp ** 2, dim=1)  # (batch_size,)
    
    # SNR = ||u_parallel||^2 / ||u_perp||^2 = cot^2(θ)
    snr = u_parallel_norm_sq / (u_perp_norm_sq + 1e-8)
    
    return snr


def compute_error_over_time(model, flow_matcher, x0_paired, x1_paired, device, t_values, epsilon, metric='cos', sigma=0.0):
    """
    对固定的配对结果，在不同时间t下计算误差
    
    Args:
        model: 训练好的模型
        flow_matcher: flow matcher实例
        x0_paired: 配对后的噪声 (batch_size, *dims)
        x1_paired: 配对后的图像 (batch_size, *dims)
        device: 设备
        t_values: 时间t的值数组
        epsilon: 单个epsilon样本 (batch_size, *dims)
        metric: 度量方式 ('cos' 或 'l2')
        sigma: sigma参数（默认0.0）
        
    Returns:
        errors: (len(t_values), batch_size) 的数组
    """
    model.eval()
    
    errors_over_time = []
    
    with torch.no_grad():
        # 理论去噪方向 ut = x1_paired - x0_paired（配对后的方向）
        ut_theoretical = x1_paired - x0_paired
        
        for t_val in t_values:
            t = torch.full((x0_paired.shape[0],), t_val, device=device, dtype=torch.float32)
            
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
            
            # 计算误差（每个配对一个值）
            if metric == 'cos':
                errors = compute_cosine_similarity(ut_theoretical, vt_predicted)
            elif metric == 'l2':
                errors = compute_l2_distance(ut_theoretical, vt_predicted)
            elif metric == 'snr':
                # SNR_dir = ||u_parallel||^2 / ||u_perp||^2
                # 使用 vt_predicted 作为 vec，ut_theoretical 作为参考方向
                errors = compute_snr_directional(vt_predicted, ut_theoretical)
            else:
                raise ValueError(f"Unknown metric: {metric}. Must be 'cos', 'l2', or 'snr'")
            
            errors_over_time.append(errors.cpu().numpy())
    
    return np.array(errors_over_time)  # Shape: (len(t_values), batch_size)


def plot_curve_over_time(results_over_time, t_values, pairing_methods, method_labels, colors, output_dir, model_info, metric='cos'):
    """绘制随时间t变化的曲线图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, method in enumerate(pairing_methods):
        # results_over_time[method]: (num_batches, num_t, batch_size)
        data = results_over_time[method]  # (num_batches, num_t, batch_size)
        
        # 计算每个t的均值
        means = []
        for t_idx in range(len(t_values)):
            # 提取所有batch中该t值下的所有误差
            t_errors = data[:, t_idx, :].flatten()  # 所有batch的所有样本
            means.append(np.mean(t_errors))
        
        means = np.array(means)
        
        # 绘制平滑曲线（无标记点）
        ax.plot(t_values, means, label=method_labels[method], 
                color=colors[idx % len(colors)], linewidth=1.5)
    
    ax.set_xlabel('Time t', fontsize=12)
    if metric == 'cos':
        metric_label = 'Cosine Similarity'
        metric_title = 'Cosine Similarities'
    elif metric == 'l2':
        metric_label = 'L2 Distance'
        metric_title = 'L2 Distances'
    elif metric == 'snr':
        metric_label = 'SNR (cot²θ)'
        metric_title = 'Directional SNR'
    else:
        metric_label = 'Error'
        metric_title = 'Errors'
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f'{metric_title} over Time (Same Pairing Result)', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # 保存图片
    png_filename = f"pairing_error_over_time_{model_info}_{metric}_curve.png"
    plt.savefig(output_dir / png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return png_filename


def plot_continuous_distribution_over_time(results_over_time, t_values, pairing_methods, method_labels, colors, output_dir, model_info, metric='cos'):
    """绘制随时间t变化的连续分布图（类似violin plot但连续）"""
    from scipy.stats import gaussian_kde
    
    if metric == 'cos':
        metric_label = 'Cosine Similarity'
    elif metric == 'l2':
        metric_label = 'L2 Distance'
    elif metric == 'snr':
        metric_label = 'SNR (cot²θ)'
    else:
        metric_label = 'Error'
    
    # 为每个配对方法创建一个图
    fig, axes = plt.subplots(len(pairing_methods), 1, figsize=(14, 3 * len(pairing_methods)))
    
    if len(pairing_methods) == 1:
        axes = [axes]
    
    for method_idx, method in enumerate(pairing_methods):
        ax = axes[method_idx]
        
        data = results_over_time[method]  # (num_batches, num_t, batch_size)
        
        # 确定y轴范围（所有t值的误差范围）
        all_errors = data.flatten()
        y_min = np.min(all_errors)
        y_max = np.max(all_errors)
        y_range = y_max - y_min
        y_min -= y_range * 0.05
        y_max += y_range * 0.05
        
        # 创建y轴采样点用于KDE
        y_samples = np.linspace(y_min, y_max, 200)
        
        # 对每个t值，计算密度分布
        for t_idx, t_val in enumerate(t_values):
            # 提取所有batch中该t值下的所有误差
            t_errors = data[:, t_idx, :].flatten()
            
            if len(t_errors) > 1:
                # 使用KDE计算密度
                try:
                    kde = gaussian_kde(t_errors)
                    density = kde(y_samples)
                    
                    # 归一化密度，使其在x轴方向有合适的宽度
                    max_density = np.max(density)
                    if max_density > 0:
                        density_normalized = density / max_density * 0.03  # 宽度为0.03
                        
                        # 绘制密度分布（左右对称）
                        ax.fill_betweenx(y_samples, t_val - density_normalized, t_val + density_normalized,
                                       alpha=0.6, color=colors[method_idx % len(colors)])
                except:
                    # 如果KDE失败，使用直方图
                    hist, bins = np.histogram(t_errors, bins=30, density=True)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    hist_normalized = hist / np.max(hist) * 0.03
                    ax.fill_betweenx(bin_centers, t_val - hist_normalized, t_val + hist_normalized,
                                   alpha=0.6, color=colors[method_idx % len(colors)])
        
        # 绘制均值曲线
        means = []
        for t_idx in range(len(t_values)):
            t_errors = data[:, t_idx, :].flatten()
            means.append(np.mean(t_errors))
        ax.plot(t_values, means, color='black', linewidth=1.5, linestyle='--', alpha=0.8, label='Mean')
        
        ax.set_xlabel('Time t', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'{method_labels[method]} - {metric_label} Distribution over Time', fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    png_filename = f"pairing_error_over_time_{model_info}_{metric}_distribution.png"
    plt.savefig(output_dir / png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return png_filename


def experiment(args):
    """主实验函数"""
    # 使用全局的 T_VALUES
    global T_VALUES
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 先加载模型以获取信息
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get('args', {})
    model_config = checkpoint.get('model_config', {})
    dataset_name = train_args.get('dataset', 'cifar10')
    sigma = args.sigma if hasattr(args, 'sigma') else 0.0
    method = train_args.get('method', 'cfm')
    
    # 创建模型
    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # 获取数据集（第一次加载，用于获取模型信息，但在else分支中会重新加载）
    if dataset_name in ['cifar10', 'mnist']:
        dataloader, _ = get_dataset(dataset_name, args.batch_size, args.data_dir)
    else:
        raise ValueError("This experiment currently only supports image datasets (cifar10, mnist)")
    
    # 创建flow matcher
    flow_matcher = create_flow_matcher('cfm', sigma)
    
    # 定义配对方法
    pairing_methods = ['otcfm', 'dct_otcfm_4x4', 'dct_hf_otcfm_4x4', 'cfm']
    method_labels = {
        'cfm': 'CFM (Random)',
        'otcfm': 'OTCFM (OT)',
        'dct_otcfm_4x4': 'DCT_OTCFM (4x4)',
        'dct_hf_otcfm_4x4': 'DCT_HF_OTCFM (4x4)'
    }
    colors = ['green', 'blue', 'orange', 'gray']
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = f"{method}_{dataset_name}_bs{args.batch_size}"
    npz_filename = f"pairing_error_over_time_{model_info}_{args.metric}.npz"
    npz_path = output_dir / npz_filename
    
    # 使用全局的 T_VALUES（如果从文件加载会覆盖）
    t_values_use = T_VALUES
    
    # 检查是否已有保存的数据
    if npz_path.exists() and not args.force_recompute:
        print(f"Loading existing data from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        t_values_use = data['t_values']
        pairing_methods_loaded = data['pairing_methods'].tolist()
        
        # 确保配对方法顺序一致
        pairing_methods = [m for m in pairing_methods if m in pairing_methods_loaded]
        
        results_over_time = {}
        for method in pairing_methods:
            results_over_time[method] = data[method]
        
        print(f"Data loaded: {len(pairing_methods)} methods, shape={results_over_time[pairing_methods[0]].shape}")
    else:
        # 需要计算数据
        print(f"Loading checkpoint from {checkpoint_path}")
        
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
        print(f"Time points: {len(t_values_use)} points from {t_values_use[0]} to {t_values_use[-1]}")
        
        # 创建flow matcher
        flow_matcher = create_flow_matcher('cfm', sigma)
        
        # 存储所有结果：{method: (num_batches, num_t, batch_size)}
        results_over_time = {method: [] for method in pairing_methods}
        
        print(f"\n{'='*60}")
        print("Starting experiment: Computing errors over time for each pairing method")
        print(f"{'='*60}")
        
        # 对每个batch采样
        for batch_idx in range(args.num_batches):
            print(f"\nBatch {batch_idx + 1}/{args.num_batches}:")
            
            # 获取数据
            x1 = next(iter(dataloader))[0].to(device)
            x0 = torch.randn_like(x1)
            epsilon = torch.randn_like(x0)
            
            # 对每个配对方法
            for pairing_method in pairing_methods:
                # 进行配对（只配对一次）
                if pairing_method == 'cfm':
                    x0_paired, x1_paired = get_paired_samples_cfm(x0, x1)
                elif pairing_method == 'otcfm':
                    x0_paired, x1_paired = get_paired_samples_otcfm(x0, x1)
                elif pairing_method == 'dct_otcfm_4x4':
                    x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=4)
                elif pairing_method == 'dct_hf_otcfm_4x4':
                    # 导入高频DCT配对函数
                    from experiment_pairing_error_fixed_t import get_paired_samples_dct_hf_otcfm
                    x0_paired, x1_paired = get_paired_samples_dct_hf_otcfm(x0, x1, high_freq_size=4)
                else:
                    raise ValueError(f"Unknown pairing method: {pairing_method}")
                
                # 对配对结果，在不同t下计算误差
                errors_over_t = compute_error_over_time(
                    model, flow_matcher, x0_paired, x1_paired, device,
                    t_values_use, epsilon, metric=args.metric, sigma=sigma
                )  # Shape: (num_t, batch_size)
                
                results_over_time[pairing_method].append(errors_over_t)
            
            print(f"  Completed all pairing methods for batch {batch_idx + 1}")
        
        # 转换为numpy数组：{method: (num_batches, num_t, batch_size)}
        for method in pairing_methods:
            results_over_time[method] = np.array(results_over_time[method])
        
        # 保存数值结果
        save_dict = {
            't_values': t_values_use,
            'pairing_methods': np.array(pairing_methods, dtype=object)
        }
        for method in pairing_methods:
            save_dict[method] = results_over_time[method]
        
        np.savez_compressed(npz_path, **save_dict)
        print(f"\nNumerical results saved: {npz_filename}")
    
    # 绘制曲线图
    print("\nGenerating curve plot...")
    curve_filename = plot_curve_over_time(
        results_over_time, t_values_use, pairing_methods, method_labels, colors,
        output_dir, model_info, metric=args.metric
    )
    print(f"Curve plot saved: {curve_filename}")
    
    # 绘制连续分布图
    print("Generating continuous distribution plot...")
    distribution_filename = plot_continuous_distribution_over_time(
        results_over_time, t_values_use, pairing_methods, method_labels, colors,
        output_dir, model_info, metric=args.metric
    )
    print(f"Distribution plot saved: {distribution_filename}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment: Compute pairing error over time for same pairing result")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for sampling')
    parser.add_argument('--num_batches', type=int, default=5,
                       help='Number of batches to sample (each batch uses new images and noise)')
    parser.add_argument('--metric', type=str, default='cos', choices=['cos', 'l2', 'snr'],
                       help='Metric type: "cos" for cosine similarity, "l2" for L2 distance, "snr" for directional SNR')
    parser.add_argument('--sigma', type=float, default=0.0,
                       help='Sigma parameter for flow matching (default: 0.0)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                       help='Directory to save results')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recompute even if data file exists')
    
    args = parser.parse_args()
    experiment(args)
