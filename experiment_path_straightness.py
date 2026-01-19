"""
实验脚本：测试配对形成的路径在流场中的"笔直程度"或"流畅度"

对于每个配对方法：
1. 获得配对后的 (x0_paired, x1_paired)
2. 对每个配对，在多个时间点 t 下：
   - 计算 xt = t * x1 + (1-t) * x0
   - 计算模型预测的速度 v(xt, t)
   - 理论方向 ut = x1 - x0 (固定不变)
3. 计算速度方向的变化：
   - 方式1：计算 v(xt, t) 与理论方向 ut 的夹角变化
   - 方式2：计算相邻时间点的 v(xt, t) 之间的角度变化
4. 对所有配对求平均，得到该配对方法的"笔直程度"指标

指标：如果路径很"笔直"，那么：
- v(xt, t) 的方向在所有 t 下都应该接近 ut（理论方向）
- 相邻时间点的 v(xt, t) 之间的角度应该很小
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
    compute_cosine_similarity
)

# 时间t的采样点（更多点以获得更平滑的评估）
T_VALUES = np.linspace(0.0, 1.0, 51)  # 从0到1，51个点（包括0和1）


def compute_angle_between_vectors(vec1, vec2):
    """计算两个向量之间的夹角（弧度）"""
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
    # 限制在 [-1, 1] 范围内，避免数值误差
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    
    # 计算夹角（弧度）
    angle = torch.acos(cos_sim)
    return angle


def compute_path_straightness(model, flow_matcher, x0_paired, x1_paired, device, t_values, epsilon, sigma=0.0):
    """
    计算配对路径的笔直程度（仅关注速度方向本身的变化剧烈程度）
    
    Args:
        model: 训练好的模型
        flow_matcher: flow matcher实例
        x0_paired: 配对后的噪声 (batch_size, *dims)
        x1_paired: 配对后的图像 (batch_size, *dims)
        device: 设备
        t_values: 时间t的值数组
        epsilon: epsilon样本 (batch_size, *dims)，用于所有t（共享）
        sigma: sigma参数（默认0.0）
        
    Returns:
        metrics: 字典，包含各种笔直程度指标
            - 'angle_change_adjacent': (num_t-1,) - 相邻时间点之间 v(xt,t) 的角度变化（平均值）
            - 'mean_angle_change': 标量 - 平均相邻角度变化
            - 'max_angle_change': 标量 - 最大相邻角度变化
            - 'std_angle_change': 标量 - 相邻角度变化的标准差
    """
    model.eval()
    
    batch_size = x0_paired.shape[0]
    num_t = len(t_values)
    
    # 存储所有时间点的预测速度
    vt_predicted_all = []  # 列表，每个元素是 (batch_size, *dims)
    
    with torch.no_grad():
        for t_val in t_values:
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
            
            # 计算xt = t * x1_paired + (1-t) * x0_paired + sigma * epsilon
            t_expanded = t.reshape(-1, *([1] * (x0_paired.dim() - 1)))
            mu_t = t_expanded * x1_paired + (1 - t_expanded) * x0_paired
            sigma_t = flow_matcher.compute_sigma_t(t)
            sigma_t_expanded = sigma_t.reshape(-1, *([1] * (mu_t.dim() - 1))) if isinstance(sigma_t, torch.Tensor) else sigma_t
            if isinstance(sigma_t_expanded, (int, float)):
                xt = mu_t + sigma_t_expanded * epsilon
            else:
                xt = mu_t + sigma_t_expanded * epsilon
            
            # 模型预测的速度
            if x0_paired.dim() == 4:  # 图像数据
                vt_predicted = model(xt, t)  # (batch_size, *dims)
            else:  # 2D数据
                vt_predicted = model(torch.cat([xt, t[:, None]], dim=-1))
            
            vt_predicted_all.append(vt_predicted)
    
    # 计算相邻时间点的角度变化（速度方向本身的变化）
    angles_change_adjacent = []
    for i in range(num_t - 1):
        # 计算相邻两个时间点的 v(xt, t) 之间的夹角
        angles = compute_angle_between_vectors(vt_predicted_all[i], vt_predicted_all[i+1])  # (batch_size,)
        angles_change_adjacent.append(angles.cpu().numpy())
    angles_change_adjacent = np.array(angles_change_adjacent)  # (num_t-1, batch_size)
    
    # 计算统计指标
    metrics = {
        # 相邻时间点之间的角度变化（对所有batch求平均）
        'angle_change_adjacent': np.mean(angles_change_adjacent, axis=1),  # (num_t-1,)
        # 总体统计
        'mean_angle_change': np.mean(angles_change_adjacent),
        'max_angle_change': np.max(angles_change_adjacent),
        'std_angle_change': np.std(angles_change_adjacent),
    }
    
    return metrics


def plot_straightness_metrics(all_metrics, t_values, pairing_methods, method_labels, colors, output_dir, model_info):
    """绘制笔直程度指标（速度方向本身的变化）"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制相邻时间点的角度变化（速度方向本身的变化剧烈程度）
    t_values_mid = (t_values[:-1] + t_values[1:]) / 2  # 中点时间
    for idx, method in enumerate(pairing_methods):
        metrics = all_metrics[method]
        angle_changes = metrics['angle_change_adjacent']  # (num_t-1,)
        # 转换为度数
        angle_changes_deg = np.degrees(angle_changes)
        ax.plot(t_values_mid, angle_changes_deg, label=method_labels[method], 
                color=colors[idx % len(colors)], linewidth=2)
    
    ax.set_xlabel('Time t (midpoint)', fontsize=12)
    ax.set_ylabel('Angle Change between Adjacent Time Points (degrees)', fontsize=12)
    ax.set_title('Change in v(xt,t) Direction between Adjacent Time Points\n(Smaller = Straighter Path)', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # 保存图片
    png_filename = f"path_straightness_{model_info}.png"
    plt.savefig(output_dir / png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return png_filename


def plot_summary_statistics(all_metrics, pairing_methods, method_labels, colors, output_dir, model_info):
    """绘制统计摘要（柱状图）"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 平均和最大相邻角度变化
    means_change = [all_metrics[method]['mean_angle_change'] for method in pairing_methods]
    maxs_change = [all_metrics[method]['max_angle_change'] for method in pairing_methods]
    means_change_deg = np.degrees(means_change)
    maxs_change_deg = np.degrees(maxs_change)
    
    x_pos = np.arange(len(pairing_methods))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, means_change_deg, width, label='Mean', alpha=0.8,
                   color=[colors[idx % len(colors)] for idx in range(len(pairing_methods))])
    bars2 = ax.bar(x_pos + width/2, maxs_change_deg, width, label='Max', alpha=0.6,
                   color=[colors[idx % len(colors)] for idx in range(len(pairing_methods))])
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Angle Change (degrees)', fontsize=12)
    ax.set_title('Angle Change in v(xt,t) Direction between Adjacent Time Points\n(Smaller = Straighter Path)', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_labels[method] for method in pairing_methods], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图片
    png_filename = f"path_straightness_summary_{model_info}.png"
    plt.savefig(output_dir / png_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return png_filename


def experiment(args):
    """主实验函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
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
    
    # 获取数据集
    if dataset_name in ['cifar10', 'mnist']:
        dataloader, _ = get_dataset(dataset_name, args.batch_size, args.data_dir)
    else:
        raise ValueError("This experiment currently only supports image datasets (cifar10, mnist)")
    
    print(f"Dataset: {dataset_name}, Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    print(f"Time points: {len(T_VALUES)} points from {T_VALUES[0]} to {T_VALUES[-1]}")
    
    # 创建flow matcher
    flow_matcher = create_flow_matcher('cfm', sigma)
    
    # 定义配对方法
    pairing_methods = ['otcfm', 'ma_tcfm', 'dct_otcfm_4x4', 'dct_otcfm_8x8', 'dct_otcfm_16x16', 'cfm']
    method_labels = {
        'cfm': 'CFM (Random)',
        'otcfm': 'OTCFM (OT)',
        'ma_tcfm': 'MA_TCFM (2x)',
        'dct_otcfm_4x4': 'DCT_OTCFM (4x4)',
        'dct_otcfm_8x8': 'DCT_OTCFM (8x8)',
        'dct_otcfm_16x16': 'DCT_OTCFM (16x16)'
    }
    colors = ['green', 'red', 'blue', 'cyan', 'magenta', 'gray']
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = f"{method}_{dataset_name}_bs{args.batch_size}"
    
    # 存储所有结果
    all_metrics = {method: [] for method in pairing_methods}
    
    print(f"\n{'='*60}")
    print("Starting experiment: Computing path straightness metrics")
    print(f"{'='*60}")
    
    # 对每个batch采样
    for batch_idx in range(args.num_batches):
        print(f"\nBatch {batch_idx + 1}/{args.num_batches}:")
        
        # 获取数据
        x1 = next(iter(dataloader))[0].to(device)
        x0 = torch.randn_like(x1)
        # 使用同一个epsilon用于所有时间点（保证一致性）
        epsilon = torch.randn_like(x0)
        
        # 对每个配对方法
        for pairing_method in pairing_methods:
            # 进行配对
            if pairing_method == 'cfm':
                x0_paired, x1_paired = get_paired_samples_cfm(x0, x1)
            elif pairing_method == 'otcfm':
                x0_paired, x1_paired = get_paired_samples_otcfm(x0, x1)
            elif pairing_method == 'ma_tcfm':
                x0_paired, x1_paired = get_paired_samples_ma_tcfm(x0, x1)
            elif pairing_method == 'dct_otcfm_4x4':
                x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=4)
            elif pairing_method == 'dct_otcfm_8x8':
                x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=8)
            elif pairing_method == 'dct_otcfm_16x16':
                x0_paired, x1_paired = get_paired_samples_dct_otcfm(x0, x1, low_freq_size=16)
            else:
                raise ValueError(f"Unknown pairing method: {pairing_method}")
            
            # 计算笔直程度指标
            metrics = compute_path_straightness(
                model, flow_matcher, x0_paired, x1_paired, device,
                T_VALUES, epsilon, sigma=sigma
            )
            
            all_metrics[pairing_method].append(metrics)
        
        print(f"  Completed all pairing methods for batch {batch_idx + 1}")
    
    # 对所有batch的结果求平均
    averaged_metrics = {}
    for method in pairing_methods:
        # 平均每个指标的各个batch结果
        avg_metrics = {}
        for key in all_metrics[method][0].keys():
            if isinstance(all_metrics[method][0][key], np.ndarray):
                # 对于数组，对所有batch求平均
                stacked = np.stack([m[key] for m in all_metrics[method]], axis=0)
                avg_metrics[key] = np.mean(stacked, axis=0)
            else:
                # 对于标量，对所有batch求平均
                avg_metrics[key] = np.mean([m[key] for m in all_metrics[method]])
        averaged_metrics[method] = avg_metrics
    
    # 打印统计摘要
    print("\n" + "="*60)
    print("Path Straightness Summary Statistics:")
    print("(Angle change in v(xt,t) direction - Smaller = Straighter Path)")
    print("="*60)
    for method in pairing_methods:
        m = averaged_metrics[method]
        print(f"\n{method_labels[method]}:")
        print(f"  Mean angle change: {np.degrees(m['mean_angle_change']):.4f}°")
        print(f"  Max angle change:  {np.degrees(m['max_angle_change']):.4f}°")
        print(f"  Std angle change:  {np.degrees(m['std_angle_change']):.4f}°")
    
    # 绘制结果
    print("\nGenerating plots...")
    curve_filename = plot_straightness_metrics(
        averaged_metrics, T_VALUES, pairing_methods, method_labels, colors,
        output_dir, model_info
    )
    print(f"Curve plot saved: {curve_filename}")
    
    summary_filename = plot_summary_statistics(
        averaged_metrics, pairing_methods, method_labels, colors,
        output_dir, model_info
    )
    print(f"Summary plot saved: {summary_filename}")
    
    # 保存数值结果
    npz_filename = f"path_straightness_{model_info}.npz"
    save_dict = {'t_values': T_VALUES, 'pairing_methods': np.array(pairing_methods, dtype=object)}
    for method in pairing_methods:
        for key, value in averaged_metrics[method].items():
            save_dict[f"{method}_{key}"] = value
    np.savez(output_dir / npz_filename, **save_dict)
    print(f"Numerical results saved: {npz_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment: Compute path straightness in flow field")
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for sampling')
    parser.add_argument('--num_batches', type=int, default=5,
                       help='Number of batches to sample (each batch uses new images and noise)')
    parser.add_argument('--sigma', type=float, default=0.0,
                       help='Sigma parameter for flow matching (default: 0.0)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--output_dir', type=str, default='./experiment_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    experiment(args)
