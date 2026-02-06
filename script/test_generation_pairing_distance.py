"""
测试脚本：比较生成图像的配对距离分布

从128个噪声生成128个图像（已经配对），然后计算：
1. 随机配对的距离分布
2. OT配对的距离分布
3. 生成关系本身的配对距离分布（原始配对）

对三个模型（CFM, OTCFM, MA_TCFM）分别进行测试和可视化
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

from model import create_model
from torchcfm.optimal_transport import OTPlanSampler
from torchdyn.core import NeuralODE
from utils import set_seed


def generate_images_from_noise(model, noise, dataset_name, device, num_steps=50):
    """从噪声生成图像"""
    model.eval()
    
    with torch.no_grad():
        if dataset_name == 'cifar10':
            # Create ODE solver
            def model_wrapper(t, x, args=None):
                if t.dim() == 0:
                    t = t.expand(x.shape[0])
                return model(x, t)
            
            node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            
            # Generate samples
            t_span = torch.linspace(0, 1, num_steps + 1).to(device)
            traj = node.trajectory(noise, t_span=t_span)
            samples = traj[-1]  # Final samples
            
            # Denormalize
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            
            return samples.cpu()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")


def compute_distances(x0, x1):
    """计算配对的距离（L2距离）"""
    # Flatten spatial dimensions
    x0_flat = x0.reshape(x0.shape[0], -1)
    x1_flat = x1.reshape(x1.shape[0], -1)
    
    # Compute L2 distances
    distances = torch.norm(x0_flat - x1_flat, dim=1)
    return distances.cpu().numpy()


def get_random_pairing(noise, images):
    """随机配对"""
    device = images.device
    indices = torch.randperm(images.shape[0], device=device)
    images_paired = images[indices]
    return noise, images_paired


def get_ot_pairing(noise, images):
    """OT配对 - 使用确定性方法从OT plan中提取最优配对"""
    import scipy.optimize
    
    # OT computation can be done on CPU for efficiency
    noise_cpu = noise.cpu()
    images_cpu = images.cpu()
    
    # Compute OT plan
    ot_sampler = OTPlanSampler(method="exact")
    pi = ot_sampler.get_map(noise_cpu, images_cpu)
    
    # Use Hungarian algorithm to get deterministic optimal pairing
    # This finds the assignment that maximizes the OT plan mass
    # For exact OT, we can use the plan directly to find the optimal assignment
    noise_flat = noise_cpu.reshape(noise_cpu.shape[0], -1)
    images_flat = images_cpu.reshape(images_cpu.shape[0], -1)
    M = torch.cdist(noise_flat, images_flat) ** 2
    
    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
    
    # Return paired samples
    noise_paired = noise_cpu[row_ind]
    images_paired = images_cpu[col_ind]
    
    # Return on same device as input
    return noise_paired.to(noise.device), images_paired.to(images.device)


def get_original_pairing(noise, images):
    """原始配对（生成关系本身）"""
    return noise, images


def get_ma_tcfm_2x_pairing(noise, images):
    """MA_TCFM 2x: 使用2x下采样的模型感知OT配对 - 使用确定性最优配对"""
    import torch.nn.functional as F
    import scipy.optimize
    
    # 2x下采样函数
    def M(x):
        if x.dim() == 4:  # (B, C, H, W)
            return F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        else:
            return x
    
    # 在变换后的空间计算OT plan
    noise_transformed = M(noise)
    images_transformed = M(images)
    
    # Compute cost matrix in transformed space
    noise_flat = noise_transformed.reshape(noise_transformed.shape[0], -1)
    images_flat = images_transformed.reshape(images_transformed.shape[0], -1)
    M_cost = torch.cdist(noise_flat, images_flat) ** 2
    
    # Use Hungarian algorithm to find optimal assignment in transformed space
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M_cost.cpu().numpy())
    
    # Convert to torch tensor
    if isinstance(row_ind, np.ndarray):
        row_ind = torch.from_numpy(row_ind).to(noise.device)
    if isinstance(col_ind, np.ndarray):
        col_ind = torch.from_numpy(col_ind).to(images.device)
    
    # 使用原始（未变换）的noise和images进行配对
    noise_paired = noise[row_ind]
    images_paired = images[col_ind]
    
    return noise_paired, images_paired


def test_model(checkpoint_path, model_name, dataset_name, device, num_samples=128, num_steps=50):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get training args and config
    train_args = checkpoint.get('args', {})
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # Generate noise
    print(f"Generating {num_samples} noise samples...")
    if dataset_name == 'cifar10':
        noise = torch.randn(num_samples, 3, 32, 32).to(device)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Generate images from noise
    print(f"Generating {num_samples} images from noise with {num_steps} steps...")
    images = generate_images_from_noise(model, noise, dataset_name, device, num_steps)
    
    # Move to CPU for distance computation (OT is more efficient on CPU)
    # IMPORTANT: All pairing methods will use the SAME noise and images
    noise_cpu = noise.cpu()
    images_cpu = images.cpu()
    
    # Compute four types of pairing distances
    print("Computing distance distributions...")
    print("Note: All methods use the SAME noise and images for fair comparison")
    
    # 1. Random pairing
    noise_rand, images_rand = get_random_pairing(noise_cpu, images_cpu)
    distances_random = compute_distances(noise_rand, images_rand)
    
    # 2. OT pairing
    noise_ot, images_ot = get_ot_pairing(noise_cpu, images_cpu)
    distances_ot = compute_distances(noise_ot, images_ot)
    
    # 3. MA_TCFM 2x pairing
    noise_ma_tcfm, images_ma_tcfm = get_ma_tcfm_2x_pairing(noise_cpu, images_cpu)
    distances_ma_tcfm = compute_distances(noise_ma_tcfm, images_ma_tcfm)
    
    # 4. Original pairing (generation relationship)
    noise_orig, images_orig = get_original_pairing(noise_cpu, images_cpu)
    distances_original = compute_distances(noise_orig, images_orig)
    
    # Statistics
    stats = {
        'random': {
            'mean': float(np.mean(distances_random)),
            'std': float(np.std(distances_random)),
            'min': float(np.min(distances_random)),
            'max': float(np.max(distances_random)),
        },
        'ot': {
            'mean': float(np.mean(distances_ot)),
            'std': float(np.std(distances_ot)),
            'min': float(np.min(distances_ot)),
            'max': float(np.max(distances_ot)),
        },
        'ma_tcfm_2x': {
            'mean': float(np.mean(distances_ma_tcfm)),
            'std': float(np.std(distances_ma_tcfm)),
            'min': float(np.min(distances_ma_tcfm)),
            'max': float(np.max(distances_ma_tcfm)),
        },
        'original': {
            'mean': float(np.mean(distances_original)),
            'std': float(np.std(distances_original)),
            'min': float(np.min(distances_original)),
            'max': float(np.max(distances_original)),
        }
    }
    
    print("\nDistance Statistics:")
    print(f"  Random pairing:    mean={stats['random']['mean']:.4f}, std={stats['random']['std']:.4f}")
    print(f"  OT pairing:        mean={stats['ot']['mean']:.4f}, std={stats['ot']['std']:.4f}")
    print(f"  MA_TCFM 2x pairing: mean={stats['ma_tcfm_2x']['mean']:.4f}, std={stats['ma_tcfm_2x']['std']:.4f}")
    print(f"  Original pairing:  mean={stats['original']['mean']:.4f}, std={stats['original']['std']:.4f}")
    
    return {
        'distances_random': distances_random,
        'distances_ot': distances_ot,
        'distances_ma_tcfm': distances_ma_tcfm,
        'distances_original': distances_original,
        'stats': stats,
        'model_name': model_name
    }


def plot_results(results, output_dir):
    """绘制结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for result in results:
        model_name = result['model_name']
        distances_random = result['distances_random']
        distances_ot = result['distances_ot']
        distances_ma_tcfm = result['distances_ma_tcfm']
        distances_original = result['distances_original']
        stats = result['stats']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. KDE plot
        ax = axes[0]
        x_min = min(distances_random.min(), distances_ot.min(), distances_ma_tcfm.min(), distances_original.min())
        x_max = max(distances_random.max(), distances_ot.max(), distances_ma_tcfm.max(), distances_original.max())
        x_range = np.linspace(x_min, x_max, 200)
        
        # Compute KDE
        kde_random = gaussian_kde(distances_random)
        kde_ot = gaussian_kde(distances_ot)
        kde_ma_tcfm = gaussian_kde(distances_ma_tcfm)
        kde_original = gaussian_kde(distances_original)
        
        ax.plot(x_range, kde_random(x_range), label='Random Pairing', linewidth=2, alpha=0.8)
        ax.plot(x_range, kde_ot(x_range), label='OT Pairing', linewidth=2, alpha=0.8)
        ax.plot(x_range, kde_ma_tcfm(x_range), label='MA_TCFM 2x Pairing', linewidth=2, alpha=0.8)
        ax.plot(x_range, kde_original(x_range), label='Original Pairing', linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{model_name}: Distance Distribution (KDE)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Histogram
        ax = axes[1]
        ax.hist(distances_random, bins=30, alpha=0.5, label='Random Pairing', density=True)
        ax.hist(distances_ot, bins=30, alpha=0.5, label='OT Pairing', density=True)
        ax.hist(distances_ma_tcfm, bins=30, alpha=0.5, label='MA_TCFM 2x Pairing', density=True)
        ax.hist(distances_original, bins=30, alpha=0.5, label='Original Pairing', density=True, histtype='step', linewidth=2)
        
        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{model_name}: Distance Distribution (Histogram)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Statistics comparison
        ax = axes[2]
        methods = ['Random', 'OT', 'MA_TCFM\n2x', 'Original']
        means = [stats['random']['mean'], stats['ot']['mean'], stats['ma_tcfm_2x']['mean'], stats['original']['mean']]
        stds = [stats['random']['std'], stats['ot']['std'], stats['ma_tcfm_2x']['std'], stats['original']['std']]
        
        x_pos = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, means, width, label='Mean', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, stds, width, label='Std', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Pairing Method', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{model_name}: Statistics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust y-axis to show differences
        y_min = min(min(means), min(stds)) * 0.95
        y_max = max(max(means), max(stds)) * 1.05
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save figure
        output_path = output_dir / f"pairing_distance_comparison_{model_name.lower()}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test generation pairing distance distributions")
    
    # Model checkpoints
    parser.add_argument('--checkpoint_cfm', type=str, required=True,
                       help='Path to CFM checkpoint')
    parser.add_argument('--checkpoint_otcfm', type=str, required=True,
                       help='Path to OTCFM checkpoint')
    parser.add_argument('--checkpoint_ma_tcfm', type=str, required=True,
                       help='Path to MA_TCFM checkpoint')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'mnist'],
                       help='Dataset name')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=128,
                       help='Number of noise samples (and generated images)')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of ODE steps for generation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./exp/experiment_results',
                       help='Directory to save results')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test all three models
    results = []
    
    # CFM
    results.append(test_model(
        args.checkpoint_cfm,
        'CFM',
        args.dataset,
        device,
        args.num_samples,
        args.num_steps
    ))
    
    # OTCFM
    results.append(test_model(
        args.checkpoint_otcfm,
        'OTCFM',
        args.dataset,
        device,
        args.num_samples,
        args.num_steps
    ))
    
    # MA_TCFM
    results.append(test_model(
        args.checkpoint_ma_tcfm,
        'MA_TCFM',
        args.dataset,
        device,
        args.num_samples,
        args.num_steps
    ))
    
    # Plot results
    print("\n" + "="*60)
    print("Plotting results...")
    print("="*60)
    plot_results(results, args.output_dir)
    
    # Save statistics
    output_dir = Path(args.output_dir)
    stats_path = output_dir / "pairing_distance_stats.json"
    import json
    stats_dict = {r['model_name']: r['stats'] for r in results}
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved statistics to {stats_path}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
