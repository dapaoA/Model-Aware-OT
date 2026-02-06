"""
分析随机配对时余弦相似度为什么这么高的原因

关键问题：为什么随机配对(x0, x1)时，理论方向 ut = x1 - x0 和模型预测方向 vt 的余弦相似度这么高？
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import get_dataset
from model import create_model, load_model_config
from flow_matcher import create_flow_matcher

def analyze_cosine_similarity():
    """分析随机配对时余弦相似度的分布和原因"""
    
    # 加载模型和数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载训练好的 CFM 模型（随机配对训练的）
    checkpoint_path = 'models/cifar10_cfm/cfm_cifar10/checkpoint_iter_400000.pt'
    config = load_model_config(checkpoint_path)
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载数据
    train_loader, _ = get_dataset('cifar10', batch_size=128, data_dir='./data')
    x1 = next(iter(train_loader))[0].to(device)
    x0 = torch.randn_like(x1)
    
    # 测试不同的时间t
    t_values = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, t_val in enumerate(t_values):
        t = torch.full((x1.shape[0],), t_val, device=device, dtype=torch.float32)
        
        # 随机配对
        indices = torch.randperm(x1.shape[0], device=x1.device)
        x1_paired = x1[indices]
        
        # 计算理论方向
        ut_theoretical = x1_paired - x0  # (B, C, H, W)
        
        # 计算 xt（在随机配对下）
        t_expanded = t.reshape(-1, *([1] * (x0.dim() - 1)))
        xt = t_expanded * x1_paired + (1 - t_expanded) * x0
        
        # 模型预测
        with torch.no_grad():
            vt_predicted = model(xt, t)  # (B, C, H, W)
        
        # 计算余弦相似度
        ut_flat = ut_theoretical.reshape(ut_theoretical.shape[0], -1)
        vt_flat = vt_predicted.reshape(vt_predicted.shape[0], -1)
        
        dot_product = (ut_flat * vt_flat).sum(dim=1)
        norm_ut = torch.norm(ut_flat, dim=1)
        norm_vt = torch.norm(vt_flat, dim=1)
        cos_sim = dot_product / (norm_ut * norm_vt + 1e-8)
        
        # 关键分析：模型预测的vt是否真的接近x1_paired - x0？
        # 还是说模型学会了"从xt推断出应该朝着某个方向去噪"？
        
        # 1. 分析 vt 和 ut 的范数
        # 2. 分析 vt 的主要成分方向
        # 3. 分析在随机配对下，xt 包含的信息量
        
        # 额外分析：如果训练时模型学习的是随机配对，那么给定 xt = t*x1 + (1-t)*x0
        # 模型会预测 vt = x1 - x0，即使这个配对是随机的
        # 这是因为模型学习的"局部模式"：给定 xt，输出 x1 - x0 的方向
        
        axes[idx].hist(cos_sim.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
        axes[idx].axvline(cos_sim.mean().item(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cos_sim.mean().item():.3f}')
        axes[idx].set_xlabel('Cosine Similarity', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].set_title(f't = {t_val:.2f}\nMean: {cos_sim.mean().item():.3f}, Std: {cos_sim.std().item():.3f}', fontsize=12)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        
        print(f"\nt = {t_val:.2f}:")
        print(f"  Mean cosine similarity: {cos_sim.mean().item():.3f}")
        print(f"  Std cosine similarity: {cos_sim.std().item():.3f}")
        print(f"  Mean ||ut||: {norm_ut.mean().item():.3f}")
        print(f"  Mean ||vt||: {norm_vt.mean().item():.3f}")
        
        # 关键观察：vt 和 ut 的范数比
        norm_ratio = norm_vt / (norm_ut + 1e-8)
        print(f"  Mean ||vt||/||ut||: {norm_ratio.mean().item():.3f}")
    
    plt.tight_layout()
    Path("exp").mkdir(parents=True, exist_ok=True)
    plt.savefig('exp/random_pairing_cosine_analysis.png', dpi=150, bbox_inches='tight')
    print("\nAnalysis saved to: exp/random_pairing_cosine_analysis.png")
    
    # 关键洞察：
    print("\n" + "="*60)
    print("关键洞察：")
    print("="*60)
    print("1. 如果训练时使用随机配对（CFM），模型学会了以下模式：")
    print("   给定 xt = t*x1 + (1-t)*x0（其中x0和x1可能随机配对），")
    print("   模型学习预测 vt = x1 - x0")
    print()
    print("2. 在测试时，即使使用随机配对：")
    print("   - xt = t*x1_paired + (1-t)*x0 包含了 x1_paired 和 x0 的信息")
    print("   - 模型会预测 vt ≈ x1_paired - x0（因为它训练时就是这样学的）")
    print("   - 所以余弦相似度高，即使配对是随机的")
    print()
    print("3. 这不是说随机配对'好'，而是说：")
    print("   - 模型学会了局部模式：从 xt 推断方向")
    print("   - 但这个方向可能不是'最优的'全局配对方向")
    print("   - 真正的测试应该是：用模型生成图像，看质量如何")
    print("="*60)


if __name__ == '__main__':
    analyze_cosine_similarity()
