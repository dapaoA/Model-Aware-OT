"""
详细测试DCT归一化的影响
"""
import torch
import numpy as np
from scipy.fft import dctn
from scipy.optimize import linear_sum_assignment
from torchcfm.conditional_flow_matching import MA_ExactOT
from dataset import get_dataset

def test_normalization_detailed():
    """详细测试归一化的影响"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用真实的CIFAR-10数据
    dataloader, _ = get_dataset('cifar10', batch_size=32, data_dir='./data')
    x1 = next(iter(dataloader))[0].to(device)  # 真实图片
    x0 = torch.randn(32, 3, 32, 32).to(device)  # 噪声
    
    # 创建DCT提取器
    ma_ot = MA_ExactOT(sigma=0.0, ma_method='dct_4x4')
    
    # 提取DCT特征（不归一化）
    x0_dct = ma_ot.M(x0)  # (B, C*16)
    x1_dct = ma_ot.M(x1)  # (B, C*16)
    
    print("="*60)
    print("归一化前的DCT特征统计")
    print("="*60)
    print(f"\nx0 (噪声) - 每个频率位置的统计:")
    for i in range(min(5, x0_dct.shape[1])):
        mean_val = x0_dct[:, i].mean().item()
        std_val = x0_dct[:, i].std().item()
        min_val = x0_dct[:, i].min().item()
        max_val = x0_dct[:, i].max().item()
        print(f"  频率位置 {i}: mean={mean_val:7.4f}, std={std_val:7.4f}, range=[{min_val:7.4f}, {max_val:7.4f}]")
    
    print(f"\nx1 (CIFAR-10) - 每个频率位置的统计:")
    for i in range(min(5, x1_dct.shape[1])):
        mean_val = x1_dct[:, i].mean().item()
        std_val = x1_dct[:, i].std().item()
        min_val = x1_dct[:, i].min().item()
        max_val = x1_dct[:, i].max().item()
        print(f"  频率位置 {i}: mean={mean_val:7.4f}, std={std_val:7.4f}, range=[{min_val:7.4f}, {max_val:7.4f}]")
    
    # 归一化
    x0_dct_norm = x0_dct.clone()
    x1_dct_norm = x1_dct.clone()
    
    # 对x0归一化
    x0_mean = x0_dct_norm.mean(dim=0, keepdim=True)
    x0_std = x0_dct_norm.std(dim=0, keepdim=True)
    x0_std = torch.clamp(x0_std, min=1e-8)
    x0_dct_norm = (x0_dct_norm - x0_mean) / x0_std
    
    # 对x1归一化
    x1_mean = x1_dct_norm.mean(dim=0, keepdim=True)
    x1_std = x1_dct_norm.std(dim=0, keepdim=True)
    x1_std = torch.clamp(x1_std, min=1e-8)
    x1_dct_norm = (x1_dct_norm - x1_mean) / x1_std
    
    print("\n" + "="*60)
    print("归一化后的DCT特征统计")
    print("="*60)
    print(f"\nx0 (噪声) - 每个频率位置的统计:")
    for i in range(min(5, x0_dct_norm.shape[1])):
        mean_val = x0_dct_norm[:, i].mean().item()
        std_val = x0_dct_norm[:, i].std().item()
        min_val = x0_dct_norm[:, i].min().item()
        max_val = x0_dct_norm[:, i].max().item()
        print(f"  频率位置 {i}: mean={mean_val:7.4f}, std={std_val:7.4f}, range=[{min_val:7.4f}, {max_val:7.4f}]")
    
    print(f"\nx1 (CIFAR-10) - 每个频率位置的统计:")
    for i in range(min(5, x1_dct_norm.shape[1])):
        mean_val = x1_dct_norm[:, i].mean().item()
        std_val = x1_dct_norm[:, i].std().item()
        min_val = x1_dct_norm[:, i].min().item()
        max_val = x1_dct_norm[:, i].max().item()
        print(f"  频率位置 {i}: mean={mean_val:7.4f}, std={std_val:7.4f}, range=[{min_val:7.4f}, {max_val:7.4f}]")
    
    # 计算距离矩阵
    cost_matrix_no_norm = torch.cdist(x0_dct, x1_dct) ** 2
    cost_matrix_norm = torch.cdist(x0_dct_norm, x1_dct_norm) ** 2
    
    # OT配对
    row_ind_no_norm, col_ind_no_norm = linear_sum_assignment(cost_matrix_no_norm.cpu().numpy())
    row_ind_norm, col_ind_norm = linear_sum_assignment(cost_matrix_norm.cpu().numpy())
    
    # 比较
    pairing_match = np.array_equal(col_ind_no_norm, col_ind_norm)
    diff_count = np.sum(col_ind_no_norm != col_ind_norm) if not pairing_match else 0
    
    print("\n" + "="*60)
    print("距离矩阵和配对比较")
    print("="*60)
    print(f"\n未归一化距离矩阵:")
    print(f"  Mean: {cost_matrix_no_norm.mean():.4f}")
    print(f"  Std: {cost_matrix_no_norm.std():.4f}")
    print(f"  Min: {cost_matrix_no_norm.min():.4f}, Max: {cost_matrix_no_norm.max():.4f}")
    print(f"  前5x5样本:\n{cost_matrix_no_norm[:5, :5].cpu().numpy()}")
    
    print(f"\n归一化后距离矩阵:")
    print(f"  Mean: {cost_matrix_norm.mean():.4f}")
    print(f"  Std: {cost_matrix_norm.std():.4f}")
    print(f"  Min: {cost_matrix_norm.min():.4f}, Max: {cost_matrix_norm.max():.4f}")
    print(f"  前5x5样本:\n{cost_matrix_norm[:5, :5].cpu().numpy()}")
    
    # 计算相关性
    cost_no_norm_flat = cost_matrix_no_norm.cpu().numpy().flatten()
    cost_norm_flat = cost_matrix_norm.cpu().numpy().flatten()
    correlation = np.corrcoef(cost_no_norm_flat, cost_norm_flat)[0, 1]
    
    print(f"\n配对结果:")
    print(f"  配对是否相同: {pairing_match}")
    if not pairing_match:
        print(f"  配对差异数量: {diff_count} / {len(col_ind_no_norm)} ({diff_count/len(col_ind_no_norm)*100:.1f}%)")
    print(f"  距离矩阵相关性: {correlation:.4f}")
    
    # 分析为什么相关性高
    print(f"\n分析:")
    print(f"  归一化确实改变了每个频率位置的分布（都变成标准正态）")
    print(f"  但距离矩阵相关性仍然很高 ({correlation:.4f})")
    print(f"  这可能是因为：")
    print(f"    1. 归一化是线性的，保持了相对距离关系")
    print(f"    2. OT配对主要依赖相对距离，而不是绝对距离")
    print(f"    3. 虽然数值变了，但样本之间的相对位置关系可能保持")
    
    # 检查归一化是否真的生效
    print(f"\n验证归一化是否生效:")
    x0_actual_mean = x0_dct_norm.mean(dim=0)
    x0_actual_std = x0_dct_norm.std(dim=0)
    x1_actual_mean = x1_dct_norm.mean(dim=0)
    x1_actual_std = x1_dct_norm.std(dim=0)
    print(f"  x0归一化后 - 前5个频率位置的均值: {x0_actual_mean[:5].cpu().numpy()}")
    print(f"  x0归一化后 - 前5个频率位置的标准差: {x0_actual_std[:5].cpu().numpy()}")
    print(f"  x1归一化后 - 前5个频率位置的均值: {x1_actual_mean[:5].cpu().numpy()}")
    print(f"  x1归一化后 - 前5个频率位置的标准差: {x1_actual_std[:5].cpu().numpy()}")

if __name__ == "__main__":
    test_normalization_detailed()
