"""
测试DCT归一化对OT配对结果的影响
"""
import torch
import numpy as np
from scipy.fft import dctn
from scipy.optimize import linear_sum_assignment
from torchcfm.conditional_flow_matching import MA_ExactOT

def test_normalization_effect():
    """测试归一化是否改变了OT配对结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建测试数据
    batch_size = 32
    x0 = torch.randn(batch_size, 3, 32, 32).to(device)  # 噪声
    x1 = torch.randn(batch_size, 3, 32, 32).to(device)  # 图片（用随机数据模拟）
    
    # 创建DCT提取器
    ma_ot = MA_ExactOT(sigma=0.0, ma_method='dct_4x4')
    
    # 提取DCT特征（不归一化）
    x0_dct = ma_ot.M(x0)  # (B, C*16)
    x1_dct = ma_ot.M(x1)  # (B, C*16)
    
    # 计算未归一化的距离矩阵和OT配对
    cost_matrix_no_norm = torch.cdist(x0_dct, x1_dct) ** 2
    row_ind_no_norm, col_ind_no_norm = linear_sum_assignment(cost_matrix_no_norm.cpu().numpy())
    
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
    
    # 计算归一化后的距离矩阵和OT配对
    cost_matrix_norm = torch.cdist(x0_dct_norm, x1_dct_norm) ** 2
    row_ind_norm, col_ind_norm = linear_sum_assignment(cost_matrix_norm.cpu().numpy())
    
    # 比较配对结果
    pairing_match = np.array_equal(col_ind_no_norm, col_ind_norm)
    
    print("="*60)
    print("DCT归一化对OT配对的影响测试")
    print("="*60)
    print(f"\nBatch size: {batch_size}")
    print(f"Feature dimension: {x0_dct.shape[1]}")
    
    print(f"\n未归一化 - x0 DCT统计:")
    print(f"  Mean per feature: {x0_dct.mean(dim=0)[:5].cpu().numpy()}")
    print(f"  Std per feature: {x0_dct.std(dim=0)[:5].cpu().numpy()}")
    print(f"  Overall mean: {x0_dct.mean():.4f}, std: {x0_dct.std():.4f}")
    
    print(f"\n未归一化 - x1 DCT统计:")
    print(f"  Mean per feature: {x1_dct.mean(dim=0)[:5].cpu().numpy()}")
    print(f"  Std per feature: {x1_dct.std(dim=0)[:5].cpu().numpy()}")
    print(f"  Overall mean: {x1_dct.mean():.4f}, std: {x1_dct.std():.4f}")
    
    print(f"\n归一化后 - x0 DCT统计:")
    print(f"  Mean per feature: {x0_dct_norm.mean(dim=0)[:5].cpu().numpy()}")
    print(f"  Std per feature: {x0_dct_norm.std(dim=0)[:5].cpu().numpy()}")
    print(f"  Overall mean: {x0_dct_norm.mean():.4f}, std: {x0_dct_norm.std():.4f}")
    
    print(f"\n归一化后 - x1 DCT统计:")
    print(f"  Mean per feature: {x1_dct_norm.mean(dim=0)[:5].cpu().numpy()}")
    print(f"  Std per feature: {x1_dct_norm.std(dim=0)[:5].cpu().numpy()}")
    print(f"  Overall mean: {x1_dct_norm.mean():.4f}, std: {x1_dct_norm.std():.4f}")
    
    print(f"\n配对结果比较:")
    print(f"  配对是否相同: {pairing_match}")
    print(f"  未归一化配对: {col_ind_no_norm[:10]}")
    print(f"  归一化后配对: {col_ind_norm[:10]}")
    
    # 计算配对差异
    if not pairing_match:
        diff_count = np.sum(col_ind_no_norm != col_ind_norm)
        print(f"  配对差异数量: {diff_count} / {batch_size} ({diff_count/batch_size*100:.1f}%)")
    
    # 比较距离矩阵
    print(f"\n距离矩阵比较:")
    print(f"  未归一化距离矩阵 - Mean: {cost_matrix_no_norm.mean():.4f}, Std: {cost_matrix_no_norm.std():.4f}")
    print(f"  归一化后距离矩阵 - Mean: {cost_matrix_norm.mean():.4f}, Std: {cost_matrix_norm.std():.4f}")
    
    # 计算距离矩阵的相关性
    cost_no_norm_flat = cost_matrix_no_norm.cpu().numpy().flatten()
    cost_norm_flat = cost_matrix_norm.cpu().numpy().flatten()
    correlation = np.corrcoef(cost_no_norm_flat, cost_norm_flat)[0, 1]
    print(f"  距离矩阵相关性: {correlation:.4f}")
    
    print("\n" + "="*60)
    print("结论:")
    if pairing_match:
        print("归一化没有改变OT配对结果！")
        print("这可能是因为归一化保持了相对距离关系。")
    else:
        print(f"归一化改变了 {diff_count}/{batch_size} 个配对。")
    print("="*60)

if __name__ == "__main__":
    test_normalization_effect()
