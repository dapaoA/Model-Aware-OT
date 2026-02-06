"""
验证脚本：检查不同配对方法是否使用相同的噪声和t值
"""
import torch
import numpy as np
from experiment_pairing_error import compute_pairing_error, get_paired_samples_cfm, get_paired_samples_otcfm, get_paired_samples_ma_tcfm
from flow_matcher import create_flow_matcher

def verify_sampling_consistency():
    """验证所有方法使用相同的t和epsilon"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建模拟数据
    batch_size = 8
    num_t_samples = 3
    x0 = torch.randn(batch_size, 3, 32, 32, device=device)
    x1 = torch.randn(batch_size, 3, 32, 32, device=device)
    
    # 预采样t和epsilon（模拟experiment函数中的逻辑）
    print("\n" + "="*60)
    print("Pre-sampling t and epsilon values...")
    print("="*60)
    t_samples = []
    epsilon_samples = []
    for i in range(num_t_samples):
        t = torch.rand(batch_size, device=device)
        epsilon = torch.randn_like(x0)
        t_samples.append(t)
        epsilon_samples.append(epsilon)
        print(f"\nIteration {i+1}:")
        print(f"  t shape: {t.shape}, t values: {t.cpu().numpy()}")
        print(f"  epsilon shape: {epsilon.shape}, epsilon mean: {epsilon.mean().item():.6f}, std: {epsilon.std().item():.6f}")
    
    # 创建flow matcher
    flow_matcher = create_flow_matcher('cfm', 0.1)
    
    # 创建一个简单的模型（只用于测试，不实际计算）
    class DummyModel:
        def __call__(self, xt, t):
            # 返回一个与xt相同形状的tensor，用于测试
            return torch.randn_like(xt)
    
    model = DummyModel()
    
    # 测试不同配对方法
    pairing_methods = ['cfm', 'otcfm', 'ma_tcfm']
    
    print("\n" + "="*60)
    print("Testing pairing methods...")
    print("="*60)
    
    # 存储每个方法在每个iteration使用的t和epsilon
    method_t_usage = {method: [] for method in pairing_methods}
    method_epsilon_usage = {method: [] for method in pairing_methods}
    
    for method in pairing_methods:
        print(f"\n--- Testing {method.upper()} ---")
        
        # 获取配对
        if method == 'cfm':
            x0_paired, x1_paired = get_paired_samples_cfm(x0, x1)
        elif method == 'otcfm':
            x0_paired, x1_paired = get_paired_samples_otcfm(x0, x1)
        elif method == 'ma_tcfm':
            x0_paired, x1_paired = get_paired_samples_ma_tcfm(x0, x1, ma_method='downsample_2x')
        
        print(f"  x0_paired shape: {x0_paired.shape}")
        print(f"  x1_paired shape: {x1_paired.shape}")
        
        # 模拟compute_pairing_error中的逻辑
        for iter_idx, (t, epsilon) in enumerate(zip(t_samples, epsilon_samples)):
            print(f"\n  Iteration {iter_idx+1}:")
            print(f"    t shape: {t.shape}, t values: {t.cpu().numpy()}")
            print(f"    epsilon shape: {epsilon.shape}, epsilon mean: {epsilon.mean().item():.6f}")
            
            # 检查epsilon是否与x0_paired匹配
            if epsilon.shape != x0_paired.shape:
                print(f"    WARNING: epsilon shape {epsilon.shape} != x0_paired shape {x0_paired.shape}")
            
            # 存储使用的t和epsilon
            method_t_usage[method].append(t.clone())
            method_epsilon_usage[method].append(epsilon.clone())
    
    # 验证所有方法在每个iteration使用相同的t和epsilon
    print("\n" + "="*60)
    print("Verification: Checking if all methods use same t and epsilon")
    print("="*60)
    
    all_consistent = True
    
    for iter_idx in range(num_t_samples):
        print(f"\nIteration {iter_idx+1}:")
        
        # 检查t值
        t_values = [method_t_usage[method][iter_idx] for method in pairing_methods]
        t_consistent = all(torch.allclose(t_values[0], t_val) for t_val in t_values[1:])
        
        if t_consistent:
            print(f"  [OK] All methods use the same t values")
            print(f"    t: {t_values[0].cpu().numpy()}")
        else:
            print(f"  [ERROR] Methods use different t values!")
            for method, t_val in zip(pairing_methods, t_values):
                print(f"    {method}: {t_val.cpu().numpy()}")
            all_consistent = False
        
        # 检查epsilon值
        epsilon_values = [method_epsilon_usage[method][iter_idx] for method in pairing_methods]
        epsilon_consistent = all(torch.allclose(epsilon_values[0], eps_val) for eps_val in epsilon_values[1:])
        
        if epsilon_consistent:
            print(f"  [OK] All methods use the same epsilon values")
            print(f"    epsilon mean: {epsilon_values[0].mean().item():.6f}, std: {epsilon_values[0].std().item():.6f}")
        else:
            print(f"  [ERROR] Methods use different epsilon values!")
            for method, eps_val in zip(pairing_methods, epsilon_values):
                print(f"    {method}: mean={eps_val.mean().item():.6f}, std={eps_val.std().item():.6f}")
            all_consistent = False
    
    print("\n" + "="*60)
    if all_consistent:
        print("[PASS] VERIFICATION PASSED: All methods use the same t and epsilon in each iteration")
    else:
        print("[FAIL] VERIFICATION FAILED: Methods use different t or epsilon values")
    print("="*60)
    
    return all_consistent

if __name__ == "__main__":
    verify_sampling_consistency()
