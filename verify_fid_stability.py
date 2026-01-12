#!/usr/bin/env python3
"""
验证FID计算的稳定性：使用10000张图片计算3个模型在10000轮时的FID
"""
import subprocess
import sys
from pathlib import Path
import json

# 模型配置
models = {
    'cfm': {
        'checkpoint': 'models/cifar10_cfm/cfm_cifar10/checkpoint_iter_10000.pt',
        'name': 'CFM'
    },
    'ma_tcfm': {
        'checkpoint': 'models/cifar10_ma_tcfm/ma_tcfm_cifar10/checkpoint_iter_10000.pt',
        'name': 'MA-TCFM'
    },
    'otcfm': {
        'checkpoint': 'models/cifar10_otcfm/otcfm_cifar10/checkpoint_iter_10000.pt',
        'name': 'OT-CFM'
    }
}

def compute_fid(model_key, checkpoint_path, output_dir):
    """计算FID"""
    print(f"\n计算 {model_key} (50000张图片, 5 steps)...")
    
    cmd = [
        sys.executable,
        'infer.py',
        '--checkpoint', str(checkpoint_path),
        '--dataset', 'cifar10',
        '--num_samples', '50000',  # 使用50000张图片（标准要求）
        '--num_steps', '5',
        '--output_dir', str(output_dir),
        '--compute_fid',
        '--seed', '42'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # 从输出中提取FID分数
        for line in result.stdout.split('\n'):
            if 'Successfully computed FID:' in line:
                fid_score = float(line.split(':')[1].strip())
                print(f"  ✓ FID: {fid_score:.4f}")
                return fid_score
            elif 'FID score:' in line and 'FID score saved' not in line:
                try:
                    fid_score = float(line.split(':')[1].strip())
                    print(f"  ✓ FID: {fid_score:.4f}")
                    return fid_score
                except:
                    pass
        
        # 如果从输出中没找到，尝试从文件中读取
        fid_file = output_dir / "fid_score.txt"
        if fid_file.exists():
            with open(fid_file, 'r') as f:
                for line in f:
                    if 'FID Score:' in line:
                        fid_score = float(line.split(':')[1].strip())
                        print(f"  ✓ FID (从文件): {fid_score:.4f}")
                        return fid_score
        
        print(f"  ✗ 无法提取FID分数")
        return None
    else:
        print(f"  ✗ 计算失败")
        print(f"  Return code: {result.returncode}")
        # 打印最后几行错误信息
        error_lines = result.stderr.split('\n')[-10:]
        for line in error_lines:
            if line.strip():
                print(f"  {line}")
        return None

def main():
    """主函数"""
    print("="*60)
    print("验证FID计算稳定性 (50000张图片, 5 steps)")
    print("计算3个模型在10000轮时的FID")
    print("="*60)
    
    results = {}
    
    for model_key, model_info in models.items():
        checkpoint_path = Path(model_info['checkpoint'])
        model_name = model_info['name']
        
        if not checkpoint_path.exists():
            print(f"\n⚠ Checkpoint不存在: {checkpoint_path}")
            continue
        
        output_dir = Path(f"fid_results/{model_key}_iter_10000_steps5_50k_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fid_score = compute_fid(model_key, checkpoint_path, output_dir)
        
        if fid_score is not None:
            results[model_key] = {
                'name': model_name,
                'fid': fid_score,
                'num_images': 50000,
                'num_steps': 5,
                'iteration': 10000
            }
    
    # 保存结果
    results_file = Path("fid_stability_verification_50k_images.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {results_file}")
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("FID 结果汇总 (50000张图片, 5 steps, iteration 10000)")
    print(f"{'='*60}")
    print(f"{'模型':<12} {'FID分数':<12}")
    print("-" * 30)
    
    for model_key, data in results.items():
        print(f"{data['name']:<12} {data['fid']:<12.4f}")
    
    # 与之前5000和10000张图片的结果对比
    print(f"\n{'='*60}")
    print("与之前结果对比 (5000张 vs 10000张 vs 50000张)")
    print(f"{'='*60}")
    
    # 读取之前的结果
    previous_file = Path("fid_training_curves_results_steps5.json")
    if previous_file.exists():
        with open(previous_file, 'r') as f:
            previous_results = json.load(f)
        
        print(f"{'模型':<12} {'5000张':<12} {'10000张':<12} {'50000张':<12}")
        print("-" * 60)
        
        # 读取10000张图片的结果
        results_10k_file = Path("fid_stability_verification_10k_images.json")
        results_10k = {}
        if results_10k_file.exists():
            with open(results_10k_file, 'r') as f:
                results_10k = json.load(f)
        
        for model_key in ['cfm', 'ma_tcfm', 'otcfm']:
            if model_key in results and model_key in previous_results:
                if '10000' in previous_results[model_key]['results']:
                    fid_5k = previous_results[model_key]['results']['10000']
                    fid_10k = results_10k.get(model_key, {}).get('fid', 'N/A')
                    fid_50k = results[model_key]['fid']
                    model_name = results[model_key]['name']
                    if fid_10k != 'N/A':
                        print(f"{model_name:<12} {fid_5k:<12.4f} {fid_10k:<12.4f} {fid_50k:<12.4f}")
                    else:
                        print(f"{model_name:<12} {fid_5k:<12.4f} {'N/A':<12} {fid_50k:<12.4f}")
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

