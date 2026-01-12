#!/usr/bin/env python3
"""
验证FID计算稳定性：使用50000张图片（分批生成）计算3个模型在10000轮时的FID
"""
import subprocess
import sys
from pathlib import Path
import json
import shutil

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

# 分批生成：每次生成10000张，共5批
batch_size = 10000
num_batches = 5
total_images = batch_size * num_batches

def compute_fid_batch(model_key, checkpoint_path, output_dir):
    """分批生成图片并计算FID"""
    print(f"\n计算 {model_key} (分批生成{total_images}张图片, 5 steps)...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 清理之前的图片
    for img_file in output_dir.glob("sample_*.png"):
        img_file.unlink()
    
    # 分批生成图片
    for batch_idx in range(num_batches):
        print(f"  生成第 {batch_idx + 1}/{num_batches} 批 ({batch_size} 张)...")
        
        batch_output_dir = output_dir / f"batch_{batch_idx}"
        batch_output_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            'infer.py',
            '--checkpoint', str(checkpoint_path),
            '--dataset', 'cifar10',
            '--num_samples', str(batch_size),
            '--num_steps', '5',
            '--output_dir', str(batch_output_dir),
            '--seed', str(42 + batch_idx)  # 每批使用不同的seed
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ✗ 第 {batch_idx + 1} 批生成失败")
            error_lines = result.stderr.split('\n')[-5:]
            for line in error_lines:
                if line.strip():
                    print(f"    {line}")
            return None
        
        # 移动图片到主目录，重命名以避免冲突
        for img_file in batch_output_dir.glob("sample_*.png"):
            old_name = img_file.name
            # sample_0000.png -> sample_0XXXX.png (XXXX = batch_idx * batch_size + original_index)
            idx = int(old_name.split('_')[1].split('.')[0])
            new_idx = batch_idx * batch_size + idx
            new_name = f"sample_{new_idx:05d}.png"
            img_file.rename(output_dir / new_name)
        
        # 清理批次目录
        shutil.rmtree(batch_output_dir)
    
    print(f"  所有图片已生成，共 {len(list(output_dir.glob('sample_*.png')))} 张")
    
    # 现在计算FID
    print(f"  计算FID分数...")
    cmd = [
        sys.executable,
        '-c',
        f'''
import sys
sys.path.insert(0, ".")
from cleanfid import fid
fid_score = fid.compute_fid(
    "{output_dir}",
    dataset_name="cifar10",
    mode="clean",
    device="cuda",
    num_workers=4,
)
print(f"FID Score: {{fid_score:.4f}}")
with open("{output_dir}/fid_score.txt", "w") as f:
    f.write(f"FID Score: {{fid_score:.4f}}\\n")
    f.write(f"Dataset: cifar10\\n")
    f.write(f"Num samples: {total_images}\\n")
    f.write(f"Num steps: 5\\n")
'''
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # 从输出中提取FID分数
        for line in result.stdout.split('\n'):
            if 'FID Score:' in line:
                fid_score = float(line.split(':')[1].strip())
                print(f"  ✓ FID: {fid_score:.4f}")
                return fid_score
        
        # 从文件中读取
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
        print(f"  ✗ FID计算失败")
        error_lines = result.stderr.split('\n')[-5:]
        for line in error_lines:
            if line.strip():
                print(f"    {line}")
        return None

def main():
    """主函数"""
    print("="*60)
    print(f"验证FID计算稳定性 ({total_images}张图片, 5 steps)")
    print("计算3个模型在10000轮时的FID")
    print(f"分批生成：每批{batch_size}张，共{num_batches}批")
    print("="*60)
    
    results = {}
    
    for model_key, model_info in models.items():
        checkpoint_path = Path(model_info['checkpoint'])
        model_name = model_info['name']
        
        if not checkpoint_path.exists():
            print(f"\n⚠ Checkpoint不存在: {checkpoint_path}")
            continue
        
        output_dir = Path(f"fid_results/{model_key}_iter_10000_steps5_50k_images")
        
        fid_score = compute_fid_batch(model_key, checkpoint_path, output_dir)
        
        if fid_score is not None:
            results[model_key] = {
                'name': model_name,
                'fid': fid_score,
                'num_images': total_images,
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
    print(f"FID 结果汇总 ({total_images}张图片, 5 steps, iteration 10000)")
    print(f"{'='*60}")
    print(f"{'模型':<12} {'FID分数':<12}")
    print("-" * 30)
    
    for model_key, data in results.items():
        print(f"{data['name']:<12} {data['fid']:<12.4f}")
    
    # 与之前结果的对比
    print(f"\n{'='*60}")
    print("与之前结果对比 (5000张 vs 10000张 vs 50000张)")
    print(f"{'='*60}")
    
    # 读取之前的结果
    previous_file = Path("fid_training_curves_results_steps5.json")
    results_10k_file = Path("fid_stability_verification_10k_images.json")
    
    previous_results = {}
    results_10k = {}
    
    if previous_file.exists():
        with open(previous_file, 'r') as f:
            previous_results = json.load(f)
    
    if results_10k_file.exists():
        with open(results_10k_file, 'r') as f:
            results_10k = json.load(f)
    
    if previous_results or results_10k:
        print(f"{'模型':<12} {'5000张':<12} {'10000张':<12} {'50000张':<12}")
        print("-" * 60)
        
        for model_key in ['cfm', 'ma_tcfm', 'otcfm']:
            if model_key in results:
                model_name = results[model_key]['name']
                fid_50k = results[model_key]['fid']
                
                fid_5k = 'N/A'
                if model_key in previous_results and '10000' in previous_results[model_key]['results']:
                    fid_5k = previous_results[model_key]['results']['10000']
                
                fid_10k = 'N/A'
                if model_key in results_10k:
                    fid_10k = results_10k[model_key]['fid']
                
                print(f"{model_name:<12} {fid_5k if fid_5k != 'N/A' else 'N/A':<12} {fid_10k if fid_10k != 'N/A' else 'N/A':<12} {fid_50k:<12.4f}")
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

