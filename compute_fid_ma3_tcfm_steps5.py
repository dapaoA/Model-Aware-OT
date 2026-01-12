#!/usr/bin/env python3
"""
计算 ma3_tcfm 模型的 FID 分数（使用5 steps），并更新训练曲线图
只计算 100000, 200000, 300000, 400000 这几个 checkpoint
"""
import subprocess
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ma3_tcfm 模型配置
ma3_tcfm_config = {
    'checkpoint_dir': 'models/cifar10_ma3_tcfm/ma3_tcfm_cifar10',
    'name': 'MA3-TCFM'
}

# 要计算的checkpoint迭代次数
iterations = [100000, 200000, 300000, 400000]

def compute_fid_for_checkpoint(model_key, checkpoint_path, output_dir):
    """为单个checkpoint计算FID"""
    print(f"\n计算 {model_key} - {checkpoint_path.name} 的FID (5 steps)...")
    
    cmd = [
        sys.executable,
        'infer.py',
        '--checkpoint', str(checkpoint_path),
        '--dataset', 'cifar10',
        '--num_samples', '5000',
        '--num_steps', '5',  # 使用5 steps
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
        print(f"  ✗ 计算失败: {result.stderr[:200]}")
        return None

def main():
    """主函数"""
    print("="*60)
    print("计算 MA3-TCFM 的 FID 分数 (使用 5 steps)")
    print("="*60)
    
    # 读取已有结果
    existing_file = Path("fid_training_curves_results_steps5.json")
    if existing_file.exists():
        with open(existing_file, 'r') as f:
            all_results = json.load(f)
        print(f"\n已加载已有结果: {existing_file}")
    else:
        all_results = {}
        print(f"\n警告: 未找到已有结果文件，将创建新文件")
    
    # 计算 ma3_tcfm 的 FID
    model_key = 'ma3_tcfm'
    checkpoint_dir = Path(ma3_tcfm_config['checkpoint_dir'])
    model_name = ma3_tcfm_config['name']
    
    print(f"\n{'='*60}")
    print(f"处理模型: {model_name} ({model_key})")
    print(f"{'='*60}")
    
    # 初始化 ma3_tcfm 的结果字典
    if model_key not in all_results:
        all_results[model_key] = {
            'name': model_name,
            'results': {}
        }
    
    results = {}
    
    for iter_num in iterations:
        checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
        
        if not checkpoint_path.exists():
            print(f"  ⚠ Checkpoint不存在: {checkpoint_path}")
            continue
        
        # 检查是否已经计算过
        iter_str = str(iter_num)
        if iter_str in all_results[model_key]['results']:
            print(f"  ⏭ 跳过 {iter_num} (已存在: {all_results[model_key]['results'][iter_str]:.4f})")
            results[iter_num] = all_results[model_key]['results'][iter_str]
            continue
        
        output_dir = Path(f"fid_results/{model_key}_iter_{iter_num}_steps5")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fid_score = compute_fid_for_checkpoint(model_key, checkpoint_path, output_dir)
        
        if fid_score is not None:
            results[iter_num] = fid_score
            all_results[model_key]['results'][iter_str] = fid_score
    
    # 保存结果到JSON文件
    results_file = Path("fid_training_curves_results_steps5.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到: {results_file}")
    
    # 绘制曲线
    print("\n绘制训练曲线...")
    plt.figure(figsize=(10, 6))
    
    # 颜色和标记：蓝色、橙色、绿色、红色（新增）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    # 定义模型顺序
    model_order = ['cfm', 'ma_tcfm', 'otcfm', 'ma3_tcfm']
    
    for idx, model_key in enumerate(model_order):
        if model_key not in all_results:
            continue
            
        model_data = all_results[model_key]
        model_name = model_data['name']
        results = model_data['results']
        
        if not results:
            continue
        
        # 将字符串键转换为整数并排序
        iterations_list = sorted([int(k) for k in results.keys()])
        fid_scores = [results[str(iter_num)] for iter_num in iterations_list]
        
        plt.plot(iterations_list, fid_scores, 
                marker=markers[idx], 
                color=colors[idx],
                linewidth=2,
                markersize=8,
                label=model_name)
    
    plt.xlabel('Training Iterations', fontsize=12)
    plt.ylabel('FID Score', fontsize=12)
    plt.title('FID Score During Training on CIFAR-10 (5 Steps)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plot_file = Path("fid_training_curves_steps5.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"曲线图已保存到: {plot_file}")
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("FID 分数汇总 (5 Steps)")
    print(f"{'='*60}")
    
    # 收集所有迭代次数
    all_iterations = sorted(set(
        [int(k) for model_data in all_results.values() 
         for k in model_data['results'].keys()]
    ))
    
    print(f"{'迭代次数':<12} {'CFM':<12} {'MA-TCFM':<12} {'OT-CFM':<12} {'MA3-TCFM':<12}")
    print("-" * 72)
    
    for iter_num in all_iterations:
        iter_str = str(iter_num)
        row = [f"{iter_num:<12}"]
        for model_key in ['cfm', 'ma_tcfm', 'otcfm', 'ma3_tcfm']:
            if model_key in all_results and iter_str in all_results[model_key]['results']:
                fid = all_results[model_key]['results'][iter_str]
                row.append(f"{fid:<12.4f}")
            else:
                row.append(f"{'N/A':<12}")
        print(" ".join(row))
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
