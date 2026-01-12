#!/usr/bin/env python3
"""
只计算10000和50000的FID（5 steps），然后合并已有结果并绘制完整曲线
"""
import subprocess
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt

# 模型配置
models = {
    'cfm': {
        'checkpoint_dir': 'models/cifar10_cfm/cfm_cifar10',
        'name': 'CFM'
    },
    'ma_tcfm': {
        'checkpoint_dir': 'models/cifar10_ma_tcfm/ma_tcfm_cifar10',
        'name': 'MA-TCFM'
    },
    'otcfm': {
        'checkpoint_dir': 'models/cifar10_otcfm/otcfm_cifar10',
        'name': 'OT-CFM'
    }
}

# 只需要计算这个
new_iterations = [20000]

def compute_fid_for_checkpoint(model_key, checkpoint_path, output_dir):
    """为单个checkpoint计算FID"""
    print(f"\n计算 {model_key} - {checkpoint_path.name} 的FID (5 steps)...")
    
    cmd = [
        sys.executable,
        'infer.py',
        '--checkpoint', str(checkpoint_path),
        '--dataset', 'cifar10',
        '--num_samples', '5000',
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
        print(f"  ✗ 计算失败: {result.stderr[:200]}")
        return None

def main():
    """主函数"""
    print("="*60)
    print("计算 10000 和 50000 的 FID (使用 5 steps)")
    print("="*60)
    
    # 读取已有结果
    existing_file = Path("fid_training_curves_results_steps5.json")
    if existing_file.exists():
        with open(existing_file, 'r') as f:
            all_results = json.load(f)
        print(f"\n已加载已有结果: {existing_file}")
    else:
        all_results = {}
        for model_key, model_info in models.items():
            all_results[model_key] = {
                'name': model_info['name'],
                'results': {}
            }
    
    # 只计算新的checkpoint
    for model_key, model_info in models.items():
        checkpoint_dir = Path(model_info['checkpoint_dir'])
        model_name = model_info['name']
        
        print(f"\n{'='*60}")
        print(f"处理模型: {model_name} ({model_key})")
        print(f"{'='*60}")
        
        for iter_num in new_iterations:
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
            
            if not checkpoint_path.exists():
                print(f"  ⚠ Checkpoint不存在: {checkpoint_path}")
                continue
            
            # 检查是否已经计算过
            iter_str = str(iter_num)
            if model_key in all_results and iter_str in all_results[model_key]['results']:
                print(f"  ⏭ 跳过 {iter_num} (已存在: {all_results[model_key]['results'][iter_str]:.4f})")
                continue
            
            output_dir = Path(f"fid_results/{model_key}_iter_{iter_num}_steps5")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fid_score = compute_fid_for_checkpoint(model_key, checkpoint_path, output_dir)
            
            if fid_score is not None:
                if model_key not in all_results:
                    all_results[model_key] = {
                        'name': model_name,
                        'results': {}
                    }
                all_results[model_key]['results'][iter_str] = fid_score
    
    # 保存合并后的结果
    results_file = Path("fid_training_curves_results_steps5.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到: {results_file}")
    
    # 绘制曲线
    print("\n绘制完整训练曲线...")
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
    markers = ['o', 's', '^']
    
    for idx, (model_key, model_data) in enumerate(all_results.items()):
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
    print("FID 分数汇总 (5 Steps) - 完整数据")
    print(f"{'='*60}")
    all_iterations = sorted([int(k) for k in set(
        [iter_str for model_data in all_results.values() 
         for iter_str in model_data['results'].keys()]
    )])
    
    print(f"{'迭代次数':<12} {'CFM':<12} {'MA-TCFM':<12} {'OT-CFM':<12}")
    print("-" * 60)
    
    for iter_num in all_iterations:
        iter_str = str(iter_num)
        row = [f"{iter_num:<12}"]
        for model_key in ['cfm', 'ma_tcfm', 'otcfm']:
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

