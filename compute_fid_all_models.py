#!/usr/bin/env python3
"""
Script to compute FID scores for all three CIFAR-10 models using clean-fid.
"""
import subprocess
import sys
from pathlib import Path

# Model checkpoints (using the latest checkpoint at 400000 iterations)
models = {
    'cfm': 'models/cifar10_cfm/cfm_cifar10/checkpoint_iter_400000.pt',
    'ma_tcfm': 'models/cifar10_ma_tcfm/ma_tcfm_cifar10/checkpoint_iter_400000.pt',
    'otcfm': 'models/cifar10_otcfm/otcfm_cifar10/checkpoint_iter_400000.pt',
}

def run_fid_computation(model_name, checkpoint_path):
    """Run FID computation for a single model."""
    print(f"\n{'='*60}")
    print(f"Computing FID for {model_name.upper()}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    output_dir = f"fid_results/{model_name}"
    
    # Run inference with FID computation
    cmd = [
        sys.executable,
        'infer.py',
        '--checkpoint', checkpoint_path,
        '--dataset', 'cifar10',
        '--num_samples', '5000',
        '--num_steps', '50',
        '--output_dir', output_dir,
        '--compute_fid',
        '--seed', '42'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Successfully computed FID for {model_name}")
        # Try to extract FID score from output
        for line in result.stdout.split('\n'):
            if 'FID score:' in line:
                print(f"  {line}")
        return True
    else:
        print(f"✗ Failed to compute FID for {model_name}")
        print(f"Error: {result.stderr}")
        return False

def main():
    """Main function to compute FID for all models."""
    print("Starting FID computation for all CIFAR-10 models...")
    print(f"Using 5000 samples per model\n")
    
    results = {}
    
    for model_name, checkpoint_path in models.items():
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {model_name}\n")
            continue
        
        success = run_fid_computation(model_name, checkpoint_path)
        results[model_name] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    # Read FID scores from output files
    fid_scores = {}
    for model_name in models.keys():
        fid_file = Path(f"fid_results/{model_name}/fid_score.txt")
        if fid_file.exists():
            with open(fid_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'FID Score:' in line:
                        score = float(line.split(':')[1].strip())
                        fid_scores[model_name] = score
                        print(f"{model_name.upper()}: FID = {score:.4f}")
    
    if fid_scores:
        print(f"\n{'='*60}")
        print("BEST MODEL")
        print(f"{'='*60}")
        best_model = min(fid_scores, key=fid_scores.get)
        best_score = fid_scores[best_model]
        print(f"\n🏆 Best model: {best_model.upper()}")
        print(f"   FID Score: {best_score:.4f}")
        print(f"\nAll scores (lower is better):")
        for model, score in sorted(fid_scores.items(), key=lambda x: x[1]):
            marker = " 🏆" if model == best_model else ""
            print(f"  {model.upper()}: {score:.4f}{marker}")

if __name__ == "__main__":
    main()

