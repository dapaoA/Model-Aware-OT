"""
Compare generation results across different models and sampling steps.
Generates samples with 1, 4, and 10 steps for CFM, OTCFM, DCT 4x4, and Reflow models.
"""
import argparse
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from model import create_model
from utils import set_seed
from torchdyn.core import NeuralODE


def generate_samples(model, dataset_name, device, num_samples, num_steps, seed=42):
    """Generate samples from the model."""
    set_seed(seed)
    model.eval()
    
    with torch.no_grad():
        if dataset_name == 'cifar10':
            x0 = torch.randn(num_samples, 3, 32, 32).to(device)
            
            # Create ODE solver
            def model_wrapper(t, x, args=None):
                if t.dim() == 0:
                    t = t.expand(x.shape[0])
                return model(x, t)
            
            node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            
            # Generate samples
            t_span = torch.linspace(0, 1, num_steps + 1).to(device)
            traj = node.trajectory(x0, t_span=t_span)
            samples = traj[-1]  # Final samples
            
            # Denormalize
            mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
            samples = samples * std + mean
            samples = samples.clamp(0, 1)
            
            return samples.cpu()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    model.train()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    train_args = checkpoint.get('args', {})
    model_config = checkpoint.get('model_config', {})
    dataset_name = train_args.get('dataset', 'cifar10')
    
    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, dataset_name


def create_comparison_grid(all_samples, output_path, num_samples_per_model=16):
    """
    Create a comparison grid showing samples from different models and steps.
    
    all_samples: dict of {model_name: {steps: samples}}
    """
    num_models = len(all_samples)
    num_steps = 3  # 1, 4, 10 steps
    step_values = [1, 4, 10]
    
    fig, axes = plt.subplots(num_models, num_steps, figsize=(15, 5 * num_models))
    if num_models == 1:
        axes = axes.reshape(1, -1)
    
    model_names = list(all_samples.keys())
    
    for model_idx, model_name in enumerate(model_names):
        for step_idx, num_steps_val in enumerate(step_values):
            ax = axes[model_idx, step_idx]
            
            if num_steps_val in all_samples[model_name]:
                samples = all_samples[model_name][num_steps_val]
                # Take first num_samples_per_model samples
                samples_to_show = samples[:num_samples_per_model]
                
                # Convert to grid
                grid = vutils.make_grid(samples_to_show, nrow=4, padding=2, normalize=False)
                grid_np = grid.permute(1, 2, 0).numpy()
                
                ax.imshow(grid_np)
                ax.axis('off')
                
                if model_idx == 0:
                    ax.set_title(f'{num_steps_val} steps', fontsize=12, fontweight='bold')
                if step_idx == 0:
                    ax.text(-0.1, 0.5, model_name, transform=ax.transAxes,
                           rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison grid to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare generation results across models and steps")
    
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate per model/step')
    parser.add_argument('--num_samples_display', type=int, default=16,
                       help='Number of samples to display in grid')
    parser.add_argument('--output_dir', type=str, default='./exp/generation_comparison',
                       help='Directory to save comparison results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define model checkpoints
    models = {
        'CFM': 'models/cifar10_cfm/cfm_cifar10/checkpoint_iter_400000.pt',
        'OTCFM': 'models/cifar10_otcfm/otcfm_cifar10/checkpoint_iter_400000.pt',
        'DCT 4x4': 'models/cifar10_dct_4x4/ma_tcfm_cifar10/checkpoint_iter_400000.pt',
        'Reflow': 'models/cifar10_otcfm_reflow/otcfm_cifar10_reflow/checkpoint_iter_200000.pt',
    }
    
    step_values = [1, 4, 10]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = {}
    
    # Generate samples for each model and step
    for model_name, checkpoint_path in models.items():
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}, skipping {model_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model, dataset_name = load_model(checkpoint_path, device)
        print(f"Model loaded: {model_name}, dataset: {dataset_name}")
        
        all_samples[model_name] = {}
        
        # Generate samples for each step count
        for num_steps in step_values:
            print(f"Generating {args.num_samples} samples with {num_steps} steps...")
            samples = generate_samples(
                model, dataset_name, device, args.num_samples, num_steps, args.seed
            )
            all_samples[model_name][num_steps] = samples
            
            # Save individual grid for this model/step
            grid_path = output_dir / f"{model_name.lower().replace(' ', '_')}_{num_steps}steps_grid.png"
            vutils.save_image(samples[:args.num_samples_display], grid_path, 
                            nrow=4, padding=2)
            print(f"Saved grid to {grid_path}")
        
        # Free up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create comparison grid
    print(f"\n{'='*60}")
    print("Creating comparison grid...")
    print(f"{'='*60}")
    comparison_path = output_dir / "generation_comparison_all_models.png"
    create_comparison_grid(all_samples, comparison_path, args.num_samples_display)
    
    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
