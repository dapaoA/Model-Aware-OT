"""
Generate comparison images from three trained models (CFM, OTCFM, MA_TCFM).
All models use the same initial noise for fair comparison.
Output: 3 rows x 10 columns = 30 images total.
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from torchdyn.core import NeuralODE

from model import create_model
from utils import set_seed


def generate_samples_from_noise(model, noise, dataset_name, device, num_steps):
    """Generate samples from given noise using the model."""
    model.eval()
    
    with torch.no_grad():
        # Create ODE solver
        def model_wrapper(t, x, args=None):
            if t.dim() == 0:
                t = t.expand(x.shape[0])
            return model(x, t)
        
        node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        
        # Generate samples
        t_span = torch.linspace(0, 1, num_steps + 1).to(device)
        traj = node.trajectory(noise, t_span=t_span)
        samples = traj[-1]  # Final samples
        
        # Denormalize
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)
        
        return samples.cpu()


def load_model(checkpoint_path, dataset_name, device):
    """Load model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get training args and config
    train_args = checkpoint.get('args', {})
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = create_model(dataset_name, model_config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def main(args):
    """Main function to generate comparison images."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check dataset
    if args.dataset != 'cifar10':
        raise ValueError(f"This script currently only supports CIFAR-10. Got: {args.dataset}")
    
    # Generate shared noise (same for all three models)
    # All three models will use the SAME 10 noise samples as starting points
    print(f"\nGenerating shared noise for all models...")
    num_samples = 10  # 10 samples per model, all using the same noise
    
    if args.dataset == 'cifar10':
        shared_noise = torch.randn(num_samples, 3, 32, 32).to(device)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print(f"Shared noise shape: {shared_noise.shape}")
    print(f"  All three models will use the SAME {num_samples} noise samples")
    
    # Load three models
    models = {}
    model_names = ['cfm', 'otcfm', 'ma_tcfm']
    checkpoint_paths = {
        'cfm': args.checkpoint_cfm,
        'otcfm': args.checkpoint_otcfm,
        'ma_tcfm': args.checkpoint_ma_tcfm,
    }
    
    print(f"\nLoading models...")
    for model_name in model_names:
        checkpoint_path = checkpoint_paths[model_name]
        models[model_name] = load_model(checkpoint_path, args.dataset, device)
        print(f"  {model_name.upper()}: Loaded")
    
    # Generate samples from each model using the SAME shared noise
    print(f"\nGenerating samples with {args.num_steps} steps...")
    print(f"  All models use the SAME {num_samples} noise samples as starting points")
    all_samples = []
    
    for model_name in model_names:
        print(f"  {model_name.upper()}: Generating from shared noise...")
        samples = generate_samples_from_noise(
            models[model_name], shared_noise, args.dataset, device, args.num_steps
        )
        all_samples.append(samples)
        print(f"    Generated {samples.shape[0]} samples")
    
    # Concatenate all samples: [CFM(10), OTCFM(10), MA_TCFM(10)] = 30 total
    all_samples_tensor = torch.cat(all_samples, dim=0)
    print(f"\nTotal samples shape: {all_samples_tensor.shape}")
    
    # Save grid image (3 rows x 10 columns)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    vutils.save_image(all_samples_tensor, output_path, nrow=10, padding=2, normalize=False)
    print(f"\nSaved comparison grid to: {output_path}")
    print(f"  Layout: 3 rows x 10 columns")
    print(f"  Row 1: CFM")
    print(f"  Row 2: OTCFM")
    print(f"  Row 3: MA_TCFM")
    print(f"  Each column uses the SAME initial noise across all three models")
    print(f"  Generation steps: {args.num_steps}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comparison images from three trained models (CFM, OTCFM, MA_TCFM)"
    )
    
    # Checkpoints
    parser.add_argument('--checkpoint_cfm', type=str, required=True,
                       help='Path to CFM checkpoint file')
    parser.add_argument('--checkpoint_otcfm', type=str, required=True,
                       help='Path to OTCFM checkpoint file')
    parser.add_argument('--checkpoint_ma_tcfm', type=str, required=True,
                       help='Path to MA_TCFM checkpoint file')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10'],
                       help='Dataset name (currently only cifar10 is supported)')
    
    # Generation parameters
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of ODE steps for generation')
    
    # Output
    parser.add_argument('--output_path', type=str, default='./comparison_three_models.png',
                       help='Path to save the comparison image (3 rows x 10 columns)')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for noise generation')
    
    args = parser.parse_args()
    main(args)
