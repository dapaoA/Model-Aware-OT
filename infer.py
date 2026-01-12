"""
Inference script for Conditional Flow Matching models.
Loads trained models and generates samples, optionally computes FID.
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torchdyn.core import NeuralODE
from PIL import Image

from model import create_model
from utils import set_seed
from torchcfm.utils import sample_8gaussians

# Import clean-fid for FID computation
try:
    from cleanfid import fid
    CLEANFID_AVAILABLE = True
except ImportError:
    CLEANFID_AVAILABLE = False
    print("Warning: clean-fid not available. FID computation will be skipped.")


def generate_samples(model, dataset_name, device, num_samples, num_steps, save_dir):
    """Generate samples from the model."""
    model.eval()
    
    with torch.no_grad():
        if dataset_name in ['cifar10', 'mnist']:
            # Image dataset
            if dataset_name == 'cifar10':
                x0 = torch.randn(num_samples, 3, 32, 32).to(device)
            else:  # mnist
                x0 = torch.randn(num_samples, 1, 28, 28).to(device)
            
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
            
            # Denormalize and save
            samples = (samples + 1) / 2
            samples = samples.clamp(0, 1)
            
            # Save individual images
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            grid_path = save_dir / "generated_samples_grid.png"
            vutils.save_image(samples, grid_path, nrow=10, padding=2)
            print(f"Saved sample grid to {grid_path}")
            
            # Save individual images (save all for FID computation)
            for i in range(num_samples):
                img_path = save_dir / f"sample_{i:04d}.png"
                vutils.save_image(samples[i], img_path)
            
            return samples.cpu()
            
        else:
            # 2D dataset
            if dataset_name == 'moons':
                x0 = sample_8gaussians(num_samples).to(device)
            else:  # 8gaussians
                x0 = sample_8gaussians(num_samples).to(device)
            
            # Create ODE solver
            def model_wrapper(t, x, args=None):
                if t.dim() == 0:
                    t = t.expand(x.shape[0])
                return model(torch.cat([x, t[:, None]], 1))
            
            node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            
            # Generate samples
            t_span = torch.linspace(0, 1, num_steps + 1).to(device)
            traj = node.trajectory(x0, t_span=t_span)
            samples = traj[-1]  # Final samples
            
            # Save visualization
            import matplotlib.pyplot as plt
            samples_np = samples.cpu().numpy()
            
            plt.figure(figsize=(8, 8))
            plt.scatter(samples_np[:, 0], samples_np[:, 1], s=10, alpha=0.6)
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.title(f"Generated samples ({num_samples} samples)")
            plt.grid(True, alpha=0.3)
            
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "generated_samples.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved samples to {save_path}")
            
            return samples.cpu()
    
    model.train()


def compute_fid(dataset_name, generated_samples_dir, device):
    """Compute FID score using clean-fid."""
    if not CLEANFID_AVAILABLE:
        print("clean-fid not available, skipping FID computation")
        return None
    
    if dataset_name not in ['cifar10', 'mnist']:
        print(f"FID computation not supported for dataset: {dataset_name}")
        return None
    
    try:
        if dataset_name == 'cifar10':
            # Use clean mode with CIFAR-10 (32x32 images)
            try:
                print("Computing FID with clean mode for CIFAR-10")
                fid_score = fid.compute_fid(
                    str(generated_samples_dir),
                    dataset_name='cifar10',
                    mode='clean',
                    device=device,
                    num_workers=4,
                )
                print(f"Successfully computed FID: {fid_score:.4f}")
            except Exception as e:
                print(f"Error computing FID: {e}")
                raise
                
        elif dataset_name == 'mnist':
            # MNIST might not be directly supported, try generic
            print("Note: MNIST FID computation may not be directly supported by clean-fid")
            fid_score = None
        else:
            fid_score = None
        
        return fid_score
    except Exception as e:
        print(f"Error computing FID: {e}")
        print("Skipping FID computation")
        return None


def infer(args):
    """Main inference function."""
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get training args and config
    train_args = checkpoint.get('args', {})
    model_config = checkpoint.get('model_config', {})
    dataset_name = train_args.get('dataset', args.dataset)
    
    if not dataset_name:
        dataset_name = args.dataset
    
    # Create model
    model = create_model(dataset_name, model_config, device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard log directory
    log_dir = output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs saved to: {log_dir}")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples with {args.num_steps} steps...")
    samples = generate_samples(
        model, dataset_name, device, args.num_samples, args.num_steps, output_dir
    )
    
    print(f"Samples generated and saved to {output_dir}")
    
    # Add generated samples to TensorBoard
    if dataset_name in ['cifar10', 'mnist']:
        # For image datasets, add grid to TensorBoard
        try:
            grid_path = output_dir / "generated_samples_grid.png"
            if grid_path.exists():
                img = Image.open(grid_path)
                import numpy as np
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                else:
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
                writer.add_image('generated/samples_grid', img_tensor, 0)
                print(f"Added sample grid to TensorBoard")
        except Exception as e:
            print(f"Warning: Could not add image to TensorBoard: {e}")
    else:
        # For 2D datasets, add scatter plot
        try:
            plot_path = output_dir / "generated_samples.png"
            if plot_path.exists():
                img = Image.open(plot_path)
                import numpy as np
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                else:
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
                writer.add_image('generated/samples_2d', img_tensor, 0)
                print(f"Added 2D plot to TensorBoard")
        except Exception as e:
            print(f"Warning: Could not add image to TensorBoard: {e}")
    
    writer.close()
    
    # Compute FID if requested and available
    if args.compute_fid:
        print("Computing FID score...")
        fid_score = compute_fid(dataset_name, output_dir, device)
        if fid_score is not None:
            print(f"FID score: {fid_score:.4f}")
            # Save FID score
            fid_path = output_dir / "fid_score.txt"
            with open(fid_path, 'w') as f:
                f.write(f"FID Score: {fid_score:.4f}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Num samples: {args.num_samples}\n")
                f.write(f"Num steps: {args.num_steps}\n")
            print(f"FID score saved to {fid_path}")
        else:
            print("FID computation skipped or failed")
    else:
        print("FID computation skipped (use --compute_fid to enable)")
    
    print("Inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Conditional Flow Matching models")
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    
    # Dataset (optional, will use from checkpoint if available)
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['moons', '8gaussians', 'cifar10', 'mnist'],
                       help='Dataset name (optional, will use from checkpoint)')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of ODE steps for generation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                       help='Directory to save generated samples')
    
    # FID
    parser.add_argument('--compute_fid', action='store_true',
                       help='Compute FID score (requires clean-fid and image dataset)')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    infer(args)
