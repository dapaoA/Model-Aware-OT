"""
Visualization utilities for denoising process and sample generation.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchdyn.core import NeuralODE

from torchcfm.utils import sample_8gaussians


def visualize_denoising_process(model, dataset_name, device, save_path, num_noises=5, num_steps=50, step_interval=5, seed=42):
    """
    Visualize denoising process.
    Creates a grid: rows = different noise samples, cols = denoising steps.
    
    Args:
        model: Trained model
        dataset_name: Name of the dataset
        device: Device to run on
        save_path: Path to save visualization
        num_noises: Number of different noise samples (rows)
        num_steps: Total number of denoising steps
        step_interval: Interval between visualized steps
        seed: Random seed for reproducibility (default: 42)
    """
    model.eval()
    
    # Save current random state
    rng_state = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    
    # Set fixed seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:
        if dataset_name in ['cifar10', 'mnist']:
            # Image dataset visualization
            num_samples = 10
            
            # Sample random noise with fixed seed
            if dataset_name == 'cifar10':
                x0 = torch.randn(num_samples, 3, 32, 32).to(device)
            else:  # mnist
                x0 = torch.randn(num_samples, 1, 28, 28).to(device)
            
            # Create ODE solver
            def model_wrapper(t, x, **kwargs):
                if t.dim() == 0:
                    t = t.expand(x.shape[0])
                return model(x, t)
            
            node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            
            # Generate trajectory
            with torch.no_grad():
                t_span = torch.linspace(0, 1, num_steps + 1).to(device)
                traj = node.trajectory(x0, t_span=t_span)
            
            # Select steps to visualize
            vis_indices = list(range(0, num_steps + 1, step_interval))
            if vis_indices[-1] != num_steps:
                vis_indices.append(num_steps)
            
            # Create figure
            fig, axes = plt.subplots(num_samples, len(vis_indices), figsize=(len(vis_indices) * 2, num_samples * 2))
            if num_samples == 1:
                axes = axes.reshape(1, -1) if len(vis_indices) > 1 else axes.reshape(1, 1)
            elif len(vis_indices) == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(num_samples):
                for j, step_idx in enumerate(vis_indices):
                    img = traj[step_idx, i].cpu()
                    # Denormalize
                    img = (img + 1) / 2
                    img = img.clamp(0, 1)
                    
                    if dataset_name == 'cifar10':
                        img = img.permute(1, 2, 0)
                    else:  # mnist
                        img = img.squeeze(0)
                    
                    axes[i, j].imshow(img.numpy(), cmap='gray' if dataset_name == 'mnist' else None)
                    axes[i, j].axis('off')
                    if i == 0:
                        axes[i, j].set_title(f'Step {step_idx}', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        else:
            # 2D dataset visualization
            num_samples = num_noises
            
            # Sample from source distribution with fixed seed
            if dataset_name == 'moons':
                x0 = sample_8gaussians(num_samples).to(device)
            else:  # 8gaussians
                x0 = sample_8gaussians(num_samples).to(device)
            
            # Create ODE solver
            def model_wrapper(t, x, **kwargs):
                if t.dim() == 0:
                    t = t.expand(x.shape[0])
                return model(torch.cat([x, t[:, None]], 1))
            
            node = NeuralODE(model_wrapper, solver="euler", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            
            # Generate trajectory
            with torch.no_grad():
                t_span = torch.linspace(0, 1, num_steps + 1).to(device)
                traj = node.trajectory(x0, t_span=t_span)
            
            # Select steps to visualize
            vis_indices = list(range(0, num_steps + 1, step_interval))
            if vis_indices[-1] != num_steps:
                vis_indices.append(num_steps)
            
            # Create figure: rows = noise samples, cols = time steps
            fig, axes = plt.subplots(num_samples, len(vis_indices), figsize=(len(vis_indices) * 2, num_samples * 2))
            if num_samples == 1:
                axes = axes.reshape(1, -1) if len(vis_indices) > 1 else axes.reshape(1, 1)
            elif len(vis_indices) == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(num_samples):
                for j, step_idx in enumerate(vis_indices):
                    traj_2d = traj[step_idx, i].cpu().numpy()
                    axes[i, j].scatter(traj_2d[0], traj_2d[1], s=50, alpha=0.7, c='blue')
                    axes[i, j].set_xlim(-15, 15)
                    axes[i, j].set_ylim(-15, 15)
                    axes[i, j].set_aspect('equal')
                    axes[i, j].axis('off')
                    if i == 0:
                        axes[i, j].set_title(f'Step {step_idx}', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    finally:
        # Restore random state
        torch.set_rng_state(rng_state)
        np.random.set_state(np_rng_state)
    
    model.train()
