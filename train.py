"""
Training script for Conditional Flow Matching models.
Supports CFM, OT-CFM, and SB-CFM methods with different datasets.
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from dataset import get_dataset, sample_source_distribution
from flow_matcher import create_flow_matcher
from model import create_model, load_model_config, create_default_config
from utils import set_seed
from utils.common import get_rng_states, set_rng_states
from utils.visualization import visualize_denoising_process
from torchcfm.utils import sample_moons, sample_8gaussians


def train(args):
    """Main training function."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if resuming from checkpoint
    resume_from = getattr(args, 'resume', None)
    if resume_from:
        print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Restore args from checkpoint (but allow override from command line)
        checkpoint_args = checkpoint.get('args', {})
        args.dataset = checkpoint_args.get('dataset', args.dataset)
        args.method = checkpoint_args.get('method', args.method)
        args.batch_size = checkpoint_args.get('batch_size', args.batch_size)
        args.lr = checkpoint_args.get('lr', args.lr)
        args.sigma = checkpoint_args.get('sigma', args.sigma)
        args.seed = checkpoint_args.get('seed', args.seed)
        args.ma_method = checkpoint_args.get('ma_method', getattr(args, 'ma_method', 'downsample_2x'))
        
        # Restore random states for reproducibility
        rng_states = checkpoint.get('rng_states')
        if rng_states:
            set_rng_states(rng_states)
            print("Restored random number generator states")
        else:
            # Fallback: use seed if RNG states not saved
            set_seed(args.seed)
            print(f"RNG states not found in checkpoint, using seed: {args.seed}")
        
        # Restore iteration
        iteration = checkpoint.get('iteration', 0)
        print(f"Resuming from iteration: {iteration}")
    else:
        # Set seed for new training
        set_seed(args.seed)
        iteration = 0
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard log directory (with model name)
    model_name = f"{args.method}_{args.dataset}"
    log_dir = save_dir / model_name / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logs saved to: {log_dir}")
    
    # Load model config
    config_path = args.model_config
    if not Path(config_path).exists():
        print(f"Config file not found at {config_path}, creating default config...")
        create_default_config(config_path)
    
    model_config = load_model_config(config_path, args.dataset)
    
    # Create model
    model = create_model(args.dataset, model_config, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # Load model weights if resuming
    if resume_from:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded from checkpoint")
    
    # Log model info to TensorBoard
    writer.add_text("model/info", f"Method: {args.method}, Dataset: {args.dataset}")
    writer.add_text("model/info", f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    writer.add_text("hyperparameters", f"Batch size: {args.batch_size}, LR: {args.lr}, Sigma: {args.sigma}")
    
    # Get dataset
    dataloader, dataset_type = get_dataset(args.dataset, args.batch_size, args.data_dir)
    
    # Create flow matcher
    sigma = args.sigma
    if args.method == 'sbcfm' and sigma <= 0:
        sigma = 0.5  # SB-CFM needs positive sigma
    ma_method = getattr(args, 'ma_method', 'downsample_2x')
    flow_matcher = create_flow_matcher(args.method, sigma, ma_method=ma_method)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Load optimizer state if resuming
    if resume_from:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded from checkpoint")
    
    # Training loop
    if resume_from:
        print(f"Resuming training: {args.method} on {args.dataset}")
    else:
        print(f"Starting training: {args.method} on {args.dataset}")
    print(f"Total iterations: {args.iterations}, Save every: {args.save_iter}")
    
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(total=args.iterations, desc="Training", unit="iter")
    
    while iteration < args.iterations:
        if dataloader is not None:
            # Image dataset
            for batch in dataloader:
                if iteration >= args.iterations:
                    break
                
                optimizer.zero_grad()
                
                if args.dataset == 'cifar10':
                    x1 = batch[0].to(device)
                    x0 = torch.randn_like(x1)
                else:  # mnist
                    x1 = batch[0].to(device)
                    x0 = torch.randn_like(x1)
                
                t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
                
                if args.dataset in ['cifar10', 'mnist']:
                    vt = model(xt, t)
                else:
                    vt = model(torch.cat([xt, t[:, None]], dim=-1))
                
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()
                optimizer.step()
                
                iteration += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log to TensorBoard
                writer.add_scalar('train/loss', loss.item(), iteration)
                
                if iteration % args.log_iter == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / iteration
                    remaining = (args.iterations - iteration) * avg_time
                    tqdm.write(f"Iteration {iteration}: loss {loss.item():.4f}, time {elapsed:.2f}s, ETA: {remaining:.2f}s")
                
                if iteration % args.save_iter == 0:
                    # Save model
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': iteration,
                        'args': vars(args),
                        'model_config': model_config,
                        'rng_states': get_rng_states(),  # Save RNG states for reproducibility
                    }
                    checkpoint_path = save_dir / model_name / f"checkpoint_iter_{iteration}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)
                    tqdm.write(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Visualize denoising
                    vis_path = save_dir / model_name / f"denoising_iter_{iteration}.png"
                    vis_path.parent.mkdir(parents=True, exist_ok=True)
                    visualize_denoising_process(
                        model, args.dataset, device, str(vis_path),
                        num_steps=args.vis_steps, step_interval=args.vis_step_interval
                    )
                    tqdm.write(f"Saved visualization to {vis_path}")
                    
                    # Add image to TensorBoard
                    try:
                        img = Image.open(vis_path)
                        import numpy as np
                        img_array = np.array(img)
                        # Convert to CHW format for TensorBoard
                        if len(img_array.shape) == 3:
                            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                        else:
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
                        writer.add_image('visualization/denoising', img_tensor, iteration)
                    except Exception as e:
                        tqdm.write(f"Warning: Could not add image to TensorBoard: {e}")
        else:
            # 2D dataset - loop training
            while iteration < args.iterations:
                optimizer.zero_grad()
                
                x0 = sample_source_distribution(args.dataset, args.batch_size, device)
                if dataset_type == 'moons':
                    x1 = sample_moons(args.batch_size).to(device)
                else:  # 8gaussians
                    x1 = sample_8gaussians(args.batch_size).to(device)
                
                t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
                vt = model(torch.cat([xt, t[:, None]], dim=-1))
                
                loss = torch.mean((vt - ut) ** 2)
                loss.backward()
                optimizer.step()
                
                iteration += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log to TensorBoard
                writer.add_scalar('train/loss', loss.item(), iteration)
                
                if iteration % args.log_iter == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / iteration
                    remaining = (args.iterations - iteration) * avg_time
                    tqdm.write(f"Iteration {iteration}: loss {loss.item():.4f}, time {elapsed:.2f}s, ETA: {remaining:.2f}s")
                
                if iteration % args.save_iter == 0:
                    # Save model
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration': iteration,
                        'args': vars(args),
                        'model_config': model_config,
                        'rng_states': get_rng_states(),  # Save RNG states for reproducibility
                    }
                    checkpoint_path = save_dir / model_name / f"checkpoint_iter_{iteration}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)
                    tqdm.write(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Visualize denoising
                    vis_path = save_dir / model_name / f"denoising_iter_{iteration}.png"
                    vis_path.parent.mkdir(parents=True, exist_ok=True)
                    visualize_denoising_process(
                        model, args.dataset, device, str(vis_path),
                        num_steps=args.vis_steps, step_interval=args.vis_step_interval
                    )
                    tqdm.write(f"Saved visualization to {vis_path}")
                    
                    # Add image to TensorBoard
                    try:
                        img = Image.open(vis_path)
                        import numpy as np
                        img_array = np.array(img)
                        # Convert to CHW format for TensorBoard
                        if len(img_array.shape) == 3:
                            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                        else:
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
                        writer.add_image('visualization/denoising', img_tensor, iteration)
                    except Exception as e:
                        tqdm.write(f"Warning: Could not add image to TensorBoard: {e}")
    
    pbar.close()
    writer.close()
    print("Training completed!")
    print(f"Final checkpoint saved at: {save_dir / model_name / f'checkpoint_iter_{iteration}.pt'}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"View with: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching models")
    
    # Training method
    parser.add_argument('--method', type=str, default='cfm',
                       choices=['cfm', 'otcfm', 'sbcfm', 'ma_otcfm', 'ma_tcfm'],
                       help='Training method: cfm, otcfm, sbcfm, ma_otcfm, or ma_tcfm')
    
    # Model-aware method (for ma_otcfm/ma_tcfm)
    parser.add_argument('--ma_method', type=str, default='downsample_2x',
                       choices=['downsample_2x', 'low_pass'],
                       help='Model-aware transformation method (for ma_otcfm/ma_tcfm): downsample_2x (default) or low_pass')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='moons',
                       choices=['moons', '8gaussians', 'cifar10', 'mnist'],
                       help='Dataset to use')
    
    # Model config
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration yaml file')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--iterations', type=int, default=20000,
                       help='Total training iterations')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--sigma', type=float, default=0.1,
                       help='Sigma parameter for flow matching')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models and visualizations')
    parser.add_argument('--save_iter', type=int, default=5000,
                       help='Save checkpoint every N iterations')
    parser.add_argument('--log_iter', type=int, default=1000,
                       help='Log loss every N iterations')
    
    # Visualization
    parser.add_argument('--vis_steps', type=int, default=50,
                       help='Number of denoising steps for visualization')
    parser.add_argument('--vis_step_interval', type=int, default=5,
                       help='Interval between visualized steps')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset storage')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    train(args)
