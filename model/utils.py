"""
Model utilities for creating and loading models.
"""
import os
import yaml
import torch

from torchcfm.models.models import MLP

# Import UNet for image datasets
try:
    from model.unet_expert import UNetExpert
except ImportError:
    UNetExpert = None

try:
    from torchcfm.models.unet.unet import UNetModelWrapper
except ImportError:
    UNetModelWrapper = None

# Import DiT for image datasets
try:
    from model.dit import DiT
except ImportError:
    DiT = None


def load_model_config(config_path, dataset_name):
    """Load model configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset-specific config
    if dataset_name in config:
        return config[dataset_name]
    elif 'default' in config:
        return config['default']
    else:
        raise ValueError(f"No config found for dataset {dataset_name} in {config_path}")


def create_model(dataset_name, config, device):
    """Create model based on dataset and config.
    
    Args:
        dataset_name: Name of the dataset
        config: Model configuration dictionary
        device: Device to place model on
        
    Returns:
        Model instance
    """
    if dataset_name in ['cifar10', 'mnist']:
        # Image dataset - use UNet or DiT based on config
        model_type = config.get('model_type', 'unet')  # Default to UNet for backward compatibility
        
        if model_type == 'dit':
            # Use DiT (Diffusion Transformer)
            if DiT is None:
                raise ImportError("DiT implementation not found. Please ensure model/dit.py exists.")
            
            if dataset_name == 'cifar10':
                input_shape = (3, 32, 32)
            else:  # mnist
                input_shape = (1, 28, 28)
            
            model = DiT(
                input_shape=input_shape,
                patch_size=config.get('patch_size', 2),
                depth=config.get('depth', 10 if dataset_name == 'cifar10' else 6),
                hidden_size=config.get('hidden_size', 256 if dataset_name == 'cifar10' else 64),
                num_heads=config.get('num_heads', 8 if dataset_name == 'cifar10' else 4),
                mlp_ratio=config.get('mlp_ratio', 4.0),
                dropout=config.get('dropout', 0.1),
                time_emb_dim=config.get('time_emb_dim', 256 if dataset_name == 'cifar10' else 64),
                class_cond=config.get('class_cond', False),
                num_classes=config.get('num_classes', None),
            ).to(device)
        else:
            # Use UNet (default)
            if UNetExpert is not None:
                model = UNetExpert(
                    in_channels=config.get('in_channels', 3),
                    out_channels=config.get('out_channels', 3),
                    time_emb_dim=config.get('time_emb_dim', 128),
                    base_channels=config.get('base_channels', 32),
                    channel_multipliers=config.get('channel_multipliers', [1, 2, 4]),
                    num_res_blocks=config.get('num_res_blocks', 2),
                    dropout=config.get('dropout', 0.1),
                ).to(device)
            elif UNetModelWrapper is not None:
                dim = (3, 32, 32) if dataset_name == 'cifar10' else (1, 28, 28)
                model = UNetModelWrapper(
                    dim=dim,
                    num_res_blocks=config.get('num_res_blocks', 2),
                    num_channels=config.get('num_channels', 128),
                    channel_mult=config.get('channel_mult', [1, 2, 2, 2]),
                    num_heads=config.get('num_heads', 4),
                    num_head_channels=config.get('num_head_channels', 64),
                    attention_resolutions=config.get('attention_resolutions', "16"),
                    dropout=config.get('dropout', 0.1),
                ).to(device)
            else:
                raise ImportError("No UNet implementation found. Please ensure model/unet_expert.py exists.")
    else:
        # 2D dataset - use MLP
        dim = config.get('dim', 2)
        model = MLP(dim=dim, time_varying=True, w=config.get('width', 64)).to(device)
    
    return model


def create_default_config(config_path):
    """Create default model configuration file if it doesn't exist."""
    default_config = {
        'moons': {'dim': 2, 'width': 64},
        '8gaussians': {'dim': 2, 'width': 64},
        'cifar10': {
            'in_channels': 3,
            'out_channels': 3,
            'time_emb_dim': 128,
            'base_channels': 32,
            'channel_multipliers': [1, 2, 4],
            'num_res_blocks': 2,
            'dropout': 0.1,
        },
        'mnist': {
            'in_channels': 1,
            'out_channels': 1,
            'time_emb_dim': 128,
            'base_channels': 32,
            'channel_multipliers': [1, 2, 4],
            'num_res_blocks': 2,
            'dropout': 0.1,
        },
    }
    
    os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f)
    
    return default_config
