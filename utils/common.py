"""
Common utility functions.
"""
import random

import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_states():
    """Get all random number generator states for reproducibility.
    
    Returns:
        dict: Dictionary containing all RNG states
    """
    states = {
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
    
    if torch.cuda.is_available():
        states['torch_cuda_rng_states'] = torch.cuda.get_rng_state_all()
    
    return states


def set_rng_states(states):
    """Restore all random number generator states.
    
    Args:
        states: Dictionary containing RNG states from get_rng_states()
    """
    if 'torch_rng_state' in states:
        torch.set_rng_state(states['torch_rng_state'])
    if 'numpy_rng_state' in states:
        np.random.set_state(states['numpy_rng_state'])
    if 'python_rng_state' in states:
        random.setstate(states['python_rng_state'])
    if 'torch_cuda_rng_states' in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states['torch_cuda_rng_states'])
