"""
Dataset utilities for loading and sampling data.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torchcfm.utils import sample_8gaussians, sample_moons


def get_dataset(dataset_name, batch_size, data_dir='./data'):
    """Get dataset and dataloader.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist', 'moons', '8gaussians')
        batch_size: Batch size for dataloader
        data_dir: Directory to store/download datasets
        
    Returns:
        Tuple of (dataloader, dataset_type)
        - For image datasets: (DataLoader, None)
        - For 2D datasets: (None, dataset_type_string)
    """
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Keep workers alive between epochs to avoid Windows spawn overhead
        )
        return dataloader, None
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,  # Keep workers alive between epochs to avoid Windows spawn overhead
        )
        return dataloader, None
    elif dataset_name == 'moons':
        return None, 'moons'
    elif dataset_name == '8gaussians':
        return None, '8gaussians'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def sample_data(dataset_name, batch_size, device):
    """Sample data from dataset.
    
    Args:
        dataset_name: Name of the dataset ('moons', '8gaussians')
        batch_size: Number of samples to generate
        device: Device to place data on
        
    Returns:
        Tensor of sampled data
    """
    if dataset_name == 'moons':
        return sample_moons(batch_size).to(device)
    elif dataset_name == '8gaussians':
        return sample_8gaussians(batch_size).to(device)
    else:
        raise ValueError(f"Cannot sample from {dataset_name} directly. Use get_dataset() for image datasets.")


def sample_source_distribution(dataset_name, num_samples, device):
    """Sample from source distribution (for 2D datasets).
    
    Args:
        dataset_name: Name of the dataset
        num_samples: Number of samples
        device: Device to place data on
        
    Returns:
        Tensor of sampled data
    """
    if dataset_name == 'moons':
        # For moons, source is 8gaussians
        return sample_8gaussians(num_samples).to(device)
    elif dataset_name == '8gaussians':
        return sample_8gaussians(num_samples).to(device)
    else:
        # For image datasets, source is standard normal
        raise ValueError(f"Use torch.randn_like() for image datasets, not this function")
