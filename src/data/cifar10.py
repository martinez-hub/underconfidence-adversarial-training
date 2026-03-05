"""CIFAR-10 dataset utilities."""

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train/test DataLoaders.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory to store/load CIFAR-10 data
        augment: Whether to apply data augmentation to training set

    Returns:
        (train_loader, test_loader) tuple
    """
    # CIFAR-10 normalization constants
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Training transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
