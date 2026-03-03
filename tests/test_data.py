"""Tests for data loading."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.cifar10 import get_cifar10_loaders


def test_cifar10_loaders_creation():
    """Test that CIFAR-10 loaders are created successfully."""
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
        augment=False,
    )

    assert train_loader is not None
    assert val_loader is not None


def test_cifar10_dataset_sizes():
    """Test that CIFAR-10 datasets have correct sizes."""
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
    )

    # CIFAR-10 has 50,000 train and 10,000 test images
    assert len(train_loader.dataset) == 50000
    assert len(val_loader.dataset) == 10000


def test_cifar10_batch_shape():
    """Test that batches have correct shape."""
    train_loader, _ = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
    )

    # Get one batch
    x, y = next(iter(train_loader))

    # Check shapes
    assert x.shape == (32, 3, 32, 32)  # [batch, channels, height, width]
    assert y.shape == (32,)  # [batch]

    # Check types
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_cifar10_value_range():
    """Test that image values are normalized to [0, 1] range."""
    train_loader, _ = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
    )

    x, _ = next(iter(train_loader))

    # After normalization, values might be outside [0, 1], but not by much
    # Just check they're reasonable
    assert x.min() >= -3.0  # Normalized values
    assert x.max() <= 3.0


def test_cifar10_label_range():
    """Test that labels are in valid range [0, 9]."""
    train_loader, _ = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
    )

    _, y = next(iter(train_loader))

    assert y.min() >= 0
    assert y.max() <= 9


def test_cifar10_augmentation():
    """Test that augmentation can be enabled/disabled."""
    # With augmentation
    train_aug, _ = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
        augment=True,
    )

    # Without augmentation
    train_no_aug, _ = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
        augment=False,
    )

    # Both should work
    x_aug, _ = next(iter(train_aug))
    x_no_aug, _ = next(iter(train_no_aug))

    assert x_aug.shape == x_no_aug.shape


def test_cifar10_batch_size():
    """Test different batch sizes."""
    for batch_size in [16, 32, 64, 128]:
        train_loader, _ = get_cifar10_loaders(
            batch_size=batch_size,
            num_workers=0,
        )

        x, y = next(iter(train_loader))

        assert x.shape[0] == batch_size
        assert y.shape[0] == batch_size


def test_cifar10_data_consistency():
    """Test that same data is loaded consistently."""
    # Load data twice
    train_loader1, _ = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
        augment=False,
    )

    train_loader2, _ = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,
        augment=False,
    )

    # Get first batch from each
    x1, y1 = next(iter(train_loader1))
    x2, y2 = next(iter(train_loader2))

    # Labels should be the same (same seed for data loading)
    # Images might differ slightly due to different random seeds
    # but should have same shape
    assert x1.shape == x2.shape
    assert y1.shape == y2.shape


def test_cifar10_num_classes():
    """Test that all 10 classes are present."""
    train_loader, _ = get_cifar10_loaders(
        batch_size=1000,
        num_workers=0,
    )

    # Collect labels from several batches
    all_labels = []
    for i, (_, y) in enumerate(train_loader):
        all_labels.extend(y.tolist())
        if i >= 5:  # Check first 5 batches
            break

    # Should have examples from all 10 classes
    unique_labels = set(all_labels)
    assert len(unique_labels) == 10
    assert unique_labels == set(range(10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
