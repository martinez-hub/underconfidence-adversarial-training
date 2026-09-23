"""Tests for data loading."""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.cifar10 import VAL_SPLIT_SIZE, get_cifar10_loaders

CIFAR10_TRAIN_TOTAL = 50000
CIFAR10_TEST_TOTAL = 10000


# Module-scoped: building the CIFAR-10 dataset objects is the expensive part,
# and every test below only reads from the loaders. Function scope rebuilt both
# full datasets 14 times, which was most of this file's runtime.
@pytest.fixture(scope="module")
def loaders_noaug():
    """Train/val/test loaders without augmentation."""
    return get_cifar10_loaders(batch_size=32, num_workers=0, augment=False)


@pytest.fixture(scope="module")
def loaders_aug():
    """Train/val/test loaders with augmentation."""
    return get_cifar10_loaders(batch_size=32, num_workers=0, augment=True)


def test_cifar10_loaders_creation(loaders_noaug):
    """Test that CIFAR-10 loaders are created successfully."""
    train_loader, val_loader, test_loader = loaders_noaug

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_cifar10_dataset_sizes(loaders_noaug):
    """Train/val partition the 50k training images; test is the official 10k."""
    train_loader, val_loader, test_loader = loaders_noaug

    assert len(val_loader.dataset) == VAL_SPLIT_SIZE
    assert len(train_loader.dataset) == CIFAR10_TRAIN_TOTAL - VAL_SPLIT_SIZE
    assert len(train_loader.dataset) + len(val_loader.dataset) == CIFAR10_TRAIN_TOTAL
    assert len(test_loader.dataset) == CIFAR10_TEST_TOTAL


def test_cifar10_train_val_splits_are_disjoint(loaders_noaug):
    """
    Validation images must not appear in training.

    Model selection runs on the validation split every epoch, so an overlap
    would leak training data into the selection signal.
    """
    train_loader, val_loader, _ = loaders_noaug

    train_indices = set(train_loader.dataset.indices)
    val_indices = set(val_loader.dataset.indices)

    assert not train_indices & val_indices
    assert len(train_indices | val_indices) == CIFAR10_TRAIN_TOTAL


def test_cifar10_split_is_deterministic():
    """The same split_seed must produce the same partition across calls."""
    first, _, _ = get_cifar10_loaders(batch_size=32, num_workers=0, augment=False)
    second, _, _ = get_cifar10_loaders(batch_size=32, num_workers=0, augment=False)

    assert first.dataset.indices == second.dataset.indices


def test_cifar10_batch_shape(loaders_noaug):
    """Test that batches have correct shape."""
    train_loader, _, _ = loaders_noaug

    # Get one batch
    x, y = next(iter(train_loader))

    # Check shapes
    assert x.shape == (32, 3, 32, 32)  # [batch, channels, height, width]
    assert y.shape == (32,)  # [batch]

    # Check types
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_cifar10_value_range(loaders_noaug):
    """
    Images must be raw pixels in [0, 1].

    Normalization happens inside the model, not here. This is the contract the
    attacks rely on: their epsilon-ball and clamp(x, 0, 1) are both expressed in
    pixel units, so a normalized loader would silently break both.
    """
    for loader in loaders_noaug:
        x, _ = next(iter(loader))
        assert x.min() >= 0.0, f"pixel below 0: {x.min()}"
        assert x.max() <= 1.0, f"pixel above 1: {x.max()}"

    # And the range must actually be used, not collapsed to a constant.
    x, _ = next(iter(loaders_noaug[0]))
    assert x.max() - x.min() > 0.5


def test_cifar10_label_range(loaders_noaug):
    """Test that labels are in valid range [0, 9]."""
    train_loader, _, _ = loaders_noaug

    _, y = next(iter(train_loader))

    assert y.min() >= 0
    assert y.max() <= 9


def test_cifar10_augmentation(loaders_aug, loaders_noaug):
    """Augmentation changes the training transform but not the batch shape."""
    train_aug = loaders_aug[0]
    train_no_aug = loaders_noaug[0]

    x_aug, _ = next(iter(train_aug))
    x_no_aug, _ = next(iter(train_no_aug))

    assert x_aug.shape == x_no_aug.shape

    # The augmented pipeline must actually differ: two passes over the same
    # index give different tensors under RandomCrop/RandomHorizontalFlip.
    torch.manual_seed(0)
    first = train_aug.dataset[0][0]
    torch.manual_seed(1)
    second = train_aug.dataset[0][0]
    assert not torch.equal(first, second), "augmentation had no effect"

    # ...while the un-augmented pipeline is deterministic.
    assert torch.equal(train_no_aug.dataset[0][0], train_no_aug.dataset[0][0])


def test_cifar10_batch_size(loaders_noaug):
    """Test different batch sizes."""
    dataset = loaders_noaug[0].dataset

    for batch_size in [16, 32, 64, 128]:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

        x, y = next(iter(loader))

        assert x.shape[0] == batch_size
        assert y.shape[0] == batch_size


def test_cifar10_data_consistency():
    """Two un-augmented loaders must yield byte-identical batches."""
    train1, _, _ = get_cifar10_loaders(batch_size=32, num_workers=0, augment=False)
    train2, _, _ = get_cifar10_loaders(batch_size=32, num_workers=0, augment=False)

    # shuffle=True on the train loader, so compare the underlying dataset
    # rather than whichever batch the sampler happens to draw.
    x1, y1 = train1.dataset[0]
    x2, y2 = train2.dataset[0]

    assert torch.equal(x1, x2), "same index gave different images"
    assert y1 == y2, "same index gave different labels"


def test_cifar10_num_classes(loaders_noaug):
    """Test that all 10 classes are present."""
    dataset = loaders_noaug[0].dataset
    loader = DataLoader(dataset, batch_size=1000, num_workers=0)

    # Collect labels from several batches
    all_labels = []
    for i, (_, y) in enumerate(loader):
        all_labels.extend(y.tolist())
        if i >= 5:  # Check first 5 batches
            break

    # Should have examples from all 10 classes
    unique_labels = set(all_labels)
    assert len(unique_labels) == 10
    assert unique_labels == set(range(10))


def test_cifar10_rejects_invalid_val_size():
    """val_size must leave a non-empty training split."""
    with pytest.raises(ValueError, match="val_size"):
        get_cifar10_loaders(batch_size=32, num_workers=0, val_size=CIFAR10_TRAIN_TOTAL)


def test_cifar10_rejects_empty_val_split():
    """
    val_size=0 must be refused, not quietly produce an empty loader.

    Trainer.validate() divides by len(val_loader) and by the accumulated
    sample count, so an empty validation split kills training with
    ZeroDivisionError after the first epoch.
    """
    with pytest.raises(ValueError, match="val_size"):
        get_cifar10_loaders(batch_size=32, num_workers=0, val_size=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
