"""CIFAR-10 dataset utilities."""

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Size of the validation split carved out of the 50k training images.
# Model selection must not touch the 10k test split, so `get_cifar10_loaders`
# returns a held-out validation loader and the test loader separately.
VAL_SPLIT_SIZE = 5000


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    augment: bool = True,
    val_size: int = VAL_SPLIT_SIZE,
    split_seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create CIFAR-10 train/validation/test DataLoaders.

    Images are returned in raw [0, 1] pixel space (``ToTensor`` only, no
    ``Normalize``). Mean/std normalization happens inside the model -- see
    ``src.models.resnet.NormalizedModel`` -- so that adversarial attacks
    operate on pixels, where an L-infinity budget of 8/255 and the attacks'
    ``clamp(x, 0, 1)`` projection are both meaningful.

    The validation loader is a held-out slice of the 50k training images. The
    test loader is the official 10k test split and should only be used for
    final evaluation, never for model selection.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory to store/load CIFAR-10 data
        augment: Whether to apply data augmentation to the training split
        val_size: Number of training images held out for validation
        split_seed: Seed for the deterministic train/validation split

    Returns:
        (train_loader, val_loader, test_loader) tuple

    Raises:
        ValueError: If val_size does not leave a non-empty training split
    """
    # Training transforms
    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    # Validation/test transforms (never augmented)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Load datasets. The validation split needs eval transforms, so the
    # training set is instantiated twice over the same files and split
    # identically; only the transform differs between the two views.
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )

    val_source = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=eval_transform,
        download=True,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=eval_transform,
        download=True,
    )

    if not 0 <= val_size < len(train_dataset):
        raise ValueError(f"val_size must be in [0, {len(train_dataset)}), got {val_size}")

    train_size = len(train_dataset) - val_size
    split_generator = torch.Generator().manual_seed(split_seed)
    train_indices, val_indices = random_split(
        range(len(train_dataset)), [train_size, val_size], generator=split_generator
    )

    train_split = torch.utils.data.Subset(train_dataset, list(train_indices))
    val_split = torch.utils.data.Subset(val_source, list(val_indices))

    # Create data loaders
    train_loader = DataLoader(
        train_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_split,
        batch_size=batch_size,
        shuffle=False,
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

    return train_loader, val_loader, test_loader
