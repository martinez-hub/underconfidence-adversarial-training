"""Checkpoint saving and loading utilities."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    **kwargs,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optional optimizer
        epoch: Current epoch number
        path: Path to save checkpoint
        **kwargs: Additional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        **kwargs,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> int:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to map checkpoint to

    Returns:
        Epoch number from checkpoint
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0)
