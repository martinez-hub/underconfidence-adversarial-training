"""Checkpoint saving and loading utilities."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


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

    Raises:
        ValueError: If epoch is negative
        IOError: If checkpoint cannot be saved
    """
    if epoch < 0:
        raise ValueError(f"Epoch must be non-negative, got {epoch}")

    path = Path(path)

    # Create parent directory
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise IOError(f"Cannot create directory {path.parent}: Permission denied") from e
    except Exception as e:
        raise IOError(f"Cannot create directory {path.parent}: {e}") from e

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        **kwargs,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    try:
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    except Exception as e:
        raise IOError(f"Failed to save checkpoint to {path}: {e}") from e


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

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If checkpoint is invalid or corrupted
        RuntimeError: If model state dict doesn't match
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if device is None:
        device = torch.device("cpu")

    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint from {path}: {e}") from e

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict' key")

    # Load model weights
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model weights loaded from: {path}")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load model state dict: {e}") from e

    # Load optimizer state if present
    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info("Optimizer state loaded")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
        else:
            logger.warning("Checkpoint does not contain optimizer state")

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Checkpoint loaded from epoch {epoch}")

    return epoch
