"""Checkpoint saving and loading utilities."""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Checkpoints written before normalization moved inside the model stored the
# backbone parameters at the top level (e.g. "conv1.weight"). The model is now
# a NormalizedModel wrapping that backbone, so those keys need the prefix.
_LEGACY_BACKBONE_PREFIX = "backbone."


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

    # Write to a temporary file and rename, so an interrupted or failing save
    # cannot truncate a previously saved checkpoint.
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        logger.info(f"Checkpoint saved: {path}")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise IOError(f"Failed to save checkpoint to {path}: {e}") from e


def _upgrade_legacy_state_dict(state_dict: dict, model: nn.Module) -> dict:
    """
    Prefix pre-NormalizedModel checkpoints so they load into the current model.

    Args:
        state_dict: State dict as stored in the checkpoint
        model: The model the weights are destined for

    Returns:
        The state dict, rewritten only if it is in the legacy layout
    """
    expected = set(model.state_dict())
    if not expected or set(state_dict) & expected:
        # Already matches (or an unrelated model) - leave it alone.
        return state_dict

    upgraded = {_LEGACY_BACKBONE_PREFIX + k: v for k, v in state_dict.items()}
    if set(upgraded) & expected:
        logger.info("Detected pre-NormalizedModel checkpoint; remapping backbone keys")
        return upgraded

    return state_dict


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

    # weights_only=False is required: checkpoints carry training metadata
    # (omegaconf DictConfig, history) alongside the tensors, which the
    # weights-only unpickler rejects. torch 2.6 made weights_only=True the
    # default, so omitting this makes every trainer-written checkpoint
    # unloadable. These are the user's own local training artifacts.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint from {path}: {e}") from e

    # Validate checkpoint structure
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict' key")

    # Load model weights
    state_dict = _upgrade_legacy_state_dict(checkpoint["model_state_dict"], model)
    try:
        model.load_state_dict(state_dict)
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
