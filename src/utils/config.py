"""Configuration loading utilities."""

import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration as OmegaConf DictConfig object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    return cfg


def setup_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device(device_str: str = "auto") -> torch.device:
    """
    Setup computation device.

    Args:
        device_str: Device specification ("auto", "cuda", "cpu", "cuda:0", etc.)

    Returns:
        PyTorch device object
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    return device
