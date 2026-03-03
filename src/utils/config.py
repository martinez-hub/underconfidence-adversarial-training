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
        try:
            device = torch.device(device_str)
        except RuntimeError as e:
            raise ValueError(f"Invalid device specification: {device_str}. Error: {e}")

    return device


def validate_config(cfg: DictConfig) -> None:
    """
    Validate configuration parameters.

    Args:
        cfg: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate meta section
    if hasattr(cfg, 'meta'):
        if hasattr(cfg.meta, 'seed'):
            if not isinstance(cfg.meta.seed, int) or cfg.meta.seed < 0:
                raise ValueError(f"meta.seed must be a non-negative integer, got {cfg.meta.seed}")

    # Validate optim section
    if hasattr(cfg, 'optim'):
        if hasattr(cfg.optim, 'lr'):
            if cfg.optim.lr <= 0:
                raise ValueError(f"optim.lr must be positive, got {cfg.optim.lr}")

        if hasattr(cfg.optim, 'epochs'):
            if cfg.optim.epochs <= 0:
                raise ValueError(f"optim.epochs must be positive, got {cfg.optim.epochs}")

        if hasattr(cfg.optim, 'weight_decay'):
            if cfg.optim.weight_decay < 0:
                raise ValueError(f"optim.weight_decay must be non-negative, got {cfg.optim.weight_decay}")

        if hasattr(cfg.optim, 'momentum'):
            if not 0 <= cfg.optim.momentum < 1:
                raise ValueError(f"optim.momentum must be in [0, 1), got {cfg.optim.momentum}")

    # Validate attack section
    if hasattr(cfg, 'attack'):
        if hasattr(cfg.attack, 'epsilon'):
            if cfg.attack.epsilon < 0:
                raise ValueError(f"attack.epsilon must be non-negative, got {cfg.attack.epsilon}")
            if cfg.attack.epsilon > 1:
                raise ValueError(f"attack.epsilon should typically be <= 1, got {cfg.attack.epsilon}")

        if hasattr(cfg.attack, 'alpha'):
            if cfg.attack.alpha <= 0:
                raise ValueError(f"attack.alpha must be positive, got {cfg.attack.alpha}")

    # Validate data section
    if hasattr(cfg, 'data'):
        if hasattr(cfg.data, 'batch_size'):
            if cfg.data.batch_size <= 0:
                raise ValueError(f"data.batch_size must be positive, got {cfg.data.batch_size}")

        if hasattr(cfg.data, 'num_workers'):
            if cfg.data.num_workers < 0:
                raise ValueError(f"data.num_workers must be non-negative, got {cfg.data.num_workers}")

    # Validate training section
    if hasattr(cfg, 'training'):
        if hasattr(cfg.training, 'attack_type'):
            valid_attacks = ['vanilla', 'pgd', 'confsmooth', 'class_ambiguity', None]
            if cfg.training.attack_type not in valid_attacks:
                raise ValueError(
                    f"training.attack_type must be one of {valid_attacks}, "
                    f"got {cfg.training.attack_type}"
                )

    # Validate UAT section
    if hasattr(cfg, 'uat'):
        if hasattr(cfg.uat, 'target_class_boost'):
            if not 0 <= cfg.uat.target_class_boost <= 1:
                raise ValueError(
                    f"uat.target_class_boost must be in [0, 1], "
                    f"got {cfg.uat.target_class_boost}"
                )

        if hasattr(cfg.uat, 'pair_mode'):
            valid_modes = ['random', 'top2', 'fixed']
            if cfg.uat.pair_mode not in valid_modes:
                raise ValueError(
                    f"uat.pair_mode must be one of {valid_modes}, "
                    f"got {cfg.uat.pair_mode}"
                )
