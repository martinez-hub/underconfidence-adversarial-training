"""Main training entry point."""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet import get_resnet18_cifar10
from src.training.trainer import Trainer
from src.utils.config import load_config, setup_device, setup_seed
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main(config_path: str, overrides: str = ""):
    """
    Main training function.

    Args:
        config_path: Path to configuration YAML file
        overrides: Comma-separated config overrides (e.g., "optim.epochs=10,optim.lr=0.01")
    """
    # Load config
    cfg = load_config(config_path)

    # Apply CLI overrides
    if overrides:
        override_dict = OmegaConf.from_dotlist(overrides.split(","))
        cfg = OmegaConf.merge(cfg, override_dict)

    logger.info("="*60)
    logger.info("Configuration:")
    logger.info("="*60)
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    logger.info("="*60)

    # Setup
    setup_seed(cfg.meta.seed)
    device = setup_device(cfg.meta.device)
    logger.info(f"Device: {device}")

    # Data
    logger.info("Loading CIFAR-10 dataset...")
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        augment=cfg.data.augment,
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Model
    logger.info("Initializing model...")
    model = get_resnet18_cifar10()
    model = model.to(device)
    logger.info(f"Model: ResNet-18 (CIFAR-10 variant)")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
    )
    logger.info(f"Optimizer: SGD (lr={cfg.optim.lr}, momentum={cfg.optim.momentum})")

    # Trainer (unified for all training methods)
    attack_type = cfg.training.get("attack_type", "vanilla")
    logger.info(f"Training mode: {attack_type}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
    )

    # Train
    logger.info("="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    trainer.fit()

    # Save final checkpoint
    final_checkpoint_path = f"{cfg.logging.output_dir}/{attack_type}_final.pt"
    from src.utils.checkpoints import save_checkpoint
    save_checkpoint(
        model,
        optimizer,
        cfg.optim.epochs,
        path=final_checkpoint_path,
    )
    logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
    logger.info("="*60)
    logger.info("Training complete!")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UAT models on CIFAR-10")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default="",
        help="Comma-separated config overrides (e.g., 'optim.epochs=10,optim.lr=0.01')"
    )
    args = parser.parse_args()

    main(args.config, args.overrides)
