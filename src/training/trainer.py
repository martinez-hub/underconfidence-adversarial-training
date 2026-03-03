"""Unified trainer for Vanilla, PGD-AT, and UAT training."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..attacks.class_ambiguity import ClassPairAmbiguityAttack
from ..attacks.confsmooth import ConfSmoothAttack
from ..attacks.pgd import PGDAttack
from ..utils.checkpoints import save_checkpoint
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Unified trainer for Vanilla, PGD-AT, and UAT training.

    All training methods use the same procedure:
    1. Generate training examples (clean or adversarial)
    2. Train on those examples with cross-entropy loss

    What differs is the input image:
    - "vanilla" or None: Train on clean images
    - "pgd": Train ONLY on PGD adversarial images (10 steps)
    - "confsmooth": Train ONLY on ConfSmooth adversarial images (5 steps)
    - "class_ambiguity": Train ONLY on Class-Pair Ambiguity adversarial images (10 steps)

    Key insight: UAT-ConfSmooth achieves comparable robustness with 50% fewer gradient steps (5 vs 10)!
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        cfg: DictConfig,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            cfg: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.optim.milestones,
            gamma=cfg.optim.gamma,
        )

        # Initialize attack based on config
        self.attack_type = cfg.training.get("attack_type", "vanilla")
        self.attack = None

        if self.attack_type == "pgd":
            self.attack = PGDAttack(
                model=self.model,
                epsilon=cfg.attack.epsilon,
                alpha=cfg.attack.alpha,
                num_steps=10,  # PGD: 10 steps
            )
            logger.info("Initialized PGD attack (10 steps)")

        elif self.attack_type == "confsmooth":
            self.attack = ConfSmoothAttack(
                model=self.model,
                epsilon=cfg.attack.epsilon,
                alpha=cfg.attack.alpha,
                num_steps=5,  # ConfSmooth: 5 steps (50% fewer!)
                num_classes=cfg.data.num_classes,
                target_class_boost=cfg.uat.get("target_class_boost", 0.01),
            )
            logger.info("Initialized ConfSmooth attack (5 steps - 50% fewer than PGD!)")

        elif self.attack_type == "class_ambiguity":
            self.attack = ClassPairAmbiguityAttack(
                model=self.model,
                epsilon=cfg.attack.epsilon,
                alpha=cfg.attack.alpha,
                num_steps=10,  # Class Ambiguity: 10 steps
                target_pair_mode=cfg.uat.get("pair_mode", "top2"),
            )
            logger.info("Initialized Class-Pair Ambiguity attack (10 steps)")

        elif self.attack_type == "vanilla" or self.attack_type is None:
            self.attack = None  # No attack (vanilla training)
            logger.info("No adversarial training (vanilla mode)")

        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Single epoch of training.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        correct_clean = 0
        correct_train = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # Generate training examples (clean or adversarial)
            if self.attack is None:
                x_train = x  # Vanilla: use clean images
            else:
                # Adversarial training: generate adversarial examples
                # Switch to eval mode for attack generation
                self.model.eval()
                x_train = self.attack.generate(x, y)
                self.model.train()

            # Train on generated examples with standard cross-entropy loss
            self.optimizer.zero_grad()
            logits_train = self.model(x_train)
            loss = F.cross_entropy(logits_train, y)
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()

            with torch.no_grad():
                # Accuracy on clean images
                logits_clean = self.model(x)
                pred_clean = logits_clean.argmax(dim=1)
                correct_clean += pred_clean.eq(y).sum().item()

                # Accuracy on training examples (clean or adversarial)
                pred_train = logits_train.argmax(dim=1)
                correct_train += pred_train.eq(y).sum().item()

                total += y.size(0)

            # Update progress bar
            if (batch_idx + 1) % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "clean": f"{100.0 * correct_clean / total:.1f}%",
                    "train": f"{100.0 * correct_train / total:.1f}%",
                })

        return {
            "train_loss": total_loss / len(self.train_loader),
            "train_acc_clean": 100.0 * correct_clean / total,
            "train_acc_train": 100.0 * correct_train / total,
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validation loop (always on clean images).

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = F.cross_entropy(logits, y)

                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)

        return {
            "val_loss": val_loss / len(self.val_loader),
            "val_acc": 100.0 * correct / total,
        }

    def fit(self):
        """Full training loop."""
        logger.info(f"Starting training: {self.cfg.optim.epochs} epochs, attack type: {self.attack_type}")

        for epoch in range(1, self.cfg.optim.epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)

            # Validation
            val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Logging
            logger.info(
                f"Epoch {epoch}/{self.cfg.optim.epochs} | "
                f"Attack: {self.attack_type} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc (clean): {train_metrics['train_acc_clean']:.2f}% | "
                f"Train Acc (train): {train_metrics['train_acc_train']:.2f}% | "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )

            # Checkpoint saving
            if epoch % self.cfg.logging.save_every == 0:
                checkpoint_path = f"{self.cfg.logging.output_dir}/{self.attack_type}_epoch{epoch}.pt"
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    path=checkpoint_path,
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        logger.info("Training complete!")
