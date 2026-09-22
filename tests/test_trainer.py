"""Tests for the unified trainer."""

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.cifar10 import get_cifar10_loaders
from src.models.resnet import get_resnet18_cifar10
from src.training.trainer import Trainer


# Module-scoped: these loaders are read-only and deterministic (shuffle=False),
# but building them instantiates all 60k CIFAR-10 images to use 32 of them.
@pytest.fixture(scope="module")
def small_data_loaders():
    """Create small data loaders for quick testing."""
    # Load full datasets first
    from torchvision import datasets, transforms

    # ToTensor only - no Normalize. The model normalizes its own input, so
    # loaders must hand over raw [0, 1] pixels.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    # Create subsets BEFORE creating DataLoaders
    train_subset = torch.utils.data.Subset(train_dataset, range(32))
    val_subset = torch.utils.data.Subset(val_dataset, range(32))

    # Now create DataLoaders with the subsets
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=16, shuffle=False, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=16, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


@pytest.fixture
def model():
    """Create a small model for testing."""
    return get_resnet18_cifar10()


@pytest.fixture
def config():
    """Create minimal config for testing."""
    return OmegaConf.create(
        {
            "training": {"attack_type": "vanilla"},
            "attack": {"epsilon": 8 / 255, "alpha": 2 / 255},
            "uat": {"target_class_boost": 0.01, "pair_mode": "top2"},
            "data": {"num_classes": 10},
            "optim": {"epochs": 1, "milestones": [], "gamma": 0.1},
            "logging": {
                "log_every": 10,
                "save_every": 1,
                "output_dir": "/tmp/test_trainer",
            },
        }
    )


def test_trainer_initialization(model, small_data_loaders, config):
    """Test that trainer initializes correctly."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=config,
    )

    assert trainer.model is not None
    assert trainer.attack_type == "vanilla"
    assert trainer.attack is None  # Vanilla has no attack


def test_trainer_vanilla_training(model, small_data_loaders, config):
    """Test vanilla training (no adversarial training)."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=config,
    )

    # Train for 1 epoch
    train_metrics = trainer.train_epoch(epoch=1)

    # Check metrics are present
    assert "train_loss" in train_metrics
    assert "train_acc_clean" in train_metrics
    assert "train_acc_train" in train_metrics

    # Check metrics are reasonable
    assert train_metrics["train_loss"] > 0
    assert 0 <= train_metrics["train_acc_clean"] <= 100
    assert 0 <= train_metrics["train_acc_train"] <= 100


def test_trainer_pgd_training(model, small_data_loaders, config):
    """Test PGD adversarial training."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    # Update config for PGD training
    config.training.attack_type = "pgd"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=config,
    )

    assert trainer.attack_type == "pgd"
    assert trainer.attack is not None

    # Spy on the attack: asserting only that train_epoch returns a dict key
    # would pass with adversarial generation removed entirely.
    calls = []
    original_generate = trainer.attack.generate

    def spy(x, *args, **kwargs):
        x_adv = original_generate(x, *args, **kwargs)
        calls.append(((x_adv - x).abs().max().item(), x_adv.shape))
        return x_adv

    trainer.attack.generate = spy

    train_metrics = trainer.train_epoch(epoch=1)

    assert "train_loss" in train_metrics
    assert len(calls) == len(train_loader), "attack was not run on every batch"
    for delta, shape in calls:
        assert delta > 0, "attack returned the clean batch unchanged"
        assert (
            delta <= config.attack.epsilon + 1e-6
        ), f"perturbation {delta} exceeds epsilon {config.attack.epsilon}"


def test_trainer_confsmooth_training(model, small_data_loaders, config):
    """Test UAT-ConfSmooth training."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    # Update config for ConfSmooth training
    config.training.attack_type = "confsmooth"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=config,
    )

    assert trainer.attack_type == "confsmooth"
    assert trainer.attack is not None

    # Spy on the attack: asserting only that train_epoch returns a dict key
    # would pass with adversarial generation removed entirely.
    calls = []
    original_generate = trainer.attack.generate

    def spy(x, *args, **kwargs):
        x_adv = original_generate(x, *args, **kwargs)
        calls.append(((x_adv - x).abs().max().item(), x_adv.shape))
        return x_adv

    trainer.attack.generate = spy

    train_metrics = trainer.train_epoch(epoch=1)

    assert "train_loss" in train_metrics
    assert len(calls) == len(train_loader), "attack was not run on every batch"
    for delta, shape in calls:
        assert delta > 0, "attack returned the clean batch unchanged"
        assert (
            delta <= config.attack.epsilon + 1e-6
        ), f"perturbation {delta} exceeds epsilon {config.attack.epsilon}"


def test_trainer_class_ambiguity_training(model, small_data_loaders, config):
    """Test UAT-ClassAmbiguity training."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    # Update config for ClassAmbiguity training
    config.training.attack_type = "class_ambiguity"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=config,
    )

    assert trainer.attack_type == "class_ambiguity"
    assert trainer.attack is not None

    # Spy on the attack: asserting only that train_epoch returns a dict key
    # would pass with adversarial generation removed entirely.
    calls = []
    original_generate = trainer.attack.generate

    def spy(x, *args, **kwargs):
        x_adv = original_generate(x, *args, **kwargs)
        calls.append(((x_adv - x).abs().max().item(), x_adv.shape))
        return x_adv

    trainer.attack.generate = spy

    train_metrics = trainer.train_epoch(epoch=1)

    assert "train_loss" in train_metrics
    assert len(calls) == len(train_loader), "attack was not run on every batch"
    for delta, shape in calls:
        assert delta > 0, "attack returned the clean batch unchanged"
        assert (
            delta <= config.attack.epsilon + 1e-6
        ), f"perturbation {delta} exceeds epsilon {config.attack.epsilon}"


def test_trainer_validation(model, small_data_loaders, config):
    """Test validation loop."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=config,
    )

    val_metrics = trainer.validate(epoch=1)

    # Check metrics
    assert "val_loss" in val_metrics
    assert "val_acc" in val_metrics
    assert val_metrics["val_loss"] > 0
    assert 0 <= val_metrics["val_acc"] <= 100


def test_trainer_invalid_attack_type(model, small_data_loaders, config):
    """Test that invalid attack type raises error."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    # Set invalid attack type
    config.training.attack_type = "invalid_attack"

    with pytest.raises(ValueError, match="Unknown attack type"):
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            cfg=config,
        )


def test_trainer_lr_scheduler(model, small_data_loaders, config):
    """Test that learning rate scheduler is created correctly."""
    train_loader, val_loader = small_data_loaders
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    # Set milestones for LR scheduler
    config.optim.milestones = [5, 10]
    config.optim.gamma = 0.1

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        cfg=config,
    )

    # Check scheduler exists
    assert trainer.scheduler is not None

    # Check initial LR
    initial_lr = trainer.optimizer.param_groups[0]["lr"]
    assert initial_lr == 0.1

    # Step scheduler and check LR hasn't changed yet (milestone not reached)
    trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == 0.1

    # Step past the first milestone (5) and confirm it actually decays --
    # otherwise a scheduler that never fires would satisfy this test.
    for _ in range(4):
        trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    # And past the second milestone (10)
    for _ in range(5):
        trainer.scheduler.step()
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
