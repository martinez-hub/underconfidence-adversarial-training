"""Tests for error handling and validation."""

import sys
from pathlib import Path
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
from src.attacks.confsmooth import ConfSmoothAttack
from src.attacks.pgd import PGDAttack
from src.models.resnet import get_resnet18_cifar10
from src.utils.checkpoints import load_checkpoint, save_checkpoint
from src.utils.config import load_config, setup_device, validate_config


# ============================================================================
# Config Validation Tests
# ============================================================================

def test_validate_config_valid():
    """Test that valid config passes validation."""
    cfg = OmegaConf.create({
        'meta': {'seed': 42},
        'optim': {'lr': 0.1, 'epochs': 10, 'weight_decay': 5e-4, 'momentum': 0.9},
        'attack': {'epsilon': 8/255, 'alpha': 2/255},
        'data': {'batch_size': 128, 'num_workers': 4},
        'training': {'attack_type': 'vanilla'},
        'uat': {'target_class_boost': 0.01, 'pair_mode': 'top2'},
    })

    # Should not raise
    validate_config(cfg)


def test_validate_config_negative_seed():
    """Test that negative seed is rejected."""
    cfg = OmegaConf.create({'meta': {'seed': -1}})

    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        validate_config(cfg)


def test_validate_config_negative_lr():
    """Test that negative learning rate is rejected."""
    cfg = OmegaConf.create({'optim': {'lr': -0.1}})

    with pytest.raises(ValueError, match="lr must be positive"):
        validate_config(cfg)


def test_validate_config_negative_epochs():
    """Test that negative epochs is rejected."""
    cfg = OmegaConf.create({'optim': {'epochs': -1}})

    with pytest.raises(ValueError, match="epochs must be positive"):
        validate_config(cfg)


def test_validate_config_negative_epsilon():
    """Test that negative epsilon is rejected."""
    cfg = OmegaConf.create({'attack': {'epsilon': -0.1}})

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        validate_config(cfg)


def test_validate_config_large_epsilon():
    """Test that epsilon > 1 is rejected."""
    cfg = OmegaConf.create({'attack': {'epsilon': 2.0}})

    with pytest.raises(ValueError, match="epsilon should typically be"):
        validate_config(cfg)


def test_validate_config_invalid_attack_type():
    """Test that invalid attack type is rejected."""
    cfg = OmegaConf.create({'training': {'attack_type': 'invalid'}})

    with pytest.raises(ValueError, match="attack_type must be one of"):
        validate_config(cfg)


def test_validate_config_invalid_pair_mode():
    """Test that invalid pair mode is rejected."""
    cfg = OmegaConf.create({'uat': {'pair_mode': 'invalid'}})

    with pytest.raises(ValueError, match="pair_mode must be one of"):
        validate_config(cfg)


# ============================================================================
# Checkpoint Validation Tests
# ============================================================================

def test_save_checkpoint_negative_epoch():
    """Test that negative epoch is rejected."""
    model = get_resnet18_cifar10()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with tempfile.NamedTemporaryFile(suffix='.pt') as f:
        with pytest.raises(ValueError, match="Epoch must be non-negative"):
            save_checkpoint(model, optimizer, epoch=-1, path=f.name)


def test_load_checkpoint_nonexistent_file():
    """Test that loading non-existent checkpoint raises error."""
    model = get_resnet18_cifar10()

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint('/nonexistent/path.pt', model)


def test_load_checkpoint_invalid_file():
    """Test that loading invalid checkpoint raises error."""
    model = get_resnet18_cifar10()

    # Create a file with invalid content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pt', delete=False) as f:
        f.write("not a valid checkpoint")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Failed to load checkpoint"):
            load_checkpoint(temp_path, model)
    finally:
        Path(temp_path).unlink()


def test_save_load_checkpoint_roundtrip():
    """Test that saving and loading checkpoint works correctly."""
    model = get_resnet18_cifar10()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / 'checkpoint.pt'

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch=5, path=str(checkpoint_path))

        # Load checkpoint
        model2 = get_resnet18_cifar10()
        epoch = load_checkpoint(str(checkpoint_path), model2)

        assert epoch == 5


# ============================================================================
# PGD Attack Validation Tests
# ============================================================================

def test_pgd_attack_none_model():
    """Test that None model is rejected."""
    with pytest.raises(ValueError, match="Model cannot be None"):
        PGDAttack(model=None)


def test_pgd_attack_negative_epsilon():
    """Test that negative epsilon is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        PGDAttack(model, epsilon=-0.1)


def test_pgd_attack_large_epsilon():
    """Test that epsilon > 1 is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="epsilon should typically be"):
        PGDAttack(model, epsilon=2.0)


def test_pgd_attack_negative_alpha():
    """Test that negative alpha is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="alpha must be positive"):
        PGDAttack(model, alpha=-0.1)


def test_pgd_attack_alpha_larger_than_epsilon():
    """Test that alpha > epsilon is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="alpha .* should typically be <= epsilon"):
        PGDAttack(model, epsilon=0.03, alpha=0.05)


def test_pgd_attack_negative_num_steps():
    """Test that negative num_steps is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="num_steps must be positive"):
        PGDAttack(model, num_steps=-1)


def test_pgd_attack_valid_params():
    """Test that valid parameters are accepted."""
    model = get_resnet18_cifar10()

    # Should not raise
    attack = PGDAttack(model, epsilon=8/255, alpha=2/255, num_steps=20)
    assert attack is not None


# ============================================================================
# ConfSmooth Attack Validation Tests
# ============================================================================

def test_confsmooth_attack_none_model():
    """Test that None model is rejected."""
    with pytest.raises(ValueError, match="Model cannot be None"):
        ConfSmoothAttack(model=None)


def test_confsmooth_attack_negative_epsilon():
    """Test that negative epsilon is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        ConfSmoothAttack(model, epsilon=-0.1)


def test_confsmooth_attack_negative_num_steps():
    """Test that negative num_steps is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="num_steps must be positive"):
        ConfSmoothAttack(model, num_steps=-1)


def test_confsmooth_attack_invalid_num_classes():
    """Test that num_classes <= 1 is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="num_classes must be > 1"):
        ConfSmoothAttack(model, num_classes=1)


def test_confsmooth_attack_invalid_target_boost():
    """Test that invalid target_class_boost is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="target_class_boost must be in"):
        ConfSmoothAttack(model, target_class_boost=-0.1)

    with pytest.raises(ValueError, match="target_class_boost must be in"):
        ConfSmoothAttack(model, target_class_boost=1.5)


def test_confsmooth_attack_valid_params():
    """Test that valid parameters are accepted."""
    model = get_resnet18_cifar10()

    # Should not raise
    attack = ConfSmoothAttack(
        model, epsilon=8/255, alpha=2/255, num_steps=20,
        num_classes=10, target_class_boost=0.01
    )
    assert attack is not None


# ============================================================================
# ClassAmbiguity Attack Validation Tests
# ============================================================================

def test_class_ambiguity_attack_none_model():
    """Test that None model is rejected."""
    with pytest.raises(ValueError, match="Model cannot be None"):
        ClassPairAmbiguityAttack(model=None)


def test_class_ambiguity_attack_negative_epsilon():
    """Test that negative epsilon is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="epsilon must be non-negative"):
        ClassPairAmbiguityAttack(model, epsilon=-0.1)


def test_class_ambiguity_attack_invalid_pair_mode():
    """Test that invalid target_pair_mode is rejected."""
    model = get_resnet18_cifar10()

    with pytest.raises(ValueError, match="target_pair_mode must be one of"):
        ClassPairAmbiguityAttack(model, target_pair_mode='invalid')


def test_class_ambiguity_attack_valid_params():
    """Test that valid parameters are accepted."""
    model = get_resnet18_cifar10()

    # Should not raise
    attack = ClassPairAmbiguityAttack(
        model, epsilon=8/255, alpha=2/255, num_steps=20,
        target_pair_mode='top2'
    )
    assert attack is not None


# ============================================================================
# Device Setup Tests
# ============================================================================

def test_setup_device_auto():
    """Test auto device selection."""
    device = setup_device('auto')
    assert device is not None
    assert device.type in ['cpu', 'cuda']


def test_setup_device_cpu():
    """Test CPU device selection."""
    device = setup_device('cpu')
    assert device.type == 'cpu'


def test_setup_device_invalid():
    """Test invalid device specification."""
    with pytest.raises(ValueError, match="Invalid device specification"):
        setup_device('invalid_device_name')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
