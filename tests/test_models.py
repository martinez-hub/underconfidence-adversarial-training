"""Tests for model definitions."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.resnet import get_resnet18_cifar10


def test_resnet18_cifar10_creation():
    """Test that ResNet-18 for CIFAR-10 can be created."""
    model = get_resnet18_cifar10()
    assert model is not None


def test_resnet18_cifar10_forward_pass():
    """Test forward pass through ResNet-18."""
    model = get_resnet18_cifar10()
    model.eval()

    # Create dummy input (batch of 4 images)
    x = torch.randn(4, 3, 32, 32)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape [batch_size, num_classes]
    assert output.shape == (4, 10)


def test_resnet18_cifar10_output_range():
    """Test that output logits are reasonable."""
    model = get_resnet18_cifar10()
    model.eval()

    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        output = model(x)

    # Logits should be finite
    assert torch.isfinite(output).all()

    # Logits should be in reasonable range (not too large)
    assert output.abs().max() < 100


def test_resnet18_cifar10_parameter_count():
    """Test that model has expected number of parameters."""
    model = get_resnet18_cifar10()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # ResNet-18 for CIFAR-10 should have ~11M parameters
    assert 10_000_000 < total_params < 12_000_000


def test_resnet18_cifar10_trainable_parameters():
    """Test that all parameters are trainable by default."""
    model = get_resnet18_cifar10()

    # All parameters should require grad
    for param in model.parameters():
        assert param.requires_grad


def test_resnet18_cifar10_batch_independence():
    """Test that predictions are independent across batch."""
    model = get_resnet18_cifar10()
    model.eval()

    # Create batch with same image repeated
    x_single = torch.randn(1, 3, 32, 32)
    x_batch = x_single.repeat(4, 1, 1, 1)

    with torch.no_grad():
        out_single = model(x_single)
        out_batch = model(x_batch)

    # All outputs in batch should be identical
    assert torch.allclose(out_batch[0], out_batch[1], atol=1e-5)
    assert torch.allclose(out_batch[0], out_single[0], atol=1e-5)


def test_resnet18_cifar10_different_batch_sizes():
    """Test that model works with different batch sizes."""
    model = get_resnet18_cifar10()
    model.eval()

    for batch_size in [1, 2, 4, 8, 16, 32]:
        x = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 10)


def test_resnet18_cifar10_gradient_flow():
    """Test that gradients flow through the model."""
    model = get_resnet18_cifar10()
    model.train()

    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    output = model(x)

    # Compute a simple loss
    loss = output.sum()
    loss.backward()

    # Check that input gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_resnet18_cifar10_training_mode():
    """Test that model can switch between train and eval modes."""
    model = get_resnet18_cifar10()

    # Default should be training mode
    assert model.training

    # Switch to eval
    model.eval()
    assert not model.training

    # Switch back to train
    model.train()
    assert model.training


def test_resnet18_cifar10_device_compatibility():
    """Test that model can be moved to different devices."""
    model = get_resnet18_cifar10()

    # CPU device
    device_cpu = torch.device('cpu')
    model_cpu = model.to(device_cpu)

    x = torch.randn(4, 3, 32, 32).to(device_cpu)

    with torch.no_grad():
        output = model_cpu(x)

    assert output.device == device_cpu


def test_resnet18_cifar10_state_dict():
    """Test that model state dict can be saved and loaded."""
    model1 = get_resnet18_cifar10()
    model2 = get_resnet18_cifar10()

    # Get state dict from model1
    state_dict = model1.state_dict()

    # Load into model2
    model2.load_state_dict(state_dict)

    # Both models should produce same output
    model1.eval()
    model2.eval()

    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-5)


def test_resnet18_cifar10_num_classes():
    """Test that output has correct number of classes."""
    model = get_resnet18_cifar10()

    # Check final layer
    assert model.fc.out_features == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
