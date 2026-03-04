"""Test Python 3.11+ and PyTorch 2.6+ compatibility."""

import sys
import pytest
import torch
import torchvision
import numpy as np


def test_python_version():
    """Verify Python 3.11+ is being used."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"
    print(f"✅ Python version: {sys.version}")


def test_pytorch_version():
    """Verify PyTorch 2.6+ is being used."""
    torch_version = tuple(map(int, torch.__version__.split('+')[0].split('.')[:2]))
    assert torch_version >= (2, 6), f"PyTorch 2.6+ required, got {torch.__version__}"
    print(f"✅ PyTorch version: {torch.__version__}")


def test_torchvision_version():
    """Verify torchvision 0.21+ is being used."""
    tv_version = tuple(map(int, torchvision.__version__.split('+')[0].split('.')[:2]))
    assert tv_version >= (0, 21), f"torchvision 0.21+ required, got {torchvision.__version__}"
    print(f"✅ torchvision version: {torchvision.__version__}")


def test_numpy_compatibility():
    """Test numpy 1.26+ compatibility."""
    np_version = tuple(map(int, np.__version__.split('.')[:2]))
    assert np_version >= (1, 26), f"numpy 1.26+ required, got {np.__version__}"
    print(f"✅ numpy version: {np.__version__}")


def test_basic_torch_operations():
    """Test basic PyTorch operations work correctly."""
    # Test tensor creation
    x = torch.randn(4, 3, 32, 32)
    assert x.shape == (4, 3, 32, 32)

    # Test GPU availability (optional)
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        assert x_gpu.is_cuda
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("ℹ️  CUDA not available (CPU only)")

    # Test autograd
    x.requires_grad = True
    y = (x ** 2).sum()
    y.backward()
    assert x.grad is not None

    print("✅ Basic PyTorch operations work correctly")


def test_imports():
    """Test all critical imports work."""
    try:
        from src.attacks.pgd import PGDAttack
        from src.attacks.confsmooth import ConfSmoothAttack
        from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
        from src.models.resnet import get_resnet18_cifar10
        from src.training.trainer import Trainer
        from src.data.cifar10 import get_cifar10_loaders
        from src.utils.config import load_config, setup_seed, setup_device
        print("✅ All imports successful")
    except Exception as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPATIBILITY VERIFICATION")
    print("="*60 + "\n")

    test_python_version()
    test_pytorch_version()
    test_torchvision_version()
    test_numpy_compatibility()
    test_basic_torch_operations()
    test_imports()

    print("\n" + "="*60)
    print("✅ ALL COMPATIBILITY TESTS PASSED")
    print("="*60 + "\n")
