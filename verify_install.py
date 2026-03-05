#!/usr/bin/env python
"""Quick script to verify installation is working correctly."""

import sys

print("=" * 60)
print("Verifying UAT Installation")
print("=" * 60)

# Test 1: Import src modules
print("\n[1/5] Testing src module imports...")
try:
    import src
    from src.data.cifar10 import get_cifar10_loaders
    from src.models.resnet import get_resnet18_cifar10
    from src.training.trainer import Trainer
    from src.attacks.pgd import PGDAttack
    from src.utils.config import load_config
    print("✅ All src modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease run: pip install -e .")
    sys.exit(1)

# Test 2: Check PyTorch
print("\n[2/5] Checking PyTorch installation...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} installed")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("❌ PyTorch not found. Please install: pip install torch torchvision")
    sys.exit(1)

# Test 3: Check other dependencies
print("\n[3/5] Checking other dependencies...")
try:
    import torchvision
    import numpy as np
    import omegaconf
    import tqdm
    print(f"✅ All dependencies installed")
    print(f"   torchvision: {torchvision.__version__}")
    print(f"   numpy: {np.__version__}")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease run: pip install -e .")
    sys.exit(1)

# Test 4: Check data directory
print("\n[4/5] Checking directory structure...")
from pathlib import Path
required_dirs = ["src", "experiments", "tests", "experiments/configs"]
missing = [d for d in required_dirs if not Path(d).exists()]
if missing:
    print(f"❌ Missing directories: {missing}")
    print("   Make sure you're in the project root directory")
    sys.exit(1)
print("✅ Directory structure looks good")

# Test 5: Try loading a config
print("\n[5/5] Testing config loading...")
try:
    config_path = "experiments/configs/smoke_test.yaml"
    if Path(config_path).exists():
        cfg = load_config(config_path)
        print(f"✅ Config loaded successfully")
        print(f"   Training mode: {cfg.training.attack_type}")
    else:
        print(f"⚠️  Config file not found: {config_path}")
except Exception as e:
    print(f"❌ Error loading config: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 Installation verified successfully!")
print("=" * 60)
print("\nYou can now run experiments:")
print("  python experiments/train.py --config experiments/configs/smoke_test.yaml")
print("\nFor more information, see: README.md")
