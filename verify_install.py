#!/usr/bin/env python
"""Quick script to verify installation is working correctly."""

import sys
import os
from pathlib import Path

print("=" * 60)
print("Verifying UAT Installation")
print("=" * 60)

# Test 0: Check current directory and Python path
print("\n[0/5] Checking environment...")
print(f"   Current directory: {Path.cwd()}")
print(f"   Python version: {sys.version.split()[0]}")
print(f"   Python path entries: {len(sys.path)}")

# Check if we're in the project root
expected_dirs = ["src", "experiments", "tests"]
missing_dirs = [d for d in expected_dirs if not Path(d).exists()]
if missing_dirs:
    print(f"⚠️  Warning: Not in project root? Missing: {missing_dirs}")
    print("   Please cd to the underconfidence-adversarial-training directory")
else:
    print(f"✅ In project root directory")

# Test 1: Import src modules
print("\n[1/5] Testing src module imports...")
try:
    import src
    print(f"   ✓ src package found at: {src.__file__ if hasattr(src, '__file__') else 'built-in'}")

    from src.data.cifar10 import get_cifar10_loaders
    print(f"   ✓ src.data.cifar10 imported")

    from src.models.resnet import get_resnet18_cifar10
    print(f"   ✓ src.models.resnet imported")

    from src.training.trainer import Trainer
    print(f"   ✓ src.training.trainer imported")

    from src.attacks.pgd import PGDAttack
    print(f"   ✓ src.attacks.pgd imported")

    from src.utils.config import load_config
    print(f"   ✓ src.utils.config imported")

    print("✅ All src modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nDiagnostic information:")
    print(f"   Python path: {sys.path[:3]}...")
    print(f"   Working directory: {os.getcwd()}")

    # Try to find where src is
    import importlib.util
    spec = importlib.util.find_spec("src")
    if spec is None:
        print(f"   'src' package not found in Python path")
        print("\n💡 Solution:")
        print("   1. Make sure you ran: pip install -e .")
        print("   2. Try reinstalling: pip uninstall underconfidence-adversarial-training && pip install -e .")
        print("   3. Check if you're in a virtual environment")
    else:
        print(f"   'src' found at: {spec.origin}")
        print(f"   But import failed with: {e}")
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
