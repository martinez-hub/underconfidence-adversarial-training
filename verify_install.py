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

# Test 1: Import src modules from OUTSIDE the repo.
# Running `import src` here would succeed from the source tree alone, because
# Python puts the script's own directory on sys.path -- so it would pass even
# with nothing installed. A subprocess with a different cwd tests the install.
print("\n[1/5] Testing src module imports (from outside the repo)...")
MODULES = [
    "src",
    "src.data.cifar10",
    "src.models.resnet",
    "src.training.trainer",
    "src.attacks.pgd",
    "src.utils.config",
]
import subprocess
import tempfile

with tempfile.TemporaryDirectory() as probe_dir:
    probe = subprocess.run(
        [sys.executable, "-c", "import " + ", ".join(MODULES)],
        cwd=probe_dir,
        capture_output=True,
        text=True,
    )

if probe.returncode != 0:
    print(f"❌ Package is not importable outside the source tree")
    print(f"   {probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else 'unknown error'}")
    print("\nDiagnostic information:")
    print(f"   Working directory: {os.getcwd()}")
    print("\n💡 Solution:")
    print("   1. Make sure you ran: pip install -e .")
    print("   2. Try reinstalling: pip uninstall underconfidence-adversarial-training && pip install -e .")
    print("   3. Check if you're in a virtual environment")
    sys.exit(1)

for module in MODULES:
    print(f"   ✓ {module} imported")
print("✅ All src modules imported successfully!")

# Now import locally too, for the config check below.
from src.utils.config import load_config

# Test 2: Check PyTorch
print("\n[2/5] Checking PyTorch installation...")
MIN_PYTHON = (3, 11)
MIN_TORCH = (2, 6)
MIN_TORCHVISION = (0, 21)


def _version_tuple(raw):
    """Parse a leading numeric version (e.g. '2.6.0+cpu') into a tuple."""
    head = raw.split("+")[0]
    parts = []
    for piece in head.split(".")[:2]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


if sys.version_info[:2] < MIN_PYTHON:
    print(f"❌ Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
          f"got {sys.version.split()[0]}")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("❌ PyTorch not found. Please install: pip install torch torchvision")
    sys.exit(1)

if _version_tuple(torch.__version__) < MIN_TORCH:
    print(f"❌ torch>={MIN_TORCH[0]}.{MIN_TORCH[1]} required, got {torch.__version__}")
    sys.exit(1)
print(f"✅ PyTorch {torch.__version__} installed")
print(f"   CUDA available: {torch.cuda.is_available()}")

# Test 3: Check other dependencies
print("\n[3/5] Checking other dependencies...")
try:
    import torchvision
    import numpy as np
    import omegaconf
    import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease run: pip install -e .")
    sys.exit(1)

if _version_tuple(torchvision.__version__) < MIN_TORCHVISION:
    print(f"❌ torchvision>={MIN_TORCHVISION[0]}.{MIN_TORCHVISION[1]} required, "
          f"got {torchvision.__version__}")
    sys.exit(1)
print(f"✅ All dependencies installed")
print(f"   torchvision: {torchvision.__version__}")
print(f"   numpy: {np.__version__}")

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
