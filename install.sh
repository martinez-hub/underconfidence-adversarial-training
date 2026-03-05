#!/bin/bash
# Installation script for UAT

set -e  # Exit on error

echo "============================================================"
echo "UAT Installation Script"
echo "============================================================"

# Check if we're in the right directory
if [ ! -d "src" ] || [ ! -d "experiments" ]; then
    echo "❌ Error: Not in project root directory"
    echo "   Please cd to underconfidence-adversarial-training/"
    exit 1
fi

echo ""
echo "[1/4] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check if Python >= 3.11
required_version="3.11"
if ! python -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)"; then
    echo "⚠️  Warning: Python 3.11+ recommended (you have $python_version)"
    echo "   Installation will continue, but some features may not work"
fi

echo ""
echo "[2/4] Checking pip..."
if ! command -v pip &> /dev/null; then
    echo "❌ Error: pip not found"
    echo "   Please install pip first"
    exit 1
fi
pip_version=$(pip --version | awk '{print $2}')
echo "   pip version: $pip_version"

echo ""
echo "[3/4] Installing package in editable mode..."
pip install -e .

if [ $? -ne 0 ]; then
    echo "❌ Installation failed"
    echo "   Try: pip install --upgrade pip setuptools wheel"
    echo "   Then run this script again"
    exit 1
fi

echo ""
echo "[4/4] Verifying installation..."
python verify_install.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ Installation completed successfully!"
    echo "============================================================"
    echo ""
    echo "Quick start:"
    echo "  python experiments/train.py --config experiments/configs/smoke_test.yaml"
    echo ""
    echo "For more information, see: README.md"
else
    echo ""
    echo "============================================================"
    echo "⚠️  Installation completed with warnings"
    echo "============================================================"
    echo ""
    echo "Some imports failed. Please check the error messages above."
    echo "You may need to install additional dependencies:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
fi
