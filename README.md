# Underconfidence Adversarial Training (UAT)

Official implementation of **"Improving Vision Model Robustness against Misclassification and Uncertainty Attacks via Underconfidence Adversarial Training"** (NLDL 2026)

*Josué Martínez-Martínez, John T Holodnak, Olivia Brown, Sheida Nabavi, Derek Aguiar, Allan Wollaber*

[[Paper (PMLR)]](https://proceedings.mlr.press/v307/marti-nez-marti-nez26a.html) | [[OpenReview]](https://openreview.net/forum?id=3upHbaUyR4)

---

## Overview

This repository implements Underconfidence Adversarial Training (UAT), a novel defense against both misclassification and underconfidence attacks. UAT achieves comparable robustness to standard adversarial training with **50% fewer gradient steps** during attack generation.

**Key Features:**
- Two novel underconfidence attacks: ConfSmooth and Class-Pair Ambiguity
- Efficient training framework (5 gradient steps vs. standard 10)
- Complete CIFAR-10 experiments with reproducible results

---

## Installation

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/martinez-hub/underconfidence-adversarial-training.git
cd underconfidence-adversarial-training

# Install package in editable mode (recommended for development)
pip install -e .

# Or install with development tools
pip install -e ".[dev]"

# Verify installation
python verify_install.py
```

**Requirements:** Python 3.11+, PyTorch 2.6+

**Note:** The `-e` flag installs the package in editable mode, which allows you to modify the source code without reinstalling. This also properly configures the `src` module for imports. Run `verify_install.py` to confirm everything is working.

### Option 2: Docker (Recommended)

```bash
# Build and run with Docker
docker build -t uat:latest .
docker run -it --rm -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints uat:latest

# Or use docker-compose
docker-compose run uat

# For GPU support (requires nvidia-docker)
docker-compose run uat-gpu
```

---

## Quick Start

### Training

```bash
# UAT with ConfSmooth (5 steps - most efficient)
python experiments/train.py --config experiments/configs/uat_confsmooth_cifar10.yaml

# UAT with Class-Pair Ambiguity (10 steps)
python experiments/train.py --config experiments/configs/uat_ambiguity_cifar10.yaml

# PGD adversarial training baseline (10 steps)
python experiments/train.py --config experiments/configs/pgd_cifar10.yaml

# Vanilla baseline (no adversarial training)
python experiments/train.py --config experiments/configs/vanilla_cifar10.yaml
```

### Evaluation

```bash
# Evaluate on clean and adversarial test sets
python experiments/eval.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_best_model.pt \
  --config experiments/configs/uat_confsmooth_cifar10.yaml

# Clean images only (faster)
python experiments/eval.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_best_model.pt \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --no-attacks
```

### Quick Smoke Test

```bash
# Run 2-epoch smoke test to verify installation
python experiments/train.py --config experiments/configs/smoke_test.yaml
```

---

## Repository Structure

```
├── src/
│   ├── attacks/          # PGD, ConfSmooth, Class-Pair Ambiguity
│   ├── training/         # Unified trainer
│   ├── models/           # ResNet-18 for CIFAR-10
│   ├── data/             # CIFAR-10 data loaders
│   └── utils/            # Config, logging, checkpoints, metrics
├── experiments/
│   ├── configs/          # Training configurations
│   ├── train.py          # Main training script
│   ├── eval.py           # Evaluation script
│   └── plot_training.py  # Training visualization
└── tests/                # Unit and integration tests
```

---

## Reproducing Paper Results

```bash
# Run all methods (vanilla, PGD-AT, UAT-ConfSmooth, UAT-Ambiguity)
python experiments/reproduce_table3.py --epochs 200 --device cuda

# Quick test mode (1 epoch per method)
python experiments/reproduce_table3.py --quick-test
```

**Expected results** after 200 epochs on CIFAR-10:
- Clean accuracy: 80-85%
- PGD robust accuracy: 45-50%
- ConfSmooth robust accuracy: 50-55%

---

## Advanced Usage

### Custom Training

```bash
# Override config parameters via command line
python experiments/train.py \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --overrides "optim.epochs=100,optim.lr=0.05,data.batch_size=256"
```

### Training Visualization

```bash
# Plot training curves
python experiments/plot_training.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_best_model.pt \
  --output plots/training_curves.png

# Compare multiple methods
python experiments/plot_training.py \
  --checkpoint "vanilla.pt,pgd.pt,uat.pt" \
  --labels "Vanilla,PGD-AT,UAT" \
  --output plots/comparison.png
```

### Checkpoint Management

```bash
# List all checkpoints with metadata
python experiments/checkpoint_utils.py list checkpoints/uat_confsmooth_cifar10/ -v

# Find best checkpoint
python experiments/checkpoint_utils.py best checkpoints/uat_confsmooth_cifar10/

# Compare checkpoints
python experiments/checkpoint_utils.py compare ckpt1.pt ckpt2.pt

# Cleanup old checkpoints (dry run)
python experiments/checkpoint_utils.py cleanup checkpoints/uat_confsmooth_cifar10/ --keep 5
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_attacks.py -v

# Run comprehensive smoke test
python tests/smoke_test_comprehensive.py
```

---

## Troubleshooting

### ModuleNotFoundError: No module named 'src'

If you encounter import errors like `ModuleNotFoundError: No module named 'src.data'`, make sure you installed the package properly:

```bash
# Reinstall in editable mode
pip install -e .
```

This ensures that the `src` module is properly added to your Python path.

### Alternative: Manual PYTHONPATH

If you prefer not to install the package, you can run scripts with:

```bash
# Set PYTHONPATH to include project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python experiments/train.py --config experiments/configs/smoke_test.yaml
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{pmlr-v307-marti-nez-marti-nez26a,
  title =      {Improving Vision Model Robustness against Misclassification and Uncertainty Attacks via Underconfidence Adversarial Training},
  author =       {Mart{\'\i}nez-Mart{\'\i}nez, Josu{\'e} and Holodnak, John T and Brown, Olivia and Nabavi, Sheida and Aguiar, Derek and Wollaber, Allan},
  booktitle =      {Proceedings of the 7th Northern Lights Deep Learning Conference (NLDL)},
  pages =      {274--286},
  year =      {2026},
  editor =      {Kim, Hyeongji and Ramírez Rivera, Adín and Ricaud, Benjamin},
  volume =      {307},
  series =      {Proceedings of Machine Learning Research},
  month =      {06--08 Jan},
  publisher =    {PMLR},
  pdf =      {https://raw.githubusercontent.com/mlresearch/v307/main/assets/marti-nez-marti-nez26a/marti-nez-marti-nez26a.pdf},
  url =      {https://proceedings.mlr.press/v307/marti-nez-marti-nez26a.html}
}
```

