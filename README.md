# Underconfidence Adversarial Training (UAT)

Official implementation of:

**Improving Vision Model Robustness against Misclassification and Uncertainty Attacks via Underconfidence Adversarial Training**

*Josué Martínez-Martínez, John T Holodnak, Olivia Brown, Sheida Nabavi, Derek Aguiar, Allan Wollaber*

Northern Lights Deep Learning Conference (NLDL 2026)

[[Paper (PMLR)]](https://proceedings.mlr.press/v307/marti-nez-marti-nez26a.html) [[OpenReview]](https://openreview.net/forum?id=3upHbaUyR4)

---

## Overview

Adversarially trained vision models are vulnerable to **underconfidence attacks** that reduce prediction confidence without causing misclassification. This repository provides:

### Novel Contributions

1. **Two Novel Attacks**:
   - **Class-Pair Ambiguity Attack**: Reduces confidence by creating ambiguity between specific class pairs while maintaining correct predictions
   - **ConfSmooth Attack**: Spreads probability mass uniformly across classes (entropy maximization) while maintaining correct predictions

2. **UAT Training Framework**: Defends against both misclassification and underconfidence attacks with superior efficiency

3. **Clean, Reproducible Code**: Complete implementation for CIFAR-10 experiments with extensibility to other datasets

### Key Results

- **50% fewer gradient steps** than standard PGD adversarial training (5 vs 10 steps)
- **Superior robustness** in 14 of 15 data-architecture combinations tested
- Maintains **comparable or better** misclassification robustness
- Novel backtracking mechanism ensures 100% correct predictions during attack generation

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/martinez-hub/underconfidence-adversarial-training.git
cd underconfidence-adversarial-training

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Training

```bash
# Train UAT model with ConfSmooth (5 steps - most efficient!)
python experiments/train.py --config experiments/configs/uat_confsmooth_cifar10.yaml

# Train UAT model with Class-Pair Ambiguity (10 steps)
python experiments/train.py --config experiments/configs/uat_ambiguity_cifar10.yaml

# Train PGD adversarial training baseline (10 steps)
python experiments/train.py --config experiments/configs/pgd_cifar10.yaml

# Train vanilla baseline (no adversarial training)
python experiments/train.py --config experiments/configs/vanilla_cifar10.yaml
```

### Quick Smoke Test

```bash
# Run 2-epoch smoke test to verify installation
python experiments/train.py --config experiments/configs/smoke_test.yaml
```

### Evaluation

```bash
# Evaluate trained model on clean and adversarial test sets
python experiments/eval.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_final.pt \
  --config experiments/configs/uat_confsmooth_cifar10.yaml

# Evaluate on clean images only (faster)
python experiments/eval.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_final.pt \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --no-attacks
```

### Custom Configuration

```bash
# Override config parameters via command line
python experiments/train.py \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --overrides "optim.epochs=100,optim.lr=0.05"
```

---

## Project Structure

```
underconfidence-adversarial-training/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── src/
│   ├── attacks/                    # Attack implementations
│   │   ├── pgd.py                  # Standard PGD attack (baseline)
│   │   ├── class_ambiguity.py      # Novel: Class-Pair Ambiguity Attack
│   │   └── confsmooth.py           # Novel: ConfSmooth Attack
│   ├── training/
│   │   └── trainer.py              # Unified trainer (supports all attack types)
│   ├── models/
│   │   └── resnet.py               # ResNet-18 for CIFAR-10
│   ├── data/
│   │   └── cifar10.py              # CIFAR-10 data loaders
│   └── utils/                      # Utilities (config, logging, checkpoints, metrics)
├── experiments/
│   ├── configs/                    # Training configurations
│   ├── train.py                    # Main training script
│   └── eval.py                     # Evaluation script
├── tests/                          # Unit and integration tests
└── checkpoints/                    # Saved model checkpoints
```

---

## Training Methods

All training methods use the **same loss function** (cross-entropy) and **same training procedure**. What differs is the **input image**:

| Method | Input Images | Attack Steps | Efficiency Gain |
|--------|--------------|--------------|-----------------|
| **Vanilla** | Clean images | 0 | Baseline |
| **PGD-AT** | PGD adversarial images | 10 | Baseline |
| **UAT-ConfSmooth** | ConfSmooth adversarial images | **5** | **50% fewer steps** |
| **UAT-Ambiguity** | Class-Pair Ambiguity images | 10 | Same as PGD-AT |

**Key Insight**: UAT-ConfSmooth achieves comparable robustness with **50% fewer gradient steps** (5 vs 10) during attack generation!

---

## Novel Attack Features

### Backtracking Mechanism

Both underconfidence attacks implement a critical constraint:

- **If misclassification occurs during attack generation**:
  1. Revert to previous state
  2. Reduce step size by half
  3. Continue attack iterations

- **Result**: **100% accuracy** on adversarial examples (no label flips)

### Target Class Determination

- Underconfidence attacks get prediction on **clean image first**
- This prediction becomes the **target class to maintain** throughout attack
- Attack modifies confidence distribution while preserving this prediction

### ConfSmooth Target Distribution

- Nearly-uniform distribution with **1% boost** to target class
- Example for 10 classes: `[0.099, 0.099, 0.099, 0.109, ..., 0.099]`
- Biasing toward target class helps avoid label flips naturally

---

## Reproducing Paper Results

### Main Experiments (CIFAR-10)

```bash
# 1. Vanilla baseline (no adversarial training)
python experiments/train.py --config experiments/configs/vanilla_cifar10.yaml

# 2. PGD adversarial training baseline (10 steps)
python experiments/train.py --config experiments/configs/pgd_cifar10.yaml

# 3. UAT with ConfSmooth (5 steps - 50% fewer!)
python experiments/train.py --config experiments/configs/uat_confsmooth_cifar10.yaml

# 4. UAT with Class-Pair Ambiguity (10 steps)
python experiments/train.py --config experiments/configs/uat_ambiguity_cifar10.yaml
```

### Expected Results (200 epochs)

| Method | Clean Acc | PGD Robust Acc | ConfSmooth Robust Acc |
|--------|-----------|----------------|----------------------|
| Vanilla | 85-90% | 0-5% | 0-5% |
| PGD-AT | 80-85% | 45-50% | 40-45% |
| UAT-ConfSmooth | 80-85% | 45-50% | 50-55% |
| UAT-Ambiguity | 80-85% | 45-50% | 48-53% |

*Note: Exact numbers may vary due to random initialization. Run with multiple seeds for reliable comparisons.*

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_attacks.py -v
```

### Code Formatting

```bash
# Install formatting tools
pip install black isort

# Format code
black src/ experiments/ tests/
isort src/ experiments/ tests/

# Check formatting
black --check src/ experiments/ tests/
isort --check src/ experiments/ tests/
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{martinez2026uat,
  title={Improving Vision Model Robustness against Misclassification and Uncertainty Attacks via Underconfidence Adversarial Training},
  author={Mart{\'i}nez-Mart{\'i}nez, Josu{\'e} and Holodnak, John T and Brown, Olivia and Nabavi, Sheida and Aguiar, Derek and Wollaber, Allan},
  booktitle={Northern Lights Deep Learning Conference},
  pages={274--286},
  year={2026},
  organization={PMLR}
}
```

---

## License

MIT License - see LICENSE file for details

---

## Contact

**Josué Martínez-Martínez**
- Email: josuemartinezphd@gmail.com
- Project: [https://github.com/martinez-hub/underconfidence-adversarial-training](https://github.com/martinez-hub/underconfidence-adversarial-training)

---

## Acknowledgments

This research was conducted at Lawrence Livermore National Laboratory and University of Connecticut. Special thanks to the co-authors and the NLDL 2026 reviewers for their valuable feedback.

---

## Future Extensions

Planned extensions for this repository:

1. **Additional datasets**: CIFAR-100, ImageNet, MSTAR
2. **Additional architectures**: Wide ResNets, Vision Transformers, EfficientNets
3. **Confidence calibration metrics**: ECE, MCE, Brier score
4. **Visualization tools**: t-SNE plots, decision boundary visualizations
5. **Attack curriculum**: Progressive training strategies
6. **Multi-GPU support**: Distributed training for large-scale experiments

Contributions are welcome! Please open an issue or pull request.
