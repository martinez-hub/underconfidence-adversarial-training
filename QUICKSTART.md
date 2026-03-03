# Quick Start Guide

Get up and running with UAT in 5 minutes!

## Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)

## Installation

```bash
# Navigate to project directory
cd /Users/josuemartinez/Documents/PersonalProjects/underconfidence-adversarial-training

# Install dependencies
pip install -r requirements.txt
```

## Run Your First Experiment (2 minutes)

### Smoke Test

Verify installation with a 2-epoch quick test:

```bash
python experiments/train.py --config experiments/configs/smoke_test.yaml
```

**Expected output:**
```
[INFO    ][...][__main__] Configuration:
...
[INFO    ][...][__main__] Loaded CIFAR-10: 50000 train, 10000 val
[INFO    ][...][__main__] Model: ResNet-18 (CIFAR-10 variant)
[INFO    ][...][__main__] Initialized ConfSmooth attack (5 steps - 50% fewer than PGD!)
[INFO    ][...][__main__] Starting training...
Epoch 1:   0%|          | 0/782 [00:00<?, ?it/s]
...
[INFO    ][...][trainer] Training complete!
```

CIFAR-10 will download automatically on first run (~170MB).

---

## Train UAT Models

### 1. UAT with ConfSmooth (Recommended - Most Efficient)

Train with ConfSmooth attack (5 gradient steps per batch):

```bash
python experiments/train.py --config experiments/configs/uat_confsmooth_cifar10.yaml
```

**Why this is recommended:**
- ✅ 50% fewer gradient steps than PGD-AT (5 vs 10)
- ✅ Comparable robustness to PGD-AT
- ✅ 2x faster training
- ✅ Novel contribution from paper

**Training time:**
- GPU (RTX 3090): ~5 hours for 200 epochs
- CPU: ~50 hours (not recommended)

### 2. PGD Adversarial Training (Baseline)

Train with standard PGD attack (10 gradient steps per batch):

```bash
python experiments/train.py --config experiments/configs/pgd_cifar10.yaml
```

**Training time:**
- GPU: ~10 hours for 200 epochs
- CPU: ~100 hours (not recommended)

### 3. Vanilla Training (No Adversarial Training)

Train without adversarial examples:

```bash
python experiments/train.py --config experiments/configs/vanilla_cifar10.yaml
```

**Training time:**
- GPU: ~2 hours for 200 epochs
- CPU: ~20 hours

---

## Evaluate Trained Models

After training completes, evaluate on clean and adversarial test sets:

```bash
python experiments/eval.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_final.pt \
  --config experiments/configs/uat_confsmooth_cifar10.yaml
```

**Expected output:**
```
[INFO    ][...][__main__] Evaluating on clean images...
Clean:   0%|          | 0/79 [00:00<?, ?it/s]
...
[INFO    ][...][__main__] Clean accuracy: 82.45%
[INFO    ][...][__main__] Clean confidence: 0.9123

[INFO    ][...][__main__] Evaluating on PGD adversarial images...
PGD:   0%|          | 0/79 [00:00<?, ?it/s]
...
[INFO    ][...][__main__] PGD accuracy: 48.32%
[INFO    ][...][__main__] PGD confidence: 0.7234

[INFO    ][...][__main__] Evaluating on ConfSmooth adversarial images...
ConfSmooth:   0%|          | 0/79 [00:00<?, ?it/s]
...
[INFO    ][...][__main__] ConfSmooth accuracy: 52.67%
[INFO    ][...][__main__] ConfSmooth confidence: 0.6543
```

**Evaluation time:** ~10 minutes on GPU

---

## Custom Training

### Override Config Parameters

```bash
# Train for fewer epochs (faster testing)
python experiments/train.py \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --overrides "optim.epochs=50"

# Change batch size and learning rate
python experiments/train.py \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --overrides "data.batch_size=256,optim.lr=0.05"

# Multiple overrides (comma-separated)
python experiments/train.py \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --overrides "optim.epochs=100,data.batch_size=64,optim.lr=0.05"
```

---

## Run Tests

Verify all attack implementations are correct:

```bash
# Install pytest (if not already installed)
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_attacks.py::test_underconfidence_attacks_maintain_correct_predictions -v
```

**Expected output:**
```
tests/test_attacks.py::test_pgd_attack_increases_loss PASSED
tests/test_attacks.py::test_pgd_satisfies_epsilon_constraint PASSED
tests/test_attacks.py::test_confsmooth_reduces_confidence PASSED
tests/test_attacks.py::test_underconfidence_attacks_maintain_correct_predictions PASSED
...
========================= 10 passed in 15.23s =========================
```

---

## Makefile Commands

Quick shortcuts for common tasks:

```bash
make help                 # Show all available commands
make run-smoke            # Run smoke test
make run-uat-confsmooth   # Train UAT-ConfSmooth
make run-pgd              # Train PGD-AT
make run-vanilla          # Train vanilla
make test                 # Run tests
make format               # Format code (black + isort)
make lint                 # Check code formatting
make clean                # Remove cache files
```

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python experiments/train.py \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --overrides "data.batch_size=64"
```

### Slow Training (CPU)

Use GPU or reduce epochs for testing:
```bash
python experiments/train.py \
  --config experiments/configs/uat_confsmooth_cifar10.yaml \
  --overrides "optim.epochs=10"
```

### CIFAR-10 Download Fails

Download manually from https://www.cs.toronto.edu/~kriz/cifar.html and place in `./data/cifar-10-batches-py/`

### Module Import Errors

Ensure you're running from project root:
```bash
cd /Users/josuemartinez/Documents/PersonalProjects/underconfidence-adversarial-training
python experiments/train.py --config experiments/configs/smoke_test.yaml
```

---

## Next Steps

1. ✅ **Run smoke test** to verify installation
2. ✅ **Run tests** to verify attacks work correctly
3. ✅ **Train UAT-ConfSmooth** for 200 epochs (recommended)
4. ✅ **Evaluate** trained model on test set
5. 📊 **Compare** with PGD-AT and vanilla baselines
6. 🚀 **Extend** to your own datasets/architectures

---

## Getting Help

- **Documentation**: See `README.md` for detailed usage
- **Implementation details**: See `IMPLEMENTATION_SUMMARY.md`
- **Contributing**: See `CONTRIBUTING.md`
- **Issues**: Open a GitHub issue

---

## Paper Citation

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

**Happy training! 🚀**
