# Training Visualization Tools

This directory contains tools for visualizing and analyzing training results.

## plot_training.py

Visualize training curves from saved checkpoints.

### Features

1. **Single Run Visualization**: Plot loss and accuracy curves
2. **Multi-Run Comparison**: Compare multiple training runs side-by-side
3. **Learning Rate Schedule**: Visualize LR changes over training
4. **Training Summary**: Print statistics about training progress

### Usage

#### Plot Single Training Run

```bash
# Basic usage (interactive display)
python experiments/plot_training.py --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt

# Save to file
python experiments/plot_training.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt \
  --output plots/confsmooth_training_curves.png

# Show summary statistics
python experiments/plot_training.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt \
  --summary

# Also plot learning rate schedule
python experiments/plot_training.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt \
  --output plots/confsmooth_training.png \
  --lr-plot
```

#### Compare Multiple Training Runs

```bash
# Compare different training methods
python experiments/plot_training.py \
  --checkpoint "checkpoints/vanilla_cifar10/vanilla_epoch200.pt,checkpoints/pgd_cifar10/pgd_epoch200.pt,checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt" \
  --labels "Vanilla,PGD-AT,UAT-ConfSmooth" \
  --output plots/method_comparison.png

# Labels will default to filename if not provided
python experiments/plot_training.py \
  --checkpoint "checkpoints/vanilla_cifar10/vanilla_epoch200.pt,checkpoints/pgd_cifar10/pgd_epoch200.pt" \
  --output plots/comparison.png
```

#### Batch Mode (No Interactive Display)

```bash
# Useful for scripts and automation
python experiments/plot_training.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt \
  --output plots/confsmooth.png \
  --no-show
```

### Output Examples

**Single Run Plot:**
- Left panel: Training and validation loss over epochs
- Right panel: Training and validation accuracy over epochs

**Comparison Plot:**
- Left panel: Validation accuracy comparison across methods
- Right panel: Validation loss comparison across methods

**Learning Rate Plot:**
- Learning rate schedule over training (log scale)

### Training Summary Example

```
============================================================
TRAINING SUMMARY
============================================================
Total Epochs: 200

Final Metrics (Epoch 200):
  Train Loss: 0.3245
  Train Acc (Clean): 88.45%
  Train Acc (Train): 54.23%
  Val Loss: 0.4156
  Val Acc: 85.32%

Best Validation Accuracy: 85.67% (Epoch 187)
Lowest Validation Loss: 0.4023 (Epoch 192)
============================================================
```

## Requirements

All plotting tools require checkpoints saved with training history. Make sure to train models with the updated trainer (which automatically saves history).

If you have old checkpoints without history, you'll need to retrain or use the eval.py script for single-epoch metrics.

## Tips

1. **High-DPI Plots**: Plots are saved at 300 DPI for publication quality
2. **Multiple Formats**: Change extension in --output to save as PNG, PDF, SVG, etc.
3. **Customization**: Edit plot_training.py to customize colors, styles, or add new plots
4. **Automation**: Use --no-show in training scripts to automatically generate plots

## Integration with Training

The trainer automatically saves training history in checkpoints:

```python
# Training history is automatically tracked
trainer = Trainer(model, train_loader, val_loader, optimizer, device, cfg)
trainer.fit()  # History is saved in checkpoints

# Load checkpoint to access history
checkpoint = torch.load("checkpoints/model.pt")
history = checkpoint["history"]
```

## Example Workflow

```bash
# 1. Train model
python experiments/train.py --config experiments/configs/uat_confsmooth_cifar10.yaml

# 2. Plot training curves
python experiments/plot_training.py \
  --checkpoint checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt \
  --output plots/uat_confsmooth_curves.png \
  --summary \
  --lr-plot

# 3. Compare with other methods
python experiments/plot_training.py \
  --checkpoint "checkpoints/vanilla_cifar10/vanilla_epoch200.pt,checkpoints/pgd_cifar10/pgd_epoch200.pt,checkpoints/uat_confsmooth_cifar10/confsmooth_epoch200.pt" \
  --labels "Vanilla,PGD-AT,UAT-ConfSmooth" \
  --output plots/all_methods_comparison.png
```
