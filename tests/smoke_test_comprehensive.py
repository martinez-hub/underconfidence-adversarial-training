"""Comprehensive smoke test to verify all project components."""

import sys
import tempfile
from pathlib import Path

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("COMPREHENSIVE SMOKE TEST")
print("="*80 + "\n")

# Test 1: Import all modules
print("Test 1: Importing all modules...")
try:
    from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
    from src.attacks.confsmooth import ConfSmoothAttack
    from src.attacks.pgd import PGDAttack
    from src.data.cifar10 import get_cifar10_loaders
    from src.models.resnet import get_resnet18_cifar10
    from src.training.trainer import Trainer
    from src.utils.calibration import (
        compute_brier_score,
        compute_calibration_metrics,
        compute_ece,
        compute_mce,
    )
    from src.utils.checkpoints import load_checkpoint, save_checkpoint
    from src.utils.config import load_config, setup_device, setup_seed, validate_config
    print("✅ All modules imported successfully\n")
except Exception as e:
    print(f"❌ Import failed: {e}\n")
    sys.exit(1)

# Test 2: Model creation
print("Test 2: Creating model...")
try:
    model = get_resnet18_cifar10()
    device = torch.device("cpu")
    model = model.to(device)
    print(f"✅ Model created: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
except Exception as e:
    print(f"❌ Model creation failed: {e}\n")
    sys.exit(1)

# Test 3: Attack initialization
print("Test 3: Initializing attacks...")
try:
    pgd = PGDAttack(model, epsilon=8/255, alpha=2/255, num_steps=5)
    confsmooth = ConfSmoothAttack(model, epsilon=8/255, alpha=2/255, num_steps=3)
    ambiguity = ClassPairAmbiguityAttack(model, epsilon=8/255, alpha=2/255, num_steps=5)
    print("✅ All attacks initialized:")
    print("   - PGD Attack")
    print("   - ConfSmooth Attack")
    print("   - ClassPairAmbiguityAttack\n")
except Exception as e:
    print(f"❌ Attack initialization failed: {e}\n")
    sys.exit(1)

# Test 4: Attack generation
print("Test 4: Generating adversarial examples...")
try:
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    with torch.no_grad():
        clean_logits = model(x)
        clean_pred = clean_logits.argmax(dim=1)

    # Test PGD
    x_adv_pgd = pgd.generate(x, y)
    assert x_adv_pgd.shape == x.shape, "PGD output shape mismatch"

    # Test ConfSmooth
    x_adv_conf = confsmooth.generate(x, y)
    assert x_adv_conf.shape == x.shape, "ConfSmooth output shape mismatch"
    with torch.no_grad():
        conf_pred = model(x_adv_conf).argmax(dim=1)
    assert (conf_pred == clean_pred).all(), "ConfSmooth changed predictions!"

    # Test ClassAmbiguity
    x_adv_amb = ambiguity.generate(x, y)
    assert x_adv_amb.shape == x.shape, "ClassAmbiguity output shape mismatch"
    with torch.no_grad():
        amb_pred = model(x_adv_amb).argmax(dim=1)
    assert (amb_pred == clean_pred).all(), "ClassAmbiguity changed predictions!"

    print("✅ All attacks generated valid adversarial examples")
    print("   - Shape preservation: ✓")
    print("   - Prediction maintenance (ConfSmooth): ✓")
    print("   - Prediction maintenance (ClassAmbiguity): ✓\n")
except Exception as e:
    print(f"❌ Attack generation failed: {e}\n")
    sys.exit(1)

# Test 5: Calibration metrics
print("Test 5: Computing calibration metrics...")
try:
    # Use more samples for calibration metrics (need at least as many samples as bins)
    x_calib = torch.randn(20, 3, 32, 32)
    with torch.no_grad():
        logits_calib = model(x_calib)
    probs = torch.softmax(logits_calib, dim=1)
    labels = torch.randint(0, 10, (20,))

    # compute_ece returns (ece, bin_boundaries, bin_accuracies, bin_confidences)
    ece, _, _, _ = compute_ece(probs, labels, n_bins=10)
    mce = compute_mce(probs, labels, n_bins=10)
    brier = compute_brier_score(probs, labels)

    assert 0 <= ece <= 1, f"ECE out of bounds: {ece}"
    assert 0 <= mce <= 1, f"MCE out of bounds: {mce}"
    assert 0 <= brier <= 2, f"Brier out of bounds: {brier}"

    print("✅ Calibration metrics computed successfully")
    print(f"   - ECE: {ece:.4f}")
    print(f"   - MCE: {mce:.4f}")
    print(f"   - Brier: {brier:.4f}\n")
except Exception as e:
    print(f"❌ Calibration metrics failed: {e}\n")
    sys.exit(1)

# Test 6: Checkpoint saving and loading
print("Test 6: Testing checkpoint save/load...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # Save checkpoint
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        history = {"train_loss": [1.0, 0.9], "val_acc": [50.0, 55.0]}
        save_checkpoint(model, optimizer, epoch=10, path=str(checkpoint_path), history=history)

        assert checkpoint_path.exists(), "Checkpoint not saved"

        # Load checkpoint
        model2 = get_resnet18_cifar10()
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
        epoch = load_checkpoint(str(checkpoint_path), model2, optimizer2)

        assert epoch == 10, f"Epoch mismatch: expected 10, got {epoch}"

        # Verify history was saved
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert "history" in checkpoint, "History not saved in checkpoint"
        assert checkpoint["history"]["train_loss"] == [1.0, 0.9], "History mismatch"

        print("✅ Checkpoint save/load working correctly")
        print("   - Checkpoint saved: ✓")
        print("   - Checkpoint loaded: ✓")
        print("   - History preserved: ✓")
        print("   - Epoch preserved: ✓\n")
except Exception as e:
    print(f"❌ Checkpoint save/load failed: {e}\n")
    sys.exit(1)

# Test 7: Config validation
print("Test 7: Testing config validation...")
try:
    from omegaconf import OmegaConf

    # Valid config
    valid_cfg = OmegaConf.create({
        'meta': {'seed': 42},
        'optim': {'lr': 0.1, 'epochs': 10},
        'attack': {'epsilon': 8/255, 'alpha': 2/255},
        'training': {'attack_type': 'vanilla'},
        'uat': {'target_class_boost': 0.01, 'pair_mode': 'top2'},
    })
    validate_config(valid_cfg)  # Should not raise

    # Invalid config
    invalid_cfg = OmegaConf.create({'optim': {'lr': -0.1}})
    try:
        validate_config(invalid_cfg)
        print("❌ Config validation failed: accepted invalid config\n")
        sys.exit(1)
    except ValueError:
        pass  # Expected

    print("✅ Config validation working correctly")
    print("   - Valid configs accepted: ✓")
    print("   - Invalid configs rejected: ✓\n")
except Exception as e:
    print(f"❌ Config validation failed: {e}\n")
    sys.exit(1)

# Test 8: Data loading
print("Test 8: Testing data loading...")
try:
    # Use very small batch for speed
    train_loader, val_loader = get_cifar10_loaders(batch_size=4, num_workers=0, augment=False)

    # Get one batch
    x_batch, y_batch = next(iter(train_loader))

    assert x_batch.shape == (4, 3, 32, 32), f"Unexpected batch shape: {x_batch.shape}"
    assert y_batch.shape == (4,), f"Unexpected label shape: {y_batch.shape}"
    assert x_batch.min() >= -3 and x_batch.max() <= 3, "Data normalization issue"

    print("✅ Data loading working correctly")
    print(f"   - Train samples: {len(train_loader.dataset):,}")
    print(f"   - Val samples: {len(val_loader.dataset):,}")
    print(f"   - Batch shape: {x_batch.shape}")
    print(f"   - Value range: [{x_batch.min():.2f}, {x_batch.max():.2f}]\n")
except Exception as e:
    print(f"❌ Data loading failed: {e}\n")
    sys.exit(1)

# Test 9: Quick training test (1 iteration)
print("Test 9: Testing training loop (1 iteration)...")
try:
    from omegaconf import OmegaConf

    # Minimal config
    cfg = OmegaConf.create({
        'training': {'attack_type': 'vanilla'},
        'attack': {'epsilon': 8/255, 'alpha': 2/255},
        'uat': {'target_class_boost': 0.01, 'pair_mode': 'top2'},
        'data': {'num_classes': 10},
        'optim': {'epochs': 1, 'milestones': [], 'gamma': 0.1},
        'logging': {'log_every': 1, 'save_every': 1, 'output_dir': '/tmp/smoke_test'},
    })

    # Create tiny loaders (just 2 batches)
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Use only 8 samples
    train_subset = torch.utils.data.Subset(train_dataset, range(8))
    val_subset = torch.utils.data.Subset(val_dataset, range(8))

    tiny_train = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=False, num_workers=0)
    tiny_val = torch.utils.data.DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=0)

    # Train for 1 epoch
    model_train = get_resnet18_cifar10().to(device)
    optimizer_train = torch.optim.SGD(model_train.parameters(), lr=0.1)

    trainer = Trainer(model_train, tiny_train, tiny_val, optimizer_train, device, cfg)

    # Train one epoch
    train_metrics = trainer.train_epoch(epoch=1)
    val_metrics = trainer.validate(epoch=1)

    # Verify metrics
    assert 'train_loss' in train_metrics, "Missing train_loss"
    assert 'val_acc' in val_metrics, "Missing val_acc"
    assert train_metrics['train_loss'] > 0, "Invalid train_loss"
    assert 0 <= val_metrics['val_acc'] <= 100, "Invalid val_acc"

    # Verify history tracking
    assert len(trainer.history['train_loss']) == 0, "History should be empty (fit() not called)"

    print("✅ Training loop working correctly")
    print(f"   - Train loss: {train_metrics['train_loss']:.4f}")
    print(f"   - Val accuracy: {val_metrics['val_acc']:.2f}%")
    print(f"   - History tracking: ✓\n")
except Exception as e:
    print(f"❌ Training loop failed: {e}\n")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("="*80)
print("✅ ALL SMOKE TESTS PASSED")
print("="*80)
print("\nTest Summary:")
print("  1. Module imports: ✅")
print("  2. Model creation: ✅")
print("  3. Attack initialization: ✅")
print("  4. Attack generation: ✅")
print("  5. Calibration metrics: ✅")
print("  6. Checkpoint save/load: ✅")
print("  7. Config validation: ✅")
print("  8. Data loading: ✅")
print("  9. Training loop: ✅")
print("\n🎉 Project is fully functional and ready for use!")
print("="*80 + "\n")
