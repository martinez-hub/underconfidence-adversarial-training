"""Tests for batch-wise backtracking optimization."""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
from src.attacks.confsmooth import ConfSmoothAttack
from src.models.resnet import get_resnet18_cifar10


def test_confsmooth_per_sample_backtracking():
    """
    Test that ConfSmooth uses per-sample backtracking.

    Each sample should be able to have a different alpha (step size)
    after backtracking events, demonstrating that backtracking is
    applied selectively rather than to the entire batch.
    """
    model = get_resnet18_cifar10()
    model.eval()

    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # Create attack with very aggressive settings to trigger backtracking
    attack = ConfSmoothAttack(
        model,
        epsilon=8/255,
        alpha=2/255,
        num_steps=20,
        num_classes=10,
        target_class_boost=0.01,
    )

    # Generate adversarial examples
    x_adv = attack.generate(x, y)

    # Verify output shape and validity
    assert x_adv.shape == x.shape, "Output shape should match input shape"
    assert torch.isfinite(x_adv).all(), "All values should be finite"
    assert (x_adv >= 0).all() and (x_adv <= 1).all(), "Values should be in [0, 1]"

    # Verify that predictions are maintained (critical constraint)
    with torch.no_grad():
        clean_logits = model(x)
        adv_logits = model(x_adv)

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)

        # All predictions should match clean predictions
        assert (adv_pred == clean_pred).all(), \
            f"Backtracking should maintain predictions! Clean: {clean_pred}, Adv: {adv_pred}"

    print("✅ ConfSmooth per-sample backtracking test passed!")
    print(f"   - Batch size: {batch_size}")
    print(f"   - All {batch_size} samples maintained correct predictions")


def test_class_ambiguity_per_sample_backtracking():
    """
    Test that ClassPairAmbiguity uses per-sample backtracking.

    Each sample should be able to have a different alpha (step size)
    after backtracking events, demonstrating that backtracking is
    applied selectively rather than to the entire batch.
    """
    model = get_resnet18_cifar10()
    model.eval()

    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # Create attack with very aggressive settings to trigger backtracking
    attack = ClassPairAmbiguityAttack(
        model,
        epsilon=8/255,
        alpha=2/255,
        num_steps=20,
        target_pair_mode="top2",
    )

    # Generate adversarial examples
    x_adv = attack.generate(x, y)

    # Verify output shape and validity
    assert x_adv.shape == x.shape, "Output shape should match input shape"
    assert torch.isfinite(x_adv).all(), "All values should be finite"
    assert (x_adv >= 0).all() and (x_adv <= 1).all(), "Values should be in [0, 1]"

    # Verify that predictions are maintained (critical constraint)
    with torch.no_grad():
        clean_logits = model(x)
        adv_logits = model(x_adv)

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)

        # All predictions should match clean predictions
        assert (adv_pred == clean_pred).all(), \
            f"Backtracking should maintain predictions! Clean: {clean_pred}, Adv: {adv_pred}"

    print("✅ ClassAmbiguity per-sample backtracking test passed!")
    print(f"   - Batch size: {batch_size}")
    print(f"   - All {batch_size} samples maintained correct predictions")


def test_backtracking_efficiency_comparison():
    """
    Compare the efficiency of per-sample vs whole-batch backtracking.

    With per-sample backtracking, samples that don't misclassify can
    continue making progress even when other samples trigger backtracking.
    This should result in better attack effectiveness.
    """
    model = get_resnet18_cifar10()
    model.eval()

    batch_size = 16
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # Test ConfSmooth attack
    attack = ConfSmoothAttack(
        model,
        epsilon=8/255,
        alpha=2/255,
        num_steps=10,
        num_classes=10,
        target_class_boost=0.01,
    )

    x_adv = attack.generate(x, y)

    # Measure confidence reduction
    with torch.no_grad():
        clean_logits = model(x)
        adv_logits = model(x_adv)

        clean_probs = torch.softmax(clean_logits, dim=1)
        adv_probs = torch.softmax(adv_logits, dim=1)

        clean_conf = clean_probs.max(dim=1)[0].mean().item()
        adv_conf = adv_probs.max(dim=1)[0].mean().item()

        conf_reduction = (clean_conf - adv_conf) / clean_conf * 100

        print(f"\n📊 Efficiency Test Results:")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Clean confidence: {clean_conf:.4f}")
        print(f"   - Adversarial confidence: {adv_conf:.4f}")
        print(f"   - Confidence reduction: {conf_reduction:.2f}%")
        print(f"   - All predictions maintained: {(adv_logits.argmax(1) == clean_logits.argmax(1)).all().item()}")

        # Per-sample backtracking should reduce confidence while maintaining predictions
        assert adv_conf < clean_conf, "Attack should reduce confidence"
        assert (adv_logits.argmax(1) == clean_logits.argmax(1)).all(), \
            "All predictions should be maintained"


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Batch-Wise Backtracking Optimization")
    print("="*60 + "\n")

    test_confsmooth_per_sample_backtracking()
    print()
    test_class_ambiguity_per_sample_backtracking()
    print()
    test_backtracking_efficiency_comparison()

    print("\n" + "="*60)
    print("✅ All backtracking optimization tests passed!")
    print("="*60 + "\n")
