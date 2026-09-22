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
    torch.manual_seed(0)
    model = get_resnet18_cifar10()
    model.eval()

    batch_size = 8
    # torch.rand, not torch.randn: attacks take raw pixels in [0, 1]. Unbounded
    # input made these tests fail (and flake) on the output-range assertion
    # below for reasons unrelated to backtracking.
    x = torch.rand(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # Create attack with very aggressive settings to trigger backtracking
    attack = ConfSmoothAttack(
        model,
        epsilon=8 / 255,
        alpha=2 / 255,
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
        assert (
            adv_pred == clean_pred
        ).all(), f"Backtracking should maintain predictions! Clean: {clean_pred}, Adv: {adv_pred}"

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
    torch.manual_seed(0)
    model = get_resnet18_cifar10()
    model.eval()

    batch_size = 8
    # torch.rand, not torch.randn: attacks take raw pixels in [0, 1]. Unbounded
    # input made these tests fail (and flake) on the output-range assertion
    # below for reasons unrelated to backtracking.
    x = torch.rand(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # Create attack with very aggressive settings to trigger backtracking
    attack = ClassPairAmbiguityAttack(
        model,
        epsilon=8 / 255,
        alpha=2 / 255,
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
        assert (
            adv_pred == clean_pred
        ).all(), f"Backtracking should maintain predictions! Clean: {clean_pred}, Adv: {adv_pred}"

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
    torch.manual_seed(0)
    model = get_resnet18_cifar10()
    model.eval()

    batch_size = 16
    x = torch.rand(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # Test ConfSmooth attack
    attack = ConfSmoothAttack(
        model,
        epsilon=8 / 255,
        alpha=2 / 255,
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
        print(
            f"   - All predictions maintained: {(adv_logits.argmax(1) == clean_logits.argmax(1)).all().item()}"
        )

        # Per-sample backtracking should reduce confidence while maintaining predictions
        assert adv_conf < clean_conf, "Attack should reduce confidence"
        assert (
            adv_logits.argmax(1) == clean_logits.argmax(1)
        ).all(), "All predictions should be maintained"


def test_backtracking_is_per_sample_not_whole_batch():
    """
    Backtracking must halve alpha only for the samples that misclassified.

    The two tests above pass just as happily against whole-batch backtracking,
    so they do not actually cover the paper's per-sample claim. This one reads
    the step sizes the attack ends with: under whole-batch backtracking every
    sample would share one alpha.
    """
    torch.manual_seed(0)
    model = get_resnet18_cifar10()
    model.eval()

    batch_size = 32
    x = torch.rand(batch_size, 3, 32, 32)
    y = torch.randint(0, 10, (batch_size,))

    # epsilon == alpha with this seed lands in the regime where only some
    # samples backtrack (11 of 32), which is exactly what distinguishes
    # per-sample from whole-batch behaviour. At 8/255 none backtrack; much
    # above 0.05 they all do.
    attack = ConfSmoothAttack(model, epsilon=0.04, alpha=0.04, num_steps=30, num_classes=10)
    x_adv, info = attack.generate(x, y, return_info=True)

    alpha_final = info["alpha_final"]
    backtracked = info["backtracked"]

    assert alpha_final.shape == (batch_size,), "alpha must be tracked per sample"
    assert backtracked.any(), (
        "no sample backtracked, so this test cannot distinguish per-sample from "
        "whole-batch behaviour - raise alpha or num_steps"
    )
    assert not backtracked.all(), (
        "every sample backtracked; pick settings where only some do so the "
        "per-sample claim is actually exercised"
    )
    assert 0 < int(backtracked.sum()) < batch_size
    assert len(torch.unique(alpha_final)) > 1, (
        f"all samples share one step size ({alpha_final[0]}), which is what "
        "whole-batch backtracking produces"
    )

    # Samples that never backtracked must still hold the initial step size.
    assert torch.allclose(
        alpha_final[~backtracked], torch.full_like(alpha_final[~backtracked], attack.alpha)
    )

    # Each backtrack halves alpha, so every value is alpha / 2^k.
    ratios = attack.alpha / alpha_final[backtracked]
    assert torch.allclose(
        ratios, torch.round(ratios)
    ), f"backtracked step sizes are not powers-of-two divisions: {alpha_final[backtracked]}"

    # And the invariant must still hold for every sample.
    with torch.no_grad():
        assert (model(x_adv).argmax(1) == model(x).argmax(1)).all()


def test_class_ambiguity_also_reports_per_sample_backtracking():
    """ClassPairAmbiguity exposes the same per-sample diagnostics."""
    torch.manual_seed(0)
    model = get_resnet18_cifar10()
    model.eval()

    x = torch.rand(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))

    attack = ClassPairAmbiguityAttack(
        model, epsilon=0.04, alpha=0.04, num_steps=30, target_pair_mode="top2"
    )
    x_adv, info = attack.generate(x, y, return_info=True)

    assert info["alpha_final"].shape == (32,)
    with torch.no_grad():
        assert (model(x_adv).argmax(1) == model(x).argmax(1)).all()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Batch-Wise Backtracking Optimization")
    print("=" * 60 + "\n")

    test_confsmooth_per_sample_backtracking()
    print()
    test_class_ambiguity_per_sample_backtracking()
    print()
    test_backtracking_efficiency_comparison()

    print("\n" + "=" * 60)
    print("✅ All backtracking optimization tests passed!")
    print("=" * 60 + "\n")
