"""Unit tests for adversarial attacks."""

import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
from src.attacks.confsmooth import ConfSmoothAttack
from src.attacks.pgd import PGDAttack
from src.models.resnet import get_resnet18_cifar10


@pytest.fixture
def model():
    """Create a test model."""
    model = get_resnet18_cifar10()
    model.eval()
    return model


@pytest.fixture
def batch():
    """Create a test batch."""
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    return x, y


def test_pgd_attack_increases_loss(model, batch):
    """PGD attack should increase loss on adversarial examples."""
    x, y = batch

    # Clean loss
    with torch.no_grad():
        clean_logits = model(x)
        clean_loss = F.cross_entropy(clean_logits, y).item()

    # Generate adversarial examples
    pgd = PGDAttack(model, epsilon=8/255, alpha=2/255, num_steps=10)
    x_adv = pgd.generate(x, y)

    # Adversarial loss
    with torch.no_grad():
        adv_logits = model(x_adv)
        adv_loss = F.cross_entropy(adv_logits, y).item()

    assert adv_loss > clean_loss, "PGD attack should increase loss"


def test_pgd_satisfies_epsilon_constraint(model, batch):
    """PGD perturbations should be within epsilon-ball."""
    x, y = batch
    epsilon = 8/255

    pgd = PGDAttack(model, epsilon=epsilon, alpha=2/255, num_steps=10)
    x_adv = pgd.generate(x, y)

    delta = (x_adv - x).abs().max().item()
    assert delta <= epsilon + 1e-6, f"Perturbation {delta} exceeds epsilon {epsilon}"


def test_confsmooth_reduces_confidence(model, batch):
    """ConfSmooth attack should reduce confidence on predictions."""
    x, y = batch

    # Clean confidence
    with torch.no_grad():
        clean_logits = model(x)
        clean_probs = F.softmax(clean_logits, dim=1)
        clean_conf = clean_probs.max(dim=1)[0].mean().item()

    # Generate ConfSmooth adversarial examples
    confsmooth = ConfSmoothAttack(model, epsilon=8/255, alpha=2/255, num_steps=20)
    x_adv = confsmooth.generate(x, y)

    # Adversarial confidence
    with torch.no_grad():
        adv_logits = model(x_adv)
        adv_probs = F.softmax(adv_logits, dim=1)
        adv_conf = adv_probs.max(dim=1)[0].mean().item()

    assert adv_conf < clean_conf, "ConfSmooth should reduce confidence"


def test_confsmooth_satisfies_epsilon_constraint(model, batch):
    """ConfSmooth perturbations should be within epsilon-ball."""
    x, y = batch
    epsilon = 8/255

    confsmooth = ConfSmoothAttack(model, epsilon=epsilon, alpha=2/255, num_steps=20)
    x_adv = confsmooth.generate(x, y)

    delta = (x_adv - x).abs().max().item()
    assert delta <= epsilon + 1e-6, f"Perturbation {delta} exceeds epsilon {epsilon}"


def test_class_ambiguity_reduces_margin(model, batch):
    """Class ambiguity attack should reduce margin between top classes."""
    x, y = batch

    # Clean margin (top-1 vs top-2 logit difference)
    with torch.no_grad():
        clean_logits = model(x)
        top2_logits = clean_logits.topk(2, dim=1)[0]
        clean_margin = (top2_logits[:, 0] - top2_logits[:, 1]).mean().item()

    # Generate ambiguity attack
    ambiguity = ClassPairAmbiguityAttack(model, epsilon=8/255, alpha=2/255, num_steps=20, target_pair_mode="top2")
    x_adv = ambiguity.generate(x, y)

    # Adversarial margin
    with torch.no_grad():
        adv_logits = model(x_adv)
        top2_logits = adv_logits.topk(2, dim=1)[0]
        adv_margin = (top2_logits[:, 0] - top2_logits[:, 1]).mean().item()

    assert adv_margin < clean_margin, "Class ambiguity should reduce margin"


def test_class_ambiguity_satisfies_epsilon_constraint(model, batch):
    """Class ambiguity perturbations should be within epsilon-ball."""
    x, y = batch
    epsilon = 8/255

    ambiguity = ClassPairAmbiguityAttack(model, epsilon=epsilon, alpha=2/255, num_steps=20)
    x_adv = ambiguity.generate(x, y)

    delta = (x_adv - x).abs().max().item()
    assert delta <= epsilon + 1e-6, f"Perturbation {delta} exceeds epsilon {epsilon}"


def test_underconfidence_attacks_maintain_correct_predictions(model, batch):
    """
    CRITICAL TEST: Underconfidence attacks should NEVER cause misclassification.

    This tests the backtracking mechanism constraint.
    """
    x, y = batch

    # Get clean predictions (target classes to maintain)
    with torch.no_grad():
        clean_logits = model(x)
        target_class = clean_logits.argmax(dim=1)

    # Test Class-Pair Ambiguity Attack
    ambiguity = ClassPairAmbiguityAttack(model, epsilon=8/255, alpha=2/255, num_steps=20)
    x_adv_ambiguity = ambiguity.generate(x, y)

    with torch.no_grad():
        logits_ambiguity = model(x_adv_ambiguity)
        pred_ambiguity = logits_ambiguity.argmax(dim=1)
        assert pred_ambiguity.eq(target_class).all(), "Class-Pair Ambiguity Attack caused misclassification!"

    # Test ConfSmooth Attack
    confsmooth = ConfSmoothAttack(model, epsilon=8/255, alpha=2/255, num_steps=20)
    x_adv_confsmooth = confsmooth.generate(x, y)

    with torch.no_grad():
        logits_confsmooth = model(x_adv_confsmooth)
        pred_confsmooth = logits_confsmooth.argmax(dim=1)
        assert pred_confsmooth.eq(target_class).all(), "ConfSmooth Attack caused misclassification!"


def test_attacks_return_correct_shape(model, batch):
    """All attacks should return tensors with the same shape as input."""
    x, y = batch
    epsilon = 8/255
    alpha = 2/255

    pgd = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)
    confsmooth = ConfSmoothAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)
    ambiguity = ClassPairAmbiguityAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)

    x_adv_pgd = pgd.generate(x, y)
    x_adv_confsmooth = confsmooth.generate(x, y)
    x_adv_ambiguity = ambiguity.generate(x, y)

    assert x_adv_pgd.shape == x.shape, "PGD output shape mismatch"
    assert x_adv_confsmooth.shape == x.shape, "ConfSmooth output shape mismatch"
    assert x_adv_ambiguity.shape == x.shape, "ClassAmbiguity output shape mismatch"


def test_attacks_produce_valid_images(model, batch):
    """All attacks should produce images in valid range [0, 1]."""
    x, y = batch
    epsilon = 8/255
    alpha = 2/255

    pgd = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)
    confsmooth = ConfSmoothAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)
    ambiguity = ClassPairAmbiguityAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)

    x_adv_pgd = pgd.generate(x, y)
    x_adv_confsmooth = confsmooth.generate(x, y)
    x_adv_ambiguity = ambiguity.generate(x, y)

    # Check valid range [0, 1]
    assert x_adv_pgd.min() >= 0.0 and x_adv_pgd.max() <= 1.0, "PGD produced invalid pixel values"
    assert x_adv_confsmooth.min() >= 0.0 and x_adv_confsmooth.max() <= 1.0, "ConfSmooth produced invalid pixel values"
    assert x_adv_ambiguity.min() >= 0.0 and x_adv_ambiguity.max() <= 1.0, "ClassAmbiguity produced invalid pixel values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
