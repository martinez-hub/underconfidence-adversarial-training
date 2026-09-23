"""Unit tests for adversarial attacks."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))

from src.attacks.class_ambiguity import ClassPairAmbiguityAttack
from src.attacks.confsmooth import ConfSmoothAttack
from src.attacks.pgd import PGDAttack
from src.models.resnet import get_resnet18_cifar10


@pytest.fixture
def model():
    """Create a test model."""
    torch.manual_seed(0)
    model = get_resnet18_cifar10()
    model.eval()
    return model


@pytest.fixture
def batch():
    """
    Create a test batch of images in [0, 1].

    torch.rand, not torch.randn: the attacks are documented to take raw pixels
    in [0, 1] and clamp their output to that range. Feeding unbounded noise made
    the clamp -- not the epsilon-ball -- decide the result, so the
    epsilon-constraint tests failed for a reason that had nothing to do with the
    projection they were meant to check. Seeded so the suite is deterministic.
    """
    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)
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
    pgd = PGDAttack(model, epsilon=8 / 255, alpha=2 / 255, num_steps=10)
    x_adv = pgd.generate(x, y)

    # Adversarial loss
    with torch.no_grad():
        adv_logits = model(x_adv)
        adv_loss = F.cross_entropy(adv_logits, y).item()

    assert adv_loss > clean_loss, "PGD attack should increase loss"


def test_pgd_satisfies_epsilon_constraint(model, batch):
    """PGD perturbations should be within epsilon-ball."""
    x, y = batch
    epsilon = 8 / 255

    pgd = PGDAttack(model, epsilon=epsilon, alpha=2 / 255, num_steps=10)
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
    confsmooth = ConfSmoothAttack(model, epsilon=8 / 255, alpha=2 / 255, num_steps=20)
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
    epsilon = 8 / 255

    confsmooth = ConfSmoothAttack(model, epsilon=epsilon, alpha=2 / 255, num_steps=20)
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
    ambiguity = ClassPairAmbiguityAttack(
        model, epsilon=8 / 255, alpha=2 / 255, num_steps=20, target_pair_mode="top2"
    )
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
    epsilon = 8 / 255

    ambiguity = ClassPairAmbiguityAttack(model, epsilon=epsilon, alpha=2 / 255, num_steps=20)
    x_adv = ambiguity.generate(x, y)

    delta = (x_adv - x).abs().max().item()
    assert delta <= epsilon + 1e-6, f"Perturbation {delta} exceeds epsilon {epsilon}"


def test_underconfidence_attacks_maintain_correct_predictions(model, batch):
    """
    CRITICAL TEST: Underconfidence attacks should NEVER cause misclassification.

    This tests the backtracking mechanism constraint.
    """
    x, y = batch
    epsilon = 8 / 255

    # Get clean predictions (target classes to maintain)
    with torch.no_grad():
        clean_logits = model(x)
        target_class = clean_logits.argmax(dim=1)

    for name, attack in (
        (
            "Class-Pair Ambiguity",
            ClassPairAmbiguityAttack(model, epsilon=epsilon, alpha=2 / 255, num_steps=20),
        ),
        ("ConfSmooth", ConfSmoothAttack(model, epsilon=epsilon, alpha=2 / 255, num_steps=20)),
    ):
        x_adv = attack.generate(x, y)

        # Prediction preserved - the actual claim under test.
        with torch.no_grad():
            pred_adv = model(x_adv).argmax(dim=1)
        assert pred_adv.eq(target_class).all(), f"{name} Attack caused misclassification!"

        # ...but a generate() that simply returned x would also satisfy that.
        # Assert the attack did real work, within its budget.
        assert not torch.allclose(x_adv, x), f"{name} Attack was a no-op"
        delta = (x_adv - x).abs().max().item()
        assert delta <= epsilon + 1e-6, f"{name} exceeded epsilon: {delta} > {epsilon}"


def test_attacks_return_correct_shape(model, batch):
    """All attacks should return tensors with the same shape as input."""
    x, y = batch
    epsilon = 8 / 255
    alpha = 2 / 255

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
    epsilon = 8 / 255
    alpha = 2 / 255

    pgd = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)
    confsmooth = ConfSmoothAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)
    ambiguity = ClassPairAmbiguityAttack(model, epsilon=epsilon, alpha=alpha, num_steps=10)

    x_adv_pgd = pgd.generate(x, y)
    x_adv_confsmooth = confsmooth.generate(x, y)
    x_adv_ambiguity = ambiguity.generate(x, y)

    # Check valid range [0, 1]
    assert x_adv_pgd.min() >= 0.0 and x_adv_pgd.max() <= 1.0, "PGD produced invalid pixel values"
    assert (
        x_adv_confsmooth.min() >= 0.0 and x_adv_confsmooth.max() <= 1.0
    ), "ConfSmooth produced invalid pixel values"
    assert (
        x_adv_ambiguity.min() >= 0.0 and x_adv_ambiguity.max() <= 1.0
    ), "ClassAmbiguity produced invalid pixel values"


@pytest.mark.parametrize("attack_cls", [PGDAttack, ConfSmoothAttack, ClassPairAmbiguityAttack])
def test_attacks_respect_epsilon_on_pipeline_data(model, attack_cls):
    """
    Attacks must honour the epsilon-ball on data straight from the real loader.

    This is the regression test for the domain bug: while the data pipeline
    normalized images, every attack silently perturbed ~73% of pixels far beyond
    epsilon (max 2.43 against a budget of 0.031), because clamp(x, 0, 1) rather
    than the projection decided the output. No test exercised loader data, so
    nothing caught it.
    """
    from src.data.cifar10 import get_cifar10_loaders

    epsilon = 8 / 255
    *_, test_loader = get_cifar10_loaders(batch_size=8, num_workers=0, augment=False)
    x, y = next(iter(test_loader))

    assert (
        x.min() >= 0.0 and x.max() <= 1.0
    ), f"loader must yield [0, 1] pixels, got [{x.min():.3f}, {x.max():.3f}]"

    x_adv = attack_cls(model, epsilon=epsilon, alpha=2 / 255, num_steps=5).generate(x, y)

    delta = (x_adv - x).abs().max().item()
    assert (
        delta <= epsilon + 1e-6
    ), f"{attack_cls.__name__} perturbation {delta} exceeds epsilon {epsilon}"
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0


@pytest.mark.parametrize("attack_cls", [PGDAttack, ConfSmoothAttack, ClassPairAmbiguityAttack])
def test_attacks_reject_out_of_domain_input(model, attack_cls):
    """
    generate() must refuse input outside [0, 1] rather than repair it.

    Clamping such input would silently manufacture a perturbation far outside
    epsilon -- a pixel of -1 becomes 0, i.e. a delta of 1.0 against a budget of
    8/255 -- and the clamped image need not preserve the prediction computed
    from the original tensor. Since epsilon and the clamp projection are both
    expressed in pixel units, out-of-domain input has no correct handling
    except refusal.
    """
    attack = attack_cls(model, epsilon=8 / 255, alpha=2 / 255, num_steps=2)
    y = torch.randint(0, 10, (2,))

    # Normalized-looking data: what the old data pipeline produced.
    with pytest.raises(ValueError, match=r"raw pixels in \[0, 1\]"):
        attack.generate(torch.randn(2, 3, 32, 32) * 2, y)

    # Just below 0 and just above 1 must both be caught.
    with pytest.raises(ValueError, match=r"raw pixels in \[0, 1\]"):
        attack.generate(torch.full((2, 3, 32, 32), -0.01), y)
    with pytest.raises(ValueError, match=r"raw pixels in \[0, 1\]"):
        attack.generate(torch.full((2, 3, 32, 32), 1.01), y)

    # The exact boundaries are valid and must NOT raise.
    attack.generate(torch.zeros(2, 3, 32, 32), y)
    attack.generate(torch.ones(2, 3, 32, 32), y)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_attacks_reject_non_finite_input(model, bad):
    """
    Non-finite pixels must be refused, NaN included.

    NaN compares false against everything, so a pure range check silently
    accepts it: min() and max() both return NaN, both bounds tests are false,
    and the attack goes on to emit non-finite "adversarial" examples that
    corrupt every metric computed from them.
    """
    attack = PGDAttack(model, epsilon=8 / 255, alpha=2 / 255, num_steps=2)
    x = torch.rand(2, 3, 32, 32)
    x[0, 0, 0, 0] = bad
    y = torch.randint(0, 10, (2,))

    with pytest.raises(ValueError, match="finite"):
        attack.generate(x, y)


def test_attacks_work_inside_no_grad(model, batch):
    """
    generate() must work even when the caller is inside torch.no_grad().

    reproduce_table3.py evaluated every attack inside a no_grad block, which
    made torch.autograd.grad raise and the Table-3 reproduction impossible.
    """
    x, y = batch

    with torch.no_grad():
        x_adv = PGDAttack(model, epsilon=8 / 255, alpha=2 / 255, num_steps=3).generate(x, y)

    assert x_adv.shape == x.shape
    assert not torch.allclose(x_adv, x), "attack was a no-op under no_grad"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
