"""Shared constructor-argument validation for the attacks.

All three attacks take the same epsilon/alpha/num_steps budget, so they share
one validator. Previously only PGDAttack rejected epsilon > 1 and
alpha > epsilon, which let the two UAT attacks be constructed with a budget
that cannot be satisfied in [0, 1] pixel space.
"""

import torch
import torch.nn as nn

# Float slack when checking the input domain, so a tensor that is in [0, 1] up
# to rounding is not rejected.
_DOMAIN_TOL = 1e-6


def validate_attack_params(
    model: nn.Module,
    epsilon: float,
    alpha: float,
    num_steps: int,
) -> None:
    """
    Validate the perturbation budget shared by every attack.

    Args:
        model: Target model to attack
        epsilon: Maximum perturbation size (L-infinity norm, in [0, 1] pixel units)
        alpha: Step size for each iteration
        num_steps: Number of attack iterations

    Raises:
        ValueError: If any parameter is invalid
    """
    if model is None:
        raise ValueError("Model cannot be None")

    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    if epsilon > 1:
        raise ValueError(f"epsilon should typically be <= 1, got {epsilon}")

    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")

    if alpha > epsilon:
        raise ValueError(f"alpha ({alpha}) should typically be <= epsilon ({epsilon})")

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")


def validate_input_domain(x: "torch.Tensor") -> None:
    """
    Require attack inputs to be raw pixels in [0, 1].

    Every attack projects with ``clamp(x + delta, 0, 1)`` and treats ``epsilon``
    as a pixel-space L-infinity budget, so out-of-domain input makes both
    meaningless: the clamp rather than the epsilon-ball decides the result.
    Repairing such input by clamping it would be worse than refusing it -- a
    pixel of -1 becomes 0, a perturbation of 1.0 against a budget of 8/255, and
    the clamped image need not preserve the prediction that was computed from
    the original tensor. Fail loudly instead.

    Normalization belongs inside the model (see
    ``src.models.resnet.NormalizedModel``), not in the data pipeline.

    Args:
        x: Candidate input batch

    Raises:
        ValueError: If any value is non-finite or lies outside [0, 1]
    """
    if x.numel() == 0:
        return

    # Check finiteness FIRST. NaN compares false against everything, so a range
    # test alone silently accepts it: min()/max() both return NaN, both
    # comparisons below are false, and the attack proceeds to produce
    # non-finite "adversarial" examples that then corrupt every downstream
    # metric. (Infinities are caught by the range test, but NaN is not.)
    if not bool(torch.isfinite(x).all()):
        raise ValueError(
            "attack inputs must be finite; got NaN or infinity. This usually "
            "means preprocessing or a checkpoint produced corrupt data."
        )

    low = float(x.min())
    high = float(x.max())
    if low < -_DOMAIN_TOL or high > 1.0 + _DOMAIN_TOL:
        raise ValueError(
            f"attack inputs must be raw pixels in [0, 1], got range "
            f"[{low:.4f}, {high:.4f}]. Normalization belongs inside the model "
            f"(src.models.resnet.NormalizedModel), not the data pipeline: an "
            f"epsilon of 8/255 and the clamp(x, 0, 1) projection are both "
            f"expressed in pixel units."
        )
