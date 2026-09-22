"""Shared constructor-argument validation for the attacks.

All three attacks take the same epsilon/alpha/num_steps budget, so they share
one validator. Previously only PGDAttack rejected epsilon > 1 and
alpha > epsilon, which let the two UAT attacks be constructed with a budget
that cannot be satisfied in [0, 1] pixel space.
"""

import torch.nn as nn


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
