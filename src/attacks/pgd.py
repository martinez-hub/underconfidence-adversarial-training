"""Projected Gradient Descent (PGD) adversarial attack."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    Generates adversarial examples by iteratively perturbing inputs
    to maximize the cross-entropy loss while staying within epsilon-ball.

    Reference: Madry et al. (2018) - Towards Deep Learning Models Resistant to Adversarial Attacks
    https://arxiv.org/abs/1706.06083
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_steps: int = 20,
        random_start: bool = True,
    ):
        """
        Initialize PGD attack.

        Args:
            model: Target model to attack
            epsilon: Maximum perturbation size (L-infinity norm)
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            random_start: Whether to start from random point in epsilon-ball
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate adversarial examples.

        Args:
            x: Clean images [batch_size, 3, 32, 32]
            y: True labels [batch_size]

        Returns:
            x_adv: Adversarial images [batch_size, 3, 32, 32]
        """
        x_adv = x.clone().detach()

        # Random initialization within epsilon-ball
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)

        # PGD attack loop
        for step in range(self.num_steps):
            # Create a leaf tensor for gradient computation
            x_adv = x_adv.detach().clone()
            x_adv.requires_grad = True

            # Forward pass
            logits = self.model(x_adv)
            loss = F.cross_entropy(logits, y)

            # Backward pass to get gradient
            grad = torch.autograd.grad(loss, x_adv)[0]

            # Take step in direction of gradient (maximize loss)
            with torch.no_grad():
                x_adv = x_adv + self.alpha * grad.sign()

                # Project back to epsilon-ball around x
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)

        return x_adv.detach()
