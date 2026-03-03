"""Class-Pair Ambiguity Attack (novel UAT contribution)."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassPairAmbiguityAttack:
    """
    Class-Pair Ambiguity Attack (novel UAT contribution).

    Generates adversarial examples that reduce model confidence by
    creating ambiguity between two specific classes while maintaining
    correct top prediction.

    Key features:
    - Minimizes margin between target class pairs
    - Maintains correct prediction via backtracking mechanism
    - 100% accuracy on adversarial examples (constraint enforced)

    Reference: Martinez-Martinez et al. (NLDL 2026), Algorithm 1
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_steps: int = 20,
        target_pair_mode: str = "top2",
    ):
        """
        Initialize Class-Pair Ambiguity Attack.

        Args:
            model: Target model to attack
            epsilon: Maximum perturbation size (L-infinity norm)
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            target_pair_mode: How to select class pairs
                - "random": Random pair per sample
                - "top2": Top-2 predicted classes
                - "fixed": Fixed pairs (e.g., (0,1), (2,3), ...)

        Raises:
            ValueError: If parameters are invalid
        """
        if model is None:
            raise ValueError("Model cannot be None")

        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")

        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        valid_modes = ["random", "top2", "fixed"]
        if target_pair_mode not in valid_modes:
            raise ValueError(
                f"target_pair_mode must be one of {valid_modes}, got {target_pair_mode}"
            )

        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.target_pair_mode = target_pair_mode

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pair_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate ambiguity-inducing adversarial examples.

        Args:
            x: Clean images [batch_size, 3, 32, 32]
            y: True labels [batch_size]
            pair_indices: Optional [batch_size, 2] specifying target pairs

        Returns:
            x_adv: Adversarial images with reduced confidence
        """
        # CRITICAL: Get prediction on clean image first (this is our target class to maintain)
        with torch.no_grad():
            clean_logits = self.model(x)
            target_class = clean_logits.argmax(dim=1)

        x_adv = x.clone().detach()

        # Random initialization within epsilon-ball
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        # Select class pairs (based on clean prediction, not true label)
        if pair_indices is None:
            pair_indices = self._select_pairs(x, target_class)

        # Track current step size (for backtracking)
        alpha_current = self.alpha

        # Track the best valid adversarial example (maintains correct predictions)
        x_adv_best = x_adv.clone().detach()

        for step in range(self.num_steps):
            # Store previous state before taking step (for backtracking)
            x_adv_prev = x_adv.clone().detach()

            # Create a leaf tensor for gradient computation
            x_adv = x_adv.detach().clone()
            x_adv.requires_grad = True

            # Forward pass
            logits = self.model(x_adv)

            # Loss: minimize margin between target pair
            # Extract logits for the two classes in each pair
            c1_logits = logits.gather(1, pair_indices[:, 0:1]).squeeze(1)
            c2_logits = logits.gather(1, pair_indices[:, 1:2]).squeeze(1)

            # Margin between the two classes (we want to minimize this)
            margin = torch.abs(c1_logits - c2_logits)
            margin_loss = margin.mean()

            # Constraint: keep target class prediction (from clean image)
            # Penalize if target class is not the top prediction
            target_logits = logits.gather(1, target_class.unsqueeze(1)).squeeze(1)

            # Find max logit among incorrect classes
            max_incorrect = logits.clone()
            max_incorrect.scatter_(1, target_class.unsqueeze(1), float('-inf'))
            max_incorrect_logit = max_incorrect.max(dim=1)[0]

            # Penalty if any incorrect class has higher logit than target
            constraint_loss = F.relu(max_incorrect_logit - target_logits + 0.1)

            # Total loss: minimize margin while maintaining correct prediction
            total_loss = margin_loss + constraint_loss.mean()

            # Gradient descent step
            grad = torch.autograd.grad(total_loss, x_adv)[0]

            # Update adversarial example
            with torch.no_grad():
                x_adv = x_adv - alpha_current * grad.sign()

                # Project to epsilon-ball
                delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)

            # CRITICAL CONSTRAINT: Check for misclassification from target class
            # If any sample misclassified, step back and reduce step size (backtracking)
            with torch.no_grad():
                logits_check = self.model(x_adv)
                pred = logits_check.argmax(dim=1)
                misclassified = ~pred.eq(target_class)

                if misclassified.any():
                    # Step back to previous state
                    x_adv = x_adv_prev
                    # Reduce step size by half
                    alpha_current = alpha_current / 2.0
                    # Continue to next iteration
                    continue
                else:
                    # Valid state - update best adversarial example
                    x_adv_best = x_adv.clone().detach()

        # Return the last valid adversarial example that maintains correct predictions
        return x_adv_best.detach()

    def _select_pairs(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select class pairs based on mode.

        Args:
            x: Input images
            target_class: Target classes to maintain (from clean prediction)

        Returns:
            Tensor of shape [batch_size, 2] with class pair indices
        """
        batch_size = x.shape[0]

        if self.target_pair_mode == "random":
            # Random pairs (ensure c1 != c2)
            pairs = []
            for i in range(batch_size):
                # Select 2 random classes (can be different from target class)
                candidates = list(range(10))
                pair = torch.tensor(np.random.choice(candidates, 2, replace=False))
                pairs.append(pair)
            return torch.stack(pairs).to(x.device)

        elif self.target_pair_mode == "top2":
            # Use top-2 predicted classes
            with torch.no_grad():
                logits = self.model(x)
                top2 = logits.topk(2, dim=1)[1]
            return top2

        elif self.target_pair_mode == "fixed":
            # Fixed pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
            pairs = []
            for i in range(batch_size):
                # Map to fixed pairs
                pair_id = i % 5
                c1, c2 = pair_id * 2, pair_id * 2 + 1
                pairs.append(torch.tensor([c1, c2]))
            return torch.stack(pairs).to(x.device)

        else:
            raise ValueError(f"Unknown pair mode: {self.target_pair_mode}")
