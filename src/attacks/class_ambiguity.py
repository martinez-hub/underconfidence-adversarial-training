"""Class-Pair Ambiguity Attack (novel UAT contribution)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._validation import validate_attack_params


class ClassPairAmbiguityAttack:
    """
    Class-Pair Ambiguity Attack (novel UAT contribution).

    Generates adversarial examples that reduce model confidence by
    creating ambiguity between two specific classes while preserving
    the model's clean-image prediction.

    Key features:
    - Minimizes margin between target class pairs
    - Preserves the model's clean-image prediction via backtracking
    - Adversarial accuracy equals clean accuracy by construction

    Note on the accuracy guarantee: the preserved quantity is the model's
    prediction on the clean image, not the ground-truth label. On a sample the
    model already misclassifies, the attack keeps that same wrong prediction.
    So adversarial accuracy matches clean accuracy; it is not 100% in absolute
    terms.

    Inputs are expected in raw [0, 1] pixel space; the model is responsible for
    any normalization (see ``src.models.resnet.NormalizedModel``).

    Reference: Martinez-Martinez et al. (NLDL 2026), Algorithm 1
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8 / 255,
        alpha: float = 2 / 255,
        num_steps: int = 20,
        target_pair_mode: str = "top2",
        num_classes: int = 10,
    ):
        """
        Initialize Class-Pair Ambiguity Attack.

        Args:
            model: Target model to attack
            epsilon: Maximum perturbation size (L-infinity norm, in [0, 1] pixel units)
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            target_pair_mode: How to select class pairs
                - "random": Random pair per sample
                - "top2": Top-2 predicted classes
                - "fixed": Fixed pairs (e.g., (0,1), (2,3), ...)
            num_classes: Number of classes, used by the "random" and "fixed" modes

        Raises:
            ValueError: If parameters are invalid
        """
        validate_attack_params(model, epsilon, alpha, num_steps)

        valid_modes = ["random", "top2", "fixed"]
        if target_pair_mode not in valid_modes:
            raise ValueError(
                f"target_pair_mode must be one of {valid_modes}, got {target_pair_mode}"
            )

        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.target_pair_mode = target_pair_mode
        self.num_classes = num_classes

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        pair_indices: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Generate ambiguity-inducing adversarial examples.

        Args:
            x: Clean images in [0, 1], shape [batch_size, 3, 32, 32]
            y: Accepted and ignored. The attack preserves the model's own clean
                prediction, so it needs no labels; the parameter exists so this
                attack is interchangeable with PGDAttack.
            pair_indices: Optional [batch_size, 2] specifying target pairs
            return_info: If True, also return a dict of per-sample backtracking
                diagnostics: ``alpha_final`` (the step size each sample ended
                with) and ``backtracked`` (which samples ever backtracked).
                Without this, the per-sample nature of the backtracking is not
                observable from outside.

        Returns:
            x_adv: Adversarial images in [0, 1], within epsilon of x, with
                reduced confidence and the clean prediction preserved.
            info: Only when return_info=True, the diagnostics dict above.
        """
        # CRITICAL: Get prediction on clean image first (this is our target class to maintain)
        with torch.no_grad():
            clean_logits = self.model(x)
            target_class = clean_logits.argmax(dim=1)

        x_adv = x.clone().detach()

        # Random initialization within epsilon-ball
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

        # Verify initial state is valid - if not, fall back to clean images
        with torch.no_grad():
            init_logits = self.model(x_adv)
            init_pred = init_logits.argmax(dim=1)
            invalid_init = ~init_pred.eq(target_class)
            if invalid_init.any():
                # Revert invalid samples to clean images. Clamped so that the
                # return value is in [0, 1] even if the caller passed
                # out-of-domain input.
                x_adv[invalid_init] = torch.clamp(x[invalid_init], 0, 1)

        # Select class pairs (based on clean prediction, not true label).
        # clean_logits is reused so "top2" costs no extra forward pass.
        if pair_indices is None:
            pair_indices = self._select_pairs(x, clean_logits)

        # Track per-sample step sizes (for backtracking)
        # Each sample can have different step size based on its backtracking history
        batch_size = x.shape[0]
        alpha_current = torch.ones(batch_size, device=x.device) * self.alpha

        # Track the best valid adversarial example (maintains correct predictions)
        # At this point, all samples in x_adv are guaranteed to be correctly classified
        x_adv_best = x_adv.clone().detach()

        # Attacks build their own graph, so grad must be enabled even when the
        # caller is inside torch.no_grad().
        with torch.enable_grad():
            for step in range(self.num_steps):
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
                max_incorrect.scatter_(1, target_class.unsqueeze(1), float("-inf"))
                max_incorrect_logit = max_incorrect.max(dim=1)[0]

                # Penalty if any incorrect class has higher logit than target
                constraint_loss = F.relu(max_incorrect_logit - target_logits + 0.1)

                # Total loss: minimize margin while maintaining correct prediction
                total_loss = margin_loss + constraint_loss.mean()

                # Gradient descent step
                grad = torch.autograd.grad(total_loss, x_adv)[0]

                # Update adversarial example with per-sample step sizes
                with torch.no_grad():
                    # Reshape alpha for broadcasting: [batch_size] -> [batch_size, 1, 1, 1]
                    alpha_broadcast = alpha_current.view(batch_size, 1, 1, 1)
                    x_adv = x_adv - alpha_broadcast * grad.sign()

                    # Project to epsilon-ball
                    delta = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                    x_adv = torch.clamp(x + delta, 0, 1)

                # CRITICAL CONSTRAINT: Check for misclassification from target class
                # Selective backtracking: Only revert misclassified samples
                with torch.no_grad():
                    logits_check = self.model(x_adv)
                    pred = logits_check.argmax(dim=1)
                    misclassified = ~pred.eq(target_class)

                    if misclassified.any():
                        # Selective revert: Revert misclassified samples to last known valid state
                        x_adv[misclassified] = x_adv_best[misclassified]
                        # Reduce step size only for misclassified samples
                        alpha_current[misclassified] /= 2.0
                        # Update best ONLY for non-misclassified samples
                        x_adv_best[~misclassified] = x_adv[~misclassified].clone().detach()
                    else:
                        # All samples valid - update all
                        x_adv_best = x_adv.clone().detach()

        # Return the last valid adversarial example that maintains correct predictions
        if return_info:
            return x_adv_best.detach(), {
                "alpha_final": alpha_current.detach().clone(),
                "backtracked": alpha_current.detach() < self.alpha,
            }
        return x_adv_best.detach()

    def _select_pairs(
        self,
        x: torch.Tensor,
        clean_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select class pairs based on mode.

        Args:
            x: Input images, used for batch size and device
            clean_logits: Model logits on the clean images, reused by "top2"

        Returns:
            Tensor of shape [batch_size, 2] with class pair indices
        """
        batch_size = x.shape[0]

        if self.target_pair_mode == "random":
            # Two distinct random classes per sample. Ranking random scores
            # gives distinct indices by construction.
            scores = torch.rand(batch_size, self.num_classes, device=x.device)
            return scores.topk(2, dim=1)[1]

        elif self.target_pair_mode == "top2":
            # Use top-2 predicted classes (no extra forward pass)
            return clean_logits.topk(2, dim=1)[1].detach()

        elif self.target_pair_mode == "fixed":
            # Fixed pairs: (0,1), (2,3), (4,5), ... cycling over the classes
            num_pairs = self.num_classes // 2
            base = (torch.arange(batch_size, device=x.device) % num_pairs) * 2
            return torch.stack([base, base + 1], dim=1)

        else:
            raise ValueError(f"Unknown pair mode: {self.target_pair_mode}")
