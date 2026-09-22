"""ConfSmooth Attack (novel UAT contribution)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._validation import validate_attack_params


class ConfSmoothAttack:
    """
    ConfSmooth Attack (novel UAT contribution).

    Generates adversarial examples that reduce model confidence by
    pushing predictions toward a nearly-uniform distribution across all classes.

    Key design: The target distribution assigns slightly more probability (~1% more)
    to the target class than to other classes. This helps guide the attack to avoid
    flipping the prediction while still significantly reducing confidence.

    Key features:
    - Minimizes KL divergence to nearly-uniform target distribution
    - Target class gets slight boost (1% more probability)
    - Preserves the model's clean-image prediction via backtracking
    - Adversarial accuracy equals clean accuracy by construction

    Note on the accuracy guarantee: the preserved quantity is the model's
    prediction on the clean image, not the ground-truth label. On a sample the
    model already misclassifies, the attack keeps that same wrong prediction.
    So adversarial accuracy matches clean accuracy; it is not 100% in absolute
    terms.

    Inputs are expected in raw [0, 1] pixel space; the model is responsible for
    any normalization (see ``src.models.resnet.NormalizedModel``).

    Reference: Martinez-Martinez et al. (NLDL 2026), Algorithm 2
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8 / 255,
        alpha: float = 2 / 255,
        num_steps: int = 20,
        num_classes: int = 10,
        target_class_boost: float = 0.01,
    ):
        """
        Initialize ConfSmooth Attack.

        Args:
            model: Target model to attack
            epsilon: Maximum perturbation size (L-infinity norm, in [0, 1] pixel units)
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            num_classes: Number of classes in the dataset
            target_class_boost: Additional probability mass for target class (default: 0.01)

        Raises:
            ValueError: If parameters are invalid
        """
        validate_attack_params(model, epsilon, alpha, num_steps)

        if num_classes <= 1:
            raise ValueError(f"num_classes must be > 1, got {num_classes}")

        if not 0 <= target_class_boost <= 1:
            raise ValueError(f"target_class_boost must be in [0, 1], got {target_class_boost}")

        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.target_class_boost = target_class_boost

        # Nearly-uniform distribution with slight bias toward target class
        # Example for 10 classes with boost=0.01:
        #   - Other classes: (1.0 - 0.01) / 10 = 0.099
        #   - Target class: 0.099 + 0.01 = 0.109
        self.other_class_prob = (1.0 - target_class_boost) / num_classes
        self.target_class_prob = self.other_class_prob + target_class_boost

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        Generate confidence-smoothing adversarial examples.

        Args:
            x: Clean images in [0, 1], shape [batch_size, 3, 32, 32]
            y: Accepted and ignored. The attack preserves the model's own clean
                prediction, so it needs no labels; the parameter exists so this
                attack is interchangeable with PGDAttack.
            return_info: If True, also return a dict of per-sample backtracking
                diagnostics: ``alpha_final`` (the step size each sample ended
                with) and ``backtracked`` (which samples ever backtracked).
                Without this, the per-sample nature of the backtracking is not
                observable from outside.

        Returns:
            x_adv: Adversarial images in [0, 1], within epsilon of x, with
                flattened confidence and the clean prediction preserved.
            info: Only when return_info=True, the diagnostics dict above.
        """
        # CRITICAL: Get prediction on clean image first (this is our target class to maintain)
        with torch.no_grad():
            clean_logits = self.model(x)
            target_class = clean_logits.argmax(dim=1)

        x_adv = x.clone().detach()
        batch_size = x.shape[0]

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

        # Construct nearly-uniform target distribution with slight bias toward target class
        # Target class = predicted class on clean image (NOT true label)
        # Example for 10-class problem with boost=0.01:
        #   [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099]
        #                           ^target class gets 0.01 more
        target_dist = torch.ones(batch_size, self.num_classes).to(x.device) * self.other_class_prob
        target_dist[torch.arange(batch_size), target_class] = self.target_class_prob

        # Track per-sample step sizes (for backtracking)
        # Each sample can have different step size based on its backtracking history
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
                log_probs = F.log_softmax(logits, dim=1)

                # Loss: KL divergence to nearly-uniform target distribution
                # KL(target || model) = sum(target * log(target / model))
                # Minimize this to push predictions toward the flat (but slightly biased) distribution
                kl_loss = F.kl_div(log_probs, target_dist, reduction="batchmean")

                # Gradient descent step
                grad = torch.autograd.grad(kl_loss, x_adv)[0]

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
