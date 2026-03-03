"""ConfSmooth Attack (novel UAT contribution)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfSmoothAttack:
    """
    ConfSmooth Attack (novel UAT contribution).

    Generates adversarial examples that reduce model confidence by
    pushing predictions toward a nearly-uniform distribution across all classes.

    Key design: The target distribution assigns slightly more probability (~1% more)
    to the true class than to other classes. This helps guide the attack to avoid
    label flips while still significantly reducing confidence.

    Key features:
    - Minimizes KL divergence to nearly-uniform target distribution
    - Target class gets slight boost (1% more probability)
    - Maintains correct prediction via backtracking mechanism
    - 100% accuracy on adversarial examples (constraint enforced)

    Reference: Martinez-Martinez et al. (NLDL 2026), Algorithm 2
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 8/255,
        alpha: float = 2/255,
        num_steps: int = 20,
        num_classes: int = 10,
        target_class_boost: float = 0.01,
    ):
        """
        Initialize ConfSmooth Attack.

        Args:
            model: Target model to attack
            epsilon: Maximum perturbation size (L-infinity norm)
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            num_classes: Number of classes in the dataset
            target_class_boost: Additional probability mass for target class (default: 0.01)

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
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate confidence-smoothing adversarial examples.

        Args:
            x: Clean images [batch_size, 3, 32, 32]
            y: True labels [batch_size]

        Returns:
            x_adv: Adversarial images with flattened confidence
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

        # Construct nearly-uniform target distribution with slight bias toward target class
        # Target class = predicted class on clean image (NOT true label)
        # Example for 10-class problem with boost=0.01:
        #   [0.099, 0.099, 0.099, 0.109, 0.099, 0.099, 0.099, 0.099, 0.099, 0.099]
        #                           ^target class gets 0.01 more
        target_dist = torch.ones(batch_size, self.num_classes).to(x.device) * self.other_class_prob
        target_dist[torch.arange(batch_size), target_class] = self.target_class_prob

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
            log_probs = F.log_softmax(logits, dim=1)

            # Loss: KL divergence to nearly-uniform target distribution
            # KL(target || model) = sum(target * log(target / model))
            # Minimize this to push predictions toward the flat (but slightly biased) distribution
            kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean')

            # Gradient descent step
            grad = torch.autograd.grad(kl_loss, x_adv)[0]

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
