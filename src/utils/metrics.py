"""Evaluation metrics."""

from typing import Tuple

import torch
import torch.nn.functional as F


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        logits: Model predictions [batch_size, num_classes]
        labels: Ground truth labels [batch_size]

    Returns:
        Accuracy as percentage (0-100)
    """
    pred = logits.argmax(dim=1)
    correct = pred.eq(labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total


def confidence_stats(
    probs: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute confidence statistics on correct predictions.

    Args:
        probs: Predicted probabilities [batch_size, num_classes]
        labels: Ground truth labels [batch_size]

    Returns:
        (mean_confidence, median_confidence) on correct predictions
    """
    pred = probs.argmax(dim=1)
    correct_mask = pred.eq(labels)

    if correct_mask.sum() == 0:
        return 0.0, 0.0

    correct_probs = probs[correct_mask]
    correct_confidences = correct_probs.max(dim=1)[0]

    mean_conf = correct_confidences.mean().item()
    median_conf = correct_confidences.median().item()

    return mean_conf, median_conf


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute prediction entropy.

    Args:
        probs: Predicted probabilities [batch_size, num_classes]

    Returns:
        Entropy values [batch_size]
    """
    # H(p) = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)  # Add small epsilon for numerical stability
    entropy = -(probs * log_probs).sum(dim=1)
    return entropy
