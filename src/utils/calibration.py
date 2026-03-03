"""
Confidence calibration metrics for model evaluation.

Implements:
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Reliability diagrams

Reference:
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- Naeini et al. (2015) "Obtaining Well Calibrated Probabilities"
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between confidence and accuracy across bins.
    Lower ECE indicates better calibration.

    Args:
        probs: Predicted probabilities [batch_size, num_classes]
        labels: True labels [batch_size]
        n_bins: Number of bins to use for calibration (default: 15)

    Returns:
        ece: Expected calibration error (scalar)
        bin_boundaries: Bin boundaries [n_bins + 1]
        bin_accuracies: Accuracy in each bin [n_bins]
        bin_confidences: Average confidence in each bin [n_bins]
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    # Convert to numpy for binning
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy().astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()

            # ECE: weighted sum of |accuracy - confidence|
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(confidence_in_bin)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return ece, bin_boundaries, np.array(bin_accuracies), np.array(bin_confidences)


def compute_mce(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum difference between confidence and accuracy across bins.
    It captures the worst-case calibration error.

    Args:
        probs: Predicted probabilities [batch_size, num_classes]
        labels: True labels [batch_size]
        n_bins: Number of bins to use (default: 15)

    Returns:
        mce: Maximum calibration error (scalar)
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    # Convert to numpy
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy().astype(float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            confidence_in_bin = confidences[in_bin].mean()

            # MCE: maximum |accuracy - confidence|
            mce = max(mce, np.abs(accuracy_in_bin - confidence_in_bin))

    return mce


def compute_brier_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute Brier Score (also known as Mean Squared Error of probabilities).

    Brier score measures the mean squared difference between predicted
    probabilities and true labels (one-hot encoded).
    Lower is better (0 = perfect, 1 = worst).

    Args:
        probs: Predicted probabilities [batch_size, num_classes]
        labels: True labels [batch_size]

    Returns:
        brier_score: Brier score (scalar)
    """
    num_classes = probs.shape[1]

    # One-hot encode labels
    targets = F.one_hot(labels, num_classes=num_classes).float()

    # Brier score: mean squared error
    brier = ((probs - targets) ** 2).sum(dim=1).mean()

    return brier.item()


def compute_calibration_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict:
    """
    Compute all calibration metrics at once.

    Args:
        probs: Predicted probabilities [batch_size, num_classes]
        labels: True labels [batch_size]
        n_bins: Number of bins for ECE/MCE (default: 15)

    Returns:
        Dictionary with calibration metrics:
        - ece: Expected calibration error
        - mce: Maximum calibration error
        - brier: Brier score
        - bin_boundaries: Bin boundaries for reliability diagram
        - bin_accuracies: Accuracy in each bin
        - bin_confidences: Average confidence in each bin
    """
    ece, bin_boundaries, bin_accuracies, bin_confidences = compute_ece(
        probs, labels, n_bins
    )
    mce = compute_mce(probs, labels, n_bins)
    brier = compute_brier_score(probs, labels)

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'bin_boundaries': bin_boundaries,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
    }


def plot_reliability_diagram(
    bin_boundaries: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    save_path: str = None,
):
    """
    Plot reliability diagram showing calibration quality.

    A reliability diagram plots confidence vs accuracy. A perfectly
    calibrated model would have points on the diagonal line y=x.

    Args:
        bin_boundaries: Bin boundaries [n_bins + 1]
        bin_accuracies: Accuracy in each bin [n_bins]
        bin_confidences: Average confidence in each bin [n_bins]
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')

    # Plot actual calibration
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    # Only plot non-empty bins
    mask = bin_accuracies > 0
    ax.plot(
        bin_confidences[mask],
        bin_accuracies[mask],
        'o-',
        label='Model calibration',
        linewidth=2,
        markersize=8,
    )

    # Add bar chart showing sample distribution
    ax2 = ax.twinx()
    ax2.bar(
        bin_centers,
        np.ones_like(bin_centers),  # Placeholder - would need sample counts
        width=1.0/len(bin_centers),
        alpha=0.3,
        color='blue',
        label='Sample distribution',
    )
    ax2.set_ylabel('Sample density', fontsize=12)
    ax2.set_ylim(0, 1.5)

    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram', fontsize=16)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)

    # Simulate predictions
    batch_size = 1000
    num_classes = 10

    # Well-calibrated model (confidence matches accuracy)
    logits = torch.randn(batch_size, num_classes)
    probs = F.softmax(logits, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Compute metrics
    metrics = compute_calibration_metrics(probs, labels, n_bins=15)

    print("Calibration Metrics:")
    print(f"  ECE:   {metrics['ece']:.4f}")
    print(f"  MCE:   {metrics['mce']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")

    # Plot reliability diagram
    plot_reliability_diagram(
        metrics['bin_boundaries'],
        metrics['bin_accuracies'],
        metrics['bin_confidences'],
        save_path='reliability_diagram.png',
    )
