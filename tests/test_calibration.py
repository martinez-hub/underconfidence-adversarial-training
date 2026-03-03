"""Tests for calibration metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.calibration import (
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
    compute_mce,
)


@pytest.fixture
def random_predictions():
    """Generate random predictions for testing."""
    torch.manual_seed(42)
    batch_size = 1000
    num_classes = 10

    logits = torch.randn(batch_size, num_classes)
    probs = F.softmax(logits, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))

    return probs, labels


@pytest.fixture
def perfect_predictions():
    """Generate perfect predictions (100% accuracy, perfect calibration)."""
    torch.manual_seed(42)
    batch_size = 1000
    num_classes = 10

    # Create one-hot predictions matching labels
    labels = torch.randint(0, num_classes, (batch_size,))
    probs = F.one_hot(labels, num_classes=num_classes).float()

    return probs, labels


@pytest.fixture
def miscalibrated_predictions():
    """Generate overconfident predictions (high confidence, low accuracy)."""
    torch.manual_seed(42)
    batch_size = 1000
    num_classes = 10

    # Generate random labels
    labels = torch.randint(0, num_classes, (batch_size,))

    # Create overconfident predictions (all 0.9 for one class, 0.01 for others)
    probs = torch.ones(batch_size, num_classes) * 0.01
    random_class = torch.randint(0, num_classes, (batch_size,))
    probs[torch.arange(batch_size), random_class] = 0.9
    probs = probs / probs.sum(dim=1, keepdim=True)  # Normalize

    return probs, labels


def test_ece_bounds(random_predictions):
    """Test that ECE is between 0 and 1."""
    probs, labels = random_predictions
    ece, _, _, _ = compute_ece(probs, labels, n_bins=15)

    assert 0.0 <= ece <= 1.0, f"ECE {ece} out of bounds [0, 1]"


def test_mce_bounds(random_predictions):
    """Test that MCE is between 0 and 1."""
    probs, labels = random_predictions
    mce = compute_mce(probs, labels, n_bins=15)

    assert 0.0 <= mce <= 1.0, f"MCE {mce} out of bounds [0, 1]"


def test_brier_bounds(random_predictions):
    """Test that Brier score is between 0 and 2."""
    probs, labels = random_predictions
    brier = compute_brier_score(probs, labels)

    # Brier score ranges from 0 (perfect) to 2 (worst for binary, approaches 2 for multi-class)
    assert 0.0 <= brier <= 2.0, f"Brier {brier} out of bounds [0, 2]"


def test_perfect_calibration(perfect_predictions):
    """Test that perfect predictions have low calibration error."""
    probs, labels = perfect_predictions

    ece, _, _, _ = compute_ece(probs, labels, n_bins=15)
    mce = compute_mce(probs, labels, n_bins=15)
    brier = compute_brier_score(probs, labels)

    # Perfect predictions should have near-zero ECE, MCE, and Brier
    assert ece < 0.01, f"ECE for perfect predictions should be near 0, got {ece}"
    assert mce < 0.01, f"MCE for perfect predictions should be near 0, got {mce}"
    assert brier < 0.01, f"Brier for perfect predictions should be near 0, got {brier}"


def test_miscalibration_detected(miscalibrated_predictions):
    """Test that miscalibrated predictions have high calibration error."""
    probs, labels = miscalibrated_predictions

    ece, _, _, _ = compute_ece(probs, labels, n_bins=15)
    mce = compute_mce(probs, labels, n_bins=15)

    # Miscalibrated predictions should have higher ECE and MCE
    # (exact values depend on random seed, but should be > 0.1)
    assert ece > 0.01, f"ECE for miscalibrated predictions should be > 0.01, got {ece}"
    assert mce > 0.01, f"MCE for miscalibrated predictions should be > 0.01, got {mce}"


def test_ece_structure(random_predictions):
    """Test that ECE returns correct structure."""
    probs, labels = random_predictions
    ece, bin_boundaries, bin_accuracies, bin_confidences = compute_ece(
        probs, labels, n_bins=15
    )

    # Check types
    assert isinstance(ece, float)
    assert isinstance(bin_boundaries, np.ndarray)
    assert isinstance(bin_accuracies, np.ndarray)
    assert isinstance(bin_confidences, np.ndarray)

    # Check shapes
    assert len(bin_boundaries) == 16  # n_bins + 1
    assert len(bin_accuracies) == 15  # n_bins
    assert len(bin_confidences) == 15  # n_bins


def test_compute_all_metrics(random_predictions):
    """Test that compute_calibration_metrics returns all expected keys."""
    probs, labels = random_predictions
    metrics = compute_calibration_metrics(probs, labels, n_bins=15)

    expected_keys = {
        'ece',
        'mce',
        'brier',
        'bin_boundaries',
        'bin_accuracies',
        'bin_confidences',
    }

    assert set(metrics.keys()) == expected_keys


def test_different_bin_counts(random_predictions):
    """Test ECE with different numbers of bins."""
    probs, labels = random_predictions

    # Test with different bin counts
    for n_bins in [5, 10, 15, 20]:
        ece, bin_boundaries, _, _ = compute_ece(probs, labels, n_bins=n_bins)

        assert len(bin_boundaries) == n_bins + 1
        assert 0.0 <= ece <= 1.0


def test_small_batch(random_predictions):
    """Test metrics work with small batch size."""
    probs, labels = random_predictions

    # Take only first 10 samples
    small_probs = probs[:10]
    small_labels = labels[:10]

    metrics = compute_calibration_metrics(small_probs, small_labels, n_bins=5)

    # Should not crash and should return valid values
    assert 0.0 <= metrics['ece'] <= 1.0
    assert 0.0 <= metrics['mce'] <= 1.0
    assert 0.0 <= metrics['brier'] <= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
