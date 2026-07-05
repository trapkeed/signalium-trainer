"""Calibration metrics for evaluating probability calibration quality.

Provides:
  - compute_ece                 — Expected Calibration Error
  - compute_reliability_data    — Reliability diagram data
  - compute_confidence_accuracy — Per-confidence-bin accuracy
"""

from __future__ import annotations

import torch


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and
    actual accuracy, weighted by the fraction of samples in each bin.

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Args:
        probs: Softmax probabilities of shape (N, num_classes).
        labels: Integer class labels of shape (N,).
        n_bins: Number of confidence bins. Default: 15.

    Returns:
        ECE as a float ∈ [0, 1]. Lower is better.
    """
    confidences, predictions = probs.max(dim=-1)
    correct = predictions.eq(labels).float()

    ece = 0.0
    n = len(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc = correct[mask].mean().item()
        bin_weight = mask.float().sum().item() / n
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_data(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict[str, list[float]]:
    """Compute reliability diagram data (confidence vs accuracy per bin).

    Args:
        probs: Softmax probabilities of shape (N, num_classes).
        labels: Integer class labels of shape (N,).
        n_bins: Number of confidence bins.

    Returns:
        Dict with keys:
            bin_midpoints — centre of each confidence bin
            accuracies    — mean accuracy per bin (None → empty bin)
            confidences   — mean predicted confidence per bin
            fractions     — fraction of samples in each bin
    """
    confidences, predictions = probs.max(dim=-1)
    correct = predictions.eq(labels).float()
    n = len(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    midpoints, accuracies, conf_vals, fractions = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i].item(), bin_boundaries[i + 1].item()
        midpoints.append((lo + hi) / 2.0)
        mask = (confidences > lo) & (confidences <= hi)
        frac = mask.float().sum().item() / n
        fractions.append(frac)
        if mask.sum() == 0:
            accuracies.append(0.0)
            conf_vals.append(float((lo + hi) / 2))
        else:
            accuracies.append(correct[mask].mean().item())
            conf_vals.append(confidences[mask].mean().item())

    return {
        "bin_midpoints": midpoints,
        "accuracies": accuracies,
        "confidences": conf_vals,
        "fractions": fractions,
    }


def compute_confidence_accuracy(
    probs: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.7,
) -> dict[str, float]:
    """Compute accuracy above and below a confidence threshold.

    Args:
        probs: Softmax probabilities of shape (N, num_classes).
        labels: Integer class labels of shape (N,).
        threshold: Confidence threshold.

    Returns:
        Dict with:
            accuracy_above — accuracy for samples above threshold
            accuracy_below — accuracy for samples at or below threshold
            coverage       — fraction of samples above threshold
            ece            — overall ECE
    """
    confidences, predictions = probs.max(dim=-1)
    correct = predictions.eq(labels).float()
    above = confidences > threshold

    acc_above = correct[above].mean().item() if above.any() else float("nan")
    acc_below = correct[~above].mean().item() if (~above).any() else float("nan")
    coverage = above.float().mean().item()
    ece = compute_ece(probs, labels)

    return {
        "accuracy_above": acc_above,
        "accuracy_below": acc_below,
        "coverage": coverage,
        "ece": ece,
    }
