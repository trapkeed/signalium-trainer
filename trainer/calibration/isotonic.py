"""Isotonic Regression — non-parametric post-hoc calibration.

Isotonic regression is applied independently to each class's predicted
probability to produce calibrated probabilities.  It is the most flexible
calibration method but requires more calibration data than temperature or
Platt scaling.

This implementation wraps sklearn's IsotonicRegression for simplicity,
applied in a one-vs-rest fashion for each class.

Reference:
    Zadrozny & Elkan (2002) "Transforming Classifier Scores into Accurate
    Multiclass Probability Estimates"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class IsotonicCalibration:
    """Non-parametric isotonic regression calibrator.

    Fits a monotonic mapping from predicted probabilities to empirical
    probabilities for each class independently (one-vs-rest).

    Requires scikit-learn to be installed.

    Usage::

        calibrator = IsotonicCalibration(num_classes=3)
        calibrator.fit(val_probs, val_labels)
        calibrated_probs = calibrator.predict(model_probs)

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int) -> None:
        self._num_classes = num_classes
        self._calibrators: Optional[list] = None

    def fit(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> "IsotonicCalibration":
        """Fit isotonic regression calibrators for each class.

        Args:
            probs: Softmax probabilities of shape (N, num_classes).
            labels: Integer class labels of shape (N,).

        Returns:
            Self (fitted calibrator).
        """
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as exc:
            raise ImportError(
                "IsotonicCalibration requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            ) from exc

        probs_np = probs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        self._calibrators = []
        for c in range(self._num_classes):
            binary_labels = (labels_np == c).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs_np[:, c], binary_labels)
            self._calibrators.append(ir)

        return self

    def predict(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply calibration and return normalised probabilities.

        Args:
            probs: Softmax probabilities of shape (N, num_classes).

        Returns:
            Calibrated, row-normalised probabilities of shape (N, num_classes).
        """
        if self._calibrators is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        probs_np = probs.detach().cpu().numpy()
        calibrated = np.stack(
            [self._calibrators[c].predict(probs_np[:, c]) for c in range(self._num_classes)],
            axis=1,
        ).astype(np.float32)

        # Row-normalise to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True).clip(min=1e-9)
        calibrated = calibrated / row_sums

        return torch.from_numpy(calibrated).to(probs.device)

    @property
    def is_fitted(self) -> bool:
        return self._calibrators is not None
