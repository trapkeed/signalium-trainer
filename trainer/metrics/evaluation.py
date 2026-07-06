"""Evaluation metrics computation using TorchMetrics and scikit-learn."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchmetrics
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)


class EvaluationMetrics:
    """Computes a full suite of classification evaluation metrics.

    Supported metrics:
    - Accuracy, Precision, Recall, F1 (macro/weighted)
    - ROC-AUC (binary and multiclass OvR)
    - Confusion matrix
    - Loss curves (passed in from training history)
    - Prediction distribution
    - Calibration (expected calibration error)

    Args:
        num_classes: Number of target classes.
        task: "binary" or "multiclass".
        device: Torch device for TorchMetrics.
    """

    def __init__(
        self,
        num_classes: int,
        task: str = "multiclass",
        device: str = "cpu",
    ) -> None:
        self._num_classes = num_classes
        self._task = task
        self._device = torch.device(device)

        self._accuracy = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self._device)
        self._precision = torchmetrics.Precision(
            task=task, num_classes=num_classes, average="macro"
        ).to(self._device)
        self._recall = torchmetrics.Recall(task=task, num_classes=num_classes, average="macro").to(
            self._device
        )
        self._f1 = torchmetrics.F1Score(task=task, num_classes=num_classes, average="macro").to(
            self._device
        )

    def compute(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute all metrics.

        Args:
            preds: Predicted class indices, shape (N,).
            targets: Ground truth class indices, shape (N,).
            probs: Class probabilities, shape (N, C) — required for AUC.

        Returns:
            Dict of metric name → value.
        """
        preds = preds.to(self._device)
        targets = targets.to(self._device)

        results: dict[str, Any] = {
            "accuracy": self._accuracy(preds, targets).item(),
            "precision_macro": self._precision(preds, targets).item(),
            "recall_macro": self._recall(preds, targets).item(),
            "f1_macro": self._f1(preds, targets).item(),
        }

        # sklearn metrics on CPU numpy arrays
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        results["classification_report"] = classification_report(
            targets_np, preds_np, zero_division=0
        )

        # ROC-AUC
        if probs is not None:
            probs_np = probs.cpu().numpy()
            try:
                if self._task == "binary":
                    results["roc_auc"] = float(roc_auc_score(targets_np, probs_np[:, 1]))
                else:
                    results["roc_auc"] = float(
                        roc_auc_score(targets_np, probs_np, multi_class="ovr", average="macro")
                    )
            except ValueError as exc:
                logger.warning("Could not compute ROC-AUC: {}", exc)

        # Confusion matrix
        results["confusion_matrix"] = self._confusion_matrix(targets_np, preds_np)

        # Prediction distribution
        results["prediction_distribution"] = {
            int(cls): int(count) for cls, count in zip(*np.unique(preds_np, return_counts=True))
        }

        # Calibration — expected calibration error (ECE)
        if probs is not None:
            results["ece"] = self._expected_calibration_error(
                probs.cpu().numpy(), targets_np, n_bins=10
            )

        self._reset()
        return results

    def _confusion_matrix(self, targets: np.ndarray, preds: np.ndarray) -> np.ndarray:
        from sklearn.metrics import confusion_matrix

        return confusion_matrix(targets, preds)

    @staticmethod
    def _expected_calibration_error(
        probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Args:
            probs: Class probabilities (N, C).
            targets: Ground truth (N,).
            n_bins: Number of confidence bins.

        Returns:
            ECE as a float in [0, 1].
        """
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = predictions == targets

        ece = 0.0
        for m in np.linspace(0, 1, n_bins + 1)[:-1]:
            m_next = m + 1.0 / n_bins
            in_bin = (confidences > m) & (confidences <= m_next)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return float(ece)

    def _reset(self) -> None:
        self._accuracy.reset()
        self._precision.reset()
        self._recall.reset()
        self._f1.reset()
