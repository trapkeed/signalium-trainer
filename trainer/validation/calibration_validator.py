"""Phase 15 — Calibration Validation.

Evaluates model probability calibration using:
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Reliability diagram data
  - Confidence–accuracy curves
  - Temperature scaling recommendation
  - Brier score (already in StatisticalMetrics, surfaced here per-class)

Leverages the existing trainer/calibration/ infrastructure.

Usage::

    validator = CalibrationValidator(n_bins=15)
    report = validator.evaluate(probs, labels)
    print(report.summary())
    data = report.reliability_chart_data()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Comprehensive calibration evaluation report.

    Attributes:
        ece: Expected Calibration Error (lower is better).
        mce: Maximum Calibration Error.
        brier_score: Multi-class Brier score.
        overconfidence_fraction: Fraction of samples where confidence > accuracy.
        underconfidence_fraction: Fraction where confidence < accuracy.
        mean_confidence: Mean max-class probability.
        mean_accuracy: Mean per-sample correctness.
        confidence_gap: mean_confidence - mean_accuracy (positive = overconfident).
        recommended_temperature: Estimated optimal temperature scaling T.
        bin_confidences: Per-bin mean confidence.
        bin_accuracies: Per-bin mean accuracy.
        bin_fractions: Per-bin sample fraction.
        bin_midpoints: Bin centre values.
        per_class_ece: ECE computed for each class independently.
    """

    ece: float = 0.0
    mce: float = 0.0
    brier_score: float = 0.0
    overconfidence_fraction: float = 0.0
    underconfidence_fraction: float = 0.0
    mean_confidence: float = 0.0
    mean_accuracy: float = 0.0
    confidence_gap: float = 0.0
    recommended_temperature: float = 1.0
    bin_confidences: list[float] = field(default_factory=list)
    bin_accuracies: list[float] = field(default_factory=list)
    bin_fractions: list[float] = field(default_factory=list)
    bin_midpoints: list[float] = field(default_factory=list)
    per_class_ece: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        direction = "overconfident" if self.confidence_gap > 0 else "underconfident"
        lines = [
            "=== Calibration Report ===",
            f"  ECE                  : {self.ece:.4f}",
            f"  MCE                  : {self.mce:.4f}",
            f"  Brier Score          : {self.brier_score:.4f}",
            f"  Mean Confidence      : {self.mean_confidence:.3f}",
            f"  Mean Accuracy        : {self.mean_accuracy:.3f}",
            f"  Confidence Gap       : {self.confidence_gap:+.3f} ({direction})",
            f"  Recommended T (temp) : {self.recommended_temperature:.2f}",
        ]
        if self.per_class_ece:
            lines.append("  Per-class ECE:")
            for cls, ece in sorted(self.per_class_ece.items()):
                lines.append(f"    Class {cls}: {ece:.4f}")
        return "\n".join(lines)

    def reliability_chart_data(self) -> dict[str, Any]:
        """JSON-safe reliability diagram data for plotting."""
        return {
            "bin_midpoints": self.bin_midpoints,
            "bin_accuracies": self.bin_accuracies,
            "bin_confidences": self.bin_confidences,
            "bin_fractions": self.bin_fractions,
            "perfect_calibration": self.bin_midpoints,  # diagonal reference
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier_score": round(self.brier_score, 4),
            "overconfidence_fraction": round(self.overconfidence_fraction, 4),
            "underconfidence_fraction": round(self.underconfidence_fraction, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_accuracy": round(self.mean_accuracy, 4),
            "confidence_gap": round(self.confidence_gap, 4),
            "recommended_temperature": round(self.recommended_temperature, 3),
            "per_class_ece": {str(k): round(v, 4) for k, v in self.per_class_ece.items()},
            "reliability_chart": self.reliability_chart_data(),
        }


# ---------------------------------------------------------------------------
# Calibration Validator
# ---------------------------------------------------------------------------


class CalibrationValidator:
    """Evaluates model probability calibration.

    Args:
        n_bins: Number of confidence bins for ECE/MCE.
        n_classes: Number of prediction classes.

    Usage::

        validator = CalibrationValidator(n_bins=15, n_classes=3)
        probs = torch.softmax(logits, dim=-1)
        report = validator.evaluate(probs, labels)
    """

    def __init__(self, n_bins: int = 15, n_classes: int = 3) -> None:
        self._bins = n_bins
        self._C = n_classes

    def evaluate(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> CalibrationReport:
        """Compute all calibration metrics.

        Args:
            probs: ``(N, n_classes)`` softmax probabilities.
            labels: ``(N,)`` integer class labels.

        Returns:
            ``CalibrationReport`` with all metrics.
        """
        p = probs.cpu().float().numpy()
        y = labels.cpu().long().numpy()
        n = len(y)
        C = self._C

        conf = p.max(axis=1)
        preds = p.argmax(axis=1)
        correct = (preds == y).astype(float)

        mean_conf = float(conf.mean())
        mean_acc = float(correct.mean())
        confidence_gap = mean_conf - mean_acc

        # Reliability bins
        bin_edges = np.linspace(0, 1, self._bins + 1)
        bin_midpoints, bin_accs, bin_confs, bin_fracs = [], [], [], []
        ece = 0.0
        mce = 0.0

        for i in range(self._bins):
            mask = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])
            mid = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            bin_midpoints.append(float(mid))
            if mask.sum() == 0:
                bin_accs.append(0.0)
                bin_confs.append(float(mid))
                bin_fracs.append(0.0)
                continue
            ba = float(correct[mask].mean())
            bc = float(conf[mask].mean())
            bf = float(mask.sum() / n)
            bin_accs.append(ba)
            bin_confs.append(bc)
            bin_fracs.append(bf)
            gap = abs(ba - bc)
            ece += bf * gap
            mce = max(mce, gap)

        # Brier score
        y_oh = np.eye(C)[y]
        brier = float(np.mean(np.sum((p - y_oh) ** 2, axis=1)))

        # Overconfidence / underconfidence fractions
        over = float((conf > correct).mean())
        under = float((conf < correct).mean())

        # Recommended temperature (binary search)
        recommended_t = self._estimate_temperature(p, y)

        # Per-class ECE
        per_class_ece: dict[int, float] = {}
        for c in range(C):
            y_bin = (y == c).astype(float)
            p_c = p[:, c]
            per_class_ece[c] = self._ece_binary(p_c, y_bin)

        return CalibrationReport(
            ece=float(ece),
            mce=float(mce),
            brier_score=brier,
            overconfidence_fraction=over,
            underconfidence_fraction=under,
            mean_confidence=mean_conf,
            mean_accuracy=mean_acc,
            confidence_gap=confidence_gap,
            recommended_temperature=recommended_t,
            bin_confidences=bin_confs,
            bin_accuracies=bin_accs,
            bin_fractions=bin_fracs,
            bin_midpoints=bin_midpoints,
            per_class_ece=per_class_ece,
        )

    def _ece_binary(self, confs: np.ndarray, correct: np.ndarray) -> float:
        bins = np.linspace(0, 1, self._bins + 1)
        n = len(confs)
        ece = 0.0
        for i in range(self._bins):
            mask = (confs >= bins[i]) & (confs < bins[i + 1])
            if mask.sum() == 0:
                continue
            ba = correct[mask].mean()
            bc = confs[mask].mean()
            ece += (mask.sum() / n) * abs(ba - bc)
        return float(ece)

    def _estimate_temperature(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Estimate optimal temperature using a grid search over NLL."""
        eps = 1e-9
        n = len(labels)
        y_oh = np.eye(self._C)[labels]
        best_t = 1.0
        best_nll = float("inf")
        for t in np.arange(0.5, 3.0, 0.1):
            # Soft temperature scaling on probs (approximate)
            log_probs = np.log(probs + eps)
            scaled = log_probs / t
            # Numerically stable softmax
            scaled -= scaled.max(axis=1, keepdims=True)
            exp_s = np.exp(scaled)
            p_t = exp_s / exp_s.sum(axis=1, keepdims=True)
            nll = -float(np.mean(np.sum(y_oh * np.log(p_t + eps), axis=1)))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        return round(best_t, 2)
