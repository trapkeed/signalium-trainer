"""Phase 17 — AEIP Expert and Ensemble Calibration.

Calibrates every expert individually and calibrates ensemble outputs.

Measures:
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Reliability (average confidence vs accuracy per bin)
  - Confidence Accuracy (fraction where confidence matches accuracy)
  - Overconfidence / Underconfidence score

Supports:
  - Per-expert calibration
  - Ensemble-level calibration
  - Temperature scaling (pure Python, no PyTorch required in calibration layer)

Usage::

    cal = ExpertCalibrator(n_bins=10)
    cal.update("expert_01", predicted_prob=0.85, true_label=1)
    report = cal.get_report("expert_01")
    ece = report.ece
    temperature = cal.fit_temperature("expert_01")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Calibration metrics for one expert or the ensemble.

    Attributes:
        entity_id: Expert ID or "ensemble".
        ece: Expected Calibration Error (lower = better).
        mce: Maximum Calibration Error.
        reliability: List of (confidence_mid, accuracy) per bin.
        avg_confidence: Mean predicted confidence.
        avg_accuracy: Mean empirical accuracy.
        overconfidence: avg_confidence - avg_accuracy (positive = overconfident).
        n_samples: Number of samples used.
        temperature: Fitted temperature scaling factor (1.0 = no scaling).
        n_bins: Number of calibration bins.
    """

    entity_id: str
    ece: float = 0.0
    mce: float = 0.0
    reliability: List[Tuple[float, float, int]] = field(default_factory=list)
    avg_confidence: float = 0.0
    avg_accuracy: float = 0.0
    overconfidence: float = 0.0
    n_samples: int = 0
    temperature: float = 1.0
    n_bins: int = 10

    @property
    def is_well_calibrated(self) -> bool:
        return self.ece < 0.05

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "ece": self.ece,
            "mce": self.mce,
            "reliability": [
                {"bin_mid": b, "accuracy": a, "n": n}
                for b, a, n in self.reliability
            ],
            "avg_confidence": self.avg_confidence,
            "avg_accuracy": self.avg_accuracy,
            "overconfidence": self.overconfidence,
            "n_samples": self.n_samples,
            "temperature": self.temperature,
            "n_bins": self.n_bins,
            "is_well_calibrated": self.is_well_calibrated,
        }


# ---------------------------------------------------------------------------
# Expert calibrator
# ---------------------------------------------------------------------------


class ExpertCalibrator:
    """Tracks calibration data per expert and computes calibration metrics.

    Args:
        n_bins: Number of equal-width confidence bins (default 10).
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._data: Dict[str, List[Tuple[float, int]]] = {}
        self._temperatures: Dict[str, float] = {}

    def update(self, entity_id: str, predicted_prob: float,
               true_label: int) -> None:
        """Record one calibration sample.

        Args:
            entity_id: Expert ID or "ensemble".
            predicted_prob: Predicted probability for the positive class.
            true_label: Ground truth label (1 = correct, 0 = incorrect).
        """
        if entity_id not in self._data:
            self._data[entity_id] = []
        self._data[entity_id].append((
            max(0.0, min(1.0, float(predicted_prob))),
            int(bool(true_label))
        ))

    def update_batch(self, entity_id: str,
                     samples: List[Tuple[float, int]]) -> None:
        """Record multiple calibration samples at once."""
        for prob, label in samples:
            self.update(entity_id, prob, label)

    def get_report(self, entity_id: str) -> CalibrationReport:
        """Compute calibration metrics for an entity."""
        data = self._data.get(entity_id, [])
        if not data:
            return CalibrationReport(entity_id=entity_id)

        n = len(data)
        bins: List[List[Tuple[float, int]]] = [[] for _ in range(self.n_bins)]

        for prob, label in data:
            idx = min(int(prob * self.n_bins), self.n_bins - 1)
            bins[idx].append((prob, label))

        ece = 0.0
        mce = 0.0
        reliability = []

        for i, bin_data in enumerate(bins):
            if not bin_data:
                continue
            bin_probs = [p for p, _ in bin_data]
            bin_labels = [l for _, l in bin_data]
            bin_n = len(bin_data)
            avg_conf = sum(bin_probs) / bin_n
            avg_acc = sum(bin_labels) / bin_n
            bin_mid = (i + 0.5) / self.n_bins
            cal_err = abs(avg_conf - avg_acc)
            ece += cal_err * (bin_n / n)
            mce = max(mce, cal_err)
            reliability.append((bin_mid, avg_acc, bin_n))

        all_probs = [p for p, _ in data]
        all_labels = [l for _, l in data]
        avg_confidence = sum(all_probs) / n
        avg_accuracy = sum(all_labels) / n
        overconfidence = avg_confidence - avg_accuracy
        temperature = self._temperatures.get(entity_id, 1.0)

        return CalibrationReport(
            entity_id=entity_id,
            ece=round(ece, 6),
            mce=round(mce, 6),
            reliability=reliability,
            avg_confidence=round(avg_confidence, 6),
            avg_accuracy=round(avg_accuracy, 6),
            overconfidence=round(overconfidence, 6),
            n_samples=n,
            temperature=temperature,
            n_bins=self.n_bins,
        )

    def fit_temperature(self, entity_id: str,
                        n_steps: int = 50,
                        lr: float = 0.1) -> float:
        """Fit temperature scaling via gradient descent on cross-entropy.

        Returns the optimal temperature T such that
        calibrated_prob = sigmoid(logit(prob) / T) reduces ECE.
        """
        data = self._data.get(entity_id, [])
        if not data:
            return 1.0

        T = 1.0
        for _ in range(n_steps):
            grad = 0.0
            for prob, label in data:
                prob = max(1e-7, min(1 - 1e-7, prob))
                logit = math.log(prob / (1 - prob))
                scaled_logit = logit / T
                scaled_prob = 1 / (1 + math.exp(-scaled_logit))
                error = scaled_prob - label
                d_prob_d_T = scaled_prob * (1 - scaled_prob) * (-logit / (T * T))
                grad += error * d_prob_d_T
            T -= lr * grad / len(data)
            T = max(0.1, min(5.0, T))

        self._temperatures[entity_id] = round(T, 4)
        return self._temperatures[entity_id]

    def apply_temperature(self, entity_id: str, prob: float) -> float:
        """Apply temperature scaling to a probability."""
        T = self._temperatures.get(entity_id, 1.0)
        if T == 1.0 or prob <= 0 or prob >= 1:
            return prob
        logit = math.log(prob / (1 - prob))
        scaled = logit / T
        return 1 / (1 + math.exp(-scaled))

    def all_reports(self) -> Dict[str, CalibrationReport]:
        """Return calibration reports for all tracked entities."""
        return {eid: self.get_report(eid) for eid in self._data}

    def entity_ids(self) -> List[str]:
        return list(self._data.keys())

    def n_samples(self, entity_id: str) -> int:
        return len(self._data.get(entity_id, []))

    def clear(self, entity_id: str) -> None:
        self._data.pop(entity_id, None)
        self._temperatures.pop(entity_id, None)
