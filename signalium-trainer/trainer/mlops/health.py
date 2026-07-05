"""Phase 18 — Model Health and Drift Detection.

Tracks model health metrics including:

  - Confidence drift
  - Prediction drift
  - Feature drift
  - Label drift
  - Calibration drift
  - Market regime drift
  - Distribution drift
  - Performance degradation
  - Latency degradation

All drift metrics use rolling window comparison (recent vs. baseline).
No automated actions are taken — only measurements and alerts.

Usage::

    health = ModelHealthMonitor()
    health.record_prediction(model_id="m1", buy_prob=0.6, sell_prob=0.2,
                              no_trade_prob=0.2, confidence=0.78,
                              true_class="buy")
    report = health.get_health_report("m1")
    print(report.confidence_drift, report.prediction_drift)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Health record
# ---------------------------------------------------------------------------


@dataclass
class HealthRecord:
    """One health observation for a model."""
    model_id: str
    buy_prob: float = 0.33
    sell_prob: float = 0.33
    no_trade_prob: float = 0.34
    confidence: float = 0.5
    uncertainty: float = 0.5
    predicted_class: str = "no_trade"
    true_class: Optional[str] = None
    features_hash: Optional[str] = None   # hash of feature vector (for drift)
    latency_ms: float = 0.0
    calibration_error: float = 0.0
    regime: str = "unknown"
    is_correct: Optional[bool] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.true_class is not None:
            self.is_correct = (self.predicted_class == self.true_class)


# ---------------------------------------------------------------------------
# Health report
# ---------------------------------------------------------------------------


@dataclass
class HealthReport:
    """Aggregated model health report.

    Attributes:
        model_id: Model these metrics belong to.
        n_records: Records in rolling window.
        confidence_drift: |current_avg_conf - baseline_avg_conf|.
        prediction_drift: KL-divergence-proxy on class distribution.
        calibration_drift: Change in calibration error from baseline.
        performance_degradation: Accuracy drop from baseline.
        latency_degradation: Latency increase from baseline.
        regime_drift: Regime distribution change.
        distribution_drift: Change in buy/sell/no_trade ratios.
        is_healthy: True if all drifts below thresholds.
        drift_alerts: List of drift type strings exceeding threshold.
        baseline_window: Window size for baseline computation.
        recent_window: Window size for recent computation.
    """

    model_id: str
    n_records: int = 0
    confidence_drift: float = 0.0
    prediction_drift: float = 0.0
    calibration_drift: float = 0.0
    performance_degradation: float = 0.0
    latency_degradation: float = 0.0
    regime_drift: float = 0.0
    distribution_drift: float = 0.0
    is_healthy: bool = True
    drift_alerts: List[str] = field(default_factory=list)
    baseline_avg_confidence: float = 0.0
    recent_avg_confidence: float = 0.0
    baseline_accuracy: float = 0.0
    recent_accuracy: float = 0.0
    baseline_avg_latency: float = 0.0
    recent_avg_latency: float = 0.0
    baseline_distribution: Dict[str, float] = field(default_factory=dict)
    recent_distribution: Dict[str, float] = field(default_factory=dict)
    baseline_window: int = 200
    recent_window: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "n_records": self.n_records,
            "confidence_drift": self.confidence_drift,
            "prediction_drift": self.prediction_drift,
            "calibration_drift": self.calibration_drift,
            "performance_degradation": self.performance_degradation,
            "latency_degradation": self.latency_degradation,
            "regime_drift": self.regime_drift,
            "distribution_drift": self.distribution_drift,
            "is_healthy": self.is_healthy,
            "drift_alerts": self.drift_alerts,
            "baseline_avg_confidence": self.baseline_avg_confidence,
            "recent_avg_confidence": self.recent_avg_confidence,
            "baseline_accuracy": self.baseline_accuracy,
            "recent_accuracy": self.recent_accuracy,
            "baseline_avg_latency": self.baseline_avg_latency,
            "recent_avg_latency": self.recent_avg_latency,
            "baseline_distribution": self.baseline_distribution,
            "recent_distribution": self.recent_distribution,
            "baseline_window": self.baseline_window,
            "recent_window": self.recent_window,
        }


# ---------------------------------------------------------------------------
# Drift thresholds
# ---------------------------------------------------------------------------


@dataclass
class DriftThresholds:
    """Configurable thresholds for drift detection."""
    confidence_drift: float = 0.10
    prediction_drift: float = 0.15
    calibration_drift: float = 0.05
    performance_degradation: float = 0.05
    latency_degradation_ms: float = 20.0
    distribution_drift: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_drift": self.confidence_drift,
            "prediction_drift": self.prediction_drift,
            "calibration_drift": self.calibration_drift,
            "performance_degradation": self.performance_degradation,
            "latency_degradation_ms": self.latency_degradation_ms,
            "distribution_drift": self.distribution_drift,
        }


# ---------------------------------------------------------------------------
# Model health monitor
# ---------------------------------------------------------------------------


class ModelHealthMonitor:
    """Tracks model health using rolling windows and drift detection.

    Args:
        baseline_window: Size of the baseline window for comparison.
        recent_window: Size of the recent window for comparison.
        thresholds: DriftThresholds for alert generation.
    """

    def __init__(self, baseline_window: int = 200,
                 recent_window: int = 50,
                 thresholds: Optional[DriftThresholds] = None) -> None:
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.thresholds = thresholds or DriftThresholds()
        self._buffers: Dict[str, deque] = {}
        self._all_records: Dict[str, List[HealthRecord]] = {}

    def record_prediction(self, model_id: str,
                           buy_prob: float = 0.33,
                           sell_prob: float = 0.33,
                           no_trade_prob: float = 0.34,
                           confidence: float = 0.5,
                           uncertainty: float = 0.5,
                           predicted_class: str = "no_trade",
                           true_class: Optional[str] = None,
                           latency_ms: float = 0.0,
                           calibration_error: float = 0.0,
                           regime: str = "unknown") -> HealthRecord:
        """Record one prediction for health tracking."""
        rec = HealthRecord(
            model_id=model_id,
            buy_prob=buy_prob, sell_prob=sell_prob, no_trade_prob=no_trade_prob,
            confidence=confidence, uncertainty=uncertainty,
            predicted_class=predicted_class, true_class=true_class,
            latency_ms=latency_ms, calibration_error=calibration_error,
            regime=regime,
        )
        total_window = self.baseline_window + self.recent_window
        if model_id not in self._buffers:
            self._buffers[model_id] = deque(maxlen=total_window)
            self._all_records[model_id] = []
        self._buffers[model_id].append(rec)
        self._all_records[model_id].append(rec)
        return rec

    def get_health_report(self, model_id: str) -> HealthReport:
        """Compute drift report for a model."""
        buf = list(self._buffers.get(model_id, []))
        n = len(buf)
        report = HealthReport(
            model_id=model_id, n_records=n,
            baseline_window=self.baseline_window,
            recent_window=self.recent_window,
        )
        if n < self.recent_window:
            return report

        baseline = buf[:self.baseline_window]
        recent = buf[-self.recent_window:]

        b_conf = _mean([r.confidence for r in baseline])
        r_conf = _mean([r.confidence for r in recent])
        report.baseline_avg_confidence = round(b_conf, 4)
        report.recent_avg_confidence = round(r_conf, 4)
        report.confidence_drift = round(abs(r_conf - b_conf), 4)

        b_dist = _class_distribution(baseline)
        r_dist = _class_distribution(recent)
        report.baseline_distribution = b_dist
        report.recent_distribution = r_dist
        report.distribution_drift = round(_distribution_drift(b_dist, r_dist), 4)
        report.prediction_drift = report.distribution_drift

        b_labeled = [r for r in baseline if r.is_correct is not None]
        r_labeled = [r for r in recent if r.is_correct is not None]
        b_acc = _mean([int(r.is_correct) for r in b_labeled]) if b_labeled else 0.0
        r_acc = _mean([int(r.is_correct) for r in r_labeled]) if r_labeled else 0.0
        report.baseline_accuracy = round(b_acc, 4)
        report.recent_accuracy = round(r_acc, 4)
        report.performance_degradation = round(max(0.0, b_acc - r_acc), 4)

        b_lat = _mean([r.latency_ms for r in baseline])
        r_lat = _mean([r.latency_ms for r in recent])
        report.baseline_avg_latency = round(b_lat, 3)
        report.recent_avg_latency = round(r_lat, 3)
        report.latency_degradation = round(max(0.0, r_lat - b_lat), 3)

        b_cal = _mean([r.calibration_error for r in baseline])
        r_cal = _mean([r.calibration_error for r in recent])
        report.calibration_drift = round(abs(r_cal - b_cal), 4)

        b_regimes = _regime_distribution(baseline)
        r_regimes = _regime_distribution(recent)
        report.regime_drift = round(_distribution_drift(b_regimes, r_regimes), 4)

        alerts = []
        thr = self.thresholds
        if report.confidence_drift > thr.confidence_drift:
            alerts.append(f"confidence_drift={report.confidence_drift:.3f}")
        if report.prediction_drift > thr.prediction_drift:
            alerts.append(f"prediction_drift={report.prediction_drift:.3f}")
        if report.calibration_drift > thr.calibration_drift:
            alerts.append(f"calibration_drift={report.calibration_drift:.3f}")
        if report.performance_degradation > thr.performance_degradation:
            alerts.append(f"performance_degradation={report.performance_degradation:.3f}")
        if report.latency_degradation > thr.latency_degradation_ms:
            alerts.append(f"latency_degradation={report.latency_degradation:.1f}ms")
        if report.distribution_drift > thr.distribution_drift:
            alerts.append(f"distribution_drift={report.distribution_drift:.3f}")

        report.drift_alerts = alerts
        report.is_healthy = len(alerts) == 0
        return report

    def all_health_reports(self) -> Dict[str, HealthReport]:
        return {mid: self.get_health_report(mid) for mid in self._buffers}

    def unhealthy_models(self) -> List[str]:
        return [mid for mid, r in self.all_health_reports().items()
                if not r.is_healthy]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _class_distribution(records: List[HealthRecord]) -> Dict[str, float]:
    if not records:
        return {"buy": 0.33, "sell": 0.33, "no_trade": 0.34}
    n = len(records)
    return {
        "buy": sum(1 for r in records if r.predicted_class == "buy") / n,
        "sell": sum(1 for r in records if r.predicted_class == "sell") / n,
        "no_trade": sum(1 for r in records if r.predicted_class == "no_trade") / n,
    }


def _regime_distribution(records: List[HealthRecord]) -> Dict[str, float]:
    if not records:
        return {}
    n = len(records)
    regimes: Dict[str, int] = {}
    for r in records:
        regimes[r.regime] = regimes.get(r.regime, 0) + 1
    return {k: v / n for k, v in regimes.items()}


def _distribution_drift(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Symmetric KL-divergence proxy over shared keys."""
    keys = set(a) | set(b)
    eps = 1e-9
    drift = 0.0
    for k in keys:
        p = a.get(k, eps)
        q = b.get(k, eps)
        p = max(p, eps)
        q = max(q, eps)
        drift += abs(p - q)
    return drift / max(len(keys), 1)
