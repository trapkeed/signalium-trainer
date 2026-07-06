"""Phase 19 — Model Health Evolution Tracker.

Tracks long-term model health trends:

  - Performance improvement / degradation
  - Confidence changes over time
  - Calibration drift history
  - Prediction entropy trends
  - False positive / negative rates
  - Long-term signal quality evolution

Advisory only — no automated actions.

Usage::

    mhe = ModelHealthEvolutionTracker()
    mhe.record_health_point("model_01",
        accuracy=0.68, confidence=0.75, calibration_error=0.04,
        false_positive_rate=0.12, false_negative_rate=0.15)
    trend = mhe.get_trend("model_01")
    report = mhe.evolution_report("model_01")
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

# ---------------------------------------------------------------------------
# Health point
# ---------------------------------------------------------------------------


@dataclass
class HealthPoint:
    """One health measurement for a model at a point in time."""

    point_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    calibration_error: float = 0.0
    prediction_entropy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    signal_rate: float = 0.0
    latency_ms: float = 0.0
    n_predictions: int = 0
    regime: str = ""
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "model_id": self.model_id,
            "accuracy": self.accuracy,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "calibration_error": self.calibration_error,
            "prediction_entropy": self.prediction_entropy,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "signal_rate": self.signal_rate,
            "latency_ms": self.latency_ms,
            "n_predictions": self.n_predictions,
            "regime": self.regime,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Health trend
# ---------------------------------------------------------------------------


@dataclass
class HealthTrend:
    """Computed long-term health trend for a model."""

    model_id: str = ""
    n_points: int = 0
    accuracy_trend: str = "stable"
    confidence_trend: str = "stable"
    calibration_trend: str = "stable"
    entropy_trend: str = "stable"
    fpr_trend: str = "stable"
    fnr_trend: str = "stable"
    avg_accuracy: float = 0.0
    avg_confidence: float = 0.5
    avg_calibration_error: float = 0.0
    avg_entropy: float = 0.0
    avg_fpr: float = 0.0
    avg_fnr: float = 0.0
    first_accuracy: float = 0.0
    latest_accuracy: float = 0.0
    accuracy_delta: float = 0.0
    overall_health: str = "stable"  # improving / stable / degrading
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "n_points": self.n_points,
            "accuracy_trend": self.accuracy_trend,
            "confidence_trend": self.confidence_trend,
            "calibration_trend": self.calibration_trend,
            "entropy_trend": self.entropy_trend,
            "fpr_trend": self.fpr_trend,
            "fnr_trend": self.fnr_trend,
            "avg_accuracy": self.avg_accuracy,
            "avg_confidence": self.avg_confidence,
            "avg_calibration_error": self.avg_calibration_error,
            "avg_entropy": self.avg_entropy,
            "avg_fpr": self.avg_fpr,
            "avg_fnr": self.avg_fnr,
            "first_accuracy": self.first_accuracy,
            "latest_accuracy": self.latest_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "overall_health": self.overall_health,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Model health evolution tracker
# ---------------------------------------------------------------------------


class ModelHealthEvolutionTracker:
    """Tracks long-term model health evolution over time.

    Args:
        window: Rolling window for short-term trend computations.
        trend_improvement_threshold: Metric improvement to classify as "improving".
        trend_degradation_threshold: Metric drop to classify as "degrading".
    """

    def __init__(
        self,
        window: int = 50,
        trend_improvement_threshold: float = 0.03,
        trend_degradation_threshold: float = 0.03,
    ) -> None:
        self.window = window
        self.improve_thr = trend_improvement_threshold
        self.degrade_thr = trend_degradation_threshold
        self._buffers: Dict[str, Deque[HealthPoint]] = {}
        self._all_points: Dict[str, List[HealthPoint]] = {}

    def record_health_point(
        self,
        model_id: str,
        accuracy: float = 0.0,
        sharpe_ratio: float = 0.0,
        profit_factor: float = 0.0,
        win_rate: float = 0.0,
        confidence: float = 0.5,
        uncertainty: float = 0.5,
        calibration_error: float = 0.0,
        prediction_entropy: float = 0.0,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        signal_rate: float = 0.0,
        latency_ms: float = 0.0,
        n_predictions: int = 0,
        regime: str = "",
        notes: str = "",
    ) -> HealthPoint:
        """Record one health measurement for a model."""
        pt = HealthPoint(
            model_id=model_id,
            accuracy=accuracy,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            confidence=confidence,
            uncertainty=uncertainty,
            calibration_error=calibration_error,
            prediction_entropy=prediction_entropy,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            signal_rate=signal_rate,
            latency_ms=latency_ms,
            n_predictions=n_predictions,
            regime=regime,
            notes=notes,
        )
        if model_id not in self._buffers:
            self._buffers[model_id] = deque(maxlen=self.window)
            self._all_points[model_id] = []
        self._buffers[model_id].append(pt)
        self._all_points[model_id].append(pt)
        return pt

    def get_trend(self, model_id: str) -> HealthTrend:
        """Compute health trend for a model."""
        pts = list(self._all_points.get(model_id, []))
        n = len(pts)
        if n == 0:
            return HealthTrend(model_id=model_id)

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        def _classify_trend(series, thr):
            if len(series) < 4:
                return "stable"
            half = len(series) // 2
            first_half = _mean(series[:half])
            second_half = _mean(series[half:])
            delta = second_half - first_half
            if delta > thr:
                return "improving"
            elif delta < -thr:
                return "degrading"
            return "stable"

        accuracies = [p.accuracy for p in pts]
        confidences = [p.confidence for p in pts]
        cal_errors = [p.calibration_error for p in pts]
        entropies = [p.prediction_entropy for p in pts]
        fprs = [p.false_positive_rate for p in pts]
        fnrs = [p.false_negative_rate for p in pts]

        acc_trend = _classify_trend(accuracies, self.improve_thr)
        conf_trend = _classify_trend(confidences, self.improve_thr)
        cal_trend = _classify_trend([-e for e in cal_errors], self.improve_thr)
        ent_trend = _classify_trend([-e for e in entropies], self.improve_thr)
        fpr_trend = _classify_trend([-r for r in fprs], self.improve_thr)
        fnr_trend = _classify_trend([-r for r in fnrs], self.improve_thr)

        avg_acc = _mean(accuracies)
        first_acc = accuracies[0] if accuracies else 0.0
        latest_acc = accuracies[-1] if accuracies else 0.0

        trends = [acc_trend, conf_trend, cal_trend]
        n_improving = trends.count("improving")
        n_degrading = trends.count("degrading")
        if n_improving > n_degrading:
            overall = "improving"
        elif n_degrading > n_improving:
            overall = "degrading"
        else:
            overall = "stable"

        return HealthTrend(
            model_id=model_id,
            n_points=n,
            accuracy_trend=acc_trend,
            confidence_trend=conf_trend,
            calibration_trend=cal_trend,
            entropy_trend=ent_trend,
            fpr_trend=fpr_trend,
            fnr_trend=fnr_trend,
            avg_accuracy=round(avg_acc, 4),
            avg_confidence=round(_mean(confidences), 4),
            avg_calibration_error=round(_mean(cal_errors), 4),
            avg_entropy=round(_mean(entropies), 4),
            avg_fpr=round(_mean(fprs), 4),
            avg_fnr=round(_mean(fnrs), 4),
            first_accuracy=round(first_acc, 4),
            latest_accuracy=round(latest_acc, 4),
            accuracy_delta=round(latest_acc - first_acc, 4),
            overall_health=overall,
        )

    def evolution_report(self, model_id: str) -> Dict[str, Any]:
        """Generate full health evolution report for a model."""
        trend = self.get_trend(model_id)
        pts = self._all_points.get(model_id, [])
        timeline = [
            {
                "timestamp": p.timestamp,
                "accuracy": p.accuracy,
                "confidence": p.confidence,
                "calibration_error": p.calibration_error,
                "fpr": p.false_positive_rate,
                "fnr": p.false_negative_rate,
            }
            for p in pts
        ]
        return {
            "model_id": model_id,
            "trend": trend.to_dict(),
            "n_points": len(pts),
            "timeline": timeline,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def all_trends(self) -> Dict[str, HealthTrend]:
        return {mid: self.get_trend(mid) for mid in self._all_points}

    def degrading_models(self) -> List[str]:
        return [mid for mid, t in self.all_trends().items() if t.overall_health == "degrading"]

    def improving_models(self) -> List[str]:
        return [mid for mid, t in self.all_trends().items() if t.overall_health == "improving"]

    def global_summary(self) -> Dict[str, Any]:
        trends = self.all_trends()
        return {
            "total_models_tracked": len(trends),
            "degrading": len(self.degrading_models()),
            "improving": len(self.improving_models()),
            "stable": sum(1 for t in trends.values() if t.overall_health == "stable"),
            "total_health_points": sum(len(v) for v in self._all_points.values()),
        }
