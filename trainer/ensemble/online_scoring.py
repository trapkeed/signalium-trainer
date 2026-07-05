"""Phase 17 — AEIP Online Scoring and Rolling Performance Windows.

Continuously evaluates expert performance with configurable rolling windows.

Supports:
  - Rolling accuracy (configurable window)
  - Rolling Sharpe ratio
  - Rolling profit factor
  - Rolling win rate
  - Rolling calibration error
  - Automatic weight adjustment triggers
  - Performance alerts (underperformance detection)

Usage::

    scorer = OnlineScorer(window=200)
    scorer.record("expert_01", predicted_class="buy", true_class="buy",
                  profit_pnl=50.0, confidence=0.75)
    metrics = scorer.get_metrics("expert_01")
    print(metrics.rolling_accuracy, metrics.rolling_sharpe)

    # Check if weight adjustment is needed
    alerts = scorer.check_alerts()
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Rolling prediction record
# ---------------------------------------------------------------------------


@dataclass
class PredictionRecord:
    """One prediction outcome for online scoring."""

    expert_id: str
    predicted_class: str
    true_class: Optional[str]
    profit_pnl: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    regime: str = "unknown"
    instrument: str = ""
    timeframe: str = ""

    @property
    def is_correct(self) -> Optional[bool]:
        if self.true_class is None:
            return None
        return self.predicted_class == self.true_class

    @property
    def is_profitable(self) -> bool:
        return self.profit_pnl > 0


# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------


@dataclass
class RollingMetrics:
    """Current rolling performance metrics for one expert.

    Attributes:
        expert_id: Expert these metrics belong to.
        window: Rolling window size.
        n_records: Number of records in current window.
        rolling_accuracy: Fraction of correct predictions in window.
        rolling_win_rate: Fraction of profitable trades in window.
        rolling_profit_factor: Sum(wins) / Sum(|losses|) in window.
        rolling_sharpe: Annualized Sharpe of PnL series in window.
        rolling_avg_confidence: Mean confidence in window.
        rolling_avg_uncertainty: Mean uncertainty in window.
        n_total_records: All-time record count.
        all_time_accuracy: All-time accuracy.
        trend: "improving" | "declining" | "stable" | "insufficient_data".
        alert: Whether performance alert is active.
        alert_reason: Reason for alert.
    """

    expert_id: str
    window: int = 200
    n_records: int = 0
    rolling_accuracy: float = 0.0
    rolling_win_rate: float = 0.0
    rolling_profit_factor: float = 1.0
    rolling_sharpe: float = 0.0
    rolling_avg_confidence: float = 0.5
    rolling_avg_uncertainty: float = 0.5
    n_total_records: int = 0
    all_time_accuracy: float = 0.0
    trend: str = "insufficient_data"
    alert: bool = False
    alert_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "window": self.window,
            "n_records": self.n_records,
            "rolling_accuracy": self.rolling_accuracy,
            "rolling_win_rate": self.rolling_win_rate,
            "rolling_profit_factor": self.rolling_profit_factor,
            "rolling_sharpe": self.rolling_sharpe,
            "rolling_avg_confidence": self.rolling_avg_confidence,
            "rolling_avg_uncertainty": self.rolling_avg_uncertainty,
            "n_total_records": self.n_total_records,
            "all_time_accuracy": self.all_time_accuracy,
            "trend": self.trend,
            "alert": self.alert,
            "alert_reason": self.alert_reason,
        }


# ---------------------------------------------------------------------------
# Online scorer
# ---------------------------------------------------------------------------


class OnlineScorer:
    """Maintains rolling performance windows for all expert models.

    Args:
        window: Rolling window size in number of predictions.
        min_window_for_metrics: Minimum records before metrics are meaningful.
        alert_accuracy_threshold: Rolling accuracy below this triggers alert.
        alert_sharpe_threshold: Rolling Sharpe below this triggers alert.
        weight_adjust_accuracy_drop: Drop in accuracy that triggers weight update.
    """

    def __init__(self, window: int = 200,
                 min_window_for_metrics: int = 20,
                 alert_accuracy_threshold: float = 0.40,
                 alert_sharpe_threshold: float = -1.0,
                 weight_adjust_accuracy_drop: float = 0.10) -> None:
        self.window = window
        self.min_window = min_window_for_metrics
        self.alert_acc_threshold = alert_accuracy_threshold
        self.alert_sharpe_threshold = alert_sharpe_threshold
        self.weight_adjust_drop = weight_adjust_accuracy_drop
        self._buffers: Dict[str, Deque[PredictionRecord]] = {}
        self._all_records: Dict[str, List[PredictionRecord]] = {}
        self._peak_accuracy: Dict[str, float] = {}

    def record(self, expert_id: str,
               predicted_class: str,
               true_class: Optional[str] = None,
               profit_pnl: float = 0.0,
               confidence: float = 0.5,
               uncertainty: float = 0.5,
               regime: str = "unknown",
               instrument: str = "",
               timeframe: str = "") -> PredictionRecord:
        """Record one prediction outcome for an expert."""
        rec = PredictionRecord(
            expert_id=expert_id,
            predicted_class=predicted_class,
            true_class=true_class,
            profit_pnl=profit_pnl,
            confidence=confidence,
            uncertainty=uncertainty,
            regime=regime,
            instrument=instrument,
            timeframe=timeframe,
        )
        if expert_id not in self._buffers:
            self._buffers[expert_id] = deque(maxlen=self.window)
            self._all_records[expert_id] = []
        self._buffers[expert_id].append(rec)
        self._all_records[expert_id].append(rec)
        return rec

    def get_metrics(self, expert_id: str) -> RollingMetrics:
        """Compute rolling metrics for one expert."""
        buf = list(self._buffers.get(expert_id, []))
        all_recs = self._all_records.get(expert_id, [])

        n = len(buf)
        if n == 0:
            return RollingMetrics(expert_id=expert_id, window=self.window)

        labeled = [r for r in buf if r.true_class is not None]
        n_correct = sum(1 for r in labeled if r.is_correct)
        rolling_acc = n_correct / len(labeled) if labeled else 0.0

        wins = [r.profit_pnl for r in buf if r.profit_pnl > 0]
        losses = [abs(r.profit_pnl) for r in buf if r.profit_pnl < 0]
        rolling_win_rate = len(wins) / n
        sum_wins = sum(wins)
        sum_losses = sum(losses)
        pf = sum_wins / sum_losses if sum_losses > 0 else (2.0 if sum_wins > 0 else 1.0)

        pnls = [r.profit_pnl for r in buf]
        sharpe = _compute_sharpe(pnls)

        avg_conf = sum(r.confidence for r in buf) / n
        avg_unc = sum(r.uncertainty for r in buf) / n

        all_labeled = [r for r in all_recs if r.true_class is not None]
        all_correct = sum(1 for r in all_labeled if r.is_correct)
        all_time_acc = all_correct / len(all_labeled) if all_labeled else 0.0

        peak = self._peak_accuracy.get(expert_id, rolling_acc)
        if rolling_acc > peak:
            self._peak_accuracy[expert_id] = rolling_acc
            peak = rolling_acc

        if n < self.min_window:
            trend = "insufficient_data"
        elif rolling_acc >= peak - 0.02:
            trend = "stable"
        elif rolling_acc >= peak - 0.05:
            trend = "declining"
        else:
            trend = "declining"

        if n >= self.min_window:
            half = n // 2
            first_labeled = [r for r in labeled[:half] if r.true_class is not None]
            second_labeled = [r for r in labeled[half:] if r.true_class is not None]
            first_acc = sum(1 for r in first_labeled if r.is_correct) / max(len(first_labeled), 1)
            second_acc = sum(1 for r in second_labeled if r.is_correct) / max(len(second_labeled), 1)
            if second_acc > first_acc + 0.03:
                trend = "improving"
            elif second_acc < first_acc - 0.03:
                trend = "declining"
            else:
                trend = "stable"

        alert = False
        alert_reason = ""
        if n >= self.min_window:
            if rolling_acc < self.alert_acc_threshold:
                alert = True
                alert_reason = f"accuracy {rolling_acc:.2%} below threshold {self.alert_acc_threshold:.2%}"
            elif sharpe < self.alert_sharpe_threshold:
                alert = True
                alert_reason = f"Sharpe {sharpe:.2f} below threshold {self.alert_sharpe_threshold:.2f}"

        return RollingMetrics(
            expert_id=expert_id,
            window=self.window,
            n_records=n,
            rolling_accuracy=round(rolling_acc, 4),
            rolling_win_rate=round(rolling_win_rate, 4),
            rolling_profit_factor=round(pf, 4),
            rolling_sharpe=round(sharpe, 4),
            rolling_avg_confidence=round(avg_conf, 4),
            rolling_avg_uncertainty=round(avg_unc, 4),
            n_total_records=len(all_recs),
            all_time_accuracy=round(all_time_acc, 4),
            trend=trend,
            alert=alert,
            alert_reason=alert_reason,
        )

    def all_metrics(self) -> Dict[str, RollingMetrics]:
        """Return rolling metrics for all tracked experts."""
        return {eid: self.get_metrics(eid) for eid in self._buffers}

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Return list of experts with active performance alerts."""
        alerts = []
        for eid, metrics in self.all_metrics().items():
            if metrics.alert:
                alerts.append({
                    "expert_id": eid,
                    "reason": metrics.alert_reason,
                    "rolling_accuracy": metrics.rolling_accuracy,
                    "rolling_sharpe": metrics.rolling_sharpe,
                    "n_records": metrics.n_records,
                })
        return alerts

    def needs_weight_adjustment(self, expert_id: str) -> bool:
        """Return True if expert's accuracy drop warrants weight update."""
        m = self.get_metrics(expert_id)
        peak = self._peak_accuracy.get(expert_id, 0.0)
        if peak == 0.0:
            return False
        return (peak - m.rolling_accuracy) >= self.weight_adjust_drop

    def regime_breakdown(self, expert_id: str) -> Dict[str, Dict[str, Any]]:
        """Return per-regime metrics for an expert from the rolling window."""
        buf = list(self._buffers.get(expert_id, []))
        regimes: Dict[str, List[PredictionRecord]] = {}
        for r in buf:
            if r.regime not in regimes:
                regimes[r.regime] = []
            regimes[r.regime].append(r)
        result = {}
        for regime, records in regimes.items():
            labeled = [r for r in records if r.true_class is not None]
            correct = sum(1 for r in labeled if r.is_correct)
            acc = correct / len(labeled) if labeled else 0.0
            pnls = [r.profit_pnl for r in records]
            result[regime] = {
                "n": len(records),
                "accuracy": round(acc, 4),
                "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
                "sharpe": round(_compute_sharpe(pnls), 4),
            }
        return result


# ---------------------------------------------------------------------------
# Sharpe helper
# ---------------------------------------------------------------------------


def _compute_sharpe(pnls: List[float], annualization: float = 252.0) -> float:
    n = len(pnls)
    if n < 2:
        return 0.0
    mean = sum(pnls) / n
    variance = sum((p - mean) ** 2 for p in pnls) / n
    std = math.sqrt(variance)
    if std < 1e-10:
        return 0.0
    return (mean / std) * math.sqrt(annualization / n * n)
