"""Phase 19 — Research Observatory.

Centralised monitoring hub for the Institutional Research Intelligence Platform.

Continuously tracks:
  - Production model performance
  - Validation results
  - Training dataset statistics
  - Feature distributions
  - Market regimes
  - Prediction quality
  - Signal outcomes
  - Trade statistics
  - Calibration
  - Latency
  - Resource utilization

Usage::

    obs = ResearchObservatory()
    obs.record_production_snapshot(model_id="m1", metrics={...})
    obs.record_signal_outcome(model_id="m1", signal="buy",
                               outcome="profit", pnl=0.012)
    report = obs.get_observatory_report()
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Production snapshot
# ---------------------------------------------------------------------------


@dataclass
class ProductionSnapshot:
    """One production monitoring snapshot for a model."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    throughput_rps: float = 0.0
    cpu_pct: float = 0.0
    gpu_pct: float = 0.0
    memory_mb: float = 0.0
    avg_confidence: float = 0.5
    avg_uncertainty: float = 0.5
    signal_rate: float = 0.0
    accuracy: float = 0.0
    calibration_error: float = 0.0
    regime: str = "unknown"
    n_predictions: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "model_id": self.model_id,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "throughput_rps": self.throughput_rps,
            "cpu_pct": self.cpu_pct,
            "gpu_pct": self.gpu_pct,
            "memory_mb": self.memory_mb,
            "avg_confidence": self.avg_confidence,
            "avg_uncertainty": self.avg_uncertainty,
            "signal_rate": self.signal_rate,
            "accuracy": self.accuracy,
            "calibration_error": self.calibration_error,
            "regime": self.regime,
            "n_predictions": self.n_predictions,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Signal outcome
# ---------------------------------------------------------------------------


@dataclass
class SignalOutcome:
    """One recorded signal outcome for research analysis."""
    outcome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    signal: str = "no_trade"       # buy / sell / no_trade
    outcome: str = "neutral"       # profit / loss / neutral
    pnl: float = 0.0
    confidence: float = 0.5
    regime: str = "unknown"
    instrument: str = ""
    timeframe: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    holding_bars: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_correct(self) -> bool:
        return (self.signal == "buy" and self.outcome == "profit") or \
               (self.signal == "sell" and self.outcome == "profit") or \
               (self.signal == "no_trade" and self.outcome == "neutral")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "model_id": self.model_id,
            "signal": self.signal,
            "outcome": self.outcome,
            "pnl": self.pnl,
            "confidence": self.confidence,
            "regime": self.regime,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "holding_bars": self.holding_bars,
            "is_correct": self.is_correct,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Validation result record
# ---------------------------------------------------------------------------


@dataclass
class ValidationRecord:
    """One recorded validation result for research tracking."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    model_version: str = ""
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    calibration_error: float = 0.0
    max_drawdown: float = 0.0
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    n_samples: int = 0
    notes: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "accuracy": self.accuracy,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "calibration_error": self.calibration_error,
            "max_drawdown": self.max_drawdown,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "n_samples": self.n_samples,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Observatory
# ---------------------------------------------------------------------------


class ResearchObservatory:
    """Centralised research monitoring hub.

    Records production snapshots, signal outcomes, validation results,
    and aggregates research signals across all active models.

    Args:
        window: Rolling window for snapshot/outcome buffers.
    """

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self._snapshots: Dict[str, Deque[ProductionSnapshot]] = {}
        self._all_snapshots: Dict[str, List[ProductionSnapshot]] = {}
        self._outcomes: Dict[str, Deque[SignalOutcome]] = {}
        self._all_outcomes: Dict[str, List[SignalOutcome]] = {}
        self._validation_records: List[ValidationRecord] = []
        self._trade_stats: Dict[str, Dict[str, Any]] = {}

    def record_production_snapshot(self, model_id: str,
                                    metrics: Optional[Dict[str, float]] = None,
                                    latency_ms: float = 0.0,
                                    throughput_rps: float = 0.0,
                                    cpu_pct: float = 0.0,
                                    gpu_pct: float = 0.0,
                                    memory_mb: float = 0.0,
                                    avg_confidence: float = 0.5,
                                    avg_uncertainty: float = 0.5,
                                    signal_rate: float = 0.0,
                                    accuracy: float = 0.0,
                                    calibration_error: float = 0.0,
                                    regime: str = "unknown",
                                    n_predictions: int = 0) -> ProductionSnapshot:
        """Record one production monitoring snapshot."""
        snap = ProductionSnapshot(
            model_id=model_id,
            metrics=metrics or {},
            latency_ms=latency_ms, throughput_rps=throughput_rps,
            cpu_pct=cpu_pct, gpu_pct=gpu_pct, memory_mb=memory_mb,
            avg_confidence=avg_confidence, avg_uncertainty=avg_uncertainty,
            signal_rate=signal_rate, accuracy=accuracy,
            calibration_error=calibration_error,
            regime=regime, n_predictions=n_predictions,
        )
        if model_id not in self._snapshots:
            self._snapshots[model_id] = deque(maxlen=self.window)
            self._all_snapshots[model_id] = []
        self._snapshots[model_id].append(snap)
        self._all_snapshots[model_id].append(snap)
        return snap

    def record_signal_outcome(self, model_id: str,
                               signal: str = "no_trade",
                               outcome: str = "neutral",
                               pnl: float = 0.0,
                               confidence: float = 0.5,
                               regime: str = "unknown",
                               instrument: str = "",
                               timeframe: str = "") -> SignalOutcome:
        """Record one signal outcome for research analysis."""
        so = SignalOutcome(
            model_id=model_id, signal=signal, outcome=outcome,
            pnl=pnl, confidence=confidence, regime=regime,
            instrument=instrument, timeframe=timeframe,
        )
        if model_id not in self._outcomes:
            self._outcomes[model_id] = deque(maxlen=self.window)
            self._all_outcomes[model_id] = []
        self._outcomes[model_id].append(so)
        self._all_outcomes[model_id].append(so)
        return so

    def record_validation(self, record: ValidationRecord) -> None:
        self._validation_records.append(record)

    def get_signal_stats(self, model_id: str) -> Dict[str, Any]:
        """Compute signal quality statistics for a model."""
        outcomes = list(self._outcomes.get(model_id, []))
        n = len(outcomes)
        if n == 0:
            return {"model_id": model_id, "n_outcomes": 0}
        n_profit = sum(1 for o in outcomes if o.outcome == "profit")
        n_loss = sum(1 for o in outcomes if o.outcome == "loss")
        total_pnl = sum(o.pnl for o in outcomes)
        gross_profit = sum(o.pnl for o in outcomes if o.pnl > 0)
        gross_loss = abs(sum(o.pnl for o in outcomes if o.pnl < 0))
        win_rate = n_profit / n if n > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        n_buy = sum(1 for o in outcomes if o.signal == "buy")
        n_sell = sum(1 for o in outcomes if o.signal == "sell")
        n_no_trade = sum(1 for o in outcomes if o.signal == "no_trade")
        return {
            "model_id": model_id,
            "n_outcomes": n,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 6),
            "gross_profit": round(gross_profit, 6),
            "gross_loss": round(gross_loss, 6),
            "profit_factor": round(min(profit_factor, 999.0), 4),
            "n_profit": n_profit,
            "n_loss": n_loss,
            "signal_distribution": {
                "buy": n_buy / n, "sell": n_sell / n,
                "no_trade": n_no_trade / n,
            },
            "avg_pnl_per_trade": round(total_pnl / n, 6),
        }

    def get_production_trend(self, model_id: str) -> Dict[str, Any]:
        """Compute production metric trends for a model."""
        snaps = list(self._snapshots.get(model_id, []))
        n = len(snaps)
        if n == 0:
            return {"model_id": model_id, "n_snapshots": 0}
        avg_conf = sum(s.avg_confidence for s in snaps) / n
        avg_lat = sum(s.latency_ms for s in snaps) / n
        avg_acc = sum(s.accuracy for s in snaps) / n
        avg_cal = sum(s.calibration_error for s in snaps) / n
        if n >= 4:
            half = n // 2
            first_conf = sum(s.avg_confidence for s in snaps[:half]) / half
            second_conf = sum(s.avg_confidence for s in snaps[half:]) / (n - half)
            conf_trend = "improving" if second_conf > first_conf + 0.02 else (
                "declining" if second_conf < first_conf - 0.02 else "stable"
            )
        else:
            conf_trend = "insufficient"
        return {
            "model_id": model_id,
            "n_snapshots": n,
            "avg_confidence": round(avg_conf, 4),
            "avg_latency_ms": round(avg_lat, 2),
            "avg_accuracy": round(avg_acc, 4),
            "avg_calibration_error": round(avg_cal, 4),
            "confidence_trend": conf_trend,
        }

    def get_observatory_report(self) -> Dict[str, Any]:
        """Generate full observatory report across all models."""
        model_ids = list(set(list(self._snapshots.keys()) +
                              list(self._outcomes.keys())))
        return {
            "report_type": "observatory_report",
            "n_models_observed": len(model_ids),
            "n_validation_records": len(self._validation_records),
            "production_trends": {
                mid: self.get_production_trend(mid) for mid in self._snapshots
            },
            "signal_stats": {
                mid: self.get_signal_stats(mid) for mid in self._outcomes
            },
            "total_snapshots": sum(
                len(v) for v in self._all_snapshots.values()
            ),
            "total_outcomes": sum(
                len(v) for v in self._all_outcomes.values()
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def n_models(self) -> int:
        return len(set(list(self._snapshots.keys()) +
                        list(self._outcomes.keys())))
