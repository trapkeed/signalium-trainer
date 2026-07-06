"""Phase 15 — Head-to-Head Model Comparison.

Compares models along all institutional dimensions:
  - Statistical metrics (accuracy, F1, calibration)
  - Trading metrics (Sharpe, profit factor, drawdown)
  - Latency and memory footprint
  - Calibration quality
  - Feature attribution consistency (if explainability is enabled)

Supports:
  - Current vs Champion
  - Champion vs Candidate
  - Champion vs Previous versions
  - Any arbitrary N-model comparison

Usage::

    comparison = ModelComparison()
    comparison.add_model("champion", model_a, probs_a, labels, trades_a, meta_a)
    comparison.add_model("candidate", model_b, probs_b, labels, trades_b, meta_b)
    report = comparison.compare()
    print(report.winner)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import numpy as np
from loguru import logger

from trainer.validation.statistical import StatisticalMetrics, StatisticalMetricsEngine
from trainer.validation.trading import TradingMetrics, TradingMetricsEngine

# ---------------------------------------------------------------------------
# Model snapshot
# ---------------------------------------------------------------------------


@dataclass
class ModelSnapshot:
    """A registered model entry in the comparison.

    Attributes:
        name: Model display name (e.g. "champion", "candidate").
        statistical: Statistical metrics.
        trading: Trading metrics.
        inference_latency_ms: Median inference latency per batch.
        model_size_mb: Number of parameters in MB.
        gpu_memory_mb: Peak GPU memory during inference.
        metadata: Arbitrary metadata dict (architecture, version, etc.).
    """

    name: str
    statistical: StatisticalMetrics
    trading: TradingMetrics
    inference_latency_ms: float = 0.0
    model_size_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


@dataclass
class ComparisonMetricDelta:
    """Delta between two model values on a single metric."""

    metric: str
    model_a_value: float
    model_b_value: float
    delta: float
    winner: str
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "model_a": round(self.model_a_value, 4),
            "model_b": round(self.model_b_value, 4),
            "delta": round(self.delta, 4),
            "winner": self.winner,
        }


@dataclass
class ComparisonReport:
    """Full head-to-head comparison report.

    Attributes:
        model_a_name, model_b_name: Names of the two models compared.
        metric_deltas: Per-metric delta list.
        winner: Name of the overall winning model.
        a_wins: Number of metrics where A is better.
        b_wins: Number of metrics where B is better.
        ties: Number of tied metrics.
        is_significant: Whether the performance difference is meaningful.
    """

    model_a_name: str = ""
    model_b_name: str = ""
    metric_deltas: list[ComparisonMetricDelta] = field(default_factory=list)
    winner: str = "tie"
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    is_significant: bool = False

    def summary(self) -> str:
        lines = [
            f"=== Model Comparison: {self.model_a_name} vs {self.model_b_name} ===",
            f"  Winner: {self.winner.upper()}  "
            f"({self.a_wins}W / {self.b_wins}W / {self.ties}T)",
            "",
            f"  {'Metric':<30s} {'A':>10s} {'B':>10s} {'Delta':>10s} {'Winner':>12s}",
            "  " + "-" * 74,
        ]
        for d in self.metric_deltas:
            lines.append(
                f"  {d.metric:<30s} {d.model_a_value:>10.4f} "
                f"{d.model_b_value:>10.4f} {d.delta:>+10.4f} {d.winner:>12s}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "winner": self.winner,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
            "is_significant": self.is_significant,
            "metric_deltas": [d.to_dict() for d in self.metric_deltas],
        }


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------


class ModelComparison:
    """Builds and computes head-to-head model comparison reports.

    Usage::

        comp = ModelComparison()
        comp.add_snapshot(snapshot_a)
        comp.add_snapshot(snapshot_b)
        report = comp.compare(snapshot_a.name, snapshot_b.name)
    """

    # Metrics and whether higher is better
    _STAT_METRICS: list[tuple[str, bool]] = [
        ("accuracy", True),
        ("f1_macro", True),
        ("roc_auc", True),
        ("pr_auc", True),
        ("ece", False),
        ("brier_score", False),
        ("log_loss", False),
    ]

    _TRADING_METRICS: list[tuple[str, bool]] = [
        ("win_rate", True),
        ("profit_factor", True),
        ("sharpe_ratio", True),
        ("sortino_ratio", True),
        ("calmar_ratio", True),
        ("max_drawdown", False),
        ("expectancy", True),
        ("avg_rr", True),
        ("tp_before_sl_accuracy", True),
    ]

    _INFRA_METRICS: list[tuple[str, bool]] = [
        ("inference_latency_ms", False),
        ("model_size_mb", False),
    ]

    def __init__(self) -> None:
        self._snapshots: dict[str, ModelSnapshot] = {}

    def add_snapshot(self, snapshot: ModelSnapshot) -> None:
        """Register a model snapshot for comparison."""
        self._snapshots[snapshot.name] = snapshot
        logger.debug(f"ModelComparison: added snapshot {snapshot.name!r}")

    @classmethod
    def from_models(
        cls,
        model_a: nn.Module,
        model_b: nn.Module,
        name_a: str,
        name_b: str,
        probs_a: torch.Tensor,
        probs_b: torch.Tensor,
        labels: torch.Tensor,
        trades_a: Optional[list] = None,
        trades_b: Optional[list] = None,
        n_classes: int = 3,
    ) -> "ModelComparison":
        """Factory: compute snapshots from raw model outputs.

        Args:
            model_a, model_b: Models to compare (for latency / size measurement).
            name_a, name_b: Display names.
            probs_a, probs_b: ``(N, C)`` softmax probabilities.
            labels: ``(N,)`` ground-truth labels.
            trades_a, trades_b: Trade result lists (optional).
            n_classes: Number of output classes.

        Returns:
            Populated ``ModelComparison`` ready for ``.compare()``.
        """
        stat_engine = StatisticalMetricsEngine(n_classes=n_classes)
        trade_engine = TradingMetricsEngine()

        comp = cls()
        for name, model, probs, trades in [
            (name_a, model_a, probs_a, trades_a or []),
            (name_b, model_b, probs_b, trades_b or []),
        ]:
            stat = stat_engine.compute(probs, labels)
            trading = trade_engine.compute(trades)
            latency_ms = _measure_latency_ms(model, probs.shape[0])
            size_mb = _model_size_mb(model)
            snapshot = ModelSnapshot(
                name=name,
                statistical=stat,
                trading=trading,
                inference_latency_ms=latency_ms,
                model_size_mb=size_mb,
            )
            comp.add_snapshot(snapshot)
        return comp

    def compare(
        self,
        name_a: str,
        name_b: str,
        significance_threshold: float = 0.005,
    ) -> ComparisonReport:
        """Compare two registered models.

        Args:
            name_a: Name of the first model.
            name_b: Name of the second model.
            significance_threshold: Minimum absolute delta to count as a win.

        Returns:
            ``ComparisonReport``.
        """
        a = self._snapshots[name_a]
        b = self._snapshots[name_b]

        deltas: list[ComparisonMetricDelta] = []
        a_wins = b_wins = ties = 0

        # Statistical metrics
        for metric, higher_is_better in self._STAT_METRICS:
            va = float(getattr(a.statistical, metric, 0.0))
            vb = float(getattr(b.statistical, metric, 0.0))
            delta = vb - va
            if higher_is_better:
                win = (
                    name_b
                    if delta > significance_threshold
                    else (name_a if delta < -significance_threshold else "tie")
                )
            else:
                win = (
                    name_a
                    if delta > significance_threshold
                    else (name_b if delta < -significance_threshold else "tie")
                )
            if win == name_a:
                a_wins += 1
            elif win == name_b:
                b_wins += 1
            else:
                ties += 1
            deltas.append(
                ComparisonMetricDelta(
                    metric=metric,
                    model_a_value=va,
                    model_b_value=vb,
                    delta=delta,
                    winner=win,
                    higher_is_better=higher_is_better,
                )
            )

        # Trading metrics
        for metric, higher_is_better in self._TRADING_METRICS:
            va = float(getattr(a.trading, metric, 0.0))
            vb = float(getattr(b.trading, metric, 0.0))
            delta = vb - va
            if higher_is_better:
                win = (
                    name_b
                    if delta > significance_threshold
                    else (name_a if delta < -significance_threshold else "tie")
                )
            else:
                win = (
                    name_a
                    if delta > significance_threshold
                    else (name_b if delta < -significance_threshold else "tie")
                )
            if win == name_a:
                a_wins += 1
            elif win == name_b:
                b_wins += 1
            else:
                ties += 1
            deltas.append(
                ComparisonMetricDelta(
                    metric=metric,
                    model_a_value=va,
                    model_b_value=vb,
                    delta=delta,
                    winner=win,
                    higher_is_better=higher_is_better,
                )
            )

        # Infrastructure metrics
        for metric, higher_is_better in self._INFRA_METRICS:
            va = float(getattr(a, metric, 0.0))
            vb = float(getattr(b, metric, 0.0))
            delta = vb - va
            win = name_a if delta > 0 else (name_b if delta < 0 else "tie")
            if win == name_a:
                a_wins += 1
            elif win == name_b:
                b_wins += 1
            else:
                ties += 1
            deltas.append(
                ComparisonMetricDelta(
                    metric=metric,
                    model_a_value=va,
                    model_b_value=vb,
                    delta=delta,
                    winner=win,
                    higher_is_better=higher_is_better,
                )
            )

        overall_winner = name_a if a_wins > b_wins else (name_b if b_wins > a_wins else "tie")
        is_significant = abs(a_wins - b_wins) >= 3

        return ComparisonReport(
            model_a_name=name_a,
            model_b_name=name_b,
            metric_deltas=deltas,
            winner=overall_winner,
            a_wins=a_wins,
            b_wins=b_wins,
            ties=ties,
            is_significant=is_significant,
        )

    def compare_all(self) -> list[ComparisonReport]:
        """Compare every pair of registered snapshots."""
        names = sorted(self._snapshots.keys())
        reports = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                reports.append(self.compare(names[i], names[j]))
        return reports


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _measure_latency_ms(model: nn.Module, batch_size: int, n_repeats: int = 10) -> float:
    """Measure median inference latency in milliseconds."""
    import time

    try:
        device = next(model.parameters()).device
        dummy = torch.randn(batch_size, 10, 5, device=device)
        times = []
        was_training = model.training
        model.eval()
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
        if was_training:
            model.train()
        return float(np.median(times))
    except Exception:
        return 0.0


def _model_size_mb(model: nn.Module) -> float:
    """Compute model parameter size in megabytes."""
    try:
        n_params = sum(p.numel() for p in model.parameters())
        return n_params * 4 / (1024**2)  # float32
    except Exception:
        return 0.0
