"""S25 — QDE Observability and Model Intelligence.

Fills in the monitoring surfaces that were not already covered by the
existing S20/S22B Research Lab modules (inference telemetry, feature/regime
drift, calibration, decision quality, autonomous performance, composite
health, etc.). See each class's docstring for exactly which of the S25
metric categories it implements.

Internal to the trainer container; every report/endpoint built on top of
this module is exposed to Signalium only through the Research Lab HTTP
server's admin-gated integration layer (``research_lab_server.py``), never
directly. No public/customer access.

Pure-Python (no numpy/torch), following the same Config/Report/Monitor
convention as ``performance_monitor.py`` / ``feature_health.py`` /
``regime_coverage.py``: a dataclass Config, a dataclass Report with
``to_dict``/``from_dict``, and a Monitor class with ``compute_report``/
``get_latest_report``/``list_reports`` backed by
``ResearchLabDatabase.upsert_observability_report`` (one shared table,
disambiguated by ``report_type`` — see rl_database.py).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryQuery, TelemetryWarehouse


# ── Shared helpers ────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (0-100). Pure Python, no numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * (len(ordered) - 1)))))
    return ordered[k]


def _histogram(
    items: list[dict[str, Any]],
    key_fn: Any,
    default: str = "UNKNOWN",
) -> dict[str, int]:
    """Count occurrences of ``key_fn(item)`` (stringified), grouping falsy/
    missing values under ``default`` so callers never get a silent gap."""
    counts: dict[str, int] = {}
    for item in items:
        key = key_fn(item)
        key_str = str(key) if key not in (None, "") else default
        counts[key_str] = counts.get(key_str, 0) + 1
    return counts


def _bucketed_histogram(
    values: list[float],
    edges: list[float],
    labels: list[str] | None = None,
) -> dict[str, int]:
    """Bucket numeric ``values`` into the bins implied by ``edges``
    (len(edges)+1 buckets: (-inf, e0], (e0, e1], ..., (e_last, +inf))."""
    if labels is None:
        labels = []
        prev = "-inf"
        for e in edges:
            labels.append(f"{prev}..{e}")
            prev = str(e)
        labels.append(f"{prev}..+inf")
    counts: dict[str, int] = {label: 0 for label in labels}
    for v in values:
        idx = 0
        while idx < len(edges) and v > edges[idx]:
            idx += 1
        counts[labels[idx]] += 1
    return counts


# ═════════════════════════════════════════════════════════════════════════════
# 1. Inference Monitoring — count, latency, throughput, queue length, and
#    system resource usage (CPU/memory/GPU), sourced from the new
#    ``tel_inference`` telemetry stream (see telemetry_models.py/
#    telemetry_warehouse.py/telemetry_server.py POST /telemetry/inference).
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class InferenceMonitorConfig:
    window_limit: int = 5000


@dataclass
class InferenceReport:
    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_samples: int
    total_inference_count: int
    avg_throughput_per_s: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_max: float
    avg_queue_length: float
    max_queue_length: float
    avg_cpu_pct: float
    avg_memory_pct: float
    avg_gpu_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_samples": self.n_samples,
            "total_inference_count": self.total_inference_count,
            "avg_throughput_per_s": self.avg_throughput_per_s,
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "latency_ms_max": self.latency_ms_max,
            "avg_queue_length": self.avg_queue_length,
            "max_queue_length": self.max_queue_length,
            "avg_cpu_pct": self.avg_cpu_pct,
            "avg_memory_pct": self.avg_memory_pct,
            "avg_gpu_pct": self.avg_gpu_pct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InferenceReport:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


class InferenceMonitor:
    """Aggregates ``tel_inference`` samples into a point-in-time serving report.

    Usage::

        monitor = InferenceMonitor(db, warehouse)
        report = monitor.compute_report(model_version="v1.0")
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        warehouse: TelemetryWarehouse,
        config: InferenceMonitorConfig | None = None,
    ) -> None:
        self._db = db
        self._wh = warehouse
        self._cfg = config or InferenceMonitorConfig()

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> InferenceReport:
        q = TelemetryQuery(
            model_version=model_version, since=since, until=until,
            limit=self._cfg.window_limit,
        )
        samples = sorted(self._wh.query_inference(q), key=lambda s: s.get("timestamp", 0))

        now = time.time()
        window_start = samples[0].get("timestamp", now) if samples else now
        window_end = samples[-1].get("timestamp", now) if samples else now

        latencies_p50 = [float(s.get("latency_ms_p50", 0.0)) for s in samples]
        latencies_p95 = [float(s.get("latency_ms_p95", 0.0)) for s in samples]
        latencies_max = [float(s.get("latency_ms_max", 0.0)) for s in samples]
        queue_lengths = [float(s.get("queue_length", 0.0)) for s in samples]
        cpu = [float(s.get("cpu_pct", 0.0)) for s in samples]
        mem = [float(s.get("memory_pct", 0.0)) for s in samples]
        gpu_values = [s.get("gpu_pct") for s in samples if s.get("gpu_pct") is not None]

        report = InferenceReport(
            report_id=f"inf-{uuid.uuid4().hex[:12]}",
            model_version=model_version or "",
            created_at=now,
            window_start=window_start,
            window_end=window_end,
            n_samples=len(samples),
            total_inference_count=sum(int(s.get("inference_count", 0)) for s in samples),
            avg_throughput_per_s=round(_mean([float(s.get("throughput_per_s", 0.0)) for s in samples]), 4),
            latency_ms_p50=round(_mean(latencies_p50), 3),
            latency_ms_p95=round(_percentile(latencies_p95, 95), 3),
            latency_ms_max=round(max(latencies_max) if latencies_max else 0.0, 3),
            avg_queue_length=round(_mean(queue_lengths), 3),
            max_queue_length=round(max(queue_lengths) if queue_lengths else 0.0, 3),
            avg_cpu_pct=round(_mean(cpu), 3),
            avg_memory_pct=round(_mean(mem), 3),
            avg_gpu_pct=round(_mean([float(g) for g in gpu_values]), 3) if gpu_values else None,
        )
        if persist:
            self._db.upsert_observability_report("inference", report.to_dict())
            logger.info(
                f"InferenceMonitor: computed report {report.report_id} "
                f"for model={model_version!r} n_samples={report.n_samples}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_observability_reports("inference", model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(self, model_version: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_observability_reports("inference", model_version=model_version, limit=limit)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Model Monitoring — decision/confidence/symbol/regime distributions
#    already existed elsewhere (qde_shadow.py, confidence_calibration
#    endpoint, regime_coverage.py); this module fills in the three that were
#    missing: risk-signal, position-size, and hold-duration distributions.
# ═════════════════════════════════════════════════════════════════════════════


def _risk_signal_of(decision: dict[str, Any]) -> str:
    """Best-effort categorical risk signal for a decision.

    Prefers an explicit ``risk_signal`` key in ``market_conditions`` (set
    upstream by Signalium, if it computes one); falls back to a
    confidence-based LOW/MEDIUM/HIGH bucketing so the distribution is never
    empty just because the upstream field is absent yet.
    """
    mc = decision.get("market_conditions") or {}
    if "risk_signal" in mc and mc["risk_signal"]:
        return str(mc["risk_signal"])
    confidence = decision.get("confidence")
    if confidence is None:
        return "UNKNOWN"
    if confidence < 0.4:
        return "HIGH"
    if confidence < 0.7:
        return "MEDIUM"
    return "LOW"


@dataclass
class ModelDistributionReport:
    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_decisions: int
    n_trades: int
    decision_distribution: dict[str, int] = field(default_factory=dict)
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    symbol_distribution: dict[str, int] = field(default_factory=dict)
    regime_distribution: dict[str, int] = field(default_factory=dict)
    risk_signal_distribution: dict[str, int] = field(default_factory=dict)
    position_size_distribution: dict[str, int] = field(default_factory=dict)
    hold_duration_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_decisions": self.n_decisions,
            "n_trades": self.n_trades,
            "decision_distribution": self.decision_distribution,
            "confidence_distribution": self.confidence_distribution,
            "symbol_distribution": self.symbol_distribution,
            "regime_distribution": self.regime_distribution,
            "risk_signal_distribution": self.risk_signal_distribution,
            "position_size_distribution": self.position_size_distribution,
            "hold_duration_distribution": self.hold_duration_distribution,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelDistributionReport:
        return cls(
            report_id=d["report_id"],
            model_version=d.get("model_version", ""),
            created_at=d.get("created_at", 0.0),
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            n_decisions=d.get("n_decisions", 0),
            n_trades=d.get("n_trades", 0),
            decision_distribution=d.get("decision_distribution", {}),
            confidence_distribution=d.get("confidence_distribution", {}),
            symbol_distribution=d.get("symbol_distribution", {}),
            regime_distribution=d.get("regime_distribution", {}),
            risk_signal_distribution=d.get("risk_signal_distribution", {}),
            position_size_distribution=d.get("position_size_distribution", {}),
            hold_duration_distribution=d.get("hold_duration_distribution", {}),
        )


_CONFIDENCE_BUCKET_EDGES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_CONFIDENCE_BUCKET_LABELS = [
    "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
    "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0",
]
_POSITION_SIZE_BUCKET_EDGES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
_HOLD_DURATION_BUCKET_EDGES_S = [60.0, 300.0, 900.0, 3600.0, 14400.0, 86400.0]
_HOLD_DURATION_BUCKET_LABELS = [
    "<1m", "1-5m", "5-15m", "15m-1h", "1-4h", "4-24h", ">24h",
]


class ModelDistributionMonitor:
    """Computes decision/confidence/symbol/regime/risk-signal/position-size/
    hold-duration distributions for a model version."""

    def __init__(self, db: ResearchLabDatabase, warehouse: TelemetryWarehouse) -> None:
        self._db = db
        self._wh = warehouse

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> ModelDistributionReport:
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000)
        decisions = sorted(self._wh.query_decisions(q), key=lambda d: d.get("timestamp", 0))
        trades = sorted(self._wh.query_trades(q), key=lambda t: t.get("timestamp", 0))

        now = time.time()
        all_ts = [d.get("timestamp", now) for d in decisions] + [t.get("timestamp", now) for t in trades]
        window_start = min(all_ts) if all_ts else now
        window_end = max(all_ts) if all_ts else now

        confidences = [float(d.get("confidence", 0.0)) for d in decisions]
        position_sizes = [
            float(d.get("position_size")) for d in decisions if d.get("position_size") is not None
        ]
        hold_durations = [float(t.get("hold_duration_s", 0.0)) for t in trades]

        report = ModelDistributionReport(
            report_id=f"mdist-{uuid.uuid4().hex[:12]}",
            model_version=model_version or "",
            created_at=now,
            window_start=window_start,
            window_end=window_end,
            n_decisions=len(decisions),
            n_trades=len(trades),
            decision_distribution=_histogram(decisions, lambda d: d.get("direction")),
            confidence_distribution=_bucketed_histogram(
                confidences, _CONFIDENCE_BUCKET_EDGES, _CONFIDENCE_BUCKET_LABELS
            ),
            symbol_distribution=_histogram(decisions, lambda d: d.get("symbol")),
            regime_distribution=_histogram(decisions, lambda d: d.get("regime")),
            risk_signal_distribution=_histogram(decisions, _risk_signal_of),
            position_size_distribution=_bucketed_histogram(
                position_sizes, _POSITION_SIZE_BUCKET_EDGES
            ),
            hold_duration_distribution=_bucketed_histogram(
                hold_durations, _HOLD_DURATION_BUCKET_EDGES_S, _HOLD_DURATION_BUCKET_LABELS
            ),
        )
        if persist:
            self._db.upsert_observability_report("model_distribution", report.to_dict())
            logger.info(
                f"ModelDistributionMonitor: computed report {report.report_id} "
                f"for model={model_version!r} n_decisions={report.n_decisions}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_observability_reports("model_distribution", model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(self, model_version: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_observability_reports("model_distribution", model_version=model_version, limit=limit)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Performance Monitoring extensions — directional accuracy, confidence
#    calibration, and most performance stats already existed
#    (performance_monitor.py, decision_quality.py, confidence_calibration
#    endpoint). This module fills in the four that were missing:
#    expected-value accuracy, risk-estimation accuracy, sizing quality, and
#    portfolio efficiency. All are computed from realized trade telemetry
#    against what the model predicted at entry.
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class AccuracyReport:
    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_trades: int
    directional_accuracy: float
    expected_value_accuracy: dict[str, Any] = field(default_factory=dict)
    risk_estimation_accuracy: dict[str, Any] = field(default_factory=dict)
    sizing_quality: dict[str, Any] = field(default_factory=dict)
    portfolio_efficiency: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_trades": self.n_trades,
            "directional_accuracy": self.directional_accuracy,
            "expected_value_accuracy": self.expected_value_accuracy,
            "risk_estimation_accuracy": self.risk_estimation_accuracy,
            "sizing_quality": self.sizing_quality,
            "portfolio_efficiency": self.portfolio_efficiency,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccuracyReport:
        return cls(
            report_id=d["report_id"],
            model_version=d.get("model_version", ""),
            created_at=d.get("created_at", 0.0),
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            n_trades=d.get("n_trades", 0),
            directional_accuracy=d.get("directional_accuracy", 0.0),
            expected_value_accuracy=d.get("expected_value_accuracy", {}),
            risk_estimation_accuracy=d.get("risk_estimation_accuracy", {}),
            sizing_quality=d.get("sizing_quality", {}),
            portfolio_efficiency=d.get("portfolio_efficiency", {}),
        )


def _predicted_direction_correct(trade: dict[str, Any]) -> bool | None:
    """Whether the trade's realized pnl agrees with its predicted direction.

    LONG is correct when pnl > 0 (price rose as predicted); SHORT is
    correct when pnl > 0 too, since pnl already encodes direction-adjusted
    return. Returns ``None`` (excluded from the accuracy denominator) for
    flat/breakeven trades and unrecognized directions.
    """
    direction = str(trade.get("direction", "")).upper()
    if direction not in ("LONG", "SHORT"):
        return None
    pnl = float(trade.get("pnl", 0.0))
    if abs(pnl) < 1e-9:
        return None
    return pnl > 0


def _expected_value_accuracy(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """How well confidence-implied expected value tracked realized pnl.

    Uses ``confidence_at_entry`` as a proxy for the model's predicted
    probability of a favorable outcome (predicted EV sign = positive iff
    confidence > 0.5), and compares against the realized pnl sign — the
    same "did the model's stated conviction pay off" question as
    directional accuracy, but scoped to confidence rather than direction,
    plus the average signed error between predicted and realized outcome.
    """
    scored = [t for t in trades if t.get("confidence_at_entry") is not None]
    if not scored:
        return {"n_trades": 0, "ev_direction_accuracy": 0.0, "mean_ev_error": 0.0}
    correct = 0
    errors = []
    for t in scored:
        conf = float(t.get("confidence_at_entry", 0.5))
        pnl = float(t.get("pnl", 0.0))
        predicted_favorable = conf > 0.5
        realized_favorable = pnl > 0
        if predicted_favorable == realized_favorable:
            correct += 1
        # Predicted EV proxy: (2*conf - 1) scaled by size; realized: pnl_pct.
        predicted_ev = (2.0 * conf - 1.0)
        realized_ev = float(t.get("pnl_pct", 0.0))
        errors.append(abs(predicted_ev - realized_ev))
    return {
        "n_trades": len(scored),
        "ev_direction_accuracy": round(correct / len(scored), 4),
        "mean_ev_error": round(_mean(errors), 4),
    }


def _risk_estimation_accuracy(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """How well the predicted risk (stop distance carried in
    ``market_conditions.predicted_stop_distance``, if present, else the
    original decision's ``stop_distance``) matched the realized adverse
    move on losing trades.

    For every losing trade with a known predicted stop distance, computes
    the ratio of realized loss to predicted risk: 1.0 means the loss was
    contained exactly at the predicted stop; > 1.0 means the realized loss
    exceeded the predicted risk (stop slipped or was not honored); < 1.0
    means the trade was closed for a smaller loss than the worst case the
    model priced in.
    """
    ratios = []
    for t in trades:
        pnl = float(t.get("pnl", 0.0))
        if pnl >= 0:
            continue
        mc = t.get("market_conditions") or {}
        stop_distance = mc.get("predicted_stop_distance") or mc.get("stop_distance")
        if not stop_distance or float(stop_distance) <= 0:
            continue
        entry_price = float(t.get("entry_price", 0.0)) or 1.0
        predicted_risk_pct = float(stop_distance) / entry_price
        realized_loss_pct = abs(float(t.get("pnl_pct", 0.0)))
        if predicted_risk_pct <= 0:
            continue
        ratios.append(realized_loss_pct / predicted_risk_pct)
    if not ratios:
        return {"n_losing_trades_scored": 0, "mean_realized_to_predicted_risk_ratio": None}
    return {
        "n_losing_trades_scored": len(ratios),
        "mean_realized_to_predicted_risk_ratio": round(_mean(ratios), 4),
        "pct_exceeded_predicted_risk": round(
            sum(1 for r in ratios if r > 1.05) / len(ratios), 4
        ),
    }


def _sizing_quality(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Correlation between position size and realized pnl_pct.

    A well-calibrated sizing policy sizes up conviction that pays off — so
    a positive correlation between ``size`` and ``pnl_pct`` is desirable
    (bigger positions on the trades that end up more profitable per unit
    risk), while a negative correlation flags a sizing policy that is, on
    net, anti-correlated with realized quality.
    """
    sized = [t for t in trades if t.get("size") is not None]
    if len(sized) < 2:
        return {"n_trades": len(sized), "size_pnl_correlation": 0.0}
    sizes = [float(t.get("size", 0.0)) for t in sized]
    pnl_pcts = [float(t.get("pnl_pct", 0.0)) for t in sized]
    mean_size = _mean(sizes)
    mean_pnl = _mean(pnl_pcts)
    cov = sum((s - mean_size) * (p - mean_pnl) for s, p in zip(sizes, pnl_pcts))
    var_size = sum((s - mean_size) ** 2 for s in sizes)
    var_pnl = sum((p - mean_pnl) ** 2 for p in pnl_pcts)
    denom = (var_size * var_pnl) ** 0.5
    correlation = cov / denom if denom > 1e-12 else 0.0
    return {
        "n_trades": len(sized),
        "size_pnl_correlation": round(correlation, 4),
        "mean_size": round(mean_size, 4),
        "mean_pnl_pct": round(mean_pnl, 4),
    }


def _portfolio_efficiency(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return generated per unit of capital deployed (sum of position
    sizes), distinct from ``performance_monitor``'s per-trade capital
    efficiency (return per unit of *gross loss*) — this is a portfolio-
    level view of how efficiently deployed size converted into return.
    """
    sized = [t for t in trades if t.get("size") is not None]
    total_pnl = sum(float(t.get("pnl", 0.0)) for t in sized)
    total_size_deployed = sum(abs(float(t.get("size", 0.0))) for t in sized)
    efficiency = total_pnl / total_size_deployed if total_size_deployed > 1e-9 else 0.0
    return {
        "n_trades": len(sized),
        "total_pnl": round(total_pnl, 4),
        "total_size_deployed": round(total_size_deployed, 4),
        "return_per_unit_size": round(efficiency, 6),
    }


class AccuracyMonitor:
    """Computes the four accuracy/efficiency metrics not already covered
    elsewhere: expected-value accuracy, risk-estimation accuracy, sizing
    quality, and portfolio efficiency — plus directional accuracy (the
    single-model equivalent of ``champion_challenger.score_direction_accuracy``,
    which only applies to two-model comparisons)."""

    def __init__(self, db: ResearchLabDatabase, warehouse: TelemetryWarehouse) -> None:
        self._db = db
        self._wh = warehouse

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> AccuracyReport:
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000)
        trades = sorted(self._wh.query_trades(q), key=lambda t: t.get("timestamp", 0))

        now = time.time()
        window_start = trades[0].get("timestamp", now) if trades else now
        window_end = trades[-1].get("timestamp", now) if trades else now

        directional = [_predicted_direction_correct(t) for t in trades]
        directional = [d for d in directional if d is not None]
        directional_accuracy = round(sum(1 for d in directional if d) / len(directional), 4) if directional else 0.0

        report = AccuracyReport(
            report_id=f"acc-{uuid.uuid4().hex[:12]}",
            model_version=model_version or "",
            created_at=now,
            window_start=window_start,
            window_end=window_end,
            n_trades=len(trades),
            directional_accuracy=directional_accuracy,
            expected_value_accuracy=_expected_value_accuracy(trades),
            risk_estimation_accuracy=_risk_estimation_accuracy(trades),
            sizing_quality=_sizing_quality(trades),
            portfolio_efficiency=_portfolio_efficiency(trades),
        )
        if persist:
            self._db.upsert_observability_report("accuracy", report.to_dict())
            logger.info(
                f"AccuracyMonitor: computed report {report.report_id} "
                f"for model={model_version!r} n_trades={report.n_trades}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_observability_reports("accuracy", model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(self, model_version: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_observability_reports("accuracy", model_version=model_version, limit=limit)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Autonomous Trading Monitoring — trade lifecycle stats already existed
#    (tel_lifecycle table); this fills in intervention frequency,
#    management-action statistics, stop-adjustment frequency, scale-in
#    frequency, average holding duration, and realized-vs-predicted risk,
#    derived from tel_lifecycle + tel_trades.
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class AutonomousTradingReport:
    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_lifecycle_events: int
    n_trades: int
    management_action_counts: dict[str, int] = field(default_factory=dict)
    intervention_frequency_per_trade: float = 0.0
    stop_adjustment_frequency_per_trade: float = 0.0
    scale_in_frequency_per_trade: float = 0.0
    avg_holding_duration_s: float = 0.0
    median_holding_duration_s: float = 0.0
    realized_vs_predicted_risk: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_lifecycle_events": self.n_lifecycle_events,
            "n_trades": self.n_trades,
            "management_action_counts": self.management_action_counts,
            "intervention_frequency_per_trade": self.intervention_frequency_per_trade,
            "stop_adjustment_frequency_per_trade": self.stop_adjustment_frequency_per_trade,
            "scale_in_frequency_per_trade": self.scale_in_frequency_per_trade,
            "avg_holding_duration_s": self.avg_holding_duration_s,
            "median_holding_duration_s": self.median_holding_duration_s,
            "realized_vs_predicted_risk": self.realized_vs_predicted_risk,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AutonomousTradingReport:
        return cls(
            report_id=d["report_id"],
            model_version=d.get("model_version", ""),
            created_at=d.get("created_at", 0.0),
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            n_lifecycle_events=d.get("n_lifecycle_events", 0),
            n_trades=d.get("n_trades", 0),
            management_action_counts=d.get("management_action_counts", {}),
            intervention_frequency_per_trade=d.get("intervention_frequency_per_trade", 0.0),
            stop_adjustment_frequency_per_trade=d.get("stop_adjustment_frequency_per_trade", 0.0),
            scale_in_frequency_per_trade=d.get("scale_in_frequency_per_trade", 0.0),
            avg_holding_duration_s=d.get("avg_holding_duration_s", 0.0),
            median_holding_duration_s=d.get("median_holding_duration_s", 0.0),
            realized_vs_predicted_risk=d.get("realized_vs_predicted_risk", {}),
        )


# Lifecycle event kinds counted as an active "management action" — i.e. the
# QDE system intervening on an already-open position, as opposed to
# open/close/session bookkeeping.
_MANAGEMENT_ACTION_KINDS = frozenset({
    "STOP_ADJUSTED", "TP_ADJUSTED", "SIZE_SCALED", "RISK_STATE_CHANGED",
})


class AutonomousTradingMonitor:
    """Aggregates ``tel_lifecycle`` + ``tel_trades`` into an autonomous-
    trading intervention/management report."""

    def __init__(self, db: ResearchLabDatabase, warehouse: TelemetryWarehouse) -> None:
        self._db = db
        self._wh = warehouse

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> AutonomousTradingReport:
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000)
        lifecycle_events = sorted(self._wh.query_lifecycle(q), key=lambda e: e.get("timestamp", 0))
        trades = sorted(self._wh.query_trades(q), key=lambda t: t.get("timestamp", 0))

        now = time.time()
        all_ts = [e.get("timestamp", now) for e in lifecycle_events] + [t.get("timestamp", now) for t in trades]
        window_start = min(all_ts) if all_ts else now
        window_end = max(all_ts) if all_ts else now

        management_events = [e for e in lifecycle_events if e.get("kind") in _MANAGEMENT_ACTION_KINDS]
        management_counts = _histogram(management_events, lambda e: e.get("kind"))
        stop_adjustments = management_counts.get("STOP_ADJUSTED", 0)
        scale_ins = management_counts.get("SIZE_SCALED", 0)

        n_trades = len(trades)
        holding_durations = [float(t.get("hold_duration_s", 0.0)) for t in trades]

        report = AutonomousTradingReport(
            report_id=f"atr-{uuid.uuid4().hex[:12]}",
            model_version=model_version or "",
            created_at=now,
            window_start=window_start,
            window_end=window_end,
            n_lifecycle_events=len(lifecycle_events),
            n_trades=n_trades,
            management_action_counts=management_counts,
            intervention_frequency_per_trade=round(len(management_events) / n_trades, 4) if n_trades else 0.0,
            stop_adjustment_frequency_per_trade=round(stop_adjustments / n_trades, 4) if n_trades else 0.0,
            scale_in_frequency_per_trade=round(scale_ins / n_trades, 4) if n_trades else 0.0,
            avg_holding_duration_s=round(_mean(holding_durations), 3),
            median_holding_duration_s=round(_percentile(holding_durations, 50), 3),
            realized_vs_predicted_risk=_risk_estimation_accuracy(trades),
        )
        if persist:
            self._db.upsert_observability_report("autonomous_trading", report.to_dict())
            logger.info(
                f"AutonomousTradingMonitor: computed report {report.report_id} "
                f"for model={model_version!r} n_trades={report.n_trades}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_observability_reports("autonomous_trading", model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(self, model_version: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_observability_reports("autonomous_trading", model_version=model_version, limit=limit)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Research Monitoring — aggregates surfaces that already exist elsewhere
#    (training queue via pipeline_jobs, active experiments, validation
#    queue, shadow deployments, pending approvals, retraining
#    recommendations) into a single "what is the Research Lab doing right
#    now" summary. Read-only — computes nothing new, just fans out to the
#    existing engines/DB accessors already wired into the server.
# ═════════════════════════════════════════════════════════════════════════════


class ResearchMonitor:
    """Fresh-computed (never persisted — this is a live status view, not a
    point-in-time report) summary of active research activity."""

    def __init__(
        self,
        db: ResearchLabDatabase,
        approval_gate: Any,
        rec_engine: Any,
    ) -> None:
        self._db = db
        self._approval_gate = approval_gate
        self._rec_engine = rec_engine

    def compute_summary(self) -> dict[str, Any]:
        pipeline_jobs = self._db.list_pipeline_jobs(limit=200)
        active_jobs = [j for j in pipeline_jobs if j.get("status") in ("QUEUED", "RUNNING")]
        training_queue = [j for j in active_jobs if j.get("current_stage") == "TRAINING"]
        validation_queue = [j for j in active_jobs if j.get("current_stage") == "VALIDATION"]

        experiments = self._db.list_experiments()
        active_experiments = [e for e in experiments if e.get("status") in ("RUNNING", "IN_PROGRESS", "PENDING")]

        shadow_tests = self._db.list_shadow_tests()
        active_shadow_tests = [s for s in shadow_tests if s.get("status") not in ("COMPLETED", "REJECTED", "PROMOTED")]

        pending_approvals = self._approval_gate.list_requests(status="PENDING") if self._approval_gate else []
        # RetrainingRecommendation lifecycle is PROPOSED -> APPROVED/DISMISSED
        # (no "PENDING" status exists) — PROPOSED is the "awaiting admin
        # decision" state that corresponds to "pending" here.
        pending_recommendations = self._rec_engine.list_recommendations(status="PROPOSED") if self._rec_engine else []

        return {
            "generated_at": round(time.time(), 3),
            "training_queue": {
                "n_pending_or_running": len(training_queue),
                "jobs": training_queue[:20],
            },
            "validation_queue": {
                "n_pending_or_running": len(validation_queue),
                "jobs": validation_queue[:20],
            },
            "active_experiments": {
                "count": len(active_experiments),
                "experiments": active_experiments[:20],
            },
            "shadow_deployments": {
                "count": len(active_shadow_tests),
                "tests": active_shadow_tests[:20],
            },
            "pending_approvals": {
                "count": len(pending_approvals),
                "requests": pending_approvals[:20],
            },
            "pending_retraining_recommendations": {
                "count": len(pending_recommendations),
                "recommendations": pending_recommendations[:20],
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# 6. QDE Health Dashboard — the single aggregated internal surface combining
#    version, deployment status, composite health/drift/confidence/
#    calibration scores, and research recommendations. This is the one
#    payload gated to exactly SuperAdmin/Admin/Founder/QuantResearcher/
#    MLResearcher/DeploymentApprover (see sso.Visibility.OBSERVABILITY_DASHBOARD),
#    excluding Auditor.
# ═════════════════════════════════════════════════════════════════════════════


class QDEHealthDashboard:
    """Aggregates the existing composite-health/deployment/recommendation
    surfaces into the one "QDE Health Dashboard" payload — computes
    nothing new, so it can never disagree with the underlying reports it
    reads from."""

    def __init__(
        self,
        deployment_manager: Any,
        composite_health_scorer: Any,
        health_monitor: Any,
        rec_engine: Any,
    ) -> None:
        self._deployment_manager = deployment_manager
        self._composite_health_scorer = composite_health_scorer
        self._health_monitor = health_monitor
        self._rec_engine = rec_engine

    def compute(self, model_version: str | None = None) -> dict[str, Any]:
        current_deployment = self._deployment_manager.get_current() if self._deployment_manager else None
        resolved_version = model_version or (current_deployment or {}).get("model_version")

        composite = None
        if self._composite_health_scorer is not None:
            composite = self._composite_health_scorer.get_latest(model_version=resolved_version)
            if composite is None:
                composite = self._composite_health_scorer.compute(model_version=resolved_version).to_dict()

        pending_recommendations = (
            self._rec_engine.list_recommendations(status="PROPOSED", model_version=resolved_version)
            if self._rec_engine else []
        )

        category_scores = {cs["category"]: cs for cs in (composite or {}).get("category_scores", [])}
        calibration_score = (category_scores.get("calibration") or {}).get("score")
        drift_score = (category_scores.get("drift") or {}).get("score")

        return {
            "generated_at": round(time.time(), 3),
            "model_version": resolved_version,
            "deployment_status": (current_deployment or {}).get("status", "NOT_DEPLOYED"),
            "deployment_updated_at": (current_deployment or {}).get("updated_at"),
            "composite_health": composite,
            "composite_score": (composite or {}).get("composite"),
            "grade": (composite or {}).get("grade"),
            "drift_score": drift_score,
            "calibration_score": calibration_score,
            "n_critical_alerts": (composite or {}).get("n_critical_alerts", 0),
            "n_warning_alerts": (composite or {}).get("n_warning_alerts", 0),
            "is_healthy": (composite or {}).get("n_critical_alerts", 0) == 0,
            "pending_research_recommendations": pending_recommendations,
        }
