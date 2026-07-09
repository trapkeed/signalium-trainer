"""S22B — Decision Quality Monitoring.

Tracks, per closed decision/trade, how well predicted confidence tracks
realized outcomes, and breaks down expectancy by regime, symbol, session,
and volatility state. Distinct from ``health_monitor.py``'s CALIBRATION
drift dimension (a single rolling ECE alert) — this module is the
queryable, persisted decision-quality *report* surface (confidence-outcome
correlation + full expectancy breakdowns), and also exposes a pure
drift-check function that ``ResearchLabHealthMonitor`` wires into its
existing CRITICAL → RetrainingRecommendation alerting path.

Pure-Python (no numpy/torch) so it can be imported by the torch-free
health monitor and Research Lab server without pulling in the training
stack.
"""

from __future__ import annotations

import math
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

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class DecisionQualityConfig:
    """Thresholds and window sizes for decision-quality tracking/drift checks."""

    baseline_window: int = 200
    rolling_window: int = 50
    min_samples: int = 10

    # Volatility bucketing (ATR-based; used when trades don't carry an
    # explicit "volatility_state" key in market_conditions).
    low_atr_threshold: float = 10.0
    high_atr_threshold: float = 25.0

    # Degradation thresholds (drift checks, wired into health monitor)
    expectancy_drift_warning: float = 0.15  # relative drop in expectancy
    expectancy_drift_critical: float = 0.35
    correlation_drift_warning: float = 0.20  # absolute drop in confidence-outcome corr
    correlation_drift_critical: float = 0.40


# ── Report ────────────────────────────────────────────────────────────────────


@dataclass
class DecisionQualityReport:
    """A point-in-time decision-quality snapshot for a model version."""

    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_trades: int
    n_decisions: int
    overall_expectancy: float
    overall_win_rate: float
    confidence_outcome_correlation: float | None
    expectancy_by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    expectancy_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    expectancy_by_session: dict[str, dict[str, Any]] = field(default_factory=dict)
    expectancy_by_volatility_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_trades": self.n_trades,
            "n_decisions": self.n_decisions,
            "overall_expectancy": round(self.overall_expectancy, 4),
            "overall_win_rate": round(self.overall_win_rate, 4),
            "confidence_outcome_correlation": (
                round(self.confidence_outcome_correlation, 4)
                if self.confidence_outcome_correlation is not None
                else None
            ),
            "expectancy_by_regime": self.expectancy_by_regime,
            "expectancy_by_symbol": self.expectancy_by_symbol,
            "expectancy_by_session": self.expectancy_by_session,
            "expectancy_by_volatility_state": self.expectancy_by_volatility_state,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecisionQualityReport:
        return cls(
            report_id=d["report_id"],
            model_version=d.get("model_version", ""),
            created_at=d.get("created_at", 0.0),
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            n_trades=d.get("n_trades", 0),
            n_decisions=d.get("n_decisions", 0),
            overall_expectancy=d.get("overall_expectancy", 0.0),
            overall_win_rate=d.get("overall_win_rate", 0.0),
            confidence_outcome_correlation=d.get("confidence_outcome_correlation"),
            expectancy_by_regime=d.get("expectancy_by_regime", {}),
            expectancy_by_symbol=d.get("expectancy_by_symbol", {}),
            expectancy_by_session=d.get("expectancy_by_session", {}),
            expectancy_by_volatility_state=d.get("expectancy_by_volatility_state", {}),
        )


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _volatility_bucket(trade: dict[str, Any], cfg: DecisionQualityConfig) -> str:
    """Bucket a trade into a volatility state.

    Prefers an explicit ``volatility_state`` key in ``market_conditions``
    (set upstream by Signalium); otherwise derives a LOW/MEDIUM/HIGH bucket
    from an ``atr`` value in the same dict. Falls back to "UNKNOWN" when
    neither is available.
    """
    mc = trade.get("market_conditions") or {}
    if "volatility_state" in mc:
        return str(mc["volatility_state"]).upper()
    atr = mc.get("atr")
    if atr is None:
        return "UNKNOWN"
    atr = float(atr)
    if atr < cfg.low_atr_threshold:
        return "LOW"
    if atr < cfg.high_atr_threshold:
        return "MEDIUM"
    return "HIGH"


def _expectancy_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Win rate, expectancy, and avg win/loss for a group of trades."""
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    pnls = [t.get("pnl", 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    return {
        "n_trades": n,
        "win_rate": round(win_rate, 4),
        "expectancy": round(expectancy, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
    }


def _group_expectancy(trades: list[dict[str, Any]], key_fn: Any) -> dict[str, dict[str, Any]]:
    """Group trades by ``key_fn(trade)`` and compute expectancy stats per group."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        key = key_fn(t) or "UNKNOWN"
        groups.setdefault(str(key), []).append(t)
    return {k: _expectancy_stats(v) for k, v in groups.items()}


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient, pure Python. ``None`` if undefined
    (fewer than 2 points, or zero variance in either series)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return None
    return cov / denom


def confidence_outcome_correlation(trades: list[dict[str, Any]]) -> float | None:
    """Correlation between ``confidence_at_entry`` and realized outcome
    (1.0 = win, 0.0 = loss/breakeven). A well-calibrated model with
    meaningful confidence should show a positive correlation; a value near
    zero or negative indicates confidence doesn't track outcomes.
    """
    usable = [t for t in trades if "confidence_at_entry" in t]
    if len(usable) < 2:
        return None
    confs = [float(t["confidence_at_entry"]) for t in usable]
    outcomes = [1.0 if t.get("pnl", 0.0) > 0 else 0.0 for t in usable]
    return _pearson_correlation(confs, outcomes)


# ── Tracker ───────────────────────────────────────────────────────────────────


class DecisionQualityTracker:
    """Computes, persists, and queries decision-quality reports.

    Usage::

        tracker = DecisionQualityTracker(db, warehouse)
        report = tracker.compute_report(model_version="v1.0")
        latest = tracker.get_latest_report(model_version="v1.0")
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        warehouse: TelemetryWarehouse,
        config: DecisionQualityConfig | None = None,
    ) -> None:
        self._db = db
        self._wh = warehouse
        self._cfg = config or DecisionQualityConfig()

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> DecisionQualityReport:
        """Compute (and by default persist) a decision-quality report over
        all trades/decisions matching the given filters."""
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000)
        trades = sorted(self._wh.query_trades(q), key=lambda t: t.get("timestamp", 0))
        decisions = self._wh.query_decisions(q)

        now = time.time()
        window_start = trades[0].get("timestamp", now) if trades else now
        window_end = trades[-1].get("timestamp", now) if trades else now

        overall = _expectancy_stats(trades)
        corr = confidence_outcome_correlation(trades)

        report = DecisionQualityReport(
            report_id=f"dqr-{uuid.uuid4().hex[:12]}",
            model_version=model_version or "",
            created_at=now,
            window_start=window_start,
            window_end=window_end,
            n_trades=len(trades),
            n_decisions=len(decisions),
            overall_expectancy=overall["expectancy"],
            overall_win_rate=overall["win_rate"],
            confidence_outcome_correlation=corr,
            expectancy_by_regime=_group_expectancy(trades, lambda t: t.get("regime")),
            expectancy_by_symbol=_group_expectancy(trades, lambda t: t.get("symbol")),
            expectancy_by_session=_group_expectancy(trades, lambda t: t.get("session")),
            expectancy_by_volatility_state=_group_expectancy(
                trades, lambda t: _volatility_bucket(t, self._cfg)
            ),
        )
        if persist:
            self._db.upsert_decision_quality_report(report.to_dict())
            logger.info(
                f"DecisionQualityTracker: computed report {report.report_id} "
                f"for model={model_version!r} n_trades={report.n_trades}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_decision_quality_reports(model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(
        self, model_version: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._db.list_decision_quality_reports(model_version=model_version, limit=limit)


# ── Drift check (wired into health_monitor's alerting path) ─────────────────


def check_decision_quality_drift(
    trades: list[dict[str, Any]],
    cfg: DecisionQualityConfig,
    split_windows: Any,
) -> dict[str, Any] | None:
    """Compare baseline vs recent windows' expectancy and confidence-outcome
    correlation; return evidence when either has degraded significantly.

    ``split_windows`` is injected (rather than imported) so this module has
    no dependency on ``health_monitor``/``dimension_evaluators`` — the
    caller (health monitor) supplies its own disjoint-window splitter.

    Returns ``None`` when there isn't enough data or nothing has degraded.
    Otherwise returns a dict with ``severity`` ("WARNING"/"CRITICAL") and
    an ``evidence`` payload ready to embed in a DriftAlert.
    """
    if len(trades) < cfg.min_samples * 2:
        return None

    baseline_t, recent_t = split_windows(
        trades, cfg.baseline_window, cfg.rolling_window, cfg.min_samples
    )
    if baseline_t is None or recent_t is None:
        return None

    b_stats = _expectancy_stats(baseline_t)
    r_stats = _expectancy_stats(recent_t)
    b_exp = b_stats["expectancy"]
    r_exp = r_stats["expectancy"]

    # Relative expectancy drop — only degradation (falling expectancy) matters.
    # Use abs(b_exp) as the normalizer so a swing from positive to negative
    # expectancy always registers as a large relative drop, and guard
    # against division by ~0 when the baseline itself was flat.
    exp_drop = 0.0
    if abs(b_exp) > 1e-6:
        exp_drop = max(0.0, (b_exp - r_exp) / abs(b_exp))
    elif r_exp < b_exp:
        exp_drop = 1.0 if r_exp < -1e-6 else 0.0

    b_corr = confidence_outcome_correlation(baseline_t)
    r_corr = confidence_outcome_correlation(recent_t)
    corr_drop = 0.0
    corr_undefined = False
    if b_corr is not None and r_corr is not None:
        corr_drop = max(0.0, b_corr - r_corr)
    elif b_corr is not None and r_corr is None:
        # Confidence went from meaningfully tracking outcomes to not being
        # informative at all (e.g. it flatlined) — that is itself a
        # degradation, not a "nothing to compare" case. Treat it as a full
        # drop from the baseline correlation.
        corr_drop = max(0.0, b_corr)
        corr_undefined = True

    triggers = []
    if exp_drop >= cfg.expectancy_drift_warning:
        triggers.append("expectancy")
    if corr_drop >= cfg.correlation_drift_warning:
        triggers.append("confidence_outcome_correlation")
    if not triggers:
        return None

    is_critical = (
        exp_drop >= cfg.expectancy_drift_critical or corr_drop >= cfg.correlation_drift_critical
    )

    return {
        "severity": "CRITICAL" if is_critical else "WARNING",
        "triggers": triggers,
        "evidence": {
            "baseline_expectancy": b_stats["expectancy"],
            "recent_expectancy": r_stats["expectancy"],
            "expectancy_relative_drop": round(exp_drop, 4),
            "baseline_confidence_outcome_correlation": (
                round(b_corr, 4) if b_corr is not None else None
            ),
            "recent_confidence_outcome_correlation": (
                round(r_corr, 4) if r_corr is not None else None
            ),
            "confidence_outcome_correlation_drop": round(corr_drop, 4),
            "confidence_outcome_correlation_became_undefined": corr_undefined,
            "expectancy_warning_threshold": cfg.expectancy_drift_warning,
            "expectancy_critical_threshold": cfg.expectancy_drift_critical,
            "correlation_warning_threshold": cfg.correlation_drift_warning,
            "correlation_critical_threshold": cfg.correlation_drift_critical,
            "n_baseline": len(baseline_t),
            "n_recent": len(recent_t),
        },
    }
