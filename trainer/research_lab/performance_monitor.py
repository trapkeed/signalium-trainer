"""S22B — Autonomous Performance Monitoring.

Tracks live autonomous-trading performance — profit factor, Sharpe ratio,
max drawdown, hit rate, risk-adjusted return, and capital efficiency —
continuously from account/decision telemetry, broken down by symbol,
regime, and user cohort.

Distinct from ``decision_quality.py`` (confidence calibration + expectancy)
and the future composite QDE health score (S22B task #8) — this module is
purely about realized economic performance. Formulas are kept
formula-identical to ``trainer/validation/trading.py``'s offline
``TradingMetricsEngine`` (profit factor, Sharpe, max drawdown, capital
efficiency), ported to pure Python so the live/telemetry path stays
import-safe in this torch-broken environment (see memory:
torch-broken-lazy-registry-import) — see ``tests/unit/test_performance_monitor.py``'s
parity tests, which assert live and offline numbers agree on the same data.

Pure-Python (no numpy/torch) throughout this module.
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

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class PerformanceMonitorConfig:
    """Thresholds and window sizes for live performance tracking/drift checks."""

    baseline_window: int = 200
    rolling_window: int = 50
    min_samples: int = 10

    # Degradation thresholds (drift checks, wired into health monitor).
    # Relative drops for profit_factor / sharpe / risk_adjusted_return /
    # capital_efficiency; absolute drops for hit_rate; absolute increase
    # for drawdown (worse drawdown = higher value).
    profit_factor_drift_warning: float = 0.20
    profit_factor_drift_critical: float = 0.45
    sharpe_drift_warning: float = 0.30
    sharpe_drift_critical: float = 0.60
    drawdown_increase_warning: float = 0.10
    drawdown_increase_critical: float = 0.25
    hit_rate_drop_warning: float = 0.10
    hit_rate_drop_critical: float = 0.20

    # Sharpe formula parity with trainer.validation.trading.TradingMetricsEngine
    # (see _sharpe_ratio below) — same defaults as that engine's constructor
    # so live and offline Sharpe agree on the same trade data.
    annualisation_factor: float = 252.0
    risk_free_rate: float = 0.0


# ── Pure metric helpers ───────────────────────────────────────────────────────


def _profit_factor(pnls: list[float]) -> float:
    """Gross profit / gross loss. ``inf`` (capped by caller) when no losses."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    if gross_loss < 1e-9:
        return float("inf")
    return gross_profit / gross_loss


def _hit_rate(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    return sum(1 for p in pnls if p > 0) / len(pnls)


def _max_drawdown(pnls: list[float]) -> float:
    """Max drawdown as a fraction of peak equity, over the cumulative pnl
    curve implied by ``pnls`` (oldest → newest).

    Mirrors ``TradingMetricsEngine``'s numpy formula
    (``peak = np.maximum.accumulate(equity)``): the running peak is the
    cummax of the equity curve *including the current point*, so the very
    first point always has zero drawdown, and a losing streak that never
    recovers above its own starting equity does not get compared against a
    phantom zero baseline. Do NOT floor ``peak`` at 0.0 — a loss-first
    sequence's peak can legitimately be negative, matching the reference.
    """
    if not pnls:
        return 0.0
    equity = 0.0
    peak: float | None = None
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = equity if peak is None else max(peak, equity)
        denom = abs(peak) if abs(peak) > 1e-9 else 1.0
        dd = (peak - equity) / denom
        max_dd = max(max_dd, dd)
    return max_dd


def _capital_efficiency(pnls: list[float]) -> float:
    """Total return per unit of capital placed at risk (gross loss), mirroring
    ``TradingMetricsEngine``'s ``capital_efficiency = total_return / gross_loss``.
    """
    total_return = sum(pnls)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    return total_return / (gross_loss + 1e-9)


def _risk_adjusted_return(pnls: list[float]) -> float:
    """Total return divided by max drawdown (a Calmar/MAR-style ratio) —
    reward earned per unit of worst-case capital erosion endured."""
    total_return = sum(pnls)
    max_dd = _max_drawdown(pnls)
    if max_dd < 1e-9:
        return float("inf") if total_return > 0 else 0.0
    return total_return / max_dd


def _sharpe_ratio(
    pnls: list[float],
    annualisation_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """Sharpe ratio, formula-identical to
    ``trainer.validation.trading.TradingMetricsEngine.compute()``'s Sharpe
    step (ported to pure Python — no numpy — for the same import-safety
    reason as the rest of this module).

    Deliberately NOT the same as ``dimension_evaluators._sharpe`` (raw
    mean/std of pnl, no normalization or annualization) — that helper is
    used for a different purpose (baseline-vs-recent *drift* comparison
    where only the direction/magnitude of change matters, not an
    absolute, cross-context-comparable Sharpe value). Live and offline
    Sharpe must agree on the same trade data, since Research Lab
    dashboards compare them side by side.
    """
    if not pnls:
        return 0.0
    # No `len(pnls) < 2` short-circuit: `TradingMetricsEngine` does not
    # special-case a single trade either — with one point, population
    # std is exactly 0.0, so the formula falls through to
    # `mean * sqrt(ann) / (0 + 1e-9)`, an epsilon-stabilized large value.
    # Matching that (rather than returning an arbitrary 0.0) is required
    # for live/offline parity on single-trade windows.
    #
    # `TradingMetricsEngine` normalizes by `max(abs(pnls.max()), 1.0)` — the
    # absolute value of the single largest (most positive) pnl, NOT the
    # largest-magnitude pnl. These two are only equivalent when the biggest
    # winner outweighs the biggest loser; a loss-dominated series (e.g. all
    # losses, or one large loss bigger than any gain) diverges if you use
    # `max(abs(p) for p in pnls)` instead. Must match exactly for parity.
    denom = max(abs(max(pnls)), 1.0)
    returns = [p / denom for p in pnls]
    rfr_per_bar = risk_free_rate / annualisation_factor
    excess = [r - rfr_per_bar for r in returns]
    m = sum(excess) / len(excess)
    var = sum((e - m) ** 2 for e in excess) / len(
        excess
    )  # population std (ddof=0), matches np.std()
    std = var**0.5
    return m * (annualisation_factor**0.5) / (std + 1e-9)


def _cap(value: float, ceiling: float = 999.0) -> float:
    """Cap +inf ratios to a large-but-finite sentinel so reports stay JSON-safe."""
    if value == float("inf"):
        return ceiling
    if value == float("-inf"):
        return -ceiling
    return value


def performance_stats(
    trades: list[dict[str, Any]],
    annualisation_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Compute the six live performance metrics for a group of trades.

    ``annualisation_factor``/``risk_free_rate`` feed ``_sharpe_ratio`` and
    default to ``TradingMetricsEngine``'s own defaults, so live and offline
    Sharpe agree on the same trade data unless a caller deliberately
    overrides them (e.g. to match a specific offline validation run's
    non-default annualisation).
    """
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "risk_adjusted_return": 0.0,
            "capital_efficiency": 0.0,
        }
    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    return {
        "n_trades": n,
        "profit_factor": round(_cap(_profit_factor(pnls)), 4),
        "sharpe_ratio": round(_sharpe_ratio(pnls, annualisation_factor, risk_free_rate), 4),
        "max_drawdown": round(_max_drawdown(pnls), 4),
        "hit_rate": round(_hit_rate(pnls), 4),
        "risk_adjusted_return": round(_cap(_risk_adjusted_return(pnls)), 4),
        "capital_efficiency": round(_cap(_capital_efficiency(pnls)), 4),
    }


def _cohort_of(trade: dict[str, Any]) -> str:
    """User cohort for a trade. Prefers an explicit ``user_cohort``/``cohort``
    key in ``market_conditions`` (set upstream by Signalium); falls back to
    ``account_id`` when present, else "UNKNOWN". No first-class cohort field
    exists on ``TradeEvent`` yet, so this mirrors the volatility-state
    derivation convention in ``decision_quality.py``.
    """
    mc = trade.get("market_conditions") or {}
    for key in ("user_cohort", "cohort"):
        if key in mc:
            return str(mc[key])
    if "account_id" in trade:
        return str(trade["account_id"])
    if "account_id" in mc:
        return str(mc["account_id"])
    return "UNKNOWN"


def _group_performance(
    trades: list[dict[str, Any]],
    key_fn: Any,
    annualisation_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        key = key_fn(t) or "UNKNOWN"
        groups.setdefault(str(key), []).append(t)
    return {
        k: performance_stats(v, annualisation_factor, risk_free_rate) for k, v in groups.items()
    }


# ── Report ────────────────────────────────────────────────────────────────────


@dataclass
class PerformanceReport:
    """A point-in-time autonomous-performance snapshot for a model version."""

    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_trades: int
    overall: dict[str, Any] = field(default_factory=dict)
    performance_by_symbol: dict[str, dict[str, Any]] = field(default_factory=dict)
    performance_by_regime: dict[str, dict[str, Any]] = field(default_factory=dict)
    performance_by_cohort: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_trades": self.n_trades,
            "overall": self.overall,
            "performance_by_symbol": self.performance_by_symbol,
            "performance_by_regime": self.performance_by_regime,
            "performance_by_cohort": self.performance_by_cohort,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerformanceReport:
        return cls(
            report_id=d["report_id"],
            model_version=d.get("model_version", ""),
            created_at=d.get("created_at", 0.0),
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            n_trades=d.get("n_trades", 0),
            overall=d.get("overall", {}),
            performance_by_symbol=d.get("performance_by_symbol", {}),
            performance_by_regime=d.get("performance_by_regime", {}),
            performance_by_cohort=d.get("performance_by_cohort", {}),
        )


# ── Tracker ───────────────────────────────────────────────────────────────────


class PerformanceMonitor:
    """Computes, persists, and queries live autonomous-performance reports.

    Usage::

        monitor = PerformanceMonitor(db, warehouse)
        report = monitor.compute_report(model_version="v1.0")
        latest = monitor.get_latest_report(model_version="v1.0")
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        warehouse: TelemetryWarehouse,
        config: PerformanceMonitorConfig | None = None,
    ) -> None:
        self._db = db
        self._wh = warehouse
        self._cfg = config or PerformanceMonitorConfig()

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> PerformanceReport:
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000)
        trades = sorted(self._wh.query_trades(q), key=lambda t: t.get("timestamp", 0))

        now = time.time()
        window_start = trades[0].get("timestamp", now) if trades else now
        window_end = trades[-1].get("timestamp", now) if trades else now

        ann = self._cfg.annualisation_factor
        rfr = self._cfg.risk_free_rate
        report = PerformanceReport(
            report_id=f"perf-{uuid.uuid4().hex[:12]}",
            model_version=model_version or "",
            created_at=now,
            window_start=window_start,
            window_end=window_end,
            n_trades=len(trades),
            overall=performance_stats(trades, ann, rfr),
            performance_by_symbol=_group_performance(trades, lambda t: t.get("symbol"), ann, rfr),
            performance_by_regime=_group_performance(trades, lambda t: t.get("regime"), ann, rfr),
            performance_by_cohort=_group_performance(trades, _cohort_of, ann, rfr),
        )
        if persist:
            self._db.upsert_performance_report(report.to_dict())
            logger.info(
                f"PerformanceMonitor: computed report {report.report_id} "
                f"for model={model_version!r} n_trades={report.n_trades}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_performance_reports(model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(
        self, model_version: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._db.list_performance_reports(model_version=model_version, limit=limit)


# ── Drift check (wired into health_monitor's alerting path) ─────────────────


def check_performance_drift(
    trades: list[dict[str, Any]],
    cfg: PerformanceMonitorConfig,
    split_windows: Any,
) -> dict[str, Any] | None:
    """Compare baseline vs recent windows' profit factor, Sharpe, drawdown,
    and hit rate; return evidence when performance has degraded significantly.

    ``split_windows`` is injected (the caller's disjoint-window splitter),
    mirroring ``decision_quality.check_decision_quality_drift``'s convention
    so this module has no dependency on ``health_monitor``.

    Returns ``None`` when there isn't enough data or nothing has degraded.
    """
    if len(trades) < cfg.min_samples * 2:
        return None

    baseline_t, recent_t = split_windows(
        trades, cfg.baseline_window, cfg.rolling_window, cfg.min_samples
    )
    if baseline_t is None or recent_t is None:
        return None

    b = performance_stats(baseline_t, cfg.annualisation_factor, cfg.risk_free_rate)
    r = performance_stats(recent_t, cfg.annualisation_factor, cfg.risk_free_rate)

    def _relative_drop(baseline: float, recent: float) -> float:
        if abs(baseline) < 1e-6:
            return 1.0 if recent < baseline - 1e-6 else 0.0
        return max(0.0, (baseline - recent) / abs(baseline))

    pf_drop = _relative_drop(b["profit_factor"], r["profit_factor"])
    sharpe_drop = _relative_drop(b["sharpe_ratio"], r["sharpe_ratio"])
    dd_increase = max(0.0, r["max_drawdown"] - b["max_drawdown"])
    hit_rate_drop = max(0.0, b["hit_rate"] - r["hit_rate"])

    triggers = []
    if pf_drop >= cfg.profit_factor_drift_warning:
        triggers.append("profit_factor")
    if sharpe_drop >= cfg.sharpe_drift_warning:
        triggers.append("sharpe_ratio")
    if dd_increase >= cfg.drawdown_increase_warning:
        triggers.append("max_drawdown")
    if hit_rate_drop >= cfg.hit_rate_drop_warning:
        triggers.append("hit_rate")
    if not triggers:
        return None

    is_critical = (
        pf_drop >= cfg.profit_factor_drift_critical
        or sharpe_drop >= cfg.sharpe_drift_critical
        or dd_increase >= cfg.drawdown_increase_critical
        or hit_rate_drop >= cfg.hit_rate_drop_critical
    )

    return {
        "severity": "CRITICAL" if is_critical else "WARNING",
        "triggers": triggers,
        "evidence": {
            "baseline_profit_factor": b["profit_factor"],
            "recent_profit_factor": r["profit_factor"],
            "profit_factor_relative_drop": round(pf_drop, 4),
            "baseline_sharpe_ratio": b["sharpe_ratio"],
            "recent_sharpe_ratio": r["sharpe_ratio"],
            "sharpe_ratio_relative_drop": round(sharpe_drop, 4),
            "baseline_max_drawdown": b["max_drawdown"],
            "recent_max_drawdown": r["max_drawdown"],
            "max_drawdown_increase": round(dd_increase, 4),
            "baseline_hit_rate": b["hit_rate"],
            "recent_hit_rate": r["hit_rate"],
            "hit_rate_drop": round(hit_rate_drop, 4),
            "n_baseline": len(baseline_t),
            "n_recent": len(recent_t),
        },
    }
