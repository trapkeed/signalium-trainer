"""S22B — Market Regime Coverage Detection.

Detects and distinctly labels three market-coverage conditions from live
decision telemetry:

  1. **Unseen conditions** — regimes appearing in live telemetry that were
     never part of the model's configured "known" training/validation
     regime set (see ``trainer/validation/regime.py``'s ``MarketRegime``
     enum for the canonical set).
  2. **Distribution shift** — the general TV-distance-based regime-histogram
     shift already computed by ``health_monitor._check_market``; this
     module re-derives it independently (small, pure, no cross-module
     import) so it can be reported as a fully queryable, persisted
     ``RegimeCoverageReport`` alongside the other two conditions rather
     than only living as a transient ``DriftAlert``.
  3. **Black swan** — statistically extreme individual moves (trade PnL or
     decision-level return magnitude) whose z-score against the trailing
     baseline distribution exceeds a configured threshold. Distinct from
     ordinary distribution shift: a single black-swan event can occur
     inside an otherwise-stable regime distribution, and a regime shift can
     happen with no black-swan moves at all — the two conditions are
     independent and must not be conflated into one signal.

Pure-Python (no numpy/torch) — reuses ``dimension_evaluators._mean/_std/_tv_distance``
for the same import-safety reason as ``decision_quality.py``/``performance_monitor.py``
(see memory: torch-broken-lazy-registry-import).
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

from trainer.research_lab.dimension_evaluators import _mean, _split_windows, _std
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryQuery, TelemetryWarehouse

# The canonical regime vocabulary a model is expected to have been
# trained/validated against (mirrors trainer/validation/regime.py's
# MarketRegime enum values). Duplicated as plain strings here rather than
# imported, since trainer.validation.regime pulls in numpy/torch and this
# module must stay import-safe in the torch-broken environment (see memory:
# torch-broken-lazy-registry-import). Any regime label observed in live
# telemetry that is NOT in this set (or a caller-supplied override) is an
# "unseen condition" by definition.
DEFAULT_KNOWN_REGIMES: frozenset[str] = frozenset(
    {
        "trending_strong", "trending_weak", "ranging", "breakout", "pullback",
        "high_volatility", "low_volatility", "compression", "expansion",
        "accumulation", "distribution", "news_period", "session_open",
        "session_close", "weekend_gap",
    }
)


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class RegimeCoverageConfig:
    """Thresholds for the three regime-coverage conditions."""

    baseline_window: int = 200
    rolling_window: int = 50
    min_samples: int = 10

    # 1. Unseen conditions: known_regimes=None falls back to
    # DEFAULT_KNOWN_REGIMES. A fraction of recent decisions whose regime
    # label isn't in the known set, at/above these thresholds, fires a
    # warning/critical unseen-condition alert.
    known_regimes: frozenset[str] | None = None
    unseen_condition_rate_warning: float = 0.05
    unseen_condition_rate_critical: float = 0.20

    # 2. Distribution shift: TV distance between baseline/recent regime
    # histograms (same thresholds/semantics as health_monitor's MARKET dim).
    distribution_shift_warning: float = 0.25
    distribution_shift_critical: float = 0.45

    # 3. Black swan: z-score threshold for an individual pnl/return to be
    # considered "extreme", plus the fraction of recent trades that must
    # cross it before the condition is reported at all.
    black_swan_zscore_threshold: float = 4.0
    black_swan_rate_warning: float = 0.02
    black_swan_rate_critical: float = 0.08


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two distributions.

    TV = 0.5 * sum |p_i - q_i| — symmetric, bounded in [0, 1]. Duplicated
    verbatim from health_monitor.py's private helper (rather than
    dimension_evaluators' differently-scaled same-named function) so this
    module's drift scores stay numerically consistent with health_monitor's
    own MARKET-dimension TV distance for the same telemetry — the two must
    never silently disagree on what counts as "shifted".
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _regime_histogram(records: list[dict[str, Any]]) -> dict[str, float]:
    """Regime frequency distribution from a list of decision/trade dicts.

    Regime labels are lowercased before counting — telemetry across the
    codebase is inconsistent about casing (e.g. "RANGING" vs "ranging"),
    and casing alone must never manufacture a false distribution shift, the
    same principle applied to unseen-condition detection below.
    """
    n = len(records)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for r in records:
        regime = (r.get("regime", "") or "").lower()
        if regime:
            counts[regime] = counts.get(regime, 0) + 1
    if not counts:
        return {}
    return {k: v / n for k, v in counts.items()}


def detect_unseen_conditions(
    records: list[dict[str, Any]], known_regimes: frozenset[str] | None
) -> dict[str, Any]:
    """Fraction + breakdown of records whose regime label is outside the
    known/trained regime vocabulary.

    Comparison is case-insensitive: telemetry across the codebase is
    inconsistent about regime-label casing (e.g. "RANGING" vs "ranging"),
    and casing alone must never manufacture a false unseen-condition alert.
    """
    known = known_regimes if known_regimes is not None else DEFAULT_KNOWN_REGIMES
    known_lower = {k.lower() for k in known}
    n = len(records)
    if n == 0:
        return {"unseen_rate": 0.0, "unseen_regimes": {}, "n_total": 0}

    unseen_counts: dict[str, int] = {}
    for r in records:
        regime = r.get("regime", "") or ""
        if regime and regime.lower() not in known_lower:
            unseen_counts[regime] = unseen_counts.get(regime, 0) + 1

    n_unseen = sum(unseen_counts.values())
    return {
        "unseen_rate": round(n_unseen / n, 4),
        "unseen_regimes": unseen_counts,
        "n_total": n,
    }


def detect_distribution_shift(
    baseline: list[dict[str, Any]], recent: list[dict[str, Any]]
) -> dict[str, Any]:
    """TV-distance-based regime histogram shift between two windows."""
    b_hist = _regime_histogram(baseline)
    r_hist = _regime_histogram(recent)
    if not b_hist or not r_hist:
        return {"tv_distance": 0.0, "baseline_dist": b_hist, "recent_dist": r_hist}
    return {
        "tv_distance": round(_tv_distance(b_hist, r_hist), 4),
        "baseline_dist": {k: round(v, 4) for k, v in b_hist.items()},
        "recent_dist": {k: round(v, 4) for k, v in r_hist.items()},
    }


def detect_black_swans(
    records: list[dict[str, Any]], baseline: list[dict[str, Any]], zscore_threshold: float
) -> dict[str, Any]:
    """Detect statistically extreme individual moves in ``records`` (pnl,
    falling back to pnl_pct), scored against ``baseline``'s own mean/std —
    NOT against ``records`` itself, so a black-swan move can't inflate the
    very baseline it's being measured against."""
    baseline_vals = [
        float(r.get("pnl", r.get("pnl_pct", 0.0))) for r in baseline
    ]
    if len(baseline_vals) < 2:
        return {"black_swan_rate": 0.0, "black_swan_events": [], "n_total": len(records)}

    mu = _mean(baseline_vals)
    sigma = _std(baseline_vals)
    if sigma < 1e-9:
        return {"black_swan_rate": 0.0, "black_swan_events": [], "n_total": len(records)}

    events = []
    for r in records:
        val = float(r.get("pnl", r.get("pnl_pct", 0.0)))
        z = (val - mu) / sigma
        if abs(z) >= zscore_threshold:
            events.append({
                "timestamp": r.get("timestamp"),
                "symbol": r.get("symbol"),
                "value": round(val, 4),
                "zscore": round(z, 3),
            })

    n = len(records)
    return {
        "black_swan_rate": round(len(events) / n, 4) if n else 0.0,
        "black_swan_events": events[:10],  # cap evidence payload size
        "n_total": n,
        "baseline_mean": round(mu, 4),
        "baseline_std": round(sigma, 4),
    }


# ── Report ────────────────────────────────────────────────────────────────────


@dataclass
class RegimeCoverageReport:
    """A point-in-time regime-coverage snapshot for a model version,
    covering all three distinct conditions."""

    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_records: int
    unseen_conditions: dict[str, Any] = field(default_factory=dict)
    distribution_shift: dict[str, Any] = field(default_factory=dict)
    black_swans: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_records": self.n_records,
            "unseen_conditions": self.unseen_conditions,
            "distribution_shift": self.distribution_shift,
            "black_swans": self.black_swans,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegimeCoverageReport:
        return cls(
            report_id=d["report_id"],
            model_version=d.get("model_version", ""),
            created_at=d.get("created_at", 0.0),
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            n_records=d.get("n_records", 0),
            unseen_conditions=d.get("unseen_conditions", {}),
            distribution_shift=d.get("distribution_shift", {}),
            black_swans=d.get("black_swans", {}),
        )


def build_regime_coverage_report(
    records: list[dict[str, Any]],
    cfg: RegimeCoverageConfig,
    model_version: str = "",
    report_id: str | None = None,
) -> RegimeCoverageReport:
    """Pure computation of a full :class:`RegimeCoverageReport`. No I/O —
    callers persist separately."""
    now = time.time()
    window_start = records[0].get("timestamp", now) if records else now
    window_end = records[-1].get("timestamp", now) if records else now

    baseline, recent = _split_windows(
        records, cfg.baseline_window, cfg.rolling_window, cfg.min_samples
    )
    # Unseen-condition detection runs over the most recent window when
    # available (that's "live" coverage), else the full record set when
    # there isn't enough history to split — unseen-vocabulary membership
    # doesn't depend on a baseline, so this fallback is safe.
    recent_for_unseen = recent if recent is not None else records

    return RegimeCoverageReport(
        report_id=report_id or f"regc-{uuid.uuid4().hex[:12]}",
        model_version=model_version,
        created_at=now,
        window_start=window_start,
        window_end=window_end,
        n_records=len(records),
        unseen_conditions=detect_unseen_conditions(recent_for_unseen, cfg.known_regimes),
        distribution_shift=(
            detect_distribution_shift(baseline, recent)
            if baseline is not None and recent is not None
            else {"tv_distance": 0.0, "baseline_dist": {}, "recent_dist": {}}
        ),
        # Black-swan scoring requires a genuine, disjoint baseline — unlike
        # unseen-condition detection, falling back to `records` for both
        # sides here would let the extreme value contaminate its own
        # baseline mean/std and silently suppress its own z-score. When
        # there isn't enough history to split, report "no data" rather than
        # a contaminated result.
        black_swans=(
            detect_black_swans(recent, baseline, cfg.black_swan_zscore_threshold)
            if baseline is not None and recent is not None
            else {"black_swan_rate": 0.0, "black_swan_events": [], "n_total": len(records)}
        ),
    )


# ── Tracker ───────────────────────────────────────────────────────────────────


class RegimeCoverageMonitor:
    """Computes, persists, and queries regime-coverage reports.

    Usage::

        monitor = RegimeCoverageMonitor(db, warehouse)
        report = monitor.compute_report(model_version="v1.0")
        latest = monitor.get_latest_report(model_version="v1.0")
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        warehouse: TelemetryWarehouse,
        config: RegimeCoverageConfig | None = None,
    ) -> None:
        self._db = db
        self._wh = warehouse
        self._cfg = config or RegimeCoverageConfig()

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> RegimeCoverageReport:
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000)
        # Decisions carry regime labels; trades carry regime + realized pnl
        # (needed for black-swan detection). Prefer trades when present
        # (they have both regime and pnl); fall back to decisions.
        trades = sorted(self._wh.query_trades(q), key=lambda t: t.get("timestamp", 0))
        records = trades if trades else sorted(
            self._wh.query_decisions(q), key=lambda d: d.get("timestamp", 0)
        )

        report = build_regime_coverage_report(
            records, self._cfg, model_version=model_version or ""
        )
        if persist:
            self._db.upsert_regime_coverage_report(report.to_dict())
            logger.info(
                f"RegimeCoverageMonitor: computed report {report.report_id} "
                f"for model={model_version!r} n_records={report.n_records} "
                f"unseen_rate={report.unseen_conditions.get('unseen_rate')} "
                f"tv_distance={report.distribution_shift.get('tv_distance')} "
                f"black_swan_rate={report.black_swans.get('black_swan_rate')}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_regime_coverage_reports(model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(
        self, model_version: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._db.list_regime_coverage_reports(model_version=model_version, limit=limit)


# ── Drift/alert checks (wired into health_monitor's alerting path) ──────────


def check_unseen_conditions(
    records: list[dict[str, Any]], cfg: RegimeCoverageConfig
) -> dict[str, Any] | None:
    """Absolute-threshold check: fraction of recent records whose regime is
    outside the known vocabulary. Returns ``None`` when below warning."""
    if len(records) < cfg.min_samples:
        return None
    result = detect_unseen_conditions(records, cfg.known_regimes)
    rate = result["unseen_rate"]
    if rate < cfg.unseen_condition_rate_warning:
        return None
    severity = (
        "CRITICAL" if rate >= cfg.unseen_condition_rate_critical else "WARNING"
    )
    return {"severity": severity, "kind": "UNSEEN_CONDITION", "evidence": result}


def check_distribution_shift(
    records: list[dict[str, Any]], cfg: RegimeCoverageConfig
) -> dict[str, Any] | None:
    """Baseline-vs-recent drift check: TV distance between regime
    histograms. Returns ``None`` when below warning or insufficient data."""
    baseline, recent = _split_windows(
        records, cfg.baseline_window, cfg.rolling_window, cfg.min_samples
    )
    if baseline is None or recent is None:
        return None
    result = detect_distribution_shift(baseline, recent)
    tv = result["tv_distance"]
    if tv < cfg.distribution_shift_warning:
        return None
    severity = (
        "CRITICAL" if tv >= cfg.distribution_shift_critical else "WARNING"
    )
    return {"severity": severity, "kind": "DISTRIBUTION_SHIFT", "evidence": result}


def check_black_swans(
    records: list[dict[str, Any]], cfg: RegimeCoverageConfig
) -> dict[str, Any] | None:
    """Absolute-threshold check: fraction of recent records that are
    statistically extreme relative to the baseline window. Returns ``None``
    when below warning or insufficient baseline data."""
    baseline, recent = _split_windows(
        records, cfg.baseline_window, cfg.rolling_window, cfg.min_samples
    )
    if baseline is None or recent is None:
        return None
    result = detect_black_swans(recent, baseline, cfg.black_swan_zscore_threshold)
    rate = result["black_swan_rate"]
    if rate < cfg.black_swan_rate_warning:
        return None
    severity = (
        "CRITICAL" if rate >= cfg.black_swan_rate_critical else "WARNING"
    )
    return {"severity": severity, "kind": "BLACK_SWAN", "evidence": result}
