"""S22B — Feature Health Monitoring.

Detects missing, stale, corrupted, and delayed features in the QDE
inference pipeline, using ``ReasoningEvent.feature_values`` telemetry
(the input features recorded at each decision's inference time).

Distinct from the existing ``DATA_QUALITY`` dimension in
``health_monitor.py`` (which covers overall telemetry staleness/event-rate
drop across all decisions) — this module is *per-feature*, checking each
individual input feature's presence, freshness, validity, and delivery
delay, independent of whether telemetry is flowing overall.

Pure-Python (no numpy/torch) for the same import-safety reason as
``decision_quality.py``/``performance_monitor.py`` (see memory:
torch-broken-lazy-registry-import).
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
class FeatureHealthConfig:
    """Thresholds for per-feature health checks.

    ``expected_features``: the full set of feature names the pipeline should
    be emitting. ``None`` means "whatever the union of observed features is"
    (missing-feature detection is then a no-op — there's nothing to compare
    against). ``value_bounds``: optional per-feature (min, max) sanity range;
    features without an entry are still checked for NaN/inf/non-numeric.
    """

    expected_features: list[str] | None = None
    value_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Staleness: seconds since a feature was last seen in any event, measured
    # from the window's latest event timestamp (not wall-clock "now"), so
    # checks are deterministic against historical telemetry too.
    staleness_warning_s: float = 3600.0
    staleness_critical_s: float = 21600.0

    # Delivery delay: ingested_at - timestamp, per event.
    delay_warning_s: float = 30.0
    delay_critical_s: float = 120.0

    # Corruption rate (fraction of observations for a feature that are
    # NaN/inf/non-numeric/out-of-bounds) before flagging.
    corruption_rate_warning: float = 0.05
    corruption_rate_critical: float = 0.20

    # Intermittent missingness: fraction of *events in the window* that a
    # given feature is present in. Distinct from ``missing_features``
    # (a feature from ``expected_features`` never observed at all, which is
    # always CRITICAL) — this catches a feature that mostly-but-not-always
    # shows up (e.g. an upstream source flaking intermittently), which the
    # expected-vs-observed set difference cannot see since the feature IS
    # in the observed set. presence_rate below `*_warning` flags WARNING,
    # below `*_critical` flags CRITICAL.
    missing_rate_warning: float = 0.95   # must be present in >= 95% of events
    missing_rate_critical: float = 0.80  # present in < 80% of events is CRITICAL

    # Reasoning-event stream freshness, measured against wall-clock time
    # (unlike per-feature `staleness_*_s`, which is window-relative and so
    # cannot detect the whole reasoning stream going dark — see
    # `check_feature_health`'s `now` parameter). Complements, rather than
    # duplicates, the DATA_QUALITY dimension's staleness check, which only
    # looks at *decision* telemetry — reasoning-event ingestion can stall
    # independently of decisions still flowing.
    stream_staleness_warning_s: float = 3600.0
    stream_staleness_critical_s: float = 14400.0


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _is_corrupted(value: Any, bounds: tuple[float, float] | None) -> bool:
    """True if a feature value is malformed, non-finite, or out of its
    configured sanity range."""
    if isinstance(value, bool):
        return False  # bools are valid 0/1-style features
    if not isinstance(value, (int, float)):
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if bounds is not None:
        lo, hi = bounds
        if value < lo or value > hi:
            return True
    return False


@dataclass
class _FeatureStats:
    n_present: int = 0
    n_corrupted: int = 0
    first_seen: float | None = None
    last_seen: float | None = None
    delay_sum: float = 0.0
    delay_max: float = 0.0
    corrupted_examples: list[Any] = field(default_factory=list)

    def to_dict(self, window_end: float, cfg: FeatureHealthConfig) -> dict[str, Any]:
        staleness_s = (window_end - self.last_seen) if self.last_seen is not None else None
        avg_delay = self.delay_sum / self.n_present if self.n_present else 0.0
        corruption_rate = self.n_corrupted / self.n_present if self.n_present else 0.0
        return {
            "n_present": self.n_present,
            "n_corrupted": self.n_corrupted,
            "corruption_rate": round(corruption_rate, 4),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "staleness_s": round(staleness_s, 1) if staleness_s is not None else None,
            "avg_delay_s": round(avg_delay, 3),
            "max_delay_s": round(self.delay_max, 3),
            "corrupted_examples": self.corrupted_examples[:3],
        }


def compute_feature_stats(
    events: list[dict[str, Any]], cfg: FeatureHealthConfig
) -> tuple[dict[str, _FeatureStats], float, float]:
    """Aggregate per-feature stats across a list of reasoning-event dicts.

    Returns ``(stats_by_feature, window_start, window_end)``.
    """
    stats: dict[str, _FeatureStats] = {}
    window_start = float("inf")
    window_end = 0.0

    for evt in events:
        ts = float(evt.get("timestamp", 0.0))
        ingested_at = float(evt.get("ingested_at", ts))
        delay = max(0.0, ingested_at - ts)
        window_start = min(window_start, ts)
        window_end = max(window_end, ts)

        feature_values = evt.get("feature_values") or {}
        for name, value in feature_values.items():
            fs = stats.setdefault(name, _FeatureStats())
            fs.n_present += 1
            fs.first_seen = ts if fs.first_seen is None else min(fs.first_seen, ts)
            fs.last_seen = ts if fs.last_seen is None else max(fs.last_seen, ts)
            fs.delay_sum += delay
            fs.delay_max = max(fs.delay_max, delay)
            bounds = cfg.value_bounds.get(name)
            if _is_corrupted(value, bounds):
                fs.n_corrupted += 1
                if len(fs.corrupted_examples) < 3:
                    fs.corrupted_examples.append(value)

    if window_start == float("inf"):
        window_start = 0.0
    return stats, window_start, window_end


# ── Report ────────────────────────────────────────────────────────────────────


@dataclass
class FeatureHealthReport:
    """A point-in-time per-feature health snapshot for a model version."""

    report_id: str
    model_version: str
    created_at: float
    window_start: float
    window_end: float
    n_events: int
    per_feature: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_features: list[str] = field(default_factory=list)
    stale_features: list[str] = field(default_factory=list)
    corrupted_features: list[str] = field(default_factory=list)
    delayed_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "model_version": self.model_version,
            "created_at": round(self.created_at, 3),
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "n_events": self.n_events,
            "per_feature": self.per_feature,
            "missing_features": self.missing_features,
            "stale_features": self.stale_features,
            "corrupted_features": self.corrupted_features,
            "delayed_features": self.delayed_features,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureHealthReport:
        return cls(
            report_id=d["report_id"],
            model_version=d.get("model_version", ""),
            created_at=d.get("created_at", 0.0),
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            n_events=d.get("n_events", 0),
            per_feature=d.get("per_feature", {}),
            missing_features=d.get("missing_features", []),
            stale_features=d.get("stale_features", []),
            corrupted_features=d.get("corrupted_features", []),
            delayed_features=d.get("delayed_features", []),
        )


def build_feature_health_report(
    events: list[dict[str, Any]],
    cfg: FeatureHealthConfig,
    model_version: str = "",
    report_id: str | None = None,
) -> FeatureHealthReport:
    """Pure computation of a full :class:`FeatureHealthReport` from a batch
    of reasoning-event dicts. No I/O — callers persist separately."""
    now = time.time()
    stats, window_start, window_end = compute_feature_stats(events, cfg)
    n_events = len(events)

    per_feature = {}
    for name, fs in stats.items():
        d = fs.to_dict(window_end, cfg)
        d["presence_rate"] = round(fs.n_present / n_events, 4) if n_events else 0.0
        per_feature[name] = d

    observed = set(stats.keys())
    expected = set(cfg.expected_features) if cfg.expected_features else set()
    # Entirely absent (never observed at all) — always the most severe case,
    # handled distinctly from intermittent missingness below.
    missing = set(expected - observed)

    stale, corrupted, delayed = [], [], []
    for name, fs in stats.items():
        staleness_s = (window_end - fs.last_seen) if fs.last_seen is not None else None
        if staleness_s is not None and staleness_s >= cfg.staleness_warning_s:
            stale.append(name)
        corruption_rate = fs.n_corrupted / fs.n_present if fs.n_present else 0.0
        if corruption_rate >= cfg.corruption_rate_warning:
            corrupted.append(name)
        avg_delay = fs.delay_sum / fs.n_present if fs.n_present else 0.0
        if avg_delay >= cfg.delay_warning_s:
            delayed.append(name)
        # Intermittent missingness: present in the observed set (so the
        # expected/observed set-difference above can't catch it) but absent
        # from a meaningful fraction of events in the window.
        presence_rate = fs.n_present / n_events if n_events else 1.0
        if presence_rate < cfg.missing_rate_warning:
            missing.add(name)

    return FeatureHealthReport(
        report_id=report_id or f"feat-{uuid.uuid4().hex[:12]}",
        model_version=model_version,
        created_at=now,
        window_start=window_start,
        window_end=window_end,
        n_events=n_events,
        per_feature=per_feature,
        missing_features=sorted(missing),
        stale_features=sorted(stale),
        corrupted_features=sorted(corrupted),
        delayed_features=sorted(delayed),
    )


# ── Tracker ───────────────────────────────────────────────────────────────────


class FeatureHealthMonitor:
    """Computes, persists, and queries feature-health reports.

    Usage::

        monitor = FeatureHealthMonitor(db, warehouse)
        report = monitor.compute_report(model_version="v1.0")
        latest = monitor.get_latest_report(model_version="v1.0")
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        warehouse: TelemetryWarehouse,
        config: FeatureHealthConfig | None = None,
    ) -> None:
        self._db = db
        self._wh = warehouse
        self._cfg = config or FeatureHealthConfig()

    def compute_report(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        persist: bool = True,
    ) -> FeatureHealthReport:
        q = TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000)
        events = sorted(self._wh.query_reasoning(q), key=lambda e: e.get("timestamp", 0))

        report = build_feature_health_report(
            events, self._cfg, model_version=model_version or ""
        )
        if persist:
            self._db.upsert_feature_health_report(report.to_dict())
            logger.info(
                f"FeatureHealthMonitor: computed report {report.report_id} "
                f"for model={model_version!r} n_events={report.n_events} "
                f"missing={report.missing_features} stale={report.stale_features} "
                f"corrupted={report.corrupted_features}"
            )
        return report

    def get_latest_report(self, model_version: str | None = None) -> dict[str, Any] | None:
        reports = self._db.list_feature_health_reports(model_version=model_version, limit=1)
        return reports[0] if reports else None

    def list_reports(
        self, model_version: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        return self._db.list_feature_health_reports(model_version=model_version, limit=limit)


# ── Drift/alert check (wired into health_monitor's alerting path) ───────────


def check_feature_health(
    events: list[dict[str, Any]],
    cfg: FeatureHealthConfig,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Evaluate absolute severity thresholds (not baseline-vs-recent drift —
    missing/stale/corrupted/delayed are each judged against a fixed
    threshold, per the task's "clear severity threshold per condition")
    and return evidence when any condition is at WARNING or above.

    ``now``: wall-clock time for the reasoning-*stream*-level freshness
    check (distinct from per-feature ``staleness_*_s``, which is measured
    relative to the window's own latest event and so cannot detect the
    whole reasoning stream going dark while decisions keep flowing — that
    gap is what this parameter closes). Defaults to ``time.time()``.

    Returns ``None`` when there's no data or nothing crosses a threshold.
    An empty ``events`` list with an explicit ``now`` still cannot be
    evaluated for stream staleness here (there's no last-seen timestamp to
    compare against) — that case is the reasoning-side analogue of
    DATA_QUALITY's "zero events" check and is left to the caller/health
    monitor's overall telemetry-completeness handling.
    """
    if not events:
        return None

    now = time.time() if now is None else now
    report = build_feature_health_report(events, cfg)
    stats, _, window_end = compute_feature_stats(events, cfg)
    expected = set(cfg.expected_features) if cfg.expected_features else set()
    observed = set(stats.keys())
    entirely_missing = expected - observed

    triggers: list[str] = []
    is_critical = False

    for name in report.missing_features:
        if name in entirely_missing:
            triggers.append(f"missing:{name}")
            # Missing entirely is always at least as severe as a critical
            # staleness/corruption breach — there is no data to fall back on.
            is_critical = True
        else:
            fs = stats[name]
            presence_rate = fs.n_present / report.n_events if report.n_events else 1.0
            triggers.append(f"intermittent_missing:{name}")
            if presence_rate < cfg.missing_rate_critical:
                is_critical = True

    for name in report.stale_features:
        fs = stats[name]
        staleness_s = window_end - fs.last_seen if fs.last_seen is not None else float("inf")
        triggers.append(f"stale:{name}")
        if staleness_s >= cfg.staleness_critical_s:
            is_critical = True

    for name in report.corrupted_features:
        fs = stats[name]
        corruption_rate = fs.n_corrupted / fs.n_present if fs.n_present else 0.0
        triggers.append(f"corrupted:{name}")
        if corruption_rate >= cfg.corruption_rate_critical:
            is_critical = True

    for name in report.delayed_features:
        fs = stats[name]
        avg_delay = fs.delay_sum / fs.n_present if fs.n_present else 0.0
        triggers.append(f"delayed:{name}")
        if avg_delay >= cfg.delay_critical_s:
            is_critical = True

    # Stream-level freshness: the whole reasoning-event feed may have gone
    # dark (window_end far in the past relative to wall-clock now), which
    # per-feature window-relative staleness cannot see by construction.
    stream_staleness_s = now - window_end
    if stream_staleness_s >= cfg.stream_staleness_warning_s:
        triggers.append("stream_stale")
        if stream_staleness_s >= cfg.stream_staleness_critical_s:
            is_critical = True

    if not triggers:
        return None

    return {
        "severity": "CRITICAL" if is_critical else "WARNING",
        "triggers": triggers,
        "evidence": {
            "n_events": report.n_events,
            "missing_features": report.missing_features,
            "stale_features": report.stale_features,
            "corrupted_features": report.corrupted_features,
            "delayed_features": report.delayed_features,
            "per_feature": report.per_feature,
            "stream_staleness_s": round(stream_staleness_s, 1),
        },
    }
