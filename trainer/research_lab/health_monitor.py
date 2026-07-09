"""S20 — Research Lab Model Health Monitor.

Continuously monitors model health across five drift dimensions:

  1. **Feature drift**     — input distribution shift (buy/sell/no_trade probs)
  2. **Confidence drift**  — output calibration shift (mean confidence delta)
  3. **Market drift**      — regime/volatility distribution shift
  4. **Data quality**      — telemetry completeness and freshness
  5. **Regime drift**      — per-regime win-rate performance delta

Each dimension emits a ``DriftAlert`` with severity WARNING or CRITICAL.
CRITICAL alerts automatically create a ``RetrainingRecommendation`` (pending
admin approval) and are deduplicated against open recommendations for the
same (model_version, dimension) pair.

Alerts are persisted in the Research Lab DB and returned by
``GET /health/alerts``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.dimension_evaluators import _split_windows
from trainer.research_lab.recommendation_models import (
    EvaluationDimension,
    EvaluationResult,
    RecommendationConfig,
    RecommendedScope,
    RecommendationStatus,
    RecommendationUrgency,
    RetrainingRecommendation,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.telemetry_warehouse import TelemetryQuery, TelemetryWarehouse


# ── Enumerations ──────────────────────────────────────────────────────────────


class DriftDimension(str, Enum):
    FEATURE = "FEATURE"
    CONFIDENCE = "CONFIDENCE"
    MARKET = "MARKET"
    DATA_QUALITY = "DATA_QUALITY"
    REGIME = "REGIME"


class AlertSeverity(str, Enum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# Mapping from DriftDimension to the EvaluationDimension used in recommendations
_DIM_TO_EVAL: dict[DriftDimension, EvaluationDimension] = {
    DriftDimension.FEATURE: EvaluationDimension.REASONING_DRIFT,
    DriftDimension.CONFIDENCE: EvaluationDimension.CONFIDENCE_DRIFT,
    DriftDimension.MARKET: EvaluationDimension.REGIME_UNDERPERFORMANCE,
    DriftDimension.DATA_QUALITY: EvaluationDimension.SHARPE_DEGRADATION,
    DriftDimension.REGIME: EvaluationDimension.REGIME_UNDERPERFORMANCE,
}


# ── DriftAlert ────────────────────────────────────────────────────────────────


@dataclass
class DriftAlert:
    """One drift alert produced by the health monitor.

    Attributes:
        alert_id: Unique identifier (``hal-*``).
        dimension: One of the five DriftDimension values.
        severity: WARNING or CRITICAL.
        model_version: The model version that triggered the alert.
        affected_symbols: Symbols with the highest drift contribution.
        affected_features: Feature names (where applicable).
        evidence: Quantitative evidence dict (dimension-specific).
        recommended_action: Human-readable recommended next step.
        created_at: Unix timestamp.
        recommendation_id: Set when a CRITICAL alert creates a
            RetrainingRecommendation.
    """

    alert_id: str
    dimension: str
    severity: str
    model_version: str
    affected_symbols: list[str]
    affected_features: list[str]
    evidence: dict[str, Any]
    recommended_action: str
    created_at: float
    recommendation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "dimension": self.dimension,
            "severity": self.severity,
            "model_version": self.model_version,
            "affected_symbols": self.affected_symbols,
            "affected_features": self.affected_features,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "created_at": round(self.created_at, 3),
            "recommendation_id": self.recommendation_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DriftAlert:
        return cls(
            alert_id=d["alert_id"],
            dimension=d["dimension"],
            severity=d["severity"],
            model_version=d.get("model_version", ""),
            affected_symbols=d.get("affected_symbols", []),
            affected_features=d.get("affected_features", []),
            evidence=d.get("evidence", {}),
            recommended_action=d.get("recommended_action", ""),
            created_at=d.get("created_at", 0.0),
            recommendation_id=d.get("recommendation_id", ""),
        )


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class HealthMonitorConfig:
    """Configurable thresholds for all five drift dimensions.

    All thresholds can be overridden without redeploying by passing a
    custom config instance to ``ResearchLabHealthMonitor``.
    """

    # Window sizes (number of decisions/trades)
    baseline_window: int = 200
    rolling_window: int = 50
    min_samples: int = 10

    # ── 1. Feature drift (output probability distribution TV distance) ─────────
    feature_warning: float = 0.08   # total-variation distance warning
    feature_critical: float = 0.16

    # ── 2. Confidence drift (|mean_recent - mean_baseline|) ────────────────────
    confidence_warning: float = 0.06
    confidence_critical: float = 0.12

    # ── 3. Market / regime distribution drift (TV on regime histogram) ─────────
    market_warning: float = 0.12
    market_critical: float = 0.25

    # ── 4. Data quality (staleness + event-rate drop) ──────────────────────────
    staleness_warning_s: float = 3_600.0   # 1 hour with no new events
    staleness_critical_s: float = 14_400.0  # 4 hours with no new events
    event_rate_drop_warning: float = 0.50  # 50 % drop in events-per-unit-time
    event_rate_drop_critical: float = 0.80  # 80 % drop

    # ── 5. Regime performance drift (win-rate drop per regime) ─────────────────
    regime_win_rate_warning: float = 0.08   # absolute drop in win rate
    regime_win_rate_critical: float = 0.15


# ── Health Check Result ───────────────────────────────────────────────────────


@dataclass
class HealthCheckResult:
    """Summary returned by a single ``run_check()`` invocation."""

    model_version: str
    alerts: list[DriftAlert]
    is_healthy: bool
    checked_at: float
    dimensions_checked: list[str]
    dimensions_healthy: list[str]
    dimensions_alerting: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "is_healthy": self.is_healthy,
            "checked_at": round(self.checked_at, 3),
            "n_alerts": len(self.alerts),
            "dimensions_checked": self.dimensions_checked,
            "dimensions_healthy": self.dimensions_healthy,
            "dimensions_alerting": self.dimensions_alerting,
            "alerts": [a.to_dict() for a in self.alerts],
        }


# ── Health Monitor ────────────────────────────────────────────────────────────


class ResearchLabHealthMonitor:
    """Orchestrates five-dimension model health monitoring for the Research Lab.

    Args:
        db: ResearchLabDatabase for alert persistence and recommendation
            deduplication.
        warehouse: TelemetryWarehouse as the data source for all checks.
        rec_engine: Optional RetrainingRecommendationEngine; when provided,
            CRITICAL alerts automatically create RetrainingRecommendations.
        config: Configurable thresholds; defaults to ``HealthMonitorConfig()``.
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        warehouse: TelemetryWarehouse,
        rec_engine: Any | None = None,
        config: HealthMonitorConfig | None = None,
    ) -> None:
        self._db = db
        self._wh = warehouse
        self._rec_engine = rec_engine
        self._cfg = config or HealthMonitorConfig()

    # ── Public: run a full health check ───────────────────────────────────────

    def run_check(
        self,
        model_version: str | None = None,
        since: float | None = None,
    ) -> HealthCheckResult:
        """Run all five drift-dimension checks and persist any alerts.

        Args:
            model_version: Restrict to a specific model version string.
                ``None`` evaluates all available data.
            since: Unix timestamp; restrict telemetry to events after this time.
        """
        mv = model_version or ""
        now = time.time()
        q = TelemetryQuery(model_version=model_version or None, since=since)

        decisions = sorted(
            self._wh.query_decisions(q), key=lambda d: d.get("timestamp", 0)
        )
        trades = sorted(
            self._wh.query_trades(q), key=lambda t: t.get("timestamp", 0)
        )

        alerts: list[DriftAlert] = []
        checked = []

        for name, fn, args in [
            (DriftDimension.CONFIDENCE.value, self._check_confidence, (decisions, mv)),
            (DriftDimension.FEATURE.value, self._check_feature, (decisions, mv)),
            (DriftDimension.MARKET.value, self._check_market, (decisions, mv)),
            (DriftDimension.DATA_QUALITY.value, self._check_data_quality, (decisions, mv, now)),
            (DriftDimension.REGIME.value, self._check_regime, (trades, mv)),
        ]:
            checked.append(name)
            try:
                alert = fn(*args)
                if alert is not None:
                    # Persist to DB
                    self._db.upsert_health_alert(alert.to_dict())
                    # Wire CRITICAL → RetrainingRecommendation
                    if alert.severity == AlertSeverity.CRITICAL.value:
                        rec_id = self._maybe_create_recommendation(alert)
                        if rec_id:
                            alert.recommendation_id = rec_id
                            self._db.upsert_health_alert(alert.to_dict())
                    alerts.append(alert)
                    logger.warning(
                        f"HealthMonitor: {name} {alert.severity} for model={mv!r} "
                        f"alert={alert.alert_id}"
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"HealthMonitor: {name} check raised {exc!r} — skipping")

        alerting_dims = [a.dimension for a in alerts]
        healthy_dims = [d for d in checked if d not in alerting_dims]

        return HealthCheckResult(
            model_version=mv,
            alerts=alerts,
            is_healthy=len(alerts) == 0,
            checked_at=now,
            dimensions_checked=checked,
            dimensions_healthy=healthy_dims,
            dimensions_alerting=alerting_dims,
        )

    # ── Public: query ─────────────────────────────────────────────────────────

    def get_alerts(
        self,
        model_version: str | None = None,
        severity: str | None = None,
        dimension: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return persisted alerts, newest first."""
        return self._db.list_health_alerts(
            model_version=model_version,
            severity=severity,
            dimension=dimension,
            limit=limit,
        )

    def get_status(self, model_version: str | None = None) -> dict[str, Any]:
        """Return a current health status summary without re-running checks."""
        recent_alerts = self._db.list_health_alerts(
            model_version=model_version, limit=200
        )
        n_critical = sum(
            1 for a in recent_alerts if a.get("severity") == AlertSeverity.CRITICAL.value
        )
        n_warning = sum(
            1 for a in recent_alerts if a.get("severity") == AlertSeverity.WARNING.value
        )
        by_dimension: dict[str, dict[str, int]] = {}
        for a in recent_alerts:
            dim = a.get("dimension", "UNKNOWN")
            sev = a.get("severity", "UNKNOWN")
            by_dimension.setdefault(dim, {})
            by_dimension[dim][sev] = by_dimension[dim].get(sev, 0) + 1

        latest_per_dim: dict[str, dict[str, Any]] = {}
        for a in recent_alerts:
            dim = a.get("dimension", "UNKNOWN")
            if dim not in latest_per_dim:
                latest_per_dim[dim] = a

        is_healthy = n_critical == 0 and n_warning == 0
        return {
            "model_version": model_version or "all",
            "is_healthy": is_healthy,
            "n_alerts_total": len(recent_alerts),
            "n_critical": n_critical,
            "n_warning": n_warning,
            "by_dimension": by_dimension,
            "latest_per_dimension": latest_per_dim,
            "dimensions": [d.value for d in DriftDimension],
        }

    # ── Dimension: Confidence drift ───────────────────────────────────────────

    def _check_confidence(
        self, decisions: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect shift in mean prediction confidence (baseline vs recent)."""
        confs = [d.get("confidence", 0.5) for d in decisions]
        baseline, recent = _split_windows(
            confs, self._cfg.baseline_window, self._cfg.rolling_window, self._cfg.min_samples
        )
        if baseline is None or recent is None:
            return None

        b_mean = sum(baseline) / len(baseline)
        r_mean = sum(recent) / len(recent)
        drift = abs(r_mean - b_mean)

        if drift < self._cfg.confidence_warning:
            return None

        severity = (
            AlertSeverity.CRITICAL.value
            if drift >= self._cfg.confidence_critical
            else AlertSeverity.WARNING.value
        )
        direction = "decreased" if r_mean < b_mean else "increased"
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.CONFIDENCE.value,
            severity=severity,
            model_version=model_version,
            affected_symbols=[],
            affected_features=["confidence"],
            evidence={
                "baseline_mean_confidence": round(b_mean, 4),
                "recent_mean_confidence": round(r_mean, 4),
                "drift": round(drift, 4),
                "warning_threshold": self._cfg.confidence_warning,
                "critical_threshold": self._cfg.confidence_critical,
                "n_baseline": len(baseline),
                "n_recent": len(recent),
            },
            recommended_action=(
                f"Confidence has {direction} by {drift:.3f} from baseline "
                f"({b_mean:.3f} → {r_mean:.3f}). "
                + ("Immediate recalibration or full retrain recommended."
                   if severity == AlertSeverity.CRITICAL.value
                   else "Monitor confidence trend; consider recalibration.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Feature drift ──────────────────────────────────────────────

    def _check_feature(
        self, decisions: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect shift in output probability distribution (buy/sell/no_trade)
        as a proxy for input feature distribution shift.

        Uses total-variation distance (half the L1 norm) between the
        baseline and recent class distributions.
        """
        if len(decisions) < self._cfg.min_samples * 2:
            return None

        baseline_d, recent_d = _split_windows(
            decisions,
            self._cfg.baseline_window,
            self._cfg.rolling_window,
            self._cfg.min_samples,
        )
        if baseline_d is None or recent_d is None:
            return None

        b_dist = _class_distribution(baseline_d)
        r_dist = _class_distribution(recent_d)
        tv = _tv_distance(b_dist, r_dist)

        if tv < self._cfg.feature_warning:
            return None

        severity = (
            AlertSeverity.CRITICAL.value
            if tv >= self._cfg.feature_critical
            else AlertSeverity.WARNING.value
        )
        # Identify which class shifted the most
        max_shift_class = max(
            ["buy", "sell", "no_trade"],
            key=lambda c: abs(r_dist.get(c, 0) - b_dist.get(c, 0)),
        )
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.FEATURE.value,
            severity=severity,
            model_version=model_version,
            affected_symbols=[],
            affected_features=["buy_prob", "sell_prob", "no_trade_prob"],
            evidence={
                "tv_distance": round(tv, 4),
                "baseline_distribution": {k: round(v, 4) for k, v in b_dist.items()},
                "recent_distribution": {k: round(v, 4) for k, v in r_dist.items()},
                "max_shift_class": max_shift_class,
                "warning_threshold": self._cfg.feature_warning,
                "critical_threshold": self._cfg.feature_critical,
                "n_baseline": len(baseline_d),
                "n_recent": len(recent_d),
            },
            recommended_action=(
                f"Output probability distribution has shifted (TV={tv:.3f}); "
                f"largest shift in class '{max_shift_class}'. "
                + ("Input distribution may have changed significantly — "
                   "full retrain on recent data recommended."
                   if severity == AlertSeverity.CRITICAL.value
                   else "Review input feature pipeline for upstream changes.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Market / regime distribution drift ─────────────────────────

    def _check_market(
        self, decisions: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect shift in market regime distribution over time.

        Computes TV distance between the regime histogram of baseline and
        recent windows.  A significant shift indicates the model is operating
        in a different market environment than it was trained on.
        """
        if len(decisions) < self._cfg.min_samples * 2:
            return None

        baseline_d, recent_d = _split_windows(
            decisions,
            self._cfg.baseline_window,
            self._cfg.rolling_window,
            self._cfg.min_samples,
        )
        if baseline_d is None or recent_d is None:
            return None

        b_regimes = _regime_histogram(baseline_d)
        r_regimes = _regime_histogram(recent_d)
        if not b_regimes or not r_regimes:
            return None  # no regime labels in telemetry

        tv = _tv_distance(b_regimes, r_regimes)
        if tv < self._cfg.market_warning:
            return None

        severity = (
            AlertSeverity.CRITICAL.value
            if tv >= self._cfg.market_critical
            else AlertSeverity.WARNING.value
        )
        # Find which regimes appeared / disappeared
        new_regimes = [r for r in r_regimes if r not in b_regimes and r_regimes[r] > 0.05]
        disappeared = [r for r in b_regimes if r not in r_regimes or r_regimes[r] < 0.01]
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.MARKET.value,
            severity=severity,
            model_version=model_version,
            affected_symbols=[],
            affected_features=["regime"],
            evidence={
                "tv_distance": round(tv, 4),
                "baseline_regime_dist": {k: round(v, 4) for k, v in b_regimes.items()},
                "recent_regime_dist": {k: round(v, 4) for k, v in r_regimes.items()},
                "new_regimes": new_regimes,
                "disappeared_regimes": disappeared,
                "warning_threshold": self._cfg.market_warning,
                "critical_threshold": self._cfg.market_critical,
            },
            recommended_action=(
                f"Market regime distribution has shifted significantly (TV={tv:.3f}). "
                + (f"New regimes: {new_regimes}. " if new_regimes else "")
                + ("Regime-conditional recalibration or retrain on recent market data recommended."
                   if severity == AlertSeverity.CRITICAL.value
                   else "Monitor regime shift; consider regime-targeted fine-tuning.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Data quality drift ─────────────────────────────────────────

    def _check_data_quality(
        self,
        decisions: list[dict],
        model_version: str,
        now: float,
    ) -> DriftAlert | None:
        """Detect telemetry completeness and freshness issues.

        Two sub-checks:
        1. **Staleness** — time since the most recent telemetry event.
        2. **Event-rate drop** — compare events-per-unit-time in recent vs
           baseline windows using the actual timestamps.
        """
        if not decisions:
            # No data at all — treat as CRITICAL staleness if model has been deployed
            return None  # cannot determine expected rate without baseline

        # 1. Staleness check
        latest_ts = max(d.get("timestamp", 0) for d in decisions)
        staleness_s = now - latest_ts

        sev: str | None = None
        if staleness_s >= self._cfg.staleness_critical_s:
            sev = AlertSeverity.CRITICAL.value
        elif staleness_s >= self._cfg.staleness_warning_s:
            sev = AlertSeverity.WARNING.value

        # 2. Event-rate drop check (only when there's enough history to compare)
        rate_evidence: dict[str, Any] = {}
        if len(decisions) >= self._cfg.min_samples * 2:
            baseline_d, recent_d = _split_windows(
                decisions,
                self._cfg.baseline_window,
                self._cfg.rolling_window,
                self._cfg.min_samples,
            )
            if baseline_d and recent_d and len(baseline_d) >= 2 and len(recent_d) >= 2:
                b_span = max(
                    decisions[len(baseline_d) - 1].get("timestamp", 0)
                    - decisions[0].get("timestamp", 1),
                    1.0,
                )
                r_span = max(
                    decisions[-1].get("timestamp", 0)
                    - decisions[len(decisions) - len(recent_d)].get("timestamp", 1),
                    1.0,
                )
                b_rate = len(baseline_d) / b_span
                r_rate = len(recent_d) / r_span
                if b_rate > 0:
                    drop = max(0.0, (b_rate - r_rate) / b_rate)
                    rate_evidence = {
                        "baseline_rate_per_s": round(b_rate, 6),
                        "recent_rate_per_s": round(r_rate, 6),
                        "rate_drop_fraction": round(drop, 4),
                    }
                    if drop >= self._cfg.event_rate_drop_critical:
                        if sev != AlertSeverity.CRITICAL.value:
                            sev = AlertSeverity.CRITICAL.value
                    elif drop >= self._cfg.event_rate_drop_warning:
                        if sev is None:
                            sev = AlertSeverity.WARNING.value

        if sev is None:
            return None

        from datetime import UTC, datetime
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.DATA_QUALITY.value,
            severity=sev,
            model_version=model_version,
            affected_symbols=[],
            affected_features=[],
            evidence={
                "staleness_s": round(staleness_s, 1),
                "latest_event_at": datetime.fromtimestamp(latest_ts, UTC).isoformat()
                if latest_ts > 0
                else "unknown",
                "n_total_events": len(decisions),
                "staleness_warning_threshold_s": self._cfg.staleness_warning_s,
                "staleness_critical_threshold_s": self._cfg.staleness_critical_s,
                **rate_evidence,
            },
            recommended_action=(
                f"Telemetry freshness degraded: last event {staleness_s / 3600:.1f}h ago. "
                + (
                    f"Event rate dropped by {rate_evidence.get('rate_drop_fraction', 0) * 100:.0f}%. "
                    if rate_evidence.get("rate_drop_fraction", 0) >= self._cfg.event_rate_drop_warning
                    else ""
                )
                + (
                    "Check telemetry pipeline and model deployment health."
                    if sev == AlertSeverity.CRITICAL.value
                    else "Investigate telemetry ingestion delays."
                )
            ),
            created_at=now,
        )

    # ── Dimension: Per-regime win-rate drift ──────────────────────────────────

    def _check_regime(
        self, trades: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect per-regime performance degradation.

        For each regime with sufficient trades in both windows, computes the
        baseline and recent win rate (fraction with pnl > 0).  If any regime
        shows a drop exceeding the configured threshold, an alert fires.
        """
        if len(trades) < self._cfg.min_samples * 2:
            return None

        baseline_t, recent_t = _split_windows(
            trades,
            self._cfg.baseline_window,
            self._cfg.rolling_window,
            self._cfg.min_samples,
        )
        if baseline_t is None or recent_t is None:
            return None

        # Group trades by regime
        def _win_rate_by_regime(window: list[dict]) -> dict[str, float]:
            groups: dict[str, list[float]] = {}
            for t in window:
                regime = t.get("regime", "unknown") or "unknown"
                groups.setdefault(regime, []).append(1.0 if t.get("pnl", 0) > 0 else 0.0)
            return {
                r: sum(v) / len(v)
                for r, v in groups.items()
                if len(v) >= max(self._cfg.min_samples // 3, 3)
            }

        b_wr = _win_rate_by_regime(baseline_t)
        r_wr = _win_rate_by_regime(recent_t)

        degraded: dict[str, dict[str, float]] = {}
        for regime in b_wr:
            if regime not in r_wr:
                continue
            drop = b_wr[regime] - r_wr[regime]
            if drop >= self._cfg.regime_win_rate_warning:
                degraded[regime] = {
                    "baseline_win_rate": round(b_wr[regime], 4),
                    "recent_win_rate": round(r_wr[regime], 4),
                    "drop": round(drop, 4),
                }

        if not degraded:
            return None

        max_drop = max(v["drop"] for v in degraded.values())
        severity = (
            AlertSeverity.CRITICAL.value
            if max_drop >= self._cfg.regime_win_rate_critical
            else AlertSeverity.WARNING.value
        )
        worst_regime = max(degraded, key=lambda r: degraded[r]["drop"])
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.REGIME.value,
            severity=severity,
            model_version=model_version,
            affected_symbols=[],
            affected_features=["regime", "win_rate"],
            evidence={
                "degraded_regimes": degraded,
                "worst_regime": worst_regime,
                "max_win_rate_drop": round(max_drop, 4),
                "warning_threshold": self._cfg.regime_win_rate_warning,
                "critical_threshold": self._cfg.regime_win_rate_critical,
                "n_baseline_trades": len(baseline_t),
                "n_recent_trades": len(recent_t),
            },
            recommended_action=(
                f"Win rate degraded in {len(degraded)} regime(s); "
                f"worst: '{worst_regime}' ({degraded[worst_regime]['baseline_win_rate']:.2f} → "
                f"{degraded[worst_regime]['recent_win_rate']:.2f}). "
                + (
                    "Regime-conditional fine-tuning or full retrain required."
                    if severity == AlertSeverity.CRITICAL.value
                    else "Review regime-specific feature quality and label alignment."
                )
            ),
            created_at=time.time(),
        )

    # ── CRITICAL → RetrainingRecommendation ───────────────────────────────────

    def _maybe_create_recommendation(self, alert: DriftAlert) -> str:
        """If the alert is CRITICAL and no open recommendation exists for the
        same (model_version, dimension) pair, create one.

        Returns the recommendation_id (or "" if skipped / rec_engine absent).
        """
        if self._rec_engine is None:
            return ""

        eval_dim = _DIM_TO_EVAL.get(DriftDimension(alert.dimension))
        if eval_dim is None:
            return ""

        # Deduplication: check open recommendations for same (mv, dimension)
        open_recs = self._rec_engine._db.list_retraining_recommendations(
            status=RecommendationStatus.PROPOSED.value,
            model_version=alert.model_version,
            limit=200,
        )
        for existing in open_recs:
            if existing.get("trigger_dimension") == eval_dim.value:
                logger.debug(
                    f"HealthMonitor: skipping duplicate recommendation for "
                    f"{eval_dim.value} model={alert.model_version!r}"
                )
                return existing.get("recommendation_id", "")

        scope = (
            RecommendedScope.FULL_RETRAIN
            if alert.dimension in (DriftDimension.DATA_QUALITY.value, DriftDimension.MARKET.value)
            else RecommendedScope.FINE_TUNE
        )
        rec = RetrainingRecommendation.new(
            model_version=alert.model_version,
            trigger_dimension=eval_dim,
            trigger_reason=(
                f"CRITICAL {alert.dimension} drift alert detected by health monitor. "
                f"{alert.recommended_action}"
            ),
            evidence=alert.evidence,
            recommended_scope=scope,
            urgency=RecommendationUrgency.CRITICAL,
        )
        self._rec_engine._db.upsert_retraining_recommendation(rec.to_dict())
        logger.info(
            f"HealthMonitor: created recommendation {rec.recommendation_id} "
            f"for CRITICAL {alert.dimension} alert on model={alert.model_version!r}"
        )
        return rec.recommendation_id


# ── Helpers ────────────────────────────────────────────────────────────────────


def _class_distribution(decisions: list[dict]) -> dict[str, float]:
    """Compute buy/sell/no_trade distribution from a list of decision dicts.

    Prefers explicit probability fields (``buy_prob`` / ``sell_prob`` /
    ``no_trade_prob``) when present.  Falls back to counting the ``direction``
    string field, which is what the live telemetry warehouse stores.
    """
    n = len(decisions)
    if n == 0:
        return {"buy": 0.33, "sell": 0.33, "no_trade": 0.34}
    # Prefer explicit probability fields if any decision carries them
    if any("buy_prob" in d for d in decisions):
        return {
            "buy": sum(d.get("buy_prob", 0.33) for d in decisions) / n,
            "sell": sum(d.get("sell_prob", 0.33) for d in decisions) / n,
            "no_trade": sum(d.get("no_trade_prob", 0.34) for d in decisions) / n,
        }
    # Fall back to direction counts (live telemetry schema)
    directions = [d.get("direction", "") or "" for d in decisions]
    return {
        "buy": sum(1 for di in directions if di == "buy") / n,
        "sell": sum(1 for di in directions if di == "sell") / n,
        "no_trade": sum(1 for di in directions if di not in ("buy", "sell")) / n,
    }


def _regime_histogram(decisions: list[dict]) -> dict[str, float]:
    """Compute regime frequency distribution from a list of decision dicts."""
    n = len(decisions)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for d in decisions:
        regime = d.get("regime", "") or ""
        if regime:
            counts[regime] = counts.get(regime, 0) + 1
    if not counts:
        return {}
    return {k: v / n for k, v in counts.items()}


def _tv_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Total-variation distance between two distributions.

    TV = 0.5 * sum |p_i - q_i| — symmetric, bounded in [0, 1].
    Keys present in one but not the other are treated as 0 in the other.
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
