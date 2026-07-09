"""S20/S22B — Research Lab Model Health Monitor.

Continuously monitors model health across thirteen drift/health dimensions:

  1. **Feature drift**       — input distribution shift (buy/sell/no_trade probs)
  2. **Confidence drift**    — output calibration shift (mean confidence delta)
  3. **Market drift**        — regime/volatility distribution shift
  4. **Data quality**        — telemetry completeness and freshness
  5. **Regime drift**        — per-regime win-rate performance delta
  6. **Entropy drift**       — prediction entropy trend (model decisiveness)
  7. **Uncertainty drift**   — dispersion (std) of confidence vs baseline
  8. **Calibration drift**   — rolling ECE-style confidence-vs-outcome gap
  9. **Decision quality**    — expectancy + confidence-outcome correlation drop
     (see ``decision_quality.py`` for the full queryable report surface:
     expectancy broken down by regime/symbol/session/volatility state)
  10. **Performance**        — profit factor/Sharpe/drawdown/hit rate drop
     (see ``performance_monitor.py`` for the full queryable report surface:
     performance broken down by symbol/regime/user cohort, plus
     risk-adjusted return and capital efficiency)
  11. **Feature health**     — missing/stale/corrupted/delayed pipeline features
     (see ``feature_health.py`` for the full queryable report surface: a
     per-feature breakdown of presence, staleness, corruption rate, and
     delivery delay)
  12. **Unseen conditions**  — live regimes outside the model's known/trained
     regime vocabulary (distinct from ordinary MARKET distribution shift)
  13. **Black swan**         — statistically extreme individual trade outcomes
     (z-score vs. trailing baseline), distinct from both distribution shift
     and per-regime win-rate degradation
     (see ``regime_coverage.py`` for the full queryable report surface
     covering all three regime-coverage conditions together)

Each dimension emits a ``DriftAlert`` with severity WARNING or CRITICAL.
CRITICAL alerts automatically create a ``RetrainingRecommendation`` (pending
admin approval) and are deduplicated against open recommendations for the
same (model_version, dimension) pair.

Alerts are persisted in the Research Lab DB and returned by
``GET /health/alerts``.
"""

from __future__ import annotations

import math
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

from trainer.research_lab.decision_quality import (
    DecisionQualityConfig,
    check_decision_quality_drift,
)
from trainer.research_lab.dimension_evaluators import _split_windows
from trainer.research_lab.performance_monitor import (
    PerformanceMonitorConfig,
    check_performance_drift,
)
from trainer.research_lab.feature_health import (
    FeatureHealthConfig,
    check_feature_health,
)
from trainer.research_lab.regime_coverage import (
    RegimeCoverageConfig,
    check_black_swans,
    check_unseen_conditions,
)
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
    ENTROPY = "ENTROPY"
    UNCERTAINTY = "UNCERTAINTY"
    CALIBRATION = "CALIBRATION"
    DECISION_QUALITY = "DECISION_QUALITY"
    PERFORMANCE = "PERFORMANCE"
    FEATURE_HEALTH = "FEATURE_HEALTH"
    UNSEEN_CONDITION = "UNSEEN_CONDITION"
    BLACK_SWAN = "BLACK_SWAN"


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
    # New S22B dimensions are all confidence/calibration-adjacent signals —
    # route them through the same evaluation category pending a dedicated
    # EvaluationDimension if the Research Lab needs finer-grained triage.
    DriftDimension.ENTROPY: EvaluationDimension.CONFIDENCE_DRIFT,
    DriftDimension.UNCERTAINTY: EvaluationDimension.CONFIDENCE_DRIFT,
    DriftDimension.CALIBRATION: EvaluationDimension.CONFIDENCE_DRIFT,
    # Decision quality (expectancy + confidence-outcome correlation) is an
    # economic-performance signal, same category as Sharpe degradation.
    DriftDimension.DECISION_QUALITY: EvaluationDimension.SHARPE_DEGRADATION,
    # Autonomous performance (profit factor/Sharpe/drawdown/hit rate) is the
    # same economic-performance category as Sharpe degradation.
    DriftDimension.PERFORMANCE: EvaluationDimension.SHARPE_DEGRADATION,
    # Feature-level pipeline health (missing/stale/corrupted/delayed inputs)
    # is mapped to REASONING_DRIFT (bad inputs corrupt reasoning) rather than
    # DATA_QUALITY's SHARPE_DEGRADATION, differentiating it from the coarse
    # telemetry staleness/event-rate check. Note FEATURE also maps to
    # REASONING_DRIFT, so coarse categories are NOT unique per dimension —
    # the actual dedup-collision safety comes from `_maybe_create_recommendation`
    # keying on the fine-grained `_health_dimension` evidence field (see memory:
    # health-monitor-dedup-collision), not from coarse-category uniqueness.
    DriftDimension.FEATURE_HEALTH: EvaluationDimension.REASONING_DRIFT,
    # Unseen market conditions and black-swan moves are both regime-coverage
    # signals, same coarse category as MARKET/REGIME (dedup safety comes
    # from the fine-grained _health_dimension key, not coarse uniqueness —
    # see the FEATURE_HEALTH comment above and memory:
    # health-monitor-dedup-collision).
    DriftDimension.UNSEEN_CONDITION: EvaluationDimension.REGIME_UNDERPERFORMANCE,
    DriftDimension.BLACK_SWAN: EvaluationDimension.REGIME_UNDERPERFORMANCE,
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
        recommendation_id: Set when this alert creates a
            RetrainingRecommendation (S22B: both CRITICAL and WARNING alerts
            route into a recommendation; urgency scales with severity).
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
    """Configurable thresholds for all thirteen drift/health dimensions.

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

    # ── 6. Prediction entropy drift (|mean_recent - mean_baseline| entropy) ────
    entropy_warning: float = 0.15   # bits (3-class max entropy ≈ 1.585 bits)
    entropy_critical: float = 0.30

    # ── 7. Uncertainty drift (|std_recent - std_baseline| of confidence) ───────
    uncertainty_warning: float = 0.05
    uncertainty_critical: float = 0.10

    # ── 8. Calibration drift (rolling ECE increase, recent vs baseline) ────────
    calibration_warning: float = 0.08
    calibration_critical: float = 0.15
    calibration_bins: int = 5

    # ── 9. Decision quality (expectancy + confidence-outcome correlation) ──────
    decision_quality_expectancy_warning: float = 0.15   # relative expectancy drop
    decision_quality_expectancy_critical: float = 0.35
    decision_quality_correlation_warning: float = 0.20  # absolute correlation drop
    decision_quality_correlation_critical: float = 0.40

    # ── 10. Autonomous performance (profit factor/Sharpe/drawdown/hit rate) ────
    performance_profit_factor_warning: float = 0.20
    performance_profit_factor_critical: float = 0.45
    performance_sharpe_warning: float = 0.30
    performance_sharpe_critical: float = 0.60
    performance_drawdown_increase_warning: float = 0.10
    performance_drawdown_increase_critical: float = 0.25
    performance_hit_rate_drop_warning: float = 0.10
    performance_hit_rate_drop_critical: float = 0.20

    # ── 11. Feature health (missing/stale/corrupted/delayed inputs) ───────────
    feature_health_expected_features: list | None = None
    feature_health_value_bounds: dict = field(default_factory=dict)
    feature_health_staleness_warning_s: float = 3600.0
    feature_health_staleness_critical_s: float = 21600.0
    feature_health_delay_warning_s: float = 30.0
    feature_health_delay_critical_s: float = 120.0
    feature_health_corruption_rate_warning: float = 0.05
    feature_health_corruption_rate_critical: float = 0.20
    feature_health_missing_rate_warning: float = 0.95
    feature_health_missing_rate_critical: float = 0.80
    feature_health_stream_staleness_warning_s: float = 3600.0
    feature_health_stream_staleness_critical_s: float = 14400.0

    # ── 12. Regime coverage: unseen conditions + black swans ──────────────────
    # (Distribution shift itself is already covered by dimension #3 MARKET;
    # these two are the additional, distinctly-labeled conditions.)
    regime_coverage_known_regimes: frozenset | None = None
    unseen_condition_rate_warning: float = 0.05
    unseen_condition_rate_critical: float = 0.20
    black_swan_zscore_threshold: float = 4.0
    black_swan_rate_warning: float = 0.02
    black_swan_rate_critical: float = 0.08


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
    """Orchestrates thirteen-dimension model health monitoring for the Research Lab.

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
        """Run all thirteen drift/health-dimension checks and persist any alerts.

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
        reasoning_events = sorted(
            self._wh.query_reasoning(q), key=lambda e: e.get("timestamp", 0)
        )

        alerts: list[DriftAlert] = []
        checked = []

        for name, fn, args in [
            (DriftDimension.CONFIDENCE.value, self._check_confidence, (decisions, mv)),
            (DriftDimension.FEATURE.value, self._check_feature, (decisions, mv)),
            (DriftDimension.MARKET.value, self._check_market, (decisions, mv)),
            (DriftDimension.DATA_QUALITY.value, self._check_data_quality, (decisions, mv, now)),
            (DriftDimension.REGIME.value, self._check_regime, (trades, mv)),
            (DriftDimension.ENTROPY.value, self._check_entropy, (decisions, mv)),
            (DriftDimension.UNCERTAINTY.value, self._check_uncertainty, (decisions, mv)),
            (DriftDimension.CALIBRATION.value, self._check_calibration, (trades, mv)),
            (DriftDimension.DECISION_QUALITY.value, self._check_decision_quality, (trades, mv)),
            (DriftDimension.PERFORMANCE.value, self._check_performance, (trades, mv)),
            (DriftDimension.FEATURE_HEALTH.value, self._check_feature_health, (reasoning_events, mv, now)),
            (DriftDimension.UNSEEN_CONDITION.value, self._check_unseen_condition, (decisions, mv)),
            (DriftDimension.BLACK_SWAN.value, self._check_black_swan, (trades, mv)),
        ]:
            checked.append(name)
            try:
                alert = fn(*args)
                if alert is not None:
                    # Persist to DB
                    self._db.upsert_health_alert(alert.to_dict())
                    # S22B: every degradation finding — CRITICAL or WARNING —
                    # must route into a Research Lab review task, not just
                    # CRITICAL ones. Urgency scales with severity so admins
                    # can triage, but nothing is silently dropped.
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

    # ── Dimension: Prediction entropy drift ───────────────────────────────────

    def _check_entropy(
        self, decisions: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect drift in mean prediction entropy (baseline vs recent).

        Entropy is computed per-decision from the 3-class (buy/sell/no_trade)
        probability distribution — either explicit probability fields or a
        proxy derived from ``confidence`` + ``direction``. Rising entropy
        indicates the model is becoming less decisive/more confused.
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

        b_entropies = [_decision_entropy(d) for d in baseline_d]
        r_entropies = [_decision_entropy(d) for d in recent_d]
        b_mean = sum(b_entropies) / len(b_entropies)
        r_mean = sum(r_entropies) / len(r_entropies)
        drift = abs(r_mean - b_mean)

        if drift < self._cfg.entropy_warning:
            return None

        severity = (
            AlertSeverity.CRITICAL.value
            if drift >= self._cfg.entropy_critical
            else AlertSeverity.WARNING.value
        )
        direction = "increased" if r_mean > b_mean else "decreased"
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.ENTROPY.value,
            severity=severity,
            model_version=model_version,
            affected_symbols=[],
            affected_features=["prediction_entropy"],
            evidence={
                "baseline_mean_entropy_bits": round(b_mean, 4),
                "recent_mean_entropy_bits": round(r_mean, 4),
                "drift": round(drift, 4),
                "warning_threshold": self._cfg.entropy_warning,
                "critical_threshold": self._cfg.entropy_critical,
                "n_baseline": len(baseline_d),
                "n_recent": len(recent_d),
            },
            recommended_action=(
                f"Prediction entropy has {direction} by {drift:.3f} bits from baseline "
                f"({b_mean:.3f} → {r_mean:.3f}). "
                + (
                    (
                        "Model is significantly less decisive than baseline — "
                        "investigate input quality and consider retraining."
                        if r_mean > b_mean
                        else "Model has become significantly more decisive/overconfident "
                        "than baseline — verify this isn't overfitting to recent conditions."
                    )
                    if severity == AlertSeverity.CRITICAL.value
                    else "Monitor decisiveness trend."
                )
            ),
            created_at=time.time(),
        )

    # ── Dimension: Uncertainty drift ──────────────────────────────────────────

    def _check_uncertainty(
        self, decisions: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect drift in confidence dispersion (std) — a proxy for the
        model's own uncertainty when no explicit uncertainty estimate is
        exposed. Distinct from ``_check_confidence``, which tracks the mean;
        this tracks the spread, so a model that becomes erratic (wider
        confidence spread) without a mean shift is still caught.
        """
        confs = [d.get("confidence", 0.5) for d in decisions]
        baseline, recent = _split_windows(
            confs, self._cfg.baseline_window, self._cfg.rolling_window, self._cfg.min_samples
        )
        if baseline is None or recent is None:
            return None

        b_std = _stdev(baseline)
        r_std = _stdev(recent)
        drift = abs(r_std - b_std)

        if drift < self._cfg.uncertainty_warning:
            return None

        severity = (
            AlertSeverity.CRITICAL.value
            if drift >= self._cfg.uncertainty_critical
            else AlertSeverity.WARNING.value
        )
        direction = "widened" if r_std > b_std else "narrowed"
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.UNCERTAINTY.value,
            severity=severity,
            model_version=model_version,
            affected_symbols=[],
            affected_features=["confidence_std"],
            evidence={
                "baseline_confidence_std": round(b_std, 4),
                "recent_confidence_std": round(r_std, 4),
                "drift": round(drift, 4),
                "warning_threshold": self._cfg.uncertainty_warning,
                "critical_threshold": self._cfg.uncertainty_critical,
                "n_baseline": len(baseline),
                "n_recent": len(recent),
            },
            recommended_action=(
                f"Confidence dispersion has {direction} by {drift:.3f} from baseline "
                f"({b_std:.3f} → {r_std:.3f}). "
                + ("Model uncertainty behavior has changed sharply — "
                   "investigate before continuing autonomous operation."
                   if severity == AlertSeverity.CRITICAL.value
                   else "Monitor uncertainty trend.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Calibration drift ──────────────────────────────────────────

    def _check_calibration(
        self, trades: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect worsening calibration (confidence vs realized outcome) using
        a rolling Expected-Calibration-Error-style metric.

        Uses ``confidence_at_entry`` and realized ``pnl`` (correct = pnl > 0)
        from closed trades, binned into ``calibration_bins`` buckets. Only
        fires when calibration error has *worsened* (recent ECE higher than
        baseline) by at least the warning threshold.
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

        b_ece = _expected_calibration_error(baseline_t, self._cfg.calibration_bins)
        r_ece = _expected_calibration_error(recent_t, self._cfg.calibration_bins)
        if b_ece is None or r_ece is None:
            return None

        drift = r_ece - b_ece  # only worsening (positive) triggers an alert
        if drift < self._cfg.calibration_warning:
            return None

        severity = (
            AlertSeverity.CRITICAL.value
            if drift >= self._cfg.calibration_critical
            else AlertSeverity.WARNING.value
        )
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.CALIBRATION.value,
            severity=severity,
            model_version=model_version,
            affected_symbols=[],
            affected_features=["confidence_at_entry", "pnl"],
            evidence={
                "baseline_ece": round(b_ece, 4),
                "recent_ece": round(r_ece, 4),
                "drift": round(drift, 4),
                "warning_threshold": self._cfg.calibration_warning,
                "critical_threshold": self._cfg.calibration_critical,
                "n_baseline": len(baseline_t),
                "n_recent": len(recent_t),
            },
            recommended_action=(
                f"Calibration error worsened by {drift:.3f} from baseline "
                f"({b_ece:.3f} → {r_ece:.3f}) — confidence no longer tracks "
                f"realized outcomes as well. "
                + ("Recalibration (temperature scaling) or full retrain recommended."
                   if severity == AlertSeverity.CRITICAL.value
                   else "Monitor confidence-vs-outcome alignment.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Decision quality (expectancy + confidence-outcome corr) ─────

    def _check_decision_quality(
        self, trades: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect degradation in decision quality: a relative drop in
        expectancy and/or a drop in how well confidence tracks realized
        outcomes, comparing baseline vs recent trade windows.

        Delegates the actual comparison to
        ``decision_quality.check_decision_quality_drift`` (the same logic
        backs the queryable ``DecisionQualityTracker`` reports), so this
        method's only job is translating that result into a ``DriftAlert``
        and slotting it into the existing alerting/recommendation path.
        """
        dq_cfg = DecisionQualityConfig(
            baseline_window=self._cfg.baseline_window,
            rolling_window=self._cfg.rolling_window,
            min_samples=self._cfg.min_samples,
            expectancy_drift_warning=self._cfg.decision_quality_expectancy_warning,
            expectancy_drift_critical=self._cfg.decision_quality_expectancy_critical,
            correlation_drift_warning=self._cfg.decision_quality_correlation_warning,
            correlation_drift_critical=self._cfg.decision_quality_correlation_critical,
        )
        result = check_decision_quality_drift(trades, dq_cfg, _split_windows)
        if result is None:
            return None

        evidence = result["evidence"]
        triggers = result["triggers"]
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.DECISION_QUALITY.value,
            severity=result["severity"],
            model_version=model_version,
            affected_symbols=[],
            affected_features=triggers,
            evidence=evidence,
            recommended_action=(
                f"Decision quality has degraded ({', '.join(triggers)}): "
                f"expectancy {evidence['baseline_expectancy']:.3f} → "
                f"{evidence['recent_expectancy']:.3f} "
                f"(relative drop {evidence['expectancy_relative_drop']:.2%}), "
                f"confidence-outcome correlation "
                f"{evidence['baseline_confidence_outcome_correlation']} → "
                f"{evidence['recent_confidence_outcome_correlation']}. "
                + ("Full retrain or expectancy-breakdown investigation recommended."
                   if result["severity"] == AlertSeverity.CRITICAL.value
                   else "Monitor expectancy and confidence-outcome trend.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Autonomous performance (profit factor/Sharpe/drawdown/hit rate) ──

    def _check_performance(
        self, trades: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect degradation in autonomous trading performance: a relative
        drop in profit factor and/or Sharpe ratio, a rise in max drawdown,
        or a drop in hit rate, comparing baseline vs recent trade windows.

        Delegates the comparison to
        ``performance_monitor.check_performance_drift`` (the same logic
        backs the queryable ``PerformanceMonitor`` reports), so this method's
        only job is translating that result into a ``DriftAlert`` and
        slotting it into the existing alerting/recommendation path.
        """
        perf_cfg = PerformanceMonitorConfig(
            baseline_window=self._cfg.baseline_window,
            rolling_window=self._cfg.rolling_window,
            min_samples=self._cfg.min_samples,
            profit_factor_drift_warning=self._cfg.performance_profit_factor_warning,
            profit_factor_drift_critical=self._cfg.performance_profit_factor_critical,
            sharpe_drift_warning=self._cfg.performance_sharpe_warning,
            sharpe_drift_critical=self._cfg.performance_sharpe_critical,
            drawdown_increase_warning=self._cfg.performance_drawdown_increase_warning,
            drawdown_increase_critical=self._cfg.performance_drawdown_increase_critical,
            hit_rate_drop_warning=self._cfg.performance_hit_rate_drop_warning,
            hit_rate_drop_critical=self._cfg.performance_hit_rate_drop_critical,
        )
        result = check_performance_drift(trades, perf_cfg, _split_windows)
        if result is None:
            return None

        evidence = result["evidence"]
        triggers = result["triggers"]
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.PERFORMANCE.value,
            severity=result["severity"],
            model_version=model_version,
            affected_symbols=[],
            affected_features=triggers,
            evidence=evidence,
            recommended_action=(
                f"Autonomous performance has degraded ({', '.join(triggers)}): "
                f"profit factor {evidence['baseline_profit_factor']:.3f} → "
                f"{evidence['recent_profit_factor']:.3f}, "
                f"Sharpe {evidence['baseline_sharpe_ratio']:.3f} → "
                f"{evidence['recent_sharpe_ratio']:.3f}, "
                f"max drawdown {evidence['baseline_max_drawdown']:.3f} → "
                f"{evidence['recent_max_drawdown']:.3f}, "
                f"hit rate {evidence['baseline_hit_rate']:.3f} → "
                f"{evidence['recent_hit_rate']:.3f}. "
                + ("Full retrain or risk-budget review recommended."
                   if result["severity"] == AlertSeverity.CRITICAL.value
                   else "Monitor performance trend across symbols/regimes/cohorts.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Feature health (missing/stale/corrupted/delayed inputs) ────

    def _check_feature_health(
        self, reasoning_events: list[dict], model_version: str, now: float | None = None
    ) -> DriftAlert | None:
        """Detect missing, stale, corrupted, or delayed pipeline features.

        Delegates to ``feature_health.check_feature_health`` (which also
        backs the queryable ``FeatureHealthMonitor`` reports) for the
        per-feature evaluation; this method only translates the result into
        a ``DriftAlert`` and slots it into the existing alerting path.
        """
        cfg = FeatureHealthConfig(
            expected_features=self._cfg.feature_health_expected_features,
            value_bounds=self._cfg.feature_health_value_bounds,
            staleness_warning_s=self._cfg.feature_health_staleness_warning_s,
            staleness_critical_s=self._cfg.feature_health_staleness_critical_s,
            delay_warning_s=self._cfg.feature_health_delay_warning_s,
            delay_critical_s=self._cfg.feature_health_delay_critical_s,
            corruption_rate_warning=self._cfg.feature_health_corruption_rate_warning,
            corruption_rate_critical=self._cfg.feature_health_corruption_rate_critical,
            missing_rate_warning=self._cfg.feature_health_missing_rate_warning,
            missing_rate_critical=self._cfg.feature_health_missing_rate_critical,
            stream_staleness_warning_s=self._cfg.feature_health_stream_staleness_warning_s,
            stream_staleness_critical_s=self._cfg.feature_health_stream_staleness_critical_s,
        )
        result = check_feature_health(reasoning_events, cfg, now=now)
        if result is None:
            return None

        evidence = result["evidence"]
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.FEATURE_HEALTH.value,
            severity=result["severity"],
            model_version=model_version,
            affected_symbols=[],
            affected_features=result["triggers"],
            evidence=evidence,
            recommended_action=(
                f"Feature pipeline health degraded — "
                f"missing: {evidence['missing_features'] or 'none'}, "
                f"stale: {evidence['stale_features'] or 'none'}, "
                f"corrupted: {evidence['corrupted_features'] or 'none'}, "
                f"delayed: {evidence['delayed_features'] or 'none'}"
                + (f", reasoning stream stale for {evidence['stream_staleness_s']:.0f}s"
                   if "stream_stale" in result["triggers"] else "")
                + ". "
                + ("Investigate upstream feature pipeline immediately — inference "
                   "quality is compromised."
                   if result["severity"] == AlertSeverity.CRITICAL.value
                   else "Monitor feature pipeline; verify upstream data sources.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Unseen market conditions ────────────────────────────────────

    def _check_unseen_condition(
        self, decisions: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect live regimes falling outside the model's known/trained
        regime vocabulary — distinct from ordinary distribution shift
        (dimension #3, MARKET): a regime can be entirely new even when the
        *known* regimes' relative frequencies haven't shifted at all.

        Delegates to ``regime_coverage.check_unseen_conditions`` (which
        also backs the queryable ``RegimeCoverageMonitor`` reports).
        """
        cfg = RegimeCoverageConfig(
            baseline_window=self._cfg.baseline_window,
            rolling_window=self._cfg.rolling_window,
            min_samples=self._cfg.min_samples,
            known_regimes=self._cfg.regime_coverage_known_regimes,
            unseen_condition_rate_warning=self._cfg.unseen_condition_rate_warning,
            unseen_condition_rate_critical=self._cfg.unseen_condition_rate_critical,
        )
        result = check_unseen_conditions(decisions, cfg)
        if result is None:
            return None

        evidence = result["evidence"]
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.UNSEEN_CONDITION.value,
            severity=result["severity"],
            model_version=model_version,
            affected_symbols=[],
            affected_features=list(evidence.get("unseen_regimes", {}).keys()),
            evidence=evidence,
            recommended_action=(
                f"Unseen market conditions detected: {evidence.get('unseen_regimes', {})} "
                f"({evidence.get('unseen_rate', 0) * 100:.1f}% of recent decisions). "
                + ("Model is operating outside its trained envelope — "
                   "urgent review of live conditions and retraining scope recommended."
                   if result["severity"] == AlertSeverity.CRITICAL.value
                   else "Monitor; consider expanding training data to cover these conditions.")
            ),
            created_at=time.time(),
        )

    # ── Dimension: Black swan detection ────────────────────────────────────────

    def _check_black_swan(
        self, trades: list[dict], model_version: str
    ) -> DriftAlert | None:
        """Detect statistically extreme individual trade outcomes (z-score
        vs the trailing baseline window) — distinct from both ordinary
        distribution shift and per-regime win-rate degradation: a single
        black-swan move can occur inside an otherwise-stable, well-performing
        regime distribution.

        Delegates to ``regime_coverage.check_black_swans`` (which also
        backs the queryable ``RegimeCoverageMonitor`` reports).
        """
        cfg = RegimeCoverageConfig(
            baseline_window=self._cfg.baseline_window,
            rolling_window=self._cfg.rolling_window,
            min_samples=self._cfg.min_samples,
            black_swan_zscore_threshold=self._cfg.black_swan_zscore_threshold,
            black_swan_rate_warning=self._cfg.black_swan_rate_warning,
            black_swan_rate_critical=self._cfg.black_swan_rate_critical,
        )
        result = check_black_swans(trades, cfg)
        if result is None:
            return None

        evidence = result["evidence"]
        return DriftAlert(
            alert_id=f"hal-{uuid.uuid4().hex[:12]}",
            dimension=DriftDimension.BLACK_SWAN.value,
            severity=result["severity"],
            model_version=model_version,
            affected_symbols=[e.get("symbol") for e in evidence.get("black_swan_events", [])],
            affected_features=[],
            evidence=evidence,
            recommended_action=(
                f"Black swan move(s) detected: {len(evidence.get('black_swan_events', []))} "
                f"extreme outcome(s) ({evidence.get('black_swan_rate', 0) * 100:.1f}% of recent trades, "
                f"z-score threshold {cfg.black_swan_zscore_threshold}). "
                + ("Review risk controls and position sizing immediately — "
                   "tail risk exposure is elevated."
                   if result["severity"] == AlertSeverity.CRITICAL.value
                   else "Monitor; confirm risk budget accounts for this tail event.")
            ),
            created_at=time.time(),
        )

    # ── CRITICAL → RetrainingRecommendation ───────────────────────────────────

    def _maybe_create_recommendation(self, alert: DriftAlert) -> str:
        """If no open recommendation exists for the same (model_version,
        dimension) pair, create one — for both CRITICAL and WARNING alerts
        (S22B: universal Research Lab task creation; urgency scales with
        severity so nothing is silently dropped).

        Returns the recommendation_id (or "" if skipped / rec_engine absent).
        """
        if self._rec_engine is None:
            return ""

        eval_dim = _DIM_TO_EVAL.get(DriftDimension(alert.dimension))
        if eval_dim is None:
            return ""

        # Deduplication key: several health DriftDimensions (e.g. ENTROPY,
        # UNCERTAINTY, CALIBRATION) all map to the same coarse EvaluationDimension
        # for recommendation categorization, so dedup must also check the
        # originating health dimension (stashed in evidence) — otherwise a CRITICAL
        # alert on one of them would silently suppress a later, distinct CRITICAL
        # alert on another (e.g. a calibration recommendation never being created
        # because an earlier entropy recommendation is still open).
        open_recs = self._rec_engine._db.list_retraining_recommendations(
            status=RecommendationStatus.PROPOSED.value,
            model_version=alert.model_version,
            limit=200,
        )
        for existing in open_recs:
            same_eval_dim = existing.get("trigger_dimension") == eval_dim.value
            same_health_dim = (
                existing.get("evidence", {}).get("_health_dimension") == alert.dimension
            )
            if same_eval_dim and same_health_dim:
                logger.debug(
                    f"HealthMonitor: skipping duplicate recommendation for "
                    f"{alert.dimension} model={alert.model_version!r}"
                )
                return existing.get("recommendation_id", "")

        if alert.dimension in (DriftDimension.DATA_QUALITY.value, DriftDimension.MARKET.value):
            scope = RecommendedScope.FULL_RETRAIN
        elif alert.dimension == DriftDimension.CALIBRATION.value:
            scope = RecommendedScope.CALIBRATION_ONLY
        else:
            scope = RecommendedScope.FINE_TUNE
        urgency = (
            RecommendationUrgency.CRITICAL
            if alert.severity == AlertSeverity.CRITICAL.value
            else RecommendationUrgency.LOW
        )
        rec = RetrainingRecommendation.new(
            model_version=alert.model_version,
            trigger_dimension=eval_dim,
            trigger_reason=(
                f"{alert.severity} {alert.dimension} drift alert detected by health monitor. "
                f"{alert.recommended_action}"
            ),
            evidence={**alert.evidence, "_health_dimension": alert.dimension},
            recommended_scope=scope,
            urgency=urgency,
        )
        self._rec_engine._db.upsert_retraining_recommendation(rec.to_dict())
        logger.info(
            f"HealthMonitor: created recommendation {rec.recommendation_id} "
            f"for {alert.severity} {alert.dimension} alert on model={alert.model_version!r}"
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
    """Compute regime frequency distribution from a list of decision dicts.

    Regime labels are lowercased before counting — telemetry across the
    codebase is inconsistent about casing (e.g. "RANGING" vs "ranging"),
    and casing alone must never manufacture a false MARKET distribution
    shift. Kept in sync with `regime_coverage._regime_histogram`'s
    identical normalization so the MARKET dimension and the persisted
    RegimeCoverageReport's distribution_shift never silently disagree on
    the same underlying telemetry (see memory: regime-label-casing)."""
    n = len(decisions)
    if n == 0:
        return {}
    counts: dict[str, int] = {}
    for d in decisions:
        regime = (d.get("regime", "") or "").lower()
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


def _decision_distribution(d: dict[str, Any]) -> dict[str, float]:
    """Per-decision buy/sell/no_trade probability distribution.

    Prefers explicit probability fields; falls back to a proxy built from
    ``confidence`` + ``direction`` (confidence on the predicted class, the
    remainder split evenly across the other two classes) which is what the
    live telemetry warehouse schema provides.
    """
    if "buy_prob" in d:
        return {
            "buy": d.get("buy_prob", 0.33),
            "sell": d.get("sell_prob", 0.33),
            "no_trade": d.get("no_trade_prob", 0.34),
        }
    conf = max(0.0, min(1.0, d.get("confidence", 0.5)))
    direction = (d.get("direction", "") or "").lower()
    remainder = (1.0 - conf) / 2.0
    dist = {"buy": remainder, "sell": remainder, "no_trade": remainder}
    key = direction if direction in dist else "no_trade"
    dist[key] = conf
    return dist


def _decision_entropy(d: dict[str, Any]) -> float:
    """Shannon entropy (bits) of a single decision's class distribution."""
    dist = _decision_distribution(d)
    eps = 1e-12
    return -sum(p * math.log2(p + eps) for p in dist.values() if p > 0)


def _stdev(values: list[float]) -> float:
    """Population standard deviation; 0.0 for fewer than two samples."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return math.sqrt(variance)


def _expected_calibration_error(
    trades: list[dict[str, Any]], n_bins: int, min_usable: int = 5
) -> float | None:
    """Rolling ECE from ``confidence_at_entry`` vs realized outcome (pnl > 0).

    Returns ``None`` when there are too few trades with confidence data
    (fewer than ``min_usable``) rather than a misleading 0.0/noisy estimate,
    so callers can skip the check instead of computing ECE from a handful
    of points. Confidence values are clipped to ``[0, 1]`` once up front and
    that clipped value is used consistently for both bin assignment and the
    bin's mean-confidence statistic, so out-of-range inputs can't distort
    the reported calibration gap.
    """
    usable = [
        {**t, "confidence_at_entry": max(0.0, min(1.0, t.get("confidence_at_entry", 0.0)))}
        for t in trades
        if "confidence_at_entry" in t
    ]
    if len(usable) < min_usable:
        return None

    n = len(usable)
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bucket = [
            t for t in usable
            if lo <= t["confidence_at_entry"] < hi
            or (i == n_bins - 1 and t["confidence_at_entry"] >= hi)
        ]
        if not bucket:
            continue
        bin_conf = sum(t["confidence_at_entry"] for t in bucket) / len(bucket)
        bin_acc = sum(1.0 for t in bucket if t.get("pnl", 0.0) > 0) / len(bucket)
        ece += (len(bucket) / n) * abs(bin_acc - bin_conf)
    return ece
