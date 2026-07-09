"""S22B — Composite QDE Health Score.

Combines the thirteen drift/health dimensions already computed by
``ResearchLabHealthMonitor`` (which itself reads calibration, decision
quality, autonomous performance, feature health, and regime coverage
signals — see ``health_monitor.py``) into a single composite score,
grouped into five higher-level categories:

  - **Calibration** — CALIBRATION, CONFIDENCE, ENTROPY, UNCERTAINTY
  - **Profitability** — PERFORMANCE, DECISION_QUALITY
  - **Stability** — DATA_QUALITY, REGIME
  - **Robustness** — FEATURE_HEALTH, UNSEEN_CONDITION, BLACK_SWAN, MARKET
  - **Drift** — FEATURE

Every one of the thirteen ``DriftDimension`` values is assigned to exactly
one category (disjoint partition), so the composite score never double- or
under-counts a monitoring signal.

The composite score (0-100) maps to one of five grades:
EXCELLENT / GOOD / WATCH / DEGRADED / CRITICAL.

This module is purely advisory — it reads persisted health-monitor alerts
and produces a queryable score. It has no authority to retrain, promote,
or roll back a model version; those actions remain gated by
``ApprovalGate`` (see ``approval_gate.py``, ``deployment_manager.py``).
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

from trainer.research_lab.health_monitor import (
    AlertSeverity,
    DriftDimension,
    ResearchLabHealthMonitor,
)
from trainer.research_lab.rl_database import ResearchLabDatabase


# ── Category partition ───────────────────────────────────────────────────────

#: Every DriftDimension maps to exactly one composite category. Kept disjoint
#: (unlike the coarse EvaluationDimension reuse in health_monitor.py) so the
#: composite score has a clean, auditable 1:1 signal→category assignment.
CATEGORY_DIMENSIONS: dict[str, list[str]] = {
    "calibration": [
        DriftDimension.CALIBRATION.value,
        DriftDimension.CONFIDENCE.value,
        DriftDimension.ENTROPY.value,
        DriftDimension.UNCERTAINTY.value,
    ],
    "profitability": [
        DriftDimension.PERFORMANCE.value,
        DriftDimension.DECISION_QUALITY.value,
    ],
    "stability": [
        DriftDimension.DATA_QUALITY.value,
        DriftDimension.REGIME.value,
    ],
    "robustness": [
        DriftDimension.FEATURE_HEALTH.value,
        DriftDimension.UNSEEN_CONDITION.value,
        DriftDimension.BLACK_SWAN.value,
        DriftDimension.MARKET.value,
    ],
    "drift": [
        DriftDimension.FEATURE.value,
    ],
}

# Sanity: every DriftDimension must appear in exactly one category. Verified
# by test_composite_health.py::test_category_partition_covers_all_dimensions.


class HealthGrade(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class CompositeHealthConfig:
    """Weights (must sum to 1.0) and per-dimension-state scores."""

    weight_calibration: float = 0.20
    weight_profitability: float = 0.30
    weight_stability: float = 0.15
    weight_robustness: float = 0.20
    weight_drift: float = 0.15

    # Per-dimension contribution when in a given state (no signal / WARNING /
    # CRITICAL). A dimension with no persisted alert is treated as healthy.
    score_healthy: float = 100.0
    score_warning: float = 60.0
    score_critical: float = 20.0

    # Composite → grade thresholds (composite >= threshold maps to the grade)
    excellent_threshold: float = 90.0
    good_threshold: float = 75.0
    watch_threshold: float = 55.0
    degraded_threshold: float = 35.0
    # Below degraded_threshold → CRITICAL

    def weights(self) -> dict[str, float]:
        return {
            "calibration": self.weight_calibration,
            "profitability": self.weight_profitability,
            "stability": self.weight_stability,
            "robustness": self.weight_robustness,
            "drift": self.weight_drift,
        }

    def validate(self) -> None:
        total = sum(self.weights().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"CompositeHealthConfig weights must sum to 1.0, got {total}")


def grade_for_composite(composite: float, cfg: CompositeHealthConfig) -> HealthGrade:
    if composite >= cfg.excellent_threshold:
        return HealthGrade.EXCELLENT
    if composite >= cfg.good_threshold:
        return HealthGrade.GOOD
    if composite >= cfg.watch_threshold:
        return HealthGrade.WATCH
    if composite >= cfg.degraded_threshold:
        return HealthGrade.DEGRADED
    return HealthGrade.CRITICAL


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class CategoryScore:
    category: str
    score: float  # 0-100
    weight: float
    weighted_score: float
    dimension_states: dict[str, str] = field(default_factory=dict)  # dim -> HEALTHY/WARNING/CRITICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "score": round(self.score, 2),
            "weight": self.weight,
            "weighted_contribution": round(self.weighted_score, 2),
            "dimension_states": self.dimension_states,
        }


@dataclass
class CompositeHealthScore:
    score_id: str = field(default_factory=lambda: f"chs-{uuid.uuid4().hex[:12]}")
    model_version: str = ""
    composite: float = 0.0  # 0-100
    grade: str = HealthGrade.GOOD.value
    category_scores: list[CategoryScore] = field(default_factory=list)
    n_critical_alerts: int = 0
    n_warning_alerts: int = 0
    created_at: float = field(default_factory=time.time)

    def category_score(self, category: str) -> CategoryScore | None:
        for cs in self.category_scores:
            if cs.category == category:
                return cs
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_id": self.score_id,
            "model_version": self.model_version,
            "composite": round(self.composite, 2),
            "grade": self.grade,
            "category_scores": [cs.to_dict() for cs in self.category_scores],
            "n_critical_alerts": self.n_critical_alerts,
            "n_warning_alerts": self.n_warning_alerts,
            "created_at": round(self.created_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompositeHealthScore:
        cats = [
            CategoryScore(
                category=cs["category"],
                score=cs["score"],
                weight=cs["weight"],
                weighted_score=cs["weighted_contribution"],
                dimension_states=cs.get("dimension_states", {}),
            )
            for cs in d.get("category_scores", [])
        ]
        return cls(
            score_id=d.get("score_id", f"chs-{uuid.uuid4().hex[:12]}"),
            model_version=d.get("model_version", ""),
            composite=d.get("composite", 0.0),
            grade=d.get("grade", HealthGrade.GOOD.value),
            category_scores=cats,
            n_critical_alerts=d.get("n_critical_alerts", 0),
            n_warning_alerts=d.get("n_warning_alerts", 0),
            created_at=d.get("created_at", 0.0),
        )


# ── Scorer ────────────────────────────────────────────────────────────────────


class CompositeQDEHealthScorer:
    """Computes and persists the composite QDE Health Score for a model
    version, from the latest per-dimension health-monitor alert state.

    Advisory only: this class has no reference to, and no ability to call,
    any retrain/promotion/rollback execution path. It only reads alerts and
    writes a score record.
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        health_monitor: ResearchLabHealthMonitor,
        config: CompositeHealthConfig | None = None,
    ) -> None:
        self._db = db
        self._monitor = health_monitor
        self._cfg = config or CompositeHealthConfig()
        self._cfg.validate()

    def compute(self, model_version: str | None = None) -> CompositeHealthScore:
        """Compute the composite score from the current persisted health
        status (does NOT re-run the underlying drift checks — call
        ``health_monitor.run_check()`` first if fresh telemetry should be
        evaluated)."""
        status = self._monitor.get_status(model_version=model_version)
        latest_per_dim: dict[str, dict[str, Any]] = status.get("latest_per_dimension", {})

        weights = self._cfg.weights()
        category_scores: list[CategoryScore] = []
        composite = 0.0
        n_critical = 0
        n_warning = 0

        for category, dims in CATEGORY_DIMENSIONS.items():
            dim_states: dict[str, str] = {}
            dim_scores: list[float] = []
            for dim in dims:
                alert = latest_per_dim.get(dim)
                severity = alert.get("severity") if alert else None
                if severity == AlertSeverity.CRITICAL.value:
                    dim_states[dim] = "CRITICAL"
                    dim_scores.append(self._cfg.score_critical)
                    n_critical += 1
                elif severity == AlertSeverity.WARNING.value:
                    dim_states[dim] = "WARNING"
                    dim_scores.append(self._cfg.score_warning)
                    n_warning += 1
                else:
                    dim_states[dim] = "HEALTHY"
                    dim_scores.append(self._cfg.score_healthy)

            cat_score = sum(dim_scores) / len(dim_scores) if dim_scores else self._cfg.score_healthy
            weight = weights[category]
            weighted = cat_score * weight
            composite += weighted
            category_scores.append(
                CategoryScore(
                    category=category,
                    score=cat_score,
                    weight=weight,
                    weighted_score=weighted,
                    dimension_states=dim_states,
                )
            )

        grade = grade_for_composite(composite, self._cfg)

        result = CompositeHealthScore(
            model_version=model_version or "",
            composite=composite,
            grade=grade.value,
            category_scores=category_scores,
            n_critical_alerts=n_critical,
            n_warning_alerts=n_warning,
        )
        self._db.upsert_composite_health_score(result.to_dict())
        logger.info(
            f"CompositeQDEHealthScorer: model={model_version!r} "
            f"composite={composite:.1f} grade={grade.value} "
            f"(critical={n_critical} warning={n_warning})"
        )
        return result

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_latest(self, model_version: str | None = None) -> dict[str, Any] | None:
        rows = self._db.list_composite_health_scores(model_version=model_version, limit=1)
        return rows[0] if rows else None

    def list_scores(
        self,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._db.list_composite_health_scores(
            model_version=model_version, since=since, until=until, limit=limit
        )
