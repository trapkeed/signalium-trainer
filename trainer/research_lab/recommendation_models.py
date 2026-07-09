"""S20 — Retraining Recommendation data models.

RetrainingRecommendation: a structured admin-actionable recommendation that
the live champion model should be retrained (or fine-tuned, recalibrated, etc.).

EvaluationResult: per-dimension scored result produced by the evaluators.

RecommendationConfig: thresholds controlling when each dimension triggers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Enumerations ──────────────────────────────────────────────────────────────


class EvaluationDimension(str, Enum):
    SHARPE_DEGRADATION = "sharpe_degradation"
    CONFIDENCE_DRIFT = "confidence_drift"
    REASONING_DRIFT = "reasoning_drift"
    REGIME_UNDERPERFORMANCE = "regime_underperformance"
    PORTFOLIO_INEFFICIENCY = "portfolio_inefficiency"


class RecommendedScope(str, Enum):
    FULL_RETRAIN = "FULL_RETRAIN"
    FINE_TUNE = "FINE_TUNE"
    CALIBRATION_ONLY = "CALIBRATION_ONLY"
    HEAD_RETRAIN = "HEAD_RETRAIN"


class RecommendationUrgency(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendationStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    DISMISSED = "DISMISSED"


# ── Evaluation result (per-dimension) ─────────────────────────────────────────


@dataclass
class EvaluationResult:
    """Scored result for one evaluation dimension."""

    dimension: EvaluationDimension
    triggered: bool
    score: float  # 0.0 = healthy, >0 = how far past threshold (threshold = 1.0)
    evidence: dict[str, Any]
    recommended_scope: RecommendedScope
    urgency: RecommendationUrgency
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "triggered": self.triggered,
            "score": round(self.score, 4),
            "evidence": self.evidence,
            "recommended_scope": self.recommended_scope.value,
            "urgency": self.urgency.value,
            "summary": self.summary,
        }


# ── Recommendation ────────────────────────────────────────────────────────────


@dataclass
class RetrainingRecommendation:
    """A structured, admin-actionable retraining recommendation.

    Starts in PROPOSED status. Admins approve or dismiss via the API.
    """

    recommendation_id: str
    model_version: str
    trigger_dimension: str           # EvaluationDimension.value
    trigger_reason: str
    evidence: dict[str, Any]
    recommended_scope: str           # RecommendedScope.value
    urgency: str                     # RecommendationUrgency.value
    status: str                      # RecommendationStatus.value
    created_at: float
    updated_at: float
    resolved_by: str = ""
    resolved_at: float | None = None
    notes: str = ""
    # All dimension results from the evaluation run that produced this recommendation
    evaluation_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        model_version: str,
        trigger_dimension: EvaluationDimension,
        trigger_reason: str,
        evidence: dict[str, Any],
        recommended_scope: RecommendedScope,
        urgency: RecommendationUrgency,
        evaluation_summary: dict[str, Any] | None = None,
        notes: str = "",
    ) -> RetrainingRecommendation:
        now = time.time()
        return cls(
            recommendation_id=f"rr-{str(uuid.uuid4())[:12]}",
            model_version=model_version,
            trigger_dimension=trigger_dimension.value,
            trigger_reason=trigger_reason,
            evidence=evidence,
            recommended_scope=recommended_scope.value,
            urgency=urgency.value,
            status=RecommendationStatus.PROPOSED.value,
            created_at=now,
            updated_at=now,
            evaluation_summary=evaluation_summary or {},
            notes=notes,
        )

    def approve(self, resolved_by: str = "admin") -> None:
        if self.status != RecommendationStatus.PROPOSED.value:
            raise ValueError(
                f"Cannot approve recommendation in status {self.status!r} — must be PROPOSED"
            )
        self.status = RecommendationStatus.APPROVED.value
        self.resolved_by = resolved_by
        self.resolved_at = time.time()
        self.updated_at = self.resolved_at

    def dismiss(self, resolved_by: str = "admin", notes: str = "") -> None:
        if self.status != RecommendationStatus.PROPOSED.value:
            raise ValueError(
                f"Cannot dismiss recommendation in status {self.status!r} — must be PROPOSED"
            )
        self.status = RecommendationStatus.DISMISSED.value
        self.resolved_by = resolved_by
        self.resolved_at = time.time()
        self.updated_at = self.resolved_at
        if notes:
            self.notes = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "model_version": self.model_version,
            "trigger_dimension": self.trigger_dimension,
            "trigger_reason": self.trigger_reason,
            "evidence": self.evidence,
            "recommended_scope": self.recommended_scope,
            "urgency": self.urgency,
            "status": self.status,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "resolved_by": self.resolved_by,
            "resolved_at": round(self.resolved_at, 3) if self.resolved_at else None,
            "notes": self.notes,
            "evaluation_summary": self.evaluation_summary,
        }


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class RecommendationConfig:
    """Thresholds and window sizes for all five evaluation dimensions."""

    # Minimum trade count before Sharpe degradation fires
    min_trades_for_evaluation: int = 5
    # Window sizes (number of trades / decisions)
    baseline_window: int = 200
    rolling_window: int = 50

    # ── Per-dimension thresholds ───────────────────────────────────────────────

    # Sharpe degradation: fraction drop from baseline that triggers
    # e.g. 0.25 → trigger if recent_sharpe < baseline_sharpe * 0.75
    sharpe_degradation_threshold: float = 0.25

    # Confidence drift: absolute shift in mean confidence
    confidence_drift_threshold: float = 0.08

    # Reasoning drift: mean absolute TV distance across output heads
    reasoning_drift_threshold: float = 0.15

    # Regime underperformance: per-regime Sharpe threshold below which a regime
    # is considered underperforming (0.0 = negative Sharpe triggers)
    regime_sharpe_threshold: float = 0.0

    # Portfolio inefficiency: Herfindahl-style concentration index
    # 1/N for N equally weighted symbols; above this suggests over-concentration
    portfolio_concentration_threshold: float = 0.5


def _urgency_from_score(score: float) -> RecommendationUrgency:
    """Map a normalized excess score to urgency.

    score = (observed_excess / threshold), so:
        score < 0.5  → LOW
        0.5–1.5     → MEDIUM
        1.5–3.0     → HIGH
        ≥ 3.0       → CRITICAL
    """
    if score < 0.5:
        return RecommendationUrgency.LOW
    if score < 1.5:
        return RecommendationUrgency.MEDIUM
    if score < 3.0:
        return RecommendationUrgency.HIGH
    return RecommendationUrgency.CRITICAL


def _elevate_urgency(
    base: RecommendationUrgency, n_triggered: int
) -> RecommendationUrgency:
    """Bump urgency when multiple dimensions fire simultaneously."""
    if n_triggered >= 4:
        return RecommendationUrgency.CRITICAL
    if n_triggered >= 3:
        order = [u for u in RecommendationUrgency]
        idx = order.index(base)
        return order[min(idx + 2, len(order) - 1)]
    if n_triggered >= 2:
        order = [u for u in RecommendationUrgency]
        idx = order.index(base)
        return order[min(idx + 1, len(order) - 1)]
    return base
