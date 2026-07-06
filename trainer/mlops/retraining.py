"""Phase 18 — Retraining Policy Manager.

Configurable retraining trigger evaluation — RECOMMENDATIONS ONLY.

IMPORTANT: This module NEVER automatically retrains any model.
It evaluates whether retraining is recommended and records the
recommendation for human review.

Supported retraining triggers:
  - Performance degradation (accuracy drops below threshold)
  - Confidence degradation (avg confidence drops)
  - Feature drift (distribution shift detected)
  - Label drift
  - Calibration drift
  - Scheduled retraining (time-based)
  - Manual retraining request

Usage::

    policies = RetrainingPolicyManager()
    policies.add_policy(RetrainingPolicy(
        policy_id="perf_degradation",
        trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
        threshold=0.05,
        cooldown_hours=24,
    ))
    rec = policies.evaluate("model_01", metrics)
    if rec:
        print(rec.reason)  # human-readable reason
        # NO automatic retraining — only a recommendation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Retraining triggers
# ---------------------------------------------------------------------------


class RetrainingTrigger:
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONFIDENCE_DEGRADATION = "confidence_degradation"
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    CALIBRATION_DRIFT = "calibration_drift"
    REGIME_DRIFT = "regime_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"

    ALL = [
        PERFORMANCE_DEGRADATION,
        CONFIDENCE_DEGRADATION,
        FEATURE_DRIFT,
        LABEL_DRIFT,
        CALIBRATION_DRIFT,
        REGIME_DRIFT,
        SCHEDULED,
        MANUAL,
    ]


# ---------------------------------------------------------------------------
# Retraining policy
# ---------------------------------------------------------------------------


@dataclass
class RetrainingPolicy:
    """One retraining trigger policy.

    Attributes:
        policy_id: Unique ID for this policy.
        trigger: RetrainingTrigger value.
        threshold: Metric threshold that triggers the recommendation.
        enabled: Whether this policy is active.
        cooldown_hours: Minimum hours between recommendations from this policy.
        priority: Recommendation priority (1 = highest).
        description: Human-readable policy description.
        last_triggered: ISO-8601 timestamp of last triggered recommendation.
    """

    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: str = RetrainingTrigger.PERFORMANCE_DEGRADATION
    threshold: float = 0.05
    enabled: bool = True
    cooldown_hours: float = 24.0
    priority: int = 2
    description: str = ""
    last_triggered: Optional[str] = None

    def is_in_cooldown(self) -> bool:
        if self.last_triggered is None:
            return False
        last = datetime.fromisoformat(self.last_triggered)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() < self.cooldown_hours * 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "trigger": self.trigger,
            "threshold": self.threshold,
            "enabled": self.enabled,
            "cooldown_hours": self.cooldown_hours,
            "priority": self.priority,
            "description": self.description,
            "last_triggered": self.last_triggered,
            "in_cooldown": self.is_in_cooldown(),
        }


# ---------------------------------------------------------------------------
# Retraining recommendation
# ---------------------------------------------------------------------------


@dataclass
class RetrainingRecommendation:
    """A recommendation to retrain a model.

    This is NEVER acted upon automatically.
    It is recorded for human review only.

    Attributes:
        recommendation_id: Unique UUID.
        model_id: Model recommended for retraining.
        trigger: What triggered this recommendation.
        reason: Human-readable explanation.
        priority: Priority level (1 = highest).
        status: "pending" | "scheduled" | "dismissed" | "acted_upon".
        metrics: Metric values that triggered the recommendation.
        policy_id: Policy that generated this recommendation.
        created_at: UTC ISO-8601 creation timestamp.
        acted_at: When a human acted on this recommendation.
        acted_by: Who acted on this recommendation.
    """

    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    trigger: str = ""
    reason: str = ""
    priority: int = 2
    status: str = "pending"
    metrics: Dict[str, float] = field(default_factory=dict)
    policy_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    acted_at: Optional[str] = None
    acted_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "model_id": self.model_id,
            "trigger": self.trigger,
            "reason": self.reason,
            "priority": self.priority,
            "status": self.status,
            "metrics": self.metrics,
            "policy_id": self.policy_id,
            "created_at": self.created_at,
            "acted_at": self.acted_at,
            "acted_by": self.acted_by,
            "note": "RECOMMENDATION ONLY — no automatic retraining.",
        }


# ---------------------------------------------------------------------------
# Policy manager
# ---------------------------------------------------------------------------


class RetrainingPolicyManager:
    """Evaluates retraining policies and generates recommendations.

    CRITICAL: This manager NEVER retrains any model.
    All recommendations are advisory only and require human action.

    Args:
        auto_retrain: MUST remain False. Automatic retraining is forbidden.
    """

    def __init__(self, auto_retrain: bool = False) -> None:
        if auto_retrain:
            raise ValueError(
                "auto_retrain=True is not permitted. "
                "Retraining recommendations are advisory only."
            )
        self.auto_retrain = False
        self._policies: Dict[str, RetrainingPolicy] = {}
        self._recommendations: List[RetrainingRecommendation] = []

    def add_policy(self, policy: RetrainingPolicy) -> str:
        """Register a retraining policy. Returns policy_id."""
        self._policies[policy.policy_id] = policy
        logger.info(f"Retraining policy added: {policy.policy_id} ({policy.trigger})")
        return policy.policy_id

    def remove_policy(self, policy_id: str) -> bool:
        return bool(self._policies.pop(policy_id, None))

    def get_policy(self, policy_id: str) -> Optional[RetrainingPolicy]:
        return self._policies.get(policy_id)

    def list_policies(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self._policies.values()]

    def evaluate(
        self, model_id: str, metrics: Dict[str, float], force_manual: bool = False
    ) -> List[RetrainingRecommendation]:
        """Evaluate all active policies against current metrics.

        Returns a list of retraining recommendations (advisory only).
        Never triggers automatic retraining.

        Args:
            model_id: Model to evaluate.
            metrics: Current model metrics dict. Keys include:
                     performance_degradation, confidence_degradation,
                     calibration_drift, feature_drift, distribution_drift,
                     regime_drift, accuracy, avg_confidence.
            force_manual: If True, always generate a manual recommendation.

        Returns:
            List of RetrainingRecommendation objects.
        """
        recommendations = []
        metric_map = {
            RetrainingTrigger.PERFORMANCE_DEGRADATION: "performance_degradation",
            RetrainingTrigger.CONFIDENCE_DEGRADATION: "confidence_degradation",
            RetrainingTrigger.CALIBRATION_DRIFT: "calibration_drift",
            RetrainingTrigger.FEATURE_DRIFT: "feature_drift",
            RetrainingTrigger.REGIME_DRIFT: "regime_drift",
        }

        for policy in self._policies.values():
            if not policy.enabled:
                continue
            if policy.is_in_cooldown():
                continue

            should_recommend = False
            trigger_value: Optional[float] = None

            if policy.trigger == RetrainingTrigger.MANUAL and force_manual:
                should_recommend = True
            elif policy.trigger == RetrainingTrigger.SCHEDULED:
                should_recommend = True
            elif policy.trigger in metric_map:
                metric_key = metric_map[policy.trigger]
                metric_val = metrics.get(metric_key, 0.0)
                trigger_value = metric_val
                should_recommend = metric_val >= policy.threshold
            else:
                metric_val = metrics.get(policy.trigger, 0.0)
                trigger_value = metric_val
                should_recommend = metric_val >= policy.threshold

            if should_recommend:
                reason = self._build_reason(policy, trigger_value, metrics)
                rec = RetrainingRecommendation(
                    model_id=model_id,
                    trigger=policy.trigger,
                    reason=reason,
                    priority=policy.priority,
                    metrics={k: round(v, 4) for k, v in metrics.items()},
                    policy_id=policy.policy_id,
                )
                self._recommendations.append(rec)
                policy.last_triggered = rec.created_at
                recommendations.append(rec)
                logger.warning(
                    f"Retraining RECOMMENDED for {model_id} "
                    f"(trigger={policy.trigger}): {reason}"
                )

        if force_manual and not any(r.trigger == RetrainingTrigger.MANUAL for r in recommendations):
            rec = RetrainingRecommendation(
                model_id=model_id,
                trigger=RetrainingTrigger.MANUAL,
                reason="Manual retraining request",
                priority=1,
                metrics=metrics,
                policy_id="manual",
            )
            self._recommendations.append(rec)
            recommendations.append(rec)

        return recommendations

    def _build_reason(
        self, policy: RetrainingPolicy, value: Optional[float], metrics: Dict[str, float]
    ) -> str:
        if value is not None:
            return (
                f"{policy.trigger}: observed {value:.4f} ≥ "
                f"threshold {policy.threshold:.4f}. "
                f"{policy.description}"
            ).strip()
        return f"{policy.trigger}: policy triggered. {policy.description}".strip()

    def acknowledge(self, recommendation_id: str, acted_by: str = "human") -> bool:
        """Mark a recommendation as acted upon."""
        for r in self._recommendations:
            if r.recommendation_id == recommendation_id:
                r.status = "acted_upon"
                r.acted_at = datetime.now(timezone.utc).isoformat()
                r.acted_by = acted_by
                return True
        return False

    def dismiss(self, recommendation_id: str) -> bool:
        """Dismiss a recommendation."""
        for r in self._recommendations:
            if r.recommendation_id == recommendation_id:
                r.status = "dismissed"
                return True
        return False

    def pending_recommendations(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        recs = [r for r in self._recommendations if r.status == "pending"]
        if model_id:
            recs = [r for r in recs if r.model_id == model_id]
        return [r.to_dict() for r in sorted(recs, key=lambda r: r.priority)]

    def all_recommendations(
        self, model_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        recs = list(reversed(self._recommendations))
        if model_id:
            recs = [r for r in recs if r.model_id == model_id]
        return [r.to_dict() for r in recs[:limit]]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_policies": len(self._policies),
            "active_policies": sum(1 for p in self._policies.values() if p.enabled),
            "total_recommendations": len(self._recommendations),
            "pending": sum(1 for r in self._recommendations if r.status == "pending"),
            "acted_upon": sum(1 for r in self._recommendations if r.status == "acted_upon"),
            "dismissed": sum(1 for r in self._recommendations if r.status == "dismissed"),
            "auto_retrain": self.auto_retrain,
            "note": "RECOMMENDATION ONLY — no automatic retraining.",
        }


def build_default_policies() -> RetrainingPolicyManager:
    """Build the default set of retraining policies."""
    mgr = RetrainingPolicyManager(auto_retrain=False)
    mgr.add_policy(
        RetrainingPolicy(
            policy_id="perf_degradation_default",
            trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
            threshold=0.05,
            cooldown_hours=24,
            priority=1,
            description="Accuracy dropped 5%+ below baseline.",
        )
    )
    mgr.add_policy(
        RetrainingPolicy(
            policy_id="confidence_degradation_default",
            trigger=RetrainingTrigger.CONFIDENCE_DEGRADATION,
            threshold=0.10,
            cooldown_hours=24,
            priority=2,
            description="Average confidence dropped 10%+ below baseline.",
        )
    )
    mgr.add_policy(
        RetrainingPolicy(
            policy_id="calibration_drift_default",
            trigger=RetrainingTrigger.CALIBRATION_DRIFT,
            threshold=0.05,
            cooldown_hours=48,
            priority=2,
            description="Calibration error drifted 5%+ from baseline.",
        )
    )
    mgr.add_policy(
        RetrainingPolicy(
            policy_id="regime_drift_default",
            trigger=RetrainingTrigger.REGIME_DRIFT,
            threshold=0.20,
            cooldown_hours=72,
            priority=3,
            description="Market regime distribution shifted significantly.",
        )
    )
    return mgr
