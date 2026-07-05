"""Phase 19 — Retraining Advisor.

Advisory engine for retraining recommendations.

IMPORTANT: This module NEVER automatically retrains any model.
All recommendations are advisory and require explicit human action.

Triggers:
  - Feature drift
  - Performance degradation
  - Calibration degradation
  - Market evolution
  - New datasets available
  - Improved labels
  - New feature releases

Every recommendation includes:
  - Reason
  - Expected benefit
  - Estimated risk
  - Priority (1–5, 1 = highest)
  - Confidence (0–1)

Usage::

    advisor = RetrainingAdvisor()
    recs = advisor.evaluate("model_01",
        context={"performance_degradation": 0.08,
                 "calibration_drift": 0.06,
                 "new_dataset_available": True})
    for rec in recs:
        print(rec.reason, rec.priority, rec.confidence)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Retraining trigger
# ---------------------------------------------------------------------------


class RetrainingTrigger:
    FEATURE_DRIFT           = "feature_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CALIBRATION_DEGRADATION = "calibration_degradation"
    MARKET_EVOLUTION        = "market_evolution"
    NEW_DATASET             = "new_dataset"
    IMPROVED_LABELS         = "improved_labels"
    NEW_FEATURES            = "new_features"
    SCHEDULED               = "scheduled"
    MANUAL                  = "manual"
    CONFIDENCE_DEGRADATION  = "confidence_degradation"
    REGIME_SHIFT            = "regime_shift"

    ALL = [
        FEATURE_DRIFT, PERFORMANCE_DEGRADATION, CALIBRATION_DEGRADATION,
        MARKET_EVOLUTION, NEW_DATASET, IMPROVED_LABELS, NEW_FEATURES,
        SCHEDULED, MANUAL, CONFIDENCE_DEGRADATION, REGIME_SHIFT,
    ]


# ---------------------------------------------------------------------------
# Retraining recommendation
# ---------------------------------------------------------------------------


@dataclass
class RetrainingRecommendation:
    """One retraining recommendation (advisory only).

    Attributes:
        recommendation_id: Unique UUID.
        model_id: Model recommended for retraining.
        trigger: RetrainingTrigger value.
        reason: Human-readable explanation.
        expected_benefit: Quantified expected improvement.
        estimated_risk: Risk of retraining (data drift, regression, etc.).
        priority: 1 (highest) to 5 (lowest).
        confidence: Advisor confidence in this recommendation (0–1).
        status: pending / scheduled / dismissed / acted_upon.
        context: Metrics/evidence that triggered this recommendation.
        created_at: UTC ISO-8601 creation timestamp.
        acted_at: When a human acted on this (if any).
        acted_by: Who acted (if any).
    """

    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    trigger: str = RetrainingTrigger.MANUAL
    reason: str = ""
    expected_benefit: str = ""
    estimated_risk: str = ""
    priority: int = 3
    confidence: float = 0.5
    status: str = "pending"
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    acted_at: Optional[str] = None
    acted_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "model_id": self.model_id,
            "trigger": self.trigger,
            "reason": self.reason,
            "expected_benefit": self.expected_benefit,
            "estimated_risk": self.estimated_risk,
            "priority": self.priority,
            "confidence": self.confidence,
            "status": self.status,
            "context": self.context,
            "created_at": self.created_at,
            "acted_at": self.acted_at,
            "acted_by": self.acted_by,
            "note": "ADVISORY ONLY — no automatic retraining.",
        }


# ---------------------------------------------------------------------------
# Advisor rule
# ---------------------------------------------------------------------------


@dataclass
class AdvisorRule:
    """One trigger rule for the retraining advisor."""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: str = RetrainingTrigger.PERFORMANCE_DEGRADATION
    context_key: str = ""
    threshold: float = 0.05
    threshold_direction: str = "above"   # above | below
    priority: int = 2
    confidence: float = 0.70
    expected_benefit: str = ""
    estimated_risk: str = "Low"
    enabled: bool = True
    cooldown_hours: float = 24.0
    last_triggered: Optional[str] = None

    def is_in_cooldown(self) -> bool:
        if not self.last_triggered:
            return False
        from datetime import timedelta
        last = datetime.fromisoformat(self.last_triggered)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).total_seconds() < self.cooldown_hours * 3600

    def evaluates(self, context: Dict[str, Any]) -> bool:
        if not self.enabled or self.is_in_cooldown():
            return False
        val = context.get(self.context_key)
        if val is None:
            return False
        if not isinstance(val, (int, float, bool)):
            return bool(val)
        if self.threshold_direction == "above":
            return float(val) >= self.threshold
        else:
            return float(val) <= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "trigger": self.trigger,
            "context_key": self.context_key,
            "threshold": self.threshold,
            "priority": self.priority,
            "confidence": self.confidence,
            "enabled": self.enabled,
            "cooldown_hours": self.cooldown_hours,
            "in_cooldown": self.is_in_cooldown(),
        }


# ---------------------------------------------------------------------------
# Retraining advisor
# ---------------------------------------------------------------------------


class RetrainingAdvisor:
    """Generates retraining recommendations from context metrics.

    CRITICAL: No automatic retraining is ever triggered.
    All recommendations require explicit human action.

    Args:
        auto_retrain: MUST remain False. Raises ValueError if True.
    """

    def __init__(self, auto_retrain: bool = False) -> None:
        if auto_retrain:
            raise ValueError(
                "auto_retrain=True is not permitted. "
                "Retraining is advisory only."
            )
        self.auto_retrain = False
        self._rules: Dict[str, AdvisorRule] = {}
        self._recommendations: List[RetrainingRecommendation] = []

    def add_rule(self, rule: AdvisorRule) -> str:
        self._rules[rule.rule_id] = rule
        return rule.rule_id

    def remove_rule(self, rule_id: str) -> bool:
        return bool(self._rules.pop(rule_id, None))

    def evaluate(self, model_id: str,
                  context: Dict[str, Any],
                  force_manual: bool = False) -> List[RetrainingRecommendation]:
        """Evaluate all rules against context and return recommendations.

        Args:
            model_id: Model to evaluate.
            context: Dict of metric key → value.
                     Keys include: performance_degradation, calibration_drift,
                     feature_drift, confidence_degradation, new_dataset_available,
                     improved_labels_available, new_features_available,
                     market_evolution_detected, regime_shift_detected.
            force_manual: If True, always generate a manual recommendation.

        Returns:
            List of RetrainingRecommendation (advisory only).
        """
        recs = []
        for rule in self._rules.values():
            if rule.evaluates(context):
                val = context.get(rule.context_key)
                rec = RetrainingRecommendation(
                    model_id=model_id,
                    trigger=rule.trigger,
                    reason=self._build_reason(rule, val, context),
                    expected_benefit=rule.expected_benefit,
                    estimated_risk=rule.estimated_risk,
                    priority=rule.priority,
                    confidence=rule.confidence,
                    context={k: round(v, 4) if isinstance(v, float) else v
                              for k, v in context.items()},
                )
                self._recommendations.append(rec)
                recs.append(rec)
                rule.last_triggered = rec.created_at

        if force_manual:
            rec = RetrainingRecommendation(
                model_id=model_id,
                trigger=RetrainingTrigger.MANUAL,
                reason="Manual retraining request",
                expected_benefit="Depends on scope of retraining",
                estimated_risk="Medium — requires validation",
                priority=1,
                confidence=1.0,
                context=context,
            )
            self._recommendations.append(rec)
            recs.append(rec)

        return recs

    def _build_reason(self, rule: AdvisorRule, value: Any,
                       context: Dict[str, Any]) -> str:
        if isinstance(value, bool):
            return (f"{rule.trigger}: {rule.context_key} is available. "
                    f"Retraining recommended.")
        if isinstance(value, (int, float)):
            direction = "exceeds" if rule.threshold_direction == "above" else "below"
            return (
                f"{rule.trigger}: {rule.context_key}={value:.4f} "
                f"{direction} threshold {rule.threshold:.4f}."
            )
        return f"{rule.trigger}: condition met for {rule.context_key}."

    def acknowledge(self, recommendation_id: str,
                    acted_by: str = "human") -> bool:
        for r in self._recommendations:
            if r.recommendation_id == recommendation_id:
                r.status = "acted_upon"
                r.acted_at = datetime.now(timezone.utc).isoformat()
                r.acted_by = acted_by
                return True
        return False

    def dismiss(self, recommendation_id: str) -> bool:
        for r in self._recommendations:
            if r.recommendation_id == recommendation_id:
                r.status = "dismissed"
                return True
        return False

    def pending(self, model_id: Optional[str] = None) -> List[Dict[str, Any]]:
        recs = [r for r in self._recommendations if r.status == "pending"]
        if model_id:
            recs = [r for r in recs if r.model_id == model_id]
        return [r.to_dict() for r in sorted(recs, key=lambda r: r.priority)]

    def all_recommendations(self, model_id: Optional[str] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
        recs = list(reversed(self._recommendations))
        if model_id:
            recs = [r for r in recs if r.model_id == model_id]
        return [r.to_dict() for r in recs[:limit]]

    def stats(self) -> Dict[str, Any]:
        return {
            "total_rules": len(self._rules),
            "active_rules": sum(1 for r in self._rules.values() if r.enabled),
            "total_recommendations": len(self._recommendations),
            "pending": sum(1 for r in self._recommendations if r.status == "pending"),
            "acted_upon": sum(1 for r in self._recommendations if r.status == "acted_upon"),
            "dismissed": sum(1 for r in self._recommendations if r.status == "dismissed"),
            "auto_retrain": False,
            "note": "ADVISORY ONLY — no automatic retraining.",
        }


def build_default_advisor() -> RetrainingAdvisor:
    """Build the default retraining advisor with standard rules."""
    advisor = RetrainingAdvisor(auto_retrain=False)
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.PERFORMANCE_DEGRADATION,
        context_key="performance_degradation",
        threshold=0.05, threshold_direction="above",
        priority=1, confidence=0.85, cooldown_hours=24,
        expected_benefit="Expected accuracy recovery of 4–8%",
        estimated_risk="Low — same architecture, updated data",
    ))
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.CALIBRATION_DEGRADATION,
        context_key="calibration_drift",
        threshold=0.05, threshold_direction="above",
        priority=2, confidence=0.75, cooldown_hours=48,
        expected_benefit="Improved calibration and confidence reliability",
        estimated_risk="Low",
    ))
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.FEATURE_DRIFT,
        context_key="feature_drift",
        threshold=0.25, threshold_direction="above",
        priority=2, confidence=0.70, cooldown_hours=48,
        expected_benefit="Alignment with current feature distributions",
        estimated_risk="Medium — feature space may have shifted significantly",
    ))
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.CONFIDENCE_DEGRADATION,
        context_key="confidence_degradation",
        threshold=0.10, threshold_direction="above",
        priority=2, confidence=0.65, cooldown_hours=24,
        expected_benefit="Restored prediction confidence",
        estimated_risk="Low",
    ))
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.NEW_DATASET,
        context_key="new_dataset_available",
        threshold=0.5, threshold_direction="above",
        priority=3, confidence=0.90, cooldown_hours=168,
        expected_benefit="Exposure to new data patterns and regimes",
        estimated_risk="Low — additional training data",
    ))
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.IMPROVED_LABELS,
        context_key="improved_labels_available",
        threshold=0.5, threshold_direction="above",
        priority=3, confidence=0.85, cooldown_hours=168,
        expected_benefit="Better label quality → improved accuracy",
        estimated_risk="Low",
    ))
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.MARKET_EVOLUTION,
        context_key="market_evolution_detected",
        threshold=0.5, threshold_direction="above",
        priority=2, confidence=0.60, cooldown_hours=72,
        expected_benefit="Adaptation to structural market change",
        estimated_risk="Medium — new regime may have limited historical data",
    ))
    advisor.add_rule(AdvisorRule(
        trigger=RetrainingTrigger.REGIME_SHIFT,
        context_key="regime_shift_detected",
        threshold=0.5, threshold_direction="above",
        priority=1, confidence=0.75, cooldown_hours=48,
        expected_benefit="Regime-specific adaptation",
        estimated_risk="Medium",
    ))
    return advisor
