"""Phase 19 — Experiment Recommendation Engine.

Automatically suggests new research experiments:

  - Try new backbone architectures
  - Increase sequence length
  - Reduce dropout
  - Test new loss weighting
  - Evaluate new features
  - Remove deprecated features
  - Train on additional assets
  - Improve calibration

All suggestions are advisory — none are automatically executed.

Usage::

    engine = ExperimentRecommendationEngine()
    experiments = engine.generate(context={
        "current_accuracy": 0.68,
        "current_calibration_error": 0.06,
        "deprecated_features": ["macd_signal_9"],
        "available_backbones": ["transformer_4h", "tcn_256"],
    })
    for exp in experiments:
        print(exp.title, exp.priority)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Experiment category
# ---------------------------------------------------------------------------


class ExperimentCategory:
    ARCHITECTURE         = "architecture"
    HYPERPARAMETER       = "hyperparameter"
    FEATURE_ENGINEERING  = "feature_engineering"
    DATA                 = "data"
    CALIBRATION          = "calibration"
    REGULARIZATION       = "regularization"
    LOSS_FUNCTION        = "loss_function"
    LABEL_QUALITY        = "label_quality"
    REGIME_SPECIALIZATION= "regime_specialization"
    ENSEMBLE             = "ensemble"

    ALL = [
        ARCHITECTURE, HYPERPARAMETER, FEATURE_ENGINEERING, DATA,
        CALIBRATION, REGULARIZATION, LOSS_FUNCTION, LABEL_QUALITY,
        REGIME_SPECIALIZATION, ENSEMBLE,
    ]


# ---------------------------------------------------------------------------
# Experiment suggestion
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSuggestion:
    """One experiment suggestion from the recommendation engine.

    Attributes:
        suggestion_id: Unique UUID.
        title: Short experiment title.
        category: ExperimentCategory value.
        description: Detailed experiment description.
        hypothesis: What is expected to improve.
        expected_benefit: Quantified expected gain.
        estimated_risk: Risk of running this experiment.
        estimated_duration: Rough training time estimate.
        priority: 1 (highest) to 5 (lowest).
        confidence: Engine confidence in this suggestion (0–1).
        prerequisites: List of conditions that must be met first.
        config_changes: Dict of suggested config changes.
        status: pending / scheduled / running / completed / abandoned.
        trigger_context: Context that generated this suggestion.
        created_at: UTC ISO-8601 creation timestamp.
    """

    suggestion_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    category: str = ExperimentCategory.HYPERPARAMETER
    description: str = ""
    hypothesis: str = ""
    expected_benefit: str = ""
    estimated_risk: str = "Low"
    estimated_duration: str = ""
    priority: int = 3
    confidence: float = 0.5
    prerequisites: List[str] = field(default_factory=list)
    config_changes: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    trigger_context: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "title": self.title,
            "category": self.category,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "expected_benefit": self.expected_benefit,
            "estimated_risk": self.estimated_risk,
            "estimated_duration": self.estimated_duration,
            "priority": self.priority,
            "confidence": self.confidence,
            "prerequisites": self.prerequisites,
            "config_changes": self.config_changes,
            "status": self.status,
            "trigger_context": self.trigger_context,
            "created_at": self.created_at,
            "note": "ADVISORY ONLY — no automatic execution.",
        }


# ---------------------------------------------------------------------------
# Experiment generator
# ---------------------------------------------------------------------------


class ExperimentRecommendationEngine:
    """Generates experiment suggestions based on current platform context.

    All suggestions are advisory — none are automatically executed.
    """

    def __init__(self) -> None:
        self._suggestions: List[ExperimentSuggestion] = []

    def generate(self, context: Dict[str, Any]) -> List[ExperimentSuggestion]:
        """Generate experiment suggestions from context.

        Context keys (all optional):
            current_accuracy: float — current model accuracy
            current_calibration_error: float — current ECE
            current_dropout: float — current dropout rate
            current_sequence_length: int — current sequence window
            current_backbone: str — current backbone name
            available_backbones: List[str] — backbones to try
            deprecated_features: List[str] — features to remove
            new_features: List[str] — new features to evaluate
            available_instruments: List[str] — new instruments
            current_instruments: List[str] — current instruments
            degrading_regimes: List[str] — regimes where model underperforms
            high_calibration_error: bool
            feature_drift_detected: bool

        Returns:
            List of ExperimentSuggestion (advisory only).
        """
        generated = []
        generated.extend(self._check_architecture(context))
        generated.extend(self._check_hyperparameters(context))
        generated.extend(self._check_feature_engineering(context))
        generated.extend(self._check_calibration(context))
        generated.extend(self._check_data_expansion(context))
        generated.extend(self._check_regime_specialization(context))
        for s in generated:
            self._suggestions.append(s)
        return generated

    def _check_architecture(self, ctx: Dict[str, Any]) -> List[ExperimentSuggestion]:
        results = []
        backbones = ctx.get("available_backbones", [])
        current = ctx.get("current_backbone", "")
        acc = ctx.get("current_accuracy", 1.0)
        for bb in backbones:
            if bb != current:
                results.append(ExperimentSuggestion(
                    title=f"Try backbone: {bb}",
                    category=ExperimentCategory.ARCHITECTURE,
                    description=f"Replace {current!r} with {bb!r}",
                    hypothesis="New backbone may capture temporal patterns more effectively",
                    expected_benefit="2–8% accuracy improvement potential",
                    estimated_risk="Medium",
                    estimated_duration="4–12 hours",
                    priority=2 if acc < 0.65 else 3,
                    confidence=0.55,
                    config_changes={"model.backbone": bb},
                    trigger_context=ctx,
                ))
        if acc < 0.60:
            results.append(ExperimentSuggestion(
                title="Increase model depth",
                category=ExperimentCategory.ARCHITECTURE,
                description="Add additional layer to the current backbone",
                hypothesis="Deeper model may capture more complex patterns",
                expected_benefit="1–5% accuracy improvement",
                estimated_risk="Medium — increased overfitting risk",
                estimated_duration="6–10 hours",
                priority=3,
                confidence=0.50,
                trigger_context=ctx,
            ))
        return results

    def _check_hyperparameters(self, ctx: Dict[str, Any]) -> List[ExperimentSuggestion]:
        results = []
        seq_len = ctx.get("current_sequence_length", 0)
        dropout = ctx.get("current_dropout", 0.0)
        acc = ctx.get("current_accuracy", 1.0)

        if seq_len and seq_len < 128:
            results.append(ExperimentSuggestion(
                title=f"Increase sequence length: {seq_len} → {seq_len * 2}",
                category=ExperimentCategory.HYPERPARAMETER,
                description=(f"Double the sequence length from {seq_len} to {seq_len * 2} "
                              f"to capture longer-term dependencies"),
                hypothesis="Longer context window improves trend detection",
                expected_benefit="1–4% accuracy improvement on trending regimes",
                estimated_risk="Low — standard hyperparameter change",
                estimated_duration="2–6 hours",
                priority=3,
                confidence=0.60,
                config_changes={"model.sequence_length": seq_len * 2},
                trigger_context=ctx,
            ))

        if dropout and dropout > 0.3 and acc > 0.65:
            results.append(ExperimentSuggestion(
                title=f"Reduce dropout: {dropout:.2f} → {dropout * 0.7:.2f}",
                category=ExperimentCategory.REGULARIZATION,
                description="Reduce dropout to allow more model capacity",
                hypothesis="Model may be under-fitting due to aggressive regularization",
                expected_benefit="0.5–2% accuracy improvement",
                estimated_risk="Low",
                estimated_duration="2–4 hours",
                priority=4,
                confidence=0.55,
                config_changes={"model.dropout": round(dropout * 0.7, 3)},
                trigger_context=ctx,
            ))

        results.append(ExperimentSuggestion(
            title="Experiment with asymmetric loss weighting",
            category=ExperimentCategory.LOSS_FUNCTION,
            description="Weight buy/sell signals higher than no_trade in the loss function",
            hypothesis="Improves precision on actionable signals",
            expected_benefit="Improved signal precision, reduced false positives",
            estimated_risk="Medium — may reduce no_trade recall",
            estimated_duration="3–6 hours",
            priority=3,
            confidence=0.60,
            trigger_context=ctx,
        ))
        return results

    def _check_feature_engineering(self, ctx: Dict[str, Any]) -> List[ExperimentSuggestion]:
        results = []
        deprecated = ctx.get("deprecated_features", [])
        new_features = ctx.get("new_features", [])

        for feat in deprecated[:5]:
            results.append(ExperimentSuggestion(
                title=f"Remove deprecated feature: {feat}",
                category=ExperimentCategory.FEATURE_ENGINEERING,
                description=f"Remove {feat!r} from training — marked deprecated",
                hypothesis="Removing noisy features improves generalization",
                expected_benefit="Cleaner feature space, faster training",
                estimated_risk="Low",
                estimated_duration="1–3 hours",
                priority=3,
                confidence=0.75,
                config_changes={"features.exclude": [feat]},
                trigger_context=ctx,
            ))

        for feat in new_features[:5]:
            results.append(ExperimentSuggestion(
                title=f"Evaluate new feature: {feat}",
                category=ExperimentCategory.FEATURE_ENGINEERING,
                description=f"Add {feat!r} to feature set and measure contribution",
                hypothesis="New feature may provide complementary signal",
                expected_benefit="0.5–3% accuracy improvement if predictive",
                estimated_risk="Low — additive experiment",
                estimated_duration="2–4 hours",
                priority=3,
                confidence=0.50,
                config_changes={"features.include": [feat]},
                trigger_context=ctx,
            ))

        if ctx.get("feature_drift_detected"):
            results.append(ExperimentSuggestion(
                title="Retrain feature normalization statistics",
                category=ExperimentCategory.FEATURE_ENGINEERING,
                description="Update mean/std statistics used for feature normalization",
                hypothesis="Drift-aligned normalization restores feature quality",
                expected_benefit="Reduced feature drift, improved model stability",
                estimated_risk="Low",
                estimated_duration="1–2 hours",
                priority=2,
                confidence=0.80,
                trigger_context=ctx,
            ))
        return results

    def _check_calibration(self, ctx: Dict[str, Any]) -> List[ExperimentSuggestion]:
        results = []
        ece = ctx.get("current_calibration_error", 0.0)
        if ece > 0.06 or ctx.get("high_calibration_error"):
            results.append(ExperimentSuggestion(
                title="Apply temperature scaling post-training",
                category=ExperimentCategory.CALIBRATION,
                description=(
                    "Apply temperature scaling to logits after training "
                    f"(current ECE={ece:.3f})"
                ),
                hypothesis="Temperature scaling reduces calibration error without retraining",
                expected_benefit=f"ECE reduction from {ece:.3f} to ~0.02–0.03",
                estimated_risk="Very low — post-hoc, does not change weights",
                estimated_duration="< 1 hour",
                priority=1,
                confidence=0.90,
                trigger_context=ctx,
            ))
            results.append(ExperimentSuggestion(
                title="Add label smoothing to loss function",
                category=ExperimentCategory.CALIBRATION,
                description="Use label smoothing ε=0.1 to prevent overconfidence",
                hypothesis="Label smoothing regularization improves calibration",
                expected_benefit="ECE improvement of 0.02–0.04",
                estimated_risk="Low",
                estimated_duration="2–4 hours",
                priority=2,
                confidence=0.70,
                config_changes={"training.label_smoothing": 0.1},
                trigger_context=ctx,
            ))
        return results

    def _check_data_expansion(self, ctx: Dict[str, Any]) -> List[ExperimentSuggestion]:
        results = []
        new_instruments = ctx.get("available_instruments", [])
        current_instruments = ctx.get("current_instruments", [])
        new_only = [i for i in new_instruments if i not in current_instruments]

        if new_only:
            results.append(ExperimentSuggestion(
                title=f"Train on {len(new_only)} additional instrument(s)",
                category=ExperimentCategory.DATA,
                description=f"Add instruments: {', '.join(new_only[:5])}",
                hypothesis="More diverse instruments improve generalization",
                expected_benefit="Improved regime coverage and model robustness",
                estimated_risk="Low — more data, same architecture",
                estimated_duration="4–8 hours",
                priority=3,
                confidence=0.70,
                trigger_context=ctx,
            ))
        return results

    def _check_regime_specialization(self, ctx: Dict[str, Any]) -> List[ExperimentSuggestion]:
        results = []
        degrading_regimes = ctx.get("degrading_regimes", [])
        for regime in degrading_regimes[:3]:
            results.append(ExperimentSuggestion(
                title=f"Regime-specific fine-tuning: {regime}",
                category=ExperimentCategory.REGIME_SPECIALIZATION,
                description=(
                    f"Fine-tune on {regime!r} regime data to recover performance"
                ),
                hypothesis=f"Model underperforms on {regime!r} due to regime drift",
                expected_benefit=f"Improved accuracy on {regime!r} regime",
                estimated_risk="Medium — may overfit to specific regime",
                estimated_duration="2–6 hours",
                priority=2,
                confidence=0.65,
                trigger_context=ctx,
            ))
        return results

    def update_status(self, suggestion_id: str, status: str) -> bool:
        for s in self._suggestions:
            if s.suggestion_id == suggestion_id:
                s.status = status
                return True
        return False

    def pending(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._suggestions
                if s.status == "pending"]

    def all_suggestions(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in reversed(self._suggestions[:limit])]

    def stats(self) -> Dict[str, Any]:
        by_cat: Dict[str, int] = {}
        for s in self._suggestions:
            by_cat[s.category] = by_cat.get(s.category, 0) + 1
        return {
            "total_suggestions": len(self._suggestions),
            "pending": sum(1 for s in self._suggestions if s.status == "pending"),
            "completed": sum(1 for s in self._suggestions if s.status == "completed"),
            "by_category": by_cat,
        }
