"""Phase 19 — Feature Lifecycle Management.

Continuously evaluates every feature across its full lifecycle:

  - Feature importance tracking
  - Stability measurement
  - Drift detection
  - Missing value rate
  - Correlation change
  - Predictive contribution
  - Age and usage frequency

Feature classification:
  Experimental → Candidate → Approved → Deprecated → Retired

Generates promotion and retirement recommendations.
No automated feature removal — recommendations only.

Usage::

    flm = FeatureLifecycleManager()
    flm.register_feature(FeatureRecord(name="rsi_14", category="technical"))
    flm.record_observation("rsi_14", importance=0.08, stability=0.92,
                            drift=0.02, missing_rate=0.001)
    rec = flm.evaluate("rsi_14")
    print(rec.classification, rec.recommendation)
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional


# ---------------------------------------------------------------------------
# Feature classification
# ---------------------------------------------------------------------------


class FeatureClassification:
    EXPERIMENTAL = "experimental"
    CANDIDATE    = "candidate"
    APPROVED     = "approved"
    DEPRECATED   = "deprecated"
    RETIRED      = "retired"

    ALL = [EXPERIMENTAL, CANDIDATE, APPROVED, DEPRECATED, RETIRED]

    ACTIVE = [EXPERIMENTAL, CANDIDATE, APPROVED]


# ---------------------------------------------------------------------------
# Feature record
# ---------------------------------------------------------------------------


@dataclass
class FeatureRecord:
    """Registry entry for one feature.

    Attributes:
        feature_id: Unique UUID.
        name: Feature name.
        category: Feature category (technical, fundamental, macro, etc.).
        classification: Current FeatureClassification value.
        created_at: When the feature was registered.
        promoted_at: When the feature was last promoted.
        deprecated_at: When the feature was deprecated (if any).
        retired_at: When the feature was retired (if any).
        usage_count: Total times this feature appeared in a training run.
        parent_feature: Parent feature name (if derived).
        description: Free-form description.
        tags: Arbitrary tag list.
    """

    feature_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    category: str = "technical"
    classification: str = FeatureClassification.EXPERIMENTAL
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    promoted_at: Optional[str] = None
    deprecated_at: Optional[str] = None
    retired_at: Optional[str] = None
    usage_count: int = 0
    parent_feature: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.classification in FeatureClassification.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "category": self.category,
            "classification": self.classification,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "deprecated_at": self.deprecated_at,
            "retired_at": self.retired_at,
            "usage_count": self.usage_count,
            "parent_feature": self.parent_feature,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Feature observation
# ---------------------------------------------------------------------------


@dataclass
class FeatureObservation:
    """One measured observation for a feature."""
    obs_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    importance: float = 0.0       # SHAP or gain-based importance (0-1)
    stability: float = 1.0        # Feature value stability over time (0-1)
    drift: float = 0.0            # Distribution drift magnitude (0-1)
    missing_rate: float = 0.0     # Fraction of NaN/missing values
    correlation_change: float = 0.0  # Change in correlation to target
    predictive_contribution: float = 0.0  # Prediction accuracy contribution
    dataset_version: str = ""
    model_version: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_id": self.obs_id,
            "name": self.name,
            "importance": self.importance,
            "stability": self.stability,
            "drift": self.drift,
            "missing_rate": self.missing_rate,
            "correlation_change": self.correlation_change,
            "predictive_contribution": self.predictive_contribution,
            "dataset_version": self.dataset_version,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Feature evaluation result
# ---------------------------------------------------------------------------


@dataclass
class FeatureEvaluation:
    """Evaluation result for one feature."""
    name: str = ""
    classification: str = FeatureClassification.EXPERIMENTAL
    avg_importance: float = 0.0
    avg_stability: float = 1.0
    avg_drift: float = 0.0
    avg_missing_rate: float = 0.0
    avg_contribution: float = 0.0
    n_observations: int = 0
    recommendation: str = "monitor"
    recommendation_reason: str = ""
    health_score: float = 1.0   # composite 0-1
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "classification": self.classification,
            "avg_importance": self.avg_importance,
            "avg_stability": self.avg_stability,
            "avg_drift": self.avg_drift,
            "avg_missing_rate": self.avg_missing_rate,
            "avg_contribution": self.avg_contribution,
            "n_observations": self.n_observations,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
            "health_score": self.health_score,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Feature lifecycle manager
# ---------------------------------------------------------------------------


class FeatureLifecycleManager:
    """Manages the full lifecycle of every feature in the platform.

    Args:
        obs_window: Rolling observation window per feature.
        importance_promote_threshold: Min importance to promote to CANDIDATE.
        stability_retire_threshold: Max stability below which to deprecate.
        drift_deprecate_threshold: Drift above which to recommend deprecation.
        missing_rate_retire_threshold: Missing rate above which to retire.
    """

    def __init__(self, obs_window: int = 100,
                 importance_promote_threshold: float = 0.03,
                 stability_retire_threshold: float = 0.50,
                 drift_deprecate_threshold: float = 0.30,
                 missing_rate_retire_threshold: float = 0.10) -> None:
        self.obs_window = obs_window
        self.importance_threshold = importance_promote_threshold
        self.stability_threshold = stability_retire_threshold
        self.drift_threshold = drift_deprecate_threshold
        self.missing_threshold = missing_rate_retire_threshold
        self._features: Dict[str, FeatureRecord] = {}
        self._observations: Dict[str, Deque[FeatureObservation]] = {}
        self._all_observations: Dict[str, List[FeatureObservation]] = {}

    def register_feature(self, record: FeatureRecord) -> str:
        self._features[record.name] = record
        return record.name

    def get_feature(self, name: str) -> Optional[FeatureRecord]:
        return self._features.get(name)

    def record_observation(self, name: str,
                            importance: float = 0.0,
                            stability: float = 1.0,
                            drift: float = 0.0,
                            missing_rate: float = 0.0,
                            correlation_change: float = 0.0,
                            predictive_contribution: float = 0.0,
                            dataset_version: str = "",
                            model_version: str = "") -> FeatureObservation:
        """Record one feature quality observation."""
        if name not in self._features:
            self._features[name] = FeatureRecord(name=name)
        self._features[name].usage_count += 1
        obs = FeatureObservation(
            name=name, importance=importance, stability=stability,
            drift=drift, missing_rate=missing_rate,
            correlation_change=correlation_change,
            predictive_contribution=predictive_contribution,
            dataset_version=dataset_version, model_version=model_version,
        )
        if name not in self._observations:
            self._observations[name] = deque(maxlen=self.obs_window)
            self._all_observations[name] = []
        self._observations[name].append(obs)
        self._all_observations[name].append(obs)
        return obs

    def evaluate(self, name: str) -> FeatureEvaluation:
        """Evaluate a feature and generate a lifecycle recommendation."""
        record = self._features.get(name)
        if record is None:
            return FeatureEvaluation(name=name,
                                      recommendation_reason="Feature not registered")
        obs_list = list(self._observations.get(name, []))
        n = len(obs_list)
        if n == 0:
            return FeatureEvaluation(name=name,
                                      classification=record.classification,
                                      recommendation_reason="No observations yet")

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        avg_imp = _mean([o.importance for o in obs_list])
        avg_stab = _mean([o.stability for o in obs_list])
        avg_drift = _mean([o.drift for o in obs_list])
        avg_miss = _mean([o.missing_rate for o in obs_list])
        avg_contrib = _mean([o.predictive_contribution for o in obs_list])

        health = (
            avg_imp * 0.3 +
            avg_stab * 0.3 +
            (1.0 - avg_drift) * 0.2 +
            (1.0 - avg_miss) * 0.1 +
            avg_contrib * 0.1
        )

        recommendation = "monitor"
        reason = ""
        cls = record.classification

        if cls == FeatureClassification.EXPERIMENTAL:
            if avg_imp >= self.importance_threshold and avg_stab >= 0.7:
                recommendation = "promote_to_candidate"
                reason = (f"Importance {avg_imp:.3f} ≥ {self.importance_threshold}, "
                           f"stability {avg_stab:.3f} ≥ 0.70")
            elif avg_miss > self.missing_threshold:
                recommendation = "retire"
                reason = f"Missing rate {avg_miss:.3f} > {self.missing_threshold}"
            else:
                recommendation = "monitor"
                reason = "Insufficient evidence for promotion"

        elif cls == FeatureClassification.CANDIDATE:
            if avg_imp >= self.importance_threshold * 1.5 and avg_stab >= 0.80:
                recommendation = "promote_to_approved"
                reason = (f"Importance {avg_imp:.3f} ≥ {self.importance_threshold*1.5:.3f}, "
                           f"stability {avg_stab:.3f} ≥ 0.80")
            elif avg_drift > self.drift_threshold:
                recommendation = "deprecate"
                reason = f"Feature drift {avg_drift:.3f} > {self.drift_threshold}"
            else:
                recommendation = "monitor"
                reason = "Accumulating candidate evidence"

        elif cls == FeatureClassification.APPROVED:
            if avg_stab < self.stability_threshold:
                recommendation = "deprecate"
                reason = f"Stability dropped to {avg_stab:.3f} < {self.stability_threshold}"
            elif avg_drift > self.drift_threshold:
                recommendation = "deprecate"
                reason = f"Drift {avg_drift:.3f} > threshold {self.drift_threshold}"
            elif avg_miss > self.missing_threshold:
                recommendation = "deprecate"
                reason = f"Missing rate {avg_miss:.3f} > {self.missing_threshold}"
            else:
                recommendation = "maintain"
                reason = "Feature meets all quality thresholds"

        elif cls == FeatureClassification.DEPRECATED:
            recommendation = "retire"
            reason = "Feature is deprecated — schedule retirement"

        else:  # RETIRED
            recommendation = "archived"
            reason = "Feature retired — no action needed"

        return FeatureEvaluation(
            name=name,
            classification=cls,
            avg_importance=round(avg_imp, 4),
            avg_stability=round(avg_stab, 4),
            avg_drift=round(avg_drift, 4),
            avg_missing_rate=round(avg_miss, 4),
            avg_contribution=round(avg_contrib, 4),
            n_observations=n,
            recommendation=recommendation,
            recommendation_reason=reason,
            health_score=round(min(max(health, 0.0), 1.0), 4),
        )

    def classify(self, name: str, classification: str,
                  actor: str = "system",
                  reason: str = "") -> bool:
        """Manually update a feature's classification."""
        record = self._features.get(name)
        if record is None:
            return False
        now = datetime.now(timezone.utc).isoformat()
        record.classification = classification
        if classification in (FeatureClassification.CANDIDATE,
                               FeatureClassification.APPROVED):
            record.promoted_at = now
        elif classification == FeatureClassification.DEPRECATED:
            record.deprecated_at = now
        elif classification == FeatureClassification.RETIRED:
            record.retired_at = now
        return True

    def evaluate_all(self) -> List[FeatureEvaluation]:
        """Evaluate all registered features."""
        return [self.evaluate(name) for name in self._features]

    def features_by_classification(self, cls: str) -> List[FeatureRecord]:
        return [f for f in self._features.values()
                if f.classification == cls]

    def promotion_candidates(self) -> List[FeatureEvaluation]:
        evals = self.evaluate_all()
        return [e for e in evals
                if e.recommendation.startswith("promote")]

    def retirement_candidates(self) -> List[FeatureEvaluation]:
        evals = self.evaluate_all()
        return [e for e in evals
                if e.recommendation in ("retire", "deprecate", "archived")]

    def summary(self) -> Dict[str, Any]:
        by_cls: Dict[str, int] = {}
        for f in self._features.values():
            by_cls[f.classification] = by_cls.get(f.classification, 0) + 1
        return {
            "total_features": len(self._features),
            "by_classification": by_cls,
            "total_observations": sum(
                len(v) for v in self._all_observations.values()
            ),
        }
