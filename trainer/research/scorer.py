"""Phase 19.2 — IRSE Core Scorer.

The ResearchScorer produces a composite Research Score (0–100) from
configurable weighted category inputs.

Classes:

  ResearchScoreInputs  — all 8 category inputs in one container
  CategoryScore        — score + sub-scores for one category
  ResearchScore        — complete composite result
  ResearchScorer       — main entry point

Usage::

    from trainer.research.scorer import ResearchScorer, ResearchScoreInputs
    from trainer.research.categories import (
        ValidationQualityInputs, TradingPerformanceInputs, ...
    )

    scorer = ResearchScorer()          # uses default weights
    inputs = ResearchScoreInputs(
        validation=ValidationQualityInputs(
            walk_forward_accuracy=0.692, n_folds=12,
        ),
        trading=TradingPerformanceInputs(
            sharpe_ratio=1.82, max_drawdown=0.067, win_rate=0.573,
        ),
    )
    result = scorer.score(inputs)
    print(f"Research Score: {result.composite:.1f}/100  Grade: {result.grade_label}")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from trainer.research.categories import (
    CalibrationInputs,
    CalibrationScorer,
    EfficiencyInputs,
    EfficiencyScorer,
    ExplainabilityInputs,
    ExplainabilityScorer,
    GeneralizationInputs,
    GeneralizationScorer,
    OperationalReadinessInputs,
    OperationalReadinessScorer,
    RobustnessInputs,
    RobustnessScorer,
    TradingPerformanceInputs,
    TradingPerformanceScorer,
    ValidationQualityInputs,
    ValidationQualityScorer,
    _clamp,
)
from trainer.research.confidence import ConfidenceCalculator, ConfidenceInputs, ConfidenceResult
from trainer.research.grades import GradeResult, GradeSystem
from trainer.research.weights import ALL_CATEGORIES, ScoringWeights

# ---------------------------------------------------------------------------
# ResearchScoreInputs
# ---------------------------------------------------------------------------


@dataclass
class ResearchScoreInputs:
    """All category inputs for one scored experiment."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    name: str = ""
    scored_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Category inputs
    validation: ValidationQualityInputs = field(default_factory=ValidationQualityInputs)
    trading: TradingPerformanceInputs = field(default_factory=TradingPerformanceInputs)
    robustness: RobustnessInputs = field(default_factory=RobustnessInputs)
    calibration: CalibrationInputs = field(default_factory=CalibrationInputs)
    generalization: GeneralizationInputs = field(default_factory=GeneralizationInputs)
    efficiency: EfficiencyInputs = field(default_factory=EfficiencyInputs)
    explainability: ExplainabilityInputs = field(default_factory=ExplainabilityInputs)
    operational: OperationalReadinessInputs = field(default_factory=OperationalReadinessInputs)

    # Confidence inputs (independent of category scores)
    confidence: ConfidenceInputs = field(default_factory=ConfidenceInputs)

    # Metadata for filtering / leaderboard
    instrument: str = ""
    timeframe: str = ""
    backbone: str = ""
    architecture: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    model_family: str = ""
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CategoryScore
# ---------------------------------------------------------------------------


@dataclass
class CategoryScore:
    """Score for one weighted category."""

    category: str
    score: float  # 0–100
    weight: float  # fraction, e.g. 0.25
    weighted_score: float  # score * weight (0–100)
    sub_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "score": round(self.score, 2),
            "weight": self.weight,
            "weighted_contribution": round(self.weighted_score, 2),
            "sub_scores": {k: round(v, 2) for k, v in self.sub_scores.items()},
        }


# ---------------------------------------------------------------------------
# ResearchScore
# ---------------------------------------------------------------------------


@dataclass
class ResearchScore:
    """Complete Research Score result for one experiment."""

    score_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    model_id: str = ""
    name: str = ""
    scored_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Composite score
    composite: float = 0.0  # 0–100

    # Grade
    grade_label: str = ""
    grade_description: str = ""
    promotion_eligible: bool = False
    deployment_eligible: bool = False

    # Category breakdown
    category_scores: list[CategoryScore] = field(default_factory=list)

    # Confidence in the score
    confidence: ConfidenceResult | None = None

    # Weights used
    weights_used: dict[str, float] = field(default_factory=dict)

    # Metadata
    instrument: str = ""
    timeframe: str = ""
    backbone: str = ""
    architecture: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    model_family: str = ""
    tags: list[str] = field(default_factory=list)

    def category_score(self, category: str) -> CategoryScore | None:
        for cs in self.category_scores:
            if cs.category == category:
                return cs
        return None

    def category_score_value(self, category: str) -> float:
        cs = self.category_score(category)
        return cs.score if cs else 0.0

    def confidence_score(self) -> float:
        return self.confidence.score if self.confidence else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_id": self.score_id,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "name": self.name,
            "scored_at": self.scored_at,
            "composite": round(self.composite, 2),
            "grade_label": self.grade_label,
            "grade_description": self.grade_description,
            "promotion_eligible": self.promotion_eligible,
            "deployment_eligible": self.deployment_eligible,
            "confidence_score": round(self.confidence_score(), 4),
            "confidence_level": self.confidence.level if self.confidence else "Unknown",
            "category_scores": [cs.to_dict() for cs in self.category_scores],
            "weights_used": self.weights_used,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "backbone": self.backbone,
            "architecture": self.architecture,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_family": self.model_family,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchScore:
        cat_scores = [
            CategoryScore(
                category=cs["category"],
                score=cs["score"],
                weight=cs["weight"],
                weighted_score=cs["weighted_contribution"],
                sub_scores=cs.get("sub_scores", {}),
            )
            for cs in d.get("category_scores", [])
        ]
        return cls(
            score_id=d.get("score_id", str(uuid.uuid4())),
            experiment_id=d.get("experiment_id", ""),
            model_id=d.get("model_id", ""),
            name=d.get("name", ""),
            scored_at=d.get("scored_at", datetime.now(UTC).isoformat()),
            composite=d.get("composite", 0.0),
            grade_label=d.get("grade_label", ""),
            grade_description=d.get("grade_description", ""),
            promotion_eligible=d.get("promotion_eligible", False),
            deployment_eligible=d.get("deployment_eligible", False),
            category_scores=cat_scores,
            weights_used=d.get("weights_used", {}),
            instrument=d.get("instrument", ""),
            timeframe=d.get("timeframe", ""),
            backbone=d.get("backbone", ""),
            architecture=d.get("architecture", ""),
            dataset_version=d.get("dataset_version", ""),
            feature_version=d.get("feature_version", ""),
            model_family=d.get("model_family", ""),
            tags=d.get("tags", []),
        )

    def summary(self) -> str:
        lines = [
            f"Research Score: {self.composite:.1f}/100  [{self.grade_label}]",
            f"Experiment:     {self.name or self.experiment_id}",
            f"Promotion:      {'YES' if self.promotion_eligible else 'NO'}",
            f"Confidence:     {self.confidence_score():.0%}" if self.confidence else "",
            "",
            "Category Breakdown:",
        ]
        for cs in self.category_scores:
            bar = "█" * int(cs.score / 10)
            lines.append(
                f"  {cs.category:<24} {cs.score:5.1f}/100  " f"({cs.weight*100:.0f}%)  {bar}"
            )
        return "\n".join(line for line in lines if line is not None)


# ---------------------------------------------------------------------------
# ResearchScorer
# ---------------------------------------------------------------------------


class ResearchScorer:
    """Compute composite Research Scores from ResearchScoreInputs.

    Thread-safe — all state is in the inputs; scorer instances are reusable.
    """

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        grade_system: GradeSystem | None = None,
    ) -> None:
        self._weights = weights or ScoringWeights.default()
        self._weights.validate()
        self._grades = grade_system or GradeSystem.default()
        self._confidence_calc = ConfidenceCalculator()

        self._category_scorers = {
            "validation_quality": ValidationQualityScorer(),
            "trading_performance": TradingPerformanceScorer(),
            "robustness": RobustnessScorer(),
            "calibration": CalibrationScorer(),
            "generalization": GeneralizationScorer(),
            "efficiency": EfficiencyScorer(),
            "explainability": ExplainabilityScorer(),
            "operational_readiness": OperationalReadinessScorer(),
        }

        self._category_inputs_map = {
            "validation_quality": "validation",
            "trading_performance": "trading",
            "robustness": "robustness",
            "calibration": "calibration",
            "generalization": "generalization",
            "efficiency": "efficiency",
            "explainability": "explainability",
            "operational_readiness": "operational",
        }

    @property
    def weights(self) -> ScoringWeights:
        return self._weights

    def score(self, inputs: ResearchScoreInputs) -> ResearchScore:
        """Compute a full Research Score from the provided inputs."""
        category_scores: list[CategoryScore] = []
        composite = 0.0
        weights_dict = self._weights.as_dict()

        for cat in ALL_CATEGORIES:
            scorer = self._category_scorers[cat]
            cat_input_attr = self._category_inputs_map[cat]
            cat_input = getattr(inputs, cat_input_attr)
            cat_score_val, sub_scores = scorer.score(cat_input)
            weight = weights_dict[cat]
            weighted = cat_score_val * weight
            composite += weighted
            category_scores.append(
                CategoryScore(
                    category=cat,
                    score=cat_score_val,
                    weight=weight,
                    weighted_score=weighted,
                    sub_scores=sub_scores,
                )
            )

        composite = _clamp(composite)

        # Grade
        grade_result: GradeResult = self._grades.assign(composite)

        # Confidence
        conf_result = self._confidence_calc.compute(inputs.confidence)

        result = ResearchScore(
            experiment_id=inputs.experiment_id,
            model_id=inputs.model_id,
            name=inputs.name,
            scored_at=inputs.scored_at,
            composite=composite,
            grade_label=grade_result.grade.label,
            grade_description=grade_result.grade.description,
            promotion_eligible=grade_result.promotion_eligible,
            deployment_eligible=grade_result.deployment_eligible,
            category_scores=category_scores,
            confidence=conf_result,
            weights_used=weights_dict,
            instrument=inputs.instrument,
            timeframe=inputs.timeframe,
            backbone=inputs.backbone,
            architecture=inputs.architecture,
            dataset_version=inputs.dataset_version,
            feature_version=inputs.feature_version,
            model_family=inputs.model_family,
            tags=inputs.tags,
        )
        logger.debug(
            f"[ResearchScorer] scored {inputs.name or inputs.experiment_id}: "
            f"{composite:.1f}/100 [{grade_result.grade.label}]"
        )
        return result

    def score_breakdown(self, inputs: ResearchScoreInputs) -> dict[str, Any]:
        """Return detailed breakdown dict (JSON-serialisable)."""
        result = self.score(inputs)
        return {
            "research_score": result.to_dict(),
            "grade_system": self._grades.all_grades(),
            "weights": self._weights.as_dict(),
            "promotion_threshold": self._grades.promotion_threshold(),
            "deployment_threshold": self._grades.deployment_threshold(),
        }

    def update_weights(self, weights: ScoringWeights) -> None:
        """Replace weights on the fly (validates first)."""
        weights.validate()
        self._weights = weights

    def sensitivity_analysis(
        self, inputs: ResearchScoreInputs, delta: float = 0.05
    ) -> dict[str, float]:
        """Return the score impact of a ±delta change in each weight."""
        base = self.score(inputs).composite
        result: dict[str, float] = {}
        base_weights = self._weights.as_dict()
        for cat in ALL_CATEGORIES:
            perturbed = dict(base_weights)
            perturbed[cat] = min(1.0, base_weights[cat] + delta)
            # renormalise
            total = sum(perturbed.values())
            perturbed = {k: v / total for k, v in perturbed.items()}
            temp_scorer = ResearchScorer(ScoringWeights.from_dict(perturbed), self._grades)
            perturbed_score = temp_scorer.score(inputs).composite
            result[cat] = round(perturbed_score - base, 3)
        return result
