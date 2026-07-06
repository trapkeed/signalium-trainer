"""Phase 19.3 — HRSA Level 3: Hierarchical Research Scorer.

Implements the three-level scoring architecture:

  Level 1  Raw metric values (RawMetricsBundle)
      ↓
  Level 2  Category scores (HRSACategoryScorer)
      ↓
  Level 3  Institutional Research Score (this module)

The final score is computed from category scores ONLY — never directly
from raw metrics.  This enforces the hierarchical flow contract.

The HierarchicalScore result includes:
  - level1_metrics     (raw values, audit trail)
  - level2_categories  (per-category scores, confidence, grade)
  - level3_composite   (0–100)
  - confidence         (weighted mean of category confidences)
  - grade_assessment   (institutional grade, research grade, promotion, risk, potential)

Usage::

    scorer = HierarchicalResearchScorer()

    bundle = RawMetricsBundle("exp_2026_001")
    bundle.set("sharpe_ratio", 1.82)
    bundle.set("max_drawdown", 0.067)
    bundle.set("walk_forward_accuracy", 0.692)
    # ... set all available metrics ...

    result = scorer.score(bundle)
    print(result.summary())
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.research.hrsa_categories import CategoryResult, HRSACategoryScorer
from trainer.research.hrsa_grades import GradeAssessment, HRSAGradeEngine
from trainer.research.raw_metrics import RawMetricsBundle, RawMetricsRegistry

# ---------------------------------------------------------------------------
# HierarchicalScore — complete scoring result
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalScore:
    """Full 3-level scoring result for one experiment."""

    score_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    name: str = ""
    scored_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Level 1
    level1_metric_count: int = 0
    level1_categories_covered: List[str] = field(default_factory=list)

    # Level 2
    level2_categories: Dict[str, CategoryResult] = field(default_factory=dict)

    # Level 3
    level3_composite: float = 0.0  # 0–100
    confidence: float = 0.0  # 0–1 weighted mean of category confidences
    total_weight_used: float = 0.0

    # Grades
    grade_assessment: Optional[GradeAssessment] = None

    # Metadata
    instrument: str = ""
    timeframe: str = ""
    backbone: str = ""
    model_family: str = ""

    @property
    def composite(self) -> float:
        return self.level3_composite

    @property
    def institutional_grade(self) -> str:
        return self.grade_assessment.institutional_grade if self.grade_assessment else ""

    @property
    def research_grade(self) -> str:
        return self.grade_assessment.research_grade if self.grade_assessment else ""

    @property
    def promotion_recommendation(self) -> str:
        return self.grade_assessment.promotion_recommendation if self.grade_assessment else ""

    @property
    def risk_rating(self) -> str:
        return self.grade_assessment.risk_rating if self.grade_assessment else ""

    @property
    def future_potential(self) -> str:
        return self.grade_assessment.future_potential if self.grade_assessment else ""

    def category_score(self, cat: str) -> float:
        r = self.level2_categories.get(cat)
        return r.score if r else 0.0

    def category_confidence(self, cat: str) -> float:
        r = self.level2_categories.get(cat)
        return r.confidence if r else 0.0

    def strongest_categories(self, n: int = 3) -> List[str]:
        scored = [(cat, r.score) for cat, r in self.level2_categories.items() if r.metric_count > 0]
        return [cat for cat, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:n]]

    def weakest_categories(self, n: int = 3) -> List[str]:
        scored = [(cat, r.score) for cat, r in self.level2_categories.items() if r.metric_count > 0]
        return [cat for cat, _ in sorted(scored, key=lambda x: x[1])[:n]]

    def category_scores_dict(self) -> Dict[str, float]:
        return {cat: r.score for cat, r in self.level2_categories.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_id": self.score_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "scored_at": self.scored_at,
            # Level 1
            "level1_metric_count": self.level1_metric_count,
            "level1_categories_covered": self.level1_categories_covered,
            # Level 2
            "level2_categories": {cat: r.to_dict() for cat, r in self.level2_categories.items()},
            # Level 3
            "level3_composite": round(self.level3_composite, 2),
            "confidence": round(self.confidence, 4),
            "total_weight_used": round(self.total_weight_used, 4),
            # Grades
            "grade_assessment": self.grade_assessment.to_dict() if self.grade_assessment else {},
            "institutional_grade": self.institutional_grade,
            "research_grade": self.research_grade,
            "promotion_recommendation": self.promotion_recommendation,
            "risk_rating": self.risk_rating,
            "future_potential": self.future_potential,
            # Metadata
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "backbone": self.backbone,
            "model_family": self.model_family,
        }

    def summary(self) -> str:
        cat_lines = []
        for cat, r in sorted(
            self.level2_categories.items(), key=lambda x: x[1].score, reverse=True
        ):
            if r.metric_count > 0:
                bar = "█" * int(r.score / 10)
                cat_lines.append(
                    f"  {r.display_name:<28} {r.score:5.1f}/100  "
                    f"[{r.grade:<10}]  conf={r.confidence:.0%}  {bar}"
                )

        grade = self.grade_assessment
        return "\n".join(
            [
                f"Hierarchical Research Score: {self.level3_composite:.1f}/100",
                f"  Experiment:  {self.name or self.experiment_id}",
                f"  Confidence:  {self.confidence:.0%}",
                f"  Grade:       {self.institutional_grade}",
                f"  Research:    {self.research_grade}",
                f"  Promotion:   {self.promotion_recommendation}",
                f"  Risk:        {self.risk_rating}",
                f"  Potential:   {self.future_potential}",
                "",
                "Level 2 — Category Breakdown:",
                *cat_lines,
            ]
        )


# ---------------------------------------------------------------------------
# HierarchicalResearchScorer
# ---------------------------------------------------------------------------


class HierarchicalResearchScorer:
    """Three-level hierarchical research scorer.

    Level 1 → Level 2 → Level 3 — no shortcutting the hierarchy.

    Thread-safe: all state is in the input bundle; the scorer is stateless.
    """

    def __init__(
        self,
        category_scorer: Optional[HRSACategoryScorer] = None,
        grade_engine: Optional[HRSAGradeEngine] = None,
        registry: Optional[RawMetricsRegistry] = None,
    ) -> None:
        self._cat_scorer = category_scorer or HRSACategoryScorer.default()
        self._grade_engine = grade_engine or HRSAGradeEngine.default()
        self._registry = registry or RawMetricsRegistry.default()

    def score(
        self,
        bundle: RawMetricsBundle,
        *,
        name: str = "",
        instrument: str = "",
        timeframe: str = "",
        backbone: str = "",
        model_family: str = "",
        trend_slope: float = 0.0,
    ) -> HierarchicalScore:
        """Compute a full HierarchicalScore from a RawMetricsBundle."""

        # === Level 2: category scores ===
        cat_results = self._cat_scorer.score_all_categories(bundle)

        # === Level 3: weighted composite ===
        composite, total_w, conf = self._aggregate(cat_results)

        # === Grades ===
        cat_scores = {cat: r.score for cat, r in cat_results.items()}
        grade = self._grade_engine.full_assessment(
            composite=composite,
            confidence=conf,
            category_scores=cat_scores,
            trend_slope=trend_slope,
        )

        result = HierarchicalScore(
            experiment_id=bundle.experiment_id,
            name=name,
            level1_metric_count=bundle.total_metrics(),
            level1_categories_covered=bundle.all_categories_covered(),
            level2_categories=cat_results,
            level3_composite=composite,
            confidence=conf,
            total_weight_used=total_w,
            grade_assessment=grade,
            instrument=instrument,
            timeframe=timeframe,
            backbone=backbone,
            model_family=model_family,
        )
        logger.debug(
            f"[HierarchicalResearchScorer] {name or bundle.experiment_id}: "
            f"{composite:.1f}/100 [{grade.institutional_grade}] conf={conf:.0%}"
        )
        return result

    def _aggregate(self, cat_results: Dict[str, CategoryResult]) -> tuple:
        """Aggregate Level 2 category scores into Level 3 composite."""
        weighted_sum = 0.0
        weighted_conf = 0.0
        total_w = 0.0

        for cat, result in cat_results.items():
            if result.metric_count == 0:
                continue  # skip categories with no data
            w = result.weight
            # Coverage-adjust the weight: low coverage → lower effective weight
            effective_w = w * result.coverage
            weighted_sum += result.score * effective_w
            weighted_conf += result.confidence * effective_w
            total_w += effective_w

        if total_w <= 0:
            return 0.0, 0.0, 0.0

        composite = max(0.0, min(100.0, weighted_sum / total_w))
        confidence = max(0.0, min(1.0, weighted_conf / total_w))
        return composite, total_w, confidence

    def score_breakdown(self, bundle: RawMetricsBundle, **kwargs) -> Dict[str, Any]:
        """Return a full breakdown dict from a bundle."""
        result = self.score(bundle, **kwargs)
        return {
            "hierarchical_score": result.to_dict(),
            "scoring_pipeline": {
                "level1_metrics": bundle.total_metrics(),
                "level2_categories": len(
                    [r for r in result.level2_categories.values() if r.metric_count > 0]
                ),
                "level3_composite": result.level3_composite,
            },
            "architecture": "3-level hierarchical (L1→L2→L3)",
        }

    def sensitivity_analysis(
        self,
        bundle: RawMetricsBundle,
        category: str,
        delta_pct: float = 10.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """Estimate score change if a category's score increased by delta_pct."""
        base = self.score(bundle, **kwargs)
        base_composite = base.level3_composite

        cat_def = self._cat_scorer.category_definitions().get(category)
        if cat_def is None:
            return {"error": f"Unknown category: {category}"}

        # Simulate higher category score
        simulated_cat_score = min(100.0, base.category_score(category) + delta_pct)
        effective_w = cat_def.weight * base.level2_categories[category].coverage
        total_w = base.total_weight_used

        if total_w > 0:
            score_delta = (
                (simulated_cat_score - base.category_score(category)) * effective_w / total_w
            )
        else:
            score_delta = 0.0

        return {
            "category": category,
            "delta_pct": delta_pct,
            "base_score": round(base_composite, 2),
            "simulated_score": round(base_composite + score_delta, 2),
            "estimated_delta": round(score_delta, 3),
        }
