"""Phase 19.3 — HRSA Score Explanations.

Generates human-readable explanations for every hierarchical research score.

Every score must be explainable.  The explainer answers:
  - Why did this model receive this score?
  - What are the largest positive contributors?
  - What are the largest negative contributors?
  - What is the weakest category and why?
  - What is the strongest category?
  - What specific improvements would raise the score?

Usage::

    explainer = HRSAExplainer()
    explanation = explainer.explain(hierarchical_score)

    print(explanation.summary())
    for rec in explanation.recommendations:
        print(f"  • {rec}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Improvement recommendation
# ---------------------------------------------------------------------------


@dataclass
class ImprovementRecommendation:
    category: str
    metric: Optional[str]
    current_score: float
    target_score: float
    estimated_composite_impact: float
    action: str
    priority: str  # "critical" | "high" | "medium" | "low"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "metric": self.metric,
            "current_score": round(self.current_score, 1),
            "target_score": round(self.target_score, 1),
            "estimated_composite_impact": round(self.estimated_composite_impact, 3),
            "action": self.action,
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# ScoreExplanation
# ---------------------------------------------------------------------------


@dataclass
class ScoreExplanation:
    score_id: str
    experiment_id: str
    name: str
    composite: float
    confidence: float

    # Structural highlights
    strongest_categories: List[Tuple[str, float]]  # [(category, score)]
    weakest_categories: List[Tuple[str, float]]

    # Metric contributors
    top_positive_metrics: List[Tuple[str, float]]  # [(metric, normalised score 0–100)]
    top_negative_metrics: List[Tuple[str, float]]

    # Narrative
    overall_summary: str
    category_narratives: Dict[str, str]

    # Improvement
    recommendations: List[ImprovementRecommendation]
    improvement_potential: float  # estimated composite gain if top recommendations actioned

    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> str:
        rec_lines = [
            f"  {i+1}. [{r.priority.upper()}] {r.action}"
            for i, r in enumerate(self.recommendations[:5])
        ]
        strongest = ", ".join(
            f"{cat} ({score:.0f})" for cat, score in self.strongest_categories[:3]
        )
        weakest = ", ".join(f"{cat} ({score:.0f})" for cat, score in self.weakest_categories[:3])
        return "\n".join(
            [
                f"Score Explanation — {self.name or self.experiment_id}",
                f"  Composite: {self.composite:.1f}/100  conf={self.confidence:.0%}",
                "",
                f"  {self.overall_summary}",
                "",
                f"  Strongest: {strongest}",
                f"  Weakest:   {weakest}",
                "",
                f"  Improvement potential: +{self.improvement_potential:.1f} pts",
                "  Top recommendations:",
                *rec_lines,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_id": self.score_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "composite": self.composite,
            "confidence": self.confidence,
            "overall_summary": self.overall_summary,
            "strongest_categories": [
                {"category": c, "score": round(s, 1)} for c, s in self.strongest_categories
            ],
            "weakest_categories": [
                {"category": c, "score": round(s, 1)} for c, s in self.weakest_categories
            ],
            "top_positive_metrics": [
                {"metric": m, "score": round(s, 1)} for m, s in self.top_positive_metrics
            ],
            "top_negative_metrics": [
                {"metric": m, "score": round(s, 1)} for m, s in self.top_negative_metrics
            ],
            "category_narratives": self.category_narratives,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "improvement_potential": round(self.improvement_potential, 2),
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Improvement action templates
# ---------------------------------------------------------------------------

_IMPROVEMENT_ACTIONS: Dict[str, Dict[str, str]] = {
    "validation_quality": {
        "default": "Increase validation fold count to 12+; improve cross-regime stability.",
        "walk_forward_accuracy": "Improve walk-forward accuracy: review feature engineering and model architecture.",
        "fold_score_std": "Reduce fold score variance: increase training data or add regularization.",
        "n_validation_folds": "Increase the number of walk-forward folds to at least 12.",
        "consistency_score": "Increase fraction of folds above threshold: review underperforming folds.",
    },
    "trading_performance": {
        "default": "Improve Sharpe ratio and reduce maximum drawdown.",
        "sharpe_ratio": "Improve Sharpe ratio: review position sizing and exit strategy.",
        "max_drawdown": "Reduce maximum drawdown: add drawdown-based position scaling.",
        "profit_factor": "Improve profit factor: optimize entry signals for higher expectancy.",
        "win_rate": "Improve win rate: refine entry conditions.",
    },
    "robustness": {
        "default": "Improve Monte Carlo stability and stress test pass rate.",
        "monte_carlo_stability": "Improve Monte Carlo stability: test with shuffled price paths.",
        "stress_test_pass_rate": "Improve stress test pass rate: add regime-specific safeguards.",
        "noise_resistance": "Add noise robustness: test with augmented noisy inputs.",
    },
    "calibration": {
        "default": "Reduce expected calibration error below 0.02.",
        "expected_calibration_error": "Apply temperature scaling or isotonic regression to calibrate outputs.",
        "brier_score": "Reduce Brier score: review probability estimation layers.",
    },
    "generalization": {
        "default": "Improve OOS performance across unseen instruments and regimes.",
        "oos_performance": "Increase OOS dataset size and evaluate on additional instruments.",
        "cross_regime_stability": "Test on additional market regimes: trending, mean-reverting, volatile.",
    },
    "efficiency": {
        "default": "Reduce inference latency and model size.",
        "inference_latency_ms": "Optimize inference: apply model quantization or pruning.",
        "model_size_mb": "Reduce model size: apply weight sharing or distillation.",
    },
    "risk_management": {
        "default": "Reduce VaR and expected shortfall; improve risk-adjusted return.",
        "value_at_risk_95": "Reduce 95% VaR: add position sizing constraints.",
        "expected_shortfall_95": "Reduce expected shortfall: add tail risk controls.",
    },
    "operational_readiness": {
        "default": "Complete deployment readiness checklist and monitoring setup.",
        "deployment_readiness": "Complete deployment readiness checklist.",
        "drift_monitoring": "Implement feature and prediction drift monitoring.",
    },
    "data_quality": {
        "default": "Improve data completeness and label quality.",
        "data_completeness": "Fill missing data gaps; document imputation approach.",
        "label_quality": "Review label generation; run label quality audit.",
    },
    "research_reproducibility": {
        "default": "Lock dataset and code versions; record all random seeds.",
        "environment_reproducibility": "Capture full environment snapshot with pinned dependencies.",
        "dataset_version_locked": "Lock dataset version to a specific tagged release.",
    },
    "explainability": {
        "default": "Improve feature attribution coverage and documentation completeness.",
        "documentation_completeness": "Complete all required documentation sections.",
    },
    "model_stability": {
        "default": "Reduce prediction variance and improve checkpoint consistency.",
        "prediction_variance": "Reduce ensemble variance or apply dropout calibration.",
    },
    "feature_quality": {
        "default": "Improve feature stability and reduce redundancy.",
        "feature_stability_index": "Audit feature stability across time periods.",
    },
    "security": {
        "default": "Improve input validation and adversarial robustness.",
        "adversarial_robustness": "Run adversarial input tests; add input sanitization.",
    },
    "compliance": {
        "default": "Complete regulatory compliance and audit trail documentation.",
        "audit_trail_completeness": "Ensure all model decisions are logged and traceable.",
    },
}


def _get_action(category: str, metric: Optional[str] = None) -> str:
    cat_actions = _IMPROVEMENT_ACTIONS.get(category, {})
    if metric and metric in cat_actions:
        return cat_actions[metric]
    return cat_actions.get("default", f"Improve {category} metrics.")


def _priority(score: float) -> str:
    if score < 40:
        return "critical"
    if score < 55:
        return "high"
    if score < 70:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# HRSAExplainer
# ---------------------------------------------------------------------------


class HRSAExplainer:
    """Generate natural-language explanations for HierarchicalScore results."""

    def explain(self, hs: Any, reference: Optional[Any] = None) -> ScoreExplanation:
        """
        Explain a HierarchicalScore.

        Args:
            hs:        The HierarchicalScore to explain.
            reference: Optional reference HierarchicalScore (e.g. champion)
                       for comparative explanations.
        """
        cats = hs.level2_categories

        # Sort categories
        scored_cats = [(cat, r.score) for cat, r in cats.items() if r.metric_count > 0]
        sorted_desc = sorted(scored_cats, key=lambda x: x[1], reverse=True)
        sorted_asc = sorted(scored_cats, key=lambda x: x[1])
        strongest = sorted_desc[:3]
        weakest = sorted_asc[:3]

        # Per-metric scores (all categories)
        all_metric_scores: List[Tuple[str, float]] = []
        for r in cats.values():
            for metric, score in r.metric_scores.items():
                all_metric_scores.append((metric, score))
        all_metric_scores.sort(key=lambda x: x[1], reverse=True)

        top_positive = all_metric_scores[:5]
        top_negative = all_metric_scores[-5:][::-1]

        # Overall narrative
        grade = hs.institutional_grade
        conf_pct = int(hs.confidence * 100)
        composite = hs.level3_composite
        summary = self._build_summary(composite, grade, conf_pct, strongest, weakest)

        # Category narratives
        narratives = self._build_category_narratives(cats, reference)

        # Recommendations
        recs = self._build_recommendations(cats, hs.total_weight_used)
        improvement_potential = min(15.0, sum(r.estimated_composite_impact for r in recs[:5]))

        return ScoreExplanation(
            score_id=hs.score_id,
            experiment_id=hs.experiment_id,
            name=hs.name,
            composite=composite,
            confidence=hs.confidence,
            strongest_categories=strongest,
            weakest_categories=weakest,
            top_positive_metrics=top_positive,
            top_negative_metrics=top_negative,
            overall_summary=summary,
            category_narratives=narratives,
            recommendations=recs,
            improvement_potential=improvement_potential,
        )

    def _build_summary(
        self,
        composite: float,
        grade: str,
        conf_pct: int,
        strongest: List[Tuple[str, float]],
        weakest: List[Tuple[str, float]],
    ) -> str:
        level = (
            "strong institutional-grade"
            if composite >= 85
            else (
                "solid candidate-level"
                if composite >= 75
                else "experimental-level" if composite >= 60 else "early-research-phase"
            )
        )
        strong_str = strongest[0][0].replace("_", " ") if strongest else "N/A"
        weak_str = weakest[0][0].replace("_", " ") if weakest else "N/A"
        return (
            f"This model achieved a {level} Research Score of {composite:.1f}/100 "
            f"[{grade}] with {conf_pct}% confidence. "
            f"Its strongest dimension is {strong_str}, "
            f"while {weak_str} presents the greatest improvement opportunity."
        )

    def _build_category_narratives(
        self, cats: Dict[str, Any], reference: Optional[Any]
    ) -> Dict[str, str]:
        narratives: Dict[str, str] = {}
        for cat, result in cats.items():
            if result.metric_count == 0:
                narratives[cat] = "No metrics provided for this category."
                continue
            grade = result.grade
            score = result.score
            conf = result.confidence
            if reference:
                ref_cat = reference.level2_categories.get(cat)
                if ref_cat and ref_cat.metric_count > 0:
                    delta = score - ref_cat.score
                    direction = "above" if delta >= 0 else "below"
                    narratives[cat] = (
                        f"{grade} ({score:.1f}/100, {conf:.0%} conf). "
                        f"{abs(delta):.1f} pts {direction} reference."
                    )
                    continue
            narratives[cat] = f"{grade} ({score:.1f}/100, {conf:.0%} conf)."
        return narratives

    def _build_recommendations(
        self, cats: Dict[str, Any], total_weight: float
    ) -> List[ImprovementRecommendation]:
        recs: List[ImprovementRecommendation] = []

        for cat, result in cats.items():
            if result.metric_count == 0 or result.score >= 90:
                continue
            weight = result.weight
            effective_w = weight * result.coverage
            gap = 90.0 - result.score

            # Find the lowest individual metric
            if result.metric_scores:
                worst_metric = min(result.metric_scores, key=result.metric_scores.get)
                worst_score = result.metric_scores[worst_metric]
            else:
                worst_metric = None
                worst_score = result.score

            impact = (gap * effective_w / total_weight) if total_weight > 0 else 0.0

            recs.append(
                ImprovementRecommendation(
                    category=cat,
                    metric=worst_metric,
                    current_score=result.score,
                    target_score=min(100.0, result.score + gap * 0.5),
                    estimated_composite_impact=round(impact * 0.5, 3),
                    action=_get_action(cat, worst_metric),
                    priority=_priority(result.score),
                )
            )

        # Sort: critical > high > medium > low; then by impact desc
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recs.sort(key=lambda r: (priority_order[r.priority], -r.estimated_composite_impact))
        return recs

    def compare_explanations(self, subject: Any, reference: Any) -> Dict[str, Any]:
        """Return a diff-style explanation comparing subject vs reference."""
        subj_cats = subject.level2_categories
        ref_cats = reference.level2_categories
        delta = subject.level3_composite - reference.level3_composite
        cat_deltas: Dict[str, float] = {}
        for cat in set(subj_cats) | set(ref_cats):
            s = subj_cats.get(cat)
            r = ref_cats.get(cat)
            if s and r and s.metric_count > 0 and r.metric_count > 0:
                cat_deltas[cat] = round(s.score - r.score, 2)

        improvements = {k: v for k, v in cat_deltas.items() if v >= 1.0}
        regressions = {k: v for k, v in cat_deltas.items() if v <= -1.0}

        return {
            "composite_delta": round(delta, 2),
            "improved": delta > 0,
            "category_improvements": dict(
                sorted(improvements.items(), key=lambda x: x[1], reverse=True)
            ),
            "category_regressions": dict(sorted(regressions.items(), key=lambda x: x[1])),
            "verdict": (
                f"Subject beats reference by {delta:+.1f} pts. "
                f"Improved in: {', '.join(list(improvements)[:3]) or 'none'}. "
                f"Regressed in: {', '.join(list(regressions)[:3]) or 'none'}."
            ),
        }
