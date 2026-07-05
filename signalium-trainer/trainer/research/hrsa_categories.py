"""Phase 19.3 — HRSA Level 2: Category Scoring.

Groups raw metrics into configurable categories and produces a Category
Score in [0, 100] for each, along with confidence, grade, trend, and
benchmark comparison.

15 default categories (vs 8 in Phase 19.2):
  validation_quality, trading_performance, calibration, generalization,
  robustness, efficiency, explainability, operational_readiness,
  risk_management, data_quality, feature_quality, model_stability,
  research_reproducibility, security, compliance

Each category has:
  - Metric mappings (name → weight within category)
  - Normalisation method (per-category or global default)
  - Confidence scoring (coverage, sample count, significance)
  - Grade assignment
  - Optional benchmark comparison

Usage::

    scorer = HRSACategoryScorer.default()
    bundle = RawMetricsBundle("exp_001")
    bundle.set("sharpe_ratio", 1.82)
    bundle.set("max_drawdown", 0.067)

    results = scorer.score_all_categories(bundle)
    for cat, result in results.items():
        print(f"{cat}: {result.score:.1f}/100  [{result.grade}]")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from trainer.research.normalization import (
    CategoryNormaliser,
    NormMethod,
    Normaliser,
)
from trainer.research.raw_metrics import RawMetricsBundle, RawMetricsRegistry


# ---------------------------------------------------------------------------
# MetricMapping — one metric within a category
# ---------------------------------------------------------------------------


@dataclass
class MetricMapping:
    name: str
    weight: float = 1.0
    normalisation_method: Optional[str] = None   # overrides category default


# ---------------------------------------------------------------------------
# CategoryDefinition
# ---------------------------------------------------------------------------


@dataclass
class CategoryDefinition:
    name: str
    display_name: str
    weight: float               # within Level 3 composite (sums to 1.0 across all)
    metric_mappings: List[MetricMapping] = field(default_factory=list)
    normalisation_method: str = NormMethod.MIN_MAX
    description: str = ""

    def metric_weights(self) -> Dict[str, float]:
        return {m.name: m.weight for m in self.metric_mappings}

    def metric_names(self) -> List[str]:
        return [m.name for m in self.metric_mappings]


# ---------------------------------------------------------------------------
# Default category definitions (15 categories)
# ---------------------------------------------------------------------------

def _default_categories() -> List[CategoryDefinition]:
    return [
        CategoryDefinition(
            name="validation_quality",
            display_name="Validation Quality",
            weight=0.15,
            metric_mappings=[
                MetricMapping("walk_forward_accuracy", 3.0),
                MetricMapping("f1_macro",              2.0),
                MetricMapping("roc_auc",               2.0),
                MetricMapping("pr_auc",                1.5),
                MetricMapping("n_validation_folds",    1.0),
                MetricMapping("fold_score_std",        1.5),
                MetricMapping("consistency_score",     2.0),
                MetricMapping("prediction_reliability", 1.5),
                MetricMapping("cross_entropy",         1.0),
            ],
        ),
        CategoryDefinition(
            name="trading_performance",
            display_name="Trading Performance",
            weight=0.15,
            metric_mappings=[
                MetricMapping("sharpe_ratio",       3.0),
                MetricMapping("sortino_ratio",       2.0),
                MetricMapping("calmar_ratio",        2.0),
                MetricMapping("profit_factor",       2.5),
                MetricMapping("max_drawdown",        3.0),
                MetricMapping("expectancy",          2.0),
                MetricMapping("win_rate",            1.5),
                MetricMapping("average_r_multiple",  1.5),
                MetricMapping("tp_before_sl_accuracy", 1.0),
                MetricMapping("capital_efficiency",  1.0),
                MetricMapping("recovery_factor",     1.0),
            ],
        ),
        CategoryDefinition(
            name="calibration",
            display_name="Calibration",
            weight=0.07,
            metric_mappings=[
                MetricMapping("expected_calibration_error", 3.0),
                MetricMapping("brier_score",                2.0),
                MetricMapping("confidence_reliability",     2.0),
                MetricMapping("prediction_entropy",         1.5),
                MetricMapping("probability_calibration",    1.5),
            ],
        ),
        CategoryDefinition(
            name="generalization",
            display_name="Generalization",
            weight=0.10,
            metric_mappings=[
                MetricMapping("cross_regime_stability",    2.5),
                MetricMapping("cross_asset_stability",     2.0),
                MetricMapping("cross_timeframe_stability", 2.0),
                MetricMapping("oos_performance",           3.0),
                MetricMapping("recent_data_performance",   2.0),
                MetricMapping("historical_stability",      1.5),
            ],
        ),
        CategoryDefinition(
            name="robustness",
            display_name="Robustness",
            weight=0.10,
            metric_mappings=[
                MetricMapping("monte_carlo_stability",      2.5),
                MetricMapping("stress_test_pass_rate",      3.0),
                MetricMapping("noise_resistance",           2.0),
                MetricMapping("data_gap_tolerance",         1.5),
                MetricMapping("spread_expansion_tolerance", 2.0),
                MetricMapping("drift_resistance",           2.0),
            ],
        ),
        CategoryDefinition(
            name="efficiency",
            display_name="Efficiency",
            weight=0.05,
            metric_mappings=[
                MetricMapping("inference_latency_ms", 3.0),
                MetricMapping("training_time_hours",  2.0),
                MetricMapping("gpu_utilization",      1.5),
                MetricMapping("ram_usage_gb",         2.0),
                MetricMapping("model_size_mb",        1.5),
                MetricMapping("storage_usage_gb",     1.0),
            ],
        ),
        CategoryDefinition(
            name="explainability",
            display_name="Explainability",
            weight=0.05,
            metric_mappings=[
                MetricMapping("explainability_coverage",    2.0),
                MetricMapping("feature_attribution_quality", 2.0),
                MetricMapping("attention_quality",          1.5),
                MetricMapping("prediction_traceability",    2.0),
                MetricMapping("decision_transparency",      1.5),
                MetricMapping("documentation_completeness", 2.0),
            ],
        ),
        CategoryDefinition(
            name="operational_readiness",
            display_name="Operational Readiness",
            weight=0.05,
            metric_mappings=[
                MetricMapping("deployment_readiness", 2.5),
                MetricMapping("rollback_readiness",   2.0),
                MetricMapping("health_monitoring",    2.0),
                MetricMapping("monitoring_coverage",  2.0),
                MetricMapping("audit_coverage",       1.5),
                MetricMapping("drift_monitoring",     2.0),
            ],
        ),
        CategoryDefinition(
            name="risk_management",
            display_name="Risk Management",
            weight=0.08,
            metric_mappings=[
                MetricMapping("value_at_risk_95",       2.5),
                MetricMapping("expected_shortfall_95",  2.5),
                MetricMapping("tail_risk_ratio",        2.0),
                MetricMapping("concentration_risk",     1.5),
                MetricMapping("market_correlation",     1.5),
                MetricMapping("max_consecutive_losses", 1.5),
                MetricMapping("risk_adjusted_return",   2.5),
            ],
        ),
        CategoryDefinition(
            name="data_quality",
            display_name="Data Quality",
            weight=0.05,
            metric_mappings=[
                MetricMapping("data_completeness",  2.5),
                MetricMapping("data_consistency",   2.5),
                MetricMapping("label_quality",      2.5),
                MetricMapping("feature_coverage",   1.5),
                MetricMapping("data_freshness",     1.5),
                MetricMapping("outlier_fraction",   2.0),
            ],
        ),
        CategoryDefinition(
            name="feature_quality",
            display_name="Feature Quality",
            weight=0.04,
            metric_mappings=[
                MetricMapping("feature_importance_concentration", 2.0),
                MetricMapping("feature_stability_index",          2.5),
                MetricMapping("feature_predictive_power",         2.5),
                MetricMapping("feature_redundancy_score",         2.0),
            ],
        ),
        CategoryDefinition(
            name="model_stability",
            display_name="Model Stability",
            weight=0.04,
            metric_mappings=[
                MetricMapping("weight_update_stability", 2.0),
                MetricMapping("prediction_variance",     2.5),
                MetricMapping("gradient_stability",      2.0),
                MetricMapping("checkpoint_consistency",  2.0),
                MetricMapping("training_loss_stability", 2.0),
            ],
        ),
        CategoryDefinition(
            name="research_reproducibility",
            display_name="Research Reproducibility",
            weight=0.04,
            metric_mappings=[
                MetricMapping("environment_reproducibility", 2.5),
                MetricMapping("random_seed_stability",       2.0),
                MetricMapping("dataset_version_locked",      2.0),
                MetricMapping("code_version_locked",         2.0),
                MetricMapping("experiment_traceability",     2.0),
            ],
        ),
        CategoryDefinition(
            name="security",
            display_name="Security",
            weight=0.02,
            metric_mappings=[
                MetricMapping("adversarial_robustness",        2.0),
                MetricMapping("input_validation_coverage",     2.0),
                MetricMapping("output_sanitization",           2.0),
                MetricMapping("access_control_completeness",   2.0),
            ],
        ),
        CategoryDefinition(
            name="compliance",
            display_name="Compliance",
            weight=0.01,
            metric_mappings=[
                MetricMapping("regulatory_compliance_score", 2.5),
                MetricMapping("audit_trail_completeness",    2.5),
                MetricMapping("data_governance_score",       2.0),
                MetricMapping("model_governance_score",      2.0),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# CategoryResult
# ---------------------------------------------------------------------------


@dataclass
class CategoryResult:
    """Score for one Level 2 category."""

    category: str
    display_name: str
    score: float              # 0–100
    confidence: float         # 0–1
    weight: float             # this category's weight in Level 3
    weighted_score: float     # score * weight (for Level 3 aggregation)
    grade: str
    metric_scores: Dict[str, float]     # metric_name → normalised score (0–100)
    raw_values: Dict[str, float]        # metric_name → raw value
    metric_count: int = 0
    coverage: float = 0.0               # fraction of expected metrics present

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "display_name": self.display_name,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 4),
            "weight": self.weight,
            "weighted_score": round(self.weighted_score, 2),
            "grade": self.grade,
            "metric_count": self.metric_count,
            "coverage": round(self.coverage, 3),
            "metric_scores": {k: round(v, 2) for k, v in self.metric_scores.items()},
            "raw_values": {k: v for k, v in self.raw_values.items()},
        }


# ---------------------------------------------------------------------------
# HRSACategoryScorer
# ---------------------------------------------------------------------------

_CATEGORY_GRADE_THRESHOLDS = [
    (90.0, "Exceptional"),
    (80.0, "Strong"),
    (70.0, "Good"),
    (60.0, "Adequate"),
    (50.0, "Weak"),
    (0.0,  "Critical"),
]


def _category_grade(score: float) -> str:
    for threshold, grade in _CATEGORY_GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "Critical"


def _category_confidence(
    coverage: float, metric_count: int, scores: List[float]
) -> float:
    """Compute confidence for a category score.

    Higher coverage + more metrics + lower score variance → higher confidence.
    """
    # Coverage factor: 100% → 1.0, 50% → 0.5
    cov_factor = coverage
    # Count factor: 10+ metrics → 1.0, 1 metric → 0.1
    count_factor = min(1.0, metric_count / 10.0)
    # Variance factor: low variance → high confidence
    if len(scores) >= 2:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        var_factor = max(0.0, 1.0 - std / 30.0)  # std of 30 pts → 0 conf
    else:
        var_factor = 0.5
    return min(1.0, (cov_factor * 0.40 + count_factor * 0.35 + var_factor * 0.25))


class HRSACategoryScorer:
    """Level 2: scores all categories from a raw metrics bundle."""

    def __init__(
        self,
        categories: Optional[List[CategoryDefinition]] = None,
        registry: Optional[RawMetricsRegistry] = None,
    ) -> None:
        self._cats: Dict[str, CategoryDefinition] = {}
        for cat in (categories or _default_categories()):
            self._cats[cat.name] = cat
        self._registry = registry or RawMetricsRegistry.default()
        self._normaliser = CategoryNormaliser()

    @classmethod
    def default(cls) -> "HRSACategoryScorer":
        return cls()

    def score_category(
        self, bundle: RawMetricsBundle, category: str
    ) -> CategoryResult:
        cat_def = self._cats.get(category)
        if cat_def is None:
            raise KeyError(f"Unknown category: {category!r}")

        metric_values = bundle.for_category(category)
        reg_metrics = self._registry.for_category(category)

        # Compute per-metric scores using the registered normalisation bounds
        score_100, norm_results = self._normaliser.normalise_category(
            metric_values=metric_values,
            metric_definitions=reg_metrics,
            method=cat_def.normalisation_method,
            metric_weights=cat_def.metric_weights(),
        )

        metric_scores = {name: r.score_100 for name, r in norm_results.items()}
        coverage = (
            len(metric_scores) / len(cat_def.metric_mappings)
            if cat_def.metric_mappings else 0.0
        )
        confidence = _category_confidence(
            coverage=coverage,
            metric_count=len(metric_scores),
            scores=list(metric_scores.values()),
        )

        return CategoryResult(
            category=category,
            display_name=cat_def.display_name,
            score=score_100,
            confidence=confidence,
            weight=cat_def.weight,
            weighted_score=score_100 * cat_def.weight,
            grade=_category_grade(score_100),
            metric_scores=metric_scores,
            raw_values={k: v for k, v in metric_values.items()},
            metric_count=len(metric_scores),
            coverage=coverage,
        )

    def score_all_categories(
        self, bundle: RawMetricsBundle
    ) -> Dict[str, CategoryResult]:
        results: Dict[str, CategoryResult] = {}
        for cat_name in self._cats:
            results[cat_name] = self.score_category(bundle, cat_name)
        return results

    def category_names(self) -> List[str]:
        return list(self._cats.keys())

    def category_definitions(self) -> Dict[str, CategoryDefinition]:
        return dict(self._cats)

    def total_weight(self) -> float:
        return sum(c.weight for c in self._cats.values())
