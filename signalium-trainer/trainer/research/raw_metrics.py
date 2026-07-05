"""Phase 19.3 — HRSA Level 1: Raw Metrics.

Defines, registers, and collects all raw measurable metrics.
Every metric has a name, value, direction (higher/lower is better),
category membership, unit, and optional sample count.

No score may be calculated directly from raw metrics — all values
must flow through the Level 2 category scoring layer.

Usage::

    # Collecting metrics for one experiment
    bundle = RawMetricsBundle("exp_2026_001")
    bundle.set("walk_forward_accuracy", 0.692)
    bundle.set("sharpe_ratio", 1.82)
    bundle.set("max_drawdown", 0.067)

    # Registry provides metadata
    reg = RawMetricsRegistry.default()
    defn = reg.get("sharpe_ratio")
    print(defn.direction)   # "higher_is_better"
    print(defn.category)    # "trading_performance"

    # Convert to Level 2 input
    category_vals = bundle.for_category("trading_performance", reg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Metric Direction
# ---------------------------------------------------------------------------


class MetricDirection:
    HIGHER = "higher_is_better"
    LOWER  = "lower_is_better"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# MetricDefinition — registered descriptor for one metric
# ---------------------------------------------------------------------------


@dataclass
class MetricDefinition:
    name: str
    direction: str              # MetricDirection constant
    category: str               # Level 2 category name
    unit: str = ""
    description: str = ""
    normalisation_lo: float = 0.0   # target range lo (used in min-max norm)
    normalisation_hi: float = 1.0   # target range hi
    is_core: bool = True            # core metrics are required; others are optional
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "category": self.category,
            "unit": self.unit,
            "description": self.description,
            "normalisation_lo": self.normalisation_lo,
            "normalisation_hi": self.normalisation_hi,
            "is_core": self.is_core,
            "tags": self.tags,
        }


# ---------------------------------------------------------------------------
# Standard metric definitions (40+ metrics across 15 categories)
# ---------------------------------------------------------------------------

_DEFAULT_METRICS: List[Dict[str, Any]] = [
    # ── validation_quality ──────────────────────────────────────────────
    dict(name="walk_forward_accuracy", direction="higher", category="validation_quality",
         unit="fraction", lo=0.50, hi=0.75, description="Walk-forward fold accuracy"),
    dict(name="precision_macro", direction="higher", category="validation_quality",
         unit="fraction", lo=0.50, hi=0.80),
    dict(name="recall_macro", direction="higher", category="validation_quality",
         unit="fraction", lo=0.50, hi=0.80),
    dict(name="f1_macro", direction="higher", category="validation_quality",
         unit="fraction", lo=0.50, hi=0.75),
    dict(name="roc_auc", direction="higher", category="validation_quality",
         unit="fraction", lo=0.55, hi=0.85),
    dict(name="pr_auc", direction="higher", category="validation_quality",
         unit="fraction", lo=0.40, hi=0.80),
    dict(name="cross_entropy", direction="lower", category="validation_quality",
         unit="nats", lo=0.30, hi=1.10),
    dict(name="n_validation_folds", direction="higher", category="validation_quality",
         unit="count", lo=4.0, hi=12.0),
    dict(name="fold_score_std", direction="lower", category="validation_quality",
         unit="fraction", lo=0.0, hi=0.10),
    dict(name="consistency_score", direction="higher", category="validation_quality",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="prediction_reliability", direction="higher", category="validation_quality",
         unit="fraction", lo=0.60, hi=0.95),

    # ── trading_performance ─────────────────────────────────────────────
    dict(name="profit_factor", direction="higher", category="trading_performance",
         unit="ratio", lo=1.0, hi=2.5),
    dict(name="expectancy", direction="higher", category="trading_performance",
         unit="R", lo=0.0, hi=0.5),
    dict(name="sharpe_ratio", direction="higher", category="trading_performance",
         unit="ratio", lo=0.5, hi=2.5),
    dict(name="sortino_ratio", direction="higher", category="trading_performance",
         unit="ratio", lo=0.5, hi=3.0),
    dict(name="calmar_ratio", direction="higher", category="trading_performance",
         unit="ratio", lo=0.5, hi=3.0),
    dict(name="max_drawdown", direction="lower", category="trading_performance",
         unit="fraction", lo=0.03, hi=0.25),
    dict(name="win_rate", direction="higher", category="trading_performance",
         unit="fraction", lo=0.40, hi=0.65),
    dict(name="average_r_multiple", direction="higher", category="trading_performance",
         unit="R", lo=0.3, hi=2.0),
    dict(name="tp_before_sl_accuracy", direction="higher", category="trading_performance",
         unit="fraction", lo=0.45, hi=0.75),
    dict(name="capital_efficiency", direction="higher", category="trading_performance",
         unit="fraction", lo=0.0, hi=1.0),
    dict(name="recovery_factor", direction="higher", category="trading_performance",
         unit="ratio", lo=1.0, hi=5.0),
    dict(name="average_mfe", direction="higher", category="trading_performance",
         unit="R", lo=0.5, hi=3.0, is_core=False),
    dict(name="average_mae", direction="lower", category="trading_performance",
         unit="R", lo=0.1, hi=1.0, is_core=False),

    # ── calibration ─────────────────────────────────────────────────────
    dict(name="expected_calibration_error", direction="lower", category="calibration",
         unit="fraction", lo=0.0, hi=0.10),
    dict(name="brier_score", direction="lower", category="calibration",
         unit="score", lo=0.10, hi=0.30),
    dict(name="prediction_entropy", direction="lower", category="calibration",
         unit="bits", lo=0.0, hi=1.5),
    dict(name="confidence_reliability", direction="higher", category="calibration",
         unit="fraction", lo=0.60, hi=0.95),
    dict(name="probability_calibration", direction="higher", category="calibration",
         unit="fraction", lo=0.70, hi=1.0),

    # ── generalization ──────────────────────────────────────────────────
    dict(name="cross_regime_stability", direction="higher", category="generalization",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="cross_asset_stability", direction="higher", category="generalization",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="cross_timeframe_stability", direction="higher", category="generalization",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="oos_performance", direction="higher", category="generalization",
         unit="fraction", lo=0.50, hi=0.75),
    dict(name="recent_data_performance", direction="higher", category="generalization",
         unit="fraction", lo=0.50, hi=0.75),
    dict(name="historical_stability", direction="higher", category="generalization",
         unit="fraction", lo=0.60, hi=1.0),

    # ── robustness ──────────────────────────────────────────────────────
    dict(name="monte_carlo_stability", direction="higher", category="robustness",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="stress_test_pass_rate", direction="higher", category="robustness",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="noise_resistance", direction="higher", category="robustness",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="data_gap_tolerance", direction="higher", category="robustness",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="spread_expansion_tolerance", direction="higher", category="robustness",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="drift_resistance", direction="higher", category="robustness",
         unit="fraction", lo=0.50, hi=1.0),

    # ── efficiency ──────────────────────────────────────────────────────
    dict(name="inference_latency_ms", direction="lower", category="efficiency",
         unit="ms", lo=5.0, hi=100.0),
    dict(name="training_time_hours", direction="lower", category="efficiency",
         unit="hours", lo=1.0, hi=48.0),
    dict(name="gpu_utilization", direction="higher", category="efficiency",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="cpu_utilization", direction="higher", category="efficiency",
         unit="fraction", lo=0.30, hi=1.0),
    dict(name="ram_usage_gb", direction="lower", category="efficiency",
         unit="GB", lo=2.0, hi=64.0),
    dict(name="model_size_mb", direction="lower", category="efficiency",
         unit="MB", lo=50.0, hi=2000.0),
    dict(name="storage_usage_gb", direction="lower", category="efficiency",
         unit="GB", lo=1.0, hi=100.0),

    # ── explainability ──────────────────────────────────────────────────
    dict(name="explainability_coverage", direction="higher", category="explainability",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="feature_attribution_quality", direction="higher", category="explainability",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="attention_quality", direction="higher", category="explainability",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="prediction_traceability", direction="higher", category="explainability",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="decision_transparency", direction="higher", category="explainability",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="documentation_completeness", direction="higher", category="explainability",
         unit="fraction", lo=0.50, hi=1.0),

    # ── operational_readiness ────────────────────────────────────────────
    dict(name="deployment_readiness", direction="higher", category="operational_readiness",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="rollback_readiness", direction="higher", category="operational_readiness",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="health_monitoring", direction="higher", category="operational_readiness",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="monitoring_coverage", direction="higher", category="operational_readiness",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="audit_coverage", direction="higher", category="operational_readiness",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="drift_monitoring", direction="higher", category="operational_readiness",
         unit="fraction", lo=0.50, hi=1.0),

    # ── risk_management (NEW) ───────────────────────────────────────────
    dict(name="value_at_risk_95", direction="lower", category="risk_management",
         unit="fraction", lo=0.01, hi=0.10),
    dict(name="expected_shortfall_95", direction="lower", category="risk_management",
         unit="fraction", lo=0.02, hi=0.15),
    dict(name="tail_risk_ratio", direction="lower", category="risk_management",
         unit="ratio", lo=0.0, hi=2.0),
    dict(name="concentration_risk", direction="lower", category="risk_management",
         unit="fraction", lo=0.0, hi=0.50),
    dict(name="market_correlation", direction="lower", category="risk_management",
         unit="correlation", lo=0.0, hi=0.80),
    dict(name="max_consecutive_losses", direction="lower", category="risk_management",
         unit="count", lo=0.0, hi=10.0),
    dict(name="risk_adjusted_return", direction="higher", category="risk_management",
         unit="ratio", lo=0.5, hi=3.0),

    # ── data_quality (NEW) ──────────────────────────────────────────────
    dict(name="data_completeness", direction="higher", category="data_quality",
         unit="fraction", lo=0.90, hi=1.0),
    dict(name="data_consistency", direction="higher", category="data_quality",
         unit="fraction", lo=0.85, hi=1.0),
    dict(name="label_quality", direction="higher", category="data_quality",
         unit="fraction", lo=0.80, hi=1.0),
    dict(name="feature_coverage", direction="higher", category="data_quality",
         unit="fraction", lo=0.70, hi=1.0),
    dict(name="data_freshness", direction="higher", category="data_quality",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="outlier_fraction", direction="lower", category="data_quality",
         unit="fraction", lo=0.0, hi=0.05),

    # ── feature_quality (NEW) ───────────────────────────────────────────
    dict(name="feature_importance_concentration", direction="lower", category="feature_quality",
         unit="gini", lo=0.0, hi=0.80),
    dict(name="feature_stability_index", direction="higher", category="feature_quality",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="feature_predictive_power", direction="higher", category="feature_quality",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="feature_redundancy_score", direction="lower", category="feature_quality",
         unit="fraction", lo=0.0, hi=0.50),

    # ── model_stability (NEW) ───────────────────────────────────────────
    dict(name="weight_update_stability", direction="higher", category="model_stability",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="prediction_variance", direction="lower", category="model_stability",
         unit="variance", lo=0.0, hi=0.10),
    dict(name="gradient_stability", direction="higher", category="model_stability",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="checkpoint_consistency", direction="higher", category="model_stability",
         unit="fraction", lo=0.80, hi=1.0),
    dict(name="training_loss_stability", direction="higher", category="model_stability",
         unit="fraction", lo=0.60, hi=1.0),

    # ── research_reproducibility (NEW) ──────────────────────────────────
    dict(name="environment_reproducibility", direction="higher", category="research_reproducibility",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="random_seed_stability", direction="higher", category="research_reproducibility",
         unit="fraction", lo=0.80, hi=1.0),
    dict(name="dataset_version_locked", direction="higher", category="research_reproducibility",
         unit="fraction", lo=0.0, hi=1.0),
    dict(name="code_version_locked", direction="higher", category="research_reproducibility",
         unit="fraction", lo=0.0, hi=1.0),
    dict(name="experiment_traceability", direction="higher", category="research_reproducibility",
         unit="fraction", lo=0.50, hi=1.0),

    # ── security (NEW) ──────────────────────────────────────────────────
    dict(name="adversarial_robustness", direction="higher", category="security",
         unit="fraction", lo=0.50, hi=1.0),
    dict(name="input_validation_coverage", direction="higher", category="security",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="output_sanitization", direction="higher", category="security",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="access_control_completeness", direction="higher", category="security",
         unit="fraction", lo=0.70, hi=1.0),

    # ── compliance (NEW) ────────────────────────────────────────────────
    dict(name="regulatory_compliance_score", direction="higher", category="compliance",
         unit="fraction", lo=0.70, hi=1.0),
    dict(name="audit_trail_completeness", direction="higher", category="compliance",
         unit="fraction", lo=0.70, hi=1.0),
    dict(name="data_governance_score", direction="higher", category="compliance",
         unit="fraction", lo=0.60, hi=1.0),
    dict(name="model_governance_score", direction="higher", category="compliance",
         unit="fraction", lo=0.60, hi=1.0),
]


# ---------------------------------------------------------------------------
# RawMetricsRegistry
# ---------------------------------------------------------------------------


class RawMetricsRegistry:
    """Central registry of all known raw metric definitions."""

    def __init__(self) -> None:
        self._definitions: Dict[str, MetricDefinition] = {}

    @classmethod
    def default(cls) -> "RawMetricsRegistry":
        """Build registry from all 80+ standard metric definitions."""
        registry = cls()
        for m in _DEFAULT_METRICS:
            direction = (
                MetricDirection.HIGHER if m["direction"] == "higher"
                else MetricDirection.LOWER if m["direction"] == "lower"
                else MetricDirection.NEUTRAL
            )
            registry.register(MetricDefinition(
                name=m["name"],
                direction=direction,
                category=m["category"],
                unit=m.get("unit", ""),
                description=m.get("description", ""),
                normalisation_lo=m.get("lo", 0.0),
                normalisation_hi=m.get("hi", 1.0),
                is_core=m.get("is_core", True),
            ))
        return registry

    def register(self, defn: MetricDefinition) -> None:
        """Register or overwrite a metric definition."""
        self._definitions[defn.name] = defn

    def get(self, name: str) -> Optional[MetricDefinition]:
        return self._definitions.get(name)

    def get_required(self, name: str) -> MetricDefinition:
        defn = self._definitions.get(name)
        if defn is None:
            raise KeyError(f"Metric not registered: {name!r}")
        return defn

    def for_category(self, category: str) -> List[MetricDefinition]:
        return [d for d in self._definitions.values() if d.category == category]

    def all_categories(self) -> List[str]:
        cats = {d.category for d in self._definitions.values()}
        return sorted(cats)

    def all_names(self) -> List[str]:
        return sorted(self._definitions.keys())

    def total(self) -> int:
        return len(self._definitions)

    def to_dict(self) -> Dict[str, Any]:
        return {name: defn.to_dict() for name, defn in self._definitions.items()}


# ---------------------------------------------------------------------------
# RawMetricsBundle — one experiment's collected metric values
# ---------------------------------------------------------------------------


class RawMetricsBundle:
    """Container for raw metric values collected for one experiment."""

    def __init__(
        self,
        experiment_id: str = "",
        registry: Optional[RawMetricsRegistry] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self._registry = registry or RawMetricsRegistry.default()
        self._values: Dict[str, float] = {}
        self._sample_counts: Dict[str, int] = {}

    def set(self, name: str, value: float, sample_count: int = 0) -> None:
        """Set a raw metric value."""
        self._values[name] = value
        if sample_count > 0:
            self._sample_counts[name] = sample_count

    def get(self, name: str) -> Optional[float]:
        return self._values.get(name)

    def get_with_default(self, name: str, default: float = 0.0) -> float:
        return self._values.get(name, default)

    def has(self, name: str) -> bool:
        return name in self._values

    def for_category(self, category: str) -> Dict[str, float]:
        """Return all metric values belonging to a category."""
        cat_metrics = self._registry.for_category(category)
        return {
            m.name: self._values[m.name]
            for m in cat_metrics
            if m.name in self._values
        }

    def coverage_for_category(self, category: str) -> float:
        """Fraction of registered category metrics that have values."""
        cat_metrics = self._registry.for_category(category)
        if not cat_metrics:
            return 0.0
        present = sum(1 for m in cat_metrics if m.name in self._values)
        return present / len(cat_metrics)

    def set_many(self, values: Dict[str, float]) -> None:
        for name, value in values.items():
            self.set(name, value)

    def all_values(self) -> Dict[str, float]:
        return dict(self._values)

    def all_categories_covered(self) -> List[str]:
        covered = set()
        for name in self._values:
            defn = self._registry.get(name)
            if defn:
                covered.add(defn.category)
        return sorted(covered)

    def total_metrics(self) -> int:
        return len(self._values)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "metrics": dict(self._values),
            "sample_counts": dict(self._sample_counts),
            "total_metrics": len(self._values),
            "categories_covered": self.all_categories_covered(),
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        registry: Optional[RawMetricsRegistry] = None,
    ) -> "RawMetricsBundle":
        bundle = cls(
            experiment_id=d.get("experiment_id", ""),
            registry=registry,
        )
        bundle.set_many(d.get("metrics", {}))
        return bundle
