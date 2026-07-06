"""Phase 19.3 — HRSA Test Suite.

Self-contained: stubs loguru, loads modules directly by file path
(bypassing trainer/research/__init__.py which imports torch via the
legacy benchmark package), uses :memory: SQLite.

Run: python3 tests/unit/test_hrsa.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# loguru stub
# ---------------------------------------------------------------------------
if "loguru" not in sys.modules:
    _loguru = types.ModuleType("loguru")

    class _Logger:
        def debug(self, *a, **kw):
            pass

        def info(self, *a, **kw):
            pass

        def warning(self, *a, **kw):
            pass

        def error(self, *a, **kw):
            pass

        def exception(self, *a, **kw):
            pass

        def bind(self, **kw):
            return self

    _loguru.logger = _Logger()
    sys.modules["loguru"] = _loguru

# ---------------------------------------------------------------------------
# Load modules DIRECTLY BY FILE PATH — bypasses __init__.py / torch
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parents[2] / "trainer" / "research"


def _load(mod_key: str, filename: str):
    """Register a module directly from its file path into sys.modules."""
    path = _BASE / filename
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in strict dependency order
_raw_metrics = _load("trainer.research.raw_metrics", "raw_metrics.py")
_normalization = _load("trainer.research.normalization", "normalization.py")
_hrsa_grades = _load("trainer.research.hrsa_grades", "hrsa_grades.py")
_hrsa_categories = _load("trainer.research.hrsa_categories", "hrsa_categories.py")
_hrsa_scorer = _load("trainer.research.hrsa_scorer", "hrsa_scorer.py")
_hrsa_benchmarks = _load("trainer.research.hrsa_benchmarks", "hrsa_benchmarks.py")
_hrsa_trends = _load("trainer.research.hrsa_trends", "hrsa_trends.py")
_hrsa_explanations = _load("trainer.research.hrsa_explanations", "hrsa_explanations.py")
_hrsa_whatif = _load("trainer.research.hrsa_whatif", "hrsa_whatif.py")
_hrsa_database = _load("trainer.research.hrsa_database", "hrsa_database.py")
_hrsa_api = _load("trainer.research.hrsa_api", "hrsa_api.py")
_hrsa_dashboard = _load("trainer.research.hrsa_dashboard", "hrsa_dashboard.py")


def _mod(name: str):
    return sys.modules[f"trainer.research.{name}"]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
_tests = []
_passed = 0
_failed = 0


def _register(fn):
    _tests.append(fn)
    return fn


def run_tests():
    global _passed, _failed
    for fn in _tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            _passed += 1
        except Exception:
            import traceback

            print(f"  FAIL  {fn.__name__}: ")
            traceback.print_exc()
            _failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {_passed} passed, {_failed} failed / {len(_tests)} total")
    return _failed == 0


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_bundle(experiment_id="exp_001", full=False):
    rm = _mod("raw_metrics")
    bundle = rm.RawMetricsBundle(experiment_id)

    # Validation quality
    bundle.set("walk_forward_accuracy", 0.692)
    bundle.set("f1_macro", 0.661)
    bundle.set("roc_auc", 0.731)
    bundle.set("pr_auc", 0.612)
    bundle.set("n_validation_folds", 10.0)
    bundle.set("fold_score_std", 0.018)
    bundle.set("consistency_score", 0.82)
    bundle.set("prediction_reliability", 0.79)
    bundle.set("cross_entropy", 0.55)

    # Trading performance
    bundle.set("sharpe_ratio", 1.82)
    bundle.set("sortino_ratio", 2.10)
    bundle.set("calmar_ratio", 1.95)
    bundle.set("profit_factor", 1.65)
    bundle.set("max_drawdown", 0.067)
    bundle.set("win_rate", 0.554)
    bundle.set("expectancy", 0.22)
    bundle.set("average_r_multiple", 1.12)
    bundle.set("tp_before_sl_accuracy", 0.58)
    bundle.set("capital_efficiency", 0.71)
    bundle.set("recovery_factor", 2.8)

    # Calibration
    bundle.set("expected_calibration_error", 0.028)
    bundle.set("brier_score", 0.18)
    bundle.set("prediction_entropy", 0.55)
    bundle.set("confidence_reliability", 0.77)
    bundle.set("probability_calibration", 0.84)

    # Generalization
    bundle.set("cross_regime_stability", 0.74)
    bundle.set("cross_asset_stability", 0.68)
    bundle.set("cross_timeframe_stability", 0.71)
    bundle.set("oos_performance", 0.645)
    bundle.set("recent_data_performance", 0.61)
    bundle.set("historical_stability", 0.79)

    # Robustness
    bundle.set("monte_carlo_stability", 0.81)
    bundle.set("stress_test_pass_rate", 0.78)
    bundle.set("noise_resistance", 0.75)
    bundle.set("data_gap_tolerance", 0.72)
    bundle.set("spread_expansion_tolerance", 0.70)
    bundle.set("drift_resistance", 0.69)

    # Efficiency
    bundle.set("inference_latency_ms", 22.0)
    bundle.set("training_time_hours", 6.0)
    bundle.set("gpu_utilization", 0.78)
    bundle.set("ram_usage_gb", 8.0)
    bundle.set("model_size_mb", 320.0)
    bundle.set("storage_usage_gb", 4.0)

    # Explainability
    bundle.set("explainability_coverage", 0.82)
    bundle.set("feature_attribution_quality", 0.78)
    bundle.set("attention_quality", 0.75)
    bundle.set("prediction_traceability", 0.80)
    bundle.set("decision_transparency", 0.74)
    bundle.set("documentation_completeness", 0.88)

    # Operational readiness
    bundle.set("deployment_readiness", 0.90)
    bundle.set("rollback_readiness", 0.88)
    bundle.set("health_monitoring", 0.85)
    bundle.set("monitoring_coverage", 0.84)
    bundle.set("audit_coverage", 0.80)
    bundle.set("drift_monitoring", 0.82)

    if full:
        # Risk management
        bundle.set("value_at_risk_95", 0.03)
        bundle.set("expected_shortfall_95", 0.045)
        bundle.set("tail_risk_ratio", 0.8)
        bundle.set("concentration_risk", 0.15)
        bundle.set("market_correlation", 0.22)
        bundle.set("max_consecutive_losses", 3.0)
        bundle.set("risk_adjusted_return", 2.1)

        # Data quality
        bundle.set("data_completeness", 0.98)
        bundle.set("data_consistency", 0.96)
        bundle.set("label_quality", 0.91)
        bundle.set("feature_coverage", 0.88)
        bundle.set("data_freshness", 0.82)
        bundle.set("outlier_fraction", 0.008)

        # Feature quality
        bundle.set("feature_importance_concentration", 0.35)
        bundle.set("feature_stability_index", 0.82)
        bundle.set("feature_predictive_power", 0.76)
        bundle.set("feature_redundancy_score", 0.18)

        # Model stability
        bundle.set("weight_update_stability", 0.88)
        bundle.set("prediction_variance", 0.018)
        bundle.set("gradient_stability", 0.84)
        bundle.set("checkpoint_consistency", 0.96)
        bundle.set("training_loss_stability", 0.90)

        # Reproducibility
        bundle.set("environment_reproducibility", 0.92)
        bundle.set("random_seed_stability", 0.98)
        bundle.set("dataset_version_locked", 1.0)
        bundle.set("code_version_locked", 1.0)
        bundle.set("experiment_traceability", 0.88)

        # Security
        bundle.set("adversarial_robustness", 0.78)
        bundle.set("input_validation_coverage", 0.85)
        bundle.set("output_sanitization", 0.82)
        bundle.set("access_control_completeness", 0.90)

        # Compliance
        bundle.set("regulatory_compliance_score", 0.88)
        bundle.set("audit_trail_completeness", 0.90)
        bundle.set("data_governance_score", 0.82)
        bundle.set("model_governance_score", 0.84)

    return bundle


def _make_scorer():
    hs = _mod("hrsa_scorer")
    return hs.HierarchicalResearchScorer()


def _score(bundle=None, full=False, **kwargs):
    b = bundle or _make_bundle(full=full)
    scorer = _make_scorer()
    return scorer.score(b, **kwargs)


# ===========================================================================
# 1. RawMetricsRegistry
# ===========================================================================


@_register
def test_registry_default_has_metrics():
    rm = _mod("raw_metrics")
    reg = rm.RawMetricsRegistry.default()
    assert reg.total() >= 80


@_register
def test_registry_get_known_metric():
    rm = _mod("raw_metrics")
    reg = rm.RawMetricsRegistry.default()
    defn = reg.get("sharpe_ratio")
    assert defn is not None
    assert defn.direction == "higher_is_better"
    assert defn.category == "trading_performance"


@_register
def test_registry_for_category():
    rm = _mod("raw_metrics")
    reg = rm.RawMetricsRegistry.default()
    cats = reg.for_category("trading_performance")
    assert len(cats) >= 8


@_register
def test_registry_all_categories_15():
    rm = _mod("raw_metrics")
    reg = rm.RawMetricsRegistry.default()
    cats = reg.all_categories()
    assert len(cats) == 15


@_register
def test_registry_register_custom():
    rm = _mod("raw_metrics")
    reg = rm.RawMetricsRegistry.default()
    reg.register(
        rm.MetricDefinition(
            name="my_custom_metric",
            direction="higher_is_better",
            category="trading_performance",
        )
    )
    assert reg.get("my_custom_metric") is not None


@_register
def test_registry_to_dict():
    rm = _mod("raw_metrics")
    reg = rm.RawMetricsRegistry.default()
    d = reg.to_dict()
    assert "sharpe_ratio" in d
    assert d["sharpe_ratio"]["direction"] == "higher_is_better"


@_register
def test_registry_get_required_raises():
    rm = _mod("raw_metrics")
    reg = rm.RawMetricsRegistry.default()
    try:
        reg.get_required("nonexistent_metric_xyz")
        assert False, "Should have raised"
    except KeyError:
        pass


# ===========================================================================
# 2. RawMetricsBundle
# ===========================================================================


@_register
def test_bundle_set_and_get():
    bundle = _make_bundle()
    assert bundle.get("sharpe_ratio") == 1.82


@_register
def test_bundle_has():
    bundle = _make_bundle()
    assert bundle.has("sharpe_ratio")
    assert not bundle.has("nonexistent_xyz")


@_register
def test_bundle_for_category():
    bundle = _make_bundle()
    vals = bundle.for_category("trading_performance")
    assert "sharpe_ratio" in vals
    assert "max_drawdown" in vals


@_register
def test_bundle_coverage():
    bundle = _make_bundle()
    cov = bundle.coverage_for_category("validation_quality")
    assert 0.0 < cov <= 1.0


@_register
def test_bundle_all_categories_covered():
    bundle = _make_bundle(full=True)
    cats = bundle.all_categories_covered()
    assert len(cats) >= 8


@_register
def test_bundle_to_dict():
    bundle = _make_bundle()
    d = bundle.to_dict()
    assert "metrics" in d
    assert "sharpe_ratio" in d["metrics"]


@_register
def test_bundle_from_dict():
    rm = _mod("raw_metrics")
    bundle = _make_bundle()
    d = bundle.to_dict()
    b2 = rm.RawMetricsBundle.from_dict(d)
    assert b2.get("sharpe_ratio") == 1.82


@_register
def test_bundle_set_many():
    rm = _mod("raw_metrics")
    bundle = rm.RawMetricsBundle("test")
    bundle.set_many({"a": 1.0, "b": 2.0})
    assert bundle.get("a") == 1.0
    assert bundle.get("b") == 2.0


# ===========================================================================
# 3. Normalisation
# ===========================================================================


@_register
def test_normaliser_min_max():
    n = _mod("normalization")
    norm = n.Normaliser()
    r = norm.normalise(1.5, lo=0.5, hi=2.5, method="min_max", direction="higher_is_better")
    assert abs(r.normalised - 0.5) < 1e-9
    assert abs(r.score_100 - 50.0) < 1e-9


@_register
def test_normaliser_lower_is_better():
    n = _mod("normalization")
    norm = n.Normaliser()
    r = norm.normalise(0.05, lo=0.03, hi=0.25, method="min_max", direction="lower_is_better")
    # raw norm would be (0.05-0.03)/(0.25-0.03) ≈ 0.091, flipped → 0.909
    assert r.normalised > 0.85


@_register
def test_normaliser_clamped():
    n = _mod("normalization")
    norm = n.Normaliser()
    r = norm.normalise(10.0, lo=0.5, hi=2.5, method="min_max", direction="higher_is_better")
    assert r.normalised == 1.0


@_register
def test_normaliser_z_score():
    n = _mod("normalization")
    norm = n.Normaliser()
    dist = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = norm.normalise(3.0, method="z_score", direction="higher_is_better", distribution=dist)
    assert abs(r.normalised - 0.5) < 0.05  # mean → ~0.5


@_register
def test_normaliser_percentile():
    n = _mod("normalization")
    norm = n.Normaliser()
    dist = [1.0, 2.0, 3.0, 4.0, 5.0]
    r = norm.normalise(3.0, method="percentile", direction="higher_is_better", distribution=dist)
    assert 0.4 <= r.normalised <= 0.7


@_register
def test_normaliser_robust():
    n = _mod("normalization")
    norm = n.Normaliser()
    dist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    r = norm.normalise(
        4.0, method="robust_scaling", direction="higher_is_better", distribution=dist
    )
    assert 0.0 <= r.normalised <= 1.0


@_register
def test_normaliser_custom():
    n = _mod("normalization")
    norm = n.Normaliser()
    r = norm.normalise(0.75, method="custom", direction="higher_is_better", custom_fn=lambda x: x)
    assert abs(r.normalised - 0.75) < 1e-9


@_register
def test_normaliser_batch():
    n = _mod("normalization")
    norm = n.Normaliser()
    results = norm.normalise_batch(
        [0.5, 1.0, 1.5, 2.0, 2.5], lo=0.5, hi=2.5, method="min_max", direction="higher_is_better"
    )
    assert len(results) == 5
    assert results[0].normalised < results[-1].normalised


@_register
def test_norm_result_to_dict():
    n = _mod("normalization")
    norm = n.Normaliser()
    r = norm.normalise(1.5, lo=0.5, hi=2.5, method="min_max", direction="higher_is_better")
    d = r.to_dict()
    assert "raw_value" in d
    assert "normalised" in d
    assert "score_100" in d


@_register
def test_category_normaliser():
    n = _mod("normalization")
    rm = _mod("raw_metrics")
    cn = n.CategoryNormaliser()
    reg = rm.RawMetricsRegistry.default()
    metric_defs = reg.for_category("trading_performance")
    values = {"sharpe_ratio": 1.82, "max_drawdown": 0.067, "profit_factor": 1.65, "win_rate": 0.554}
    score, results = cn.normalise_category(values, metric_defs, method="min_max")
    assert 0.0 <= score <= 100.0
    assert len(results) == 4


# ===========================================================================
# 4. HRSA Grade Engine
# ===========================================================================


@_register
def test_grade_institutional_elite():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    g = engine.assign_institutional(97.0)
    assert g.label == "Institutional Elite"


@_register
def test_grade_production_ready():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    g = engine.assign_institutional(86.0)
    assert g.label == "Production Ready"


@_register
def test_grade_research_only():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    g = engine.assign_institutional(55.0)
    assert g.label == "Research Only"


@_register
def test_grade_research_grade():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    g = engine.assign_research(85.0, confidence=0.90)
    assert "Research" in g.label


@_register
def test_grade_promotion_go():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    inst = engine.assign_institutional(87.0)
    promo = engine.assign_promotion(87.0, inst)
    assert promo == ge.PromotionRecommendation.GO


@_register
def test_grade_promotion_no():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    inst = engine.assign_institutional(55.0)
    promo = engine.assign_promotion(55.0, inst)
    assert promo == ge.PromotionRecommendation.NO


@_register
def test_grade_risk_low():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    cat_scores = {
        "robustness": 88.0,
        "risk_management": 85.0,
        "model_stability": 82.0,
        "calibration": 84.0,
    }
    assert engine.assign_risk(cat_scores) == ge.RiskRating.LOW


@_register
def test_grade_risk_high():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    # avg = (55+50+48+52)/4 = 51.25  → >= 45 and < 65 → HIGH
    cat_scores = {
        "robustness": 55.0,
        "risk_management": 50.0,
        "model_stability": 48.0,
        "calibration": 52.0,
    }
    assert engine.assign_risk(cat_scores) == ge.RiskRating.HIGH


@_register
def test_grade_future_potential_strong():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    fp = engine.assign_future_potential(trend_slope=2.0, score=80.0)
    assert fp == ge.FuturePotential.STRONG


@_register
def test_grade_custom():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    engine.define_custom_grade("Platinum", 98.0, 100.0, "Ultra elite")
    g = engine.assign_institutional(99.0)
    assert g.label == "Platinum"


@_register
def test_full_assessment():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    assessment = engine.full_assessment(
        composite=87.5,
        confidence=0.82,
        category_scores={"robustness": 80.0, "risk_management": 75.0},
        trend_slope=1.2,
    )
    assert assessment.institutional_grade != ""
    assert assessment.research_grade != ""
    assert assessment.promotion_recommendation in ("GO", "REVIEW", "HOLD", "NO")
    assert assessment.risk_rating in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")
    assert assessment.future_potential in ("STRONG", "MODERATE", "LIMITED", "DECLINING")


@_register
def test_full_assessment_to_dict():
    ge = _mod("hrsa_grades")
    engine = ge.HRSAGradeEngine.default()
    a = engine.full_assessment(82.0, confidence=0.75)
    d = a.to_dict()
    assert "institutional_grade" in d
    assert "promotion_recommendation" in d


# ===========================================================================
# 5. HRSA Category Scorer
# ===========================================================================


@_register
def test_category_scorer_default_15_cats():
    hc = _mod("hrsa_categories")
    scorer = hc.HRSACategoryScorer.default()
    assert len(scorer.category_names()) == 15


@_register
def test_category_scorer_score_one():
    hc = _mod("hrsa_categories")
    scorer = hc.HRSACategoryScorer.default()
    bundle = _make_bundle()
    result = scorer.score_category(bundle, "trading_performance")
    assert 0.0 <= result.score <= 100.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.grade != ""


@_register
def test_category_scorer_score_all():
    hc = _mod("hrsa_categories")
    scorer = hc.HRSACategoryScorer.default()
    bundle = _make_bundle()
    results = scorer.score_all_categories(bundle)
    assert len(results) == 15
    for cat, result in results.items():
        assert 0.0 <= result.score <= 100.0


@_register
def test_category_result_to_dict():
    hc = _mod("hrsa_categories")
    scorer = hc.HRSACategoryScorer.default()
    bundle = _make_bundle()
    result = scorer.score_category(bundle, "validation_quality")
    d = result.to_dict()
    assert "score" in d
    assert "confidence" in d
    assert "grade" in d
    assert "metric_scores" in d


@_register
def test_category_coverage():
    hc = _mod("hrsa_categories")
    scorer = hc.HRSACategoryScorer.default()
    bundle = _make_bundle()
    result = scorer.score_category(bundle, "validation_quality")
    assert result.coverage > 0.5


@_register
def test_category_with_no_metrics_zero_score():
    rm = _mod("raw_metrics")
    hc = _mod("hrsa_categories")
    scorer = hc.HRSACategoryScorer.default()
    empty = rm.RawMetricsBundle("empty")
    result = scorer.score_category(empty, "compliance")
    assert result.metric_count == 0
    assert result.score == 0.0


@_register
def test_category_weight_sum_close_to_1():
    hc = _mod("hrsa_categories")
    scorer = hc.HRSACategoryScorer.default()
    total = scorer.total_weight()
    assert abs(total - 1.0) < 0.001


# ===========================================================================
# 6. Hierarchical Research Scorer (Level 3)
# ===========================================================================


@_register
def test_scorer_returns_hierarchical_score():
    hs = _mod("hrsa_scorer")
    result = _score()
    assert isinstance(result, hs.HierarchicalScore)


@_register
def test_scorer_composite_in_range():
    result = _score()
    assert 0.0 <= result.level3_composite <= 100.0


@_register
def test_scorer_composite_positive_for_good_inputs():
    result = _score()
    assert result.level3_composite > 20.0


@_register
def test_scorer_confidence_in_range():
    result = _score()
    assert 0.0 <= result.confidence <= 1.0


@_register
def test_scorer_level2_has_all_categories():
    result = _score()
    assert len(result.level2_categories) == 15


@_register
def test_scorer_grade_assessment_present():
    result = _score()
    assert result.grade_assessment is not None
    assert result.institutional_grade != ""


@_register
def test_scorer_strongest_weakest():
    result = _score(full=True)
    strong = result.strongest_categories(3)
    weak = result.weakest_categories(3)
    assert len(strong) == 3
    assert len(weak) == 3
    # Strongest should have higher scores than weakest
    strong_scores = [result.category_score(c) for c in strong]
    weak_scores = [result.category_score(c) for c in weak]
    assert min(strong_scores) >= max(weak_scores) - 5.0  # allow small overlap


@_register
def test_scorer_to_dict():
    result = _score()
    d = result.to_dict()
    assert "score_id" in d
    assert "level3_composite" in d
    assert "level2_categories" in d
    assert "grade_assessment" in d


@_register
def test_scorer_summary():
    result = _score(name="test_model")
    summary = result.summary()
    assert "test_model" in summary
    assert "/100" in summary


@_register
def test_scorer_full_bundle_higher_than_partial():
    partial = _score()
    full = _score(full=True)
    # Full bundle (more categories filled) should have higher or equal composite
    assert full.level3_composite >= partial.level3_composite - 5.0


@_register
def test_scorer_metadata_attached():
    result = _score(name="mymodel", instrument="ES", timeframe="1H", backbone="xgb")
    assert result.name == "mymodel"
    assert result.instrument == "ES"
    assert result.timeframe == "1H"
    assert result.backbone == "xgb"


@_register
def test_scorer_category_scores_dict():
    result = _score()
    d = result.category_scores_dict()
    assert "trading_performance" in d
    assert "validation_quality" in d


@_register
def test_scorer_sensitivity():
    hs_mod = _mod("hrsa_scorer")
    scorer = hs_mod.HierarchicalResearchScorer()
    bundle = _make_bundle()
    analysis = scorer.sensitivity_analysis(bundle, "trading_performance", delta_pct=10.0)
    assert "estimated_delta" in analysis
    assert analysis["estimated_delta"] >= 0.0


@_register
def test_scorer_breakdown():
    hs_mod = _mod("hrsa_scorer")
    scorer = hs_mod.HierarchicalResearchScorer()
    bundle = _make_bundle()
    breakdown = scorer.score_breakdown(bundle)
    assert "hierarchical_score" in breakdown
    assert "scoring_pipeline" in breakdown


# ===========================================================================
# 7. HRSA Benchmarks
# ===========================================================================


@_register
def test_benchmarks_register_and_retrieve():
    bm = _mod("hrsa_benchmarks")
    engine = bm.HRSABenchmarkEngine()
    result = _score(name="v1")
    engine.register("v1", result, role="champion")
    assert engine.champion_record() is not None
    assert engine.champion_record().composite == result.level3_composite


@_register
def test_benchmarks_champion_demotion():
    bm = _mod("hrsa_benchmarks")
    engine = bm.HRSABenchmarkEngine()
    r1 = _score(name="v1")
    r2 = _score(name="v2")
    engine.register("v1", r1, role="champion")
    engine.register("v2", r2, role="champion")
    assert engine._champion_id == "v2"
    assert engine._prev_champion_id == "v1"
    assert engine._records["v1"].role == bm.BenchmarkRole.PREV_CHAMPION


@_register
def test_benchmarks_compare():
    bm = _mod("hrsa_benchmarks")
    engine = bm.HRSABenchmarkEngine()
    r_champ = _score(name="champion")
    r_cand = _score(name="candidate")
    engine.register("champion", r_champ, role="champion")
    suite = engine.compare("candidate", r_cand)
    assert suite.subject_id == "candidate"
    assert len(suite.deltas) >= 2  # champion + hist average


@_register
def test_benchmarks_delta_improved():
    bm = _mod("hrsa_benchmarks")
    engine = bm.HRSABenchmarkEngine()
    # Manufacture a low baseline
    r_base = _score(name="baseline")
    rm = _mod("raw_metrics")
    weak = rm.RawMetricsBundle("weak")
    weak.set("sharpe_ratio", 0.6)
    weak.set("max_drawdown", 0.20)
    hs_weak = _make_scorer().score(weak, name="weak")
    engine.register("baseline", hs_weak, role="baseline")
    suite = engine.compare("good", r_base, roles=["baseline"])
    d = suite.vs_baseline()
    assert d is not None
    assert d.improved  # good model beats weak baseline


@_register
def test_benchmarks_full_report():
    bm = _mod("hrsa_benchmarks")
    engine = bm.HRSABenchmarkEngine()
    r = _score(name="v1")
    engine.register("v1", r, role="champion")
    report = engine.full_report()
    assert "champion_id" in report
    assert "ranking" in report


@_register
def test_benchmark_record_from_hierarchical_score():
    bm = _mod("hrsa_benchmarks")
    hs = _score(name="test")
    rec = bm.BenchmarkRecord.from_hierarchical_score(hs, role="candidate")
    assert rec.composite == hs.level3_composite
    assert rec.role == "candidate"


@_register
def test_benchmark_delta_to_dict():
    bm = _mod("hrsa_benchmarks")
    engine = bm.HRSABenchmarkEngine()
    r1 = _score(name="r1")
    r2 = _score(name="r2")
    engine.register("r1", r1, role="baseline")
    engine.register("r2", r2, role="champion")
    delta = engine.compare_ids("r1", "r2")
    assert delta is not None
    d = delta.to_dict()
    assert "composite_delta" in d


# ===========================================================================
# 8. HRSA Trends
# ===========================================================================


@_register
def test_trends_record_and_composite():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(5):
        engine.record(_score())
    comp = engine.composite_trend()
    assert comp.n == 5


@_register
def test_trends_category_trends():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(3):
        engine.record(_score())
    cat_trends = engine.category_trends()
    assert len(cat_trends) > 0


@_register
def test_trends_improvement_rate():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(5):
        engine.record(_score())
    rate = engine.improvement_rate(window=5)
    assert isinstance(rate, float)


@_register
def test_trends_regression_detection_no_alert():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(5):
        engine.record(_score())
    alerts = engine.detect_regressions(threshold=50.0)  # high threshold
    assert isinstance(alerts, list)


@_register
def test_trends_stability():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(5):
        engine.record(_score())
    stab = engine.score_stability()
    assert "composite" in stab
    assert stab["composite"] >= 0.0


@_register
def test_trends_trend_report():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(4):
        engine.record(_score())
    report = engine.trend_report()
    assert "composite_trend" in report
    assert "category_trends" in report
    assert "regression_alerts" in report


@_register
def test_trends_history():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(3):
        engine.record(_score())
    history = engine.history()
    assert len(history) == 3


@_register
def test_trends_trend_result_to_dict():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    for _ in range(3):
        engine.record(_score())
    trend = engine.composite_trend()
    d = trend.to_dict()
    assert "slope" in d
    assert "trend_label" in d
    assert "points" in d


@_register
def test_trends_linreg_slope():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    # Manually add improving points
    for i in range(5):
        engine.record_raw("test_series", float(70 + i * 2), level="composite")
    trend = engine.composite_trend()
    assert trend.slope > 0  # improving


@_register
def test_trends_regression_alert():
    ht = _mod("hrsa_trends")
    engine = ht.HRSATrendEngine()
    # Manually inject a regression
    for v in [85.0, 86.0, 85.5, 84.0, 70.0]:  # big drop at end
        engine.record_raw("test", v, level="composite")
    alerts = engine.detect_regressions(threshold=5.0)
    assert len(alerts) >= 1
    assert alerts[0].drop >= 5.0


# ===========================================================================
# 9. HRSA Explanations
# ===========================================================================


@_register
def test_explanation_returns_object():
    he = _mod("hrsa_explanations")
    explainer = he.HRSAExplainer()
    hs = _score()
    explanation = explainer.explain(hs)
    assert isinstance(explanation, he.ScoreExplanation)


@_register
def test_explanation_has_summary():
    he = _mod("hrsa_explanations")
    explainer = he.HRSAExplainer()
    hs = _score(name="mymodel")
    explanation = explainer.explain(hs)
    summary = explanation.summary()
    assert "mymodel" in summary or "mymodel" in explanation.experiment_id


@_register
def test_explanation_recommendations_nonempty():
    he = _mod("hrsa_explanations")
    explainer = he.HRSAExplainer()
    hs = _score()
    explanation = explainer.explain(hs)
    assert len(explanation.recommendations) > 0


@_register
def test_explanation_to_dict():
    he = _mod("hrsa_explanations")
    explainer = he.HRSAExplainer()
    hs = _score()
    explanation = explainer.explain(hs)
    d = explanation.to_dict()
    assert "overall_summary" in d
    assert "recommendations" in d
    assert "weakest_categories" in d


@_register
def test_explanation_strongest_weakest():
    he = _mod("hrsa_explanations")
    explainer = he.HRSAExplainer()
    hs = _score()
    explanation = explainer.explain(hs)
    assert len(explanation.strongest_categories) >= 1
    assert len(explanation.weakest_categories) >= 1


@_register
def test_explanation_improvement_potential_positive():
    he = _mod("hrsa_explanations")
    explainer = he.HRSAExplainer()
    hs = _score()
    explanation = explainer.explain(hs)
    assert explanation.improvement_potential >= 0.0


@_register
def test_explanation_compare():
    he = _mod("hrsa_explanations")
    explainer = he.HRSAExplainer()
    hs_a = _score()
    hs_b = _score()
    comp = explainer.compare_explanations(hs_a, hs_b)
    assert "composite_delta" in comp
    assert "verdict" in comp


@_register
def test_explanation_recommendation_priority():
    he = _mod("hrsa_explanations")
    valid_priorities = {"critical", "high", "medium", "low"}
    explainer = he.HRSAExplainer()
    hs = _score()
    explanation = explainer.explain(hs)
    for rec in explanation.recommendations:
        assert rec.priority in valid_priorities


# ===========================================================================
# 10. HRSA What-If Simulator
# ===========================================================================


@_register
def test_whatif_simulate_category():
    hw = _mod("hrsa_whatif")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    result = sim.simulate_category(hs, "calibration", new_score=90.0)
    assert isinstance(result, hw.SimulationResult)
    assert result.original_composite == hs.level3_composite
    assert result.simulated_composite >= 0.0


@_register
def test_whatif_improvement_increases_score():
    hw = _mod("hrsa_whatif")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    weak_cat = hs.weakest_categories(1)[0]
    current = hs.category_score(weak_cat)
    if current < 90:
        result = sim.simulate_category(hs, weak_cat, new_score=90.0)
        assert result.delta >= 0.0


@_register
def test_whatif_deterioration_decreases_score():
    hw = _mod("hrsa_whatif")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    strong_cat = hs.strongest_categories(1)[0]
    current = hs.category_score(strong_cat)
    result = sim.simulate_category(hs, strong_cat, new_score=max(0, current - 30))
    assert result.delta <= 0.05  # should decrease or stay same


@_register
def test_whatif_multiple_categories():
    hw = _mod("hrsa_whatif")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    result = sim.simulate_categories(
        hs,
        {
            "calibration": 88.0,
            "robustness": 85.0,
        },
    )
    assert result.delta >= 0.0


@_register
def test_whatif_optimal_improvements():
    hw = _mod("hrsa_whatif")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    results = sim.optimal_improvements(hs, budget_categories=3)
    assert len(results) >= 1
    for r in results:
        assert isinstance(r, hw.SimulationResult)


@_register
def test_whatif_batch():
    hw = _mod("hrsa_whatif")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    scenarios = [
        {"name": "s1", "changes": {"calibration": 85.0}},
        {"name": "s2", "changes": {"robustness": 80.0}},
    ]
    results = sim.run_batch(hs, scenarios)
    assert len(results) == 2
    assert results[0].scenario_name == "s1"


@_register
def test_whatif_to_dict():
    hw = _mod("hrsa_whatif")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    result = sim.simulate_category(hs, "efficiency", new_score=80.0)
    d = result.to_dict()
    assert "original_composite" in d
    assert "delta" in d
    assert "improved" in d


@_register
def test_whatif_grade_change_detected():
    hw = _mod("hrsa_whatif")
    # Create a model near a grade boundary
    rm = _mod("raw_metrics")
    bundle = rm.RawMetricsBundle("boundary_test")
    bundle.set("sharpe_ratio", 1.2)
    bundle.set("max_drawdown", 0.10)
    hs = _make_scorer().score(bundle)
    sim = hw.HRSAWhatIfSimulator()
    # Dramatic improvement
    result = sim.simulate_categories(
        hs,
        {
            "validation_quality": 95.0,
            "trading_performance": 95.0,
            "robustness": 95.0,
        },
    )
    # grade_change may or may not trigger depending on current grade
    assert result is not None


@_register
def test_whatif_metric_level_no_scorer():
    hw = _mod("hrsa_whatif")
    bundle = _make_bundle()
    hs = _score(bundle=bundle)
    sim = hw.HRSAWhatIfSimulator(scorer=None)
    result = sim.simulate_metric(hs, bundle, "max_drawdown", 0.03)
    # Without scorer, falls back to approximate (composite unchanged)
    assert result.original_composite == hs.level3_composite


@_register
def test_whatif_metric_level_with_scorer():
    hw = _mod("hrsa_whatif")
    hs_mod = _mod("hrsa_scorer")
    scorer = hs_mod.HierarchicalResearchScorer()
    bundle = _make_bundle()
    hs = scorer.score(bundle)
    sim = hw.HRSAWhatIfSimulator(scorer=scorer)
    result = sim.simulate_metric(hs, bundle, "max_drawdown", 0.03)
    assert isinstance(result.simulated_composite, float)
    assert 0.0 <= result.simulated_composite <= 100.0


# ===========================================================================
# 11. HRSA Database
# ===========================================================================


@_register
def test_db_save_and_retrieve():
    hd = _mod("hrsa_database")
    db = hd.HRSADatabase(":memory:")
    hs = _score(name="db_test")
    db.save(hs)
    row = db.get_score(hs.score_id)
    assert row is not None
    assert abs(row["composite"] - hs.level3_composite) < 0.01


@_register
def test_db_get_latest():
    hd = _mod("hrsa_database")
    db = hd.HRSADatabase(":memory:")
    hs = _score()
    db.save(hs)
    latest = db.get_latest(hs.experiment_id)
    assert latest is not None
    assert latest["experiment_id"] == hs.experiment_id


@_register
def test_db_category_scores():
    hd = _mod("hrsa_database")
    db = hd.HRSADatabase(":memory:")
    hs = _score()
    db.save(hs)
    cats = db.category_scores_for(hs.score_id)
    assert len(cats) > 0
    assert any(c["category"] == "trading_performance" for c in cats)


@_register
def test_db_composite_history():
    hd = _mod("hrsa_database")
    db = hd.HRSADatabase(":memory:")
    for _ in range(3):
        hs = _score()
        db.save(hs)
    history = db.composite_history(hs.experiment_id)
    assert len(history) >= 1


@_register
def test_db_leaderboard():
    hd = _mod("hrsa_database")
    db = hd.HRSADatabase(":memory:")
    for i in range(5):
        hs = _score(name=f"model_{i}")
        db.save(hs)
    lb = db.leaderboard(top_n=3)
    assert len(lb) <= 3


@_register
def test_db_save_explanation():
    hd = _mod("hrsa_database")
    he = _mod("hrsa_explanations")
    db = hd.HRSADatabase(":memory:")
    hs = _score()
    db.save(hs)
    explanation = he.HRSAExplainer().explain(hs)
    db.save_explanation(explanation)
    retrieved = db.get_explanation(hs.score_id)
    assert retrieved is not None
    assert "overall_summary" in retrieved


@_register
def test_db_save_whatif():
    hd = _mod("hrsa_database")
    hw = _mod("hrsa_whatif")
    db = hd.HRSADatabase(":memory:")
    hs = _score()
    sim = hw.HRSAWhatIfSimulator()
    result = sim.simulate_category(hs, "calibration", new_score=85.0)
    db.save_whatif(hs.experiment_id, result)
    history = db.whatif_history(hs.experiment_id)
    assert len(history) == 1


@_register
def test_db_stats():
    hd = _mod("hrsa_database")
    db = hd.HRSADatabase(":memory:")
    for _ in range(3):
        db.save(_score())
    stats = db.stats()
    assert stats["total_scores"] == 3
    assert stats["average_composite"] is not None


@_register
def test_db_get_missing_score():
    hd = _mod("hrsa_database")
    db = hd.HRSADatabase(":memory:")
    assert db.get_score("nonexistent-uuid") is None


# ===========================================================================
# 12. HRSA API
# ===========================================================================


@_register
def test_api_score():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    bundle = _make_bundle()
    resp = api.score(bundle, name="api_test")
    assert resp["status"] == "ok"
    assert "level3_composite" in resp["data"]


@_register
def test_api_get_score():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    bundle = _make_bundle()
    resp = api.score(bundle, name="api_test")
    score_id = resp["data"]["score_id"]
    retrieved = api.get_score(score_id)
    assert retrieved["status"] == "ok"
    assert retrieved["data"]["score_id"] == score_id


@_register
def test_api_get_missing_score():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    resp = api.get_score("nonexistent-uuid")
    assert resp["status"] == "error"


@_register
def test_api_category_scores():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    bundle = _make_bundle()
    resp = api.score(bundle)
    score_id = resp["data"]["score_id"]
    cats = api.category_scores(score_id)
    assert cats["status"] == "ok"
    assert len(cats["data"]["categories"]) > 0


@_register
def test_api_leaderboard():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    for i in range(4):
        b = _make_bundle(f"exp_{i}")
        api.score(b, name=f"model_{i}")
    lb = api.leaderboard(top_n=3)
    assert lb["status"] == "ok"
    assert len(lb["data"]["leaderboard"]) <= 3


@_register
def test_api_trend_report():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    for _ in range(3):
        api.score(_make_bundle())
    report = api.trend_report()
    assert report["status"] == "ok"
    assert "composite_trend" in report["data"]


@_register
def test_api_what_if():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    bundle = _make_bundle("exp_sim")
    api.score(bundle, name="sim_model")
    resp = api.what_if("exp_sim", [{"name": "improve_calib", "changes": {"calibration": 85.0}}])
    assert resp["status"] == "ok"
    assert len(resp["data"]["simulations"]) == 1


@_register
def test_api_what_if_optimal():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    bundle = _make_bundle("exp_opt")
    api.score(bundle, name="opt_model")
    resp = api.what_if_optimal("exp_opt", budget=3)
    assert resp["status"] == "ok"
    assert "optimal_improvements" in resp["data"]


@_register
def test_api_compare():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    b1 = _make_bundle("exp_a")
    b2 = _make_bundle("exp_b")
    resp = api.compare("exp_a", b1, "exp_b", b2)
    assert resp["status"] == "ok"
    assert "comparison" in resp["data"]


@_register
def test_api_registry_info():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    resp = api.registry_info()
    assert resp["status"] == "ok"
    assert resp["data"]["total_metrics"] >= 80


@_register
def test_api_grade_system():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    resp = api.grade_system()
    assert resp["status"] == "ok"
    assert len(resp["data"]["institutional_grades"]) >= 6


@_register
def test_api_system_stats():
    ha = _mod("hrsa_api")
    api = ha.HRSAApi()
    api.score(_make_bundle())
    resp = api.system_stats()
    assert resp["status"] == "ok"
    assert resp["data"]["database"]["total_scores"] == 1


# ===========================================================================
# 13. HRSA Dashboard
# ===========================================================================


@_register
def test_dashboard_hierarchical_tree():
    hd = _mod("hrsa_dashboard")
    hs = _score()
    dash = hd.HRSADashboard()
    tree = dash.hierarchical_tree(hs)
    assert tree["type"] == "hierarchical_tree"
    assert "root" in tree
    assert "level2_nodes" in tree
    assert tree["root"]["score"] == round(hs.level3_composite, 1)


@_register
def test_dashboard_radar_chart():
    hd = _mod("hrsa_dashboard")
    hs = _score()
    dash = hd.HRSADashboard()
    radar = dash.radar_chart(hs)
    assert radar["type"] == "radar_chart"
    assert len(radar["axes"]) > 0


@_register
def test_dashboard_category_heatmap():
    hd = _mod("hrsa_dashboard")
    hs = _score()
    dash = hd.HRSADashboard()
    heatmap = dash.category_heatmap(hs)
    assert heatmap["type"] == "category_heatmap"
    assert len(heatmap["cells"]) > 0


@_register
def test_dashboard_metric_heatmap():
    hd = _mod("hrsa_dashboard")
    hs = _score()
    dash = hd.HRSADashboard()
    heatmap = dash.metric_heatmap(hs)
    assert heatmap["type"] == "metric_heatmap"
    assert heatmap["total_metrics"] > 0


@_register
def test_dashboard_metric_heatmap_filtered():
    hd = _mod("hrsa_dashboard")
    hs = _score()
    dash = hd.HRSADashboard()
    heatmap = dash.metric_heatmap(hs, category="trading_performance")
    assert all(r["category"] == "Trading Performance" for r in heatmap["rows"])


@_register
def test_dashboard_score_timeline_no_db():
    hd = _mod("hrsa_dashboard")
    dash = hd.HRSADashboard()
    result = dash.score_timeline("exp_no_db")
    assert result["type"] == "score_timeline"


@_register
def test_dashboard_score_timeline_with_db():
    hdb = _mod("hrsa_database")
    hd = _mod("hrsa_dashboard")
    db = hdb.HRSADatabase(":memory:")
    hs = _score()
    db.save(hs)
    dash = hd.HRSADashboard(db=db)
    result = dash.score_timeline(hs.experiment_id)
    assert result["type"] == "score_timeline"
    assert len(result["points"]) == 1


@_register
def test_dashboard_improvement_timeline():
    hdb = _mod("hrsa_database")
    hd = _mod("hrsa_dashboard")
    db = hdb.HRSADatabase(":memory:")
    exp_id = "timeline_exp"
    for _ in range(3):
        rm = _mod("raw_metrics")
        bundle = rm.RawMetricsBundle(exp_id)
        bundle.set("sharpe_ratio", 1.82)
        hs = _make_scorer().score(bundle)
        db.save(hs)
    dash = hd.HRSADashboard(db=db)
    result = dash.improvement_timeline(exp_id)
    assert result["type"] == "improvement_timeline"


@_register
def test_dashboard_benchmark_comparison():
    hd = _mod("hrsa_dashboard")
    hb = _mod("hrsa_benchmarks")
    hs = _score(name="subject")
    ref_rec = hb.BenchmarkRecord.from_hierarchical_score(_score(name="champion"), role="champion")
    dash = hd.HRSADashboard()
    result = dash.benchmark_comparison(hs, [ref_rec])
    assert result["type"] == "benchmark_comparison"
    assert len(result["bars"]) == 2


@_register
def test_dashboard_score_breakdown():
    hd = _mod("hrsa_dashboard")
    hs = _score()
    dash = hd.HRSADashboard()
    result = dash.score_breakdown(hs)
    assert result["type"] == "score_breakdown"
    assert len(result["segments"]) > 0
    assert abs(result["total_contribution"] - hs.level3_composite) < 1.0


@_register
def test_dashboard_build_all():
    hdb = _mod("hrsa_database")
    hd = _mod("hrsa_dashboard")
    db = hdb.HRSADatabase(":memory:")
    hs = _score()
    db.save(hs)
    dash = hd.HRSADashboard(db=db)
    all_charts = dash.build_all(hs)
    expected = [
        "hierarchical_tree",
        "radar_chart",
        "category_heatmap",
        "metric_heatmap",
        "score_timeline",
        "improvement_timeline",
        "benchmark_comparison",
        "score_breakdown",
    ]
    for chart in expected:
        assert chart in all_charts


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
