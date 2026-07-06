"""Phase 19.2 — IRSE Test Suite.

Tests for:
  - ScoringWeights validation and normalisation
  - GradeSystem assignment and thresholds
  - All 8 category scorers (score range, sub-scores, edge cases)
  - ConfidenceCalculator
  - ResearchScorer (composite, breakdown, sensitivity)
  - ScoreTrendTracker (history, trend, regression, champion)
  - IRSELeaderboard (add, rank, filter, report)
  - IRSEBenchmarker (register, compare, report)
  - IRSEScoreDatabase (CRUD, leaderboard query, history, weights)
  - IRSEApi (all endpoint envelopes)
  - IRSEScoreDashboard (all 7 dashboard types)
  - IRSEScoreExporter (JSON, CSV, HTML, Markdown)
"""

import importlib.util
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: load Phase 19.2 modules directly (bypass __init__.py / torch)
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parents[2] / "trainer" / "research"


def _load(module_name: str, file_name: str):
    path = _BASE / file_name
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- loguru stub (not installed in CI Nix env) ----------------------------
if "loguru" not in sys.modules:
    _loguru = types.ModuleType("loguru")
    _logger_cls = type(
        "Logger",
        (),
        {
            "debug": staticmethod(lambda *a, **k: None),
            "info": staticmethod(lambda *a, **k: None),
            "warning": staticmethod(lambda *a, **k: None),
            "error": staticmethod(lambda *a, **k: None),
        },
    )
    _loguru.logger = _logger_cls()
    sys.modules["loguru"] = _loguru

# Load in dependency order
_weights = _load("trainer.research.weights", "weights.py")
_grades = _load("trainer.research.grades", "grades.py")
_categories = _load("trainer.research.categories", "categories.py")
_confidence = _load("trainer.research.confidence", "confidence.py")
_scorer = _load("trainer.research.scorer", "scorer.py")
_trends = _load("trainer.research.trends", "trends.py")
_lb_mod = _load("trainer.research.leaderboard_irse", "leaderboard_irse.py")
_bm_mod = _load("trainer.research.benchmarking", "benchmarking.py")
_db_mod = _load("trainer.research.score_database", "score_database.py")
_api_mod = _load("trainer.research.score_api", "score_api.py")
_dash_mod = _load("trainer.research.score_dashboard", "score_dashboard.py")
_exp_mod = _load("trainer.research.score_export", "score_export.py")

# Pull classes into local scope
ScoringWeights = _weights.ScoringWeights
GradeSystem = _grades.GradeSystem
Grade = _grades.Grade

ValidationQualityInputs = _categories.ValidationQualityInputs
ValidationQualityScorer = _categories.ValidationQualityScorer
TradingPerformanceInputs = _categories.TradingPerformanceInputs
TradingPerformanceScorer = _categories.TradingPerformanceScorer
RobustnessInputs = _categories.RobustnessInputs
RobustnessScorer = _categories.RobustnessScorer
CalibrationInputs = _categories.CalibrationInputs
CalibrationScorer = _categories.CalibrationScorer
GeneralizationInputs = _categories.GeneralizationInputs
GeneralizationScorer = _categories.GeneralizationScorer
EfficiencyInputs = _categories.EfficiencyInputs
EfficiencyScorer = _categories.EfficiencyScorer
ExplainabilityInputs = _categories.ExplainabilityInputs
ExplainabilityScorer = _categories.ExplainabilityScorer
OperationalReadinessInputs = _categories.OperationalReadinessInputs
OperationalReadinessScorer = _categories.OperationalReadinessScorer

ConfidenceInputs = _confidence.ConfidenceInputs
ConfidenceCalculator = _confidence.ConfidenceCalculator

ResearchScoreInputs = _scorer.ResearchScoreInputs
ResearchScorer = _scorer.ResearchScorer
ResearchScore = _scorer.ResearchScore

ScoreTrendTracker = _trends.ScoreTrendTracker
ScoreSnapshot = _trends.ScoreSnapshot

IRSELeaderboard = _lb_mod.IRSELeaderboard
IRSELeaderboardEntry = _lb_mod.IRSELeaderboardEntry

IRSEBenchmarker = _bm_mod.IRSEBenchmarker
ModelRole = _bm_mod.ModelRole

IRSEScoreDatabase = _db_mod.IRSEScoreDatabase
IRSEApi = _api_mod.IRSEApi
IRSEScoreDashboard = _dash_mod.IRSEScoreDashboard
IRSEScoreExporter = _exp_mod.IRSEScoreExporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_inputs(
    *,
    wf_acc: float = 0.70,
    sharpe: float = 1.80,
    drawdown: float = 0.08,
    monte_carlo: float = 0.85,
    ece: float = 0.04,
    oos: float = 0.68,
    latency_ms: float = 20.0,
    doc_coverage: float = 0.90,
    deploy_ready: float = 0.90,
    name: str = "test_exp",
    experiment_id: str = "",
) -> ResearchScoreInputs:
    import uuid as _uuid

    return ResearchScoreInputs(
        experiment_id=experiment_id or str(_uuid.uuid4()),
        name=name,
        validation=ValidationQualityInputs(
            walk_forward_accuracy=wf_acc,
            n_folds=12,
            cross_regime_stability=0.02,
            cross_instrument_stability=0.03,
            consistency_score=0.85,
            prediction_reliability=0.82,
        ),
        trading=TradingPerformanceInputs(
            profit_factor=1.65,
            expectancy=0.30,
            sharpe_ratio=sharpe,
            sortino_ratio=2.20,
            calmar_ratio=1.80,
            max_drawdown=drawdown,
            win_rate=0.57,
            average_r_multiple=1.40,
            tp_before_sl_accuracy=0.62,
            capital_efficiency=0.75,
        ),
        robustness=RobustnessInputs(
            monte_carlo_stability=monte_carlo,
            stress_test_pass_rate=0.88,
            noise_resistance=0.82,
            data_gap_tolerance=0.78,
            spread_expansion_tolerance=0.80,
            execution_delay_tolerance=0.75,
            volatility_shift_tolerance=0.77,
        ),
        calibration=CalibrationInputs(
            expected_calibration_error=ece,
            brier_score=0.16,
            confidence_reliability=0.81,
            prediction_entropy=0.35,
            probability_calibration=0.89,
        ),
        generalization=GeneralizationInputs(
            unseen_instrument_performance=oos,
            unseen_timeframe_performance=0.66,
            unseen_regime_performance=0.64,
            out_of_sample_performance=0.67,
            recent_data_performance=0.70,
            historical_stability=0.82,
        ),
        efficiency=EfficiencyInputs(
            inference_latency_ms=latency_ms,
            training_time_hours=8.0,
            gpu_utilization=0.78,
            cpu_utilization=0.60,
            ram_usage_gb=12.0,
            model_size_mb=250.0,
            storage_footprint_gb=5.0,
        ),
        explainability=ExplainabilityInputs(
            feature_attribution_coverage=0.88,
            attention_quality=0.75,
            prediction_traceability=0.80,
            decision_transparency=0.70,
            documentation_completeness=doc_coverage,
        ),
        operational=OperationalReadinessInputs(
            deployment_readiness=deploy_ready,
            monitoring_coverage=0.85,
            health_checks=0.90,
            drift_monitoring=0.80,
            rollback_readiness=0.75,
            audit_completeness=0.88,
        ),
        confidence=ConfidenceInputs(
            n_validation_folds=12,
            n_trade_samples=15000,
            regime_coverage=0.80,
            instrument_count=5,
            timeframe_count=3,
            p_value=0.015,
            fold_score_std=0.02,
            data_quality_score=0.92,
            label_quality_score=0.88,
            reproducibility_score=0.95,
        ),
        instrument="ES",
        timeframe="1h",
        backbone="transformer",
        architecture="encoder_decoder",
        dataset_version="v3.2",
        feature_version="v2.1",
        model_family="signalium_v3",
    )


def _make_full_system():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    trends = ScoreTrendTracker()
    bm = IRSEBenchmarker()
    db = IRSEScoreDatabase(":memory:")
    api = IRSEApi(scorer, lb, trends, bm, db)
    dash = IRSEScoreDashboard(lb, trends, bm, db)
    exp = IRSEScoreExporter(lb, trends, bm, db)
    return scorer, lb, trends, bm, db, api, dash, exp


# ===========================================================================
# 1. ScoringWeights
# ===========================================================================


def test_weights_default_sums_to_one():
    w = ScoringWeights.default()
    total = sum(w.as_dict().values())
    assert abs(total - 1.0) < 1e-9, f"Sum={total}"


def test_weights_validate_ok():
    w = ScoringWeights.default()
    w.validate()  # should not raise


def test_weights_validate_fails_on_bad_sum():
    w = ScoringWeights(validation_quality=0.50)  # sum > 1
    try:
        w.validate()
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_weights_from_dict():
    d = {
        "validation_quality": 0.30,
        "trading_performance": 0.30,
        "robustness": 0.10,
        "calibration": 0.10,
        "generalization": 0.10,
        "efficiency": 0.05,
        "explainability": 0.025,
        "operational_readiness": 0.025,
    }
    w = ScoringWeights.from_dict(d)
    w.validate()
    assert abs(w.validation_quality - 0.30) < 1e-9


def test_weights_normalise():
    w = ScoringWeights(
        validation_quality=2.0,
        trading_performance=2.0,
        robustness=1.0,
        calibration=1.0,
        generalization=1.0,
        efficiency=0.5,
        explainability=0.5,
        operational_readiness=0.5,
    )
    n = w.normalise()
    n.validate()


def test_weights_to_json():
    w = ScoringWeights.default()
    j = w.to_json()
    import json

    d = json.loads(j)
    assert "validation_quality" in d


# ===========================================================================
# 2. GradeSystem
# ===========================================================================


def test_grade_institutional_elite():
    gs = GradeSystem.default()
    result = gs.assign(97.5)
    assert result.label == "Institutional Elite"
    assert result.promotion_eligible
    assert result.deployment_eligible


def test_grade_institutional():
    gs = GradeSystem.default()
    result = gs.assign(91.0)
    assert result.label == "Institutional"
    assert result.promotion_eligible


def test_grade_production_ready():
    gs = GradeSystem.default()
    result = gs.assign(87.0)
    assert result.label == "Production Ready"


def test_grade_candidate():
    gs = GradeSystem.default()
    result = gs.assign(82.0)
    assert result.label == "Candidate"
    assert result.promotion_eligible


def test_grade_experimental():
    gs = GradeSystem.default()
    result = gs.assign(75.0)
    assert result.label == "Experimental"
    assert not result.promotion_eligible


def test_grade_research_only():
    gs = GradeSystem.default()
    result = gs.assign(55.0)
    assert result.label == "Research Only"
    assert not result.promotion_eligible


def test_grade_exact_boundary_95():
    gs = GradeSystem.default()
    result = gs.assign(95.0)
    assert result.label == "Institutional Elite"


def test_grade_to_dict():
    gs = GradeSystem.default()
    result = gs.assign(88.0)
    d = result.to_dict()
    assert "grade_label" in d
    assert "promotion_eligible" in d


def test_grade_invalid_score():
    gs = GradeSystem.default()
    try:
        gs.assign(101.0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_grade_thresholds():
    gs = GradeSystem.default()
    assert gs.promotion_threshold() <= 80.0
    assert gs.deployment_threshold() <= 85.0


# ===========================================================================
# 3. Category Scorers
# ===========================================================================


def test_validation_quality_high_inputs():
    scorer = ValidationQualityScorer()
    inputs = ValidationQualityInputs(
        walk_forward_accuracy=0.73,
        walk_forward_f1_macro=0.71,
        n_folds=15,
        cross_regime_stability=0.01,
        cross_instrument_stability=0.01,
        cross_timeframe_stability=0.01,
        consistency_score=0.95,
        prediction_reliability=0.90,
    )
    score, sub = scorer.score(inputs)
    assert 0.0 <= score <= 100.0
    assert score >= 70.0, f"Expected high score, got {score}"
    assert "walk_forward_accuracy" in sub


def test_validation_quality_zero_inputs():
    scorer = ValidationQualityScorer()
    score, _ = scorer.score(ValidationQualityInputs())
    assert 0.0 <= score <= 100.0


def test_trading_performance_strong():
    scorer = TradingPerformanceScorer()
    inputs = TradingPerformanceInputs(
        profit_factor=2.0,
        sharpe_ratio=2.2,
        max_drawdown=0.05,
        win_rate=0.60,
        average_r_multiple=1.8,
        expectancy=0.45,
        sortino_ratio=2.8,
        calmar_ratio=2.5,
        tp_before_sl_accuracy=0.68,
        capital_efficiency=0.85,
    )
    score, sub = scorer.score(inputs)
    assert 0 <= score <= 100
    assert score >= 60.0, f"Expected strong trading score, got {score}"
    assert "sharpe_ratio" in sub


def test_trading_performance_zeros():
    scorer = TradingPerformanceScorer()
    score, _ = scorer.score(TradingPerformanceInputs())
    assert 0.0 <= score <= 100.0


def test_robustness_perfect():
    scorer = RobustnessScorer()
    inputs = RobustnessInputs(
        monte_carlo_stability=1.0,
        stress_test_pass_rate=1.0,
        noise_resistance=1.0,
        data_gap_tolerance=1.0,
        spread_expansion_tolerance=1.0,
        execution_delay_tolerance=1.0,
        volatility_shift_tolerance=1.0,
    )
    score, _ = scorer.score(inputs)
    assert abs(score - 100.0) < 1e-6


def test_robustness_zeros():
    scorer = RobustnessScorer()
    score, _ = scorer.score(RobustnessInputs())
    assert score == 0.0


def test_calibration_low_ece():
    scorer = CalibrationScorer()
    inputs = CalibrationInputs(
        expected_calibration_error=0.01,
        brier_score=0.12,
        confidence_reliability=0.90,
        prediction_entropy=0.20,
        probability_calibration=0.95,
    )
    score, sub = scorer.score(inputs)
    assert score >= 70.0, f"Expected well-calibrated score, got {score}"
    assert "expected_calibration_error" in sub


def test_generalization_oos():
    scorer = GeneralizationScorer()
    inputs = GeneralizationInputs(
        unseen_instrument_performance=0.70,
        unseen_timeframe_performance=0.68,
        unseen_regime_performance=0.66,
        out_of_sample_performance=0.72,
        recent_data_performance=0.69,
        historical_stability=0.85,
    )
    score, _ = scorer.score(inputs)
    assert score >= 50.0


def test_efficiency_fast():
    scorer = EfficiencyScorer()
    inputs = EfficiencyInputs(
        inference_latency_ms=8.0,
        training_time_hours=4.0,
        gpu_utilization=0.85,
        cpu_utilization=0.70,
        ram_usage_gb=8.0,
        model_size_mb=100.0,
        storage_footprint_gb=2.0,
    )
    score, _ = scorer.score(inputs)
    assert score >= 60.0


def test_explainability_full():
    scorer = ExplainabilityScorer()
    inputs = ExplainabilityInputs(
        feature_attribution_coverage=1.0,
        attention_quality=1.0,
        prediction_traceability=1.0,
        decision_transparency=1.0,
        documentation_completeness=1.0,
    )
    score, _ = scorer.score(inputs)
    assert abs(score - 100.0) < 1e-6


def test_operational_readiness_full():
    scorer = OperationalReadinessScorer()
    inputs = OperationalReadinessInputs(
        deployment_readiness=1.0,
        monitoring_coverage=1.0,
        health_checks=1.0,
        drift_monitoring=1.0,
        rollback_readiness=1.0,
        audit_completeness=1.0,
    )
    score, _ = scorer.score(inputs)
    assert abs(score - 100.0) < 1e-6


# ===========================================================================
# 4. Confidence
# ===========================================================================


def test_confidence_high():
    calc = ConfidenceCalculator()
    inputs = ConfidenceInputs(
        n_validation_folds=12,
        n_trade_samples=20000,
        regime_coverage=0.90,
        instrument_count=5,
        timeframe_count=3,
        p_value=0.005,
        fold_score_std=0.01,
        data_quality_score=0.95,
        label_quality_score=0.92,
        reproducibility_score=0.98,
    )
    result = calc.compute(inputs)
    assert 0.0 <= result.score <= 1.0
    assert result.score >= 0.70, f"Expected high confidence, got {result.score}"
    assert result.level in ("High", "Very High")


def test_confidence_low():
    calc = ConfidenceCalculator()
    inputs = ConfidenceInputs(n_validation_folds=2, n_trade_samples=500, p_value=0.5)
    result = calc.compute(inputs)
    assert result.score < 0.5
    assert result.level in ("Very Low", "Low")


def test_confidence_to_dict():
    calc = ConfidenceCalculator()
    result = calc.compute(ConfidenceInputs())
    d = result.to_dict()
    assert "score" in d
    assert "level" in d
    assert "sub_scores" in d


# ===========================================================================
# 5. ResearchScorer
# ===========================================================================


def test_scorer_basic():
    scorer = ResearchScorer()
    inputs = _make_inputs()
    rs = scorer.score(inputs)
    assert 0.0 <= rs.composite <= 100.0
    assert rs.grade_label != ""
    assert len(rs.category_scores) == 8


def test_scorer_returns_correct_category_count():
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs())
    cats = {cs.category for cs in rs.category_scores}
    expected = {
        "validation_quality",
        "trading_performance",
        "robustness",
        "calibration",
        "generalization",
        "efficiency",
        "explainability",
        "operational_readiness",
    }
    assert cats == expected


def test_scorer_weighted_sum_correct():
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs())
    reconstructed = sum(cs.weighted_score for cs in rs.category_scores)
    assert (
        abs(reconstructed - rs.composite) < 1e-4
    ), f"Weighted sum {reconstructed} ≠ composite {rs.composite}"


def test_scorer_breakdown():
    scorer = ResearchScorer()
    inputs = _make_inputs()
    bd = scorer.score_breakdown(inputs)
    assert "research_score" in bd
    assert "weights" in bd


def test_scorer_to_dict_roundtrip():
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs())
    d = rs.to_dict()
    rs2 = ResearchScore.from_dict(d)
    # to_dict() rounds composite to 2 dp — allow 0.01 tolerance
    assert abs(rs2.composite - rs.composite) < 0.01


def test_scorer_summary():
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs(name="summary_test"))
    s = rs.summary()
    assert "Research Score" in s
    assert "summary_test" in s


def test_scorer_sensitivity_analysis():
    scorer = ResearchScorer()
    inputs = _make_inputs()
    sensitivity = scorer.sensitivity_analysis(inputs, delta=0.05)
    assert len(sensitivity) == 8
    for v in sensitivity.values():
        assert isinstance(v, float)


def test_scorer_update_weights():
    scorer = ResearchScorer()
    new_w = ScoringWeights.from_dict(
        {
            "validation_quality": 0.40,
            "trading_performance": 0.30,
            "robustness": 0.10,
            "calibration": 0.05,
            "generalization": 0.05,
            "efficiency": 0.04,
            "explainability": 0.03,
            "operational_readiness": 0.03,
        }
    )
    scorer.update_weights(new_w)
    rs = scorer.score(_make_inputs())
    assert 0.0 <= rs.composite <= 100.0


def test_scorer_zero_inputs():
    scorer = ResearchScorer()
    rs = scorer.score(ResearchScoreInputs())
    assert 0.0 <= rs.composite <= 100.0


def test_scorer_confidence_present():
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs())
    assert rs.confidence is not None
    assert 0.0 <= rs.confidence_score() <= 1.0


# ===========================================================================
# 6. ScoreTrendTracker
# ===========================================================================


def test_trend_record_and_history():
    tracker = ScoreTrendTracker()
    tracker.record("exp_001", 75.0, name="v1")
    tracker.record("exp_001", 78.5, name="v1_b")
    tracker.record("exp_002", 82.0, name="v2")
    history = tracker.score_history()
    assert len(history) == 3


def test_trend_record_from_research_score():
    scorer = ResearchScorer()
    tracker = ScoreTrendTracker()
    rs = scorer.score(_make_inputs())
    snap = tracker.record_from_research_score(rs)
    assert snap.composite == rs.composite


def test_trend_improvement_trend_improving():
    tracker = ScoreTrendTracker()
    for i, score in enumerate([60.0, 65.0, 70.0, 75.0, 80.0]):
        tracker.record(f"exp_{i:03d}", score)
    trend = tracker.improvement_trend()
    assert trend["trend"] == "improving"
    assert trend["slope"] > 0


def test_trend_improvement_trend_declining():
    tracker = ScoreTrendTracker()
    for i, score in enumerate([80.0, 75.0, 70.0, 65.0, 60.0]):
        tracker.record(f"exp_{i:03d}", score)
    trend = tracker.improvement_trend()
    assert trend["trend"] == "declining"


def test_trend_regression_detection():
    tracker = ScoreTrendTracker()
    tracker.record("exp_001", 85.0, name="v1")
    tracker.record("exp_001", 84.0, name="v1b")
    tracker.record("exp_001", 72.0, name="v1c")  # big drop
    alerts = tracker.detect_regressions(window=5, threshold=5.0)
    assert len(alerts) == 1
    assert alerts[0].experiment_id == "exp_001"
    assert alerts[0].drop >= 10.0


def test_trend_no_regression_small_drop():
    tracker = ScoreTrendTracker()
    tracker.record("exp_001", 85.0)
    tracker.record("exp_001", 83.0)  # < 5 drop
    alerts = tracker.detect_regressions(threshold=5.0)
    assert len(alerts) == 0


def test_trend_champion_timeline():
    tracker = ScoreTrendTracker()
    tracker.record_champion_change("exp_002", "exp_001", 82.0, 78.0, "better sharpe")
    tracker.record_champion_change("exp_003", "exp_002", 85.0, 82.0, "better OOS")
    tl = tracker.champion_timeline()
    assert len(tl) == 2
    assert tl[0]["new_champion_id"] == "exp_002"


def test_trend_distribution():
    tracker = ScoreTrendTracker()
    for i, s in enumerate([60, 70, 75, 80, 85, 90, 95]):
        tracker.record(f"e{i}", float(s))
    dist = tracker.score_distribution()
    assert dist["n"] == 7
    assert dist["min"] == 60.0
    assert dist["max"] == 95.0
    assert 70.0 <= dist["mean"] <= 80.0


def test_trend_report_structure():
    tracker = ScoreTrendTracker()
    tracker.record("exp_001", 78.0)
    report = tracker.trend_report()
    assert "total_scored_experiments" in report
    assert "improvement_trend" in report
    assert "score_distribution" in report


# ===========================================================================
# 7. IRSELeaderboard
# ===========================================================================


def test_leaderboard_add_and_rank():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    inputs1 = _make_inputs(name="model_a", wf_acc=0.65)
    inputs2 = _make_inputs(name="model_b", wf_acc=0.73, sharpe=2.1)
    rs1 = scorer.score(inputs1)
    rs2 = scorer.score(inputs2)
    lb.add(rs1)
    lb.add(rs2)
    top = lb.top(10)
    assert len(top) == 2
    assert top[0]["rank"] == 1


def test_leaderboard_deduplication():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    inputs = _make_inputs(name="model_a")
    rs1 = scorer.score(inputs)
    # Same experiment_id — add twice
    lb.add(rs1)
    lb.add(rs1)
    assert len(lb.all_entries()) == 1


def test_leaderboard_rank_by_trading():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    rs1 = scorer.score(_make_inputs(name="low_sharpe", sharpe=0.8))
    rs2 = scorer.score(_make_inputs(name="high_sharpe", sharpe=2.5))
    lb.add(rs1)
    lb.add(rs2)
    ranked = lb.rank_by("trading_score")
    assert ranked[0]["name"] == "high_sharpe"


def test_leaderboard_filter_instrument():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    inp1 = _make_inputs(name="es_model")
    inp1.instrument = "ES"
    inp2 = _make_inputs(name="nq_model")
    inp2.instrument = "NQ"
    lb.add(scorer.score(inp1))
    lb.add(scorer.score(inp2))
    filtered = lb.filter(instrument="ES")
    assert len(filtered.all_entries()) == 1
    assert filtered.all_entries()[0].instrument == "ES"


def test_leaderboard_promotion_candidates():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    # Use fully-populated fixture so composite crosses the 80-pt threshold
    high = _make_inputs(name="high", wf_acc=0.74, sharpe=2.3, drawdown=0.04)
    low = _make_inputs(
        name="low",
        wf_acc=0.52,
        sharpe=0.6,
        drawdown=0.20,
        monte_carlo=0.0,
        ece=0.20,
        oos=0.50,
        latency_ms=95.0,
        doc_coverage=0.0,
        deploy_ready=0.0,
    )
    rs_high = scorer.score(high)
    rs_low = scorer.score(low)
    lb.add(rs_high)
    lb.add(rs_low)
    candidates = lb.promotion_candidates()
    # High scorer (composite > low scorer) should appear if eligible
    # Verify that promotion_candidates returns only eligible entries
    for c in candidates:
        assert c["promotion_eligible"] is True
    # And the low scorer is genuinely lower
    assert rs_high.composite > rs_low.composite


def test_leaderboard_grade_distribution():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    lb.add(scorer.score(_make_inputs(name="a")))
    dist = lb.grade_distribution()
    assert isinstance(dist, dict)
    assert sum(dist.values()) == len(lb.all_entries())


def test_leaderboard_report_structure():
    scorer = ResearchScorer()
    lb = IRSELeaderboard()
    lb.add(scorer.score(_make_inputs(name="x")))
    report = lb.leaderboard_report()
    assert "total_entries" in report
    assert "top_entries" in report
    assert "grade_distribution" in report


# ===========================================================================
# 8. IRSEBenchmarker
# ===========================================================================


def test_benchmarker_register_and_compare():
    scorer = ResearchScorer()
    bm = IRSEBenchmarker()
    rs1 = scorer.score(_make_inputs(name="baseline", wf_acc=0.62))
    rs2 = scorer.score(_make_inputs(name="candidate", wf_acc=0.72))
    bm.register("exp_001", rs1, ModelRole.BASELINE)
    bm.register("exp_002", rs2, ModelRole.CANDIDATE)
    bm.set_champion("exp_001")
    comp = bm.compare("exp_002", against="exp_001")
    assert comp.subject_id == "exp_002"
    assert comp.reference_id == "exp_001"
    assert isinstance(comp.composite_delta, float)


def test_benchmarker_compare_improved():
    scorer = ResearchScorer()
    bm = IRSEBenchmarker()
    rs_low = scorer.score(_make_inputs(name="low", wf_acc=0.55, sharpe=0.8))
    rs_high = scorer.score(_make_inputs(name="high", wf_acc=0.74, sharpe=2.2))
    bm.register("low", rs_low, ModelRole.BASELINE)
    bm.register("high", rs_high, ModelRole.CANDIDATE)
    bm.set_champion("low")
    comp = bm.compare("high", against="low")
    assert comp.composite_improved


def test_benchmarker_stronger_categories():
    scorer = ResearchScorer()
    bm = IRSEBenchmarker()
    rs1 = scorer.score(_make_inputs(name="ref"))
    rs2 = scorer.score(_make_inputs(name="cand", deploy_ready=1.0, doc_coverage=1.0))
    bm.register("ref", rs1, ModelRole.CHAMPION)
    bm.register("cand", rs2)
    bm.set_champion("ref")
    comp = bm.compare("cand")
    assert isinstance(comp.stronger_categories, list)
    assert isinstance(comp.weaker_categories, list)


def test_benchmarker_benchmark_report():
    scorer = ResearchScorer()
    bm = IRSEBenchmarker()
    rs = scorer.score(_make_inputs(name="x"))
    bm.register("x", rs, ModelRole.CHAMPION)
    bm.set_champion("x")
    report = bm.benchmark_report("x")
    assert "subject" in report
    assert "champion_id" in report


def test_benchmarker_full_report():
    scorer = ResearchScorer()
    bm = IRSEBenchmarker()
    for i in range(3):
        rs = scorer.score(_make_inputs(name=f"model_{i}"))
        bm.register(f"exp_{i:03d}", rs, ModelRole.EXPERIMENTAL)
    report = bm.full_benchmark_report()
    assert report["total_registered"] == 3
    assert "ranking" in report


# ===========================================================================
# 9. IRSEScoreDatabase
# ===========================================================================


def test_db_save_and_retrieve():
    db = IRSEScoreDatabase(":memory:")
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs(name="db_test"))
    rs_dict = rs.to_dict()
    db.save_research_score(rs_dict)
    retrieved = db.get_research_score(rs.score_id)
    assert retrieved is not None
    # to_dict() rounds composite to 2 dp before storage — allow 0.01 tolerance
    assert abs(retrieved["composite"] - rs.composite) < 0.01


def test_db_latest_for_experiment():
    db = IRSEScoreDatabase(":memory:")
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs(name="exp_a"))
    rs_dict = rs.to_dict()
    db.save_research_score(rs_dict)
    latest = db.get_latest_for_experiment(rs.experiment_id)
    assert latest is not None
    assert latest["experiment_id"] == rs.experiment_id


def test_db_category_scores_stored():
    db = IRSEScoreDatabase(":memory:")
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs(name="cat_test"))
    db.save_research_score(rs.to_dict())
    retrieved = db.get_research_score(rs.score_id)
    assert len(retrieved["category_scores"]) == 8


def test_db_leaderboard_query():
    db = IRSEScoreDatabase(":memory:")
    scorer = ResearchScorer()
    for i in range(5):
        rs = scorer.score(_make_inputs(name=f"lb_{i}"))
        db.save_research_score(rs.to_dict())
    results = db.query_leaderboard(limit=3, order_by="composite")
    assert len(results) <= 3


def test_db_score_history():
    db = IRSEScoreDatabase(":memory:")
    db.save_score_snapshot(
        {
            "snapshot_id": "snap_001",
            "experiment_id": "exp_001",
            "composite": 78.5,
            "grade_label": "Experimental",
            "recorded_at": "2026-01-01T00:00:00+00:00",
        }
    )
    history = db.get_score_history()
    assert len(history) == 1
    assert history[0]["composite"] == 78.5


def test_db_weight_configuration():
    db = IRSEScoreDatabase(":memory:")
    config_id = db.save_weight_configuration(
        "custom_v1",
        {
            "validation_quality": 0.40,
            "trading_performance": 0.30,
            "robustness": 0.10,
            "calibration": 0.05,
            "generalization": 0.05,
            "efficiency": 0.04,
            "explainability": 0.03,
            "operational_readiness": 0.03,
        },
        description="Heavy validation",
        set_active=True,
    )
    assert config_id
    active = db.get_active_weights()
    assert active is not None
    assert abs(active["validation_quality"] - 0.40) < 1e-9


def test_db_stats():
    db = IRSEScoreDatabase(":memory:")
    scorer = ResearchScorer()
    rs = scorer.score(_make_inputs(name="stats_test"))
    db.save_research_score(rs.to_dict())
    stats = db.stats()
    assert stats["research_scores"] == 1
    assert stats["category_scores"] == 8


# ===========================================================================
# 10. IRSEApi
# ===========================================================================


def test_api_score_experiment():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    inputs = _make_inputs(name="api_test")
    resp = api.score_experiment(inputs)
    assert resp["status"] == "ok"
    assert "composite" in resp["data"]
    assert "score_id" in resp["data"]


def test_api_get_score():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    inputs = _make_inputs(name="api_get_test")
    score_resp = api.score_experiment(inputs)
    score_id = score_resp["data"]["score_id"]
    get_resp = api.get_score(score_id)
    assert get_resp["status"] == "ok"
    assert get_resp["data"]["score_id"] == score_id


def test_api_get_missing_score():
    _, _, _, _, _, api, _, _ = _make_full_system()
    resp = api.get_score("nonexistent-id")
    assert resp["status"] == "error"


def test_api_get_leaderboard():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="a"))
    api.score_experiment(_make_inputs(name="b"))
    resp = api.get_leaderboard()
    assert resp["status"] == "ok"
    assert "top_entries" in resp["data"]


def test_api_leaderboard_filter():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    inp = _make_inputs(name="filtered_model")
    inp.instrument = "NQ"
    api.score_experiment(inp)
    resp = api.get_leaderboard(instrument="NQ")
    assert resp["status"] == "ok"


def test_api_trend_report():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="trend_test"))
    resp = api.trend_report()
    assert resp["status"] == "ok"
    assert "total_scored_experiments" in resp["data"]


def test_api_get_weights():
    _, _, _, _, _, api, _, _ = _make_full_system()
    resp = api.get_weights()
    assert resp["status"] == "ok"
    assert "validation_quality" in resp["data"]


def test_api_set_weights():
    _, _, _, _, _, api, _, _ = _make_full_system()
    resp = api.set_weights(
        {
            "validation_quality": 0.35,
            "trading_performance": 0.30,
            "robustness": 0.10,
            "calibration": 0.07,
            "generalization": 0.08,
            "efficiency": 0.04,
            "explainability": 0.03,
            "operational_readiness": 0.03,
        }
    )
    assert resp["status"] == "ok"
    assert resp["data"]["applied"]


def test_api_system_stats():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="stats"))
    resp = api.system_stats()
    assert resp["status"] == "ok"
    assert "database" in resp["data"]


def test_api_export_leaderboard_json():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="export_test"))
    resp = api.export_leaderboard(fmt="json")
    assert resp["status"] == "ok"


def test_api_export_leaderboard_csv():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="export_csv"))
    resp = api.export_leaderboard(fmt="csv")
    assert resp["status"] == "ok"


# ===========================================================================
# 11. IRSEScoreDashboard
# ===========================================================================


def test_dashboard_timeline_empty():
    _, lb, trends, bm, db, _, dash, _ = _make_full_system()
    d = dash.research_score_timeline()
    assert d["type"] == "research_score_timeline"
    assert "series" in d


def test_dashboard_timeline_with_data():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="tl_test"))
    d = dash.research_score_timeline()
    assert d["type"] == "research_score_timeline"


def test_dashboard_radar_chart():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    inputs = _make_inputs(name="radar_model")
    api.score_experiment(inputs)
    entries = lb.all_entries()
    exp_id = entries[0].experiment_id
    d = dash.radar_chart(exp_id)
    assert d["type"] == "radar_chart"
    assert len(d["axes"]) == 8


def test_dashboard_radar_missing_experiment():
    _, lb, trends, bm, db, _, dash, _ = _make_full_system()
    d = dash.radar_chart("missing_id")
    assert "error" in d


def test_dashboard_category_breakdown():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    for i in range(3):
        api.score_experiment(_make_inputs(name=f"cb_{i}"))
    d = dash.category_breakdown(top_n=3)
    assert d["type"] == "category_breakdown"
    assert len(d["series"]) == 8


def test_dashboard_leaderboard():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="lb_dash"))
    d = dash.leaderboard_dashboard(top_n=5)
    assert d["type"] == "leaderboard"
    assert "total_entries" in d


def test_dashboard_score_heatmap():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="heat_a"))
    api.score_experiment(_make_inputs(name="heat_b"))
    d = dash.score_heatmap(top_n=2)
    assert d["type"] == "score_heatmap"
    assert len(d["category_labels"]) == 8


def test_dashboard_historical_evolution():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="hist_test"))
    d = dash.historical_evolution()
    assert d["type"] == "historical_evolution"
    assert "all_series" in d


def test_dashboard_build_all():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="all_test"))
    entries = lb.all_entries()
    exp_id = entries[0].experiment_id
    result = dash.build_all(top_n=5, sample_experiment_id=exp_id)
    assert "dashboards" in result
    assert result["count"] == len(result["dashboards"])


# ===========================================================================
# 12. IRSEScoreExporter
# ===========================================================================


def test_exporter_json_leaderboard():
    scorer, lb, trends, bm, db, api, dash, exp = _make_full_system()
    api.score_experiment(_make_inputs(name="json_exp"))
    data = exp.export_json()
    import json as _json

    parsed = _json.loads(data)
    assert "leaderboard" in parsed


def test_exporter_json_single_experiment():
    scorer, lb, trends, bm, db, api, dash, exporter = _make_full_system()
    inputs = _make_inputs(name="single_exp")
    api.score_experiment(inputs)
    data = exporter.export_json(experiment_id=inputs.experiment_id)
    import json as _json

    parsed = _json.loads(data)
    assert isinstance(parsed, (dict, list))


def test_exporter_csv():
    scorer, lb, trends, bm, db, api, dash, exporter = _make_full_system()
    api.score_experiment(_make_inputs(name="csv_exp"))
    csv_data = exporter.export_csv(top_n=10)
    assert "composite" in csv_data
    assert "csv_exp" in csv_data


def test_exporter_csv_empty():
    _, lb, trends, bm, db, _, _, exporter = _make_full_system()
    csv_data = exporter.export_csv()
    assert csv_data == ""


def test_exporter_html():
    scorer, lb, trends, bm, db, api, dash, exporter = _make_full_system()
    api.score_experiment(_make_inputs(name="html_exp"))
    html = exporter.export_html(top_n=5)
    assert "<!DOCTYPE html>" in html
    assert "IRSE" in html
    assert "html_exp" in html


def test_exporter_research_report_markdown():
    scorer, lb, trends, bm, db, api, dash, exporter = _make_full_system()
    api.score_experiment(_make_inputs(name="report_exp"))
    report = exporter.export_research_report(as_markdown=True)
    assert "# Institutional Research Scoring Engine" in report
    assert "Top 10" in report


def test_exporter_research_report_json():
    scorer, lb, trends, bm, db, api, dash, exporter = _make_full_system()
    api.score_experiment(_make_inputs(name="report_json"))
    import json as _json

    report = exporter.export_research_report(as_markdown=False)
    parsed = _json.loads(report)
    assert "leaderboard" in parsed or "top_10" in parsed


def test_exporter_score_card():
    scorer, lb, trends, bm, db, api, dash, exporter = _make_full_system()
    inputs = _make_inputs(name="card_exp")
    api.score_experiment(inputs)
    card = exporter.experiment_score_card(inputs.experiment_id)
    assert "Research Score Card" in card
    assert "card_exp" in card


def test_exporter_score_card_missing():
    _, lb, trends, bm, db, _, _, exporter = _make_full_system()
    card = exporter.experiment_score_card("nonexistent")
    assert "No score found" in card


# ===========================================================================
# Main runner (no pytest dependency)
# ===========================================================================

if __name__ == "__main__":
    import traceback as _tb

    tests = [
        (name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            _tb.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed / {len(tests)} total")
    if failed:
        sys.exit(1)
