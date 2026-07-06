"""Phase 15 — Institutional Validation Platform Unit Tests.

Tests every module of the trainer/validation/ package.

Coverage targets:
  - Windows validators (SlidingWindow, Expanding, Anchored, Nested)
  - Leakage detector
  - Regime classifier
  - Instrument / timeframe registry
  - Statistical metrics engine (17 metrics)
  - Trading metrics engine (21 metrics) + TradeSimulator
  - Monte Carlo validator
  - Robustness tester
  - Calibration validator
  - Drift analyzer
  - Head-to-head model comparison
  - Institutional leaderboard
  - Promotion policy
  - Failure analyzer
  - Validation database (SQLite)
  - Visualization data builder
  - Report exporter
  - Validation API
  - InstitutionalValidationEngine (integration)
"""

from __future__ import annotations

import json
import math
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Imports under test ────────────────────────────────────────────────────

from trainer.validation.windows import (
    SlidingWindowValidator,
    ExpandingWindowValidator,
    AnchoredWalkForward,
    NestedValidator,
    WindowFold,
)
from trainer.validation.leakage import LeakageDetector, LeakageReport, LeakageDetected
from trainer.validation.regime import MarketRegime, MarketRegimeClassifier, RegimeDistribution
from trainer.validation.instrument import (
    INSTRUMENTS,
    TIMEFRAMES,
    register_instrument,
    register_timeframe,
    get_all_instruments,
    get_all_timeframes,
    InstrumentResult,
    MultiAssetValidationResult,
    MultiAssetValidator,
)
from trainer.validation.statistical import StatisticalMetrics, StatisticalMetricsEngine
from trainer.validation.trading import (
    TradeResult,
    TradingMetrics,
    TradingMetricsEngine,
    TradeSimulator,
)
from trainer.validation.monte_carlo import MonteCarloValidator, MonteCarloResult
from trainer.validation.robustness import RobustnessTester, RobustnessReport
from trainer.validation.calibration_validator import CalibrationValidator, CalibrationReport
from trainer.validation.drift import DriftAnalyzer, DriftReport
from trainer.validation.comparison import (
    ModelSnapshot,
    ModelComparison,
    ComparisonReport,
)
from trainer.validation.leaderboard import LeaderboardEntry, InstitutionalLeaderboard
from trainer.validation.promotion import PromotionRule, PromotionPolicy, PromotionDecision
from trainer.validation.failure_analysis import FailureAnalyzer, FailureReport, WorstPerformer
from trainer.validation.database import ValidationDatabase, ValidationRunRecord
from trainer.validation.head_validator import PredictionHeadValidator, HeadResult
from trainer.validation.visualization import VisualizationDataBuilder
from trainer.validation.export import ReportExporter, ExportConfig
from trainer.validation.api import ValidationAPI, _ok, _error
from trainer.validation.engine import (
    InstitutionalValidationEngine,
    EngineConfig,
    ValidationResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def small_xy():
    """Small (N=600, seq=10, feat=8) dataset."""
    torch.manual_seed(0)
    x = torch.randn(600, 10, 8)
    y = torch.randint(0, 3, (600,))
    return x, y


@pytest.fixture
def probs_3class():
    """(200, 3) softmax probabilities with labels."""
    torch.manual_seed(1)
    logits = torch.randn(200, 3)
    probs = torch.softmax(logits, dim=-1)
    labels = torch.randint(0, 3, (200,))
    return probs, labels


@pytest.fixture
def simple_model():
    """Minimal 3-class model for testing."""
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(80, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
    )


@pytest.fixture
def trade_list():
    """Sample list of TradeResult objects."""
    rng = np.random.default_rng(42)
    trades = []
    for _ in range(50):
        pnl = float(rng.normal(5.0, 20.0))
        tp = pnl > 0
        sl = not tp
        trades.append(
            TradeResult(
                direction=int(rng.choice([1, -1])),
                entry_price=1.1000,
                exit_price=1.1050 if pnl > 0 else 1.0980,
                stop_loss=1.0980,
                take_profit=1.1050,
                holding_bars=int(rng.integers(1, 50)),
                pnl_pips=pnl,
                mfe_pips=abs(pnl) + 2,
                mae_pips=abs(pnl) * 0.3,
                tp_hit=tp,
                sl_hit=sl,
                confidence=float(rng.uniform(0.55, 0.90)),
            )
        )
    return trades


@pytest.fixture
def tmp_db(tmp_path):
    db = ValidationDatabase(tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture
def tmp_lb(tmp_path):
    return InstitutionalLeaderboard(store_path=tmp_path / "lb.json")


# ══════════════════════════════════════════════════════════════════════════
# 1 — Windows validators
# ══════════════════════════════════════════════════════════════════════════


class TestSlidingWindowValidator:
    def test_generates_correct_folds(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=100, step_size=100)
        folds = list(v.split(x, y))
        assert len(folds) >= 3

    def test_fold_sizes(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=100, step_size=100)
        for fold in v.split(x, y):
            assert fold.train_size == 200
            assert fold.val_size == 100

    def test_chronological_ordering(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=100, step_size=100)
        folds = list(v.split(x, y))
        for i in range(1, len(folds)):
            assert folds[i].train_start > folds[i - 1].train_start

    def test_strategy_label(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=100, step_size=100)
        fold = next(v.split(x, y))
        assert fold.strategy == "sliding_window"

    def test_gap_size(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=50, step_size=100, gap_size=10)
        fold = next(v.split(x, y))
        assert fold.val_start == fold.train_end + 10

    def test_with_test_window(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=50, test_size=50, step_size=100)
        fold = next(v.split(x, y))
        assert fold.has_test
        assert fold.test_size == 50

    def test_max_folds(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=100, step_size=50, max_folds=3)
        folds = list(v.split(x, y))
        assert len(folds) == 3

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            SlidingWindowValidator(train_size=0, val_size=100, step_size=10)

    def test_num_folds(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=100, step_size=100)
        n = v.num_folds(600)
        assert n >= 3


class TestExpandingWindowValidator:
    def test_train_grows(self, small_xy):
        x, y = small_xy
        v = ExpandingWindowValidator(min_train_size=200, val_size=100, step_size=100)
        folds = list(v.split(x, y))
        train_sizes = [f.train_size for f in folds]
        assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1))

    def test_strategy_label(self, small_xy):
        x, y = small_xy
        v = ExpandingWindowValidator(min_train_size=200, val_size=100, step_size=100)
        fold = next(v.split(x, y))
        assert fold.strategy == "expanding_window"


class TestAnchoredWalkForward:
    def test_anchor_start(self, small_xy):
        x, y = small_xy
        v = AnchoredWalkForward(initial_train_size=200, val_size=100, step_size=100)
        for fold in v.split(x, y):
            assert fold.train_start == 0

    def test_strategy_label(self, small_xy):
        x, y = small_xy
        v = AnchoredWalkForward(initial_train_size=200, val_size=100, step_size=100)
        fold = next(v.split(x, y))
        assert fold.strategy == "anchored_walk_forward"


class TestNestedValidator:
    def test_generates_nested_folds(self, small_xy):
        x, y = small_xy
        v = NestedValidator(
            outer_min_train=300,
            outer_val_size=100,
            outer_step=100,
            inner_train_size=150,
            inner_val_size=50,
            inner_step=50,
        )
        folds = list(v.split(x, y))
        assert len(folds) > 0
        for nf in folds:
            assert len(nf.inner_folds) > 0


# ══════════════════════════════════════════════════════════════════════════
# 2 — Leakage Detector
# ══════════════════════════════════════════════════════════════════════════


class TestLeakageDetector:
    def test_clean_data_passes(self, small_xy):
        x, y = small_xy
        det = LeakageDetector(abort_on_detect=False)
        report = det.check(x[:400], y[:400], x[450:550], y[450:550])
        assert isinstance(report, LeakageReport)
        assert not report.is_critical

    def test_nan_detected(self, small_xy):
        x, y = small_xy
        x_bad = x.clone()
        x_bad[0, 0, 0] = float("nan")
        det = LeakageDetector(abort_on_detect=False)
        report = det.check(x_bad[:400], y[:400], x[450:550], y[450:550])
        assert report.nan_inf_features > 0

    def test_abort_raises(self, small_xy):
        """When abort_on_detect=True and leakage is found, LeakageDetected is raised."""
        x, y = small_xy
        # Create trivially correlated features (target leakage)
        x_leaky = x.clone()
        y_float = y.float().unsqueeze(1).expand(-1, x.shape[1]).unsqueeze(2)
        x_leaky[:, :, 0:1] = y_float  # perfect correlation
        det = LeakageDetector(
            abort_on_detect=True,
            target_correlation_threshold=0.50,
        )
        with pytest.raises((LeakageDetected, Exception)):
            det.check(x_leaky[:400], y[:400], x_leaky[450:550], y[450:550])

    def test_report_has_summary(self, small_xy):
        x, y = small_xy
        det = LeakageDetector(abort_on_detect=False)
        report = det.check(x[:400], y[:400], x[450:550], y[450:550])
        summary = report.summary()
        assert "===" in summary

    def test_to_dict(self, small_xy):
        x, y = small_xy
        det = LeakageDetector(abort_on_detect=False)
        report = det.check(x[:400], y[:400], x[450:550], y[450:550])
        d = report.to_dict()
        assert "future_leakage" in d
        assert "nan_inf_features" in d


# ══════════════════════════════════════════════════════════════════════════
# 3 — Market Regime Classifier
# ══════════════════════════════════════════════════════════════════════════


class TestMarketRegimeClassifier:
    @pytest.fixture
    def synthetic_ohlcv(self):
        np.random.seed(42)
        n = 100
        close = 1.1 + np.cumsum(np.random.normal(0, 0.001, n))
        high = close + np.abs(np.random.normal(0, 0.002, n))
        low = close - np.abs(np.random.normal(0, 0.002, n))
        volume = np.random.uniform(1000, 5000, n)
        ohlcv = np.column_stack([close, high, low, close, volume])
        return torch.tensor(ohlcv, dtype=torch.float32)

    def test_produces_labels(self, synthetic_ohlcv):
        clf = MarketRegimeClassifier()
        labels = clf.classify(synthetic_ohlcv)
        assert len(labels) == len(synthetic_ohlcv)

    def test_all_labels_valid(self, synthetic_ohlcv):
        clf = MarketRegimeClassifier()
        labels = clf.classify(synthetic_ohlcv)
        valid = set(r.value for r in MarketRegime)
        assert all(lbl.value in valid for lbl in labels)

    def test_regime_distribution(self, synthetic_ohlcv):
        clf = MarketRegimeClassifier()
        labels = clf.classify(synthetic_ohlcv)
        dist = RegimeDistribution.from_labels(labels)
        assert dist.total == len(synthetic_ohlcv)
        assert sum(dist.counts.values()) == dist.total

    def test_distribution_summary(self, synthetic_ohlcv):
        clf = MarketRegimeClassifier()
        labels = clf.classify(synthetic_ohlcv)
        dist = RegimeDistribution.from_labels(labels)
        summary = dist.summary()
        assert "bars" in summary

    def test_short_input(self):
        ohlcv = torch.randn(1, 5)
        clf = MarketRegimeClassifier()
        labels = clf.classify(ohlcv)
        assert len(labels) == 1


# ══════════════════════════════════════════════════════════════════════════
# 4 — Instrument Registry
# ══════════════════════════════════════════════════════════════════════════


class TestInstrumentRegistry:
    def test_canonical_instruments(self):
        assert len(INSTRUMENTS) == 14
        for sym in ["EURUSD", "BTCUSD", "XAUUSD", "NAS100"]:
            assert sym in INSTRUMENTS

    def test_canonical_timeframes(self):
        assert len(TIMEFRAMES) == 8
        for tf in ["1M", "1H", "4H", "1D"]:
            assert tf in TIMEFRAMES

    def test_register_custom_instrument(self):
        register_instrument("USDZAR", "forex", 0.001, "US Dollar / South African Rand")
        assert "USDZAR" in get_all_instruments()

    def test_register_custom_timeframe(self):
        register_timeframe("2H", 120)
        assert "2H" in get_all_timeframes()
        assert get_all_timeframes()["2H"] == 120

    def test_instrument_metadata(self):
        assert INSTRUMENTS["EURUSD"]["type"] == "forex"
        assert INSTRUMENTS["BTCUSD"]["type"] == "crypto"
        assert INSTRUMENTS["XAUUSD"]["type"] == "commodity"


# ══════════════════════════════════════════════════════════════════════════
# 5 — Statistical Metrics Engine
# ══════════════════════════════════════════════════════════════════════════


class TestStatisticalMetricsEngine:
    def test_all_17_metrics_present(self, probs_3class):
        probs, labels = probs_3class
        engine = StatisticalMetricsEngine(n_classes=3)
        m = engine.compute(probs, labels)
        assert isinstance(m, StatisticalMetrics)
        d = m.to_dict()
        required = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc",
            "pr_auc",
            "log_loss",
            "cross_entropy",
            "mae",
            "rmse",
            "mape",
            "r_squared",
            "ece",
            "brier_score",
            "prediction_entropy",
            "confidence_mean",
            "mce",
        ]
        for key in required:
            assert key in d, f"Missing metric: {key}"

    def test_accuracy_bounds(self, probs_3class):
        probs, labels = probs_3class
        m = StatisticalMetricsEngine(n_classes=3).compute(probs, labels)
        assert 0.0 <= m.accuracy <= 1.0

    def test_f1_bounds(self, probs_3class):
        probs, labels = probs_3class
        m = StatisticalMetricsEngine(n_classes=3).compute(probs, labels)
        assert 0.0 <= m.f1_macro <= 1.0

    def test_ece_bounds(self, probs_3class):
        probs, labels = probs_3class
        m = StatisticalMetricsEngine(n_classes=3).compute(probs, labels)
        assert 0.0 <= m.ece <= 1.0

    def test_perfect_model(self):
        n, C = 100, 3
        y = torch.randint(0, C, (n,))
        p = torch.zeros(n, C)
        p.scatter_(1, y.unsqueeze(1), 1.0)
        m = StatisticalMetricsEngine(n_classes=C).compute(p, y)
        assert m.accuracy == pytest.approx(1.0, abs=1e-5)

    def test_to_dict_all_floats(self, probs_3class):
        probs, labels = probs_3class
        m = StatisticalMetricsEngine(n_classes=3).compute(probs, labels)
        d = m.to_dict()
        for k, v in d.items():
            assert isinstance(v, (int, float)), f"{k} is not numeric"

    def test_roc_auc_random(self, probs_3class):
        probs, labels = probs_3class
        m = StatisticalMetricsEngine(n_classes=3).compute(probs, labels)
        assert 0.0 <= m.roc_auc <= 1.0

    def test_brier_score(self, probs_3class):
        probs, labels = probs_3class
        m = StatisticalMetricsEngine(n_classes=3).compute(probs, labels)
        assert m.brier_score >= 0.0


# ══════════════════════════════════════════════════════════════════════════
# 6 — Trading Metrics Engine
# ══════════════════════════════════════════════════════════════════════════


class TestTradingMetricsEngine:
    def test_21_metrics_present(self, trade_list):
        m = TradingMetricsEngine().compute(trade_list)
        d = m.to_dict()
        required = [
            "win_rate",
            "profit_factor",
            "avg_rr",
            "avg_holding_bars",
            "trade_frequency",
            "avg_mfe_pips",
            "avg_mae_pips",
            "expectancy",
            "max_drawdown",
            "avg_drawdown",
            "recovery_factor",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "ulcer_index",
            "mar_ratio",
            "risk_of_ruin",
            "capital_efficiency",
            "return_skewness",
            "return_kurtosis",
            "tail_risk_cvar5",
            "tp_before_sl_accuracy",
        ]
        for key in required:
            assert key in d, f"Missing metric: {key}"

    def test_win_rate_bounds(self, trade_list):
        m = TradingMetricsEngine().compute(trade_list)
        assert 0.0 <= m.win_rate <= 1.0

    def test_empty_trades(self):
        m = TradingMetricsEngine().compute([])
        assert m.n_trades == 0
        assert m.win_rate == 0.0

    def test_n_trades(self, trade_list):
        m = TradingMetricsEngine().compute(trade_list)
        assert m.n_trades == len(trade_list)

    def test_max_drawdown_non_negative(self, trade_list):
        m = TradingMetricsEngine().compute(trade_list)
        assert m.max_drawdown >= 0.0

    def test_profit_factor_positive(self, trade_list):
        m = TradingMetricsEngine().compute(trade_list)
        assert m.profit_factor >= 0.0

    def test_risk_of_ruin_bounds(self, trade_list):
        m = TradingMetricsEngine().compute(trade_list)
        assert 0.0 <= m.risk_of_ruin <= 1.0


class TestTradeSimulator:
    def test_simulate_returns_list(self, probs_3class):
        probs, labels = probs_3class
        probs_np = probs.numpy()
        ohlcv = np.random.rand(200, 5) * 0.01 + 1.1
        sim = TradeSimulator(pip_size=0.0001, spread_pips=1.0)
        trades = sim.simulate(probs_np, ohlcv)
        assert isinstance(trades, list)

    def test_simulate_trade_fields(self, probs_3class):
        probs, labels = probs_3class
        probs_np = probs.numpy()
        ohlcv = np.random.rand(200, 5) * 0.01 + 1.1
        sim = TradeSimulator(pip_size=0.0001)
        trades = sim.simulate(probs_np, ohlcv)
        if trades:
            t = trades[0]
            assert hasattr(t, "pnl_pips")
            assert hasattr(t, "confidence")


# ══════════════════════════════════════════════════════════════════════════
# 7 — Monte Carlo Validator
# ══════════════════════════════════════════════════════════════════════════


class TestMonteCarloValidator:
    def test_runs_and_returns_result(self, trade_list):
        mc = MonteCarloValidator(n_simulations=50, seed=0)
        result = mc.run(trade_list)
        assert isinstance(result, MonteCarloResult)
        assert result.n_simulations == 50

    def test_equity_bands_ordered(self, trade_list):
        mc = MonteCarloValidator(n_simulations=100, seed=0)
        result = mc.run(trade_list)
        assert result.final_equity_p5 <= result.final_equity_p50 <= result.final_equity_p95

    def test_ruin_probability_bounds(self, trade_list):
        mc = MonteCarloValidator(n_simulations=100, seed=0)
        result = mc.run(trade_list)
        assert 0.0 <= result.ruin_probability <= 1.0

    def test_summary_string(self, trade_list):
        mc = MonteCarloValidator(n_simulations=50, seed=0)
        result = mc.run(trade_list)
        assert "Monte Carlo" in result.summary()

    def test_to_dict(self, trade_list):
        mc = MonteCarloValidator(n_simulations=50, seed=0)
        result = mc.run(trade_list)
        d = result.to_dict()
        assert "final_equity_p50" in d
        assert "ruin_probability" in d

    def test_empty_trades(self):
        mc = MonteCarloValidator(n_simulations=50, seed=0)
        result = mc.run([])
        assert result.n_simulations == 50

    def test_reproducible(self, trade_list):
        mc1 = MonteCarloValidator(n_simulations=50, seed=7)
        mc2 = MonteCarloValidator(n_simulations=50, seed=7)
        r1 = mc1.run(trade_list)
        r2 = mc2.run(trade_list)
        assert r1.final_equity_p50 == pytest.approx(r2.final_equity_p50, abs=1e-3)


# ══════════════════════════════════════════════════════════════════════════
# 8 — Robustness Tester
# ══════════════════════════════════════════════════════════════════════════


class TestRobustnessTester:
    def test_returns_report(self, simple_model, small_xy):
        x, y = small_xy
        tester = RobustnessTester(seed=0)
        report = tester.evaluate(simple_model, x[:100], y[:100])
        assert isinstance(report, RobustnessReport)
        assert 0.0 <= report.overall_robustness_score <= 1.0

    def test_scenarios_present(self, simple_model, small_xy):
        x, y = small_xy
        tester = RobustnessTester(seed=0)
        report = tester.evaluate(simple_model, x[:100], y[:100])
        assert len(report.scenarios) > 0

    def test_summary_string(self, simple_model, small_xy):
        x, y = small_xy
        tester = RobustnessTester(seed=0)
        report = tester.evaluate(simple_model, x[:100], y[:100])
        summary = report.summary()
        assert "Robustness" in summary

    def test_to_dict(self, simple_model, small_xy):
        x, y = small_xy
        tester = RobustnessTester(seed=0)
        report = tester.evaluate(simple_model, x[:100], y[:100])
        d = report.to_dict()
        assert "overall_robustness_score" in d
        assert "scenarios" in d


# ══════════════════════════════════════════════════════════════════════════
# 9 — Calibration Validator
# ══════════════════════════════════════════════════════════════════════════


class TestCalibrationValidator:
    def test_evaluate_returns_report(self, probs_3class):
        probs, labels = probs_3class
        v = CalibrationValidator(n_bins=10, n_classes=3)
        report = v.evaluate(probs, labels)
        assert isinstance(report, CalibrationReport)

    def test_ece_bounds(self, probs_3class):
        probs, labels = probs_3class
        v = CalibrationValidator()
        report = v.evaluate(probs, labels)
        assert 0.0 <= report.ece <= 1.0

    def test_summary(self, probs_3class):
        probs, labels = probs_3class
        v = CalibrationValidator()
        report = v.evaluate(probs, labels)
        assert "ECE" in report.summary()

    def test_reliability_chart(self, probs_3class):
        probs, labels = probs_3class
        v = CalibrationValidator(n_bins=10)
        report = v.evaluate(probs, labels)
        chart = report.reliability_chart_data()
        assert "bin_midpoints" in chart
        assert len(chart["bin_midpoints"]) == 10

    def test_temperature_recommendation(self, probs_3class):
        probs, labels = probs_3class
        v = CalibrationValidator()
        report = v.evaluate(probs, labels)
        assert report.recommended_temperature > 0.0

    def test_per_class_ece(self, probs_3class):
        probs, labels = probs_3class
        v = CalibrationValidator(n_classes=3)
        report = v.evaluate(probs, labels)
        assert len(report.per_class_ece) == 3


# ══════════════════════════════════════════════════════════════════════════
# 10 — Drift Analyzer
# ══════════════════════════════════════════════════════════════════════════


class TestDriftAnalyzer:
    def test_no_drift_on_identical(self, small_xy):
        x, y = small_xy
        analyzer = DriftAnalyzer(psi_threshold=0.20)
        report = analyzer.detect(x[:400], x[:400], y[:400], y[:400])
        assert report.drift_severity in ("none", "minor")

    def test_detects_severe_drift(self, small_xy):
        x, y = small_xy
        x_shifted = x * 10 + 50  # massive shift
        analyzer = DriftAnalyzer(psi_threshold=0.05)
        report = analyzer.detect(x[:400], x_shifted[:400], y[:400], y[:400])
        assert report.drift_severity in ("moderate", "severe")

    def test_stores_history(self, small_xy):
        x, y = small_xy
        analyzer = DriftAnalyzer()
        analyzer.detect(x[:400], x[400:], y[:400], y[400:])
        analyzer.detect(x[:400], x[400:], y[:400], y[400:])
        assert len(analyzer.history()) == 2

    def test_to_dict(self, small_xy):
        x, y = small_xy
        analyzer = DriftAnalyzer()
        report = analyzer.detect(x[:400], x[400:], y[:400], y[400:])
        d = report.to_dict()
        assert "drift_severity" in d
        assert "feature_drift" in d

    def test_summary_string(self, small_xy):
        x, y = small_xy
        analyzer = DriftAnalyzer()
        report = analyzer.detect(x[:400], x[400:], y[:400], y[400:])
        assert "Drift Report" in report.summary()


# ══════════════════════════════════════════════════════════════════════════
# 11 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════


class TestModelComparison:
    @pytest.fixture
    def two_snapshots(self, probs_3class, trade_list):
        probs, labels = probs_3class
        stat = StatisticalMetricsEngine(n_classes=3).compute(probs, labels)
        trading = TradingMetricsEngine().compute(trade_list)
        snap_a = ModelSnapshot(
            name="champion",
            statistical=stat,
            trading=trading,
            inference_latency_ms=5.0,
            model_size_mb=10.0,
        )
        snap_b = ModelSnapshot(
            name="candidate",
            statistical=stat,
            trading=trading,
            inference_latency_ms=4.0,
            model_size_mb=8.0,
        )
        return snap_a, snap_b

    def test_compare_returns_report(self, two_snapshots):
        snap_a, snap_b = two_snapshots
        comp = ModelComparison()
        comp.add_snapshot(snap_a)
        comp.add_snapshot(snap_b)
        report = comp.compare("champion", "candidate")
        assert isinstance(report, ComparisonReport)

    def test_report_has_winner(self, two_snapshots):
        snap_a, snap_b = two_snapshots
        comp = ModelComparison()
        comp.add_snapshot(snap_a)
        comp.add_snapshot(snap_b)
        report = comp.compare("champion", "candidate")
        assert report.winner in ("champion", "candidate", "tie")

    def test_metric_deltas(self, two_snapshots):
        snap_a, snap_b = two_snapshots
        comp = ModelComparison()
        comp.add_snapshot(snap_a)
        comp.add_snapshot(snap_b)
        report = comp.compare("champion", "candidate")
        assert len(report.metric_deltas) > 0

    def test_summary_string(self, two_snapshots):
        snap_a, snap_b = two_snapshots
        comp = ModelComparison()
        comp.add_snapshot(snap_a)
        comp.add_snapshot(snap_b)
        report = comp.compare("champion", "candidate")
        assert "vs" in report.summary()

    def test_to_dict(self, two_snapshots):
        snap_a, snap_b = two_snapshots
        comp = ModelComparison()
        comp.add_snapshot(snap_a)
        comp.add_snapshot(snap_b)
        report = comp.compare("champion", "candidate")
        d = report.to_dict()
        assert "winner" in d
        assert "metric_deltas" in d


# ══════════════════════════════════════════════════════════════════════════
# 12 — Institutional Leaderboard
# ══════════════════════════════════════════════════════════════════════════


class TestInstitutionalLeaderboard:
    def _make_entry(self, model_id="m1", acc=0.60, sharpe=1.2):
        return LeaderboardEntry(
            model_id=model_id,
            model_name=f"Model-{model_id}",
            validation_accuracy=acc,
            sharpe_ratio=sharpe,
            profit_factor=1.5,
            max_drawdown=0.10,
            win_rate=0.55,
            validation_f1=0.60,
            ece=0.05,
        )

    def test_add_and_retrieve(self, tmp_lb):
        e = self._make_entry()
        tmp_lb.add(e)
        assert tmp_lb.get("m1") is not None

    def test_top_k(self, tmp_lb):
        for i in range(5):
            tmp_lb.add(self._make_entry(f"m{i}", acc=0.50 + i * 0.05))
        top = tmp_lb.top(3)
        assert len(top) == 3

    def test_filter_by_status(self, tmp_lb):
        e = self._make_entry("x1")
        e.promotion_status = "champion"
        tmp_lb.add(e)
        results = tmp_lb.filter(promotion_status="champion")
        assert any(r.model_id == "x1" for r in results)

    def test_search(self, tmp_lb):
        e = self._make_entry("find_me")
        e.model_name = "UniqueModelAlpha"
        tmp_lb.add(e)
        results = tmp_lb.search("uniquemodelalpha")
        assert any(r.model_id == "find_me" for r in results)

    def test_update(self, tmp_lb):
        tmp_lb.add(self._make_entry("u1"))
        tmp_lb.update("u1", {"validation_accuracy": 0.99})
        assert tmp_lb.get("u1").validation_accuracy == pytest.approx(0.99)

    def test_format_table(self, tmp_lb):
        for i in range(3):
            tmp_lb.add(self._make_entry(f"t{i}"))
        table = tmp_lb.format_table()
        assert "Model" in table

    def test_compare(self, tmp_lb):
        tmp_lb.add(self._make_entry("a1", acc=0.60))
        tmp_lb.add(self._make_entry("a2", acc=0.70))
        result = tmp_lb.compare("a1", "a2")
        assert "comparison" in result

    def test_persistence(self, tmp_path):
        lb = InstitutionalLeaderboard(store_path=tmp_path / "lb.json")
        lb.add(self._make_entry("persist_me"))
        lb2 = InstitutionalLeaderboard(store_path=tmp_path / "lb.json")
        assert lb2.get("persist_me") is not None


# ══════════════════════════════════════════════════════════════════════════
# 13 — Promotion Policy
# ══════════════════════════════════════════════════════════════════════════


class TestPromotionPolicy:
    def test_all_rules_pass(self):
        policy = PromotionPolicy(
            min_accuracy=0.50,
            min_sharpe=0.5,
            max_drawdown=0.30,
            require_manual_approval=False,
        )
        decision = policy.evaluate(
            "m1",
            {"accuracy": 0.60, "sharpe_ratio": 1.0, "max_drawdown": 0.15},
        )
        assert decision.is_eligible
        assert decision.is_promoted

    def test_fails_accuracy(self):
        policy = PromotionPolicy(min_accuracy=0.70, require_manual_approval=False)
        decision = policy.evaluate("m1", {"accuracy": 0.50})
        assert not decision.is_eligible
        assert len(decision.failed_rules) > 0

    def test_manual_approval_required(self):
        policy = PromotionPolicy(min_accuracy=0.50, require_manual_approval=True)
        decision = policy.evaluate("m1", {"accuracy": 0.60})
        assert decision.is_eligible
        assert not decision.is_promoted

    def test_manual_approve(self):
        policy = PromotionPolicy(min_accuracy=0.50, require_manual_approval=True)
        decision = policy.evaluate("m1", {"accuracy": 0.60})
        approved = policy.approve(decision, approver="researcher_a")
        assert approved.is_promoted
        assert "researcher_a" in approved.notes

    def test_approve_ineligible_raises(self):
        policy = PromotionPolicy(min_accuracy=0.90, require_manual_approval=True)
        decision = policy.evaluate("m1", {"accuracy": 0.50})
        with pytest.raises(ValueError):
            policy.approve(decision)

    def test_to_dict(self):
        policy = PromotionPolicy(min_accuracy=0.55, min_sharpe=1.0)
        decision = policy.evaluate("m1", {"accuracy": 0.60, "sharpe_ratio": 1.2})
        d = decision.to_dict()
        assert "is_eligible" in d
        assert "failed_rules" in d

    def test_describe(self):
        policy = PromotionPolicy(min_accuracy=0.55, max_drawdown=0.20)
        rules = policy.describe()
        assert len(rules) == 2

    def test_custom_rule(self):
        policy = PromotionPolicy()
        rule = PromotionRule("win_rate", 0.50, is_max=False, label="Min Win Rate")
        policy.add_rule(rule)
        dec = policy.evaluate("m1", {"win_rate": 0.45})
        assert not dec.is_eligible

    def test_no_rules_permissive(self):
        policy = PromotionPolicy(require_manual_approval=False)
        decision = policy.evaluate("m1", {"accuracy": 0.99})
        assert decision.is_eligible


# ══════════════════════════════════════════════════════════════════════════
# 14 — Failure Analyzer
# ══════════════════════════════════════════════════════════════════════════


class TestFailureAnalyzer:
    def test_analyze_with_regimes(self):
        regime_perf = {
            "trending_strong": {"accuracy": 0.70},
            "ranging": {"accuracy": 0.40},
            "high_volatility": {"accuracy": 0.38},
        }
        analyzer = FailureAnalyzer(top_k=3)
        report = analyzer.analyze(regime_performance=regime_perf)
        assert len(report.worst_regimes) <= 3
        assert report.worst_regimes[0].value < 0.50

    def test_recommendations_generated(self):
        regime_perf = {"ranging": {"accuracy": 0.30}}
        analyzer = FailureAnalyzer()
        report = analyzer.analyze(regime_performance=regime_perf)
        assert len(report.recommendations) > 0

    def test_summary_string(self):
        analyzer = FailureAnalyzer()
        report = analyzer.analyze()
        assert "Failure" in report.summary()

    def test_to_dict(self):
        analyzer = FailureAnalyzer()
        report = analyzer.analyze(regime_performance={"ranging": {"accuracy": 0.35}})
        d = report.to_dict()
        assert "worst_regimes" in d
        assert "recommendations" in d


# ══════════════════════════════════════════════════════════════════════════
# 15 — Validation Database
# ══════════════════════════════════════════════════════════════════════════


class TestValidationDatabase:
    def _make_record(self, model_id="test_model"):
        return ValidationRunRecord(
            run_id=str(uuid.uuid4()),
            model_id=model_id,
            model_version="1.0.0",
            statistical_metrics={"accuracy": 0.60},
            trading_metrics={"sharpe_ratio": 1.2},
            status="completed",
        )

    def test_save_and_retrieve(self, tmp_db):
        record = self._make_record()
        tmp_db.save_run(record)
        retrieved = tmp_db.get_run(record.run_id)
        assert retrieved is not None
        assert retrieved.model_id == "test_model"

    def test_list_runs(self, tmp_db):
        for i in range(5):
            tmp_db.save_run(self._make_record(f"model_{i}"))
        runs = tmp_db.list_runs(limit=10)
        assert len(runs) == 5

    def test_filter_by_model_id(self, tmp_db):
        tmp_db.save_run(self._make_record("alpha"))
        tmp_db.save_run(self._make_record("beta"))
        runs = tmp_db.list_runs(model_id="alpha")
        assert all(r.model_id == "alpha" for r in runs)

    def test_update_status(self, tmp_db):
        r = self._make_record()
        tmp_db.save_run(r)
        tmp_db.update_status(r.run_id, "failed")
        retrieved = tmp_db.get_run(r.run_id)
        assert retrieved.status == "failed"

    def test_count_runs(self, tmp_db):
        for _ in range(3):
            tmp_db.save_run(self._make_record())
        assert tmp_db.count_runs() == 3

    def test_get_best_run(self, tmp_db):
        r1 = self._make_record("m1")
        r1.leaderboard_score = 0.50
        r2 = self._make_record("m1")
        r2.leaderboard_score = 0.90
        tmp_db.save_run(r1)
        tmp_db.save_run(r2)
        best = tmp_db.get_best_run("m1")
        assert best.leaderboard_score == pytest.approx(0.90)

    def test_delete_run(self, tmp_db):
        r = self._make_record()
        tmp_db.save_run(r)
        deleted = tmp_db.delete_run(r.run_id)
        assert deleted
        assert tmp_db.get_run(r.run_id) is None

    def test_metrics_serialised_correctly(self, tmp_db):
        r = self._make_record()
        r.statistical_metrics = {"accuracy": 0.75, "f1_macro": 0.70}
        tmp_db.save_run(r)
        retrieved = tmp_db.get_run(r.run_id)
        assert retrieved.statistical_metrics["accuracy"] == pytest.approx(0.75)


# ══════════════════════════════════════════════════════════════════════════
# 16 — Prediction Head Validator
# ══════════════════════════════════════════════════════════════════════════


class TestPredictionHeadValidator:
    def test_returns_dict(self, simple_model, small_xy):
        x, y = small_xy
        v = PredictionHeadValidator()
        results = v.validate(simple_model, x[:100], y[:100])
        assert isinstance(results, dict)
        assert "classification_head" in results

    def test_classification_head_active(self, simple_model, small_xy):
        x, y = small_xy
        v = PredictionHeadValidator()
        results = v.validate(simple_model, x[:100], y[:100])
        assert results["classification_head"].is_active
        assert 0.0 <= results["classification_head"].accuracy <= 1.0

    def test_inactive_heads_marked(self, simple_model, small_xy):
        x, y = small_xy
        v = PredictionHeadValidator()
        results = v.validate(simple_model, x[:100], y[:100])
        # Most heads should be inactive for a simple Sequential model
        inactive = [r for r in results.values() if not r.is_active]
        assert len(inactive) > 0

    def test_to_dict(self, simple_model, small_xy):
        x, y = small_xy
        v = PredictionHeadValidator()
        results = v.validate(simple_model, x[:100], y[:100])
        d = results["classification_head"].to_dict()
        assert "accuracy" in d
        assert "head_type" in d


# ══════════════════════════════════════════════════════════════════════════
# 17 — Visualization Data Builder
# ══════════════════════════════════════════════════════════════════════════


class TestVisualizationDataBuilder:
    def test_walk_forward_timeline(self, small_xy):
        x, y = small_xy
        v = SlidingWindowValidator(train_size=200, val_size=100, step_size=100)
        folds = list(v.split(x, y))
        data = VisualizationDataBuilder.build_walk_forward_timeline(folds)
        assert data["chart_type"] == "walk_forward_timeline"
        assert len(data["folds"]) == len(folds)

    def test_equity_curves(self, trade_list):
        mc = MonteCarloValidator(n_simulations=50).run(trade_list)
        data = VisualizationDataBuilder.build_equity_curves(mc)
        assert data["chart_type"] == "equity_curves"
        assert "p50" in data

    def test_calibration_curve(self, probs_3class):
        probs, labels = probs_3class
        report = CalibrationValidator(n_bins=10).evaluate(probs, labels)
        data = VisualizationDataBuilder.build_calibration_curve(report)
        assert data["chart_type"] == "calibration_curve"
        assert len(data["bin_midpoints"]) == 10

    def test_trade_distribution(self, trade_list):
        data = VisualizationDataBuilder.build_trade_distribution(trade_list)
        assert data["chart_type"] == "trade_distribution"
        assert "n_trades" in data

    def test_regime_chart(self):
        perf = {"ranging": {"accuracy": 0.40}, "trending_strong": {"accuracy": 0.70}}
        data = VisualizationDataBuilder.build_regime_chart(perf)
        assert data["chart_type"] == "regime_performance"
        assert len(data["regime_names"]) == 2


# ══════════════════════════════════════════════════════════════════════════
# 18 — Report Exporter
# ══════════════════════════════════════════════════════════════════════════


class TestReportExporter:
    def _make_record(self):
        r = ValidationRunRecord(run_id=str(uuid.uuid4()), model_id="test")
        r.statistical_metrics = {"accuracy": 0.60, "f1_macro": 0.58}
        r.trading_metrics = {"sharpe_ratio": 1.2, "profit_factor": 1.5}
        r.calibration = {"ece": 0.05, "mce": 0.12}
        r.promotion_decision = {"is_eligible": True, "is_promoted": False}
        r.visualization_data = {}
        return r

    def test_exports_json(self, tmp_path):
        exporter = ReportExporter(
            output_dir=tmp_path, config=ExportConfig(formats=["json"], output_dir=tmp_path)
        )
        paths = exporter.export_all(self._make_record())
        assert len(paths) == 1
        assert paths[0].suffix == ".json"
        data = json.loads(paths[0].read_text())
        assert "run_id" in data

    def test_exports_csv(self, tmp_path):
        exporter = ReportExporter(
            output_dir=tmp_path, config=ExportConfig(formats=["csv"], output_dir=tmp_path)
        )
        paths = exporter.export_all(self._make_record())
        assert len(paths) == 1
        assert paths[0].suffix == ".csv"

    def test_exports_html(self, tmp_path):
        exporter = ReportExporter(
            output_dir=tmp_path, config=ExportConfig(formats=["html"], output_dir=tmp_path)
        )
        paths = exporter.export_all(self._make_record())
        assert any(p.suffix == ".html" for p in paths)
        content = paths[0].read_text()
        assert "<html>" in content


# ══════════════════════════════════════════════════════════════════════════
# 19 — Validation API
# ══════════════════════════════════════════════════════════════════════════


class TestValidationAPI:
    @pytest.fixture
    def api_setup(self, tmp_path):
        db = ValidationDatabase(tmp_path / "api_test.db")
        lb = InstitutionalLeaderboard()
        policy = PromotionPolicy(require_manual_approval=True)
        api = ValidationAPI(db, lb, policy)
        yield api, db, lb
        db.close()

    def test_get_report_not_found(self, api_setup):
        api, _, _ = api_setup
        resp = api.get_report("nonexistent-id")
        assert resp["status"] == "error"
        assert resp["code"] == "NOT_FOUND"

    def test_get_report_found(self, api_setup):
        api, db, _ = api_setup
        r = ValidationRunRecord(run_id="test-run-001", model_id="m1")
        r.statistical_metrics = {"accuracy": 0.60}
        r.trading_metrics = {"sharpe_ratio": 1.0}
        r.calibration = {}
        r.drift = {}
        r.monte_carlo = {}
        r.robustness = {}
        r.promotion_decision = {}
        db.save_run(r)
        resp = api.get_report("test-run-001")
        assert resp["status"] == "ok"

    def test_list_reports_empty(self, api_setup):
        api, _, _ = api_setup
        resp = api.list_reports()
        assert resp["status"] == "ok"
        assert resp["data"]["total"] == 0

    def test_get_leaderboard_empty(self, api_setup):
        api, _, _ = api_setup
        resp = api.get_leaderboard()
        assert resp["status"] == "ok"
        assert resp["data"]["total"] == 0

    def test_search_leaderboard(self, api_setup):
        api, _, lb = api_setup
        e = LeaderboardEntry(model_id="search_target", model_name="UniqueSearchTarget")
        lb.add(e)
        resp = api.search_leaderboard("UniqueSearchTarget")
        assert resp["status"] == "ok"

    def test_retire_model(self, api_setup):
        api, _, lb = api_setup
        lb.add(LeaderboardEntry(model_id="to_retire", model_name="OldModel"))
        resp = api.retire_model("to_retire", reason="Superseded by v2")
        assert resp["status"] == "ok"

    def test_ok_response_structure(self):
        resp = _ok({"key": "value"}, "test message")
        assert resp["status"] == "ok"
        assert "timestamp" in resp

    def test_error_response_structure(self):
        resp = _error("something went wrong", "TEST_ERROR")
        assert resp["status"] == "error"
        assert resp["code"] == "TEST_ERROR"


# ══════════════════════════════════════════════════════════════════════════
# 20 — Integration: Full Engine Pipeline
# ══════════════════════════════════════════════════════════════════════════


class TestInstitutionalValidationEngine:
    def test_full_pipeline_completes(self, simple_model, small_xy, tmp_path):
        x, y = small_xy
        policy = PromotionPolicy(min_accuracy=0.10, require_manual_approval=False)
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            leaderboard_path=tmp_path / "lb.json",
            policy=policy,
            export_dir=tmp_path / "reports",
            config=EngineConfig(
                min_train_size=300,
                val_size=100,
                step_size=100,
                n_mc_simulations=10,
            ),
        )
        result = engine.validate(
            model=simple_model,
            x=x,
            y=y,
            model_id="integration_test",
            model_version="1.0.0",
        )
        engine.close()
        assert isinstance(result, ValidationResult)
        assert result.run_id != ""
        assert result.n_folds > 0

    def test_result_has_statistical_metrics(self, simple_model, small_xy, tmp_path):
        x, y = small_xy
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            export_dir=tmp_path / "reports",
            config=EngineConfig(
                min_train_size=300,
                val_size=100,
                step_size=100,
                n_mc_simulations=10,
            ),
        )
        result = engine.validate(simple_model, x, y, "test_model")
        engine.close()
        assert result.statistical is not None
        assert 0.0 <= result.statistical.accuracy <= 1.0

    def test_promotion_decision_present(self, simple_model, small_xy, tmp_path):
        x, y = small_xy
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            export_dir=tmp_path / "reports",
            config=EngineConfig(
                min_train_size=300,
                val_size=100,
                step_size=100,
                n_mc_simulations=10,
            ),
        )
        result = engine.validate(simple_model, x, y, "p_test")
        engine.close()
        assert result.promotion_decision is not None
        assert isinstance(result.promotion_decision.is_eligible, bool)

    def test_leaderboard_entry_added(self, simple_model, small_xy, tmp_path):
        x, y = small_xy
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            leaderboard_path=tmp_path / "lb.json",
            export_dir=tmp_path / "reports",
            config=EngineConfig(
                min_train_size=300,
                val_size=100,
                step_size=100,
                n_mc_simulations=10,
            ),
        )
        result = engine.validate(simple_model, x, y, "lb_test")
        assert engine.leaderboard.get("lb_test") is not None
        engine.close()

    def test_database_record_saved(self, simple_model, small_xy, tmp_path):
        x, y = small_xy
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            export_dir=tmp_path / "reports",
            config=EngineConfig(
                min_train_size=300,
                val_size=100,
                step_size=100,
                n_mc_simulations=10,
            ),
        )
        result = engine.validate(simple_model, x, y, "db_test")
        record = engine.database.get_run(result.run_id)
        engine.close()
        assert record is not None
        assert record.model_id == "db_test"

    def test_summary_string(self, simple_model, small_xy, tmp_path):
        x, y = small_xy
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            export_dir=tmp_path / "reports",
            config=EngineConfig(
                min_train_size=300,
                val_size=100,
                step_size=100,
                n_mc_simulations=10,
            ),
        )
        result = engine.validate(simple_model, x, y, "summary_test")
        engine.close()
        summary = result.summary()
        assert "Institutional Validation" in summary
        assert result.model_id in summary

    def test_small_dataset_fails_gracefully(self, simple_model, tmp_path):
        x = torch.randn(50, 10, 8)
        y = torch.randint(0, 3, (50,))
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            export_dir=tmp_path / "reports",
            config=EngineConfig(min_train_size=300, val_size=100, step_size=100),
        )
        result = engine.validate(simple_model, x, y, "tiny_model")
        engine.close()
        assert result.status in ("failed", "aborted_leakage", "completed")

    def test_api_accessible(self, simple_model, small_xy, tmp_path):
        x, y = small_xy
        engine = InstitutionalValidationEngine(
            db_path=tmp_path / "v.db",
            export_dir=tmp_path / "reports",
            config=EngineConfig(
                min_train_size=300,
                val_size=100,
                step_size=100,
                n_mc_simulations=10,
            ),
        )
        result = engine.validate(simple_model, x, y, "api_test")
        api_resp = engine.api.get_report(result.run_id)
        engine.close()
        assert api_resp["status"] == "ok"
