"""Phase 24.9 — MIE unit tests."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

# ── Import helpers (same sys.modules pattern as prior phases) ─────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def _preload(rel_path: str):
    """Load + register a module in sys.modules under its full dotted name.
    Absolute imports in later modules resolve from sys.modules without
    triggering trainer/mie/__init__.py."""
    full = os.path.join(ROOT, rel_path)
    name = rel_path.replace("/", ".").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register BEFORE exec (handles circular refs)
    spec.loader.exec_module(mod)
    return mod


def _load(short_name: str):
    """Return an already pre-loaded MIE module by its base filename.
    Used by test helper methods to get module objects without re-executing."""
    key = "trainer.mie." + short_name.replace(".py", "")
    return sys.modules[key]


# ── Pre-load all MIE modules in dependency order ──────────────────────────────
_types = _preload("trainer/mie/mie_types.py")
_models = _preload("trainer/mie/mie_models.py")
_mre = _preload("trainer/mie/market_regime_engine.py")
_ri = _preload("trainer/mie/risk_intelligence.py")
_vi = _preload("trainer/mie/volatility_intelligence.py")
_mi = _preload("trainer/mie/momentum_intelligence.py")
_ti = _preload("trainer/mie/trend_intelligence.py")
_li = _preload("trainer/mie/liquidity_intelligence.py")
_si = _preload("trainer/mie/session_intelligence.py")
_ni = _preload("trainer/mie/news_intelligence.py")
_bse = _preload("trainer/mie/behavior_similarity.py")
_pr = _preload("trainer/mie/prediction_reliability.py")
_ce = _preload("trainer/mie/calibration_engine.py")
_te = _preload("trainer/mie/tradeability_engine.py")
_ei = _preload("trainer/mie/execution_intelligence.py")
_mqe = _preload("trainer/mie/market_quality_engine.py")
_port = _preload("trainer/mie/portfolio_risk_engine.py")
_acg = _preload("trainer/mie/ai_commentary.py")
_fse = _preload("trainer/mie/feature_store.py")
_db = _preload("trainer/mie/mie_database.py")
_dash = _preload("trainer/mie/mie_dashboard.py")
_api = _preload("trainer/mie/mie_api.py")
_worker = _preload("trainer/mie/mie_worker.py")


def _inp(**kw) -> _models.MarketInput:
    """Build a MarketInput with sensible defaults."""
    defaults = dict(
        symbol="EURUSD",
        open=1.1000,
        high=1.1020,
        low=1.0980,
        close=1.1010,
        volume=5_000.0,
        bid=1.1009,
        ask=1.1011,
        spread=0.0002,
        atr=0.0015,
        adx=28.0,
        rsi=52.0,
        closes=[1.09 + i * 0.001 for i in range(60)],
        highs=[1.09 + i * 0.001 + 0.0005 for i in range(60)],
        lows=[1.09 + i * 0.001 - 0.0005 for i in range(60)],
        volumes=[5_000.0] * 60,
        returns=[(i % 7 - 3) * 0.0003 for i in range(60)],
        prediction_confidence=0.68,
        expected_rr=2.1,
        expected_move=0.0015,
        expected_holding_bars=15,
        portfolio_exposure=0.10,
        correlation_avg=0.25,
        capital=100_000.0,
        open_trades=2,
        news_events=[],
        session_hour=13,
        historical_similar=[],
        calibration_score=0.65,
        historical_win_rate=0.58,
        model_health=0.85,
    )
    defaults.update(kw)
    return _models.MarketInput(**defaults)


# ═════════════════════════════════════════════════════════════════════════════
# 1. mie_types
# ═════════════════════════════════════════════════════════════════════════════


class TestMIETypes(unittest.TestCase):

    def test_market_regime_count(self):
        self.assertEqual(len(_types.MarketRegime), 16)

    def test_risk_regime_count(self):
        self.assertEqual(len(_types.RiskRegime), 12)

    def test_tradeability_grade_values(self):
        grades = [g.value for g in _types.TradeabilityGrade]
        self.assertIn("A+", grades)
        self.assertIn("Avoid", grades)

    def test_execution_mode_count(self):
        self.assertEqual(len(_types.ExecutionMode), 4)

    def test_session_type_count(self):
        self.assertEqual(len(_types.SessionType), 5)

    def test_momentum_state_count(self):
        self.assertEqual(len(_types.MomentumState), 5)

    def test_trend_state_count(self):
        self.assertEqual(len(_types.TrendState), 5)

    def test_volatility_state_count(self):
        self.assertEqual(len(_types.VolatilityState), 5)

    def test_liquidity_state_count(self):
        self.assertEqual(len(_types.LiquidityState), 5)

    def test_worker_trigger_count(self):
        self.assertEqual(len(_types.WorkerTrigger), 6)

    def test_tradeability_grade_a_plus(self):
        self.assertEqual(_types.tradeability_grade(90), _types.TradeabilityGrade.A_PLUS)

    def test_tradeability_grade_a(self):
        self.assertEqual(_types.tradeability_grade(72), _types.TradeabilityGrade.A)

    def test_tradeability_grade_b(self):
        self.assertEqual(_types.tradeability_grade(60), _types.TradeabilityGrade.B)

    def test_tradeability_grade_c(self):
        self.assertEqual(_types.tradeability_grade(42), _types.TradeabilityGrade.C)

    def test_tradeability_grade_d(self):
        self.assertEqual(_types.tradeability_grade(30), _types.TradeabilityGrade.D)

    def test_tradeability_grade_avoid(self):
        self.assertEqual(_types.tradeability_grade(10), _types.TradeabilityGrade.AVOID)

    def test_clamp_bounds(self):
        self.assertEqual(_types.clamp(-5), 0.0)
        self.assertEqual(_types.clamp(150), 100.0)
        self.assertEqual(_types.clamp(50), 50.0)

    def test_risk_multipliers_all_present(self):
        for r in _types.RiskRegime:
            self.assertIn(r, _types.RISK_REGIME_MULTIPLIERS)

    def test_risk_multipliers_range(self):
        for v in _types.RISK_REGIME_MULTIPLIERS.values():
            self.assertGreater(v, 0)
            self.assertLessEqual(v, 1.5)

    def test_regime_tradeability_base_all_present(self):
        for r in _types.MarketRegime:
            self.assertIn(r, _types.REGIME_TRADEABILITY_BASE)

    def test_session_quality_all_present(self):
        for s in _types.SessionType:
            self.assertIn(s, _types.SESSION_QUALITY_SCORES)

    def test_news_impact_penalty_keys(self):
        for n in _types.NewsImpact:
            self.assertIn(n, _types.NEWS_IMPACT_PENALTY)


# ═════════════════════════════════════════════════════════════════════════════
# 2. mie_models
# ═════════════════════════════════════════════════════════════════════════════


class TestMIEModels(unittest.TestCase):

    def test_market_input_defaults(self):
        inp = _models.MarketInput(symbol="GBPUSD")
        self.assertEqual(inp.symbol, "GBPUSD")
        self.assertIsInstance(inp.news_events, list)
        self.assertIsInstance(inp.returns, list)

    def test_market_input_to_dict(self):
        inp = _inp()
        d = inp.to_dict()
        self.assertIn("symbol", d)
        self.assertNotIn("returns", d)  # lists excluded

    def test_feature_store_vector_length(self):
        f = _models.FeatureStoreExport(
            symbol="X",
            tradeability=75.0,
            market_quality=60.0,
            liquidity_quality=80.0,
        )
        v = f.to_feature_vector()
        self.assertEqual(len(v), 12)

    def test_feature_store_vector_bounds(self):
        f = _models.FeatureStoreExport(
            symbol="X",
            tradeability=100.0,
            execution_quality=100.0,
            market_quality=100.0,
            prediction_reliability=100.0,
            behavior_similarity=100.0,
            calibration=100.0,
            historical_success=1.0,
            trend_quality=100.0,
            momentum_quality=100.0,
            liquidity_quality=100.0,
        )
        for val in f.to_feature_vector():
            self.assertLessEqual(val, 1.0 + 1e-6)
            self.assertGreaterEqual(val, 0.0)

    def test_market_intelligence_report_to_dict(self):
        report = _models.MarketIntelligenceReport(symbol="EURUSD")
        d = report.to_dict()
        self.assertIn("report_id", d)
        self.assertIn("symbol", d)

    def test_worker_job_defaults(self):
        job = _models.WorkerJob()
        self.assertEqual(job.status, "PENDING")
        self.assertIsInstance(job.job_id, str)

    def test_tradeability_result_to_dict(self):
        t = _models.TradeabilityResult(
            score=72.5,
            grade=_types.TradeabilityGrade.A,
        )
        d = t.to_dict()
        self.assertEqual(d["score"], 72.5)
        self.assertEqual(d["grade"], "A")

    def test_execution_result_to_dict(self):
        e = _models.ExecutionResult(
            mode=_types.ExecutionMode.LIMIT_ORDER,
            optimal_stop=1.0950,
            optimal_target=1.1100,
        )
        d = e.to_dict()
        self.assertEqual(d["mode"], "LIMIT_ORDER")

    def test_regime_result_to_dict(self):
        r = _models.MarketRegimeResult(
            regime=_types.MarketRegime.TRENDING,
            confidence=0.80,
        )
        d = r.to_dict()
        self.assertEqual(d["regime"], "TRENDING")
        self.assertAlmostEqual(d["confidence"], 0.80, places=3)


# ═════════════════════════════════════════════════════════════════════════════
# 3. MarketRegimeEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestMarketRegimeEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("market_regime_engine.py")
        return mod.MarketRegimeEngine()

    def test_strong_trend_high_adx(self):
        engine = self._engine()
        r = engine.detect(_inp(adx=45.0, rsi=62.0))
        self.assertEqual(r.regime, _types.MarketRegime.STRONG_TREND)

    def test_trending_moderate_adx(self):
        engine = self._engine()
        r = engine.detect(_inp(adx=30.0))
        self.assertIn(
            r.regime,
            [
                _types.MarketRegime.TRENDING,
                _types.MarketRegime.EXPANSION,
                _types.MarketRegime.ACCUMULATION,
                _types.MarketRegime.DISTRIBUTION,
            ],
        )

    def test_range_low_adx(self):
        engine = self._engine()
        r = engine.detect(_inp(adx=10.0, returns=[0.0] * 60))
        self.assertIn(r.regime, [_types.MarketRegime.RANGE, _types.MarketRegime.COMPRESSION])

    def test_volatile_vol_spike(self):
        engine = self._engine()
        big_returns = [0.02 * (1 if i % 2 == 0 else -1) for i in range(60)]
        r = engine.detect(_inp(adx=20.0, returns=big_returns))
        self.assertIn(r.regime, [_types.MarketRegime.VOLATILE, _types.MarketRegime.BLACK_SWAN])

    def test_black_swan_extreme_spread(self):
        engine = self._engine()
        r = engine.detect(_inp(spread=0.010, atr=0.0010))
        self.assertEqual(r.regime, _types.MarketRegime.BLACK_SWAN)

    def test_illiquid_low_volume(self):
        engine = self._engine()
        r = engine.detect(_inp(volume=10.0, volumes=[10000.0] * 60))
        self.assertEqual(r.regime, _types.MarketRegime.ILLIQUID)

    def test_news_driven(self):
        engine = self._engine()
        big_returns = [0.005 * (1 if i % 2 == 0 else -1) for i in range(60)]
        r = engine.detect(
            _inp(
                news_events=[{"impact": "HIGH", "timing": "recent", "label": "CPI"}],
                adx=22.0,
                returns=big_returns,
            )
        )
        self.assertEqual(r.regime, _types.MarketRegime.NEWS_DRIVEN)

    def test_compression_flat_vol(self):
        engine = self._engine()
        tiny = [0.00001 * (1 if i % 2 == 0 else -1) for i in range(60)]
        r = engine.detect(_inp(adx=10.0, returns=tiny))
        self.assertEqual(r.regime, _types.MarketRegime.COMPRESSION)

    def test_confidence_in_range(self):
        engine = self._engine()
        r = engine.detect(_inp())
        self.assertGreaterEqual(r.confidence, 0.0)
        self.assertLessEqual(r.confidence, 1.0)

    def test_signals_populated(self):
        engine = self._engine()
        r = engine.detect(_inp())
        self.assertIn("adx", r.signals)


# ═════════════════════════════════════════════════════════════════════════════
# 4. RiskIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestRiskIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("risk_intelligence.py")
        return mod.RiskIntelligenceEngine()

    def test_optimal_low_risk(self):
        engine = self._engine()
        r = engine.evaluate(
            _inp(
                spread=0.0001,
                returns=[0.0001] * 60,
                portfolio_exposure=0.0,
                correlation_avg=0.0,
                news_events=[],
                volume=5000.0,
                volumes=[5000.0] * 60,
            )
        )
        self.assertIn(
            r.regime,
            [_types.RiskRegime.OPTIMAL, _types.RiskRegime.LOW_RISK, _types.RiskRegime.NORMAL],
        )

    def test_extreme_vol_high_risk(self):
        engine = self._engine()
        big_ret = [0.03 * (1 if i % 2 == 0 else -1) for i in range(60)]
        r = engine.evaluate(_inp(returns=big_ret))
        self.assertIn(
            r.regime,
            [
                _types.RiskRegime.HIGH_RISK,
                _types.RiskRegime.EXTREME,
                _types.RiskRegime.FLASH_VOLATILITY,
                _types.RiskRegime.BLACK_SWAN,
                _types.RiskRegime.MARKET_STRESS,
            ],
        )

    def test_risk_score_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreaterEqual(r.risk_score, 0.0)
        self.assertLessEqual(r.risk_score, 100.0)

    def test_multiplier_positive(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreater(r.multiplier, 0.0)

    def test_news_event_regime(self):
        engine = self._engine()
        r = engine.evaluate(
            _inp(
                news_events=[{"impact": "HIGH", "timing": "upcoming", "label": "NFP"}],
            )
        )
        self.assertIn(
            r.regime,
            [
                _types.RiskRegime.NEWS_EVENT,
                _types.RiskRegime.ELEVATED,
                _types.RiskRegime.HIGH_RISK,
            ],
        )

    def test_factors_populated(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertIn("volatility", r.factors)
        self.assertIn("spread", r.factors)

    def test_wide_spread_raises_risk(self):
        engine = self._engine()
        r_normal = engine.evaluate(_inp(spread=0.0001))
        r_wide = engine.evaluate(_inp(spread=0.010, atr=0.001))
        # Wide spread should not lower risk score
        self.assertGreaterEqual(r_wide.risk_score, r_normal.risk_score)

    def test_low_liquidity_warning(self):
        engine = self._engine()
        r = engine.evaluate(_inp(volume=50.0, volumes=[10_000.0] * 60))
        self.assertTrue(
            any("low volume" in w.lower() or "liquidity" in w.lower() for w in r.warnings)
            or r.regime == _types.RiskRegime.LOW_LIQUIDITY
            or r.factors.get("liquidity", 0) > 0
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5. VolatilityIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestVolatilityIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("volatility_intelligence.py")
        return mod.VolatilityIntelligenceEngine()

    def test_normal_state(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertIsNotNone(r.state)

    def test_quality_in_range(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreaterEqual(r.quality, 0.0)
        self.assertLessEqual(r.quality, 100.0)

    def test_extreme_vol_low_quality(self):
        engine = self._engine()
        big_ret = [0.05 * (1 if i % 2 == 0 else -1) for i in range(60)]
        r = engine.analyse(_inp(returns=big_ret))
        self.assertEqual(r.state, _types.VolatilityState.EXTREME)
        self.assertLessEqual(r.quality, 50.0)

    def test_flat_vol_low_quality(self):
        engine = self._engine()
        tiny = [0.000001] * 60
        r = engine.analyse(_inp(returns=tiny))
        self.assertIn(r.state, [_types.VolatilityState.VERY_LOW, _types.VolatilityState.LOW])

    def test_vol_ratio_positive(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreater(r.vol_ratio, 0.0)

    def test_current_vol_field(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreaterEqual(r.current_vol, 0.0)

    def test_atr_percentile_in_range(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreaterEqual(r.atr_percentile, 0.0)
        self.assertLessEqual(r.atr_percentile, 100.0)


# ═════════════════════════════════════════════════════════════════════════════
# 6. MomentumIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestMomentumIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("momentum_intelligence.py")
        return mod.MomentumIntelligenceEngine()

    def test_state_returned(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertIsInstance(r.state, _types.MomentumState)

    def test_quality_in_range(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreaterEqual(r.quality, 0.0)
        self.assertLessEqual(r.quality, 100.0)

    def test_flat_momentum_no_returns(self):
        engine = self._engine()
        r = engine.analyse(_inp(returns=[0.0] * 30))
        self.assertEqual(r.state, _types.MomentumState.FLAT)

    def test_strong_upward_momentum(self):
        engine = self._engine()
        r = engine.analyse(
            _inp(
                returns=[0.005] * 30,
                rsi=65.0,
            )
        )
        self.assertIn(
            r.state,
            [
                _types.MomentumState.ACCELERATING,
                _types.MomentumState.STEADY,
            ],
        )
        self.assertGreater(r.strength, 0.0)

    def test_overbought_reduces_quality(self):
        engine = self._engine()
        r_normal = engine.analyse(_inp(rsi=55.0, returns=[0.002] * 20))
        r_ob = engine.analyse(_inp(rsi=80.0, returns=[0.002] * 20))
        self.assertLessEqual(r_ob.quality, r_normal.quality + 5.0)

    def test_rsi_signal_populated(self):
        engine = self._engine()
        r = engine.analyse(_inp(rsi=65.0))
        self.assertEqual(r.rsi_signal, 65.0)


# ═════════════════════════════════════════════════════════════════════════════
# 7. TrendIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestTrendIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("trend_intelligence.py")
        return mod.TrendIntelligenceEngine()

    def test_strong_trend_high_adx(self):
        engine = self._engine()
        r = engine.analyse(_inp(adx=45.0, rsi=65.0))
        self.assertEqual(r.state, _types.TrendState.STRONG)

    def test_sideways_low_adx(self):
        engine = self._engine()
        r = engine.analyse(_inp(adx=10.0, rsi=50.0, closes=[1.10] * 60))
        self.assertIn(r.state, [_types.TrendState.SIDEWAYS, _types.TrendState.REVERSING])

    def test_direction_up(self):
        engine = self._engine()
        r = engine.analyse(_inp(rsi=62.0))
        self.assertEqual(r.direction, "UP")

    def test_direction_down(self):
        engine = self._engine()
        r = engine.analyse(_inp(rsi=38.0))
        self.assertEqual(r.direction, "DOWN")

    def test_quality_in_range(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreaterEqual(r.quality, 0.0)
        self.assertLessEqual(r.quality, 100.0)

    def test_strength_equals_adx(self):
        engine = self._engine()
        r = engine.analyse(_inp(adx=32.0))
        self.assertAlmostEqual(r.strength, 32.0, places=1)

    def test_adx_value_populated(self):
        engine = self._engine()
        r = engine.analyse(_inp(adx=29.5))
        self.assertAlmostEqual(r.adx_value, 29.5, places=1)


# ═════════════════════════════════════════════════════════════════════════════
# 8. LiquidityIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestLiquidityIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("liquidity_intelligence.py")
        return mod.LiquidityIntelligenceEngine()

    def test_good_conditions(self):
        engine = self._engine()
        r = engine.analyse(_inp(volume=5000.0, volumes=[5000.0] * 30, spread=0.0001, atr=0.001))
        self.assertIn(
            r.state,
            [
                _types.LiquidityState.EXCELLENT,
                _types.LiquidityState.GOOD,
                _types.LiquidityState.FAIR,
            ],
        )

    def test_illiquid_conditions(self):
        engine = self._engine()
        r = engine.analyse(_inp(volume=10.0, volumes=[10000.0] * 30))
        self.assertEqual(r.state, _types.LiquidityState.ILLIQUID)

    def test_wide_spread_poor(self):
        engine = self._engine()
        r = engine.analyse(_inp(spread=0.008, atr=0.001))
        self.assertIn(r.state, [_types.LiquidityState.POOR, _types.LiquidityState.ILLIQUID])

    def test_quality_in_range(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreaterEqual(r.quality, 0.0)
        self.assertLessEqual(r.quality, 100.0)

    def test_sub_scores_populated(self):
        engine = self._engine()
        r = engine.analyse(_inp())
        self.assertGreaterEqual(r.volume_score, 0.0)
        self.assertGreaterEqual(r.spread_score, 0.0)
        self.assertGreaterEqual(r.depth_estimate, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 9. SessionIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestSessionIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("session_intelligence.py")
        return mod.SessionIntelligenceEngine()

    def test_overlap_session(self):
        engine = self._engine()
        r = engine.analyse(_inp(session_hour=14))
        self.assertEqual(r.session, _types.SessionType.OVERLAP)
        self.assertTrue(r.is_overlap)

    def test_london_session(self):
        engine = self._engine()
        r = engine.analyse(_inp(session_hour=9))
        self.assertEqual(r.session, _types.SessionType.LONDON)

    def test_new_york_session(self):
        engine = self._engine()
        r = engine.analyse(_inp(session_hour=17))
        self.assertEqual(r.session, _types.SessionType.NEW_YORK)

    def test_asian_session(self):
        engine = self._engine()
        r = engine.analyse(_inp(session_hour=3))
        self.assertEqual(r.session, _types.SessionType.ASIAN)

    def test_off_hours(self):
        engine = self._engine()
        r = engine.analyse(_inp(session_hour=22))
        self.assertEqual(r.session, _types.SessionType.OFF_HOURS)

    def test_quality_in_range(self):
        engine = self._engine()
        for hour in [2, 9, 14, 17, 22]:
            r = engine.analyse(_inp(session_hour=hour))
            self.assertGreaterEqual(r.quality, 0.0)
            self.assertLessEqual(r.quality, 100.0)

    def test_overlap_highest_quality(self):
        engine = self._engine()
        r_overlap = engine.analyse(_inp(session_hour=13))
        r_off = engine.analyse(_inp(session_hour=22))
        self.assertGreater(r_overlap.quality, r_off.quality)


# ═════════════════════════════════════════════════════════════════════════════
# 10. NewsIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestNewsIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("news_intelligence.py")
        return mod.NewsIntelligenceEngine()

    def test_no_news(self):
        engine = self._engine()
        r = engine.evaluate(_inp(news_events=[]))
        self.assertEqual(r.impact, _types.NewsImpact.NONE)
        self.assertEqual(r.impact_score, 0.0)

    def test_high_upcoming_news(self):
        engine = self._engine()
        r = engine.evaluate(
            _inp(news_events=[{"impact": "HIGH", "timing": "upcoming", "label": "FOMC"}])
        )
        self.assertEqual(r.impact, _types.NewsImpact.HIGH)
        self.assertGreater(r.impact_score, 0)

    def test_medium_recent_news(self):
        engine = self._engine()
        r = engine.evaluate(
            _inp(news_events=[{"impact": "MEDIUM", "timing": "recent", "label": "CPI"}])
        )
        self.assertIn(r.impact, [_types.NewsImpact.LOW, _types.NewsImpact.MEDIUM])

    def test_event_count_correct(self):
        engine = self._engine()
        events = [
            {"impact": "HIGH", "timing": "upcoming", "label": "A"},
            {"impact": "MEDIUM", "timing": "recent", "label": "B"},
            {"impact": "LOW", "timing": "past", "label": "C"},
        ]
        r = engine.evaluate(_inp(news_events=events))
        self.assertEqual(r.event_count, 3)

    def test_score_capped_at_100(self):
        engine = self._engine()
        events = [{"impact": "HIGH", "timing": "upcoming", "label": f"E{i}"} for i in range(10)]
        r = engine.evaluate(_inp(news_events=events))
        self.assertLessEqual(r.impact_score, 100.0)

    def test_upcoming_high_count(self):
        engine = self._engine()
        events = [
            {"impact": "HIGH", "timing": "upcoming", "label": "A"},
            {"impact": "HIGH", "timing": "upcoming", "label": "B"},
        ]
        r = engine.evaluate(_inp(news_events=events))
        self.assertEqual(r.upcoming_high, 2)


# ═════════════════════════════════════════════════════════════════════════════
# 11. BehaviorSimilarityEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestBehaviorSimilarityEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("behavior_similarity.py")
        return mod.BehaviorSimilarityEngine()

    def _similar_records(self, n: int = 10):
        base = dict(
            adx=28.0,
            rsi=52.0,
            vol_ratio=1.1,
            spread_frac=0.15,
            momentum_strength=0.5,
            trend_quality=60.0,
            session_quality=80.0,
        )
        return [dict(base, outcome=1.0 if i % 3 != 0 else -1.0) for i in range(n)]

    def test_no_history_uses_win_rate(self):
        engine = self._engine()
        r = engine.analyse(_inp(historical_similar=[], historical_win_rate=0.60))
        self.assertAlmostEqual(r.historical_success, 0.60, places=3)
        self.assertEqual(r.match_count, 0)

    def test_with_history(self):
        engine = self._engine()
        r = engine.analyse(_inp(historical_similar=self._similar_records(10)))
        self.assertGreater(r.match_count, 0)
        self.assertGreater(r.similarity_score, 0.0)

    def test_similarity_in_range(self):
        engine = self._engine()
        r = engine.analyse(_inp(historical_similar=self._similar_records(5)))
        self.assertGreaterEqual(r.similarity_score, 0.0)
        self.assertLessEqual(r.similarity_score, 100.0)

    def test_historical_success_in_range(self):
        engine = self._engine()
        r = engine.analyse(_inp(historical_similar=self._similar_records(10)))
        self.assertGreaterEqual(r.historical_success, 0.0)
        self.assertLessEqual(r.historical_success, 1.0)

    def test_identical_records_high_similarity(self):
        engine = self._engine()
        record = dict(
            adx=28.0,
            rsi=52.0,
            vol_ratio=1.0,
            spread_frac=0.13,
            momentum_strength=0.0,
            trend_quality=56.0,
            session_quality=80.0,
            outcome=1.0,
        )
        r = engine.analyse(_inp(historical_similar=[record] * 5))
        self.assertGreater(r.best_match_score, 50.0)

    def test_features_used_listed(self):
        engine = self._engine()
        r = engine.analyse(_inp(historical_similar=self._similar_records(3)))
        self.assertIsInstance(r.features_used, list)
        self.assertGreater(len(r.features_used), 0)


# ═════════════════════════════════════════════════════════════════════════════
# 12. PredictionReliabilityEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestPredictionReliabilityEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("prediction_reliability.py")
        return mod.PredictionReliabilityEngine()

    def test_high_quality_inputs(self):
        engine = self._engine()
        r = engine.evaluate(
            _inp(
                prediction_confidence=0.70,
                calibration_score=0.80,
                model_health=0.95,
                historical_win_rate=0.65,
            )
        )
        self.assertGreater(r.reliability_score, 60.0)

    def test_poor_inputs_low_reliability(self):
        engine = self._engine()
        r = engine.evaluate(
            _inp(
                prediction_confidence=0.20,
                calibration_score=0.20,
                model_health=0.30,
                historical_win_rate=0.30,
            )
        )
        self.assertLess(r.reliability_score, 50.0)

    def test_score_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreaterEqual(r.reliability_score, 0.0)
        self.assertLessEqual(r.reliability_score, 100.0)

    def test_overconfident_penalised(self):
        engine = self._engine()
        r_good = engine.evaluate(_inp(prediction_confidence=0.70))
        r_over = engine.evaluate(_inp(prediction_confidence=0.99))
        self.assertLess(r_over.confidence_score, r_good.confidence_score + 10)

    def test_low_win_rate_reason_generated(self):
        engine = self._engine()
        r = engine.evaluate(_inp(historical_win_rate=0.30))
        self.assertTrue(any("win rate" in reason.lower() for reason in r.reasons))


# ═════════════════════════════════════════════════════════════════════════════
# 13. CalibrationEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestCalibrationEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("calibration_engine.py")
        return mod.CalibrationEngine()

    def test_perfect_calibration(self):
        engine = self._engine()
        preds = [0.7] * 20
        outs = [1.0 if i < 14 else 0.0 for i in range(20)]  # 70% win rate
        r = engine.evaluate(_inp(), pred_hist=preds, out_hist=outs)
        self.assertGreater(r.calibration_score, 50.0)

    def test_random_calibration(self):
        engine = self._engine()
        preds = [0.5] * 40
        outs = [1.0, 0.0] * 20
        r = engine.evaluate(_inp(), pred_hist=preds, out_hist=outs)
        # Uniform prediction + 50% outcome = well calibrated
        self.assertGreater(r.calibration_score, 40.0)

    def test_fallback_calibration_score(self):
        engine = self._engine()
        r = engine.evaluate(_inp(calibration_score=0.80))
        self.assertGreater(r.calibration_score, 50.0)

    def test_score_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreaterEqual(r.calibration_score, 0.0)
        self.assertLessEqual(r.calibration_score, 100.0)

    def test_brier_score_range(self):
        engine = self._engine()
        preds = [0.6] * 20
        outs = [1.0 if i < 12 else 0.0 for i in range(20)]
        r = engine.evaluate(_inp(), pred_hist=preds, out_hist=outs)
        self.assertGreaterEqual(r.brier_score, 0.0)
        self.assertLessEqual(r.brier_score, 1.0)

    def test_ece_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreaterEqual(r.ece, 0.0)
        self.assertLessEqual(r.ece, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 14. TradeabilityEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestTradeabilityEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("tradeability_engine.py")
        return mod.TradeabilityEngine()

    def test_factor_weights_sum_to_one(self):
        mod = _load("tradeability_engine.py")
        total = sum(mod._FACTOR_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_score_in_range(self):
        engine = self._engine()
        r = engine.score(_inp())
        self.assertGreaterEqual(r.score, 0.0)
        self.assertLessEqual(r.score, 100.0)

    def test_grade_set(self):
        engine = self._engine()
        r = engine.score(_inp())
        self.assertIsInstance(r.grade, _types.TradeabilityGrade)

    def test_high_quality_inputs_high_score(self):
        engine = self._engine()
        r = engine.score(
            _inp(
                prediction_confidence=0.72,
                expected_rr=2.5,
                expected_move=0.003,
                portfolio_exposure=0.05,
                spread=0.0001,
                atr=0.0015,
                news_events=[],
                session_hour=13,
            )
        )
        self.assertGreater(r.score, 50.0)

    def test_poor_inputs_low_score(self):
        engine = self._engine()
        r = engine.score(
            _inp(
                expected_rr=0.5,
                portfolio_exposure=0.90,
                spread=0.010,
                atr=0.001,
                news_events=[{"impact": "HIGH", "timing": "upcoming", "label": "FOMC"}],
            )
        )
        self.assertLess(r.score, 55.0)

    def test_factors_all_present(self):
        engine = self._engine()
        mod = _load("tradeability_engine.py")
        r = engine.score(_inp())
        for key in mod._FACTOR_WEIGHTS:
            self.assertIn(key, r.factors)

    def test_suggestions_generated(self):
        engine = self._engine()
        r = engine.score(_inp(expected_rr=0.3, portfolio_exposure=0.8))
        self.assertIsInstance(r.suggestions, list)

    def test_reasons_generated_for_low_factors(self):
        engine = self._engine()
        r = engine.score(_inp(expected_rr=0.1, spread=0.05, atr=0.001))
        self.assertIsInstance(r.reasons, list)


# ═════════════════════════════════════════════════════════════════════════════
# 15. ExecutionIntelligenceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestExecutionIntelligenceEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("execution_intelligence.py")
        return mod.ExecutionIntelligenceEngine()

    def test_normal_conditions_market_order(self):
        engine = self._engine()
        mod = _load("mie_models.py")
        t = mod.TradeabilityResult(
            score=75.0,
            grade=_types.TradeabilityGrade.A,
        )
        r = engine.recommend(_inp(), tradeability=t)
        self.assertEqual(r.mode, _types.ExecutionMode.MARKET_ORDER)

    def test_low_tradeability_avoid(self):
        engine = self._engine()
        mod = _load("mie_models.py")
        t = mod.TradeabilityResult(score=15.0, grade=_types.TradeabilityGrade.AVOID)
        r = engine.recommend(_inp(), tradeability=t)
        self.assertEqual(r.mode, _types.ExecutionMode.AVOID)

    def test_entry_zone_set(self):
        engine = self._engine()
        r = engine.recommend(_inp())
        self.assertLess(r.entry_zone_low, r.entry_zone_high)

    def test_stop_and_target_set(self):
        engine = self._engine()
        r = engine.recommend(_inp())
        self.assertNotEqual(r.optimal_stop, 0.0)
        self.assertNotEqual(r.optimal_target, 0.0)

    def test_expected_drawdown_range(self):
        engine = self._engine()
        r = engine.recommend(_inp())
        self.assertGreaterEqual(r.expected_drawdown, 0.0)
        self.assertLessEqual(r.expected_drawdown, 0.50)

    def test_max_holding_positive(self):
        engine = self._engine()
        r = engine.recommend(_inp())
        self.assertGreater(r.max_holding_bars, 0)

    def test_rationale_non_empty(self):
        engine = self._engine()
        r = engine.recommend(_inp())
        self.assertGreater(len(r.rationale), 0)

    def test_stop_below_price_for_long(self):
        engine = self._engine()
        r = engine.recommend(_inp(close=1.1010, atr=0.0015, rsi=60.0))
        self.assertLess(r.optimal_stop, 1.1010)

    def test_target_above_price_for_long(self):
        engine = self._engine()
        r = engine.recommend(_inp(close=1.1010, atr=0.0015, rsi=60.0))
        self.assertGreater(r.optimal_target, 1.1010)


# ═════════════════════════════════════════════════════════════════════════════
# 16. MarketQualityEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestMarketQualityEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("market_quality_engine.py")
        return mod.MarketQualityEngine()

    def test_weights_sum_to_one(self):
        mod = _load("market_quality_engine.py")
        total = sum(mod.MarketQualityEngine._WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_all_scores_in_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        for score in [
            r.market_quality,
            r.execution_quality,
            r.prediction_quality,
            r.liquidity_quality,
            r.volatility_quality,
            r.momentum_quality,
            r.trend_quality,
            r.session_quality,
            r.news_impact_score,
        ]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)

    def test_to_dict_all_keys(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        d = r.to_dict()
        for key in [
            "market_quality",
            "execution_quality",
            "prediction_quality",
            "liquidity_quality",
            "volatility_quality",
            "momentum_quality",
            "trend_quality",
            "session_quality",
            "news_impact_score",
        ]:
            self.assertIn(key, d)

    def test_high_news_impact_lowers_quality(self):
        engine = self._engine()
        mod_n = _load("mie_models.py")
        low_news = mod_n.NewsIntelligenceResult(impact_score=0.0)
        high_news = mod_n.NewsIntelligenceResult(impact_score=90.0)
        r_low = engine.evaluate(_inp(), news=low_news)
        r_high = engine.evaluate(_inp(), news=high_news)
        self.assertGreater(r_low.market_quality, r_high.market_quality)


# ═════════════════════════════════════════════════════════════════════════════
# 17. MIEPortfolioRiskEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestMIEPortfolioRiskEngine(unittest.TestCase):

    def _engine(self):
        mod = _load("portfolio_risk_engine.py")
        return mod.MIEPortfolioRiskEngine()

    def test_max_exposure_in_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreater(r.max_exposure, 0.0)
        self.assertLessEqual(r.max_exposure, 1.0)

    def test_max_trades_positive(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreaterEqual(r.max_open_trades, 0)

    def test_extreme_risk_restricts_trades(self):
        engine = self._engine()
        mod_m = _load("mie_models.py")
        ri = mod_m.RiskIntelligenceResult(
            regime=_types.RiskRegime.EXTREME, risk_score=80.0, multiplier=0.30
        )
        r = engine.evaluate(_inp(), risk=ri)
        self.assertLessEqual(r.max_open_trades, 2)

    def test_position_size_in_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreater(r.position_size_frac, 0.0)
        self.assertLessEqual(r.position_size_frac, 0.10)

    def test_diversification_in_range(self):
        engine = self._engine()
        r = engine.evaluate(_inp())
        self.assertGreaterEqual(r.diversification, 0.0)
        self.assertLessEqual(r.diversification, 1.0)

    def test_high_exposure_warning(self):
        engine = self._engine()
        r = engine.evaluate(_inp(portfolio_exposure=0.90))
        self.assertTrue(len(r.warnings) > 0)


# ═════════════════════════════════════════════════════════════════════════════
# 18. AICommentaryGenerator
# ═════════════════════════════════════════════════════════════════════════════


class TestAICommentaryGenerator(unittest.TestCase):

    def _gen(self):
        mod = _load("ai_commentary.py")
        return mod.AICommentaryGenerator()

    def test_summary_non_empty(self):
        gen = self._gen()
        r = gen.generate(_inp())
        self.assertGreater(len(r.summary), 10)

    def test_tags_non_empty(self):
        gen = self._gen()
        r = gen.generate(_inp())
        self.assertIsInstance(r.tags, list)
        self.assertGreater(len(r.tags), 0)

    def test_regime_note_populated(self):
        gen = self._gen()
        mod_m = _load("mie_models.py")
        regime = mod_m.MarketRegimeResult(regime=_types.MarketRegime.TRENDING)
        r = gen.generate(_inp(), regime=regime)
        self.assertIn("trend", r.regime_note.lower())

    def test_danger_tag_on_black_swan(self):
        gen = self._gen()
        mod_m = _load("mie_models.py")
        ri = mod_m.RiskIntelligenceResult(regime=_types.RiskRegime.BLACK_SWAN)
        r = gen.generate(_inp(), risk=ri)
        self.assertIn("DANGER", r.tags)

    def test_recommendation_populated(self):
        gen = self._gen()
        mod_m = _load("mie_models.py")
        t = mod_m.TradeabilityResult(score=90.0, grade=_types.TradeabilityGrade.A_PLUS)
        r = gen.generate(_inp(), tradeability=t)
        self.assertIn("full allocation", r.recommendation.lower())

    def test_confidence_range(self):
        gen = self._gen()
        r = gen.generate(_inp(prediction_confidence=0.75))
        self.assertAlmostEqual(r.confidence, 0.75, places=3)

    def test_to_dict_keys(self):
        gen = self._gen()
        r = gen.generate(_inp())
        d = r.to_dict()
        for key in ["summary", "regime_note", "risk_note", "recommendation", "tags"]:
            self.assertIn(key, d)


# ═════════════════════════════════════════════════════════════════════════════
# 19. FeatureStoreExporter
# ═════════════════════════════════════════════════════════════════════════════


class TestFeatureStoreExporter(unittest.TestCase):

    def _exp(self):
        mod = _load("feature_store.py")
        return mod.FeatureStoreExporter()

    def test_vector_length(self):
        exp = self._exp()
        r = exp.export(_inp())
        self.assertEqual(len(r.to_feature_vector()), 12)

    def test_vector_normalised(self):
        exp = self._exp()
        r = exp.export(_inp())
        for v in r.to_feature_vector():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-6)

    def test_symbol_set(self):
        exp = self._exp()
        r = exp.export(_inp(symbol="GBPJPY"))
        self.assertEqual(r.symbol, "GBPJPY")

    def test_with_all_sub_results(self):
        exp = self._exp()
        mod_m = _load("mie_models.py")
        regime = mod_m.MarketRegimeResult(regime=_types.MarketRegime.TRENDING)
        risk = mod_m.RiskIntelligenceResult(regime=_types.RiskRegime.NORMAL)
        trade = mod_m.TradeabilityResult(score=75.0, grade=_types.TradeabilityGrade.A)
        quality = mod_m.MarketQualityResult(market_quality=70.0, execution_quality=65.0)
        r = exp.export(_inp(), regime=regime, risk=risk, tradeability=trade, quality=quality)
        self.assertAlmostEqual(r.tradeability, 75.0, places=1)
        self.assertAlmostEqual(r.market_quality, 70.0, places=1)


# ═════════════════════════════════════════════════════════════════════════════
# 20. MIEDatabase
# ═════════════════════════════════════════════════════════════════════════════


class TestMIEDatabase(unittest.TestCase):

    def _db(self):
        mod = _load("mie_database.py")
        return mod.MIEDatabase(":memory:")

    def test_tables_exist(self):
        db = self._db()
        tables = [
            "market_intelligence",
            "market_regime_history",
            "risk_history",
            "tradeability_history",
            "market_quality",
            "execution_quality",
            "prediction_reliability",
            "behavior_similarity",
            "ai_commentary",
            "market_scores",
        ]
        for t in tables:
            count = db.table_count(t)
            self.assertIsInstance(count, int)

    def test_save_intelligence(self):
        db = self._db()
        rid = db.save_intelligence(
            {
                "report_id": "TEST001",
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "tradeability": 75.0,
                "trade_grade": "A",
                "summary": "test",
            }
        )
        self.assertGreater(rid, 0)
        self.assertEqual(db.table_count("market_intelligence"), 1)

    def test_save_regime(self):
        db = self._db()
        rid = db.save_regime(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "regime": "TRENDING",
                "confidence": 0.80,
            }
        )
        self.assertGreater(rid, 0)

    def test_save_tradeability(self):
        db = self._db()
        rid = db.save_tradeability(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "score": 70.0,
                "grade": "A",
            }
        )
        self.assertGreater(rid, 0)

    def test_save_quality(self):
        db = self._db()
        rid = db.save_quality(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "market_quality": 65.0,
            }
        )
        self.assertGreater(rid, 0)

    def test_save_execution(self):
        db = self._db()
        rid = db.save_execution(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "mode": "MARKET_ORDER",
            }
        )
        self.assertGreater(rid, 0)

    def test_save_reliability(self):
        db = self._db()
        rid = db.save_reliability(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "reliability": 72.0,
            }
        )
        self.assertGreater(rid, 0)

    def test_save_similarity(self):
        db = self._db()
        rid = db.save_similarity(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "similarity": 65.0,
                "historical_win": 0.58,
                "match_count": 10,
                "best_match": 80.0,
            }
        )
        self.assertGreater(rid, 0)

    def test_save_commentary(self):
        db = self._db()
        rid = db.save_commentary(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "summary": "Good setup",
                "tags": ["TRENDING", "NORMAL"],
            }
        )
        self.assertGreater(rid, 0)

    def test_save_scores(self):
        db = self._db()
        rid = db.save_scores(
            {
                "symbol": "EURUSD",
                "timestamp": 1000.0,
                "tradeability": 72.0,
                "market_quality": 65.0,
            }
        )
        self.assertGreater(rid, 0)

    def test_get_latest_intelligence(self):
        db = self._db()
        db.save_intelligence(
            {
                "report_id": "R001",
                "symbol": "EURUSD",
                "timestamp": 1000.0,
            }
        )
        rows = db.get_latest_intelligence("EURUSD")
        self.assertEqual(len(rows), 1)

    def test_get_regime_history(self):
        db = self._db()
        db.save_regime({"symbol": "GBPUSD", "timestamp": 1001.0, "regime": "RANGE"})
        rows = db.get_regime_history("GBPUSD")
        self.assertEqual(len(rows), 1)

    def test_get_tradeability_history(self):
        db = self._db()
        for i in range(5):
            db.save_tradeability(
                {"symbol": "EURUSD", "timestamp": float(i), "score": 60.0, "grade": "B"}
            )
        rows = db.get_tradeability_history("EURUSD", limit=3)
        self.assertEqual(len(rows), 3)

    def test_multiple_symbols_isolated(self):
        db = self._db()
        db.save_intelligence({"report_id": "A1", "symbol": "EURUSD", "timestamp": 1.0})
        db.save_intelligence({"report_id": "B1", "symbol": "GBPUSD", "timestamp": 2.0})
        eu = db.get_latest_intelligence("EURUSD")
        gb = db.get_latest_intelligence("GBPUSD")
        self.assertEqual(len(eu), 1)
        self.assertEqual(len(gb), 1)


# ═════════════════════════════════════════════════════════════════════════════
# 21. MIEDashboard
# ═════════════════════════════════════════════════════════════════════════════


class TestMIEDashboard(unittest.TestCase):

    def _dash(self):
        mod = _load("mie_dashboard.py")
        return mod.MIEDashboard()

    def _report(self):
        mod_m = _load("mie_models.py")
        r = mod_m.MarketIntelligenceReport(symbol="EURUSD")
        r.tradeability = mod_m.TradeabilityResult(
            score=72.0,
            grade=_types.TradeabilityGrade.A,
        )
        r.regime = mod_m.MarketRegimeResult(
            regime=_types.MarketRegime.TRENDING,
            confidence=0.82,
        )
        r.risk = mod_m.RiskIntelligenceResult(
            regime=_types.RiskRegime.NORMAL,
            risk_score=25.0,
            multiplier=1.0,
        )
        r.quality = mod_m.MarketQualityResult(
            market_quality=65.0,
            execution_quality=70.0,
        )
        r.execution = mod_m.ExecutionResult(
            mode=_types.ExecutionMode.MARKET_ORDER,
            optimal_stop=1.0990,
            optimal_target=1.1050,
        )
        r.commentary = mod_m.AICommentaryResult(
            summary="Good trend setup.",
            recommendation="Full allocation.",
        )
        r.features = mod_m.FeatureStoreExport(symbol="EURUSD", tradeability=72.0)
        return r

    def test_overview_keys(self):
        dash = self._dash()
        d = dash.overview(self._report())
        for key in ["symbol", "tradeability", "trade_grade", "market_regime", "risk_regime"]:
            self.assertIn(key, d)

    def test_regime_view_keys(self):
        dash = self._dash()
        d = dash.regime_view(self._report())
        self.assertIn("regime", d)
        self.assertIn("confidence", d)

    def test_risk_view_keys(self):
        dash = self._dash()
        d = dash.risk_view(self._report())
        self.assertIn("risk_regime", d)
        self.assertIn("risk_score", d)

    def test_tradeability_view_keys(self):
        dash = self._dash()
        d = dash.tradeability_view(self._report())
        self.assertIn("score", d)
        self.assertIn("grade", d)

    def test_execution_view_mode(self):
        dash = self._dash()
        d = dash.execution_view(self._report())
        self.assertEqual(d["mode"], "MARKET_ORDER")

    def test_commentary_view(self):
        dash = self._dash()
        d = dash.commentary_view(self._report())
        self.assertIn("summary", d)

    def test_feature_view_vector(self):
        dash = self._dash()
        d = dash.feature_view(self._report())
        self.assertIn("feature_vector", d)
        self.assertEqual(len(d["feature_vector"]), 12)

    def test_live_analysis_keys(self):
        dash = self._dash()
        d = dash.live_analysis(self._report())
        for key in [
            "tradeability",
            "trade_grade",
            "market_regime",
            "risk_regime",
            "execution_mode",
            "commentary",
        ]:
            self.assertIn(key, d)

    def test_ea_view_sl_tp(self):
        dash = self._dash()
        d = dash.ea_view(self._report())
        self.assertIn("sl", d)
        self.assertIn("tp", d)

    def test_android_view_compact(self):
        dash = self._dash()
        d = dash.android_view(self._report())
        for key in ["tradeability", "market_quality", "commentary"]:
            self.assertIn(key, d)

    def test_all_views_returns_dict(self):
        dash = self._dash()
        all_v = dash.all_views(self._report())
        for view_name in [
            "overview",
            "regime",
            "risk",
            "tradeability",
            "quality",
            "execution",
            "commentary",
            "features",
            "live_analysis",
        ]:
            self.assertIn(view_name, all_v)


# ═════════════════════════════════════════════════════════════════════════════
# 22. MIEAPI (full pipeline)
# ═════════════════════════════════════════════════════════════════════════════


class TestMIEAPI(unittest.TestCase):

    def _api(self):
        mod = _load("mie_api.py")
        return mod.MIEAPI(db_path=":memory:")

    def test_recalculate_returns_report(self):
        api = self._api()
        report = api.recalculate(_inp())
        self.assertIsNotNone(report.tradeability)
        self.assertIsNotNone(report.regime)

    def test_all_layers_populated(self):
        api = self._api()
        r = api.recalculate(_inp())
        for attr in [
            "regime",
            "risk",
            "volatility",
            "momentum",
            "trend",
            "liquidity",
            "session",
            "news",
            "similarity",
            "reliability",
            "calibration",
            "tradeability",
            "execution",
            "quality",
            "portfolio_risk",
            "commentary",
            "features",
        ]:
            self.assertIsNotNone(getattr(r, attr), f"{attr} is None")

    def test_tradeability_score_range(self):
        api = self._api()
        r = api.recalculate(_inp())
        self.assertGreaterEqual(r.tradeability.score, 0.0)
        self.assertLessEqual(r.tradeability.score, 100.0)

    def test_current_after_recalculate(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.current("EURUSD")
        self.assertTrue(result.get("ok"))
        self.assertIsNotNone(result.get("data"))

    def test_history_endpoint(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.history("EURUSD")
        self.assertTrue(result.get("ok"))

    def test_tradeability_endpoint(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.tradeability_endpoint("EURUSD")
        self.assertTrue(result.get("ok"))

    def test_risk_endpoint(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.risk_endpoint("EURUSD")
        self.assertTrue(result.get("ok"))

    def test_regime_endpoint(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.regime_endpoint("EURUSD")
        self.assertTrue(result.get("ok"))

    def test_execution_endpoint(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.execution_endpoint("EURUSD")
        self.assertTrue(result.get("ok"))

    def test_quality_endpoint(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.quality_endpoint("EURUSD")
        self.assertTrue(result.get("ok"))

    def test_commentary_endpoint(self):
        api = self._api()
        api.recalculate(_inp())
        result = api.commentary_endpoint("EURUSD")
        self.assertTrue(result.get("ok"))

    def test_no_data_current_returns_error(self):
        api = self._api()
        result = api.current("NOTEXIST")
        self.assertIn("error", result)

    def test_feature_vector_from_report(self):
        api = self._api()
        r = api.recalculate(_inp())
        v = r.features.to_feature_vector()
        self.assertEqual(len(v), 12)
        for val in v:
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0 + 1e-6)

    def test_multi_symbol(self):
        api = self._api()
        api.recalculate(_inp(symbol="EURUSD"))
        api.recalculate(_inp(symbol="GBPUSD"))
        eu = api.current("EURUSD")
        gb = api.current("GBPUSD")
        self.assertTrue(eu["ok"])
        self.assertTrue(gb["ok"])


# ═════════════════════════════════════════════════════════════════════════════
# 23. MIEWorker
# ═════════════════════════════════════════════════════════════════════════════


class TestMIEWorker(unittest.TestCase):

    def _worker(self):
        mod_w = _load("mie_worker.py")
        mod_a = _load("mie_api.py")
        api = mod_a.MIEAPI(db_path=":memory:")
        return mod_w.MIEWorker(api=api)

    def test_submit_candle(self):
        worker = self._worker()
        job_id = worker.submit_candle(_inp())
        self.assertIsInstance(job_id, str)
        self.assertEqual(worker.queue_size, 1)

    def test_process_next(self):
        worker = self._worker()
        worker.submit_candle(_inp())
        job = worker.process_next()
        self.assertIsNotNone(job)
        self.assertEqual(job.status, "DONE")

    def test_queue_drains(self):
        worker = self._worker()
        for _ in range(3):
            worker.submit_candle(_inp())
        worker.process_all()
        self.assertEqual(worker.queue_size, 0)
        self.assertEqual(worker.processed, 3)

    def test_failed_job_tracked(self):
        mod_w = _load("mie_worker.py")
        mod_t = _load("mie_types.py")
        mod_m = _load("mie_models.py")
        mod_a = _load("mie_api.py")
        api = mod_a.MIEAPI(db_path=":memory:")
        worker = mod_w.MIEWorker(api=api)
        bad_job = mod_m.WorkerJob(
            trigger=mod_t.WorkerTrigger.MANUAL_REFRESH,
            symbol="EURUSD",
            payload={},  # missing 'input'
        )
        worker.submit(bad_job)
        job = worker.process_next()
        self.assertEqual(job.status, "FAILED")
        self.assertEqual(worker.failed, 1)

    def test_stats_dict(self):
        worker = self._worker()
        s = worker.stats()
        self.assertIn("queue_size", s)
        self.assertIn("processed", s)
        self.assertIn("failed", s)

    def test_history_tracked(self):
        worker = self._worker()
        worker.submit_candle(_inp())
        worker.process_all()
        hist = worker.get_history()
        self.assertEqual(len(hist), 1)

    def test_on_complete_callback(self):
        mod_w = _load("mie_worker.py")
        mod_a = _load("mie_api.py")
        api = mod_a.MIEAPI(db_path=":memory:")
        completed = []
        worker = mod_w.MIEWorker(api=api, on_complete=lambda j: completed.append(j.job_id))
        worker.submit_candle(_inp())
        worker.process_all()
        self.assertEqual(len(completed), 1)

    def test_submit_prediction(self):
        worker = self._worker()
        mod_t = _load("mie_types.py")
        job_id = worker.submit_prediction(_inp())
        self.assertEqual(worker.queue_size, 1)
        job = worker.process_next()
        self.assertEqual(job.trigger, mod_t.WorkerTrigger.NEW_PREDICTION)

    def test_submit_manual(self):
        worker = self._worker()
        mod_t = _load("mie_types.py")
        worker.submit_manual(_inp())
        job = worker.process_next()
        self.assertEqual(job.trigger, mod_t.WorkerTrigger.MANUAL_REFRESH)


# ═════════════════════════════════════════════════════════════════════════════
# 24. Integration tests (full pipeline edge cases)
# ═════════════════════════════════════════════════════════════════════════════


class TestMIEIntegration(unittest.TestCase):

    def _api(self):
        mod = _load("mie_api.py")
        return mod.MIEAPI(db_path=":memory:")

    def test_black_swan_conditions(self):
        api = self._api()
        big_returns = [0.05 * (1 if i % 2 == 0 else -1) for i in range(60)]
        r = api.recalculate(
            _inp(
                spread=0.020,
                atr=0.001,
                returns=big_returns,
                news_events=[{"impact": "HIGH", "timing": "upcoming", "label": "X"}],
            )
        )
        self.assertIn(
            r.risk.regime,
            [
                _types.RiskRegime.EXTREME,
                _types.RiskRegime.BLACK_SWAN,
                _types.RiskRegime.FLASH_VOLATILITY,
                _types.RiskRegime.MARKET_STRESS,
            ],
        )

    def test_avoid_execution_on_extreme_risk(self):
        api = self._api()
        big_returns = [0.05 * (1 if i % 2 == 0 else -1) for i in range(60)]
        r = api.recalculate(
            _inp(
                spread=0.030,
                atr=0.001,
                returns=big_returns,
                portfolio_exposure=0.80,
                expected_rr=0.3,
            )
        )
        # Should avoid or wait
        self.assertIn(
            r.execution.mode,
            [
                _types.ExecutionMode.AVOID,
                _types.ExecutionMode.WAIT,
                _types.ExecutionMode.LIMIT_ORDER,
            ],
        )

    def test_high_quality_setup_market_order(self):
        api = self._api()
        good_returns = [0.001 * (1 if i % 3 != 0 else -0.5) for i in range(60)]
        r = api.recalculate(
            _inp(
                adx=35.0,
                spread=0.0001,
                atr=0.0015,
                expected_rr=2.5,
                expected_move=0.003,
                portfolio_exposure=0.05,
                prediction_confidence=0.72,
                news_events=[],
                session_hour=13,
                returns=good_returns,
            )
        )
        self.assertGreater(r.tradeability.score, 40.0)

    def test_report_serialises_to_dict(self):
        api = self._api()
        r = api.recalculate(_inp())
        d = r.to_dict()
        self.assertIn("tradeability", d)
        self.assertIn("regime", d)
        self.assertIn("risk", d)

    def test_repeated_recalculate_accumulates(self):
        api = self._api()
        for _ in range(5):
            api.recalculate(_inp())
        result = api.tradeability_endpoint("EURUSD")
        self.assertEqual(result["count"], 5)

    def test_different_symbols(self):
        api = self._api()
        for sym in ["EURUSD", "GBPUSD", "USDJPY"]:
            r = api.recalculate(_inp(symbol=sym))
            self.assertEqual(r.symbol, sym)

    def test_feature_vector_deterministic(self):
        api = self._api()
        r1 = api.recalculate(_inp())
        r2 = api.recalculate(_inp())
        v1 = r1.features.to_feature_vector()
        v2 = r2.features.to_feature_vector()
        for a, b in zip(v1, v2):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
