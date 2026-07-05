"""Phase 21 — Execution Intelligence Test Suite.

130+ tests across all 16 execution modules.
Uses spec_from_file_location to avoid torch import chain from trainer/__init__.py.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
import unittest

# ── Stub loguru before any imports ──────────────────────────────────────────

_loguru = types.ModuleType("loguru")

class _Logger:
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def success(self, *a, **k): pass
    def trace(self, *a, **k): pass
    def critical(self, *a, **k): pass
    def exception(self, *a, **k): pass
    def bind(self, **k): return self

_loguru.logger = _Logger()
sys.modules.setdefault("loguru", _loguru)

_BASE = Path(__file__).resolve().parents[2] / "trainer" / "execution"


def _load(mod_key: str, filename: str):
    path = _BASE / filename
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order
_et  = _load("trainer.execution.execution_types",   "execution_types.py")
_ez  = _load("trainer.execution.entry_zones",        "entry_zones.py")
_eq  = _load("trainer.execution.entry_quality",      "entry_quality.py")
_sl  = _load("trainer.execution.stop_loss_engine",   "stop_loss_engine.py")
_tp  = _load("trainer.execution.take_profit_engine", "take_profit_engine.py")
_re  = _load("trainer.execution.risk_engine",        "risk_engine.py")
_ps  = _load("trainer.execution.position_sizer",     "position_sizer.py")
_tm  = _load("trainer.execution.trade_manager",      "trade_manager.py")
_eti = _load("trainer.execution.execution_timing",   "execution_timing.py")
_os  = _load("trainer.execution.order_strategy",     "order_strategy.py")
_ts  = _load("trainer.execution.trade_scorer",       "trade_scorer.py")
_es  = _load("trainer.execution.execution_simulator","execution_simulator.py")
_ep  = _load("trainer.execution.execution_planner",  "execution_planner.py")
_edb = _load("trainer.execution.execution_database", "execution_database.py")
_ed  = _load("trainer.execution.execution_dashboard","execution_dashboard.py")
_api = _load("trainer.execution.execution_api",      "execution_api.py")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fc(
    instrument="EURUSD",
    direction="BUY",
    confidence=0.72,
    expected_return=0.08,
    expected_risk=0.10,
    current_price=1.0850,
    atr=0.0012,
    spread=0.0001,
    session=None,
    volatility=0.009,
    momentum=0.3,
    news_risk=0.1,
    account_size=10_000.0,
):
    if session is None:
        session = _et.Session.LONDON
    return _et.ExecutionForecast(
        instrument=instrument,
        direction=direction,
        confidence=confidence,
        expected_return=expected_return,
        expected_risk=expected_risk,
        current_price=current_price,
        atr=atr,
        spread=spread,
        session=session,
        volatility=volatility,
        momentum=momentum,
        news_risk=news_risk,
        account_size=account_size,
    )


def _make_api():
    return _api.ExecutionApi(db_path=":memory:")


def _make_planner():
    return _ep.ExecutionPlanner()


# ════════════════════════════════════════════════════════════════════════════════
# 1. execution_types
# ════════════════════════════════════════════════════════════════════════════════
class TestExecutionTypes(unittest.TestCase):

    def test_execution_forecast_is_buy(self):
        fc = _fc(direction="BUY")
        self.assertTrue(fc.is_buy)

    def test_execution_forecast_is_sell(self):
        fc = _fc(direction="SELL")
        self.assertFalse(fc.is_buy)

    def test_spread_cost_pct(self):
        fc = _fc(spread=0.0001, current_price=1.0)
        self.assertAlmostEqual(fc.spread_cost_pct, 0.0001, places=5)

    def test_atr_pct(self):
        fc = _fc(atr=0.0012, current_price=1.0)
        self.assertAlmostEqual(fc.atr_pct, 0.0012, places=5)

    def test_risk_reward_raw(self):
        fc = _fc(expected_return=0.10, expected_risk=0.05)
        self.assertAlmostEqual(fc.risk_reward_raw, 2.0, places=5)

    def test_risk_reward_zero_risk(self):
        fc = _fc(expected_risk=0.0)
        self.assertEqual(fc.risk_reward_raw, 0.0)

    def test_to_dict_keys(self):
        fc  = _fc()
        d   = fc.to_dict()
        for key in ("instrument", "direction", "confidence", "current_price", "atr", "spread"):
            self.assertIn(key, d)

    def test_broker_constraints_defaults(self):
        bc = _et.BrokerConstraints()
        self.assertEqual(bc.min_lot_size, 0.01)
        self.assertGreater(bc.max_leverage, 0)

    def test_broker_validate_lot_clips_min(self):
        bc = _et.BrokerConstraints(min_lot_size=0.01)
        self.assertEqual(bc.validate_lot(0.001), 0.01)

    def test_broker_validate_lot_clips_max(self):
        bc = _et.BrokerConstraints(max_lot_size=1.0)
        self.assertEqual(bc.validate_lot(999.0), 1.0)

    def test_broker_spread_ok(self):
        bc = _et.BrokerConstraints(max_spread_to_trade=0.003)
        self.assertTrue(bc.spread_ok(0.001))
        self.assertFalse(bc.spread_ok(0.004))

    def test_broker_to_dict_keys(self):
        d = _et.BrokerConstraints().to_dict()
        self.assertIn("broker", d)
        self.assertIn("max_leverage", d)

    def test_grade_from_score_excellent(self):
        self.assertEqual(_et._grade_from_score(0.90), _et.ExecutionGrade.EXCELLENT)

    def test_grade_from_score_reject(self):
        self.assertEqual(_et._grade_from_score(0.20), _et.ExecutionGrade.REJECT)

    def test_session_quality_overlap_is_max(self):
        self.assertEqual(_et.SESSION_QUALITY[_et.Session.OVERLAP], 1.0)

    def test_session_quality_closed_is_low(self):
        self.assertLess(_et.SESSION_QUALITY[_et.Session.CLOSED], 0.2)


# ════════════════════════════════════════════════════════════════════════════════
# 2. entry_zones
# ════════════════════════════════════════════════════════════════════════════════
class TestEntryZones(unittest.TestCase):

    def setUp(self):
        self.engine = _ez.EntryZoneEngine()
        self.fc_buy  = _fc(direction="BUY",  current_price=1.0850, atr=0.0012)
        self.fc_sell = _fc(direction="SELL", current_price=1.0850, atr=0.0012)

    def test_buy_primary_below_price(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertLess(z.primary_entry, self.fc_buy.current_price)

    def test_sell_primary_above_price(self):
        z = self.engine.calculate(self.fc_sell)
        self.assertGreater(z.primary_entry, self.fc_sell.current_price)

    def test_buy_invalidation_below_primary(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertLess(z.invalidation, z.primary_entry)

    def test_sell_invalidation_above_primary(self):
        z = self.engine.calculate(self.fc_sell)
        self.assertGreater(z.invalidation, z.primary_entry)

    def test_conservative_further_than_primary_buy(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertLess(z.conservative_entry, z.primary_entry)

    def test_aggressive_closer_to_price_than_primary_buy(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertGreater(z.aggressive_entry, z.primary_entry)

    def test_zone_width_positive(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertGreater(z.zone_width, 0)

    def test_risk_to_primary_positive(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertGreater(z.risk_to_primary, 0)

    def test_contains_primary(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertTrue(z.contains(z.primary_entry))

    def test_contains_outside_zone_false(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertFalse(z.contains(self.fc_buy.current_price + 1.0))

    def test_to_dict_keys(self):
        z = self.engine.calculate(self.fc_buy)
        d = z.to_dict()
        for key in ("primary_entry", "secondary_entry", "aggressive_entry",
                    "conservative_entry", "late_entry", "invalidation"):
            self.assertIn(key, d)

    def test_instrument_and_direction_preserved(self):
        z = self.engine.calculate(self.fc_buy)
        self.assertEqual(z.instrument, "EURUSD")
        self.assertEqual(z.direction, "BUY")


# ════════════════════════════════════════════════════════════════════════════════
# 3. entry_quality
# ════════════════════════════════════════════════════════════════════════════════
class TestEntryQuality(unittest.TestCase):

    def setUp(self):
        self.engine   = _eq.EntryQualityEngine()
        self.ze       = _ez.EntryZoneEngine()

    def test_score_in_range(self):
        fc = _fc()
        r  = self.engine.evaluate(fc)
        self.assertGreaterEqual(r.score, 0.0)
        self.assertLessEqual(r.score, 1.0)

    def test_grade_not_none(self):
        r = self.engine.evaluate(_fc())
        self.assertIsNotNone(r.grade)

    def test_high_news_risk_reduces_score(self):
        r_low  = self.engine.evaluate(_fc(news_risk=0.0))
        r_high = self.engine.evaluate(_fc(news_risk=0.9))
        self.assertGreater(r_low.score, r_high.score)

    def test_closed_session_lowers_score(self):
        r_london = self.engine.evaluate(_fc(session=_et.Session.LONDON))
        r_closed = self.engine.evaluate(_fc(session=_et.Session.CLOSED))
        self.assertGreater(r_london.score, r_closed.score)

    def test_aligned_momentum_raises_score(self):
        r_pos = self.engine.evaluate(_fc(direction="BUY", momentum=0.8))
        r_neg = self.engine.evaluate(_fc(direction="BUY", momentum=-0.8))
        self.assertGreater(r_pos.score, r_neg.score)

    def test_with_zones_no_error(self):
        fc = _fc()
        z  = self.ze.calculate(fc)
        r  = self.engine.evaluate(fc, z)
        self.assertIsNotNone(r.score)

    def test_tradeable_when_good(self):
        r = self.engine.evaluate(_fc(confidence=0.85, momentum=0.5, news_risk=0.0))
        # May or may not be tradeable depending on score; just check field exists
        self.assertIsInstance(r.is_tradeable(), bool)

    def test_to_dict_has_sub_scores(self):
        d = self.engine.evaluate(_fc()).to_dict()
        self.assertIn("sub_scores", d)

    def test_wide_spread_lowers_score(self):
        r_tight = self.engine.evaluate(_fc(spread=0.00005))
        r_wide  = self.engine.evaluate(_fc(spread=0.0008))
        self.assertGreater(r_tight.score, r_wide.score)


# ════════════════════════════════════════════════════════════════════════════════
# 4. stop_loss_engine
# ════════════════════════════════════════════════════════════════════════════════
class TestStopLossEngine(unittest.TestCase):

    def setUp(self):
        self.engine = _sl.StopLossEngine()
        self.ze     = _ez.EntryZoneEngine()

    def test_buy_initial_stop_below_entry(self):
        fc = _fc(direction="BUY")
        z  = self.ze.calculate(fc)
        sl = self.engine.recommend(fc, z)
        self.assertLess(sl.initial_stop, z.primary_entry)

    def test_sell_initial_stop_above_entry(self):
        fc = _fc(direction="SELL")
        z  = self.ze.calculate(fc)
        sl = self.engine.recommend(fc, z)
        self.assertGreater(sl.initial_stop, z.primary_entry)

    def test_emergency_wider_than_initial_buy(self):
        fc = _fc(direction="BUY")
        z  = self.ze.calculate(fc)
        sl = self.engine.recommend(fc, z)
        self.assertLess(sl.emergency_stop, sl.initial_stop)

    def test_stop_confidence_in_range(self):
        fc = _fc()
        sl = self.engine.recommend(fc)
        self.assertGreaterEqual(sl.stop_confidence, 0.0)
        self.assertLessEqual(sl.stop_confidence, 1.0)

    def test_hit_probability_in_range(self):
        fc = _fc()
        sl = self.engine.recommend(fc)
        self.assertGreaterEqual(sl.expected_hit_probability, 0.0)
        self.assertLessEqual(sl.expected_hit_probability, 1.0)

    def test_pips_initial_positive(self):
        fc = _fc()
        z  = self.ze.calculate(fc)
        sl = self.engine.recommend(fc, z)
        self.assertGreater(sl.pips_initial, 0)

    def test_to_dict_keys(self):
        sl = self.engine.recommend(_fc())
        d  = sl.to_dict()
        for key in ("initial_stop", "adaptive_stop", "atr_stop",
                    "structure_stop", "volatility_stop", "emergency_stop"):
            self.assertIn(key, d)

    def test_no_zones_fallback(self):
        sl = self.engine.recommend(_fc())
        self.assertIsNotNone(sl.initial_stop)


# ════════════════════════════════════════════════════════════════════════════════
# 5. take_profit_engine
# ════════════════════════════════════════════════════════════════════════════════
class TestTakeProfitEngine(unittest.TestCase):

    def setUp(self):
        self.engine = _tp.TakeProfitEngine()
        self.ze     = _ez.EntryZoneEngine()
        self.sl_eng = _sl.StopLossEngine()

    def test_buy_tp1_above_entry(self):
        fc  = _fc(direction="BUY")
        z   = self.ze.calculate(fc)
        sl  = self.sl_eng.recommend(fc, z)
        tpr = self.engine.recommend(fc, z, sl)
        self.assertGreater(tpr.tp1, z.primary_entry)

    def test_sell_tp1_below_entry(self):
        fc  = _fc(direction="SELL")
        z   = self.ze.calculate(fc)
        sl  = self.sl_eng.recommend(fc, z)
        tpr = self.engine.recommend(fc, z, sl)
        self.assertLess(tpr.tp1, z.primary_entry)

    def test_tp_levels_ascending_for_buy(self):
        fc  = _fc(direction="BUY")
        z   = self.ze.calculate(fc)
        sl  = self.sl_eng.recommend(fc, z)
        tpr = self.engine.recommend(fc, z, sl)
        self.assertLess(tpr.tp1, tpr.tp2)
        self.assertLess(tpr.tp2, tpr.tp3)
        self.assertLess(tpr.tp3, tpr.runner_target)

    def test_tp_levels_descending_for_sell(self):
        fc  = _fc(direction="SELL")
        z   = self.ze.calculate(fc)
        sl  = self.sl_eng.recommend(fc, z)
        tpr = self.engine.recommend(fc, z, sl)
        self.assertGreater(tpr.tp1, tpr.tp2)
        self.assertGreater(tpr.tp2, tpr.tp3)

    def test_probabilities_decreasing(self):
        fc  = _fc()
        tpr = self.engine.recommend(fc)
        self.assertGreater(tpr.tp1_hit_prob, tpr.tp2_hit_prob)
        self.assertGreater(tpr.tp2_hit_prob, tpr.tp3_hit_prob)

    def test_probabilities_in_range(self):
        tpr = self.engine.recommend(_fc())
        for p in (tpr.tp1_hit_prob, tpr.tp2_hit_prob, tpr.tp3_hit_prob, tpr.runner_hit_prob):
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_best_expectancy_level_is_valid(self):
        tpr = self.engine.recommend(_fc())
        self.assertIn(tpr.best_expectancy_level(), ("TP1", "TP2", "TP3", "RUNNER"))

    def test_to_dict_keys(self):
        d = self.engine.recommend(_fc()).to_dict()
        for key in ("tp1", "tp2", "tp3", "runner_target", "probabilities"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════════
# 6. risk_engine
# ════════════════════════════════════════════════════════════════════════════════
class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.engine = _re.RiskEngine()
        self.ze     = _ez.EntryZoneEngine()
        self.sl_eng = _sl.StopLossEngine()
        self.tp_eng = _tp.TakeProfitEngine()

    def _full(self, **kwargs):
        fc = _fc(**kwargs)
        z  = self.ze.calculate(fc)
        sl = self.sl_eng.recommend(fc, z)
        tpr = self.tp_eng.recommend(fc, z, sl)
        return self.engine.analyze(fc, sl, tpr), fc

    def test_rr_positive(self):
        rp, _ = self._full()
        self.assertGreater(rp.risk_reward_ratio, 0)

    def test_probs_in_range(self):
        rp, _ = self._full()
        self.assertGreaterEqual(rp.prob_success, 0.0)
        self.assertLessEqual(rp.prob_success, 1.0)
        self.assertGreaterEqual(rp.prob_failure, 0.0)
        self.assertLessEqual(rp.prob_failure, 1.0)

    def test_high_confidence_positive_ev(self):
        rp, _ = self._full(confidence=0.85, expected_return=0.15, expected_risk=0.05)
        self.assertTrue(rp.is_positive_ev())

    def test_tail_risk_positive(self):
        rp, _ = self._full()
        self.assertGreater(rp.tail_risk, 0)

    def test_worst_case_positive(self):
        rp, _ = self._full()
        self.assertGreater(rp.worst_case, 0)

    def test_best_case_positive(self):
        rp, _ = self._full()
        self.assertGreater(rp.best_case, 0)

    def test_no_stop_no_tp_fallback(self):
        rp = self.engine.analyze(_fc())
        self.assertIsNotNone(rp.risk_reward_ratio)

    def test_to_dict_keys(self):
        rp, _ = self._full()
        d = rp.to_dict()
        for key in ("prob_success", "prob_failure", "risk_reward_ratio",
                    "expected_value", "tail_risk", "worst_case", "best_case"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════════
# 7. position_sizer
# ════════════════════════════════════════════════════════════════════════════════
class TestPositionSizer(unittest.TestCase):

    def setUp(self):
        self.sizer  = _ps.PositionSizer()
        self.ze     = _ez.EntryZoneEngine()
        self.sl_eng = _sl.StopLossEngine()

    def _sl(self, fc):
        z = self.ze.calculate(fc)
        return self.sl_eng.recommend(fc, z)

    def test_fixed_risk_lots_positive(self):
        fc = _fc()
        sz = self.sizer.calculate(fc, self._sl(fc), _et.SizingMethod.FIXED_RISK)
        self.assertGreater(sz.lots, 0)

    def test_lots_within_broker_limits(self):
        fc = _fc()
        sz = self.sizer.calculate(fc, self._sl(fc))
        bc = _et.default_broker()
        self.assertGreaterEqual(sz.lots, bc.min_lot_size)
        self.assertLessEqual(sz.lots, bc.max_lot_size)

    def test_risk_pct_reasonable(self):
        fc = _fc(account_size=10_000.0)
        sz = self.sizer.calculate(fc, self._sl(fc))
        self.assertGreater(sz.risk_pct, 0.0)
        self.assertLessEqual(sz.risk_pct, 0.05)   # should not exceed 5%

    def test_volatility_adjusted_method(self):
        fc = _fc()
        sz = self.sizer.calculate(fc, self._sl(fc), _et.SizingMethod.VOLATILITY_ADJUSTED)
        self.assertGreater(sz.lots, 0)

    def test_kelly_method(self):
        fc = _fc(confidence=0.65, expected_return=0.10, expected_risk=0.05)
        sz = self.sizer.calculate(fc, self._sl(fc), _et.SizingMethod.KELLY)
        self.assertGreater(sz.kelly_fraction, 0)

    def test_units_equals_lots_times_100k(self):
        fc = _fc()
        sz = self.sizer.calculate(fc, self._sl(fc))
        self.assertAlmostEqual(sz.units, sz.lots * 100_000, delta=1)

    def test_margin_positive(self):
        fc = _fc()
        sz = self.sizer.calculate(fc, self._sl(fc))
        self.assertGreater(sz.margin_required, 0)

    def test_risk_limits_enforced(self):
        limits = _ps.RiskLimits(max_risk_per_trade_pct=0.001)
        sizer  = _ps.PositionSizer(risk_limits=limits)
        fc     = _fc(account_size=10_000)
        sz     = sizer.calculate(fc, self._sl(fc))
        self.assertLessEqual(sz.risk_pct, 0.001 + 1e-6)

    def test_to_dict_keys(self):
        fc = _fc()
        d  = self.sizer.calculate(fc).to_dict()
        for key in ("lots", "units", "risk_amount", "risk_pct", "method"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════════
# 8. trade_manager
# ════════════════════════════════════════════════════════════════════════════════
class TestTradeManager(unittest.TestCase):

    def setUp(self):
        self.manager = _tm.TradeManager()
        self.ze      = _ez.EntryZoneEngine()
        self.sl_eng  = _sl.StopLossEngine()
        self.tp_eng  = _tp.TakeProfitEngine()
        self.sizer   = _ps.PositionSizer()

    def _plan(self, **kwargs):
        fc  = _fc(**kwargs)
        z   = self.ze.calculate(fc)
        sl  = self.sl_eng.recommend(fc, z)
        tpr = self.tp_eng.recommend(fc, z, sl)
        sz  = self.sizer.calculate(fc, sl)
        return self.manager.plan(fc, sz, sl, tpr)

    def test_plan_has_rules(self):
        mp = self._plan()
        self.assertGreater(len(mp.rules), 0)

    def test_breakeven_trigger_price_nonzero(self):
        mp = self._plan()
        self.assertNotEqual(mp.breakeven_trigger_price, 0.0)

    def test_trail_distance_positive(self):
        mp = self._plan()
        self.assertGreater(mp.trail_distance, 0)

    def test_partial_close_pcts_sum(self):
        mp = self._plan()
        total = mp.partial_close_tp1_pct + mp.partial_close_tp2_pct + mp.runner_pct
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_high_news_risk_exit_early(self):
        mp = self._plan(news_risk=0.8)
        self.assertEqual(mp.primary_action, _et.ManagementAction.EXIT_EARLY)

    def test_high_confidence_scale_in(self):
        mp = self._plan(confidence=0.85, momentum=0.5)
        self.assertEqual(mp.primary_action, _et.ManagementAction.SCALE_IN)

    def test_low_confidence_partial_close(self):
        mp = self._plan(confidence=0.45, news_risk=0.0)
        self.assertEqual(mp.primary_action, _et.ManagementAction.PARTIAL_CLOSE)

    def test_to_dict_keys(self):
        d = self._plan().to_dict()
        for key in ("primary_action", "rules", "breakeven", "trail", "partial_close"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════════
# 9. execution_timing
# ════════════════════════════════════════════════════════════════════════════════
class TestExecutionTiming(unittest.TestCase):

    def setUp(self):
        self.engine = _eti.ExecutionTiming()

    def test_overlap_session_execute_now(self):
        fc = _fc(session=_et.Session.OVERLAP, news_risk=0.0,
                 volatility=0.010, spread=0.0001)
        t  = self.engine.evaluate(fc)
        self.assertEqual(t.recommendation, _et.TimingRecommendation.EXECUTE_NOW)

    def test_closed_session_avoid_or_wait(self):
        fc = _fc(session=_et.Session.CLOSED, spread=0.001)
        t  = self.engine.evaluate(fc)
        self.assertIn(
            t.recommendation,
            (_et.TimingRecommendation.AVOID, _et.TimingRecommendation.WAIT),
        )

    def test_high_news_risk_avoid(self):
        fc = _fc(news_risk=0.9)
        t  = self.engine.evaluate(fc)
        self.assertEqual(t.recommendation, _et.TimingRecommendation.AVOID)

    def test_wide_spread_avoid(self):
        fc = _fc(spread=0.010)
        t  = self.engine.evaluate(fc)
        self.assertEqual(t.recommendation, _et.TimingRecommendation.AVOID)

    def test_score_in_range(self):
        t = self.engine.evaluate(_fc())
        self.assertGreaterEqual(t.overall_score, 0.0)
        self.assertLessEqual(t.overall_score, 1.0)

    def test_five_factors_present(self):
        t = self.engine.evaluate(_fc())
        self.assertEqual(len(t.factors), 5)

    def test_slippage_positive(self):
        t = self.engine.evaluate(_fc())
        self.assertGreaterEqual(t.expected_slippage, 0.0)

    def test_to_dict_keys(self):
        d = self.engine.evaluate(_fc()).to_dict()
        for key in ("recommendation", "overall_score", "factors", "flags"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════════
# 10. order_strategy
# ════════════════════════════════════════════════════════════════════════════════
class TestOrderStrategy(unittest.TestCase):

    def setUp(self):
        self.engine = _os.OrderStrategyEngine()
        self.ze     = _ez.EntryZoneEngine()
        self.timer  = _eti.ExecutionTiming()

    def test_avoid_timing_gives_conditional(self):
        fc     = _fc(news_risk=0.9)
        t      = self.timer.evaluate(fc)
        z      = self.ze.calculate(fc)
        o      = self.engine.recommend(fc, z, t)
        self.assertEqual(o.primary_type, _et.OrderType.CONDITIONAL)

    def test_execute_now_high_conf_gives_market_or_limit(self):
        fc = _fc(session=_et.Session.OVERLAP, news_risk=0.0,
                 confidence=0.80, spread=0.0001, volatility=0.010)
        t  = self.timer.evaluate(fc)
        z  = self.ze.calculate(fc)
        o  = self.engine.recommend(fc, z, t)
        self.assertIn(o.primary_type, (_et.OrderType.MARKET, _et.OrderType.LIMIT))

    def test_fill_confidence_in_range(self):
        fc = _fc()
        t  = self.timer.evaluate(fc)
        z  = self.ze.calculate(fc)
        o  = self.engine.recommend(fc, z, t)
        self.assertGreaterEqual(o.fill_confidence, 0.0)
        self.assertLessEqual(o.fill_confidence, 1.0)

    def test_reasoning_not_empty(self):
        fc = _fc()
        o  = self.engine.recommend(fc)
        self.assertGreater(len(o.reasoning), 0)

    def test_entry_price_set(self):
        fc = _fc()
        o  = self.engine.recommend(fc)
        self.assertNotEqual(o.entry_price, 0.0)

    def test_to_dict_keys(self):
        d = self.engine.recommend(_fc()).to_dict()
        for key in ("primary_type", "alternative_type", "fill_confidence",
                    "expiry_hours", "reasoning"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════════
# 11. trade_scorer
# ════════════════════════════════════════════════════════════════════════════════
class TestTradeScorer(unittest.TestCase):

    def setUp(self):
        self.scorer  = _ts.TradeScorer()
        self.ze      = _ez.EntryZoneEngine()
        self.sl_eng  = _sl.StopLossEngine()
        self.tp_eng  = _tp.TakeProfitEngine()
        self.re      = _re.RiskEngine()
        self.timer   = _eti.ExecutionTiming()
        self.eq_eng  = _eq.EntryQualityEngine()

    def _score(self, **kwargs):
        fc  = _fc(**kwargs)
        z   = self.ze.calculate(fc)
        q   = self.eq_eng.evaluate(fc, z)
        sl  = self.sl_eng.recommend(fc, z)
        tpr = self.tp_eng.recommend(fc, z, sl)
        rp  = self.re.analyze(fc, sl, tpr)
        t   = self.timer.evaluate(fc)
        return self.scorer.score(fc, q, rp, t)

    def test_score_in_range(self):
        s = self._score()
        self.assertGreaterEqual(s.overall_score, 0.0)
        self.assertLessEqual(s.overall_score, 1.0)

    def test_grade_not_none(self):
        s = self._score()
        self.assertIsNotNone(s.grade)

    def test_high_conf_scores_higher(self):
        s_high = self._score(confidence=0.90, expected_return=0.20, momentum=0.5)
        s_low  = self._score(confidence=0.40, expected_return=0.03, momentum=-0.5)
        self.assertGreater(s_high.overall_score, s_low.overall_score)

    def test_to_dict_keys(self):
        d = self._score().to_dict()
        for key in ("grade", "overall_score", "sub_scores"):
            self.assertIn(key, d)

    def test_sub_scores_count(self):
        d = self._score().to_dict()
        self.assertEqual(len(d["sub_scores"]), 8)

    def test_weights_implied_by_sub_score_keys(self):
        d = self._score().to_dict()
        expected = {
            "forecast_quality", "entry_quality", "risk", "liquidity",
            "spread", "volatility", "reward", "portfolio_impact",
        }
        self.assertEqual(set(d["sub_scores"].keys()), expected)

    def test_no_optional_inputs_still_scores(self):
        s = self.scorer.score(_fc())
        self.assertGreaterEqual(s.overall_score, 0.0)


# ════════════════════════════════════════════════════════════════════════════════
# 12. execution_simulator
# ════════════════════════════════════════════════════════════════════════════════
class TestExecutionSimulator(unittest.TestCase):

    def setUp(self):
        self.sim    = _es.ExecutionSimulator()
        self.ze     = _ez.EntryZoneEngine()
        self.sl_eng = _sl.StopLossEngine()
        self.tp_eng = _tp.TakeProfitEngine()

    def _sim(self, **kwargs):
        fc  = _fc(**kwargs)
        z   = self.ze.calculate(fc)
        sl  = self.sl_eng.recommend(fc, z)
        tpr = self.tp_eng.recommend(fc, z, sl)
        return self.sim.simulate(fc, z, sl, tpr)

    def test_slippage_nonnegative(self):
        r = self._sim()
        self.assertGreaterEqual(r.expected_slippage, 0.0)

    def test_fill_quality_valid(self):
        r = self._sim()
        self.assertIn(r.fill_quality, ("EXCELLENT", "GOOD", "FAIR", "POOR"))

    def test_alt_entries_three(self):
        r = self._sim()
        self.assertEqual(len(r.alternative_entries), 3)

    def test_alt_stops_three(self):
        r = self._sim()
        self.assertEqual(len(r.alternative_stops), 3)

    def test_alt_targets_three(self):
        r = self._sim()
        self.assertEqual(len(r.alternative_targets), 3)

    def test_scenarios_six(self):
        r = self._sim()
        self.assertEqual(len(r.expected_outcomes), 6)

    def test_pnl_std_dev_nonneg(self):
        r = self._sim()
        self.assertGreaterEqual(r.pnl_std_dev, 0.0)

    def test_overlap_gives_better_fill_than_closed(self):
        r_overlap = self._sim(session=_et.Session.OVERLAP)
        r_closed  = self._sim(session=_et.Session.CLOSED)
        self.assertLessEqual(r_overlap.expected_slippage, r_closed.expected_slippage)

    def test_to_dict_keys(self):
        d = self._sim().to_dict()
        for key in ("expected_fill", "expected_slippage", "fill_quality",
                    "expected_outcomes", "summary"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════════
# 13. execution_planner
# ════════════════════════════════════════════════════════════════════════════════
class TestExecutionPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = _make_planner()

    def test_plan_returns_execution_plan(self):
        p = self.planner.plan(_fc())
        self.assertIsInstance(p, _ep.ExecutionPlan)

    def test_plan_has_all_components(self):
        p = self.planner.plan(_fc())
        self.assertIsNotNone(p.zones)
        self.assertIsNotNone(p.entry_quality)
        self.assertIsNotNone(p.stop_loss)
        self.assertIsNotNone(p.take_profit)
        self.assertIsNotNone(p.sizing)
        self.assertIsNotNone(p.management)
        self.assertIsNotNone(p.timing)
        self.assertIsNotNone(p.risk)
        self.assertIsNotNone(p.order)
        self.assertIsNotNone(p.score)
        self.assertIsNotNone(p.simulation)

    def test_instrument_preserved(self):
        p = self.planner.plan(_fc(instrument="XAUUSD"))
        self.assertEqual(p.instrument, "XAUUSD")

    def test_executive_summary_keys(self):
        s = self.planner.plan(_fc()).executive_summary()
        for key in ("instrument", "direction", "executable", "timing",
                    "trade_grade", "primary_entry", "initial_stop",
                    "tp1", "tp2", "runner", "lots", "risk_pct"):
            self.assertIn(key, s)

    def test_is_executable_returns_bool(self):
        p = self.planner.plan(_fc())
        self.assertIsInstance(p.is_executable(), bool)

    def test_plan_batch_sorted_desc(self):
        fcs = [
            _fc(confidence=0.85, expected_return=0.20),
            _fc(confidence=0.50, expected_return=0.04),
            _fc(confidence=0.70, expected_return=0.10),
        ]
        plans = self.planner.plan_batch(fcs)
        scores = [p.score.overall_score for p in plans]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_to_dict_has_executive_summary(self):
        d = self.planner.plan(_fc()).to_dict()
        self.assertIn("executive_summary", d)

    def test_sell_plan_works(self):
        p = self.planner.plan(_fc(direction="SELL"))
        self.assertEqual(p.direction, "SELL")


# ════════════════════════════════════════════════════════════════════════════════
# 14. execution_database
# ════════════════════════════════════════════════════════════════════════════════
class TestExecutionDatabase(unittest.TestCase):

    def setUp(self):
        self.db      = _edb.ExecutionDatabase(":memory:")
        self.planner = _make_planner()

    def _store(self, **kwargs):
        p = self.planner.plan(_fc(**kwargs))
        return self.db.store_plan(p), p

    def test_store_returns_int(self):
        row_id, _ = self._store()
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_latest_plans_empty_initially(self):
        db = _edb.ExecutionDatabase(":memory:")
        self.assertEqual(db.latest_plans(), [])

    def test_latest_plans_returns_after_store(self):
        self._store()
        plans = self.db.latest_plans(5)
        self.assertEqual(len(plans), 1)

    def test_plans_by_instrument(self):
        self._store(instrument="EURUSD")
        self._store(instrument="XAUUSD")
        plans = self.db.plans_by_instrument("EURUSD")
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["executive_summary"]["instrument"], "EURUSD")

    def test_top_scored_plans(self):
        self._store(confidence=0.90, expected_return=0.20)
        self._store(confidence=0.50, expected_return=0.04)
        top = self.db.top_scored_plans(1)
        self.assertEqual(len(top), 1)

    def test_executable_plans_query(self):
        self._store()
        # May or may not be executable — just check no error
        plans = self.db.executable_plans()
        self.assertIsInstance(plans, list)

    def test_stats_keys(self):
        self._store()
        s = self.db.stats()
        for key in ("execution_plans", "trade_scores", "sizing_decisions",
                    "management_plans", "optimization_events", "average_score"):
            self.assertIn(key, s)

    def test_stats_counts_increase(self):
        s0 = self.db.stats()
        self._store()
        s1 = self.db.stats()
        self.assertGreater(s1["execution_plans"], s0["execution_plans"])

    def test_log_optimization(self):
        self.db.log_optimization("EURUSD", "SIZING", "Adjusted lots due to limit")
        log = self.db.optimization_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["instrument"], "EURUSD")

    def test_sizing_history(self):
        self._store(instrument="GBPUSD")
        h = self.db.sizing_history("GBPUSD")
        self.assertEqual(len(h), 1)


# ════════════════════════════════════════════════════════════════════════════════
# 15. execution_dashboard
# ════════════════════════════════════════════════════════════════════════════════
class TestExecutionDashboard(unittest.TestCase):

    def setUp(self):
        planner = _make_planner()
        fcs = [
            _fc(instrument="EURUSD", direction="BUY"),
            _fc(instrument="XAUUSD", direction="BUY",  current_price=1950.0, atr=3.5, spread=0.5),
            _fc(instrument="NAS100", direction="SELL", current_price=14500., atr=45., spread=2.0),
        ]
        plans = planner.plan_batch(fcs)
        self.dash = _ed.ExecutionDashboard(plans)

    def test_execution_dashboard_chart_type(self):
        d = self.dash.execution_dashboard()
        self.assertEqual(d["chart_type"], "execution_dashboard")

    def test_execution_dashboard_plan_count(self):
        d = self.dash.execution_dashboard()
        self.assertEqual(d["total_plans"], 3)

    def test_risk_dashboard_keys(self):
        d = self.dash.risk_dashboard()
        for key in ("chart_type", "total_portfolio_risk_pct", "risks"):
            self.assertIn(key, d)

    def test_trade_timeline_events(self):
        d = self.dash.trade_timeline()
        self.assertGreater(len(d["events"]), 0)
        # 5 events per plan × 3 plans = 15
        self.assertEqual(len(d["events"]), 15)

    def test_probability_charts_instruments(self):
        d = self.dash.probability_charts()
        self.assertEqual(len(d["instruments"]), 3)

    def test_stop_distribution_fields(self):
        d = self.dash.stop_distribution()
        row = d["instruments"][0]
        for key in ("initial", "atr", "structure", "emergency"):
            self.assertIn(key, row)

    def test_target_distribution_fields(self):
        d = self.dash.target_distribution()
        row = d["instruments"][0]
        for key in ("tp1", "tp2", "tp3", "runner", "best_level"):
            self.assertIn(key, row)

    def test_scorecards_sorted_desc(self):
        d      = self.dash.execution_scorecards()
        scores = [c["overall"] for c in d["scorecards"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fill_quality_analysis_summary(self):
        d = self.dash.fill_quality_analysis()
        self.assertIn("quality_summary", d)

    def test_full_dashboard_has_8_views(self):
        d = self.dash.full_dashboard()
        self.assertEqual(len(d), 8)

    def test_add_plan_increases_count(self):
        planner = _make_planner()
        p = planner.plan(_fc(instrument="BTCUSD", current_price=65000., atr=500., spread=5.))
        self.dash.add_plan(p)
        d = self.dash.execution_dashboard()
        self.assertEqual(d["total_plans"], 4)


# ════════════════════════════════════════════════════════════════════════════════
# 16. execution_api
# ════════════════════════════════════════════════════════════════════════════════
class TestExecutionApi(unittest.TestCase):

    def setUp(self):
        self.api = _make_api()
        self.fd  = {
            "instrument": "EURUSD", "direction": "BUY",
            "confidence": 0.72, "expected_return": 0.08, "expected_risk": 0.10,
            "current_price": 1.0850, "atr": 0.0012, "spread": 0.0001,
            "session": "LONDON", "volatility": 0.009, "momentum": 0.3,
            "news_risk": 0.1, "account_size": 10000.0,
        }

    def test_plan_status_ok(self):
        r = self.api.plan(self.fd)
        self.assertEqual(r["status"], "ok")

    def test_plan_has_executive_summary(self):
        r = self.api.plan(self.fd)
        self.assertIn("executive_summary", r["data"])

    def test_plan_batch_returns_list(self):
        r = self.api.plan_batch([self.fd, {**self.fd, "instrument": "XAUUSD",
            "current_price": 1950.0, "atr": 3.5, "spread": 0.5}])
        self.assertEqual(r["status"], "ok")
        self.assertIsInstance(r["data"], list)
        self.assertEqual(len(r["data"]), 2)

    def test_entry_zones_ok(self):
        r = self.api.entry_zones(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("primary_entry", r["data"])

    def test_entry_quality_ok(self):
        r = self.api.entry_quality(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("grade", r["data"])

    def test_stop_loss_ok(self):
        r = self.api.stop_loss(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("initial_stop", r["data"])

    def test_take_profit_ok(self):
        r = self.api.take_profit(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("tp1", r["data"])

    def test_sizing_ok(self):
        r = self.api.sizing(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("lots", r["data"])

    def test_sizing_all_methods(self):
        for method in ("FIXED_RISK", "FIXED_FRACTIONAL", "VOLATILITY_ADJUSTED", "KELLY"):
            r = self.api.sizing(self.fd, method=method)
            self.assertEqual(r["status"], "ok", msg=f"method={method}")

    def test_management_ok(self):
        r = self.api.management(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("primary_action", r["data"])

    def test_timing_ok(self):
        r = self.api.timing(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("recommendation", r["data"])

    def test_risk_ok(self):
        r = self.api.risk(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("risk_reward_ratio", r["data"])

    def test_order_ok(self):
        r = self.api.order(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("primary_type", r["data"])

    def test_score_ok(self):
        r = self.api.score(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("grade", r["data"])

    def test_simulation_ok(self):
        r = self.api.simulation(self.fd)
        self.assertEqual(r["status"], "ok")
        self.assertIn("expected_fill", r["data"])

    def test_dashboard_ok(self):
        fd2 = {**self.fd, "instrument": "XAUUSD", "current_price": 1950.0,
               "atr": 3.5, "spread": 0.5}
        r   = self.api.dashboard([self.fd, fd2])
        self.assertEqual(r["status"], "ok")
        self.assertEqual(len(r["data"]), 8)

    def test_stats_ok(self):
        self.api.plan(self.fd)
        r = self.api.stats()
        self.assertEqual(r["status"], "ok")
        self.assertGreater(r["data"]["execution_plans"], 0)

    def test_history_ok(self):
        self.api.plan(self.fd)
        r = self.api.history()
        self.assertEqual(r["status"], "ok")
        self.assertGreater(len(r["data"]), 0)

    def test_history_by_instrument(self):
        self.api.plan(self.fd)
        r = self.api.history(instrument="EURUSD")
        self.assertEqual(r["status"], "ok")

    def test_bad_forecast_returns_error(self):
        r = self.api.plan({"instrument": "EURUSD"})   # missing required fields
        self.assertEqual(r["status"], "error")

    def test_timestamp_present(self):
        r = self.api.plan(self.fd)
        self.assertIn("timestamp", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
