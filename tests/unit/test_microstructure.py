"""Phase 22 — Unit tests for the Market Microstructure Intelligence Engine.

Run with:  python3 -m unittest tests/unit/test_microstructure.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MS_DIR = ROOT / "trainer" / "microstructure"


def _load(name: str, rel: str):
    path = ROOT / "trainer" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order
mt   = _load("trainer.microstructure.micro_types",        "microstructure/micro_types.py")
mm   = _load("trainer.microstructure.micro_models",       "microstructure/micro_models.py")
pa   = _load("trainer.microstructure.price_action",       "microstructure/price_action.py")
vm   = _load("trainer.microstructure.volatility_micro",   "microstructure/volatility_micro.py")
mom  = _load("trainer.microstructure.momentum_micro",     "microstructure/momentum_micro.py")
lp   = _load("trainer.microstructure.liquidity_proxy",    "microstructure/liquidity_proxy.py")
me   = _load("trainer.microstructure.market_efficiency",  "microstructure/market_efficiency.py")
ti   = _load("trainer.microstructure.tick_interface",     "microstructure/tick_interface.py")
ofi  = _load("trainer.microstructure.order_flow_interface","microstructure/order_flow_interface.py")
emb  = _load("trainer.microstructure.micro_embeddings",   "microstructure/micro_embeddings.py")
sc   = _load("trainer.microstructure.state_classifier",   "microstructure/state_classifier.py")
ms_  = _load("trainer.microstructure.micro_scorer",       "microstructure/micro_scorer.py")
mdb  = _load("trainer.microstructure.micro_database",     "microstructure/micro_database.py")
mdash = _load("trainer.microstructure.micro_dashboard",   "microstructure/micro_dashboard.py")
mapi  = _load("trainer.microstructure.micro_api",         "microstructure/micro_api.py")

# Aliases
MarketState           = mt.MarketState
MomentumState         = mt.MomentumState
VolatilityState       = mt.VolatilityState
LiquidityState        = mt.LiquidityState
TrendStrength         = mt.TrendStrength
DataGranularity       = mt.DataGranularity
EMBEDDING_TOTAL       = mt.EMBEDDING_TOTAL
trend_strength_from_score = mt.trend_strength_from_score

OHLCBar               = mm.OHLCBar
MicrostructureFeatures = mm.MicrostructureFeatures
MicrostructureScore   = mm.MicrostructureScore
MarketStateRecord     = mm.MarketStateRecord
MicrostructureEmbedding = mm.MicrostructureEmbedding

PriceActionAnalyzer   = pa.PriceActionAnalyzer
VolatilityMicrostructure = vm.VolatilityMicrostructure
MomentumMicrostructure = mom.MomentumMicrostructure
LiquidityProxy        = lp.LiquidityProxy
MarketEfficiency      = me.MarketEfficiency
TickInterface         = ti.TickInterface
TickBar               = ti.TickBar
OrderFlowInterface    = ofi.OrderFlowInterface
OrderBookLevel        = ofi.OrderBookLevel
OrderFlowBar          = ofi.OrderFlowBar
MicrostructureEmbedder = emb.MicrostructureEmbedder
MarketStateClassifier = sc.MarketStateClassifier
MicrostructureScorer  = ms_.MicrostructureScorer
MicrostructureDatabase = mdb.MicrostructureDatabase
MicrostructureDashboard = mdash.MicrostructureDashboard
MicrostructureAPI     = mapi.MicrostructureAPI


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_bar(
    ts=None, o=1.10000, h=1.10200, l=1.09800, c=1.10150,
    vol=1000.0, tf="H1", instrument="EURUSD",
) -> OHLCBar:
    return OHLCBar(
        timestamp  = ts or time.time(),
        open       = o,
        high       = h,
        low        = l,
        close      = c,
        volume     = vol,
        timeframe  = tf,
        instrument = instrument,
    )


def make_trending_bars(n=20, direction=1, base=1.10000, step=0.00050) -> list:
    bars = []
    ts = time.time() - n * 3600
    price = base
    for i in range(n):
        o = price
        c = price + direction * step
        h = max(o, c) + 0.00020
        l = min(o, c) - 0.00010
        bars.append(OHLCBar(ts + i * 3600, o, h, l, c, 1000.0, "H1", "EURUSD"))
        price = c
    return bars


def make_ranging_bars(n=20, base=1.10000, amplitude=0.00100) -> list:
    bars = []
    ts = time.time() - n * 3600
    import math
    for i in range(n):
        mid = base + amplitude * math.sin(i * 0.8)
        o = mid - 0.00020
        c = mid + 0.00020
        h = max(o, c) + 0.00010
        l = min(o, c) - 0.00010
        bars.append(OHLCBar(ts + i * 3600, o, h, l, c, 500.0, "H1", "EURUSD"))
    return bars


def make_features(
    trend_energy=0.70,
    directional_persistence=0.75,
    candle_efficiency=0.70,
    atr_expansion=0.60,
    atr_compression=0.10,
    volatility_shock=0.0,
    spread_proxy=0.20,
    gap_frequency=0.05,
    large_candle_frequency=0.10,
    liquidity_vacuum_prob=0.10,
    breakout_strength=0.30,
    reversal_strength=0.10,
    momentum_exhaustion=0.20,
    momentum_acceleration=0.40,
    directional_efficiency=0.65,
    body_ratio=0.70,
    close_position=0.80,
    impulse_strength=0.70,
    continuation_probability=0.65,
    compression_zone_score=0.15,
    range_expansion=0.60,
    noise_ratio=0.25,
    volatility_clustering=0.30,
    volatility_persistence=0.40,
    range_compression=0.05,
    momentum_deceleration=0.10,
    impulse_decay=0.20,
    trend_efficiency=0.70,
    mean_reversion_probability=0.30,
    randomness_index=0.30,
    fractal_dimension=1.3,
    directional_stability=0.70,
    noise_level=0.25,
    reaction_zone_score=0.20,
) -> MicrostructureFeatures:
    f = MicrostructureFeatures()
    for k, v in locals().items():
        if k != "f" and hasattr(f, k):
            setattr(f, k, v)
    return f


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMicroTypes(unittest.TestCase):
    def test_market_states_count(self):
        states = list(MarketState)
        self.assertEqual(len(states), 12)

    def test_all_market_states(self):
        values = {s.value for s in MarketState}
        for v in ("TRENDING","STRONG_TREND","WEAK_TREND","RANGE","COMPRESSION",
                   "EXPANSION","BREAKOUT","FAKE_BREAKOUT","REVERSAL",
                   "HIGH_VOLATILITY","LOW_VOLATILITY","TRANSITION"):
            self.assertIn(v, values)

    def test_momentum_states(self):
        vals = {s.value for s in MomentumState}
        for v in ("ACCELERATING","DECELERATING","EXHAUSTED","NEUTRAL","REVERSING"):
            self.assertIn(v, vals)

    def test_volatility_states(self):
        vals = {s.value for s in VolatilityState}
        self.assertIn("SHOCK", vals)
        self.assertIn("EXPANDING", vals)

    def test_liquidity_states(self):
        vals = {s.value for s in LiquidityState}
        self.assertIn("DEEP", vals)
        self.assertIn("VACUUM", vals)

    def test_trend_strength_from_score(self):
        self.assertEqual(trend_strength_from_score(0.90), TrendStrength.VERY_STRONG)
        self.assertEqual(trend_strength_from_score(0.65), TrendStrength.STRONG)
        self.assertEqual(trend_strength_from_score(0.45), TrendStrength.MODERATE)
        self.assertEqual(trend_strength_from_score(0.25), TrendStrength.WEAK)
        self.assertEqual(trend_strength_from_score(0.10), TrendStrength.NONE)

    def test_embedding_total(self):
        self.assertEqual(EMBEDDING_TOTAL, 32)

    def test_data_granularity_ohlc(self):
        self.assertEqual(DataGranularity.OHLC.value, "OHLC")


class TestOHLCBar(unittest.TestCase):
    def test_range(self):
        b = make_bar(h=1.10200, l=1.09800)
        self.assertAlmostEqual(b.range, 0.00400, places=5)

    def test_body(self):
        b = make_bar(o=1.10000, c=1.10150)
        self.assertAlmostEqual(b.body, 0.00150, places=5)

    def test_is_bullish(self):
        self.assertTrue(make_bar(o=1.100, c=1.101).is_bullish)
        self.assertFalse(make_bar(o=1.101, c=1.100).is_bullish)

    def test_mid(self):
        b = make_bar(h=1.102, l=1.098)
        self.assertAlmostEqual(b.mid, 1.100, places=3)

    def test_to_dict(self):
        b = make_bar()
        d = b.to_dict()
        self.assertIn("open", d)
        self.assertIn("close", d)
        self.assertIn("instrument", d)

    def test_zero_range_bar(self):
        b = make_bar(o=1.1, h=1.1, l=1.1, c=1.1)
        self.assertAlmostEqual(b.range, 0.0, places=5)


class TestMicrostructureFeatures(unittest.TestCase):
    def test_default_features(self):
        f = MicrostructureFeatures()
        self.assertFalse(f.has_tick_features)
        self.assertEqual(f.granularity, DataGranularity.OHLC.value)

    def test_to_dict(self):
        f = MicrostructureFeatures()
        d = f.to_dict()
        self.assertIn("body_ratio", d)
        self.assertIn("trend_efficiency", d)
        self.assertIn("momentum_acceleration", d)

    def test_price_action_vector_length(self):
        f = MicrostructureFeatures()
        v = f.price_action_vector()
        self.assertEqual(len(v), 8)

    def test_liquidity_vector_length(self):
        f = MicrostructureFeatures()
        v = f.liquidity_vector()
        self.assertEqual(len(v), 6)

    def test_momentum_vector_length(self):
        v = MicrostructureFeatures().momentum_vector()
        self.assertEqual(len(v), 6)

    def test_volatility_vector_length(self):
        v = MicrostructureFeatures().volatility_vector()
        self.assertEqual(len(v), 6)

    def test_efficiency_vector_length(self):
        v = MicrostructureFeatures().efficiency_vector()
        self.assertEqual(len(v), 6)

    def test_tick_features_none_by_default(self):
        f = MicrostructureFeatures()
        self.assertIsNone(f.tick_velocity)
        self.assertIsNone(f.tick_imbalance)


class TestPriceActionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PriceActionAnalyzer()

    def test_analyse_bar_body_ratio(self):
        b = make_bar(o=1.10000, h=1.10200, l=1.09800, c=1.10200)  # full body
        r = self.analyzer.analyse_bar(b)
        self.assertGreater(r["body_ratio"], 0.40)

    def test_analyse_bar_upper_wick(self):
        b = make_bar(o=1.10000, h=1.10300, l=1.09900, c=1.10050)
        r = self.analyzer.analyse_bar(b)
        self.assertGreater(r["upper_wick_ratio"], 0.0)

    def test_analyse_bar_close_position(self):
        b = make_bar(o=1.10000, h=1.10200, l=1.09800, c=1.10200)
        r = self.analyzer.analyse_bar(b)
        self.assertAlmostEqual(r["close_position"], 1.0, places=1)

    def test_analyse_bar_bullish_directional_efficiency(self):
        b = make_bar(o=1.10000, h=1.10200, l=1.09900, c=1.10180)
        r = self.analyzer.analyse_bar(b)
        self.assertGreater(r["directional_efficiency"], 0.0)

    def test_analyse_bar_bearish_directional_efficiency(self):
        b = make_bar(o=1.10200, h=1.10250, l=1.09900, c=1.09950)
        r = self.analyzer.analyse_bar(b)
        self.assertLess(r["directional_efficiency"], 0.0)

    def test_analyse_bar_zero_range(self):
        b = make_bar(o=1.1, h=1.1, l=1.1, c=1.1)
        r = self.analyzer.analyse_bar(b)
        self.assertEqual(r["body_ratio"], 0.0)

    def test_retracement_quality_single_bar(self):
        q = self.analyzer.retracement_quality([make_bar()])
        self.assertEqual(q, 0.5)

    def test_retracement_quality_two_bars(self):
        bars = make_trending_bars(n=5)
        q = self.analyzer.retracement_quality(bars)
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)

    def test_continuation_probability_range(self):
        bars = make_trending_bars(n=10)
        cp = self.analyzer.continuation_probability(bars)
        self.assertGreaterEqual(cp, 0.0)
        self.assertLessEqual(cp, 1.0)

    def test_analyse_window_fills_features(self):
        bars = make_trending_bars(n=5)
        f = MicrostructureFeatures()
        result = self.analyzer.analyse_window(bars, f)
        self.assertGreater(result.body_ratio, 0.0)

    def test_analyse_window_empty(self):
        f = MicrostructureFeatures()
        result = self.analyzer.analyse_window([], f)
        self.assertEqual(result.body_ratio, 0.0)  # unchanged

    def test_compression_plus_expansion_equals_1(self):
        b = make_bar(o=1.10000, h=1.10200, l=1.09800, c=1.10150)
        r = self.analyzer.analyse_bar(b)
        self.assertAlmostEqual(r["compression"] + r["expansion"], 1.0, places=4)


class TestVolatilityMicrostructure(unittest.TestCase):
    def setUp(self):
        self.engine = VolatilityMicrostructure()

    def test_single_bar_returns_unchanged(self):
        f = MicrostructureFeatures()
        result = self.engine.analyse_window([make_bar()], f)
        self.assertEqual(result.atr_expansion, 0.0)

    def test_trending_bars_produce_expansion_or_normal(self):
        bars = make_trending_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.atr_expansion, 0.0)
        self.assertLessEqual(result.atr_expansion, 1.0)

    def test_atr_expansion_bounds(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.atr_expansion, 0.0)
        self.assertLessEqual(result.atr_expansion, 1.0)

    def test_atr_compression_bounds(self):
        bars = make_ranging_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.atr_compression, 0.0)
        self.assertLessEqual(result.atr_compression, 1.0)

    def test_noise_ratio_bounds(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.noise_ratio, 0.0)
        self.assertLessEqual(result.noise_ratio, 1.0)

    def test_volatility_shock_high_for_large_bar(self):
        # Make bars with one very large candle at the end
        bars = make_ranging_bars(n=14)
        large_bar = make_bar(h=1.15000, l=1.08000)   # huge range
        bars.append(large_bar)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreater(result.volatility_shock, 0.0)

    def test_volatility_clustering_bounds(self):
        bars = make_trending_bars(n=10)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.volatility_clustering, 0.0)
        self.assertLessEqual(result.volatility_clustering, 1.0)

    def test_autocorr_lag1_constant_series(self):
        series = [0.001] * 10
        ac = VolatilityMicrostructure._autocorr_lag1(series)
        self.assertAlmostEqual(ac, 0.0, places=5)


class TestMomentumMicrostructure(unittest.TestCase):
    def setUp(self):
        self.engine = MomentumMicrostructure()

    def test_insufficient_bars(self):
        f = MicrostructureFeatures()
        result = self.engine.analyse_window([make_bar(), make_bar()], f)
        self.assertEqual(result.trend_energy, 0.0)  # unchanged

    def test_trend_energy_high_for_trending(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreater(result.trend_energy, 0.30)

    def test_directional_persistence_high_for_trending(self):
        bars = make_trending_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreater(result.directional_persistence, 0.50)

    def test_directional_persistence_bounds(self):
        bars = make_ranging_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.directional_persistence, 0.0)
        self.assertLessEqual(result.directional_persistence, 1.0)

    def test_impulse_decay_bounds(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.impulse_decay, 0.0)
        self.assertLessEqual(result.impulse_decay, 1.0)

    def test_momentum_exhaustion_bounds(self):
        bars = make_ranging_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.momentum_exhaustion, 0.0)
        self.assertLessEqual(result.momentum_exhaustion, 1.0)

    def test_acceleration_or_deceleration_not_both(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        # At most one of acceleration or deceleration can be non-zero
        self.assertFalse(result.momentum_acceleration > 0 and result.momentum_deceleration > 0)


class TestLiquidityProxy(unittest.TestCase):
    def setUp(self):
        self.engine = LiquidityProxy()

    def test_empty_bars(self):
        f = MicrostructureFeatures()
        result = self.engine.analyse_window([], f)
        self.assertEqual(result.spread_proxy, 0.0)

    def test_spread_proxy_bounds(self):
        bars = make_trending_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.spread_proxy, 0.0)
        self.assertLessEqual(result.spread_proxy, 1.0)

    def test_gap_frequency_zero_for_clean_trend(self):
        bars = make_trending_bars(n=15)
        # Adjust opens to close of previous bar (no gaps)
        for i in range(1, len(bars)):
            bars[i].open = bars[i - 1].close
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertEqual(result.gap_frequency, 0.0)

    def test_breakout_strength_bounds(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.breakout_strength, 0.0)
        self.assertLessEqual(result.breakout_strength, 1.0)

    def test_reversal_strength_bounds(self):
        bars = make_ranging_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.reversal_strength, 0.0)
        self.assertLessEqual(result.reversal_strength, 1.0)

    def test_liquidity_vacuum_prob_bounds(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.liquidity_vacuum_prob, 0.0)
        self.assertLessEqual(result.liquidity_vacuum_prob, 1.0)

    def test_compression_zone_score_bounds(self):
        bars = make_ranging_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.compression_zone_score, 0.0)
        self.assertLessEqual(result.compression_zone_score, 1.0)

    def test_single_bar_no_gaps(self):
        f = MicrostructureFeatures()
        result = self.engine.analyse_window([make_bar()], f)
        self.assertEqual(result.gap_frequency, 0.0)


class TestMarketEfficiency(unittest.TestCase):
    def setUp(self):
        self.engine = MarketEfficiency()

    def test_insufficient_bars(self):
        f = MicrostructureFeatures()
        result = self.engine.analyse_window([make_bar(), make_bar()], f)
        self.assertEqual(result.trend_efficiency, 0.0)

    def test_trend_efficiency_high_for_trending(self):
        bars = make_trending_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreater(result.trend_efficiency, 0.50)

    def test_trend_efficiency_low_for_ranging(self):
        bars = make_ranging_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertLess(result.trend_efficiency, 0.60)

    def test_randomness_index_complement(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertAlmostEqual(
            result.trend_efficiency + result.randomness_index, 1.0, places=4
        )

    def test_fractal_dimension_bounds(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.fractal_dimension, 1.0)
        self.assertLessEqual(result.fractal_dimension, 2.0)

    def test_directional_stability_bounds(self):
        bars = make_trending_bars(n=15)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.directional_stability, 0.0)
        self.assertLessEqual(result.directional_stability, 1.0)

    def test_directional_stability_high_for_strong_trend(self):
        bars = make_trending_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreater(result.directional_stability, 0.50)

    def test_mean_reversion_prob_bounds(self):
        bars = make_ranging_bars(n=20)
        f = MicrostructureFeatures()
        result = self.engine.analyse_window(bars, f)
        self.assertGreaterEqual(result.mean_reversion_probability, 0.0)
        self.assertLessEqual(result.mean_reversion_probability, 1.0)


class TestTickInterface(unittest.TestCase):
    def setUp(self):
        self.ti = TickInterface()

    def _make_ticks(self, n=20, direction=1):
        ticks = []
        ts = time.time()
        price = 1.10000
        for i in range(n):
            side = "BUY" if direction > 0 else "SELL"
            ticks.append(TickBar(ts + i * 0.5, price + direction * i * 0.00001, 1.0, side))
        return ticks

    def test_not_available(self):
        self.assertFalse(self.ti.is_available())

    def test_tick_velocity_none_for_empty(self):
        self.assertIsNone(self.ti.tick_velocity(None))
        self.assertIsNone(self.ti.tick_velocity([]))

    def test_tick_velocity_positive(self):
        ticks = self._make_ticks(10)
        v = self.ti.tick_velocity(ticks)
        self.assertIsNotNone(v)
        self.assertGreater(v, 0.0)

    def test_tick_imbalance_bullish(self):
        ticks = [TickBar(time.time() + i, 1.10 + i * 0.00001, 1.0, "BUY") for i in range(10)]
        imb = self.ti.tick_imbalance(ticks)
        self.assertIsNotNone(imb)
        self.assertAlmostEqual(imb, 1.0, places=4)

    def test_tick_imbalance_balanced(self):
        ticks = [
            TickBar(time.time() + i, 1.10, 1.0, "BUY" if i % 2 == 0 else "SELL")
            for i in range(10)
        ]
        imb = self.ti.tick_imbalance(ticks)
        self.assertIsNotNone(imb)
        self.assertAlmostEqual(abs(imb), 0.0, places=4)

    def test_tick_momentum_bounds(self):
        ticks = self._make_ticks(30)
        mom = self.ti.tick_momentum(ticks)
        self.assertIsNotNone(mom)
        self.assertGreaterEqual(mom, -1.0)
        self.assertLessEqual(mom, 1.0)

    def test_tick_entropy_balanced(self):
        ticks = [
            TickBar(time.time() + i, 1.10, 1.0, "BUY" if i % 2 == 0 else "SELL")
            for i in range(10)
        ]
        ent = self.ti.tick_entropy(ticks)
        self.assertIsNotNone(ent)
        self.assertAlmostEqual(ent, 1.0, places=3)  # max entropy

    def test_enrich_features_with_ticks(self):
        ticks = self._make_ticks(10)
        f = MicrostructureFeatures()
        result = self.ti.enrich_features(f, ticks)
        self.assertTrue(result.has_tick_features)

    def test_enrich_features_no_ticks(self):
        f = MicrostructureFeatures()
        result = self.ti.enrich_features(f, None)
        self.assertFalse(result.has_tick_features)


class TestOrderFlowInterface(unittest.TestCase):
    def setUp(self):
        self.ofi = OrderFlowInterface()

    def test_not_available(self):
        self.assertFalse(self.ofi.is_available())

    def test_volume_delta_none_for_empty(self):
        self.assertIsNone(self.ofi.volume_delta(None))

    def test_bid_ask_imbalance_none_for_empty(self):
        self.assertIsNone(self.ofi.bid_ask_imbalance(None))

    def test_order_book_level_imbalance(self):
        level = OrderBookLevel(price=1.10000, bid_size=100.0, ask_size=50.0)
        # (100-50)/150
        self.assertAlmostEqual(level.imbalance, 0.3333, places=3)

    def test_market_depth_none_for_empty(self):
        self.assertIsNone(self.ofi.market_depth(None))

    def test_market_depth_with_levels(self):
        levels = [
            OrderBookLevel(1.10000, 100.0, 80.0),
            OrderBookLevel(1.10010, 90.0, 70.0),
        ]
        depth = self.ofi.market_depth(levels)
        self.assertIsNotNone(depth)
        self.assertAlmostEqual(depth["bid_depth"], 190.0, places=1)
        self.assertAlmostEqual(depth["ask_depth"], 150.0, places=1)

    def test_data_quality_report(self):
        report = self.ofi.data_quality_report()
        self.assertFalse(report["available"])
        self.assertIn("current_source", report)
        self.assertEqual(report["current_source"], "OHLC")


class TestMicrostructureEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = MicrostructureEmbedder()

    def test_embedding_dim(self):
        self.assertEqual(self.embedder.embedding_dim(), 32)

    def test_sub_dims(self):
        dims = self.embedder.sub_dims()
        self.assertEqual(sum(dims.values()), 32)

    def test_embed_returns_32_dims(self):
        f = make_features()
        emb = self.embedder.embed(f)
        self.assertEqual(emb.dim, 32)
        self.assertEqual(len(emb.embedding), 32)

    def test_embedding_values_in_range(self):
        f = make_features()
        emb = self.embedder.embed(f)
        for v in emb.embedding:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_sub_vector_lengths(self):
        f = make_features()
        emb = self.embedder.embed(f)
        self.assertEqual(len(emb.price_action_state), 8)
        self.assertEqual(len(emb.liquidity_state), 6)
        self.assertEqual(len(emb.momentum_state), 6)
        self.assertEqual(len(emb.volatility_state), 6)
        self.assertEqual(len(emb.efficiency_state), 6)

    def test_to_dict(self):
        f = make_features()
        emb = self.embedder.embed(f)
        d = emb.to_dict()
        self.assertIn("embedding", d)
        self.assertIn("dim", d)
        self.assertEqual(d["dim"], 32)

    def test_batch_embed(self):
        features = [make_features() for _ in range(5)]
        embs = self.embedder.batch_embed(features, "EURUSD", "H1")
        self.assertEqual(len(embs), 5)
        for e in embs:
            self.assertEqual(e.dim, 32)

    def test_different_features_different_embedding(self):
        f1 = make_features(trend_energy=0.90, atr_expansion=0.90)
        f2 = make_features(trend_energy=0.10, atr_expansion=0.10)
        e1 = self.embedder.embed(f1)
        e2 = self.embedder.embed(f2)
        self.assertNotEqual(e1.embedding, e2.embedding)


class TestMicrostructureScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = MicrostructureScorer()

    def test_score_returns_all_fields(self):
        f = make_features()
        s = self.scorer.score(f)
        self.assertIsInstance(s.trend_score, float)
        self.assertIsInstance(s.overall_score, float)

    def test_all_scores_bounded(self):
        f = make_features()
        s = self.scorer.score(f)
        for attr in ("trend_score","liquidity_score","momentum_score",
                     "efficiency_score","volatility_score","compression_score",
                     "expansion_score","overall_score"):
            v = getattr(s, attr)
            self.assertGreaterEqual(v, 0.0, msg=f"{attr} < 0")
            self.assertLessEqual(v, 1.0, msg=f"{attr} > 1")

    def test_trending_features_high_trend_score(self):
        f = make_features(trend_energy=0.90, directional_persistence=0.90,
                          candle_efficiency=0.80, directional_efficiency=0.85)
        s = self.scorer.score(f)
        self.assertGreater(s.trend_score, 0.60)

    def test_compressed_features_high_compression(self):
        f = make_features(atr_compression=0.80, compression_zone_score=0.75,
                          range_compression=1.0, trend_energy=0.10)
        s = self.scorer.score(f)
        self.assertGreater(s.compression_score, 0.50)

    def test_volatile_features_high_volatility(self):
        f = make_features(atr_expansion=0.90, volatility_shock=1.0, range_expansion=1.0)
        s = self.scorer.score(f)
        self.assertGreater(s.volatility_score, 0.50)

    def test_weights_sum_to_1(self):
        w = self.scorer.weights()
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)

    def test_score_to_dict(self):
        f = make_features()
        s = self.scorer.score(f)
        d = s.to_dict()
        self.assertIn("trend_score", d)
        self.assertIn("overall_score", d)


class TestMarketStateClassifier(unittest.TestCase):
    def setUp(self):
        self.scorer     = MicrostructureScorer()
        self.classifier = MarketStateClassifier()

    def _classify(self, features):
        score = self.scorer.score(features)
        return self.classifier.classify(features, score)

    def test_strong_trend_classification(self):
        f = make_features(trend_energy=0.95, directional_persistence=0.90,
                          candle_efficiency=0.90, directional_efficiency=0.90)
        record = self._classify(f)
        self.assertIn(record.state, (MarketState.STRONG_TREND, MarketState.TRENDING))

    def test_compression_classification(self):
        f = make_features(
            trend_energy=0.05, directional_persistence=0.40,
            atr_compression=0.90, compression_zone_score=0.85,
            range_compression=1.0, atr_expansion=0.02,
            volatility_shock=0.0, breakout_strength=0.05,
            reversal_strength=0.05,
        )
        record = self._classify(f)
        self.assertIn(record.state, (
            MarketState.COMPRESSION, MarketState.LOW_VOLATILITY,
            MarketState.RANGE, MarketState.TRANSITION,
        ))

    def test_confidence_in_range(self):
        record = self._classify(make_features())
        self.assertGreaterEqual(record.confidence, 0.0)
        self.assertLessEqual(record.confidence, 1.0)

    def test_record_has_all_states(self):
        record = self._classify(make_features())
        self.assertIsInstance(record.state, MarketState)
        self.assertIsInstance(record.momentum_state, MomentumState)
        self.assertIsInstance(record.volatility_state, VolatilityState)
        self.assertIsInstance(record.liquidity_state, LiquidityState)
        self.assertIsInstance(record.trend_strength, TrendStrength)

    def test_to_dict(self):
        record = self._classify(make_features())
        d = record.to_dict()
        self.assertIn("state", d)
        self.assertIn("confidence", d)
        self.assertIn("reasoning", d)

    def test_reasoning_not_empty(self):
        record = self._classify(make_features())
        self.assertGreater(len(record.reasoning), 0)

    def test_high_volatility_classification(self):
        f = make_features(
            atr_expansion=0.95, volatility_shock=1.0,
            volatility_clustering=0.90, noise_ratio=0.80,
        )
        record = self._classify(f)
        self.assertEqual(record.state, MarketState.HIGH_VOLATILITY)

    def test_breakout_classification(self):
        f = make_features(
            breakout_strength=0.90, atr_expansion=0.80,
            range_expansion=0.85, impulse_strength=0.90,
            directional_persistence=0.80,
        )
        record = self._classify(f)
        self.assertIn(record.state, (
            MarketState.BREAKOUT, MarketState.STRONG_TREND,
            MarketState.HIGH_VOLATILITY, MarketState.EXPANSION,
        ))


class TestMicrostructureDatabase(unittest.TestCase):
    def setUp(self):
        self.db = MicrostructureDatabase(":memory:")

    def test_save_features(self):
        f = MicrostructureFeatures()
        self.db.save_features(f)
        self.assertEqual(self.db.count_features(), 1)

    def test_save_multiple_features(self):
        for _ in range(5):
            self.db.save_features(MicrostructureFeatures())
        self.assertEqual(self.db.count_features(), 5)

    def test_save_embedding(self):
        emb = MicrostructureEmbedding(
            timestamp=time.time(), embedding=[0.5] * 32,
            price_action_state=[0.5]*8, liquidity_state=[0.5]*6,
            momentum_state=[0.5]*6, volatility_state=[0.5]*6,
            efficiency_state=[0.5]*6,
        )
        self.db.save_embedding(emb)
        self.assertEqual(self.db.count_embeddings(), 1)

    def test_save_market_state(self):
        f = make_features()
        score = MicrostructureScorer().score(f)
        record = MarketStateClassifier().classify(f, score)
        self.db.save_market_state(record, "EURUSD", "H1")
        self.assertEqual(self.db.count_market_states(), 1)

    def test_save_score(self):
        f = make_features()
        score = MicrostructureScorer().score(f)
        self.db.save_score(score, time.time(), "EURUSD", "H1")
        self.assertEqual(self.db.count_scores(), 1)

    def test_save_liquidity(self):
        f = make_features()
        self.db.save_liquidity(f, "NORMAL")
        with self.db._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM liquidity_estimates").fetchone()[0]
        self.assertEqual(count, 1)

    def test_save_momentum(self):
        f = make_features()
        self.db.save_momentum(f, "ACCELERATING")
        with self.db._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM momentum_metrics").fetchone()[0]
        self.assertEqual(count, 1)

    def test_state_distribution(self):
        f = make_features()
        score = MicrostructureScorer().score(f)
        record = MarketStateClassifier().classify(f, score)
        for _ in range(3):
            self.db.save_market_state(record, "EURUSD", "H1")
        dist = self.db.state_distribution()
        self.assertIn(record.state.value, dist)
        self.assertEqual(dist[record.state.value], 3)

    def test_recent_states(self):
        f = make_features()
        score = MicrostructureScorer().score(f)
        record = MarketStateClassifier().classify(f, score)
        self.db.save_market_state(record, "EURUSD", "H1")
        rows = self.db.recent_states()
        self.assertEqual(len(rows), 1)

    def test_stats(self):
        self.db.save_features(MicrostructureFeatures())
        s = self.db.stats()
        self.assertEqual(s["features"], 1)
        self.assertIn("embeddings", s)


class TestMicrostructureDashboard(unittest.TestCase):
    def setUp(self):
        self.dash    = MicrostructureDashboard()
        self.scorer  = MicrostructureScorer()
        self.clf     = MarketStateClassifier()
        self.embedder = MicrostructureEmbedder()
        bars = make_trending_bars(n=10)
        self.features = []
        self.scores   = []
        self.states   = []
        self.embeddings = []
        for b in bars:
            f = MicrostructureFeatures(timestamp=b.timestamp)
            f.trend_energy = 0.70
            f.directional_persistence = 0.75
            s = self.scorer.score(f)
            sr = self.clf.classify(f, s)
            e = self.embedder.embed(f)
            self.features.append(f)
            self.scores.append(s)
            self.states.append(sr)
            self.embeddings.append(e)

    def test_microstructure_timeline(self):
        d = self.dash.microstructure_timeline(self.features, self.scores)
        self.assertEqual(d["view"], "MICROSTRUCTURE_TIMELINE")
        self.assertEqual(d["total"], len(self.features))

    def test_market_state_timeline(self):
        d = self.dash.market_state_timeline(self.states)
        self.assertEqual(d["view"], "MARKET_STATE_TIMELINE")

    def test_momentum_evolution(self):
        d = self.dash.momentum_evolution(self.features)
        self.assertEqual(d["view"], "MOMENTUM_EVOLUTION")
        self.assertIn("trend_energy", d["series"])

    def test_volatility_evolution(self):
        d = self.dash.volatility_evolution(self.features)
        self.assertEqual(d["view"], "VOLATILITY_EVOLUTION")
        self.assertIn("atr_expansion", d["series"])

    def test_liquidity_estimates(self):
        d = self.dash.liquidity_estimates(self.features, self.states)
        self.assertEqual(d["view"], "LIQUIDITY_ESTIMATES")

    def test_microstructure_heatmaps(self):
        d = self.dash.microstructure_heatmaps(self.features, self.states)
        self.assertEqual(d["view"], "MICROSTRUCTURE_HEATMAPS")

    def test_embedding_explorer(self):
        d = self.dash.embedding_explorer(self.embeddings)
        self.assertEqual(d["view"], "EMBEDDING_EXPLORER")
        self.assertEqual(d["dim"], 32)
        self.assertEqual(d["total"], len(self.embeddings))

    def test_all_dashboards_7_views(self):
        all_d = self.dash.all_dashboards(
            self.features, self.scores, self.states, self.embeddings
        )
        self.assertEqual(len(all_d), 7)

    def test_empty_returns_gracefully(self):
        d = self.dash.microstructure_timeline([], [])
        self.assertEqual(d["periods"], [])


class TestMicrostructureAPI(unittest.TestCase):
    def setUp(self):
        self.api  = MicrostructureAPI(db_path=":memory:")
        self.bars = make_trending_bars(n=20)

    def test_analyse_returns_all_keys(self):
        result = self.api.analyse(self.bars, "EURUSD", "H1")
        self.assertIn("features", result)
        self.assertIn("score", result)
        self.assertIn("market_state", result)
        self.assertIn("embedding", result)

    def test_analyse_empty_bars(self):
        result = self.api.analyse([], "EURUSD", "H1")
        self.assertIn("error", result)

    def test_current_state_after_analyse(self):
        self.api.analyse(self.bars, "EURUSD", "H1")
        state = self.api.current_state()
        self.assertIsNotNone(state)
        self.assertIn("state", state)

    def test_current_state_before_analyse(self):
        api2 = MicrostructureAPI(":memory:")
        self.assertIsNone(api2.current_state())

    def test_embedding_after_analyse(self):
        self.api.analyse(self.bars, "EURUSD", "H1")
        emb = self.api.embedding()
        self.assertIsNotNone(emb)
        self.assertEqual(emb["dim"], 32)

    def test_scores_after_analyse(self):
        self.api.analyse(self.bars, "EURUSD", "H1")
        scores = self.api.scores()
        self.assertEqual(len(scores), 1)
        self.assertIn("trend_score", scores[0])

    def test_historical(self):
        for _ in range(3):
            self.api.analyse(self.bars, "EURUSD", "H1")
        hist = self.api.historical()
        self.assertEqual(hist["total"], 3)

    def test_get_dashboard_all(self):
        self.api.analyse(self.bars, "EURUSD", "H1")
        all_d = self.api.get_dashboard("all")
        self.assertEqual(len(all_d), 7)

    def test_get_dashboard_single(self):
        self.api.analyse(self.bars, "EURUSD", "H1")
        d = self.api.get_dashboard("momentum_evolution")
        self.assertEqual(d["view"], "MOMENTUM_EVOLUTION")

    def test_get_dashboard_unknown(self):
        d = self.api.get_dashboard("nonexistent")
        self.assertIn("error", d)

    def test_system_stats(self):
        self.api.analyse(self.bars, "EURUSD", "H1")
        s = self.api.system_stats()
        self.assertIn("in_memory", s)
        self.assertIn("database", s)
        self.assertEqual(s["embedding_dim"], 32)
        self.assertFalse(s["tick_available"])
        self.assertFalse(s["order_flow_available"])

    def test_order_flow_status(self):
        report = self.api.order_flow_status()
        self.assertFalse(report["available"])
        self.assertEqual(report["current_source"], "OHLC")

    def test_multiple_analyses_accumulate(self):
        for _ in range(5):
            self.api.analyse(self.bars, "EURUSD", "H1")
        s = self.api.system_stats()
        self.assertEqual(s["in_memory"]["features"], 5)
        self.assertEqual(s["database"]["features"], 5)

    def test_scores_last_n(self):
        for _ in range(5):
            self.api.analyse(self.bars, "EURUSD", "H1")
        s3 = self.api.scores(last_n=3)
        self.assertEqual(len(s3), 3)


if __name__ == "__main__":
    unittest.main()
