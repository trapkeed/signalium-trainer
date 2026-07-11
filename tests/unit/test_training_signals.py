"""Unit tests for Phase S4 hindsight training-signal labelers."""

from __future__ import annotations

import unittest

from trainer.lifecycle.lifecycle_states import LifecycleState
from trainer.microstructure.micro_models import OHLCBar
from trainer.training_signals.lifecycle_transition_labels import HindsightLifecycleLabeler
from trainer.training_signals.outcome_path import TradeOutcomePath
from trainer.training_signals.portfolio_allocation_labels import HindsightAllocationLabeler
from trainer.training_signals.position_size_labels import HindsightPositionSizeLabeler
from trainer.training_signals.stop_loss_labels import HindsightStopLossLabeler
from trainer.training_signals.take_profit_labels import HindsightTakeProfitLabeler


def _bar(t: float, o: float, h: float, low: float, c: float) -> OHLCBar:
    return OHLCBar(timestamp=t, open=o, high=h, low=low, close=c)


def _winning_long_path(instrument: str = "EURUSD") -> TradeOutcomePath:
    entry = 1.1000
    bars = [
        _bar(1, entry, 1.1010, 1.0990, 1.1005),  # small dip then up
        _bar(2, 1.1005, 1.1030, 1.1000, 1.1025),
        _bar(3, 1.1025, 1.1060, 1.1020, 1.1055),  # MFE bar
        _bar(4, 1.1055, 1.1058, 1.1040, 1.1045),  # pulls back a bit
        _bar(5, 1.1045, 1.1050, 1.1035, 1.1040),  # closes off the highs
    ]
    return TradeOutcomePath(instrument=instrument, direction=1, entry_price=entry, future_bars=bars)


def _losing_long_path(instrument: str = "GBPUSD") -> TradeOutcomePath:
    entry = 1.2500
    bars = [
        _bar(1, entry, 1.2505, 1.2470, 1.2480),
        _bar(2, 1.2480, 1.2485, 1.2430, 1.2440),
        _bar(3, 1.2440, 1.2445, 1.2400, 1.2410),
    ]
    return TradeOutcomePath(instrument=instrument, direction=1, entry_price=entry, future_bars=bars)


def _flat_path(instrument: str = "USDJPY") -> TradeOutcomePath:
    entry = 150.00
    return TradeOutcomePath(instrument=instrument, direction=1, entry_price=entry, future_bars=[])


class TestOutcomePath(unittest.TestCase):
    def test_favorable_and_adverse_excursion_long(self):
        path = _winning_long_path()
        mfe, mfe_idx = path.max_favorable_excursion()
        self.assertAlmostEqual(mfe, 1.1060 - 1.1000, places=5)
        self.assertEqual(mfe_idx, 2)

    def test_realized_return_pct_long(self):
        path = _winning_long_path()
        expected = (1.1040 - 1.1000) / 1.1000
        self.assertAlmostEqual(path.realized_return_pct(), expected, places=6)

    def test_short_direction_excursions_are_mirrored(self):
        entry = 1.5000
        bars = [_bar(1, entry, 1.5010, 1.4950, 1.4960)]
        path = TradeOutcomePath(instrument="EURUSD", direction=-1, entry_price=entry, future_bars=bars)
        self.assertAlmostEqual(path.favorable_excursion(bars[0]), entry - 1.4950, places=5)
        self.assertAlmostEqual(path.adverse_excursion(bars[0]), 1.5010 - entry, places=5)


class TestHindsightStopLossLabeler(unittest.TestCase):
    def setUp(self):
        self.labeler = HindsightStopLossLabeler()

    def test_winning_trade_stop_survives_to_mfe(self):
        label = self.labeler.label(_winning_long_path())
        self.assertEqual(label.survives_to_index, 2)
        self.assertGreater(label.optimal_stop_distance, 0.0)
        self.assertLess(label.optimal_stop_price, label.entry_price)

    def test_no_future_bars_returns_zero_distance(self):
        label = self.labeler.label(_flat_path())
        self.assertEqual(label.optimal_stop_distance, 0.0)
        self.assertEqual(label.survives_to_index, -1)

    def test_losing_trade_uses_mae_through_best_point_reached(self):
        # The path here reaches its (small) MFE at bar 0, so the hindsight
        # stop only needs to survive the adverse excursion within bar 0 —
        # not the full-path MAE, which happens later and is irrelevant once
        # the best achievable exit has already passed.
        path = _losing_long_path()
        label = self.labeler.label(path)
        mfe, mfe_idx = path.max_favorable_excursion()
        self.assertGreater(mfe, 0.0)
        expected = path.max_adverse_excursion_through(mfe_idx)
        self.assertAlmostEqual(label.optimal_stop_distance, expected, places=5)
        self.assertLess(label.optimal_stop_distance, path.max_adverse_excursion())


class TestHindsightTakeProfitLabeler(unittest.TestCase):
    def setUp(self):
        self.labeler = HindsightTakeProfitLabeler()

    def test_target_equals_mfe_price(self):
        path = _winning_long_path()
        label = self.labeler.label(path)
        self.assertAlmostEqual(label.optimal_target_price, 1.1060, places=5)
        self.assertEqual(label.bars_to_reach, 3)

    def test_never_favorable_gives_zero_distance(self):
        entry = 1.0
        bars = [_bar(1, entry, entry, entry - 0.01, entry - 0.005)]
        path = TradeOutcomePath(instrument="EURUSD", direction=1, entry_price=entry, future_bars=bars)
        label = self.labeler.label(path)
        self.assertEqual(label.optimal_target_distance, 0.0)


class TestHindsightPositionSizeLabeler(unittest.TestCase):
    def setUp(self):
        self.labeler = HindsightPositionSizeLabeler()

    def test_winning_trade_label_above_half(self):
        label = self.labeler.label(_winning_long_path())
        self.assertGreater(label.size_signal_label, 0.5)
        self.assertTrue(0.0 <= label.size_signal_label <= 1.0)

    def test_losing_trade_label_below_half(self):
        label = self.labeler.label(_losing_long_path())
        self.assertLess(label.size_signal_label, 0.5)

    def test_zero_risk_positive_return_saturates_toward_high_end(self):
        # Zero realized adverse excursion caps risk_adjusted_return at the
        # saturation constant (tanh(1) ~= 0.762), landing above the neutral
        # midpoint but below the asymptotic ceiling of 1.0.
        entry = 1.0
        bars = [_bar(1, entry, entry + 0.01, entry, entry + 0.01)]
        path = TradeOutcomePath(instrument="EURUSD", direction=1, entry_price=entry, future_bars=bars)
        label = self.labeler.label(path)
        self.assertGreater(label.size_signal_label, 0.85)
        self.assertLess(label.size_signal_label, 1.0)


class TestHindsightAllocationLabeler(unittest.TestCase):
    def setUp(self):
        self.labeler = HindsightAllocationLabeler()

    def test_weights_sum_to_one_when_positive_scores_exist(self):
        label = self.labeler.label([_winning_long_path("EURUSD"), _losing_long_path("GBPUSD")])
        total = sum(label.optimal_weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertGreater(label.optimal_weights["EURUSD"], label.optimal_weights["GBPUSD"])

    def test_all_losers_gives_all_zero_weights(self):
        label = self.labeler.label([_losing_long_path("AUDUSD"), _losing_long_path("NZDUSD")])
        self.assertTrue(all(w == 0.0 for w in label.optimal_weights.values()))

    def test_empty_input_returns_empty_label(self):
        label = self.labeler.label([])
        self.assertEqual(label.instruments, [])
        self.assertEqual(label.optimal_weights, {})


class TestHindsightLifecycleLabeler(unittest.TestCase):
    def setUp(self):
        self.labeler = HindsightLifecycleLabeler()

    def test_winning_trade_produces_full_sequence(self):
        label = self.labeler.label(_winning_long_path())
        states = [t.state for t in label.transitions]
        self.assertEqual(states[0], LifecycleState.ACTIVE)
        self.assertIn(LifecycleState.PROTECTED, states)
        self.assertEqual(states[-1], LifecycleState.CLOSED)

    def test_never_favorable_goes_straight_to_stopped(self):
        entry = 1.0
        bars = [_bar(1, entry, entry, entry - 0.01, entry - 0.005)]
        path = TradeOutcomePath(instrument="EURUSD", direction=1, entry_price=entry, future_bars=bars)
        label = self.labeler.label(path)
        states = [t.state for t in label.transitions]
        self.assertEqual(states, [LifecycleState.ACTIVE, LifecycleState.STOPPED])

    def test_no_future_bars_only_active(self):
        label = self.labeler.label(_flat_path())
        self.assertEqual(len(label.transitions), 1)
        self.assertEqual(label.transitions[0].state, LifecycleState.ACTIVE)

    def test_transitions_are_monotonic_in_bar_index(self):
        label = self.labeler.label(_winning_long_path())
        indices = [t.bar_index for t in label.transitions]
        self.assertEqual(indices, sorted(indices))


if __name__ == "__main__":
    unittest.main()
