"""Unit tests for Phase S3 — the live position-management loop."""

from __future__ import annotations

import unittest

from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.qde.qde_outputs import Direction, QDEOutputHeads
from trainer.qde.qde_position_manager import (
    REVIEW_INTERVAL_SECONDS,
    PositionManagementAction,
    PositionTelemetry,
    QDEPositionManager,
)


def _telemetry(
    unrealised_pct: float = 0.0,
    news_risk: float = 0.1,
    free_margin_pct: float = 1.0,
    seconds_since_last_review: float = REVIEW_INTERVAL_SECONDS,
    **head_overrides,
) -> PositionTelemetry:
    heads = QDEOutputHeads(
        instrument="EURUSD",
        direction=Direction.LONG,
        confidence=head_overrides.pop("confidence", 0.6),
        reduce_signal=head_overrides.pop("reduce_signal", 0.0),
        scale_signal=head_overrides.pop("scale_signal", 0.0),
        exit_horizon=head_overrides.pop("exit_horizon", 0.6),
        management_confidence=head_overrides.pop("management_confidence", 0.6),
        risk_signal=head_overrides.pop("risk_signal", 0.2),
        **head_overrides,
    )
    return PositionTelemetry(
        instrument="EURUSD",
        position=PositionMetrics(lots=1.0, unrealised_pct=unrealised_pct),
        market=MarketContext(news_risk=news_risk),
        heads=heads,
        free_margin_pct=free_margin_pct,
        seconds_since_last_review=seconds_since_last_review,
    )


class TestQDEPositionManager(unittest.TestCase):
    def setUp(self):
        self.mgr = QDEPositionManager(version_id="qde-v1")

    def test_due_for_review_respects_interval(self):
        self.assertTrue(self.mgr.due_for_review(_telemetry(seconds_since_last_review=30.0)))
        self.assertFalse(self.mgr.due_for_review(_telemetry(seconds_since_last_review=10.0)))

    def test_emergency_news_risk_forces_close(self):
        decision = self.mgr.review(_telemetry(news_risk=0.9, reduce_signal=0.0))
        self.assertEqual(decision.action, PositionManagementAction.CLOSE)
        self.assertEqual(decision.confidence, 1.0)

    def test_high_reduce_signal_closes(self):
        decision = self.mgr.review(_telemetry(reduce_signal=0.7))
        self.assertEqual(decision.action, PositionManagementAction.CLOSE)

    def test_moderate_reduce_signal_reduces(self):
        decision = self.mgr.review(_telemetry(reduce_signal=0.4))
        self.assertEqual(decision.action, PositionManagementAction.REDUCE)

    def test_high_scale_signal_with_margin_and_profit_adds(self):
        decision = self.mgr.review(
            _telemetry(unrealised_pct=0.01, scale_signal=0.8, free_margin_pct=0.5)
        )
        self.assertEqual(decision.action, PositionManagementAction.ADD)

    def test_scale_signal_ignored_without_free_margin(self):
        decision = self.mgr.review(
            _telemetry(unrealised_pct=0.01, scale_signal=0.8, free_margin_pct=0.1)
        )
        self.assertNotEqual(decision.action, PositionManagementAction.ADD)

    def test_profit_with_fading_confidence_locks_profit(self):
        decision = self.mgr.review(
            _telemetry(unrealised_pct=0.03, confidence=0.4, reduce_signal=0.1, exit_horizon=0.6)
        )
        self.assertEqual(decision.action, PositionManagementAction.LOCK_PROFIT)

    def test_profit_with_short_horizon_moves_to_breakeven(self):
        decision = self.mgr.review(
            _telemetry(unrealised_pct=0.005, confidence=0.6, exit_horizon=0.2, reduce_signal=0.0)
        )
        self.assertEqual(decision.action, PositionManagementAction.MOVE_TO_BREAKEVEN)

    def test_profit_with_confidence_trails(self):
        decision = self.mgr.review(
            _telemetry(
                unrealised_pct=0.005,
                confidence=0.6,
                exit_horizon=0.7,
                management_confidence=0.6,
                reduce_signal=0.0,
            )
        )
        self.assertEqual(decision.action, PositionManagementAction.TRAIL)

    def test_no_signal_crossed_holds(self):
        decision = self.mgr.review(
            _telemetry(unrealised_pct=0.0, confidence=0.5, reduce_signal=0.0, scale_signal=0.0)
        )
        self.assertEqual(decision.action, PositionManagementAction.HOLD)

    def test_decision_always_carries_required_fields(self):
        decision = self.mgr.review(_telemetry())
        self.assertIsInstance(decision.action, PositionManagementAction)
        self.assertIsInstance(decision.confidence, float)
        self.assertTrue(decision.reasoning)
        self.assertIsInstance(decision.management_confidence, float)
        self.assertIsInstance(decision.risk_signal, float)

    def test_to_dict_serializes_all_required_fields(self):
        decision = self.mgr.review(_telemetry())
        d = decision.to_dict()
        for key in ("action", "confidence", "reasoning", "management_confidence", "risk_signal"):
            self.assertIn(key, d)

    def test_all_seven_actions_are_reachable(self):
        self.assertEqual(
            {a.value for a in PositionManagementAction},
            {
                "HOLD",
                "REDUCE",
                "ADD",
                "CLOSE",
                "MOVE_TO_BREAKEVEN",
                "TRAIL",
                "LOCK_PROFIT",
            },
        )


if __name__ == "__main__":
    unittest.main()
