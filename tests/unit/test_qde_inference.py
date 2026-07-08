"""Unit tests for live QDE inference (`trainer/qde/qde_inference.py`) and its
active-model resolution (`trainer/qde/qde_registry.py`), plus the
`PositionSizer.from_decision` / `TradeManager.plan_from_decision` executors.

Unlike `tests/unit/test_execution.py`, these modules have no torch
dependency in their own import chains (verified via
`trainer/registry/__init__.py`'s lazy `ModelRegistry` export), so this file
uses normal package imports rather than `spec_from_file_location`.

Run directly (bypassing the torch-importing `tests/conftest.py`), e.g.:
    python -m pytest tests/unit/test_qde_inference.py --noconftest -o addopts="" -q
"""

from __future__ import annotations

import sys
import types
import unittest

# ── Stub loguru before any trainer imports ──────────────────────────────────

if "loguru" not in sys.modules:
    _loguru = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, _name):
            return lambda *a, **k: None

        def bind(self, **k):
            return self

    _loguru.logger = _Logger()
    sys.modules["loguru"] = _loguru

from trainer.execution.execution_types import ExecutionForecast, ManagementAction, Session
from trainer.execution.position_sizer import PositionSizer
from trainer.execution.stop_loss_engine import StopLossEngine
from trainer.execution.take_profit_engine import TakeProfitEngine
from trainer.execution.trade_manager import TradeManager
from trainer.lifecycle.lifecycle_models import PositionMetrics
from trainer.qde.qde_inference import (
    EntryAction,
    ExitAction,
    HeuristicQDEDecisionModel,
    QDEAccountSnapshot,
    QDEDecision,
    QDEDecisionModel,
    QDEInferenceInput,
    QDEMemorySnapshot,
    ScaleAction,
    default_portfolio_snapshot,
)
from trainer.qde.qde_models import QDEStage, QDEVersion
from trainer.qde.qde_registry import QDERegistry
from trainer.registry.unified_registry import UnifiedModelRegistry


def _fc(
    instrument="EURUSD",
    direction="BUY",
    confidence=0.72,
    current_price=1.0850,
    atr=0.0012,
    spread=0.0001,
    account_size=10_000.0,
    news_risk=0.1,
    momentum=0.3,
    volatility=0.009,
):
    return ExecutionForecast(
        instrument=instrument,
        direction=direction,
        confidence=confidence,
        expected_return=0.08,
        expected_risk=0.10,
        current_price=current_price,
        atr=atr,
        spread=spread,
        session=Session.LONDON,
        volatility=volatility,
        momentum=momentum,
        news_risk=news_risk,
        account_size=account_size,
    )


def _build_input(
    forecast=None,
    position=None,
    account=None,
    memory=None,
    market=None,
):
    from trainer.lifecycle.lifecycle_models import MarketContext

    fc = forecast or _fc()
    return QDEInferenceInput(
        forecast=fc,
        market=market
        or MarketContext(
            price=fc.current_price,
            atr=fc.atr,
            spread=fc.spread,
            volatility=fc.volatility,
            momentum=fc.momentum,
            news_risk=fc.news_risk,
            session=fc.session.value,
            trend_valid=True,
        ),
        portfolio=default_portfolio_snapshot(),
        position=position or PositionMetrics(),
        account=account or QDEAccountSnapshot(),
        memory=memory or QDEMemorySnapshot(instrument=fc.instrument),
        entry_price_hint=fc.current_price,
        stop_price_hint=fc.current_price - fc.atr * 1.5,
        take_profit_hint=fc.current_price + fc.atr * 3,
    )


# ════════════════════════════════════════════════════════════════════════════
# Snapshot types
# ════════════════════════════════════════════════════════════════════════════
class TestQDEAccountSnapshot(unittest.TestCase):
    def test_free_margin_pct(self):
        acc = QDEAccountSnapshot(equity=10_000.0, margin_available=4_000.0)
        self.assertAlmostEqual(acc.free_margin_pct, 0.4)

    def test_free_margin_pct_zero_equity(self):
        acc = QDEAccountSnapshot(equity=0.0, margin_available=100.0)
        self.assertEqual(acc.free_margin_pct, 0.0)

    def test_to_dict_keys(self):
        d = QDEAccountSnapshot().to_dict()
        for key in ("balance", "equity", "margin_available", "open_positions_count"):
            self.assertIn(key, d)


class TestQDEMemorySnapshot(unittest.TestCase):
    def _record(self, direction_correct, pnl_pct, decision_type_value="BUY", grade="B"):
        class _Outcome:
            actual_direction_correct = direction_correct
            actual_pnl_pct = pnl_pct

        class _DecisionType:
            value = decision_type_value

        record = types.SimpleNamespace(
            has_outcome=True,
            outcome=_Outcome(),
            decision_type=_DecisionType(),
            grade=grade,
        )
        return record

    def test_empty_records_returns_defaults(self):
        snap = QDEMemorySnapshot.from_records("EURUSD", [])
        self.assertEqual(snap.instrument, "EURUSD")
        self.assertEqual(snap.recent_decision_count, 0)
        self.assertIsNone(snap.recent_win_rate)

    def test_from_records_computes_win_rate_and_avg_pnl(self):
        records = [
            self._record(True, 0.02),
            self._record(False, -0.01),
        ]
        snap = QDEMemorySnapshot.from_records("EURUSD", records)
        self.assertEqual(snap.recent_decision_count, 2)
        self.assertAlmostEqual(snap.recent_win_rate, 0.5)
        self.assertAlmostEqual(snap.recent_avg_pnl_pct, 0.005)
        self.assertEqual(snap.last_decision_type, "BUY")
        self.assertEqual(snap.last_decision_grade, "B")

    def test_to_dict_handles_none_win_rate(self):
        d = QDEMemorySnapshot(instrument="EURUSD").to_dict()
        self.assertIsNone(d["recent_win_rate"])


class TestDefaultPortfolioSnapshot(unittest.TestCase):
    def test_neutral_defaults(self):
        p = default_portfolio_snapshot()
        self.assertEqual(p.instrument_count, 0)
        self.assertEqual(p.diversification_score, 1.0)
        self.assertEqual(p.correlation_risk, 0.0)


# ════════════════════════════════════════════════════════════════════════════
# QDEDecision
# ════════════════════════════════════════════════════════════════════════════
class TestQDEDecision(unittest.TestCase):
    def test_is_entry_true_only_for_enter(self):
        d = QDEDecision(instrument="EURUSD", direction="BUY", entry_action=EntryAction.ENTER)
        self.assertTrue(d.is_entry)
        d2 = QDEDecision(instrument="EURUSD", direction="BUY", entry_action=EntryAction.WAIT)
        self.assertFalse(d2.is_entry)

    def test_primary_management_action_defaults_to_hold(self):
        d = QDEDecision(instrument="EURUSD", direction="BUY", entry_action=EntryAction.WAIT)
        self.assertEqual(d.primary_management_action, ManagementAction.HOLD)

    def test_primary_management_action_uses_first_entry(self):
        d = QDEDecision(
            instrument="EURUSD",
            direction="BUY",
            entry_action=EntryAction.ENTER,
            management_actions=[ManagementAction.SCALE_IN, ManagementAction.TRAIL_STOP],
        )
        self.assertEqual(d.primary_management_action, ManagementAction.SCALE_IN)

    def test_to_dict_keys(self):
        d = QDEDecision(instrument="EURUSD", direction="BUY", entry_action=EntryAction.ENTER)
        out = d.to_dict()
        for key in (
            "instrument",
            "entry_action",
            "exit_action",
            "scale_action",
            "position_size_lots",
            "risk_allocation_pct",
            "management_actions",
            "confidence",
        ):
            self.assertIn(key, out)


# ════════════════════════════════════════════════════════════════════════════
# HeuristicQDEDecisionModel
# ════════════════════════════════════════════════════════════════════════════
class TestHeuristicQDEDecisionModel(unittest.TestCase):
    def setUp(self):
        self.model = HeuristicQDEDecisionModel(version=None)

    def test_is_a_qde_decision_model(self):
        self.assertIsInstance(self.model, QDEDecisionModel)

    def test_rejects_on_low_free_margin(self):
        account = QDEAccountSnapshot(equity=10_000.0, margin_available=100.0)  # 1%
        decision = self.model.decide(_build_input(account=account))
        self.assertEqual(decision.entry_action, EntryAction.REJECT)
        self.assertIn("margin", decision.rationale.lower())

    def test_rejects_on_low_conviction(self):
        decision = self.model.decide(_build_input(forecast=_fc(confidence=0.1)))
        self.assertEqual(decision.entry_action, EntryAction.REJECT)

    def test_waits_on_marginal_conviction(self):
        decision = self.model.decide(_build_input(forecast=_fc(confidence=0.50, news_risk=0.0)))
        self.assertEqual(decision.entry_action, EntryAction.WAIT)

    def test_enters_on_high_conviction(self):
        decision = self.model.decide(_build_input(forecast=_fc(confidence=0.9, news_risk=0.0)))
        self.assertEqual(decision.entry_action, EntryAction.ENTER)
        self.assertGreater(decision.position_size_lots, 0.0)
        self.assertGreater(decision.risk_allocation_pct, 0.0)

    def test_exits_existing_position_on_high_news_risk(self):
        position = PositionMetrics(lots=0.5, entry_price=1.08, current_price=1.085)
        decision = self.model.decide(
            _build_input(forecast=_fc(confidence=0.9, news_risk=0.8), position=position)
        )
        self.assertEqual(decision.exit_action, ExitAction.EXIT_NOW)
        self.assertEqual(decision.management_actions, [ManagementAction.EXIT_EARLY])

    def test_scales_in_on_very_high_conviction_and_margin(self):
        position = PositionMetrics(lots=0.5, entry_price=1.08, current_price=1.085)
        account = QDEAccountSnapshot(equity=10_000.0, margin_available=5_000.0)
        decision = self.model.decide(
            _build_input(
                forecast=_fc(confidence=0.95, news_risk=0.0),
                position=position,
                account=account,
            )
        )
        self.assertEqual(decision.scale_action, ScaleAction.SCALE_IN)
        self.assertIn(ManagementAction.SCALE_IN, decision.management_actions)

    def test_memory_win_rate_shifts_conviction(self):
        good_memory = QDEMemorySnapshot(instrument="EURUSD", recent_win_rate=1.0)
        bad_memory = QDEMemorySnapshot(instrument="EURUSD", recent_win_rate=0.0)
        fc = _fc(confidence=0.6, news_risk=0.0)
        good = self.model.decide(_build_input(forecast=fc, memory=good_memory))
        bad = self.model.decide(_build_input(forecast=fc, memory=bad_memory))
        self.assertGreater(good.confidence, bad.confidence)

    def test_version_id_propagates(self):
        import time

        now = time.time()
        version = QDEVersion(
            version_id="v-42",
            version_number=42,
            label="test",
            stage=QDEStage.ACTIVE,
            created_at=now,
            updated_at=now,
        )
        model = HeuristicQDEDecisionModel(version=version)
        decision = model.decide(_build_input(forecast=_fc(confidence=0.9, news_risk=0.0)))
        self.assertEqual(decision.model_version_id, "v-42")


# ════════════════════════════════════════════════════════════════════════════
# QDERegistry.get_active_model / register_decision_model
# ════════════════════════════════════════════════════════════════════════════
class TestQDEActiveModelResolution(unittest.TestCase):
    def setUp(self):
        self.storage_path = None  # use default tmp file; unique per registry instance below
        import tempfile

        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self.registry = QDERegistry(storage_path=self._tmp.name)

    def tearDown(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_falls_back_to_heuristic_when_no_active_version(self):
        model = self.registry.get_active_model()
        self.assertIsInstance(model, HeuristicQDEDecisionModel)

    def test_returns_heuristic_bound_to_active_version_when_none_registered(self):
        import time

        now = time.time()
        version = QDEVersion(
            version_id="v-1",
            version_number=1,
            label="v1",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
        )
        self.registry.register(version)
        self.registry.set_active("v-1")

        model = self.registry.get_active_model()
        self.assertIsInstance(model, HeuristicQDEDecisionModel)
        self.assertEqual(model.version_id, "v-1")

    def test_registered_decision_model_is_returned_for_active_version(self):
        import time

        now = time.time()
        version = QDEVersion(
            version_id="v-2",
            version_number=2,
            label="v2",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
        )
        self.registry.register(version)
        self.registry.set_active("v-2")

        class _FakeModel(QDEDecisionModel):
            def decide(self, inputs):
                return QDEDecision(
                    instrument=inputs.forecast.instrument,
                    direction=inputs.forecast.direction,
                    entry_action=EntryAction.ENTER,
                )

        fake = _FakeModel()
        self.registry.register_decision_model("v-2", fake)

        resolved = self.registry.get_active_model()
        self.assertIs(resolved, fake)
        self.assertEqual(resolved.version_id, "v-2")


class TestUnifiedRegistryActiveNative(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self.registry = UnifiedModelRegistry(storage_path=self._tmp.name)

    def tearDown(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_returns_none_when_no_champion(self):
        self.assertIsNone(self.registry.get_active_native("qde"))


# ════════════════════════════════════════════════════════════════════════════
# PositionSizer.from_decision / TradeManager.plan_from_decision
# ════════════════════════════════════════════════════════════════════════════
class TestPositionSizerFromDecision(unittest.TestCase):
    def setUp(self):
        self.sizer = PositionSizer()
        self.stop_engine = StopLossEngine()
        from trainer.execution.entry_zones import EntryZoneEngine

        self.zone_engine = EntryZoneEngine()

    def test_zero_lots_when_decision_has_no_size(self):
        fc = _fc()
        decision = QDEDecision(instrument="EURUSD", direction="BUY", entry_action=EntryAction.REJECT)
        result = self.sizer.from_decision(fc, decision)
        self.assertEqual(result.lots, 0.0)
        self.assertEqual(result.method, "QDE_DECISION")

    def test_lots_reflect_decision_size(self):
        fc = _fc()
        zones = self.zone_engine.calculate(fc)
        stop = self.stop_engine.recommend(fc, zones)
        decision = QDEDecision(
            instrument="EURUSD",
            direction="BUY",
            entry_action=EntryAction.ENTER,
            position_size_lots=1.0,
            risk_allocation_pct=0.01,
        )
        result = self.sizer.from_decision(fc, decision, stop)
        self.assertGreater(result.lots, 0.0)
        self.assertGreater(result.margin_required, 0.0)


class TestTradeManagerFromDecision(unittest.TestCase):
    def setUp(self):
        self.manager = TradeManager()
        self.stop_engine = StopLossEngine()
        self.tp_engine = TakeProfitEngine()
        from trainer.execution.entry_zones import EntryZoneEngine

        self.zone_engine = EntryZoneEngine()

    def test_primary_action_matches_decision(self):
        fc = _fc()
        zones = self.zone_engine.calculate(fc)
        stop = self.stop_engine.recommend(fc, zones)
        tp = self.tp_engine.recommend(fc, zones, stop)
        decision = QDEDecision(
            instrument="EURUSD",
            direction="BUY",
            entry_action=EntryAction.ENTER,
            management_actions=[ManagementAction.SCALE_IN, ManagementAction.TRAIL_STOP],
        )
        plan = self.manager.plan_from_decision(fc, decision, stop_loss=stop, take_profit=tp)
        self.assertEqual(plan.primary_action, ManagementAction.SCALE_IN)
        rule_actions = [r.action for r in plan.rules]
        self.assertIn(ManagementAction.TRAIL_STOP, rule_actions)

    def test_defaults_to_hold_when_no_management_actions(self):
        fc = _fc()
        decision = QDEDecision(instrument="EURUSD", direction="BUY", entry_action=EntryAction.WAIT)
        plan = self.manager.plan_from_decision(fc, decision)
        self.assertEqual(plan.primary_action, ManagementAction.HOLD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
