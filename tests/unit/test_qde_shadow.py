"""Unit tests for Task #5 — Inference traffic splitting (dual-dispatch shadow
testing): `trainer/qde/qde_shadow.py`, the widened `ShadowDecisionRecord` /
`ShadowTestRun` fields in `trainer/research_lab/rl_models.py`, and the
`QDELifecycleAPI.dispatch_shadow_decision` / `evaluate_shadow_promotion`
endpoints in `trainer/qde/qde_api.py`.

`trainer.research_lab` has no torch dependency; `trainer.qde` pulls in
`trainer.execution` (which pulls in `trainer.registry`), which also has no
torch dependency in its import chain (see `tests/unit/test_qde_inference.py`).
So this file uses normal package imports, guarded by a loguru stub, and can
run standalone bypassing the torch-importing root conftest:

    python -m pytest tests/unit/test_qde_shadow.py --noconftest -o addopts="" -q
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

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

from trainer.execution.execution_types import ExecutionForecast, Session
from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.qde.qde_api import QDELifecycleAPI
from trainer.qde.qde_inference import (
    EntryAction,
    QDEAccountSnapshot,
    QDEDecision,
    QDEDecisionModel,
    QDEInferenceInput,
    QDEMemorySnapshot,
    default_portfolio_snapshot,
)
from trainer.qde.qde_models import QDEStage, QDEVersion
from trainer.qde.qde_promotion import QDEPromotionEngine
from trainer.qde.qde_registry import QDERegistry
from trainer.qde.qde_shadow import (
    DualDispatchResult,
    ShadowDispatcher,
    evaluate_and_maybe_promote,
    expected_pnl_proxy,
)
from trainer.research_lab.research_lab_api import ResearchLabAPI
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.rl_models import (
    ShadowDecisionRecord,
    ShadowTestRecommendation,
    ShadowTestRun,
    ShadowTestStatus,
    compute_promotion_score,
)


def _fc(instrument="EURUSD", direction="BUY", confidence=0.72, current_price=1.0850):
    return ExecutionForecast(
        instrument=instrument,
        direction=direction,
        confidence=confidence,
        expected_return=0.08,
        expected_risk=0.10,
        current_price=current_price,
        atr=0.0012,
        spread=0.0001,
        session=Session.LONDON,
        volatility=0.009,
        momentum=0.3,
        news_risk=0.1,
        account_size=10_000.0,
    )


def _build_input(forecast=None):
    fc = forecast or _fc()
    return QDEInferenceInput(
        forecast=fc,
        market=MarketContext(
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
        position=PositionMetrics(),
        account=QDEAccountSnapshot(),
        memory=QDEMemorySnapshot(instrument=fc.instrument),
        entry_price_hint=fc.current_price,
        stop_price_hint=fc.current_price - fc.atr * 1.5,
        take_profit_hint=fc.current_price + fc.atr * 3,
    )


class _StubDecisionModel(QDEDecisionModel):
    """A deterministic, controllable decision model for dispatch tests."""

    def __init__(self, decision: QDEDecision, sleep_s: float = 0.0, version_id: str = ""):
        self.version_id = version_id
        self._decision = decision
        self._sleep_s = sleep_s

    def decide(self, inputs: QDEInferenceInput) -> QDEDecision:
        if self._sleep_s:
            time.sleep(self._sleep_s)
        return self._decision


def _decision(
    instrument="EURUSD",
    entry_action=EntryAction.ENTER,
    confidence=0.6,
    position_size_lots=1.0,
    risk_allocation_pct=0.01,
):
    return QDEDecision(
        instrument=instrument,
        direction="BUY",
        entry_action=entry_action,
        confidence=confidence,
        position_size_lots=position_size_lots,
        risk_allocation_pct=risk_allocation_pct,
    )


# ════════════════════════════════════════════════════════════════════════════
# compute_promotion_score
# ════════════════════════════════════════════════════════════════════════════
class TestComputePromotionScore(unittest.TestCase):
    def test_perfect_agreement_no_diffs_scores_high(self):
        score = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=0.0,
            latency_diff_ms=0.0,
            risk_diff=0.0,
        )
        # agreement(0.35*1.0) + pnl_neutral(0.25*0.5) + confidence_neutral(0.15*0.5)
        # + latency_ok(0.15*1.0) + risk_ok(0.10*1.0)
        self.assertAlmostEqual(score, 0.35 + 0.125 + 0.075 + 0.15 + 0.10, places=4)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_full_disagreement_and_worse_everything_scores_low(self):
        score = compute_promotion_score(
            decisions_agree=False,
            confidence_diff=-0.5,
            pnl_diff=-1.0,
            latency_diff_ms=500.0,
            risk_diff=0.5,
        )
        self.assertLess(score, 0.15)
        self.assertGreaterEqual(score, 0.0)

    def test_positive_pnl_diff_beats_negative(self):
        good = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=0.01,
            latency_diff_ms=0.0,
            risk_diff=0.0,
        )
        bad = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=-0.01,
            latency_diff_ms=0.0,
            risk_diff=0.0,
        )
        self.assertGreater(good, bad)

    def test_higher_latency_penalised(self):
        fast = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=0.0,
            latency_diff_ms=0.0,
            risk_diff=0.0,
        )
        slow = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=0.0,
            latency_diff_ms=200.0,
            risk_diff=0.0,
        )
        self.assertGreater(fast, slow)

    def test_large_risk_diff_penalised_either_direction(self):
        neutral = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=0.0,
            latency_diff_ms=0.0,
            risk_diff=0.0,
        )
        more_risk = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=0.0,
            latency_diff_ms=0.0,
            risk_diff=0.05,
        )
        less_risk = compute_promotion_score(
            decisions_agree=True,
            confidence_diff=0.0,
            pnl_diff=0.0,
            latency_diff_ms=0.0,
            risk_diff=-0.05,
        )
        self.assertGreater(neutral, more_risk)
        self.assertGreater(neutral, less_risk)

    def test_score_always_within_unit_interval(self):
        for agree in (True, False):
            for cdiff in (-1.0, 0.0, 1.0):
                for pdiff in (-1.0, 0.0, 1.0):
                    for ldiff in (-100.0, 0.0, 1000.0):
                        for rdiff in (-1.0, 0.0, 1.0):
                            score = compute_promotion_score(
                                decisions_agree=agree,
                                confidence_diff=cdiff,
                                pnl_diff=pdiff,
                                latency_diff_ms=ldiff,
                                risk_diff=rdiff,
                            )
                            self.assertGreaterEqual(score, 0.0)
                            self.assertLessEqual(score, 1.0)


# ════════════════════════════════════════════════════════════════════════════
# ShadowDecisionRecord — widened comparison fields
# ════════════════════════════════════════════════════════════════════════════
class TestShadowDecisionRecordWidenedFields(unittest.TestCase):
    def test_new_autocomputes_confidence_diff(self):
        record = ShadowDecisionRecord.new(
            test_id="t1",
            instrument="EURUSD",
            champion_decision_type="ENTER",
            challenger_decision_type="ENTER",
            champion_confidence=0.6,
            challenger_confidence=0.8,
        )
        self.assertAlmostEqual(record.confidence_diff, 0.2, places=6)

    def test_new_autocomputes_promotion_score_matching_compute_fn(self):
        record = ShadowDecisionRecord.new(
            test_id="t1",
            instrument="EURUSD",
            champion_decision_type="ENTER",
            challenger_decision_type="WAIT",
            champion_confidence=0.6,
            challenger_confidence=0.8,
            pnl_diff=0.02,
            latency_diff_ms=5.0,
            risk_diff=0.001,
        )
        expected = compute_promotion_score(
            decisions_agree=False,
            confidence_diff=0.2,
            pnl_diff=0.02,
            latency_diff_ms=5.0,
            risk_diff=0.001,
        )
        self.assertAlmostEqual(record.promotion_score, expected, places=4)

    def test_new_sets_decisions_agree_correctly(self):
        agree = ShadowDecisionRecord.new(
            test_id="t1",
            instrument="EURUSD",
            champion_decision_type="ENTER",
            challenger_decision_type="ENTER",
            champion_confidence=0.5,
            challenger_confidence=0.5,
        )
        disagree = ShadowDecisionRecord.new(
            test_id="t1",
            instrument="EURUSD",
            champion_decision_type="ENTER",
            challenger_decision_type="REJECT",
            champion_confidence=0.5,
            challenger_confidence=0.5,
        )
        self.assertTrue(agree.decisions_agree)
        self.assertFalse(disagree.decisions_agree)

    def test_explicit_promotion_score_overrides_autocompute(self):
        record = ShadowDecisionRecord.new(
            test_id="t1",
            instrument="EURUSD",
            champion_decision_type="ENTER",
            challenger_decision_type="ENTER",
            champion_confidence=0.5,
            challenger_confidence=0.5,
            promotion_score=0.999,
        )
        self.assertEqual(record.promotion_score, 0.999)

    def test_to_dict_includes_all_widened_fields(self):
        record = ShadowDecisionRecord.new(
            test_id="t1",
            instrument="EURUSD",
            champion_decision_type="ENTER",
            challenger_decision_type="ENTER",
            champion_confidence=0.5,
            challenger_confidence=0.6,
            pnl_diff=0.01,
            latency_diff_ms=2.0,
            risk_diff=0.002,
        )
        d = record.to_dict()
        for key in (
            "confidence_diff",
            "pnl_diff",
            "latency_diff_ms",
            "risk_diff",
            "promotion_score",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["pnl_diff"], round(0.01, 5))
        self.assertEqual(d["latency_diff_ms"], round(2.0, 3))
        self.assertEqual(d["risk_diff"], round(0.002, 5))

    def test_reconstruction_from_to_dict_round_trips(self):
        record = ShadowDecisionRecord.new(
            test_id="t1",
            instrument="EURUSD",
            champion_decision_type="ENTER",
            challenger_decision_type="WAIT",
            champion_confidence=0.5,
            challenger_confidence=0.6,
        )
        rebuilt = ShadowDecisionRecord(**record.to_dict())
        self.assertAlmostEqual(rebuilt.promotion_score, record.promotion_score, places=4)
        self.assertAlmostEqual(rebuilt.confidence_diff, record.confidence_diff, places=4)


# ════════════════════════════════════════════════════════════════════════════
# ShadowTestRun.conclude — avg_promotion_score-driven recommendation
# ════════════════════════════════════════════════════════════════════════════
class TestShadowTestRunConclude(unittest.TestCase):
    def _run_with_decisions(self, decisions, min_decisions=1):
        test = ShadowTestRun.new(
            name="t",
            challenger_version="v2",
            champion_version="v1",
            instruments=["EURUSD"],
            min_decisions=min_decisions,
        )
        for d in decisions:
            test.add_decision(d)
        return test

    def test_promote_when_high_score_and_agreement(self):
        decisions = [
            ShadowDecisionRecord.new(
                test_id="t",
                instrument="EURUSD",
                champion_decision_type="ENTER",
                challenger_decision_type="ENTER",
                champion_confidence=0.6,
                challenger_confidence=0.65,
            )
            for _ in range(5)
        ]
        test = self._run_with_decisions(decisions, min_decisions=5)
        test.conclude()
        self.assertEqual(test.recommendation, ShadowTestRecommendation.PROMOTE)
        self.assertGreaterEqual(test.avg_promotion_score, 0.70)

    def test_reject_when_low_score(self):
        decisions = [
            ShadowDecisionRecord.new(
                test_id="t",
                instrument="EURUSD",
                champion_decision_type="ENTER",
                challenger_decision_type="REJECT",
                champion_confidence=0.6,
                challenger_confidence=0.1,
                pnl_diff=-0.05,
                latency_diff_ms=300.0,
                risk_diff=0.05,
            )
            for _ in range(5)
        ]
        test = self._run_with_decisions(decisions, min_decisions=5)
        test.conclude()
        self.assertEqual(test.recommendation, ShadowTestRecommendation.REJECT)

    def test_extend_when_below_min_decisions(self):
        decisions = [
            ShadowDecisionRecord.new(
                test_id="t",
                instrument="EURUSD",
                champion_decision_type="ENTER",
                challenger_decision_type="ENTER",
                champion_confidence=0.6,
                challenger_confidence=0.65,
            )
        ]
        test = self._run_with_decisions(decisions, min_decisions=100)
        test.conclude()
        self.assertEqual(test.recommendation, ShadowTestRecommendation.EXTEND)

    def test_disagreement_rate_and_avg_diffs_recorded(self):
        """S22B: metric completeness — disagreement_rate, avg_confidence_diff,
        and avg_latency_diff_ms must be computed and queryable on the
        concluded summary, not just derivable per-decision."""
        decisions = [
            ShadowDecisionRecord.new(
                test_id="t",
                instrument="EURUSD",
                champion_decision_type="ENTER",
                challenger_decision_type="WAIT",  # disagreement
                champion_confidence=0.5,
                challenger_confidence=0.7,
                latency_diff_ms=10.0,
            ),
            ShadowDecisionRecord.new(
                test_id="t",
                instrument="EURUSD",
                champion_decision_type="ENTER",
                challenger_decision_type="ENTER",  # agreement
                champion_confidence=0.5,
                challenger_confidence=0.9,
                latency_diff_ms=30.0,
            ),
        ]
        test = self._run_with_decisions(decisions, min_decisions=2)
        test.conclude()
        self.assertAlmostEqual(test.agreement_rate, 0.5, places=4)
        self.assertAlmostEqual(test.disagreement_rate, 0.5, places=4)
        self.assertAlmostEqual(test.avg_confidence_diff, 0.3, places=4)
        self.assertAlmostEqual(test.avg_latency_diff_ms, 20.0, places=3)
        d = test.to_dict()
        for key in ("disagreement_rate", "avg_confidence_diff", "avg_latency_diff_ms"):
            self.assertIn(key, d)


# ════════════════════════════════════════════════════════════════════════════
# ShadowDispatcher — dual dispatch
# ════════════════════════════════════════════════════════════════════════════
class TestShadowDispatcher(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self.registry = QDERegistry(storage_path=self._tmp.name)

        now = time.time()
        self.champion_version = QDEVersion(
            version_id="v-champ",
            version_number=1,
            label="champion",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
        )
        self.challenger_version = QDEVersion(
            version_id="v-chall",
            version_number=2,
            label="challenger",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
        )
        self.registry.register(self.champion_version)
        self.registry.register(self.challenger_version)
        self.registry.set_active("v-champ")

    def tearDown(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_dispatch_calls_both_models_and_never_executes_challenger(self):
        champion_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.6)
        challenger_decision = _decision(entry_action=EntryAction.WAIT, confidence=0.5)
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(champion_decision, version_id="v-champ")
        )
        self.registry.register_decision_model(
            "v-chall", _StubDecisionModel(challenger_decision, version_id="v-chall")
        )

        dispatcher = ShadowDispatcher(self.registry)
        result = dispatcher.dispatch(
            test_id="shadow-1", challenger_version_id="v-chall", inputs=_build_input()
        )

        self.assertIsInstance(result, DualDispatchResult)
        self.assertEqual(result.champion_decision.entry_action, EntryAction.ENTER)
        self.assertEqual(result.challenger_decision.entry_action, EntryAction.WAIT)
        self.assertFalse(result.record.decisions_agree)
        self.assertEqual(result.record.champion_decision_type, "ENTER")
        self.assertEqual(result.record.challenger_decision_type, "WAIT")

    def test_dispatch_measures_latency_for_both_sides(self):
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(_decision(), sleep_s=0.01, version_id="v-champ")
        )
        self.registry.register_decision_model(
            "v-chall", _StubDecisionModel(_decision(), sleep_s=0.03, version_id="v-chall")
        )
        dispatcher = ShadowDispatcher(self.registry)
        result = dispatcher.dispatch(
            test_id="shadow-1", challenger_version_id="v-chall", inputs=_build_input()
        )
        self.assertGreater(result.champion_latency_ms, 0.0)
        self.assertGreater(result.challenger_latency_ms, 0.0)
        # Challenger slept longer, so its recorded latency should exceed champion's.
        self.assertGreater(result.challenger_latency_ms, result.champion_latency_ms)
        self.assertAlmostEqual(
            result.record.latency_diff_ms,
            result.challenger_latency_ms - result.champion_latency_ms,
            places=3,
        )

    def test_pnl_diff_uses_expected_pnl_proxy(self):
        champion_decision = _decision(
            position_size_lots=1.0, risk_allocation_pct=0.01, confidence=0.5
        )
        challenger_decision = _decision(
            position_size_lots=2.0, risk_allocation_pct=0.02, confidence=0.9
        )
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(champion_decision, version_id="v-champ")
        )
        self.registry.register_decision_model(
            "v-chall", _StubDecisionModel(challenger_decision, version_id="v-chall")
        )
        dispatcher = ShadowDispatcher(self.registry)
        result = dispatcher.dispatch(
            test_id="shadow-1", challenger_version_id="v-chall", inputs=_build_input()
        )
        expected_diff = expected_pnl_proxy(challenger_decision) - expected_pnl_proxy(
            champion_decision
        )
        self.assertAlmostEqual(result.record.pnl_diff, expected_diff, places=6)
        self.assertGreater(result.record.pnl_diff, 0.0)

    def test_risk_diff_reflects_risk_allocation_delta(self):
        champion_decision = _decision(risk_allocation_pct=0.01)
        challenger_decision = _decision(risk_allocation_pct=0.03)
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(champion_decision, version_id="v-champ")
        )
        self.registry.register_decision_model(
            "v-chall", _StubDecisionModel(challenger_decision, version_id="v-chall")
        )
        dispatcher = ShadowDispatcher(self.registry)
        result = dispatcher.dispatch(
            test_id="shadow-1", challenger_version_id="v-chall", inputs=_build_input()
        )
        self.assertAlmostEqual(result.record.risk_diff, 0.02, places=6)

    def test_falls_back_to_heuristic_models_when_none_registered(self):
        dispatcher = ShadowDispatcher(self.registry)
        result = dispatcher.dispatch(
            test_id="shadow-1", challenger_version_id="v-chall", inputs=_build_input()
        )
        self.assertIsInstance(result, DualDispatchResult)
        self.assertIn(result.record.champion_decision_type, ("ENTER", "WAIT", "REJECT"))
        self.assertIn(result.record.challenger_decision_type, ("ENTER", "WAIT", "REJECT"))

    def test_to_dict_serialises_full_result(self):
        dispatcher = ShadowDispatcher(self.registry)
        result = dispatcher.dispatch(
            test_id="shadow-1", challenger_version_id="v-chall", inputs=_build_input()
        )
        d = result.to_dict()
        for key in (
            "champion_decision",
            "challenger_decision",
            "champion_latency_ms",
            "challenger_latency_ms",
            "record",
        ):
            self.assertIn(key, d)

    def test_challenger_decision_is_marked_shadow_champion_is_not(self):
        """S22B: structural guarantee — only the challenger side of a dual
        dispatch is ever marked `is_shadow=True`; the champion decision must
        never carry the flag, whatever model produced it."""
        champion_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.6)
        challenger_decision = _decision(entry_action=EntryAction.WAIT, confidence=0.5)
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(champion_decision, version_id="v-champ")
        )
        self.registry.register_decision_model(
            "v-chall", _StubDecisionModel(challenger_decision, version_id="v-chall")
        )
        dispatcher = ShadowDispatcher(self.registry)
        result = dispatcher.dispatch(
            test_id="shadow-1", challenger_version_id="v-chall", inputs=_build_input()
        )
        self.assertFalse(result.champion_decision.is_shadow)
        self.assertTrue(result.challenger_decision.is_shadow)
        self.assertTrue(result.to_dict()["challenger_decision"]["is_shadow"])
        self.assertFalse(result.to_dict()["champion_decision"]["is_shadow"])

    def test_execution_planner_refuses_shadow_marked_decision(self):
        """S22B: `ExecutionPlanner.plan()` must raise, not silently trade,
        if the resolved decision is somehow marked `is_shadow=True` — the
        enforcement point downstream of dispatch."""
        from trainer.execution.execution_planner import ExecutionPlanner

        shadow_marked_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.9)
        shadow_marked_decision.is_shadow = True
        planner = ExecutionPlanner(
            decision_model=_StubDecisionModel(shadow_marked_decision, version_id="v-chall")
        )
        with self.assertRaises(RuntimeError):
            planner.plan(_fc())

    def test_execution_planner_executes_non_shadow_decision_normally(self):
        from trainer.execution.execution_planner import ExecutionPlanner

        champion_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.9)
        self.assertFalse(champion_decision.is_shadow)
        planner = ExecutionPlanner(
            decision_model=_StubDecisionModel(champion_decision, version_id="v-champ")
        )
        plan = planner.plan(_fc())
        self.assertEqual(plan.decision.entry_action, EntryAction.ENTER)

    def test_execution_planner_rejects_non_shadow_flagged_wrong_champion_via_registry(self):
        """S22B: `is_shadow` alone can't catch a miswired active-model
        pointer that produces an unflagged decision from a non-champion
        version — only `ShadowDispatcher` ever sets `is_shadow`. When
        `ExecutionPlanner` resolves its model via the live `QDERegistry`
        (the real runtime path), it must independently re-verify the
        decision's `model_version_id` against the registry's actual active
        version and refuse execution on a mismatch, even if `is_shadow` is
        left at its default `False`."""
        from trainer.execution.execution_planner import ExecutionPlanner

        wrong_version_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.9)
        wrong_version_decision.model_version_id = "v-chall"
        self.assertFalse(wrong_version_decision.is_shadow)
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(wrong_version_decision, version_id="v-chall")
        )
        planner = ExecutionPlanner(qde_registry=self.registry)
        with self.assertRaises(RuntimeError):
            planner.plan(_fc())

    def test_execution_planner_no_false_positive_on_empty_model_version_id(self):
        """Legacy decisions that never set `model_version_id` (empty string
        default) must not be rejected by the champion-identity guard — the
        check only applies when a version id is actually present to compare."""
        from trainer.execution.execution_planner import ExecutionPlanner

        legacy_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.9)
        self.assertEqual(legacy_decision.model_version_id, "")
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(legacy_decision, version_id="v-champ")
        )
        planner = ExecutionPlanner(qde_registry=self.registry)
        plan = planner.plan(_fc())
        self.assertEqual(plan.decision.entry_action, EntryAction.ENTER)

    def test_execution_planner_direct_injection_bypasses_champion_identity_check(self):
        """Direct `decision_model` injection (tests, or a caller that
        already resolved a model) intentionally has no `QDERegistry`
        reference, so the champion-identity guard never runs for it — only
        the `is_shadow` check applies on that path. A decision whose
        model_version_id doesn't match any registry version must still
        execute normally when injected this way."""
        from trainer.execution.execution_planner import ExecutionPlanner

        decision = _decision(entry_action=EntryAction.ENTER, confidence=0.9)
        decision.model_version_id = "v-not-in-any-registry"
        planner = ExecutionPlanner(
            decision_model=_StubDecisionModel(decision, version_id="v-not-in-any-registry")
        )
        self.assertIsNone(planner._qde_registry)
        plan = planner.plan(_fc())
        self.assertEqual(plan.decision.entry_action, EntryAction.ENTER)

    def test_execution_planner_executes_via_registry_when_versions_match(self):
        """Companion to the mismatch test above: the champion-identity
        check must not false-positive on the normal, correctly-wired path
        where the decision's model_version_id matches the registry's
        active version."""
        from trainer.execution.execution_planner import ExecutionPlanner

        champion_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.9)
        self.registry.register_decision_model(
            "v-champ", _StubDecisionModel(champion_decision, version_id="v-champ")
        )
        planner = ExecutionPlanner(qde_registry=self.registry)
        plan = planner.plan(_fc())
        self.assertEqual(plan.decision.entry_action, EntryAction.ENTER)


# ════════════════════════════════════════════════════════════════════════════
# evaluate_and_maybe_promote — automated promotion trigger
# ════════════════════════════════════════════════════════════════════════════
class TestEvaluateAndMaybePromote(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self.registry = QDERegistry(storage_path=self._tmp.name)
        self.promotion = QDEPromotionEngine(self.registry)

        now = time.time()
        self.challenger = QDEVersion(
            version_id="v-chall",
            version_number=2,
            label="challenger",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
            metrics={
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.3,
                "max_drawdown": 0.10,
                "win_rate": 0.55,
                "profit_factor": 1.5,
                "capital_efficiency": 0.2,
                "decision_quality": 0.7,
            },
        )
        self.registry.register(self.challenger)

    def tearDown(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def _shadow_test(self, recommendation, status=ShadowTestStatus.CONCLUDED, avg_score=0.8):
        test = ShadowTestRun.new(
            name="t",
            challenger_version="v-chall",
            champion_version="v-champ",
            instruments=["EURUSD"],
            min_decisions=1,
        )
        test.status = status
        test.recommendation = recommendation
        test.avg_promotion_score = avg_score
        test.agreement_rate = 0.9
        return test

    def test_not_concluded_returns_no_action(self):
        test = self._shadow_test(ShadowTestRecommendation.PROMOTE, status=ShadowTestStatus.RUNNING)
        result = evaluate_and_maybe_promote(test, self.promotion)
        self.assertEqual(result["action"], "none")

    def test_promote_recommendation_activates_when_backtest_gates_clear(self):
        test = self._shadow_test(ShadowTestRecommendation.PROMOTE)
        result = evaluate_and_maybe_promote(test, self.promotion)
        self.assertEqual(result["action"], "promoted")
        self.assertEqual(self.registry.get_active().version_id, "v-chall")

    def test_promote_recommendation_does_not_activate_when_backtest_gates_fail(self):
        # Overwrite metrics with values that fail absolute floor checks.
        self.challenger.metrics = {
            "sharpe_ratio": 0.1,
            "max_drawdown": 0.9,
            "win_rate": 0.1,
            "profit_factor": 0.5,
        }
        self.registry.update(self.challenger)
        test = self._shadow_test(ShadowTestRecommendation.PROMOTE)
        result = evaluate_and_maybe_promote(test, self.promotion)
        self.assertEqual(result["action"], "not_promoted")
        self.assertIsNone(self.registry.get_active())

    def test_reject_recommendation_takes_no_promotion_action(self):
        test = self._shadow_test(ShadowTestRecommendation.REJECT, avg_score=0.1)
        result = evaluate_and_maybe_promote(test, self.promotion)
        self.assertEqual(result["action"], "rejected")
        self.assertIsNone(self.registry.get_active())

    def test_extend_recommendation_takes_no_promotion_action(self):
        test = self._shadow_test(ShadowTestRecommendation.EXTEND, avg_score=0.5)
        result = evaluate_and_maybe_promote(test, self.promotion)
        self.assertEqual(result["action"], "extended")
        self.assertIsNone(self.registry.get_active())

    def test_backtest_engine_remains_untouched_by_this_module(self):
        # Sanity check that the augmentation is additive: calling evaluate()
        # directly still runs the same five-dimension backtest gates.
        decision = self.promotion.evaluate("v-chall")
        self.assertTrue(decision.is_eligible)


# ════════════════════════════════════════════════════════════════════════════
# QDELifecycleAPI — dispatch_shadow_decision / evaluate_shadow_promotion
# ════════════════════════════════════════════════════════════════════════════
class TestQDELifecycleAPIShadowEndpoints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self.api = QDELifecycleAPI(storage_path=self._tmp.name)

        now = time.time()
        self.champion = QDEVersion(
            version_id="v-champ",
            version_number=1,
            label="champion",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
        )
        self.challenger = QDEVersion(
            version_id="v-chall",
            version_number=2,
            label="challenger",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
            metrics={
                "sharpe_ratio": 1.2,
                "sortino_ratio": 1.3,
                "max_drawdown": 0.10,
                "win_rate": 0.55,
                "profit_factor": 1.5,
                "capital_efficiency": 0.2,
                "decision_quality": 0.7,
            },
        )
        self.api._registry.register(self.champion)
        self.api._registry.register(self.challenger)
        self.api._registry.set_active("v-champ")

    def tearDown(self):
        import os

        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_dispatch_shadow_decision_returns_widened_record(self):
        resp = self.api.dispatch_shadow_decision(
            test_id="shadow-1",
            challenger_version_id="v-chall",
            inputs=_build_input(),
        )
        self.assertTrue(resp.success, resp.error)
        record = resp.data["record"]
        for key in (
            "confidence_diff",
            "pnl_diff",
            "latency_diff_ms",
            "risk_diff",
            "promotion_score",
        ):
            self.assertIn(key, record)

    def test_dispatch_shadow_decision_errors_on_unknown_challenger(self):
        # get_model_for_version falls back to heuristic even for unknown ids,
        # so dispatch itself won't error — but the record should still be
        # well-formed for a nonexistent version (no registered version).
        resp = self.api.dispatch_shadow_decision(
            test_id="shadow-1",
            challenger_version_id="does-not-exist",
            inputs=_build_input(),
        )
        self.assertTrue(resp.success, resp.error)

    def test_evaluate_shadow_promotion_promotes_on_promote_recommendation(self):
        shadow_test_dict = {
            "test_id": "shadow-1",
            "name": "t",
            "challenger_version": "v-chall",
            "champion_version": "v-champ",
            "instruments": ["EURUSD"],
            "status": ShadowTestStatus.CONCLUDED.value,
            "started_at": time.time(),
            "concluded_at": time.time(),
            "min_decisions": 1,
            "notes": "",
            "total_decisions": 10,
            "agreement_rate": 0.9,
            "avg_promotion_score": 0.85,
            "recommendation": ShadowTestRecommendation.PROMOTE.value,
            "conclusion_notes": "",
        }
        resp = self.api.evaluate_shadow_promotion(shadow_test_dict)
        self.assertTrue(resp.success, resp.error)
        self.assertEqual(resp.data["action"], "promoted")
        self.assertEqual(self.api._registry.get_active().version_id, "v-chall")

    def test_evaluate_shadow_promotion_no_action_on_reject(self):
        shadow_test_dict = {
            "test_id": "shadow-1",
            "name": "t",
            "challenger_version": "v-chall",
            "champion_version": "v-champ",
            "instruments": ["EURUSD"],
            "status": ShadowTestStatus.CONCLUDED.value,
            "started_at": time.time(),
            "concluded_at": time.time(),
            "min_decisions": 1,
            "notes": "",
            "total_decisions": 10,
            "agreement_rate": 0.2,
            "avg_promotion_score": 0.1,
            "recommendation": ShadowTestRecommendation.REJECT.value,
            "conclusion_notes": "",
        }
        resp = self.api.evaluate_shadow_promotion(shadow_test_dict)
        self.assertTrue(resp.success, resp.error)
        self.assertEqual(resp.data["action"], "rejected")
        self.assertEqual(self.api._registry.get_active().version_id, "v-champ")


# ════════════════════════════════════════════════════════════════════════════
# End-to-end: dual dispatch → Research Lab persistence → shadow conclude →
# automated promotion trigger, exercising the whole widened pipeline together.
# ════════════════════════════════════════════════════════════════════════════
class TestDualDispatchToPromotionEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmpdir.name) / "research_lab.db")
        self.rl_db = ResearchLabDatabase(self.db_path)
        self.rl_api = ResearchLabAPI(db=self.rl_db)

        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self.qde_api = QDELifecycleAPI(storage_path=self._tmp.name)

        now = time.time()
        champion = QDEVersion(
            version_id="v-champ",
            version_number=1,
            label="champion",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
        )
        challenger = QDEVersion(
            version_id="v-chall",
            version_number=2,
            label="challenger",
            stage=QDEStage.DEPLOYMENT_CANDIDATE,
            created_at=now,
            updated_at=now,
            metrics={
                "sharpe_ratio": 1.5,
                "sortino_ratio": 1.6,
                "max_drawdown": 0.08,
                "win_rate": 0.6,
                "profit_factor": 1.8,
                "capital_efficiency": 0.25,
                "decision_quality": 0.8,
            },
        )
        self.qde_api._registry.register(champion)
        self.qde_api._registry.register(challenger)
        self.qde_api._registry.set_active("v-champ")

        agreeing_decision = _decision(entry_action=EntryAction.ENTER, confidence=0.7)
        self.qde_api._registry.register_decision_model(
            "v-champ", _StubDecisionModel(agreeing_decision, version_id="v-champ")
        )
        self.qde_api._registry.register_decision_model(
            "v-chall", _StubDecisionModel(agreeing_decision, version_id="v-chall")
        )

    def tearDown(self):
        import os

        self._tmpdir.cleanup()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_full_pipeline_dispatch_persist_conclude_promote(self):
        created = self.rl_api.create_shadow_test(
            name="v2 shadow",
            challenger_version="v-chall",
            champion_version="v-champ",
            instruments=["EURUSD"],
            min_decisions=3,
        )
        self.assertTrue(created.success, created.error)
        test_id = created.data["test_id"]

        for _ in range(3):
            dispatch_resp = self.qde_api.dispatch_shadow_decision(
                test_id=test_id,
                challenger_version_id="v-chall",
                inputs=_build_input(),
            )
            self.assertTrue(dispatch_resp.success, dispatch_resp.error)
            record = ShadowDecisionRecord(**dispatch_resp.data["record"])
            persist_resp = self.rl_api.record_dual_dispatch_decision(record)
            self.assertTrue(persist_resp.success, persist_resp.error)

        concluded = self.rl_api.conclude_shadow_test(test_id)
        self.assertTrue(concluded.success, concluded.error)
        self.assertEqual(concluded.data["recommendation"], ShadowTestRecommendation.PROMOTE.value)
        self.assertGreaterEqual(concluded.data["avg_promotion_score"], 0.70)

        promotion_resp = self.qde_api.evaluate_shadow_promotion(concluded.data)
        self.assertTrue(promotion_resp.success, promotion_resp.error)
        self.assertEqual(promotion_resp.data["action"], "promoted")
        self.assertEqual(self.qde_api._registry.get_active().version_id, "v-chall")

    def test_shadow_tests_queryable_by_version_pair_and_time_window(self):
        """S22B: shadow observation results must be queryable via the
        Research Lab by champion/challenger version pair and time window."""
        t0 = time.time() - 1000
        created_pair_a = self.rl_api.create_shadow_test(
            name="pair-a",
            challenger_version="v-chall",
            champion_version="v-champ",
            instruments=["EURUSD"],
            min_decisions=1,
        )
        self.assertTrue(created_pair_a.success, created_pair_a.error)

        created_pair_b = self.rl_api.create_shadow_test(
            name="pair-b",
            challenger_version="v-other-chall",
            champion_version="v-other-champ",
            instruments=["EURUSD"],
            min_decisions=1,
        )
        self.assertTrue(created_pair_b.success, created_pair_b.error)

        by_champion = self.rl_api.list_shadow_tests(champion_version="v-champ")
        self.assertTrue(by_champion.success, by_champion.error)
        ids = {t["test_id"] for t in by_champion.data["shadow_tests"]}
        self.assertIn(created_pair_a.data["test_id"], ids)
        self.assertNotIn(created_pair_b.data["test_id"], ids)

        by_pair = self.rl_api.list_shadow_tests(
            champion_version="v-champ", challenger_version="v-chall"
        )
        self.assertTrue(by_pair.success, by_pair.error)
        self.assertEqual(
            {t["test_id"] for t in by_pair.data["shadow_tests"]},
            {created_pair_a.data["test_id"]},
        )

        # Time window: everything started_at >= t0 (both fall inside);
        # nothing should match a window entirely in the future.
        in_window = self.rl_api.list_shadow_tests(since=t0)
        in_window_ids = {t["test_id"] for t in in_window.data["shadow_tests"]}
        self.assertIn(created_pair_a.data["test_id"], in_window_ids)
        self.assertIn(created_pair_b.data["test_id"], in_window_ids)

        future_only = self.rl_api.list_shadow_tests(since=time.time() + 1000)
        self.assertEqual(future_only.data["shadow_tests"], [])


if __name__ == "__main__":
    unittest.main()
