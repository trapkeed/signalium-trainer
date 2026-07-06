"""Phase 21.6 — Unit tests for the Decision Memory & Self-Evaluation Engine.

Run with:  python3 -m unittest tests/unit/test_memory.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
import unittest
from pathlib import Path

# ── Spec-from-file loader ─────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent
MEMORY_DIR = ROOT / "trainer" / "memory"


def _load(name: str, rel: str):
    """Load a trainer.memory sub-module without importing torch."""
    path = ROOT / "trainer" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load in dependency order
dt = _load("trainer.memory.decision_types", "memory/decision_types.py")
dm = _load("trainer.memory.decision_models", "memory/decision_models.py")
mem = _load("trainer.memory.decision_memory", "memory/decision_memory.py")
ot = _load("trainer.memory.outcome_tracker", "memory/outcome_tracker.py")
qe = _load("trainer.memory.quality_engine", "memory/quality_engine.py")
ce = _load("trainer.memory.counterfactual_engine", "memory/counterfactual_engine.py")
pd = _load("trainer.memory.pattern_discovery", "memory/pattern_discovery.py")
se = _load("trainer.memory.self_evaluation", "memory/self_evaluation.py")
kb = _load("trainer.memory.knowledge_base", "memory/knowledge_base.py")
dl = _load("trainer.memory.decision_lineage", "memory/decision_lineage.py")
db_ = _load("trainer.memory.decision_benchmarking", "memory/decision_benchmarking.py")
ms = _load("trainer.memory.memory_search", "memory/memory_search.py")
mdb = _load("trainer.memory.memory_database", "memory/memory_database.py")
mdash = _load("trainer.memory.memory_dashboard", "memory/memory_dashboard.py")
mapi_mod = _load("trainer.memory.memory_api", "memory/memory_api.py")

# Aliases
DecisionType = dt.DecisionType
OutcomeType = dt.OutcomeType
QualityFactor = dt.QualityFactor
MarketRegime = dt.MarketRegime
VolatilityRegime = dt.VolatilityRegime
DecisionGrade = dt.DecisionGrade
QUALITY_WEIGHTS = dt.QUALITY_WEIGHTS
POSITIVE_OUTCOMES = dt.POSITIVE_OUTCOMES
NEGATIVE_OUTCOMES = dt.NEGATIVE_OUTCOMES
grade_from_score = dt.grade_from_score

DecisionSnapshot = dm.DecisionSnapshot
DecisionReasoning = dm.DecisionReasoning
DecisionOutcome = dm.DecisionOutcome
DecisionScore = dm.DecisionScore
CounterfactualResult = dm.CounterfactualResult
DecisionRecord = dm.DecisionRecord

DecisionMemory = mem.DecisionMemory
OutcomeTracker = ot.OutcomeTracker
QualityEngine = qe.QualityEngine
CounterfactualEngine = ce.CounterfactualEngine
PatternDiscovery = pd.PatternDiscovery
SelfEvaluationEngine = se.SelfEvaluationEngine
KnowledgeBase = kb.KnowledgeBase
DecisionLineage = dl.DecisionLineage
DecisionBenchmarking = db_.DecisionBenchmarking
MemorySearch = ms.MemorySearch
SearchQuery = ms.SearchQuery
MemoryDatabase = mdb.MemoryDatabase
MemoryDashboard = mdash.MemoryDashboard
MemoryAPI = mapi_mod.MemoryAPI


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_snap(
    instrument="EURUSD",
    regime=MarketRegime.TRENDING_UP.value,
    session="LONDON",
    confidence=0.70,
    tp_prob=0.65,
    sl_prob=0.30,
    price=1.10000,
    atr=0.00100,
    spread=0.00002,
    volatility=0.01,
    expected_move=0.00150,
    expected_holding_h=24.0,
    portfolio_exposure=0.05,
    correlation_risk=0.10,
    model_version="v1.0",
    volatility_regime=VolatilityRegime.NORMAL.value,
) -> DecisionSnapshot:
    return DecisionSnapshot(
        price=price,
        atr=atr,
        spread=spread,
        volatility=volatility,
        session=session,
        liquidity_estimate=0.90,
        market_regime=regime,
        volatility_regime=volatility_regime,
        timeframe="H1",
        forecast_direction="BUY",
        forecast_confidence=confidence,
        expected_move=expected_move,
        expected_holding_h=expected_holding_h,
        tp_probability=tp_prob,
        sl_probability=sl_prob,
        model_version=model_version,
        champion_model="ensemble_v3",
        research_score=0.72,
        ensemble_output=0.68,
        portfolio_exposure=portfolio_exposure,
        correlation_risk=correlation_risk,
        risk_metrics={"var": 0.002, "max_drawdown": 0.05},
        feature_snapshot={"rsi": 58.0, "macd": 0.001},
        expert_weights={"trend": 0.5, "momentum": 0.3, "value": 0.2},
    )


def make_rea(
    primary=None,
    supporting=None,
    contradicting=None,
    risks=None,
    alternatives=None,
    rationale="Strong trend with momentum alignment.",
) -> DecisionReasoning:
    return DecisionReasoning(
        primary_reasons=primary or ["Uptrend confirmed", "RSI bullish"],
        supporting_factors=supporting or ["MACD positive", "Session peak"],
        contradicting_factors=contradicting or ["High spread"],
        risk_factors=risks or ["News risk", "Low liquidity"],
        uncertainty_sources=["Forecast uncertainty ±0.002"],
        alternatives_considered=alternatives or ["WAIT", "NO_TRADE", "SCALE_IN"],
        final_rationale=rationale,
    )


def make_record(
    decision_type=None,
    instrument="EURUSD",
    regime=MarketRegime.TRENDING_UP.value,
    session="LONDON",
    confidence=0.70,
    tp_prob=0.65,
    sl_prob=0.30,
    tags=None,
) -> DecisionRecord:
    return DecisionRecord.create(
        decision_type=decision_type or DecisionType.BUY,
        instrument=instrument,
        snapshot=make_snap(
            instrument=instrument,
            regime=regime,
            session=session,
            confidence=confidence,
            tp_prob=tp_prob,
            sl_prob=sl_prob,
        ),
        reasoning=make_rea(),
        tags=tags or ["test"],
    )


def add_positive_outcome(record: DecisionRecord, pnl=0.0015, holding_h=20.0) -> None:
    record.outcome = DecisionOutcome(
        outcome_type=OutcomeType.HIT_TP,
        actual_direction_correct=True,
        actual_pnl=pnl,
        actual_pnl_pct=pnl / 1.10,
        hit_tp=True,
        hit_sl=False,
        holding_time_h=holding_h,
        recorded_at=time.time(),
    )


def add_negative_outcome(record: DecisionRecord, pnl=-0.0010, holding_h=5.0) -> None:
    record.outcome = DecisionOutcome(
        outcome_type=OutcomeType.HIT_SL,
        actual_direction_correct=False,
        actual_pnl=pnl,
        actual_pnl_pct=pnl / 1.10,
        hit_tp=False,
        hit_sl=True,
        holding_time_h=holding_h,
        recorded_at=time.time(),
    )


# ── Test suite ────────────────────────────────────────────────────────────────


class TestDecisionTypes(unittest.TestCase):
    """Tests for decision_types.py"""

    def test_decision_type_values(self):
        self.assertEqual(DecisionType.BUY.value, "BUY")
        self.assertEqual(DecisionType.SELL.value, "SELL")
        self.assertEqual(DecisionType.NO_TRADE.value, "NO_TRADE")

    def test_all_decision_types_present(self):
        types = {t.value for t in DecisionType}
        for v in (
            "BUY",
            "SELL",
            "NO_TRADE",
            "WAIT",
            "CANCEL",
            "SCALE_IN",
            "SCALE_OUT",
            "MOVE_STOP",
            "MOVE_TARGET",
            "PARTIAL_CLOSE",
            "EXIT",
        ):
            self.assertIn(v, types)

    def test_outcome_types(self):
        self.assertEqual(OutcomeType.HIT_TP.value, "HIT_TP")
        self.assertEqual(OutcomeType.HIT_SL.value, "HIT_SL")
        self.assertEqual(OutcomeType.MISSED_OPPORTUNITY.value, "MISSED_OPPORTUNITY")

    def test_all_outcome_types_present(self):
        types = {t.value for t in OutcomeType}
        for v in (
            "CORRECT_DIRECTION",
            "INCORRECT_DIRECTION",
            "HIT_TP",
            "HIT_SL",
            "CANCELLED",
            "EXPIRED",
            "MISSED_OPPORTUNITY",
            "PARTIAL_SUCCESS",
            "NO_TRADE",
            "UNEXPECTED_MARKET_EVENT",
        ):
            self.assertIn(v, types)

    def test_quality_factors(self):
        self.assertEqual(len(QUALITY_WEIGHTS), 10)

    def test_quality_weights_sum(self):
        self.assertAlmostEqual(sum(QUALITY_WEIGHTS.values()), 1.0, places=9)

    def test_individual_weights(self):
        self.assertAlmostEqual(QUALITY_WEIGHTS[QualityFactor.FORECAST_ACCURACY], 0.20)
        self.assertAlmostEqual(QUALITY_WEIGHTS[QualityFactor.EXECUTION_QUALITY], 0.15)
        self.assertAlmostEqual(QUALITY_WEIGHTS[QualityFactor.RISK_MANAGEMENT], 0.15)

    def test_grade_from_score_excellent(self):
        self.assertEqual(grade_from_score(0.90), DecisionGrade.EXCELLENT)

    def test_grade_from_score_good(self):
        self.assertEqual(grade_from_score(0.75), DecisionGrade.GOOD)

    def test_grade_from_score_acceptable(self):
        self.assertEqual(grade_from_score(0.60), DecisionGrade.ACCEPTABLE)

    def test_grade_from_score_poor(self):
        self.assertEqual(grade_from_score(0.45), DecisionGrade.POOR)

    def test_grade_from_score_reject(self):
        self.assertEqual(grade_from_score(0.20), DecisionGrade.REJECT)

    def test_positive_outcomes_set(self):
        self.assertIn(OutcomeType.HIT_TP, POSITIVE_OUTCOMES)
        self.assertIn(OutcomeType.CORRECT_DIRECTION, POSITIVE_OUTCOMES)

    def test_negative_outcomes_set(self):
        self.assertIn(OutcomeType.HIT_SL, NEGATIVE_OUTCOMES)
        self.assertIn(OutcomeType.INCORRECT_DIRECTION, NEGATIVE_OUTCOMES)


class TestDecisionModels(unittest.TestCase):
    """Tests for decision_models.py"""

    def test_snapshot_to_dict(self):
        snap = make_snap()
        d = snap.to_dict()
        self.assertEqual(d["session"], "LONDON")
        self.assertIn("risk_metrics", d)
        self.assertIn("feature_snapshot", d)

    def test_reasoning_to_dict(self):
        rea = make_rea()
        d = rea.to_dict()
        self.assertIn("primary_reasons", d)
        self.assertIn("final_rationale", d)
        self.assertIsInstance(d["primary_reasons"], list)

    def test_outcome_to_dict(self):
        out = DecisionOutcome(
            outcome_type=OutcomeType.HIT_TP,
            actual_pnl=0.0015,
            hit_tp=True,
            recorded_at=time.time(),
        )
        d = out.to_dict()
        self.assertEqual(d["outcome_type"], "HIT_TP")
        self.assertTrue(d["hit_tp"])

    def test_decision_record_create(self):
        r = make_record()
        self.assertIsNotNone(r.decision_id)
        self.assertEqual(len(r.decision_id), 36)  # UUID length
        self.assertFalse(r.has_outcome)
        self.assertFalse(r.has_score)
        self.assertTrue(r.is_entry_decision)

    def test_decision_record_not_entry(self):
        r = make_record(decision_type=DecisionType.WAIT)
        self.assertFalse(r.is_entry_decision)

    def test_decision_record_to_dict(self):
        r = make_record()
        d = r.to_dict()
        self.assertIn("decision_id", d)
        self.assertIn("snapshot", d)
        self.assertIn("reasoning", d)
        self.assertIsNone(d["outcome"])

    def test_decision_record_summary(self):
        r = make_record()
        s = r.summary()
        self.assertIn("decision_id", s)
        self.assertIn("confidence", s)
        self.assertEqual(s["instrument"], "EURUSD")

    def test_decision_score_to_dict(self):
        score = DecisionScore(
            overall=0.75,
            grade=DecisionGrade.GOOD,
            sub_scores={"FORECAST_ACCURACY": 0.80},
            notes=[],
        )
        d = score.to_dict()
        self.assertEqual(d["grade"], "GOOD")
        self.assertAlmostEqual(d["overall"], 0.75)

    def test_counterfactual_result_to_dict(self):
        cf = CounterfactualResult(
            scenario="WAIT instead of BUY",
            alternative_action="WAIT",
            estimated_pnl=0.0,
            delta_pnl=-0.0015,
            better_than_actual=False,
        )
        d = cf.to_dict()
        self.assertIn("scenario", d)
        self.assertFalse(d["better_than_actual"])


class TestDecisionMemory(unittest.TestCase):
    """Tests for decision_memory.py"""

    def setUp(self):
        self.mem = DecisionMemory()

    def test_record_and_retrieve(self):
        r = make_record()
        self.mem.store(r)
        fetched = self.mem.get(r.decision_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.instrument, "EURUSD")

    def test_record_decision_method(self):
        r = self.mem.record_decision(
            decision_type=DecisionType.SELL,
            instrument="GBPUSD",
            snapshot=make_snap(instrument="GBPUSD"),
            reasoning=make_rea(),
        )
        self.assertEqual(r.decision_type, DecisionType.SELL)
        self.assertEqual(self.mem.total(), 1)

    def test_total(self):
        for _ in range(5):
            self.mem.store(make_record())
        self.assertEqual(self.mem.total(), 5)

    def test_record_outcome(self):
        r = make_record()
        self.mem.store(r)
        ok = self.mem.record_outcome(
            decision_id=r.decision_id,
            outcome_type=OutcomeType.HIT_TP,
            actual_pnl=0.0015,
            hit_tp=True,
        )
        self.assertTrue(ok)
        fetched = self.mem.get(r.decision_id)
        self.assertTrue(fetched.has_outcome)
        self.assertTrue(fetched.outcome.hit_tp)

    def test_record_outcome_unknown_id(self):
        ok = self.mem.record_outcome("unknown-id", OutcomeType.CANCELLED)
        self.assertFalse(ok)

    def test_filter_by_instrument(self):
        self.mem.store(make_record(instrument="EURUSD"))
        self.mem.store(make_record(instrument="GBPUSD"))
        self.mem.store(make_record(instrument="EURUSD"))
        eur = self.mem.filter(instrument="EURUSD")
        self.assertEqual(len(eur), 2)

    def test_filter_by_type(self):
        self.mem.store(make_record(decision_type=DecisionType.BUY))
        self.mem.store(make_record(decision_type=DecisionType.SELL))
        buys = self.mem.filter(decision_type=DecisionType.BUY)
        self.assertEqual(len(buys), 1)

    def test_filter_by_confidence(self):
        self.mem.store(make_record(confidence=0.80))
        self.mem.store(make_record(confidence=0.50))
        hi = self.mem.filter(min_confidence=0.75)
        self.assertEqual(len(hi), 1)

    def test_filter_has_outcome(self):
        r1 = make_record()
        add_positive_outcome(r1)
        r2 = make_record()
        self.mem.store(r1)
        self.mem.store(r2)
        with_out = self.mem.filter(has_outcome=True)
        self.assertEqual(len(with_out), 1)

    def test_decisions_with_outcomes(self):
        r1 = make_record()
        add_positive_outcome(r1)
        r2 = make_record()
        self.mem.store(r1)
        self.mem.store(r2)
        self.assertEqual(len(self.mem.decisions_with_outcomes()), 1)
        self.assertEqual(len(self.mem.decisions_without_outcomes()), 1)

    def test_stats(self):
        for _ in range(3):
            self.mem.store(make_record())
        s = self.mem.stats()
        self.assertEqual(s["total_decisions"], 3)
        self.assertIn("decision_type_dist", s)

    def test_all_records(self):
        self.mem.store(make_record())
        self.mem.store(make_record())
        self.assertEqual(len(self.mem.all_records()), 2)


class TestOutcomeTracker(unittest.TestCase):
    """Tests for outcome_tracker.py"""

    def setUp(self):
        self.tracker = OutcomeTracker()

    def _records_with_outcomes(self, n_pos=3, n_neg=2):
        records = []
        for _ in range(n_pos):
            r = make_record()
            add_positive_outcome(r)
            records.append(r)
        for _ in range(n_neg):
            r = make_record()
            add_negative_outcome(r)
            records.append(r)
        return records

    def test_summarise_basic(self):
        records = self._records_with_outcomes(3, 2)
        s = self.tracker.summarise(records)
        self.assertEqual(s.total, 5)
        self.assertEqual(s.with_outcome, 5)
        self.assertEqual(s.positive_count, 3)
        self.assertEqual(s.negative_count, 2)

    def test_win_rate(self):
        records = self._records_with_outcomes(6, 4)
        s = self.tracker.summarise(records)
        self.assertAlmostEqual(s.win_rate, 0.60, places=4)

    def test_hit_tp_count(self):
        records = self._records_with_outcomes(3, 0)
        s = self.tracker.summarise(records)
        self.assertEqual(s.hit_tp_count, 3)

    def test_hit_sl_count(self):
        records = self._records_with_outcomes(0, 3)
        s = self.tracker.summarise(records)
        self.assertEqual(s.hit_sl_count, 3)

    def test_empty_records(self):
        s = self.tracker.summarise([])
        self.assertEqual(s.total, 0)

    def test_records_without_outcomes(self):
        records = [make_record() for _ in range(3)]
        s = self.tracker.summarise(records)
        self.assertEqual(s.with_outcome, 0)

    def test_direction_accuracy(self):
        records = []
        for _ in range(4):
            r = make_record()
            add_positive_outcome(r)
            records.append(r)
        for _ in range(1):
            r = make_record()
            add_negative_outcome(r)
            records.append(r)
        acc = self.tracker.direction_accuracy(records)
        self.assertAlmostEqual(acc, 0.80, places=4)

    def test_tp_hit_rate(self):
        records = []
        for _ in range(3):
            r = make_record()
            add_positive_outcome(r)
            records.append(r)
        for _ in range(2):
            r = make_record()
            add_negative_outcome(r)
            records.append(r)
        self.assertAlmostEqual(self.tracker.tp_hit_rate(records), 0.60, places=4)

    def test_sl_hit_rate(self):
        records = []
        for _ in range(3):
            r = make_record()
            add_positive_outcome(r)
            records.append(r)
        for _ in range(2):
            r = make_record()
            add_negative_outcome(r)
            records.append(r)
        self.assertAlmostEqual(self.tracker.sl_hit_rate(records), 0.40, places=4)

    def test_by_session_grouping(self):
        r1 = make_record(session="LONDON")
        add_positive_outcome(r1)
        r2 = make_record(session="NY")
        add_negative_outcome(r2)
        s = self.tracker.summarise([r1, r2])
        self.assertIn("LONDON", s.by_session)
        self.assertIn("NY", s.by_session)

    def test_summary_to_dict(self):
        records = self._records_with_outcomes(2, 1)
        s = self.tracker.summarise(records)
        d = s.to_dict()
        self.assertIn("win_rate", d)
        self.assertIn("by_instrument", d)


class TestQualityEngine(unittest.TestCase):
    """Tests for quality_engine.py"""

    def setUp(self):
        self.engine = QualityEngine()

    def test_score_returns_decision_score(self):
        r = make_record()
        add_positive_outcome(r)
        s = self.engine.score(r)
        self.assertIsInstance(s.overall, float)
        self.assertIsInstance(s.grade, DecisionGrade)

    def test_score_bounds(self):
        r = make_record()
        add_positive_outcome(r)
        s = self.engine.score(r)
        self.assertGreaterEqual(s.overall, 0.0)
        self.assertLessEqual(s.overall, 1.0)

    def test_score_has_10_sub_scores(self):
        r = make_record()
        add_positive_outcome(r)
        s = self.engine.score(r)
        self.assertEqual(len(s.sub_scores), 10)

    def test_positive_outcome_grades_well(self):
        r = make_record(confidence=0.80, tp_prob=0.75, sl_prob=0.15)
        add_positive_outcome(r)
        s = self.engine.score(r)
        self.assertGreater(s.overall, 0.50)

    def test_negative_outcome_grades_poorly(self):
        r = make_record(confidence=0.90, tp_prob=0.30, sl_prob=0.50)
        add_negative_outcome(r)
        s = self.engine.score(r)
        self.assertLess(s.overall, 0.70)

    def test_score_and_attach(self):
        r = make_record()
        add_positive_outcome(r)
        s = self.engine.score_and_attach(r)
        self.assertIsNotNone(r.score)
        self.assertEqual(r.score, s)

    def test_batch_score(self):
        records = [make_record() for _ in range(5)]
        for r in records:
            add_positive_outcome(r)
        scores = self.engine.batch_score(records)
        self.assertEqual(len(scores), 5)

    def test_quality_weights_dict(self):
        w = self.engine.quality_weights()
        self.assertEqual(len(w), 10)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)

    def test_no_outcome_score(self):
        r = make_record()
        s = self.engine.score(r)  # no outcome — uses probabilities
        self.assertGreaterEqual(s.overall, 0.0)
        self.assertLessEqual(s.overall, 1.0)

    def test_grade_excellent(self):
        r = make_record(confidence=0.85, tp_prob=0.80, sl_prob=0.10)
        r.snapshot.liquidity_estimate = 1.0
        r.snapshot.portfolio_exposure = 0.01
        r.snapshot.correlation_risk = 0.01
        r.snapshot.expected_move = 0.003
        add_positive_outcome(r)
        r.outcome.hit_tp = True
        r.outcome.actual_direction_correct = True
        s = self.engine.score(r)
        self.assertIn(s.grade.value, ("EXCELLENT", "GOOD"))

    def test_sl_hit_penalises_risk(self):
        r = make_record(sl_prob=0.60)
        add_negative_outcome(r)
        s = self.engine.score(r)
        rm_score = s.sub_scores.get("RISK_MANAGEMENT", 1.0)
        self.assertLess(rm_score, 0.50)


class TestCounterfactualEngine(unittest.TestCase):
    """Tests for counterfactual_engine.py"""

    def setUp(self):
        self.engine = CounterfactualEngine()

    def test_analyse_no_outcome(self):
        r = make_record()
        results = self.engine.analyse(r)
        self.assertEqual(results, [])

    def test_analyse_with_outcome(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        results = self.engine.analyse(r)
        self.assertGreater(len(results), 0)

    def test_wait_counterfactual_present(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        results = self.engine.analyse(r)
        actions = [c.alternative_action for c in results]
        self.assertIn("WAIT", actions)

    def test_no_trade_counterfactual(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        results = self.engine.analyse(r)
        actions = [c.alternative_action for c in results]
        self.assertIn("NO_TRADE", actions)

    def test_larger_tp_counterfactual(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        results = self.engine.analyse(r)
        actions = [c.alternative_action for c in results]
        self.assertIn("LARGER_TP", actions)

    def test_tighter_sl_counterfactual(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        results = self.engine.analyse(r)
        actions = [c.alternative_action for c in results]
        self.assertIn("TIGHTER_SL", actions)

    def test_counterfactual_to_dict(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        results = self.engine.analyse(r)
        d = results[0].to_dict()
        self.assertIn("scenario", d)
        self.assertIn("delta_pnl", d)

    def test_best_alternative(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        best = self.engine.best_alternative(r)
        self.assertIsNotNone(best)
        self.assertIsInstance(best.estimated_pnl, float)

    def test_better_alternatives_count(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_negative_outcome(r)  # SL hit — alternatives may be better
        count = self.engine.better_alternatives_count(r)
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_early_trail_on_tp_hit(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        results = self.engine.analyse(r)
        actions = [c.alternative_action for c in results]
        self.assertIn("EARLY_TRAIL", actions)

    def test_record_counterfactuals_attached(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        self.engine.analyse(r)
        self.assertGreater(len(r.counterfactuals), 0)


class TestPatternDiscovery(unittest.TestCase):
    """Tests for pattern_discovery.py"""

    def setUp(self):
        self.engine = PatternDiscovery()

    def _make_records(self, n_pos=5, n_neg=5, regime=None):
        records = []
        for _ in range(n_pos):
            r = make_record(regime=regime or MarketRegime.TRENDING_UP.value)
            add_positive_outcome(r)
            records.append(r)
        for _ in range(n_neg):
            r = make_record(regime=regime or MarketRegime.RANGING.value, session="NY")
            add_negative_outcome(r)
            records.append(r)
        return records

    def test_insufficient_data(self):
        report = self.engine.discover([])
        self.assertIn("Insufficient", report.summary)

    def test_discovers_patterns(self):
        records = self._make_records(5, 5)
        report = self.engine.discover(records)
        self.assertGreater(len(report.patterns), 0)

    def test_pattern_has_fields(self):
        records = self._make_records(5, 5)
        report = self.engine.discover(records)
        if report.patterns:
            p = report.patterns[0]
            self.assertIsInstance(p.win_rate, float)
            self.assertIsInstance(p.sample_size, int)
            self.assertIn(p.strength, ("STRONG", "MODERATE", "WEAK"))
            self.assertIn(p.direction, ("POSITIVE", "NEGATIVE"))

    def test_top_positive_negative_lists(self):
        records = self._make_records(5, 5)
        report = self.engine.discover(records)
        self.assertIsInstance(report.top_positive, list)
        self.assertIsInstance(report.top_negative, list)

    def test_report_to_dict(self):
        records = self._make_records(5, 5)
        report = self.engine.discover(records)
        d = report.to_dict()
        self.assertIn("patterns", d)
        self.assertIn("summary", d)


class TestSelfEvaluation(unittest.TestCase):
    """Tests for self_evaluation.py"""

    def setUp(self):
        self.engine = SelfEvaluationEngine()

    def _make_mixed_records(self, n_pos=4, n_neg=2):
        records = []
        for _ in range(n_pos):
            r = make_record()
            add_positive_outcome(r)
            records.append(r)
        for _ in range(n_neg):
            r = make_record()
            add_negative_outcome(r)
            records.append(r)
        return records

    def test_empty_records(self):
        a = self.engine.evaluate([])
        self.assertIn("not available", a.summary)

    def test_assessment_has_fields(self):
        records = self._make_mixed_records()
        a = self.engine.evaluate(records)
        self.assertIsInstance(a.strengths, list)
        self.assertIsInstance(a.weaknesses, list)
        self.assertIsInstance(a.common_mistakes, list)
        self.assertIsInstance(a.failure_modes, list)
        self.assertIsInstance(a.best_conditions, list)
        self.assertIsInstance(a.worst_conditions, list)
        self.assertIsInstance(a.opportunities, list)

    def test_win_rate_computed(self):
        records = self._make_mixed_records(6, 4)
        a = self.engine.evaluate(records)
        self.assertAlmostEqual(a.overall_win_rate, 0.60, places=4)

    def test_grade_assigned(self):
        records = self._make_mixed_records()
        a = self.engine.evaluate(records)
        self.assertIn(a.grade, ("EXCELLENT", "GOOD", "ACCEPTABLE", "POOR", "REJECT"))

    def test_to_dict(self):
        records = self._make_mixed_records()
        a = self.engine.evaluate(records)
        d = a.to_dict()
        self.assertIn("period", d)
        self.assertIn("strengths", d)
        self.assertIn("grade", d)

    def test_period_label(self):
        records = self._make_mixed_records()
        a = self.engine.evaluate(records, period="MONTHLY")
        self.assertEqual(a.period, "MONTHLY")

    def test_summary_not_empty(self):
        records = self._make_mixed_records()
        a = self.engine.evaluate(records)
        self.assertGreater(len(a.summary), 0)


class TestKnowledgeBase(unittest.TestCase):
    """Tests for knowledge_base.py"""

    def setUp(self):
        self.kb = KnowledgeBase()

    def test_add_and_get(self):
        e = self.kb.create_entry(
            entry_type="MODEL",
            title="Ensemble v3",
            description="Champion ensemble model",
        )
        fetched = self.kb.get(e.entry_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "Ensemble v3")

    def test_total(self):
        self.kb.create_entry("MODEL", "A", "desc A")
        self.kb.create_entry("EXPERIMENT", "B", "desc B")
        self.assertEqual(self.kb.total(), 2)

    def test_search_by_text(self):
        self.kb.create_entry("MODEL", "Alpha Model", "alpha test model")
        self.kb.create_entry("EXPERIMENT", "Beta Exp", "beta experiment")
        results = self.kb.search(query="alpha")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Alpha Model")

    def test_search_by_type(self):
        self.kb.create_entry("MODEL", "M1", "d")
        self.kb.create_entry("EXPERIMENT", "E1", "d")
        self.kb.create_entry("MODEL", "M2", "d")
        models = self.kb.search(entry_type="MODEL")
        self.assertEqual(len(models), 2)

    def test_link_entries(self):
        a = self.kb.create_entry("MODEL", "Model A", "d")
        b = self.kb.create_entry("EXPERIMENT", "Exp B", "d")
        ok = self.kb.link_entries(a.entry_id, b.entry_id)
        self.assertTrue(ok)
        self.assertIn(b.entry_id, a.links)
        self.assertIn(a.entry_id, b.links)

    def test_index_decision(self):
        r = make_record()
        e = self.kb.index_decision(r)
        self.assertEqual(e.entry_type, "DECISION")
        self.assertIn(r.decision_id, e.decision_ids)

    def test_entries_for_decision(self):
        r = make_record()
        self.kb.index_decision(r)
        entries = self.kb.entries_for_decision(r.decision_id)
        self.assertEqual(len(entries), 1)

    def test_by_type_stats(self):
        self.kb.create_entry("MODEL", "M", "d")
        self.kb.create_entry("MODEL", "M2", "d")
        self.kb.create_entry("EXPERIMENT", "E", "d")
        bt = self.kb.by_type()
        self.assertEqual(bt["MODEL"], 2)
        self.assertEqual(bt["EXPERIMENT"], 1)

    def test_search_no_results(self):
        self.kb.create_entry("MODEL", "Alpha", "desc")
        results = self.kb.search(query="zzznotfound")
        self.assertEqual(len(results), 0)

    def test_stats(self):
        self.kb.create_entry("MODEL", "M", "d")
        s = self.kb.stats()
        self.assertIn("total_entries", s)
        self.assertEqual(s["total_entries"], 1)


class TestDecisionLineage(unittest.TestCase):
    """Tests for decision_lineage.py"""

    def setUp(self):
        self.lineage = DecisionLineage()

    def test_create_lineage(self):
        rec = self.lineage.create_lineage(
            decision_id="dec-001",
            model_version="v1.5",
            dataset_name="DS_2024",
            feature_set="FS_v3",
            training_run="RUN_42",
            experiment="EXP_007",
            git_commit="abc1234",
        )
        self.assertIsNotNone(rec)
        self.assertGreater(rec.depth, 0)

    def test_full_lineage_depth(self):
        rec = self.lineage.create_lineage(
            decision_id="dec-002",
            model_version="v2.0",
            dataset_name="DS_2025",
            feature_set="FS_v4",
            training_run="RUN_50",
            experiment="EXP_010",
            git_commit="def5678",
        )
        # Decision + Forecast + Model + Dataset + FeatureSet + TrainingRun + Experiment + GitCommit = 8
        self.assertEqual(rec.depth, 8)

    def test_get_by_decision(self):
        self.lineage.create_lineage(decision_id="dec-abc", model_version="v1")
        rec = self.lineage.get_by_decision("dec-abc")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.decision_id, "dec-abc")

    def test_get_by_decision_missing(self):
        self.assertIsNone(self.lineage.get_by_decision("nonexistent"))

    def test_lineage_node_types(self):
        rec = self.lineage.create_lineage(
            decision_id="dec-xyz",
            model_version="vtest",
            git_commit="commitabc",
        )
        types = [n.node_type for n in rec.nodes]
        self.assertIn("DECISION", types)
        self.assertIn("MODEL", types)
        self.assertIn("GIT_COMMIT", types)

    def test_root_and_leaf(self):
        rec = self.lineage.create_lineage(
            decision_id="dec-rl",
            model_version="v1",
            git_commit="cmt123",
        )
        self.assertEqual(rec.leaf.node_type, "DECISION")
        self.assertEqual(rec.root.node_type, "GIT_COMMIT")

    def test_to_dict(self):
        rec = self.lineage.create_lineage("dec-td", model_version="v1")
        d = rec.to_dict()
        self.assertIn("lineage_id", d)
        self.assertIn("nodes", d)
        self.assertIsInstance(d["nodes"], list)

    def test_decisions_by_model(self):
        self.lineage.create_lineage("d1", model_version="v99")
        self.lineage.create_lineage("d2", model_version="v99")
        self.lineage.create_lineage("d3", model_version="v1")
        decs = self.lineage.decisions_by_model("v99")
        self.assertEqual(len(decs), 2)

    def test_stats(self):
        self.lineage.create_lineage("d1", model_version="v1")
        self.lineage.create_lineage("d2", model_version="v1", git_commit="abc")
        s = self.lineage.stats()
        self.assertEqual(s["total_lineages"], 2)
        self.assertGreater(s["avg_depth"], 0)


class TestDecisionBenchmarking(unittest.TestCase):
    """Tests for decision_benchmarking.py"""

    def setUp(self):
        self.bench = DecisionBenchmarking()

    def _make_records_for_bench(self, n_pos=6, n_neg=4):
        records = []
        for _ in range(n_pos):
            r = make_record()
            add_positive_outcome(r)
            records.append(r)
        for _ in range(n_neg):
            r = make_record()
            add_negative_outcome(r)
            records.append(r)
        return records

    def test_default_baselines_present(self):
        baselines = self.bench.list_baselines()
        names = [b["name"] for b in baselines]
        self.assertIn("Random Strategy", names)

    def test_benchmark_returns_report(self):
        records = self._make_records_for_bench()
        report = self.bench.benchmark(records)
        self.assertIsInstance(report.total_decisions, int)
        self.assertGreater(len(report.comparisons), 0)

    def test_verdict_assigned(self):
        records = self._make_records_for_bench()
        report = self.bench.benchmark(records)
        self.assertIn(report.overall_verdict, ("BETTER", "WORSE", "SIMILAR"))

    def test_create_baseline_from_records(self):
        records = self._make_records_for_bench(7, 3)
        bl = self.bench.create_baseline_from_records("HISTORICAL_Q1", records, "Q1-2025")
        self.assertEqual(bl.name, "HISTORICAL_Q1")
        self.assertEqual(bl.sample_size, 10)
        self.assertAlmostEqual(bl.win_rate, 0.70, places=4)

    def test_benchmark_empty(self):
        report = self.bench.benchmark([])
        self.assertIn("No completed", report.summary)

    def test_comparison_to_dict(self):
        records = self._make_records_for_bench()
        report = self.bench.benchmark(records)
        if report.comparisons:
            d = report.comparisons[0].to_dict()
            self.assertIn("verdict", d)
            self.assertIn("baseline", d)

    def test_add_custom_baseline(self):
        from trainer.memory.decision_benchmarking import BenchmarkBaseline

        bl = BenchmarkBaseline("MY_BASELINE", 0.55, 0.65, 0.0003, 50)
        self.bench.add_baseline(bl)
        names = [b["name"] for b in self.bench.list_baselines()]
        self.assertIn("MY_BASELINE", names)


class TestMemorySearch(unittest.TestCase):
    """Tests for memory_search.py"""

    def setUp(self):
        self.search = MemorySearch()
        self.records = []
        for instr in ("EURUSD", "GBPUSD", "USDJPY"):
            r = make_record(instrument=instr, confidence=0.70)
            add_positive_outcome(r)
            self.records.append(r)
        r_low = make_record(instrument="EURUSD", confidence=0.40)
        add_negative_outcome(r_low)
        self.records.append(r_low)

    def test_search_all(self):
        result = self.search.search(self.records, SearchQuery())
        self.assertEqual(result.total_found, 4)

    def test_search_by_instrument(self):
        q = SearchQuery(instrument="EURUSD")
        result = self.search.search(self.records, q)
        self.assertEqual(result.total_found, 2)

    def test_search_by_type(self):
        q = SearchQuery(decision_type="BUY")
        result = self.search.search(self.records, q)
        self.assertEqual(result.total_found, 4)

    def test_search_by_confidence(self):
        q = SearchQuery(min_confidence=0.65)
        result = self.search.search(self.records, q)
        self.assertEqual(result.total_found, 3)

    def test_search_profitable(self):
        q = SearchQuery(profitable=True)
        result = self.search.search(self.records, q)
        self.assertEqual(result.total_found, 3)

    def test_search_hit_tp(self):
        q = SearchQuery(hit_tp=True)
        result = self.search.search(self.records, q)
        self.assertEqual(result.total_found, 3)

    def test_search_hit_sl(self):
        q = SearchQuery(hit_sl=True)
        result = self.search.search(self.records, q)
        self.assertEqual(result.total_found, 1)

    def test_search_direction_correct(self):
        q = SearchQuery(direction_correct=True)
        result = self.search.search(self.records, q)
        self.assertEqual(result.total_found, 3)

    def test_incorrect_buys(self):
        result = self.search.incorrect_buys(self.records)
        self.assertGreaterEqual(result.total_found, 0)

    def test_high_confidence_decisions(self):
        result = self.search.high_confidence_decisions(self.records, min_conf=0.65)
        self.assertEqual(result.total_found, 3)

    def test_limit_respected(self):
        records = [make_record() for _ in range(20)]
        q = SearchQuery(limit=5)
        result = self.search.search(records, q)
        self.assertLessEqual(len(result.records), 5)

    def test_sort_by_confidence(self):
        q = SearchQuery(sort_by="confidence")
        result = self.search.search(self.records, q)
        confs = [r.snapshot.forecast_confidence for r in result.records]
        self.assertEqual(confs, sorted(confs, reverse=True))

    def test_to_dict(self):
        result = self.search.search(self.records, SearchQuery())
        d = result.to_dict()
        self.assertIn("total_found", d)
        self.assertIn("records", d)


class TestMemoryDatabase(unittest.TestCase):
    """Tests for memory_database.py"""

    def setUp(self):
        self.db = MemoryDatabase(":memory:")

    def test_save_and_count(self):
        r = make_record()
        self.db.save_decision(r)
        self.assertEqual(self.db.count_decisions(), 1)

    def test_save_multiple(self):
        for _ in range(5):
            self.db.save_decision(make_record())
        self.assertEqual(self.db.count_decisions(), 5)

    def test_save_with_outcome(self):
        r = make_record()
        add_positive_outcome(r)
        self.db.save_decision(r)
        self.assertEqual(self.db.count_outcomes(), 1)

    def test_save_with_score(self):
        r = make_record()
        r.score = DecisionScore(overall=0.75, grade=DecisionGrade.GOOD, sub_scores={}, notes=[])
        self.db.save_decision(r)
        self.assertEqual(self.db.count_scores(), 1)

    def test_save_with_counterfactuals(self):
        r = make_record(decision_type=DecisionType.BUY)
        add_positive_outcome(r)
        ce = CounterfactualEngine()
        ce.analyse(r)
        self.db.save_decision(r)
        self.assertGreater(self.db.count_counterfactuals(), 0)

    def test_load_decisions(self):
        for _ in range(3):
            self.db.save_decision(make_record())
        rows = self.db.load_decisions()
        self.assertEqual(len(rows), 3)

    def test_load_decisions_by_instrument(self):
        self.db.save_decision(make_record(instrument="EURUSD"))
        self.db.save_decision(make_record(instrument="GBPUSD"))
        rows = self.db.load_decisions(instrument="EURUSD")
        self.assertEqual(len(rows), 1)

    def test_decision_ids(self):
        r = make_record()
        self.db.save_decision(r)
        ids = self.db.decision_ids()
        self.assertIn(r.decision_id, ids)

    def test_stats(self):
        r = make_record()
        add_positive_outcome(r)
        self.db.save_decision(r)
        s = self.db.stats()
        self.assertEqual(s["decisions"], 1)
        self.assertEqual(s["outcomes"], 1)

    def test_idempotent_save(self):
        r = make_record()
        self.db.save_decision(r)
        self.db.save_decision(r)  # INSERT OR REPLACE
        self.assertEqual(self.db.count_decisions(), 1)


class TestMemoryDashboard(unittest.TestCase):
    """Tests for memory_dashboard.py"""

    def setUp(self):
        self.dash = MemoryDashboard()
        self.records = []
        for _ in range(4):
            r = make_record()
            add_positive_outcome(r)
            self.records.append(r)
        for _ in range(2):
            r = make_record(session="NY", regime=MarketRegime.RANGING.value)
            add_negative_outcome(r)
            self.records.append(r)

    def test_decision_timeline(self):
        d = self.dash.decision_timeline(self.records)
        self.assertEqual(d["view"], "DECISION_TIMELINE")
        self.assertEqual(d["total"], 6)

    def test_decision_tree(self):
        d = self.dash.decision_tree(self.records)
        self.assertEqual(d["view"], "DECISION_TREE")
        self.assertIn("EURUSD", d["tree"])

    def test_quality_distribution_no_scores(self):
        d = self.dash.quality_distribution(self.records)
        self.assertEqual(d["view"], "QUALITY_DISTRIBUTION")
        self.assertEqual(d["total_scored"], 0)

    def test_quality_distribution_with_scores(self):
        engine = QualityEngine()
        for r in self.records:
            engine.score_and_attach(r)
        d = self.dash.quality_distribution(self.records)
        self.assertEqual(d["total_scored"], 6)

    def test_failure_analysis(self):
        d = self.dash.failure_analysis(self.records)
        self.assertEqual(d["view"], "FAILURE_ANALYSIS")
        self.assertEqual(d["total_failures"], 2)

    def test_counterfactual_explorer_empty(self):
        d = self.dash.counterfactual_explorer(self.records)
        self.assertEqual(d["view"], "COUNTERFACTUAL_EXPLORER")
        self.assertEqual(d["total"], 0)

    def test_counterfactual_explorer_with_cf(self):
        ce = CounterfactualEngine()
        for r in self.records:
            ce.analyse(r)
        d = self.dash.counterfactual_explorer(self.records)
        self.assertGreater(d["total"], 0)

    def test_knowledge_graph(self):
        d = self.dash.knowledge_graph(self.records)
        self.assertEqual(d["view"], "KNOWLEDGE_GRAPH")
        self.assertGreater(len(d["nodes"]), 0)

    def test_learning_timeline(self):
        engine = QualityEngine()
        for r in self.records:
            engine.score_and_attach(r)
        d = self.dash.learning_timeline(self.records)
        self.assertEqual(d["view"], "LEARNING_TIMELINE")

    def test_decision_heatmaps(self):
        d = self.dash.decision_heatmaps(self.records)
        self.assertEqual(d["view"], "DECISION_HEATMAPS")
        self.assertGreater(d["total"], 0)

    def test_all_dashboards(self):
        all_d = self.dash.all_dashboards(self.records)
        self.assertIn("decision_timeline", all_d)
        self.assertIn("failure_analysis", all_d)
        self.assertEqual(len(all_d), 8)


class TestMemoryAPI(unittest.TestCase):
    """Integration tests for memory_api.py"""

    def setUp(self):
        self.api = MemoryAPI(db_path=":memory:")

    def test_record_and_get_decision(self):
        record = self.api.record_decision(
            decision_type=DecisionType.BUY,
            instrument="EURUSD",
            snapshot=make_snap(),
            reasoning=make_rea(),
        )
        d = self.api.get_decision(record.decision_id)
        self.assertIsNotNone(d)
        self.assertEqual(d["instrument"], "EURUSD")

    def test_get_decision_missing(self):
        self.assertIsNone(self.api.get_decision("nonexistent"))

    def test_record_outcome_via_api(self):
        record = self.api.record_decision(
            decision_type=DecisionType.BUY,
            instrument="EURUSD",
            snapshot=make_snap(),
            reasoning=make_rea(),
        )
        ok = self.api.record_outcome(
            decision_id=record.decision_id,
            outcome_type=OutcomeType.HIT_TP,
            actual_pnl=0.0015,
            hit_tp=True,
            direction_correct=True,
        )
        self.assertTrue(ok)

    def test_score_decision(self):
        record = self.api.record_decision(
            decision_type=DecisionType.BUY,
            instrument="EURUSD",
            snapshot=make_snap(),
            reasoning=make_rea(),
        )
        self.api.record_outcome(record.decision_id, OutcomeType.HIT_TP, hit_tp=True)
        score = self.api.score_decision(record.decision_id)
        self.assertIsNotNone(score)
        self.assertIn("overall", score)
        self.assertIn("grade", score)

    def test_score_all(self):
        for _ in range(3):
            r = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        result = self.api.score_all()
        self.assertEqual(result["scored"], 3)

    def test_analyse_counterfactuals(self):
        record = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        self.api.record_outcome(
            record.decision_id, OutcomeType.HIT_TP, hit_tp=True, actual_pnl=0.0015
        )
        cfs = self.api.analyse_counterfactuals(record.decision_id)
        self.assertGreater(len(cfs), 0)

    def test_summarise_outcomes(self):
        record = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        self.api.record_outcome(record.decision_id, OutcomeType.HIT_TP, hit_tp=True)
        summary = self.api.summarise_outcomes()
        self.assertIn("win_rate", summary)

    def test_discover_patterns(self):
        for _ in range(5):
            r = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
            self.api.record_outcome(r.decision_id, OutcomeType.HIT_TP, hit_tp=True)
        for _ in range(5):
            r = self.api.record_decision(
                DecisionType.SELL,
                "GBPUSD",
                make_snap(instrument="GBPUSD", session="NY"),
                make_rea(),
            )
            self.api.record_outcome(r.decision_id, OutcomeType.HIT_SL, hit_sl=True)
        report = self.api.discover_patterns()
        self.assertIn("patterns", report)

    def test_self_evaluate(self):
        for _ in range(3):
            r = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
            self.api.record_outcome(r.decision_id, OutcomeType.HIT_TP, hit_tp=True)
        a = self.api.self_evaluate()
        self.assertIn("grade", a)
        self.assertIn("strengths", a)

    def test_search_decisions(self):
        for _ in range(3):
            self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        self.api.record_decision(
            DecisionType.SELL, "GBPUSD", make_snap(instrument="GBPUSD"), make_rea()
        )
        result = self.api.search_decisions(SearchQuery(instrument="EURUSD"))
        self.assertEqual(result["total_found"], 3)

    def test_knowledge_search(self):
        record = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        self.api.index_decision_in_knowledge_base(record.decision_id)
        results = self.api.knowledge_search(query="EURUSD")
        self.assertGreater(len(results), 0)

    def test_build_lineage(self):
        record = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        lin = self.api.build_lineage(
            decision_id=record.decision_id,
            model_version="v2.0",
            git_commit="abc1234",
        )
        self.assertIsNotNone(lin)
        self.assertIn("lineage_id", lin)

    def test_benchmark(self):
        for _ in range(6):
            r = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
            self.api.record_outcome(r.decision_id, OutcomeType.HIT_TP, hit_tp=True)
        for _ in range(4):
            r = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
            self.api.record_outcome(r.decision_id, OutcomeType.HIT_SL, hit_sl=True)
        report = self.api.benchmark()
        self.assertIn("overall_verdict", report)

    def test_get_dashboard_all(self):
        record = self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        self.api.record_outcome(record.decision_id, OutcomeType.HIT_TP, hit_tp=True)
        all_dash = self.api.get_dashboard("all")
        self.assertIn("decision_timeline", all_dash)
        self.assertEqual(len(all_dash), 8)

    def test_get_dashboard_single(self):
        d = self.api.get_dashboard("failure_analysis")
        self.assertEqual(d["view"], "FAILURE_ANALYSIS")

    def test_get_dashboard_unknown(self):
        d = self.api.get_dashboard("nonexistent")
        self.assertIn("error", d)

    def test_system_stats(self):
        self.api.record_decision(DecisionType.BUY, "EURUSD", make_snap(), make_rea())
        s = self.api.system_stats()
        self.assertIn("memory", s)
        self.assertIn("database", s)
        self.assertIn("knowledge", s)
        self.assertIn("lineage", s)


if __name__ == "__main__":
    unittest.main()
