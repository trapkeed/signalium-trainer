"""Phase 21.5 — Trade Lifecycle Intelligence Engine Tests.

Uses spec_from_file_location to avoid importing trainer (which imports torch).
Run: python3 -m unittest tests/unit/test_lifecycle.py -v
"""

import importlib.util
import os
import sys
import time
import unittest

# ── spec_from_file_location loader ───────────────────────────────────────────

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(rel_path: str, mod_name: str):
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load all modules in dependency order
_states = _load("trainer/lifecycle/lifecycle_states.py", "trainer.lifecycle.lifecycle_states")
_models = _load("trainer/lifecycle/lifecycle_models.py", "trainer.lifecycle.lifecycle_models")
_sm = _load("trainer/lifecycle/state_machine.py", "trainer.lifecycle.state_machine")
_te = _load("trainer/lifecycle/transition_engine.py", "trainer.lifecycle.transition_engine")
_pm = _load("trainer/lifecycle/position_monitor.py", "trainer.lifecycle.position_monitor")
_lce = _load("trainer/lifecycle/lifecycle_confidence.py", "trainer.lifecycle.lifecycle_confidence")
_ee = _load("trainer/lifecycle/event_engine.py", "trainer.lifecycle.event_engine")
_are = _load("trainer/lifecycle/ai_reasoning.py", "trainer.lifecycle.ai_reasoning")
_tl = _load("trainer/lifecycle/trade_lifecycle.py", "trainer.lifecycle.trade_lifecycle")
_la = _load("trainer/lifecycle/lifecycle_analytics.py", "trainer.lifecycle.lifecycle_analytics")
_lo = _load("trainer/lifecycle/lifecycle_optimizer.py", "trainer.lifecycle.lifecycle_optimizer")
_db = _load("trainer/lifecycle/lifecycle_database.py", "trainer.lifecycle.lifecycle_database")
_dash = _load("trainer/lifecycle/lifecycle_dashboard.py", "trainer.lifecycle.lifecycle_dashboard")
_api_mod = _load("trainer/lifecycle/lifecycle_api.py", "trainer.lifecycle.lifecycle_api")

# Aliases
LifecycleState = _states.LifecycleState
TERMINAL_STATES = _states.TERMINAL_STATES
CLOSING_STATES = _states.CLOSING_STATES
POSITION_OPEN_STATES = _states.POSITION_OPEN_STATES
PRE_ENTRY_STATES = _states.PRE_ENTRY_STATES
VALID_TRANSITIONS = _states.VALID_TRANSITIONS
STATE_ORDER = _states.STATE_ORDER
STATE_EXPECTED_DURATION_HOURS = _states.STATE_EXPECTED_DURATION_HOURS
is_terminal = _states.is_terminal
is_position_open = _states.is_position_open
can_transition = _states.can_transition
valid_next_states = _states.valid_next_states

MarketContext = _models.MarketContext
PositionMetrics = _models.PositionMetrics
StateRecord = _models.StateRecord
LifecycleEvent = _models.LifecycleEvent
EventType = _models.EventType
TransitionResult = _models.TransitionResult
LifecycleConfidence = _models.LifecycleConfidence

StateMachine = _sm.StateMachine
TransitionRule = _sm.TransitionRule

TransitionEngine = _te.TransitionEngine

PositionMonitor = _pm.PositionMonitor
ManagementRecommendation = _pm.ManagementRecommendation

LifecycleConfidenceEngine = _lce.LifecycleConfidenceEngine

EventEngine = _ee.EventEngine

AiReasoningEngine = _are.AiReasoningEngine

TradeLifecycle = _tl.TradeLifecycle

LifecycleAnalytics = _la.LifecycleAnalytics

LifecycleOptimizer = _lo.LifecycleOptimizer

LifecycleDatabase = _db.LifecycleDatabase

LifecycleDashboard = _dash.LifecycleDashboard

LifecycleAPI = _api_mod.LifecycleAPI


def _ctx(**kw) -> MarketContext:
    defaults = dict(
        price=1.25,
        atr=0.001,
        spread=0.00015,
        volatility=0.012,
        momentum=0.3,
        news_risk=0.15,
        session="LONDON",
        trend_valid=True,
    )
    defaults.update(kw)
    return MarketContext(**defaults)


def _metrics(**kw) -> PositionMetrics:
    defaults = dict(
        lots=0.1,
        entry_price=1.25,
        current_price=1.252,
        unrealised_pnl=0.002,
        unrealised_pct=0.016,
        stop_price=1.249,
        target_price=1.256,
        time_in_trade_h=3.0,
    )
    defaults.update(kw)
    return PositionMetrics(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleStates
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleStates(unittest.TestCase):

    def test_total_state_count(self):
        self.assertEqual(len(LifecycleState), 20)

    def test_terminal_states_count(self):
        self.assertGreaterEqual(len(TERMINAL_STATES), 4)

    def test_closed_is_terminal(self):
        self.assertIn(LifecycleState.CLOSED, TERMINAL_STATES)

    def test_invalidated_is_terminal(self):
        self.assertIn(LifecycleState.INVALIDATED, TERMINAL_STATES)

    def test_cancelled_is_terminal(self):
        self.assertIn(LifecycleState.CANCELLED, TERMINAL_STATES)

    def test_expired_is_terminal(self):
        self.assertIn(LifecycleState.EXPIRED, TERMINAL_STATES)

    def test_active_not_terminal(self):
        self.assertNotIn(LifecycleState.ACTIVE, TERMINAL_STATES)

    def test_position_open_states_non_empty(self):
        self.assertGreater(len(POSITION_OPEN_STATES), 0)

    def test_active_is_position_open(self):
        self.assertIn(LifecycleState.ACTIVE, POSITION_OPEN_STATES)

    def test_idea_not_position_open(self):
        self.assertNotIn(LifecycleState.IDEA, POSITION_OPEN_STATES)

    def test_pre_entry_states_non_empty(self):
        self.assertGreater(len(PRE_ENTRY_STATES), 0)

    def test_valid_transitions_has_all_states(self):
        for state in LifecycleState:
            self.assertIn(state, VALID_TRANSITIONS)

    def test_idea_can_go_to_watchlist(self):
        self.assertIn(LifecycleState.WATCHLIST, VALID_TRANSITIONS[LifecycleState.IDEA])

    def test_closed_has_no_transitions(self):
        self.assertEqual(len(VALID_TRANSITIONS[LifecycleState.CLOSED]), 0)

    def test_invalidated_has_no_transitions(self):
        self.assertEqual(len(VALID_TRANSITIONS[LifecycleState.INVALIDATED]), 0)

    def test_can_transition_true(self):
        self.assertTrue(can_transition(LifecycleState.IDEA, LifecycleState.WATCHLIST))

    def test_can_transition_false(self):
        self.assertFalse(can_transition(LifecycleState.CLOSED, LifecycleState.ACTIVE))

    def test_is_terminal_true(self):
        self.assertTrue(is_terminal(LifecycleState.CLOSED))

    def test_is_terminal_false(self):
        self.assertFalse(is_terminal(LifecycleState.ACTIVE))

    def test_is_position_open(self):
        self.assertTrue(is_position_open(LifecycleState.ACTIVE))
        self.assertFalse(is_position_open(LifecycleState.IDEA))

    def test_state_order_has_all_states(self):
        for state in LifecycleState:
            self.assertIn(state, STATE_ORDER)

    def test_expected_duration_hours_positive(self):
        for state, h in STATE_EXPECTED_DURATION_HOURS.items():
            self.assertGreaterEqual(h, 0.0)

    def test_closing_states_non_empty(self):
        self.assertGreater(len(CLOSING_STATES), 0)

    def test_stopped_is_closing_state(self):
        self.assertIn(LifecycleState.STOPPED, CLOSING_STATES)

    def test_valid_next_states_idea(self):
        nxt = valid_next_states(LifecycleState.IDEA)
        self.assertIn(LifecycleState.WATCHLIST, nxt)


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleModels
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleModels(unittest.TestCase):

    def test_market_context_to_dict_keys(self):
        ctx = _ctx()
        d = ctx.to_dict()
        for k in (
            "price",
            "atr",
            "spread",
            "volatility",
            "momentum",
            "news_risk",
            "session",
            "trend_valid",
        ):
            self.assertIn(k, d)

    def test_position_metrics_to_dict_keys(self):
        m = _metrics()
        d = m.to_dict()
        for k in ("lots", "entry_price", "current_price", "unrealised_pct"):
            self.assertIn(k, d)

    def test_lifecycle_event_create(self):
        ev = LifecycleEvent.create(
            "T001", EventType.STATE_CHANGED, LifecycleState.ACTIVE, "State changed", severity="INFO"
        )
        self.assertEqual(ev.trade_id, "T001")
        self.assertIsNotNone(ev.event_id)
        self.assertIsNotNone(ev.timestamp)

    def test_lifecycle_event_to_dict(self):
        ev = LifecycleEvent.create("T001", "FOO", LifecycleState.IDEA, "desc")
        d = ev.to_dict()
        for k in ("event_id", "trade_id", "event_type", "state", "timestamp", "severity"):
            self.assertIn(k, d)

    def test_state_record_to_dict(self):
        r = StateRecord(
            state=LifecycleState.IDEA,
            timestamp=time.time(),
            confidence=0.6,
            reason="test",
            market_context=_ctx(),
        )
        d = r.to_dict()
        self.assertIn("state", d)
        self.assertIn("confidence", d)
        self.assertIn("market_context", d)

    def test_transition_result_to_dict(self):
        tr = TransitionResult(
            from_state=LifecycleState.IDEA,
            to_state=LifecycleState.WATCHLIST,
            allowed=True,
            confidence=0.7,
            expected_benefit=0.5,
            risk_rating="LOW",
        )
        d = tr.to_dict()
        self.assertIn("allowed", d)
        self.assertIn("confidence", d)
        self.assertTrue(d["allowed"])

    def test_lifecycle_confidence_to_dict(self):
        lc = LifecycleConfidence(
            state=LifecycleState.ACTIVE,
            confidence=0.75,
            expected_duration_h=24.0,
            expected_reward=0.07,
            expected_risk=0.05,
            transition_prob=0.80,
            failure_prob=0.20,
        )
        d = lc.to_dict()
        for k in ("state", "confidence", "expected_duration_h", "transition_prob", "failure_prob"):
            self.assertIn(k, d)


# ─────────────────────────────────────────────────────────────────────────────
# TestStateMachine
# ─────────────────────────────────────────────────────────────────────────────
class TestStateMachine(unittest.TestCase):

    def setUp(self):
        self.sm = StateMachine()

    def test_valid_transition_allowed(self):
        result = self.sm.validate_transition(
            LifecycleState.IDEA, LifecycleState.WATCHLIST, _ctx(), 0.65
        )
        self.assertTrue(result.allowed)

    def test_invalid_transition_blocked(self):
        result = self.sm.validate_transition(
            LifecycleState.CLOSED, LifecycleState.ACTIVE, _ctx(), 0.65
        )
        self.assertFalse(result.allowed)

    def test_transition_with_no_context(self):
        result = self.sm.validate_transition(LifecycleState.WATCHLIST, LifecycleState.PREPARATION)
        self.assertIsNotNone(result)

    def test_result_has_confidence(self):
        result = self.sm.validate_transition(
            LifecycleState.PREPARATION, LifecycleState.ENTRY_READY, _ctx(), 0.70
        )
        self.assertGreater(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_result_has_risk_rating(self):
        result = self.sm.validate_transition(
            LifecycleState.ACTIVE, LifecycleState.PROTECTED, _ctx(), 0.75
        )
        self.assertIn(result.risk_rating, ("LOW", "MEDIUM", "HIGH", "CRITICAL"))

    def test_expected_benefit_range(self):
        result = self.sm.validate_transition(
            LifecycleState.PROTECTED, LifecycleState.RUNNER, _ctx(), 0.75
        )
        self.assertGreaterEqual(result.expected_benefit, 0.0)
        self.assertLessEqual(result.expected_benefit, 1.0)

    def test_news_risk_blocks_entry_ready(self):
        ctx = _ctx(news_risk=0.80)
        result = self.sm.validate_transition(
            LifecycleState.PREPARATION, LifecycleState.ENTRY_READY, ctx, 0.70
        )
        self.assertFalse(result.allowed)

    def test_trend_invalid_blocks_entry_ready(self):
        ctx = _ctx(trend_valid=False)
        result = self.sm.validate_transition(
            LifecycleState.PREPARATION, LifecycleState.ENTRY_READY, ctx, 0.70
        )
        self.assertFalse(result.allowed)

    def test_all_rules_non_empty(self):
        self.assertGreater(len(self.sm.all_rules()), 10)

    def test_rules_for_state_active(self):
        rules = self.sm.rules_for_state(LifecycleState.ACTIVE)
        self.assertGreater(len(rules), 0)

    def test_get_rule_returns_rule(self):
        rule = self.sm.get_rule(LifecycleState.ACTIVE, LifecycleState.PROTECTED)
        self.assertIsNotNone(rule)

    def test_get_rule_missing_returns_none(self):
        rule = self.sm.get_rule(LifecycleState.CLOSED, LifecycleState.ACTIVE)
        self.assertIsNone(rule)

    def test_transition_result_has_reason(self):
        result = self.sm.validate_transition(
            LifecycleState.IDEA, LifecycleState.WATCHLIST, _ctx(), 0.65
        )
        self.assertIsInstance(result.reason, str)
        self.assertGreater(len(result.reason), 0)

    def test_timeout_hours_positive(self):
        result = self.sm.validate_transition(
            LifecycleState.WATCHLIST, LifecycleState.PREPARATION, _ctx(), 0.65
        )
        self.assertGreater(result.timeout_hours, 0.0)

    def test_emergency_exit_from_active(self):
        result = self.sm.validate_transition(
            LifecycleState.ACTIVE, LifecycleState.EMERGENCY_EXIT, _ctx(), 0.65
        )
        self.assertTrue(result.allowed)


# ─────────────────────────────────────────────────────────────────────────────
# TestTransitionEngine
# ─────────────────────────────────────────────────────────────────────────────
class TestTransitionEngine(unittest.TestCase):

    def setUp(self):
        self.te = TransitionEngine()

    def test_evaluate_returns_recommendation(self):
        rec = self.te.evaluate(LifecycleState.IDEA, 0.65, _ctx())
        self.assertIsNotNone(rec)

    def test_terminal_state_returns_stay(self):
        rec = self.te.evaluate(LifecycleState.CLOSED, 1.0)
        self.assertTrue(rec.should_stay)
        self.assertIsNone(rec.recommended)

    def test_recommendation_has_confidence(self):
        rec = self.te.evaluate(LifecycleState.WATCHLIST, 0.70, _ctx())
        self.assertGreater(rec.confidence, 0.0)

    def test_emergency_exit_on_high_news_risk(self):
        ctx = _ctx(news_risk=0.90)
        rec = self.te.evaluate(LifecycleState.ACTIVE, 0.75, ctx)
        self.assertEqual(rec.recommended, LifecycleState.EMERGENCY_EXIT)

    def test_emergency_exit_on_extreme_vol(self):
        ctx = _ctx(volatility=0.040)
        rec = self.te.evaluate(LifecycleState.ACTIVE, 0.75, ctx)
        self.assertEqual(rec.recommended, LifecycleState.EMERGENCY_EXIT)

    def test_invalidation_on_low_confidence(self):
        rec = self.te.evaluate(LifecycleState.WATCHLIST, 0.30, _ctx())
        self.assertIn(rec.recommended, (LifecycleState.INVALIDATED, None))

    def test_all_candidates_returned(self):
        rec = self.te.evaluate(LifecycleState.ACTIVE, 0.70, _ctx())
        self.assertGreater(len(rec.all_candidates), 0)

    def test_possible_transitions_non_empty(self):
        results = self.te.possible_transitions(LifecycleState.ACTIVE, 0.70, _ctx())
        self.assertGreater(len(results), 0)

    def test_recommendation_reason_non_empty(self):
        rec = self.te.evaluate(LifecycleState.PREPARATION, 0.70, _ctx())
        self.assertIsInstance(rec.reason, str)

    def test_to_dict(self):
        rec = self.te.evaluate(LifecycleState.IDEA, 0.65, _ctx())
        d = rec.to_dict()
        self.assertIn("current_state", d)
        self.assertIn("candidates", d)


# ─────────────────────────────────────────────────────────────────────────────
# TestPositionMonitor
# ─────────────────────────────────────────────────────────────────────────────
class TestPositionMonitor(unittest.TestCase):

    def setUp(self):
        self.pm = PositionMonitor()

    def test_evaluate_returns_plan(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.ACTIVE, _metrics(), _ctx(), 0.70)
        self.assertIsNotNone(plan)

    def test_to_dict_keys(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.ACTIVE)
        d = plan.to_dict()
        for k in ("instrument", "state", "primary", "risk_level", "actions"):
            self.assertIn(k, d)

    def test_high_news_risk_exit_immediately(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.ACTIVE, context=_ctx(news_risk=0.90))
        self.assertEqual(plan.primary, ManagementRecommendation.EXIT_IMMEDIATELY)

    def test_extreme_vol_reduce_exposure(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.ACTIVE, context=_ctx(volatility=0.040))
        self.assertEqual(plan.primary, ManagementRecommendation.REDUCE_EXPOSURE)

    def test_high_confidence_scale_in(self):
        plan = self.pm.evaluate(
            "EURUSD",
            LifecycleState.ACTIVE,
            context=_ctx(news_risk=0.1, volatility=0.010, momentum=0.5),
            confidence=0.80,
        )
        self.assertEqual(plan.primary, ManagementRecommendation.SCALE_IN)

    def test_low_confidence_position_open_reduce(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.ACTIVE, confidence=0.40)
        self.assertIn(
            plan.primary,
            (
                ManagementRecommendation.REDUCE_EXPOSURE,
                ManagementRecommendation.HOLD,
                ManagementRecommendation.EXIT_IMMEDIATELY,
            ),
        )

    def test_exit_ready_triggers_exit(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.EXIT_READY)
        self.assertEqual(plan.primary, ManagementRecommendation.EXIT_IMMEDIATELY)

    def test_actions_list(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.ACTIVE)
        self.assertIsInstance(plan.actions, list)

    def test_risk_level_not_empty(self):
        plan = self.pm.evaluate("EURUSD", LifecycleState.RUNNER)
        self.assertIn(plan.risk_level, ("LOW", "MEDIUM", "HIGH", "CRITICAL"))


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleConfidenceEngine
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleConfidenceEngine(unittest.TestCase):

    def setUp(self):
        self.engine = LifecycleConfidenceEngine()

    def test_compute_returns_confidence_object(self):
        lc = self.engine.compute(LifecycleState.ACTIVE, 0.70, _ctx())
        self.assertIsNotNone(lc)

    def test_confidence_in_range(self):
        lc = self.engine.compute(LifecycleState.WATCHLIST, 0.60)
        self.assertGreater(lc.confidence, 0.0)
        self.assertLessEqual(lc.confidence, 1.0)

    def test_transition_prob_in_range(self):
        lc = self.engine.compute(LifecycleState.ENTRY_READY, 0.72)
        self.assertGreater(lc.transition_prob, 0.0)
        self.assertLessEqual(lc.transition_prob, 1.0)

    def test_failure_prob_in_range(self):
        lc = self.engine.compute(LifecycleState.ACTIVE, 0.65)
        self.assertGreaterEqual(lc.failure_prob, 0.0)
        self.assertLessEqual(lc.failure_prob, 1.0)

    def test_expected_duration_positive(self):
        lc = self.engine.compute(LifecycleState.RUNNER, 0.70)
        self.assertGreater(lc.expected_duration_h, 0.0)

    def test_high_news_risk_increases_failure(self):
        lc_safe = self.engine.compute(LifecycleState.ACTIVE, 0.70, _ctx(news_risk=0.10))
        lc_risk = self.engine.compute(LifecycleState.ACTIVE, 0.70, _ctx(news_risk=0.80))
        self.assertGreater(lc_risk.failure_prob, lc_safe.failure_prob)

    def test_to_dict_keys(self):
        lc = self.engine.compute(LifecycleState.PROTECTED, 0.80)
        d = lc.to_dict()
        for k in ("state", "confidence", "expected_duration_h", "transition_prob", "failure_prob"):
            self.assertIn(k, d)

    def test_closed_state_zero_duration(self):
        lc = self.engine.compute(LifecycleState.CLOSED, 1.0)
        self.assertEqual(lc.expected_duration_h, 0.0)

    def test_notes_is_list(self):
        lc = self.engine.compute(LifecycleState.ACTIVE, 0.65, _ctx(news_risk=0.60))
        self.assertIsInstance(lc.notes, list)


# ─────────────────────────────────────────────────────────────────────────────
# TestEventEngine
# ─────────────────────────────────────────────────────────────────────────────
class TestEventEngine(unittest.TestCase):

    def setUp(self):
        self.ee = EventEngine()

    def test_detect_events_returns_list(self):
        events = self.ee.detect_events("T001", LifecycleState.ACTIVE, _ctx())
        self.assertIsInstance(events, list)

    def test_no_context_returns_empty(self):
        events = self.ee.detect_events("T001", LifecycleState.ACTIVE, None)
        self.assertEqual(len(events), 0)

    def test_confidence_raised_event(self):
        events = self.ee.detect_events(
            "T001",
            LifecycleState.ACTIVE,
            _ctx(),
            prev_confidence=0.60,
            curr_confidence=0.75,
        )
        types = [e.event_type for e in events]
        self.assertIn("CONFIDENCE_RAISED", types)

    def test_confidence_dropped_event(self):
        events = self.ee.detect_events(
            "T001",
            LifecycleState.ACTIVE,
            _ctx(),
            prev_confidence=0.80,
            curr_confidence=0.65,
        )
        types = [e.event_type for e in events]
        self.assertIn("CONFIDENCE_DROPPED", types)

    def test_extreme_vol_event(self):
        events = self.ee.detect_events("T001", LifecycleState.ACTIVE, _ctx(volatility=0.040))
        types = [e.event_type for e in events]
        self.assertIn("VOLATILITY_EXPANDED", types)

    def test_critical_news_event(self):
        events = self.ee.detect_events("T001", LifecycleState.ACTIVE, _ctx(news_risk=0.80))
        sevs = [e.severity for e in events]
        self.assertIn("CRITICAL", sevs)

    def test_trend_invalid_event(self):
        events = self.ee.detect_events(
            "T001",
            LifecycleState.WATCHLIST,
            _ctx(trend_valid=False),
            prev_context=_ctx(trend_valid=True),
        )
        types = [e.event_type for e in events]
        self.assertIn("TREND_INVALID", types)

    def test_emit_state_change(self):
        ev = self.ee.emit_state_change(
            "T001", LifecycleState.IDEA, LifecycleState.WATCHLIST, "Setup detected"
        )
        self.assertEqual(ev.event_type, "STATE_CHANGED")
        self.assertEqual(ev.state, LifecycleState.WATCHLIST)

    def test_emit_emergency_severity_critical(self):
        ev = self.ee.emit_state_change(
            "T001", LifecycleState.ACTIVE, LifecycleState.EMERGENCY_EXIT, "Critical"
        )
        self.assertEqual(ev.severity, "CRITICAL")

    def test_event_to_dict(self):
        ev = self.ee.emit_state_change(
            "T001", LifecycleState.IDEA, LifecycleState.WATCHLIST, "Setup"
        )
        d = ev.to_dict()
        for k in ("event_id", "trade_id", "event_type", "state", "severity"):
            self.assertIn(k, d)

    def test_target_approaching_event(self):
        metrics = _metrics(unrealised_pct=0.020)
        events = self.ee.detect_events("T001", LifecycleState.ACTIVE, _ctx(), metrics)
        types = [e.event_type for e in events]
        self.assertIn("TARGET_APPROACHING", types)


# ─────────────────────────────────────────────────────────────────────────────
# TestAiReasoningEngine
# ─────────────────────────────────────────────────────────────────────────────
class TestAiReasoningEngine(unittest.TestCase):

    def setUp(self):
        self.are = AiReasoningEngine()

    def test_generate_returns_report(self):
        report = self.are.generate("T001", "EURUSD", "BUY", LifecycleState.ACTIVE, 0.75)
        self.assertIsNotNone(report)

    def test_headline_not_empty(self):
        report = self.are.generate("T001", "EURUSD", "BUY", LifecycleState.ACTIVE, 0.75)
        self.assertGreater(len(report.headline), 0)

    def test_rationale_is_list(self):
        report = self.are.generate("T001", "EURUSD", "BUY", LifecycleState.WATCHLIST, 0.65)
        self.assertIsInstance(report.rationale, list)
        self.assertGreater(len(report.rationale), 0)

    def test_risk_factors_elevated_news(self):
        report = self.are.generate(
            "T001", "EURUSD", "BUY", LifecycleState.ACTIVE, 0.70, context=_ctx(news_risk=0.60)
        )
        self.assertGreater(len(report.risk_factors), 0)

    def test_action_items_active(self):
        report = self.are.generate("T001", "EURUSD", "BUY", LifecycleState.ACTIVE, 0.70)
        self.assertGreater(len(report.action_items), 0)

    def test_confidence_narrative_improving(self):
        report = self.are.generate(
            "T001", "EURUSD", "BUY", LifecycleState.ACTIVE, 0.80, prev_confidence=0.65
        )
        self.assertIn("increased", report.confidence_narrative.lower())

    def test_confidence_narrative_declining(self):
        report = self.are.generate(
            "T001", "EURUSD", "BUY", LifecycleState.ACTIVE, 0.55, prev_confidence=0.75
        )
        self.assertIn("declined", report.confidence_narrative.lower())

    def test_outlook_not_empty(self):
        report = self.are.generate("T001", "EURUSD", "BUY", LifecycleState.RUNNER, 0.72)
        self.assertGreater(len(report.outlook), 0)

    def test_to_dict_keys(self):
        report = self.are.generate("T001", "EURUSD", "BUY", LifecycleState.ACTIVE, 0.70)
        d = report.to_dict()
        for k in ("headline", "rationale", "risk_factors", "action_items", "outlook"):
            self.assertIn(k, d)

    def test_stopped_rationale(self):
        report = self.are.generate("T001", "EURUSD", "BUY", LifecycleState.STOPPED, 0.0)
        self.assertGreater(len(report.rationale), 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestTradeLifecycle
# ─────────────────────────────────────────────────────────────────────────────
class TestTradeLifecycle(unittest.TestCase):

    def _new(self, conf=0.60):
        return TradeLifecycle.new("EURUSD", "BUY", confidence=conf, context=_ctx())

    def test_new_starts_at_idea(self):
        tl = self._new()
        self.assertEqual(tl.current_state, LifecycleState.IDEA)

    def test_trade_id_not_empty(self):
        tl = self._new()
        self.assertGreater(len(tl.trade_id), 0)

    def test_initial_history_length(self):
        tl = self._new()
        self.assertEqual(len(tl.state_history), 1)

    def test_valid_transition_succeeds(self):
        tl = self._new()
        ok = tl.transition_to(LifecycleState.WATCHLIST, "Setup detected")
        self.assertTrue(ok)
        self.assertEqual(tl.current_state, LifecycleState.WATCHLIST)

    def test_invalid_transition_fails(self):
        tl = self._new()
        ok = tl.transition_to(LifecycleState.ACTIVE, "Skip ahead")
        self.assertFalse(ok)
        self.assertEqual(tl.current_state, LifecycleState.IDEA)

    def test_state_history_grows(self):
        tl = self._new()
        tl.transition_to(LifecycleState.WATCHLIST, "Setup")
        self.assertEqual(len(tl.state_history), 2)

    def test_events_generated_on_transition(self):
        tl = self._new()
        tl.transition_to(LifecycleState.WATCHLIST, "Setup")
        self.assertGreater(len(tl.events), 0)

    def test_terminal_after_closed(self):
        tl = self._new()
        # Manual path to CLOSED via CLOSING
        tl.transition_to(LifecycleState.WATCHLIST, "1")
        tl.transition_to(LifecycleState.PREPARATION, "2")
        tl.transition_to(LifecycleState.ENTRY_READY, "3")
        tl.transition_to(LifecycleState.PENDING_ENTRY, "4")
        tl.transition_to(LifecycleState.ACTIVE, "5")
        tl.transition_to(LifecycleState.EXIT_READY, "6")
        tl.transition_to(LifecycleState.CLOSING, "7")
        tl.transition_to(LifecycleState.CLOSED, "8")
        self.assertTrue(tl.is_terminal)

    def test_no_transition_after_terminal(self):
        tl = self._new()
        tl.transition_to(LifecycleState.INVALIDATED, "Invalid")
        ok = tl.transition_to(LifecycleState.WATCHLIST, "Revival")
        self.assertFalse(ok)

    def test_is_position_open(self):
        tl = self._new()
        self.assertFalse(tl.is_position_open)
        tl.transition_to(LifecycleState.WATCHLIST, "1")
        tl.transition_to(LifecycleState.PREPARATION, "2")
        tl.transition_to(LifecycleState.ENTRY_READY, "3")
        tl.transition_to(LifecycleState.PENDING_ENTRY, "4")
        tl.transition_to(LifecycleState.ACTIVE, "5")
        self.assertTrue(tl.is_position_open)

    def test_age_hours_positive(self):
        tl = self._new()
        self.assertGreaterEqual(tl.age_hours, 0.0)

    def test_recommend_transition_returns_rec(self):
        tl = self._new()
        rec = tl.recommend_transition(_ctx())
        self.assertIsNotNone(rec)

    def test_lifecycle_confidence_returns_lc(self):
        tl = self._new()
        lc = tl.lifecycle_confidence(_ctx())
        self.assertIsNotNone(lc)

    def test_position_management_returns_plan(self):
        tl = self._new()
        plan = tl.position_management(_ctx())
        self.assertIsNotNone(plan)

    def test_reasoning_returns_report(self):
        tl = self._new()
        report = tl.reasoning(_ctx())
        self.assertIsNotNone(report)

    def test_to_dict_keys(self):
        tl = self._new()
        d = tl.to_dict()
        for k in (
            "trade_id",
            "instrument",
            "direction",
            "current_state",
            "confidence",
            "state_history",
            "events",
        ):
            self.assertIn(k, d)

    def test_summary_compact(self):
        tl = self._new()
        s = tl.summary()
        self.assertIn("trade_id", s)
        self.assertNotIn("state_history", s)

    def test_detect_events(self):
        tl = self._new()
        evs = tl.detect_events(_ctx(), prev_conf=0.50, prev_context=_ctx(news_risk=0.0))
        self.assertIsInstance(evs, list)

    def test_critical_events_filtered(self):
        tl = self._new()
        tl.detect_events(_ctx(news_risk=0.85))
        # critical_events property returns CRITICAL only
        self.assertIsInstance(tl.critical_events, list)

    def test_state_count(self):
        tl = self._new()
        tl.transition_to(LifecycleState.WATCHLIST, "s")
        self.assertEqual(tl.state_count, 2)

    def test_event_count(self):
        tl = self._new()
        tl.transition_to(LifecycleState.WATCHLIST, "s")
        self.assertGreater(tl.event_count, 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleAnalytics
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleAnalytics(unittest.TestCase):

    def _history(self, states):
        now = time.time()
        records = []
        for i, st in enumerate(states):
            r = StateRecord(state=st, timestamp=now + i * 3600, confidence=0.70, reason="test")
            if i > 0:
                records[i - 1].duration_h = 1.0
            records.append(r)
        return records

    def test_empty_histories(self):
        la = LifecycleAnalytics()
        report = la.analyse([])
        self.assertEqual(report.total_trades, 0)

    def test_total_trades(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.CLOSED])
        report = la.analyse([h, h], [LifecycleState.CLOSED, LifecycleState.CLOSED])
        self.assertEqual(report.total_trades, 2)

    def test_completed_count(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.CLOSED])
        report = la.analyse([h], [LifecycleState.CLOSED])
        self.assertEqual(report.completed_trades, 1)

    def test_cancelled_rate(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.CANCELLED])
        report = la.analyse([h], [LifecycleState.CANCELLED])
        self.assertGreater(report.cancellation_rate, 0.0)

    def test_state_analytics_non_empty(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.WATCHLIST, LifecycleState.CLOSED])
        report = la.analyse([h])
        self.assertGreater(len(report.state_analytics), 0)

    def test_to_dict_keys(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.CLOSED])
        report = la.analyse([h])
        d = report.to_dict()
        for k in ("total_trades", "failure_rate", "state_analytics", "top_paths"):
            self.assertIn(k, d)

    def test_transition_frequency(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.WATCHLIST])
        freq = la.transition_frequency([h])
        self.assertIn("IDEA→WATCHLIST", freq)

    def test_state_time_distribution(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.WATCHLIST])
        dist = la.state_time_distribution([h])
        self.assertIn("IDEA", dist)

    def test_top_paths_non_empty(self):
        la = LifecycleAnalytics()
        h = self._history([LifecycleState.IDEA, LifecycleState.CLOSED])
        report = la.analyse([h, h])
        self.assertGreater(len(report.top_paths), 0)


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleOptimizer
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleOptimizer(unittest.TestCase):

    def _history(self, states):
        now = time.time()
        records = []
        for i, st in enumerate(states):
            r = StateRecord(state=st, timestamp=now + i * 3600, confidence=0.70, reason="test")
            if i > 0:
                records[i - 1].duration_h = 1.0
            records.append(r)
        return records

    def test_empty_returns_report(self):
        lo = LifecycleOptimizer()
        report = lo.optimise([])
        self.assertEqual(report.total_analysed, 0)

    def test_basic_report(self):
        lo = LifecycleOptimizer()
        h = self._history([LifecycleState.IDEA, LifecycleState.ACTIVE, LifecycleState.CLOSED])
        report = lo.optimise([h])
        self.assertIsNotNone(report)

    def test_summary_not_empty(self):
        lo = LifecycleOptimizer()
        h = self._history([LifecycleState.IDEA, LifecycleState.ACTIVE])
        report = lo.optimise([h])
        self.assertGreater(len(report.summary), 0)

    def test_findings_is_list(self):
        lo = LifecycleOptimizer()
        h = self._history([LifecycleState.IDEA])
        report = lo.optimise([h])
        self.assertIsInstance(report.findings, list)

    def test_to_dict_keys(self):
        lo = LifecycleOptimizer()
        h = self._history([LifecycleState.IDEA, LifecycleState.CLOSED])
        report = lo.optimise([h])
        d = report.to_dict()
        for k in ("total_analysed", "findings", "summary", "key_insight"):
            self.assertIn(k, d)

    def test_high_expiry_rate_generates_finding(self):
        lo = LifecycleOptimizer()
        histories = [
            self._history([LifecycleState.IDEA, LifecycleState.WATCHLIST]) for _ in range(10)
        ]
        final = [LifecycleState.EXPIRED] * 10
        report = lo.optimise(histories, final)
        cats = [f.category for f in report.findings]
        self.assertIn("TIMING", cats)

    def test_finding_has_priority(self):
        lo = LifecycleOptimizer()
        histories = [self._history([LifecycleState.IDEA, LifecycleState.EXPIRED]) for _ in range(5)]
        final = [LifecycleState.EXPIRED] * 5
        report = lo.optimise(histories, final)
        if report.findings:
            self.assertIsInstance(report.findings[0].priority, int)


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleDatabase
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleDatabase(unittest.TestCase):

    def setUp(self):
        self.db = LifecycleDatabase()

    def test_stats_zero_on_empty(self):
        s = self.db.stats()
        self.assertEqual(s["trade_lifecycle"], 0)
        self.assertEqual(s["lifecycle_events"], 0)

    def test_upsert_lifecycle(self):
        self.db.upsert_lifecycle(
            {
                "trade_id": "T001",
                "instrument": "EURUSD",
                "direction": "BUY",
                "current_state": "IDEA",
                "confidence": 0.60,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )
        s = self.db.stats()
        self.assertEqual(s["trade_lifecycle"], 1)

    def test_get_lifecycle(self):
        self.db.upsert_lifecycle(
            {
                "trade_id": "T002",
                "instrument": "GBPUSD",
                "direction": "SELL",
                "current_state": "WATCHLIST",
                "confidence": 0.65,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )
        row = self.db.get_lifecycle("T002")
        self.assertIsNotNone(row)
        self.assertEqual(row["trade_id"], "T002")

    def test_get_lifecycle_missing(self):
        self.assertIsNone(self.db.get_lifecycle("NONEXISTENT"))

    def test_insert_event(self):
        ev = LifecycleEvent.create("T001", "STATE_CHANGED", LifecycleState.IDEA, "test")
        self.db.insert_event(ev.to_dict())
        s = self.db.stats()
        self.assertEqual(s["lifecycle_events"], 1)

    def test_get_events_for_trade(self):
        ev = LifecycleEvent.create("T003", "FOO", LifecycleState.IDEA, "test")
        self.db.insert_event(ev.to_dict())
        rows = self.db.get_events_for_trade("T003")
        self.assertEqual(len(rows), 1)

    def test_insert_state_record(self):
        r = StateRecord(
            state=LifecycleState.IDEA, timestamp=time.time(), confidence=0.60, reason="test"
        )
        self.db.insert_state_record("T001", r.to_dict())
        s = self.db.stats()
        self.assertEqual(s["state_history"], 1)

    def test_get_state_history(self):
        r = StateRecord(
            state=LifecycleState.IDEA, timestamp=time.time(), confidence=0.60, reason="test2"
        )
        self.db.insert_state_record("T004", r.to_dict())
        rows = self.db.get_state_history("T004")
        self.assertEqual(len(rows), 1)

    def test_upsert_transition_rule(self):
        self.db.upsert_transition_rule(
            {
                "from_state": "IDEA",
                "to_state": "WATCHLIST",
                "entry_conditions": [],
                "exit_conditions": [],
                "validation_rules": [],
                "timeout_hours": 24.0,
                "base_confidence": 0.60,
                "expected_benefit": 0.4,
                "risk_rating": "LOW",
                "description": "Test rule",
            }
        )
        rules = self.db.get_transition_rules()
        self.assertEqual(len(rules), 1)

    def test_save_analytics_snapshot(self):
        row_id = self.db.save_analytics_snapshot(
            {
                "total_trades": 5,
                "completed_trades": 3,
            }
        )
        self.assertGreater(row_id, 0)

    def test_get_latest_analytics(self):
        self.db.save_analytics_snapshot({"total_trades": 10})
        snap = self.db.get_latest_analytics()
        self.assertIsNotNone(snap)
        self.assertEqual(snap["total_trades"], 10)

    def test_save_optimisation(self):
        row_id = self.db.save_optimisation(
            {
                "total_analysed": 3,
                "findings": [],
                "summary": "Test",
            }
        )
        self.assertGreater(row_id, 0)

    def test_list_lifecycles(self):
        self.db.upsert_lifecycle(
            {
                "trade_id": "T010",
                "instrument": "EURUSD",
                "direction": "BUY",
                "current_state": "ACTIVE",
                "confidence": 0.75,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        )
        rows = self.db.list_lifecycles()
        self.assertGreater(len(rows), 0)

    def test_upsert_updates_existing(self):
        base = {
            "trade_id": "T020",
            "instrument": "EURUSD",
            "direction": "BUY",
            "current_state": "IDEA",
            "confidence": 0.60,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self.db.upsert_lifecycle(base)
        base["current_state"] = "WATCHLIST"
        base["confidence"] = 0.68
        self.db.upsert_lifecycle(base)
        row = self.db.get_lifecycle("T020")
        self.assertEqual(row["current_state"], "WATCHLIST")


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleDashboard
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleDashboard(unittest.TestCase):

    def setUp(self):
        self.dash = LifecycleDashboard()
        now = time.time()
        self.history = [
            StateRecord(LifecycleState.IDEA, now, 0.60, "created", duration_h=2.0),
            StateRecord(LifecycleState.WATCHLIST, now + 7200, 0.65, "setup", duration_h=3.0),
            StateRecord(LifecycleState.ACTIVE, now + 18000, 0.72, "active"),
        ]
        self.events = [
            LifecycleEvent.create("T001", "STATE_CHANGED", LifecycleState.IDEA, "created"),
        ]

    def test_lifecycle_timeline_view(self):
        view = self.dash.lifecycle_timeline(
            "T001", "EURUSD", "BUY", self.history, LifecycleState.ACTIVE
        )
        self.assertEqual(view["view"], "lifecycle_timeline")
        self.assertEqual(len(view["timeline"]), 3)

    def test_timeline_total_hours(self):
        view = self.dash.lifecycle_timeline(
            "T001", "EURUSD", "BUY", self.history, LifecycleState.ACTIVE
        )
        self.assertGreater(view["total_hours"], 0.0)

    def test_state_transition_graph(self):
        view = self.dash.state_transition_graph(self.history)
        self.assertEqual(view["view"], "state_transition_graph")
        self.assertIn("edges", view)

    def test_trade_journey(self):
        view = self.dash.trade_journey("T001", "EURUSD", "BUY", self.history, self.events)
        self.assertEqual(view["view"], "trade_journey")
        self.assertEqual(view["total_states"], 3)

    def test_confidence_timeline(self):
        view = self.dash.confidence_timeline("T001", self.history)
        self.assertEqual(view["view"], "confidence_timeline")
        self.assertIn("trend", view)
        self.assertIn(view["trend"], ("IMPROVING", "DECLINING", "STABLE"))

    def test_transition_matrix_view(self):
        view = self.dash.transition_matrix([self.history])
        self.assertEqual(view["view"], "transition_matrix")
        self.assertIn("prob_matrix", view)

    def test_lifecycle_statistics_view(self):
        view = self.dash.lifecycle_statistics([self.history])
        self.assertEqual(view["view"], "lifecycle_statistics")
        self.assertIn("state_visit_counts", view)

    def test_confidence_trend_improving(self):
        h = [
            StateRecord(LifecycleState.IDEA, time.time(), 0.50, "low"),
            StateRecord(LifecycleState.ACTIVE, time.time() + 3600, 0.80, "high"),
        ]
        view = self.dash.confidence_timeline("T001", h)
        self.assertEqual(view["trend"], "IMPROVING")

    def test_confidence_trend_declining(self):
        h = [
            StateRecord(LifecycleState.IDEA, time.time(), 0.80, "high"),
            StateRecord(LifecycleState.ACTIVE, time.time() + 3600, 0.50, "low"),
        ]
        view = self.dash.confidence_timeline("T001", h)
        self.assertEqual(view["trend"], "DECLINING")


# ─────────────────────────────────────────────────────────────────────────────
# TestLifecycleAPI
# ─────────────────────────────────────────────────────────────────────────────
class TestLifecycleAPI(unittest.TestCase):

    def setUp(self):
        self.api = LifecycleAPI()

    def test_create_lifecycle(self):
        r = self.api.create_lifecycle("EURUSD", "BUY", 0.65, _ctx())
        self.assertTrue(r.success)
        self.assertIn("trade_id", r.data)

    def test_create_stores_in_active(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        self.assertTrue(r.success)
        tid = r.data["trade_id"]
        self.assertIn(tid, self.api._active)

    def test_get_lifecycle_found(self):
        r = self.api.create_lifecycle("GBPUSD", "SELL")
        tid = r.data["trade_id"]
        r2 = self.api.get_lifecycle(tid)
        self.assertTrue(r2.success)
        self.assertEqual(r2.data["trade_id"], tid)

    def test_get_lifecycle_not_found(self):
        r = self.api.get_lifecycle("NONEXISTENT")
        self.assertFalse(r.success)

    def test_transition_valid(self):
        r = self.api.create_lifecycle("EURUSD", "BUY", 0.65)
        tid = r.data["trade_id"]
        r2 = self.api.transition(tid, LifecycleState.WATCHLIST, "Setup")
        self.assertTrue(r2.success)

    def test_transition_invalid_blocked(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.transition(tid, LifecycleState.ACTIVE, "Skip")
        self.assertFalse(r2.success)

    def test_auto_transition_returns(self):
        r = self.api.create_lifecycle("EURUSD", "BUY", 0.70, _ctx())
        tid = r.data["trade_id"]
        r2 = self.api.auto_transition(tid, _ctx())
        self.assertTrue(r2.success)

    def test_recommend_transition_read_only(self):
        r = self.api.create_lifecycle("EURUSD", "BUY", 0.70)
        tid = r.data["trade_id"]
        r2 = self.api.recommend_transition(tid)
        self.assertTrue(r2.success)
        self.assertIn("current_state", r2.data)

    def test_position_management(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.position_management(tid)
        self.assertTrue(r2.success)
        self.assertIn("primary", r2.data)

    def test_lifecycle_confidence(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.lifecycle_confidence(tid)
        self.assertTrue(r2.success)
        self.assertIn("confidence", r2.data)

    def test_detect_events(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.detect_events(tid, _ctx())
        self.assertTrue(r2.success)
        self.assertIn("new_events", r2.data)

    def test_reasoning(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.reasoning(tid)
        self.assertTrue(r2.success)
        self.assertIn("headline", r2.data)

    def test_state_history(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.state_history(tid)
        self.assertTrue(r2.success)
        self.assertGreater(r2.data["count"], 0)

    def test_lifecycle_analytics(self):
        self.api.create_lifecycle("EURUSD", "BUY")
        r = self.api.lifecycle_analytics()
        self.assertTrue(r.success)
        self.assertIn("total_trades", r.data)

    def test_optimisation(self):
        self.api.create_lifecycle("EURUSD", "BUY")
        r = self.api.optimisation()
        self.assertTrue(r.success)
        self.assertIn("summary", r.data)

    def test_timeline_dashboard(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.timeline_dashboard(tid)
        self.assertTrue(r2.success)
        self.assertEqual(r2.data["view"], "lifecycle_timeline")

    def test_confidence_dashboard(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        tid = r.data["trade_id"]
        r2 = self.api.confidence_dashboard(tid)
        self.assertTrue(r2.success)
        self.assertEqual(r2.data["view"], "confidence_timeline")

    def test_transition_dashboard(self):
        self.api.create_lifecycle("EURUSD", "BUY")
        r = self.api.transition_dashboard()
        self.assertTrue(r.success)
        self.assertEqual(r.data["view"], "transition_matrix")

    def test_statistics_dashboard(self):
        self.api.create_lifecycle("EURUSD", "BUY")
        r = self.api.statistics_dashboard()
        self.assertTrue(r.success)
        self.assertEqual(r.data["view"], "lifecycle_statistics")

    def test_db_stats(self):
        r = self.api.db_stats()
        self.assertTrue(r.success)
        self.assertIn("trade_lifecycle", r.data)

    def test_active_trades(self):
        self.api.create_lifecycle("EURUSD", "BUY")
        r = self.api.active_trades()
        self.assertTrue(r.success)
        self.assertGreater(r.data["count"], 0)

    def test_api_response_to_dict(self):
        r = self.api.create_lifecycle("EURUSD", "BUY")
        d = r.to_dict()
        for k in ("success", "endpoint", "data"):
            self.assertIn(k, d)

    def test_transition_missing_trade(self):
        r = self.api.transition("NONEXISTENT", LifecycleState.WATCHLIST, "x")
        self.assertFalse(r.success)

    def test_full_lifecycle_path_api(self):
        r = self.api.create_lifecycle("EURUSD", "BUY", 0.70)
        tid = r.data["trade_id"]
        path = [
            (LifecycleState.WATCHLIST, "Setup"),
            (LifecycleState.PREPARATION, "Prep"),
            (LifecycleState.ENTRY_READY, "Ready"),
            (LifecycleState.PENDING_ENTRY, "Pending"),
            (LifecycleState.ACTIVE, "Active"),
            (LifecycleState.INVALIDATED, "Invalid"),
        ]
        for state, reason in path:
            resp = self.api.transition(tid, state, reason)
            if not resp.success:
                break
        r2 = self.api.get_lifecycle(tid)
        self.assertTrue(r2.success)


if __name__ == "__main__":
    unittest.main()
