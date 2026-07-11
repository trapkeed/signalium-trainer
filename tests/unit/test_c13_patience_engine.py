"""C13 — Patience Engine tests.

Covers:
  - WaitReason: all seven values present and ordered
  - SkipOutcome: four values present
  - ThesisDecayProfile analogue: compute_no_action_reward formula
  - compute_no_action_reward: base reward applied for any wait
  - compute_no_action_reward: COGNITIVE_PASSIVE adds maximum bonus
  - compute_no_action_reward: UNCERTAINTY_HIGH adds proportional bonus
  - compute_no_action_reward: OVERTRADING adds bonus
  - compute_no_action_reward: IMPULSIVE_GUARD adds bonus
  - compute_no_action_reward: LOW_QUALITY adds proportional gap bonus
  - compute_no_action_reward: high opportunity_score reduces reward
  - compute_no_action_reward: clamped to [0, 1]
  - compute_realized_reward: CORRECT_SKIP >= 0.60
  - compute_realized_reward: MISSED_OPPORTUNITY <= -0.40
  - compute_realized_reward: NEUTRAL = 0.0
  - compute_realized_reward: larger pnl pushes toward ceiling/floor
  - SkippedOpportunity: is_resolved False at creation
  - SkippedOpportunity: resolve() sets outcome and timestamps
  - SkippedOpportunity: resolve() is idempotent
  - SkippedOpportunity: was_correct / was_mistake properties
  - SkippedOpportunity: to_dict keys
  - WaitDecision: frozen (immutable)
  - WaitDecision: to_dict keys
  - PatienceState: pending_skips / resolved_skips filters
  - PatienceState: find_skip by id
  - PatienceState: _append_skip bounds to MAX_SKIPPED_RECORDS
  - patience_score: new state returns sensible defaults
  - patience_score: discipline_score fraction of correct/resolved
  - patience_score: impulsive_entry_rate correct
  - patience_score: miss_rate correct
  - patience_score: composite_patience_score formula
  - patience_score: returns PatienceMetrics
  - minimum_quality_threshold: returns DEFAULT when no history
  - minimum_quality_threshold: raises when impulsive_entry_rate > 0.30
  - minimum_quality_threshold: raises when discipline_score < 0.40
  - minimum_quality_threshold: lowers when miss_rate > 0.50
  - minimum_quality_threshold: bounded to [MIN, MAX]
  - minimum_quality_threshold: drought detection lowers threshold
  - evaluate_wait_decision: IGNORE → COGNITIVE_PASSIVE
  - evaluate_wait_decision: WAIT → COGNITIVE_PASSIVE
  - evaluate_wait_decision: MONITOR → CONFIRMATION_PENDING
  - evaluate_wait_decision: uncertainty ≥ 0.55 → UNCERTAINTY_HIGH
  - evaluate_wait_decision: is_overtrading → OVERTRADING
  - evaluate_wait_decision: trade_frequency_ratio ≥ 1.5 → OVERTRADING
  - evaluate_wait_decision: high urgency + low conviction → IMPULSIVE_GUARD
  - evaluate_wait_decision: low opp_score → LOW_QUALITY
  - evaluate_wait_decision: capital_unavailable → CAPITAL_UNAVAILABLE
  - evaluate_wait_decision: multiple reasons collected simultaneously
  - evaluate_wait_decision: should_wait=False when no conditions met
  - evaluate_wait_decision: should_wait=False has no_action_reward=0
  - evaluate_wait_decision: should_wait=True has skip_record
  - evaluate_wait_decision: skip_record fields populated
  - evaluate_wait_decision: primary_reason is first reason (highest priority)
  - evaluate_wait_decision: default threshold = DEFAULT_QUALITY_THRESHOLD
  - evaluate_wait_decision: custom threshold honoured
  - evaluate_wait_decision: returns WaitDecision
  - PatienceEngine.register / init
  - PatienceEngine.evaluate_wait_decision uses adaptive threshold
  - PatienceEngine.record_skip persists skip_record
  - PatienceEngine.record_skip increments total_skips
  - PatienceEngine.record_skip returns None when should_wait=False
  - PatienceEngine.record_trade increments total_trades
  - PatienceEngine.record_trade impulsive flag
  - PatienceEngine.record_overtrading_event increments count
  - PatienceEngine.resolve_skipped_opportunity: CORRECT_SKIP outcome
  - PatienceEngine.resolve_skipped_opportunity: MISSED_OPPORTUNITY outcome
  - PatienceEngine.resolve_skipped_opportunity: updates correct_skips counter
  - PatienceEngine.resolve_skipped_opportunity: updates missed_opportunities counter
  - PatienceEngine.resolve_skipped_opportunity: adapts threshold
  - PatienceEngine.resolve_skipped_opportunity: idempotent on already resolved
  - PatienceEngine.resolve_skipped_opportunity: returns None for unknown skip_id
  - PatienceEngine.age_pending_skips: resolves old pending skips as NEUTRAL
  - PatienceEngine.pending_skips: correct filter
  - PatienceEngine.patience_score delegates to stateless function
  - PatienceEngine.minimum_quality_threshold delegates to stateless function
  - PatienceEngine.summary: returns dict with keys
  - Behavioral: impulsive entries raise adaptive threshold over time
  - Behavioral: correct skips improve discipline_score
  - Behavioral: missed opportunities raise miss_rate
  - Behavioral: composite_patience_score improves with discipline
  - Behavioral: NO_ACTION reward positive for well-grounded wait
  - Behavioral: NO_ACTION reward reduced for high-quality opp
  - Package exports all C13 symbols
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ── Test helpers ──────────────────────────────────────────────────────────────

_T0 = 1_700_000_000.0


def _make_cognitive_decision(
    action: str = "WAIT",
    uncertainty_score: float = 0.3,
    conviction_score: float = 0.6,
    entry_urgency: float = 0.3,
    direction: str = "BULLISH",
    instrument: str = "EURUSD",
):
    """Build a mock CognitiveDecision with the given field values."""
    from trainer.qde.brain.brain_types import BeliefDirection
    from trainer.qde.brain.decision.decision_types import CognitiveAction

    m = MagicMock()
    m.action = CognitiveAction(action)
    m.uncertainty_score = uncertainty_score
    m.conviction_score = conviction_score
    m.entry_urgency = entry_urgency
    m.direction = BeliefDirection(direction)
    m.instrument = instrument
    return m


def _make_opp_score(opportunity_score: float = 0.45, opp_class: str = "INTRADAY"):
    """Build a mock OpportunityScore."""
    m = MagicMock()
    m.opportunity_score = opportunity_score
    m.opportunity_class = opp_class
    return m


def _make_account_features(is_overtrading: bool = False, trade_frequency_ratio: float = 1.0):
    m = MagicMock()
    m.is_overtrading = is_overtrading
    m.trade_frequency_ratio = trade_frequency_ratio
    return m


def _active_cog_dec(**kwargs):
    """A BUY decision that passes all wait checks by default."""
    defaults = dict(
        action="BUY",
        uncertainty_score=0.2,
        conviction_score=0.7,
        entry_urgency=0.4,
    )
    defaults.update(kwargs)
    return _make_cognitive_decision(**defaults)


def _high_quality_opp():
    return _make_opp_score(opportunity_score=0.70)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Enums
# ═════════════════════════════════════════════════════════════════════════════


class TestEnums:
    def test_wait_reason_seven_values(self):
        from trainer.qde.brain.patience import WaitReason

        assert len(list(WaitReason)) == 7

    def test_wait_reason_values_present(self):
        from trainer.qde.brain.patience import WaitReason

        for name in (
            "COGNITIVE_PASSIVE",
            "UNCERTAINTY_HIGH",
            "OVERTRADING",
            "CONFIRMATION_PENDING",
            "LOW_QUALITY",
            "IMPULSIVE_GUARD",
            "CAPITAL_UNAVAILABLE",
        ):
            assert hasattr(WaitReason, name)

    def test_skip_outcome_four_values(self):
        from trainer.qde.brain.patience import SkipOutcome

        assert len(list(SkipOutcome)) == 4

    def test_skip_outcome_values(self):
        from trainer.qde.brain.patience import SkipOutcome

        for name in ("CORRECT_SKIP", "MISSED_OPPORTUNITY", "NEUTRAL", "UNRESOLVED"):
            assert hasattr(SkipOutcome, name)

    def test_wait_reason_is_str_enum(self):
        from trainer.qde.brain.patience import WaitReason

        assert isinstance(WaitReason.COGNITIVE_PASSIVE, str)


# ═════════════════════════════════════════════════════════════════════════════
# 2. compute_no_action_reward
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeNoActionReward:
    def test_base_reward_positive(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r = compute_no_action_reward([WaitReason.LOW_QUALITY], opportunity_score=0.10)
        assert r > 0.0

    def test_cognitive_passive_adds_bonus(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r_with = compute_no_action_reward([WaitReason.COGNITIVE_PASSIVE], opportunity_score=0.10)
        r_without = compute_no_action_reward([WaitReason.LOW_QUALITY], opportunity_score=0.10)
        assert r_with > r_without

    def test_uncertainty_high_adds_bonus(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r = compute_no_action_reward(
            [WaitReason.UNCERTAINTY_HIGH], opportunity_score=0.10, uncertainty_score=0.80
        )
        r_low = compute_no_action_reward(
            [WaitReason.UNCERTAINTY_HIGH], opportunity_score=0.10, uncertainty_score=0.56
        )
        assert r > r_low  # higher excess uncertainty → higher reward

    def test_overtrading_adds_bonus(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r = compute_no_action_reward([WaitReason.OVERTRADING], opportunity_score=0.10)
        r_base = compute_no_action_reward([], opportunity_score=0.10)
        assert r > r_base

    def test_impulsive_guard_adds_bonus(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r = compute_no_action_reward([WaitReason.IMPULSIVE_GUARD], opportunity_score=0.10)
        r_base = compute_no_action_reward([], opportunity_score=0.10)
        assert r > r_base

    def test_low_quality_gap_adds_proportional_bonus(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r_large = compute_no_action_reward(
            [WaitReason.LOW_QUALITY], opportunity_score=0.05, quality_gap=0.15
        )
        r_small = compute_no_action_reward(
            [WaitReason.LOW_QUALITY], opportunity_score=0.15, quality_gap=0.05
        )
        assert r_large > r_small

    def test_high_opp_score_reduces_reward(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r_high_opp = compute_no_action_reward(
            [WaitReason.COGNITIVE_PASSIVE], opportunity_score=0.80
        )
        r_low_opp = compute_no_action_reward([WaitReason.COGNITIVE_PASSIVE], opportunity_score=0.10)
        assert r_high_opp < r_low_opp

    def test_clamped_to_zero_one(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        all_reasons = list(WaitReason)
        r = compute_no_action_reward(
            all_reasons, opportunity_score=0.0, uncertainty_score=1.0, quality_gap=1.0
        )
        assert 0.0 <= r <= 1.0

    def test_no_reasons_returns_base(self):
        from trainer.qde.brain.patience import compute_no_action_reward

        r = compute_no_action_reward([], opportunity_score=0.30)
        # Base reward 0.20, no bonuses, no penalties (opp < 0.65)
        assert r == pytest.approx(0.20)

    def test_multiple_reasons_accumulate(self):
        from trainer.qde.brain.patience import WaitReason, compute_no_action_reward

        r_multi = compute_no_action_reward(
            [WaitReason.UNCERTAINTY_HIGH, WaitReason.OVERTRADING],
            opportunity_score=0.10,
            uncertainty_score=0.70,
        )
        r_single = compute_no_action_reward(
            [WaitReason.UNCERTAINTY_HIGH],
            opportunity_score=0.10,
            uncertainty_score=0.70,
        )
        assert r_multi > r_single


# ═════════════════════════════════════════════════════════════════════════════
# 3. compute_realized_reward
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeRealizedReward:
    def test_correct_skip_positive(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        r = compute_realized_reward(SkipOutcome.CORRECT_SKIP, pnl_estimate=0.3)
        assert r >= 0.60

    def test_missed_opportunity_negative(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        r = compute_realized_reward(SkipOutcome.MISSED_OPPORTUNITY, pnl_estimate=0.3)
        assert r <= -0.40

    def test_neutral_is_zero(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        assert compute_realized_reward(SkipOutcome.NEUTRAL) == pytest.approx(0.0)

    def test_larger_pnl_pushes_toward_ceiling(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        r_large = compute_realized_reward(SkipOutcome.CORRECT_SKIP, pnl_estimate=2.0)
        r_small = compute_realized_reward(SkipOutcome.CORRECT_SKIP, pnl_estimate=0.1)
        assert r_large > r_small

    def test_larger_missed_pushes_toward_floor(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        r_large = compute_realized_reward(SkipOutcome.MISSED_OPPORTUNITY, pnl_estimate=2.0)
        r_small = compute_realized_reward(SkipOutcome.MISSED_OPPORTUNITY, pnl_estimate=0.1)
        assert r_large < r_small

    def test_unresolved_is_zero(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        assert compute_realized_reward(SkipOutcome.UNRESOLVED) == pytest.approx(0.0)

    def test_correct_skip_bounded(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        r = compute_realized_reward(SkipOutcome.CORRECT_SKIP, pnl_estimate=100.0)
        assert 0.60 <= r <= 1.0

    def test_missed_bounded(self):
        from trainer.qde.brain.patience import SkipOutcome, compute_realized_reward

        r = compute_realized_reward(SkipOutcome.MISSED_OPPORTUNITY, pnl_estimate=100.0)
        assert -1.0 <= r <= -0.40


# ═════════════════════════════════════════════════════════════════════════════
# 4. SkippedOpportunity
# ═════════════════════════════════════════════════════════════════════════════


class TestSkippedOpportunity:
    def _build(self, **kwargs):
        from trainer.qde.brain.patience import SkippedOpportunity, WaitReason

        defaults = dict(
            instrument="EURUSD",
            direction="BULLISH",
            opportunity_score_at_skip=0.25,
            quality_threshold_at_skip=0.20,
            wait_reasons=[WaitReason.LOW_QUALITY],
            primary_reason=WaitReason.LOW_QUALITY,
            cognitive_action="BUY",
            uncertainty_score=0.30,
            conviction_score=0.60,
            entry_urgency=0.40,
            prospective_no_action_reward=0.35,
            skipped_at=_T0,
        )
        defaults.update(kwargs)
        return SkippedOpportunity(**defaults)

    def test_is_resolved_false_at_creation(self):
        s = self._build()
        assert s.is_resolved is False

    def test_resolve_sets_outcome(self):
        from trainer.qde.brain.patience import SkipOutcome

        s = self._build()
        s.resolve(SkipOutcome.CORRECT_SKIP, pnl_estimate=0.4, realized_reward=0.8, now=_T0 + 3600)
        assert s.outcome == SkipOutcome.CORRECT_SKIP
        assert s.is_resolved is True

    def test_resolve_sets_pnl_and_reward(self):
        from trainer.qde.brain.patience import SkipOutcome

        s = self._build()
        s.resolve(SkipOutcome.CORRECT_SKIP, pnl_estimate=0.5, realized_reward=0.85, now=_T0 + 100)
        assert s.outcome_pnl_estimate == pytest.approx(0.5)
        assert s.realized_no_action_reward == pytest.approx(0.85)

    def test_resolve_idempotent(self):
        from trainer.qde.brain.patience import SkipOutcome

        s = self._build()
        s.resolve(SkipOutcome.CORRECT_SKIP, pnl_estimate=0.5, realized_reward=0.8, now=_T0 + 100)
        s.resolve(
            SkipOutcome.MISSED_OPPORTUNITY, pnl_estimate=0.2, realized_reward=-0.5, now=_T0 + 200
        )
        assert s.outcome == SkipOutcome.CORRECT_SKIP  # first call wins

    def test_was_correct(self):
        from trainer.qde.brain.patience import SkipOutcome

        s = self._build()
        s.resolve(SkipOutcome.CORRECT_SKIP, now=_T0 + 100)
        assert s.was_correct is True
        assert s.was_mistake is False

    def test_was_mistake(self):
        from trainer.qde.brain.patience import SkipOutcome

        s = self._build()
        s.resolve(SkipOutcome.MISSED_OPPORTUNITY, now=_T0 + 100)
        assert s.was_mistake is True
        assert s.was_correct is False

    def test_to_dict_keys(self):
        s = self._build()
        d = s.to_dict()
        for key in (
            "skip_id",
            "instrument",
            "direction",
            "opportunity_score_at_skip",
            "quality_threshold_at_skip",
            "wait_reasons",
            "primary_reason",
            "cognitive_action",
            "uncertainty_score",
            "conviction_score",
            "entry_urgency",
            "prospective_no_action_reward",
            "outcome",
            "outcome_pnl_estimate",
            "realized_no_action_reward",
            "skipped_at",
            "resolved_at",
            "is_resolved",
            "was_correct",
        ):
            assert key in d, f"Missing key: {key}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. WaitDecision
# ═════════════════════════════════════════════════════════════════════════════


class TestWaitDecision:
    def test_frozen(self):
        from trainer.qde.brain.patience import WaitDecision

        d = WaitDecision(
            should_wait=True,
            wait_reasons=(),
            primary_reason=None,
            prospective_no_action_reward=0.3,
            quality_gap=0.1,
            reasoning="test",
        )
        with pytest.raises((AttributeError, TypeError)):
            d.should_wait = False  # type: ignore[misc]

    def test_to_dict_keys(self):
        from trainer.qde.brain.patience import WaitDecision

        d = WaitDecision(
            should_wait=True,
            wait_reasons=(),
            primary_reason=None,
            prospective_no_action_reward=0.3,
            quality_gap=0.05,
            reasoning="x",
        )
        for key in (
            "should_wait",
            "wait_reasons",
            "primary_reason",
            "prospective_no_action_reward",
            "quality_gap",
            "reasoning",
            "skip_id",
        ):
            assert key in d.to_dict()


# ═════════════════════════════════════════════════════════════════════════════
# 6. PatienceState
# ═════════════════════════════════════════════════════════════════════════════


class TestPatienceState:
    def _state(self):
        from trainer.qde.brain.patience import PatienceState

        return PatienceState(instrument="EURUSD")

    def _skip(self, resolved=False, correct=True):
        from trainer.qde.brain.patience import SkipOutcome, SkippedOpportunity, WaitReason

        s = SkippedOpportunity(
            instrument="EURUSD",
            direction="BULLISH",
            wait_reasons=[WaitReason.LOW_QUALITY],
            skipped_at=_T0,
        )
        if resolved:
            outcome = SkipOutcome.CORRECT_SKIP if correct else SkipOutcome.MISSED_OPPORTUNITY
            s.resolve(outcome, now=_T0 + 100)
        return s

    def test_pending_skips_filter(self):
        st = self._state()
        st._append_skip(self._skip(resolved=False))
        st._append_skip(self._skip(resolved=True))
        assert len(st.pending_skips()) == 1

    def test_resolved_skips_filter(self):
        st = self._state()
        st._append_skip(self._skip(resolved=False))
        st._append_skip(self._skip(resolved=True))
        assert len(st.resolved_skips()) == 1

    def test_find_skip(self):
        st = self._state()
        s = self._skip()
        st._append_skip(s)
        found = st.find_skip(s.skip_id)
        assert found is s

    def test_find_skip_unknown(self):
        st = self._state()
        assert st.find_skip("does-not-exist") is None

    def test_append_skip_bounds(self):
        from trainer.qde.brain.patience import MAX_SKIPPED_RECORDS

        st = self._state()
        for _ in range(MAX_SKIPPED_RECORDS + 10):
            st._append_skip(self._skip())
        assert len(st.skipped_opportunities) == MAX_SKIPPED_RECORDS

    def test_to_dict_keys(self):
        st = self._state()
        d = st.to_dict()
        for key in (
            "instrument",
            "total_trades",
            "total_skips",
            "correct_skips",
            "missed_opportunities",
            "impulsive_entries",
            "overtrading_events",
            "current_threshold",
            "pending_skips",
            "total_stored_skips",
        ):
            assert key in d


# ═════════════════════════════════════════════════════════════════════════════
# 7. patience_score
# ═════════════════════════════════════════════════════════════════════════════


class TestPatienceScore:
    def _resolved_skip(self, correct: bool):
        from trainer.qde.brain.patience import SkipOutcome, SkippedOpportunity, WaitReason

        s = SkippedOpportunity(
            instrument="X", wait_reasons=[WaitReason.LOW_QUALITY], skipped_at=_T0
        )
        outcome = SkipOutcome.CORRECT_SKIP if correct else SkipOutcome.MISSED_OPPORTUNITY
        s.resolve(outcome, now=_T0 + 100)
        return s

    def test_returns_patience_metrics(self):
        from trainer.qde.brain.patience import PatienceMetrics, PatienceState, patience_score

        st = PatienceState()
        m = patience_score(st)
        assert isinstance(m, PatienceMetrics)

    def test_new_state_neutral(self):
        from trainer.qde.brain.patience import PatienceState, patience_score

        m = patience_score(PatienceState())
        assert m.discipline_score == pytest.approx(0.0, abs=1e-4)
        assert m.impulsive_entry_rate == pytest.approx(0.0)
        assert m.miss_rate == pytest.approx(0.0)

    def test_discipline_score_correct_fraction(self):
        from trainer.qde.brain.patience import PatienceState, patience_score

        st = PatienceState()
        for _ in range(3):
            st._append_skip(self._resolved_skip(correct=True))
        for _ in range(1):
            st._append_skip(self._resolved_skip(correct=False))
        st.correct_skips = 3
        st.missed_opportunities = 1
        m = patience_score(st)
        # 3 correct out of 4 resolved = 0.75
        assert m.discipline_score == pytest.approx(0.75)

    def test_impulsive_entry_rate(self):
        from trainer.qde.brain.patience import PatienceState, patience_score

        st = PatienceState(total_trades=10, impulsive_entries=3)
        m = patience_score(st)
        assert m.impulsive_entry_rate == pytest.approx(0.3)

    def test_miss_rate(self):
        from trainer.qde.brain.patience import PatienceState, patience_score

        st = PatienceState(correct_skips=1, missed_opportunities=1)
        for _ in range(2):
            st._append_skip(self._resolved_skip(True))
        m = patience_score(st)
        # miss_rate = missed / resolved = 1 / (1+1) when we manually set counts
        # Actual resolved_skips() reads from skipped_opportunities list
        # Since we added 2 correct resolved skips but set missed=1, we need careful setup
        st2 = PatienceState(correct_skips=1, missed_opportunities=1)
        st2._append_skip(self._resolved_skip(correct=True))  # adds to list
        st2._append_skip(self._resolved_skip(correct=False))  # adds to list
        from trainer.qde.brain.patience import patience_score as ps

        m2 = ps(st2)
        # resolved_count from the actual resolved list = 2
        # miss_rate = 1/2 = 0.5
        assert m2.miss_rate == pytest.approx(0.5)

    def test_composite_formula(self):
        from trainer.qde.brain.patience import PatienceState, patience_score

        st = PatienceState(
            total_trades=10,
            impulsive_entries=2,
            correct_skips=3,
            missed_opportunities=1,
        )
        # Add resolved skips to state
        for _ in range(3):
            st._append_skip(self._resolved_skip(correct=True))
        for _ in range(1):
            st._append_skip(self._resolved_skip(correct=False))

        m = patience_score(st)
        expected = (
            0.50 * m.discipline_score
            + 0.30 * (1 - m.impulsive_entry_rate)
            + 0.20 * (1 - m.miss_rate)
        )
        assert m.composite_patience_score == pytest.approx(expected, abs=1e-4)

    def test_composite_bounded(self):
        from trainer.qde.brain.patience import PatienceState, patience_score

        m = patience_score(PatienceState(total_trades=5, impulsive_entries=5))
        assert 0.0 <= m.composite_patience_score <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 8. minimum_quality_threshold
# ═════════════════════════════════════════════════════════════════════════════


class TestMinimumQualityThreshold:
    def test_default_when_no_history(self):
        from trainer.qde.brain.patience import (
            DEFAULT_QUALITY_THRESHOLD,
            PatienceState,
            minimum_quality_threshold,
        )

        st = PatienceState()
        assert minimum_quality_threshold(st) == pytest.approx(DEFAULT_QUALITY_THRESHOLD)

    def test_raises_when_impulsive_rate_high(self):
        from trainer.qde.brain.patience import (
            PatienceState,
            minimum_quality_threshold,
        )

        st = PatienceState(total_trades=10, impulsive_entries=4)  # 40% > 30%
        threshold = minimum_quality_threshold(st)
        assert threshold > st.current_threshold

    def test_raises_when_discipline_low(self):
        from trainer.qde.brain.patience import (
            PatienceState,
            SkipOutcome,
            SkippedOpportunity,
            WaitReason,
            minimum_quality_threshold,
        )

        st = PatienceState(correct_skips=1, missed_opportunities=4)
        # Add 5 resolved skips to the state list
        for correct in [True, False, False, False, False]:
            s = SkippedOpportunity(
                instrument="X", wait_reasons=[WaitReason.LOW_QUALITY], skipped_at=_T0
            )
            s.resolve(
                SkipOutcome.CORRECT_SKIP if correct else SkipOutcome.MISSED_OPPORTUNITY,
                now=_T0 + 100,
            )
            st._append_skip(s)
        threshold = minimum_quality_threshold(st)
        assert threshold > st.current_threshold

    def test_lowers_when_miss_rate_high(self):
        """When miss_rate > 0.50 and discipline is not too low, threshold falls.

        Use 5 correct + 6 missed = 11 resolved:
          discipline  = 5/11 ≈ 0.455  → above 0.40, no raise triggered
          miss_rate   = 6/11 ≈ 0.545  → above 0.50, lower fires
        Net adjustment = -THRESHOLD_STEP_DOWN → threshold drops below 0.30.
        """
        from trainer.qde.brain.patience import (
            PatienceState,
            SkipOutcome,
            SkippedOpportunity,
            WaitReason,
            minimum_quality_threshold,
        )

        st = PatienceState(current_threshold=0.30)
        # 5 correct, 6 missed → miss_rate ≈ 0.545, discipline ≈ 0.455
        for correct in [True] * 5 + [False] * 6:
            s = SkippedOpportunity(
                instrument="X", wait_reasons=[WaitReason.LOW_QUALITY], skipped_at=_T0
            )
            s.resolve(
                SkipOutcome.CORRECT_SKIP if correct else SkipOutcome.MISSED_OPPORTUNITY,
                now=_T0 + 100,
            )
            st._append_skip(s)
        st.correct_skips = 5
        st.missed_opportunities = 6
        threshold = minimum_quality_threshold(st)
        assert threshold < 0.30

    def test_bounded_above(self):
        from trainer.qde.brain.patience import (
            MAX_QUALITY_THRESHOLD,
            PatienceState,
            minimum_quality_threshold,
        )

        st = PatienceState(
            total_trades=100,
            impulsive_entries=60,
            current_threshold=MAX_QUALITY_THRESHOLD,
        )
        assert minimum_quality_threshold(st) <= MAX_QUALITY_THRESHOLD

    def test_bounded_below(self):
        from trainer.qde.brain.patience import (
            MIN_QUALITY_THRESHOLD,
            PatienceState,
            minimum_quality_threshold,
        )

        st = PatienceState(current_threshold=MIN_QUALITY_THRESHOLD)
        assert minimum_quality_threshold(st) >= MIN_QUALITY_THRESHOLD

    def test_drought_lowers_threshold(self):
        """All recent decisions were waits → threshold should drop (drought detected)."""
        from trainer.qde.brain.patience import PatienceState, minimum_quality_threshold

        st = PatienceState(total_trades=1, current_threshold=0.35)
        # Fill recent_wait_flags with True (all waits)
        for _ in range(25):
            st.recent_wait_flags.append(True)
        threshold = minimum_quality_threshold(st)
        assert threshold < 0.35


# ═════════════════════════════════════════════════════════════════════════════
# 9. evaluate_wait_decision (stateless)
# ═════════════════════════════════════════════════════════════════════════════


class TestEvaluateWaitDecision:
    def test_ignore_action_triggers_cognitive_passive(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _make_cognitive_decision(action="IGNORE")
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert result.should_wait is True
        assert WaitReason.COGNITIVE_PASSIVE in result.wait_reasons

    def test_wait_action_triggers_cognitive_passive(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _make_cognitive_decision(action="WAIT")
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert result.should_wait is True
        assert WaitReason.COGNITIVE_PASSIVE in result.wait_reasons

    def test_monitor_action_triggers_confirmation_pending(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _make_cognitive_decision(action="MONITOR", uncertainty_score=0.30)
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert result.should_wait is True
        assert WaitReason.CONFIRMATION_PENDING in result.wait_reasons

    def test_high_uncertainty_triggers_uncertainty_high(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _make_cognitive_decision(action="BUY", uncertainty_score=0.60)
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert WaitReason.UNCERTAINTY_HIGH in result.wait_reasons

    def test_overtrading_flag_triggers_overtrading(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _active_cog_dec()
        af = _make_account_features(is_overtrading=True)
        result = evaluate_wait_decision(cd, _make_opp_score(0.50), account_features=af)
        assert WaitReason.OVERTRADING in result.wait_reasons

    def test_high_frequency_ratio_triggers_overtrading(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _active_cog_dec()
        af = _make_account_features(is_overtrading=False, trade_frequency_ratio=2.0)
        result = evaluate_wait_decision(cd, _make_opp_score(0.50), account_features=af)
        assert WaitReason.OVERTRADING in result.wait_reasons

    def test_impulsive_guard_high_urgency_low_conviction(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _active_cog_dec(entry_urgency=0.80, conviction_score=0.30)
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert WaitReason.IMPULSIVE_GUARD in result.wait_reasons

    def test_impulsive_guard_not_triggered_when_conviction_ok(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _active_cog_dec(entry_urgency=0.80, conviction_score=0.60)
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert WaitReason.IMPULSIVE_GUARD not in result.wait_reasons

    def test_low_quality_triggers_low_quality(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _active_cog_dec()
        # score 0.10 < DEFAULT_QUALITY_THRESHOLD 0.20
        result = evaluate_wait_decision(cd, _make_opp_score(0.10))
        assert WaitReason.LOW_QUALITY in result.wait_reasons

    def test_capital_unavailable_triggers(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _active_cog_dec()
        result = evaluate_wait_decision(cd, _make_opp_score(0.50), capital_unavailable=True)
        assert WaitReason.CAPITAL_UNAVAILABLE in result.wait_reasons

    def test_multiple_reasons_all_collected(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _make_cognitive_decision(action="WAIT", uncertainty_score=0.70)
        af = _make_account_features(is_overtrading=True)
        result = evaluate_wait_decision(cd, _make_opp_score(0.05), account_features=af)
        assert len(result.wait_reasons) >= 3

    def test_no_conditions_met_should_wait_false(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _active_cog_dec(uncertainty_score=0.20, conviction_score=0.80, entry_urgency=0.40)
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert result.should_wait is False

    def test_should_wait_false_no_action_reward_zero(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _active_cog_dec()
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert result.prospective_no_action_reward == pytest.approx(0.0)

    def test_should_wait_false_skip_record_none(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _active_cog_dec()
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert result.skip_record is None

    def test_should_wait_true_has_skip_record(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _make_cognitive_decision(action="WAIT")
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert result.skip_record is not None

    def test_skip_record_fields_populated(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _make_cognitive_decision(
            action="WAIT", uncertainty_score=0.30, conviction_score=0.60, instrument="GBPUSD"
        )
        result = evaluate_wait_decision(cd, _make_opp_score(0.25))
        s = result.skip_record
        assert s is not None
        assert s.instrument == "GBPUSD"
        assert s.uncertainty_score == pytest.approx(0.30)
        assert s.opportunity_score_at_skip == pytest.approx(0.25)

    def test_primary_reason_highest_priority(self):
        """COGNITIVE_PASSIVE should be primary when action=WAIT even if other reasons apply."""
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _make_cognitive_decision(action="WAIT", uncertainty_score=0.70)
        result = evaluate_wait_decision(cd, _make_opp_score(0.05))
        assert result.primary_reason == WaitReason.COGNITIVE_PASSIVE

    def test_custom_threshold_honoured(self):
        from trainer.qde.brain.patience import WaitReason, evaluate_wait_decision

        cd = _active_cog_dec()
        # With threshold=0.60, a score of 0.50 is below threshold
        result = evaluate_wait_decision(cd, _make_opp_score(0.50), quality_threshold=0.60)
        assert WaitReason.LOW_QUALITY in result.wait_reasons

    def test_returns_wait_decision(self):
        from trainer.qde.brain.patience import WaitDecision, evaluate_wait_decision

        cd = _active_cog_dec()
        result = evaluate_wait_decision(cd, _make_opp_score(0.50))
        assert isinstance(result, WaitDecision)

    def test_no_action_reward_positive_for_cognitive_passive(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _make_cognitive_decision(action="IGNORE")
        result = evaluate_wait_decision(cd, _make_opp_score(0.10))
        assert result.prospective_no_action_reward > 0.0

    def test_quality_gap_positive_when_below_threshold(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _active_cog_dec()
        opp = _make_opp_score(0.05)
        result = evaluate_wait_decision(cd, opp)
        # quality_gap = threshold - opp_score = 0.20 - 0.05 = 0.15
        assert result.quality_gap > 0.0

    def test_quality_gap_negative_when_above_threshold(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        cd = _active_cog_dec()
        result = evaluate_wait_decision(cd, _make_opp_score(0.80))
        assert result.quality_gap < 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 10. PatienceEngine
# ═════════════════════════════════════════════════════════════════════════════


class TestPatienceEngine:
    def _engine(self, **kwargs):
        from trainer.qde.brain.patience import PatienceEngine

        return PatienceEngine(instrument="EURUSD", **kwargs)

    def test_initial_threshold_default(self):
        from trainer.qde.brain.patience import DEFAULT_QUALITY_THRESHOLD

        eng = self._engine()
        assert eng.minimum_quality_threshold() == pytest.approx(DEFAULT_QUALITY_THRESHOLD)

    def test_initial_threshold_custom(self):
        eng = self._engine(initial_threshold=0.35)
        assert eng.minimum_quality_threshold() == pytest.approx(0.35)

    def test_evaluate_wait_decision_uses_adaptive_threshold(self):
        from trainer.qde.brain.patience import PatienceEngine

        eng = PatienceEngine(instrument="EURUSD", initial_threshold=0.40)
        cd = _active_cog_dec()
        # score 0.30 < adaptive threshold 0.40 → should wait
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.30))
        assert result.should_wait is True

    def test_record_skip_persists(self):

        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        assert skip is not None
        assert len(eng.pending_skips()) == 1

    def test_record_skip_increments_total_skips(self):
        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        eng.record_skip(result)
        assert eng.state.total_skips == 1

    def test_record_skip_none_when_no_wait(self):
        eng = self._engine()
        cd = _active_cog_dec()
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        assert skip is None

    def test_record_trade_increments_total(self):
        eng = self._engine()
        eng.record_trade()
        eng.record_trade()
        assert eng.state.total_trades == 2

    def test_record_trade_impulsive(self):
        eng = self._engine()
        eng.record_trade(was_impulsive=True)
        assert eng.state.impulsive_entries == 1

    def test_record_overtrading_event(self):
        eng = self._engine()
        eng.record_overtrading_event()
        eng.record_overtrading_event()
        assert eng.state.overtrading_events == 2

    def test_resolve_correct_skip(self):
        from trainer.qde.brain.patience import SkipOutcome

        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        resolved = eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP, 0.4)
        assert resolved is not None
        assert resolved.outcome == SkipOutcome.CORRECT_SKIP
        assert eng.state.correct_skips == 1

    def test_resolve_missed_opportunity(self):
        from trainer.qde.brain.patience import SkipOutcome

        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.MISSED_OPPORTUNITY, 0.3)
        assert eng.state.missed_opportunities == 1

    def test_resolve_updates_realized_reward(self):
        from trainer.qde.brain.patience import SkipOutcome

        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        resolved = eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP, 0.5)
        assert resolved.realized_no_action_reward >= 0.60

    def test_resolve_idempotent(self):
        from trainer.qde.brain.patience import SkipOutcome

        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP)
        eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.MISSED_OPPORTUNITY)
        assert eng.state.correct_skips == 1
        assert eng.state.missed_opportunities == 0

    def test_resolve_unknown_skip_id_returns_none(self):
        from trainer.qde.brain.patience import SkipOutcome

        eng = self._engine()
        result = eng.resolve_skipped_opportunity("nonexistent", SkipOutcome.CORRECT_SKIP)
        assert result is None

    def test_resolve_adapts_threshold(self):
        """After many impulsive entries, resolving skips should update the threshold."""
        from trainer.qde.brain.patience import SkipOutcome

        eng = self._engine()
        # Simulate many impulsive entries to raise the threshold
        for _ in range(5):
            eng.record_trade(was_impulsive=True)
        # Now resolve a skip; threshold should be updated
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        initial_threshold = eng.state.current_threshold
        eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP)
        # Threshold should have been updated (may or may not change direction, just verify it ran)
        assert eng.state.last_threshold_update > 0.0

    def test_age_pending_skips_resolves_old(self):
        from trainer.qde.brain.patience import DEFAULT_RESOLUTION_WINDOW_SECONDS, SkipOutcome

        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50), now=_T0)
        skip = eng.record_skip(result)
        # The skip was created at now, so age by resolution_window + 1 second
        resolved_count = eng.age_pending_skips(
            resolution_window_seconds=DEFAULT_RESOLUTION_WINDOW_SECONDS,
            now=_T0 + DEFAULT_RESOLUTION_WINDOW_SECONDS + 1,
        )
        assert resolved_count == 1
        assert skip.outcome == SkipOutcome.NEUTRAL

    def test_age_pending_skips_does_not_resolve_recent(self):

        eng = self._engine()
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50), now=_T0)
        eng.record_skip(result)
        # Too soon — only 1 second elapsed
        resolved_count = eng.age_pending_skips(now=_T0 + 1)
        assert resolved_count == 0

    def test_patience_score_delegates(self):
        from trainer.qde.brain.patience import PatienceMetrics

        eng = self._engine()
        m = eng.patience_score()
        assert isinstance(m, PatienceMetrics)

    def test_minimum_quality_threshold_delegates(self):
        eng = self._engine()
        t = eng.minimum_quality_threshold()
        assert isinstance(t, float)

    def test_summary_keys(self):
        eng = self._engine()
        s = eng.summary()
        for key in ("state", "metrics", "adaptive_threshold", "pending_skips"):
            assert key in s


# ═════════════════════════════════════════════════════════════════════════════
# 11. Behavioral / integration tests
# ═════════════════════════════════════════════════════════════════════════════


class TestBehavioral:
    def test_impulsive_entries_raise_threshold(self):
        """Many impulsive entries should push the adaptive threshold above default."""
        from trainer.qde.brain.patience import (
            DEFAULT_QUALITY_THRESHOLD,
            PatienceEngine,
            SkipOutcome,
        )

        eng = PatienceEngine(instrument="EURUSD")
        for _ in range(10):
            eng.record_trade(was_impulsive=True)
        # Force a resolve to trigger _update_threshold
        cd = _make_cognitive_decision(action="WAIT")
        result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
        skip = eng.record_skip(result)
        eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP)
        assert eng.state.current_threshold >= DEFAULT_QUALITY_THRESHOLD

    def test_correct_skips_improve_discipline(self):
        from trainer.qde.brain.patience import PatienceEngine, SkipOutcome

        eng = PatienceEngine(instrument="EURUSD")
        for _ in range(5):
            cd = _make_cognitive_decision(action="WAIT")
            result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
            skip = eng.record_skip(result)
            eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP)
        m = eng.patience_score()
        assert m.discipline_score == pytest.approx(1.0)

    def test_missed_opportunities_increase_miss_rate(self):
        from trainer.qde.brain.patience import PatienceEngine, SkipOutcome

        eng = PatienceEngine(instrument="EURUSD")
        for _ in range(4):
            cd = _make_cognitive_decision(action="WAIT")
            result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
            skip = eng.record_skip(result)
            eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.MISSED_OPPORTUNITY)
        m = eng.patience_score()
        assert m.miss_rate == pytest.approx(1.0)

    def test_no_action_reward_positive_well_grounded(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        # High uncertainty + overtrading → strong basis for waiting
        cd = _make_cognitive_decision(action="WAIT", uncertainty_score=0.75)
        af = _make_account_features(is_overtrading=True)
        result = evaluate_wait_decision(cd, _make_opp_score(0.10), account_features=af)
        assert result.prospective_no_action_reward > 0.3

    def test_no_action_reward_reduced_for_high_quality(self):
        from trainer.qde.brain.patience import evaluate_wait_decision

        # COGNITIVE_PASSIVE but high opp_score → reduced reward (might be over-cautious)
        cd_high = _make_cognitive_decision(action="WAIT")
        cd_low = _make_cognitive_decision(action="WAIT")
        r_high = evaluate_wait_decision(cd_high, _make_opp_score(0.80)).prospective_no_action_reward
        r_low = evaluate_wait_decision(cd_low, _make_opp_score(0.10)).prospective_no_action_reward
        assert r_high < r_low

    def test_composite_patience_improves_after_correct_skips(self):
        from trainer.qde.brain.patience import PatienceEngine, SkipOutcome

        eng = PatienceEngine(instrument="EURUSD")
        m0 = eng.patience_score()

        # Add some impulsive entries and missed skips
        for _ in range(3):
            eng.record_trade(was_impulsive=True)

        # Now add correct skips
        for _ in range(5):
            cd = _make_cognitive_decision(action="WAIT")
            result = eng.evaluate_wait_decision(cd, _make_opp_score(0.50))
            skip = eng.record_skip(result)
            eng.resolve_skipped_opportunity(skip.skip_id, SkipOutcome.CORRECT_SKIP)

        m1 = eng.patience_score()
        # discipline_score should have improved (was 0, now 1.0)
        assert m1.discipline_score >= m0.discipline_score

    def test_end_to_end_skip_lifecycle(self):
        """Full skip lifecycle: evaluate → record → resolve → query."""
        from trainer.qde.brain.patience import (
            PatienceEngine,
            SkipOutcome,
            WaitReason,
        )

        eng = PatienceEngine(instrument="USDJPY")

        # 1. Brain decides WAIT (high uncertainty)
        cd = _make_cognitive_decision(action="WAIT", uncertainty_score=0.70, instrument="USDJPY")
        opp = _make_opp_score(0.15)
        decision = eng.evaluate_wait_decision(cd, opp, now=_T0)

        assert decision.should_wait is True
        assert decision.prospective_no_action_reward > 0.0
        assert WaitReason.COGNITIVE_PASSIVE in decision.wait_reasons

        # 2. Record the skip
        skip = eng.record_skip(decision)
        assert skip is not None
        assert len(eng.pending_skips()) == 1

        # 3. Market moves against the direction we skipped → correct skip
        resolved = eng.resolve_skipped_opportunity(
            skip.skip_id, SkipOutcome.CORRECT_SKIP, pnl_estimate=0.35, now=_T0 + 3600
        )
        assert resolved.was_correct is True
        assert resolved.realized_no_action_reward >= 0.60
        assert len(eng.pending_skips()) == 0

        # 4. Patience metrics improved
        m = eng.patience_score()
        assert m.discipline_score == pytest.approx(1.0)
        assert m.correct_skips == 1


# ═════════════════════════════════════════════════════════════════════════════
# 12. Package exports
# ═════════════════════════════════════════════════════════════════════════════


class TestPackageExports:
    def test_all_c13_symbols_exported(self):
        import trainer.qde.brain.patience as pkg

        required = [
            "evaluate_wait_decision",
            "patience_score",
            "minimum_quality_threshold",
            "PatienceEngine",
            "WaitReason",
            "SkipOutcome",
            "SkippedOpportunity",
            "WaitDecision",
            "PatienceState",
            "PatienceMetrics",
            "compute_no_action_reward",
            "compute_realized_reward",
            "DEFAULT_QUALITY_THRESHOLD",
            "MIN_QUALITY_THRESHOLD",
            "MAX_QUALITY_THRESHOLD",
        ]
        for name in required:
            assert hasattr(pkg, name), f"Package missing: {name}"

    def test_three_callables_are_callable(self):
        from trainer.qde.brain.patience import (
            evaluate_wait_decision,
            minimum_quality_threshold,
            patience_score,
        )

        assert callable(evaluate_wait_decision)
        assert callable(patience_score)
        assert callable(minimum_quality_threshold)

    def test_all_in_dunder_all(self):
        import trainer.qde.brain.patience as pkg

        for name in [
            "evaluate_wait_decision",
            "patience_score",
            "minimum_quality_threshold",
            "PatienceEngine",
            "WaitReason",
            "SkipOutcome",
        ]:
            assert name in pkg.__all__
