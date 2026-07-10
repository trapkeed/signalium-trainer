"""QDE Brain — setup queue tests.

Covers the opportunity-queue state machine (WATCHING/ARMED/CONFIRMING/
READY/EXECUTING/COOLDOWN): observing a setup over time, entry only after
sustained evidence accumulates, integration with hypotheses/attention/
working memory/narratives, and the fact that this vocabulary stays
distinct from `qde_observer_types.ObserverState`.
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_types import BeliefDirection, NarrativeTone
from trainer.qde.brain.conviction_engine import ConvictionProfile, ConvictionScore
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.hypothesis_lifecycle import HypothesisMaturity
from trainer.qde.brain.opportunity_tracker import Opportunity, OpportunityTracker
from trainer.qde.brain.setup_lifecycle import (
    DEFAULT_SETUP_THRESHOLDS,
    SetupEvidence,
    SetupState,
    SetupThresholds,
    is_valid_transition,
    next_state,
)
from trainer.qde.brain.setup_queue import SetupQueue
from trainer.qde.brain.uncertainty_engine import UncertaintyLevel, UncertaintyScore

SYM = "EURUSD"


def _conviction(conviction=0.8, confidence=0.8, profile=ConvictionProfile.ALIGNED_HIGH):
    return ConvictionScore(
        hypothesis_id="h1",
        instrument=SYM,
        confidence=confidence,
        conviction=conviction,
        profile=profile,
    )


def _uncertainty(composite=0.1, level=UncertaintyLevel.LOW):
    return UncertaintyScore(instrument=SYM, composite=composite, level=level)


# ── SetupState / transition map ─────────────────────────────────────────────


def test_setup_states_are_distinct_from_observer_state_vocabulary():
    from trainer.qde.qde_observer_types import ObserverState

    setup_values = {s.value for s in SetupState}
    observer_values = {s.value for s in ObserverState}
    # WATCHING/ARMED/CONFIRMING/READY/EXECUTING are queue-specific; ARMED and
    # COOLDOWN happen to be shared words with ObserverState, everything else
    # in each vocabulary is unique to that lifecycle.
    assert setup_values - observer_values == {
        "WATCHING",
        "CONFIRMING",
        "READY",
        "EXECUTING",
    }


def test_valid_transitions_cannot_skip_stages():
    assert is_valid_transition(SetupState.WATCHING, SetupState.ARMED)
    assert not is_valid_transition(SetupState.WATCHING, SetupState.READY)
    assert not is_valid_transition(SetupState.WATCHING, SetupState.EXECUTING)
    assert is_valid_transition(SetupState.READY, SetupState.EXECUTING)
    assert is_valid_transition(SetupState.EXECUTING, SetupState.COOLDOWN)
    assert is_valid_transition(SetupState.COOLDOWN, SetupState.WATCHING)


# ── next_state pure function ────────────────────────────────────────────────


def test_next_state_arms_on_sufficient_maturity_and_attention():
    evidence = SetupEvidence(
        maturity=HypothesisMaturity.BUILDING,
        conviction=_conviction(),
        uncertainty=_uncertainty(),
        attention_weight=0.5,
    )
    assert next_state(SetupState.WATCHING, evidence) == SetupState.ARMED


def test_next_state_stays_watching_without_attention():
    evidence = SetupEvidence(
        maturity=HypothesisMaturity.BUILDING,
        conviction=_conviction(),
        uncertainty=_uncertainty(),
        attention_weight=0.0,
    )
    assert next_state(SetupState.WATCHING, evidence) == SetupState.WATCHING


def test_next_state_requires_sustained_confirmation_ticks():
    evidence = SetupEvidence(
        maturity=HypothesisMaturity.HIGH_CONVICTION,
        conviction=_conviction(),
        uncertainty=_uncertainty(),
        attention_weight=0.5,
    )
    # One tick of CONFIRMING evidence is not enough on its own.
    assert (
        next_state(SetupState.CONFIRMING, evidence, confirmation_ticks=1) == SetupState.CONFIRMING
    )
    # Sustained evidence (>= min_confirmation_ticks) promotes to READY.
    assert next_state(SetupState.CONFIRMING, evidence, confirmation_ticks=2) == SetupState.READY


def test_next_state_high_uncertainty_never_advances_only_retreats():
    high_uncertainty = _uncertainty(composite=0.9, level=UncertaintyLevel.EXTREME)
    evidence = SetupEvidence(
        maturity=HypothesisMaturity.EXECUTABLE,
        conviction=_conviction(),
        uncertainty=high_uncertainty,
        attention_weight=0.9,
    )
    assert next_state(SetupState.CONFIRMING, evidence, confirmation_ticks=5) == SetupState.ARMED
    assert next_state(SetupState.ARMED, evidence) == SetupState.WATCHING


def test_next_state_invalidated_hypothesis_forces_cooldown():
    evidence = SetupEvidence(maturity=HypothesisMaturity.INVALIDATED)
    assert next_state(SetupState.CONFIRMING, evidence, confirmation_ticks=5) == SetupState.COOLDOWN
    assert next_state(SetupState.ARMED, evidence) == SetupState.COOLDOWN


def test_next_state_ready_expires_after_missed_entry_window():
    evidence = SetupEvidence(
        maturity=HypothesisMaturity.EXECUTABLE,
        conviction=_conviction(),
        uncertainty=_uncertainty(),
        dwell_seconds=1000.0,
    )
    thresholds = SetupThresholds(max_ready_dwell_seconds=900.0)
    assert next_state(SetupState.READY, evidence, thresholds=thresholds) == SetupState.COOLDOWN


def test_next_state_executing_orphan_safety_net():
    evidence = SetupEvidence(dwell_seconds=120.0)
    thresholds = SetupThresholds(max_executing_dwell_seconds=60.0)
    assert next_state(SetupState.EXECUTING, evidence, thresholds=thresholds) == SetupState.COOLDOWN


def test_next_state_cooldown_rearms_after_elapsed_and_not_terminal():
    evidence = SetupEvidence(maturity=HypothesisMaturity.OBSERVING, dwell_seconds=400.0)
    thresholds = SetupThresholds(cooldown_seconds=300.0)
    assert next_state(SetupState.COOLDOWN, evidence, thresholds=thresholds) == SetupState.WATCHING


def test_breakout_pending_narrative_shortens_required_confirmation():
    evidence = SetupEvidence(
        maturity=HypothesisMaturity.HIGH_CONVICTION,
        conviction=_conviction(),
        uncertainty=_uncertainty(),
        attention_weight=0.5,
        narrative_tone=NarrativeTone.BREAKOUT_PENDING,
    )
    thresholds = SetupThresholds(min_confirmation_ticks=2)
    # Only 1 tick needed when narrative corroborates the setup.
    assert (
        next_state(SetupState.CONFIRMING, evidence, confirmation_ticks=1, thresholds=thresholds)
        == SetupState.READY
    )


# ── OpportunityTracker ───────────────────────────────────────────────────────


def test_tracker_open_dedupes_by_hypothesis_id():
    tracker = OpportunityTracker()
    store: dict[str, Opportunity] = {}
    a = tracker.open(store, SYM, "hyp-1", BeliefDirection.BULLISH)
    b = tracker.open(store, SYM, "hyp-1", BeliefDirection.BULLISH)
    assert a.opportunity_id == b.opportunity_id
    assert len(store) == 1


def test_tracker_mark_executing_requires_ready_state():
    tracker = OpportunityTracker()
    store: dict[str, Opportunity] = {}
    opp = tracker.open(store, SYM, "hyp-1")
    assert tracker.mark_executing(store, opp.opportunity_id) is None  # still WATCHING
    opp.state = SetupState.READY
    result = tracker.mark_executing(store, opp.opportunity_id)
    assert result is not None
    assert result.state == SetupState.EXECUTING
    assert result.previous_state == SetupState.READY


def test_tracker_complete_moves_to_cooldown():
    tracker = OpportunityTracker()
    store: dict[str, Opportunity] = {}
    opp = tracker.open(store, SYM, "hyp-1")
    opp.state = SetupState.EXECUTING
    result = tracker.complete(store, opp.opportunity_id)
    assert result.state == SetupState.COOLDOWN


def test_tracker_prune_stale_drops_long_cooled_opportunities():
    tracker = OpportunityTracker()
    store: dict[str, Opportunity] = {}
    opp = tracker.open(store, SYM, "hyp-1")
    opp.state = SetupState.COOLDOWN
    opp.state_entered_at = time.time() - 4000.0
    removed = tracker.prune_stale(store, now=time.time())
    assert removed == 1
    assert store == {}


# ── SetupQueue integration ───────────────────────────────────────────────────


def test_setup_queue_progresses_a_reinforced_hypothesis_to_ready():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    queue = SetupQueue(brain, hypothesis_engine=hypotheses)

    hyp = hypotheses.create(SYM, "trend continuation", BeliefDirection.BULLISH, confidence=0.5)
    for _ in range(4):
        hypotheses.reinforce(hyp.hypothesis_id, "confirming evidence")

    state = brain.get_state(SYM)
    state.attention.weight = 0.5  # enough attention for the opportunity to arm

    now = time.time()
    seen_states = []
    for i in range(4):
        queue.observe(SYM, now=now + i * 250)
        opp = next(iter(state.opportunities.values()))
        seen_states.append(opp.state)

    # Must pass through every intermediate stage — never jump straight
    # from WATCHING to READY.
    assert SetupState.WATCHING not in seen_states[1:]
    assert seen_states[-1] in (SetupState.CONFIRMING, SetupState.READY)
    assert len(hypotheses.registry.active(SYM)) == 1


def test_setup_queue_does_not_arm_without_attention():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    queue = SetupQueue(brain, hypothesis_engine=hypotheses)

    hyp = hypotheses.create(SYM, "quiet setup", BeliefDirection.BULLISH, confidence=0.5)
    hypotheses.reinforce(hyp.hypothesis_id, "evidence")

    queue.observe(SYM)
    state = brain.get_state(SYM)
    opp = next(iter(state.opportunities.values()))
    assert opp.state == SetupState.WATCHING


def test_setup_queue_ready_opportunities_seam_and_execution_handoff():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    queue = SetupQueue(brain, hypothesis_engine=hypotheses)

    hyp = hypotheses.create(SYM, "trend continuation", BeliefDirection.BULLISH, confidence=0.5)
    for _ in range(4):
        hypotheses.reinforce(hyp.hypothesis_id, "confirming evidence")
    state = brain.get_state(SYM)
    state.attention.weight = 0.5

    now = time.time()
    opp = None
    for i in range(4):
        queue.observe(SYM, now=now + i * 250)
        candidates = queue.ready_opportunities(SYM)
        if candidates:
            opp = candidates[0]
            break

    assert opp is not None, "expected the setup to reach READY given sustained reinforcement"
    assert opp not in queue.ready_opportunities("GBPUSD")  # scoped per symbol

    executed = queue.mark_executing(SYM, opp.opportunity_id)
    assert executed.state == SetupState.EXECUTING
    assert queue.ready_opportunities(SYM) == []

    completed = queue.complete(SYM, opp.opportunity_id)
    assert completed.state == SetupState.COOLDOWN


def test_setup_queue_invalidated_hypothesis_cools_down_its_opportunity():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    queue = SetupQueue(brain, hypothesis_engine=hypotheses)

    hyp = hypotheses.create(SYM, "false start", BeliefDirection.BEARISH, confidence=0.4)
    state = brain.get_state(SYM)
    state.attention.weight = 0.5
    queue.observe(SYM)

    hypotheses.invalidate(hyp.hypothesis_id, "structure broke down")
    queue.observe(SYM)

    opp = next(iter(state.opportunities.values()))
    assert opp.state == SetupState.COOLDOWN


def test_setup_queue_logs_transitions_to_working_memory():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    queue = SetupQueue(brain, hypothesis_engine=hypotheses)

    hyp = hypotheses.create(SYM, "trend continuation", BeliefDirection.BULLISH, confidence=0.5)
    for _ in range(4):
        hypotheses.reinforce(hyp.hypothesis_id, "confirming evidence")
    state = brain.get_state(SYM)
    state.attention.weight = 0.5

    queue.observe(SYM)
    kinds = [item.kind for item in state.working_memory]
    assert "OPPORTUNITY_UPDATE" in kinds


def test_setup_queue_uses_narrative_context():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    queue = SetupQueue(brain, hypothesis_engine=hypotheses)

    hyp = hypotheses.create(SYM, "trend continuation", BeliefDirection.BULLISH, confidence=0.5)
    for _ in range(4):
        hypotheses.reinforce(hyp.hypothesis_id, "confirming evidence")

    from trainer.qde.brain.brain_models import MarketNarrative

    state = brain.get_state(SYM)
    state.attention.weight = 0.5
    state.narrative = MarketNarrative(
        instrument=SYM,
        summary="EURUSD coiling ahead of a breakout",
        tone=NarrativeTone.BREAKOUT_PENDING,
    )

    queue.observe(SYM)
    opp = next(iter(state.opportunities.values()))
    assert opp.narrative_snapshot == "EURUSD coiling ahead of a breakout"


def test_brain_state_round_trips_opportunities_through_persistence():
    from trainer.qde.brain.brain import _state_from_dict

    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    queue = SetupQueue(brain, hypothesis_engine=hypotheses)

    hyp = hypotheses.create(SYM, "trend continuation", BeliefDirection.BULLISH, confidence=0.5)
    state = brain.get_state(SYM)
    state.attention.weight = 0.5
    queue.observe(SYM)

    raw = state.to_dict()
    assert "opportunities" in raw
    restored = _state_from_dict(raw)
    assert len(restored.opportunities) == len(state.opportunities)
    for opp_id, opp in state.opportunities.items():
        restored_opp = restored.opportunities[opp_id]
        assert restored_opp.state == opp.state
        assert restored_opp.hypothesis_id == opp.hypothesis_id


def test_default_setup_thresholds_are_reasonable():
    assert DEFAULT_SETUP_THRESHOLDS.min_confirmation_ticks >= 1
    assert 0.0 < DEFAULT_SETUP_THRESHOLDS.ready_min_conviction <= 1.0
