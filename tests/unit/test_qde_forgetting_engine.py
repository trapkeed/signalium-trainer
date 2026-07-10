"""QDE Brain — forgetting / memory decay tests.

Covers `memory_decay_engine` (pure recency/relevance/context/regime decay
math) and `forgetting_engine.ForgettingEngine` (the orchestration seam that
applies that math to a live `BrainState` without duplicating the decay
mechanisms `working_memory`/`hypotheses` already own).
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_models import (
    BrainState,
    LongTermMemoryEntry,
    MarketNarrative,
    SymbolPersonality,
    WorkingMemoryItem,
)
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.forgetting_engine import ForgettingEngine
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.memory_decay_engine import (
    REGIME_NEUTRAL_QUALITY,
    context_decay,
    recency_weight,
    regime_decay,
    relevance_weight,
)

SYM = "EURUSD"


# ── memory_decay_engine: recency_weight ─────────────────────────────────────


def test_recency_weight_is_full_at_zero_age():
    assert recency_weight(0.0) == 1.0


def test_recency_weight_halves_at_half_life():
    half_life = 3600.0
    assert abs(recency_weight(half_life, half_life) - 0.5) < 1e-9


def test_recency_weight_decreases_monotonically_with_age():
    half_life = 3600.0
    w1 = recency_weight(100.0, half_life)
    w2 = recency_weight(10000.0, half_life)
    assert w1 > w2 > 0.0


# ── memory_decay_engine: relevance_weight ───────────────────────────────────


def test_relevance_weight_favors_recent_over_old():
    recent = relevance_weight(0.8, age_seconds=60.0, context_match=True)
    old = relevance_weight(0.8, age_seconds=10_000_000.0, context_match=True)
    assert recent > old


def test_relevance_weight_penalizes_context_mismatch():
    matched = relevance_weight(0.8, age_seconds=0.0, context_match=True)
    mismatched = relevance_weight(0.8, age_seconds=0.0, context_match=False, mismatch_penalty=0.5)
    assert mismatched < matched
    assert mismatched > 0.0  # penalized, not discarded


# ── memory_decay_engine: context_decay ──────────────────────────────────────


def test_context_decay_no_change_while_context_valid():
    assert (
        context_decay(0.9, elapsed_since_context_changed_seconds=5000.0, context_still_valid=True)
        == 0.9
    )


def test_context_decay_fades_confidence_once_context_invalid():
    faded = context_decay(
        0.9, elapsed_since_context_changed_seconds=1800.0, context_still_valid=False
    )
    assert 0.0 < faded < 0.9


def test_context_decay_approaches_zero_for_large_elapsed():
    faded = context_decay(
        0.9, elapsed_since_context_changed_seconds=1_000_000.0, context_still_valid=False
    )
    assert faded < 0.001


# ── memory_decay_engine: regime_decay ───────────────────────────────────────


def test_regime_decay_leaves_current_regime_untouched():
    quality = {"TRENDING": 0.8, "RANGE": -0.3}
    last_seen = {"TRENDING": time.time(), "RANGE": time.time() - 100_000.0}
    now = time.time()
    decayed = regime_decay(quality, last_seen, current_regime="TRENDING", now=now)
    assert decayed["TRENDING"] == 0.8


def test_regime_decay_pulls_obsolete_regime_toward_neutral():
    now = time.time()
    quality = {"TRENDING": 0.8, "RANGE": -0.6}
    last_seen = {"TRENDING": now, "RANGE": now - 100_000.0}
    decayed = regime_decay(quality, last_seen, current_regime="TRENDING", now=now)
    assert decayed["RANGE"] < REGIME_NEUTRAL_QUALITY  # still negative, but...
    assert abs(decayed["RANGE"]) < abs(-0.6)  # ...moved toward neutral


def test_regime_decay_updates_last_seen_for_current_regime():
    now = time.time()
    last_seen: dict[str, float] = {}
    regime_decay({"TRENDING": 0.5}, last_seen, current_regime="TRENDING", now=now)
    assert last_seen["TRENDING"] == now


def test_regime_decay_leaves_never_before_seen_regime_alone():
    now = time.time()
    decayed = regime_decay({"RANGE": 0.4}, {}, current_regime="TRENDING", now=now)
    assert decayed["RANGE"] == 0.4  # no last_seen entry yet -> not assumed stale


# ── ForgettingEngine: working memory + hypothesis expiry delegation ────────


def test_forget_delegates_working_memory_pruning():
    state = BrainState.new(SYM)
    now = time.time()
    state.working_memory.append(
        WorkingMemoryItem(
            instrument=SYM,
            kind="tick",
            content={},
            salience=0.01,
            created_at=now - 100_000.0,
            last_reinforced_at=now - 100_000.0,
        )
    )
    engine = ForgettingEngine()
    report = engine.forget(state, now=now)
    assert report.working_memory_pruned == 1
    assert state.working_memory == []


def test_forget_delegates_hypothesis_expiry():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp = hypotheses.create(SYM, "stale setup", BeliefDirection.BULLISH, confidence=0.4)
    state = brain.get_state(SYM)
    hyp.last_updated_at = time.time() - hyp.max_age_seconds - 1.0

    engine = ForgettingEngine()
    report = engine.forget(state, now=time.time())
    assert report.hypotheses_expired == 1
    assert state.hypotheses[hyp.hypothesis_id].status.value == "EXPIRED"


# ── ForgettingEngine: context decay of hypotheses ───────────────────────────


def test_forget_context_decays_hypothesis_when_regime_changes():
    brain = QDEBrain()
    state = brain.get_state(SYM)
    state.narrative = MarketNarrative(instrument=SYM, market_regime="TRENDING")
    hypotheses = HypothesisEngine(brain)
    hyp = hypotheses.create(SYM, "trend continuation", BeliefDirection.BULLISH, confidence=0.8)
    assert hyp.formed_regime == "TRENDING"

    # Market regime moves on; some time passes without the hypothesis
    # being touched again.
    state.narrative.market_regime = "RANGE"
    now = time.time() + 900.0

    engine = ForgettingEngine()
    report = engine.forget(state, now=now)
    assert report.hypotheses_context_decayed == 1
    assert state.hypotheses[hyp.hypothesis_id].confidence < 0.8


def test_forget_invalidates_hypothesis_once_context_decay_bottoms_out():
    brain = QDEBrain()
    state = brain.get_state(SYM)
    state.narrative = MarketNarrative(instrument=SYM, market_regime="TRENDING")
    hypotheses = HypothesisEngine(brain)
    hyp = hypotheses.create(SYM, "trend continuation", BeliefDirection.BULLISH, confidence=0.3)

    state.narrative.market_regime = "RANGE"
    # Long enough to fully context-decay, but well under `max_age_seconds`
    # so ordinary staleness expiry (`HypothesisTracker.age`) doesn't fire
    # first and mask what we're actually testing here.
    now = time.time() + 3600.0
    assert now - hyp.last_updated_at < hyp.max_age_seconds

    engine = ForgettingEngine()
    engine.forget(state, now=now)
    assert state.hypotheses[hyp.hypothesis_id].is_terminal()
    assert state.hypotheses[hyp.hypothesis_id].status.value == "INVALIDATED"


def test_forget_does_not_context_decay_hypothesis_without_formed_regime():
    brain = QDEBrain()
    state = brain.get_state(SYM)
    hypotheses = HypothesisEngine(brain)
    hyp = hypotheses.create(SYM, "no narrative yet", BeliefDirection.BULLISH, confidence=0.5)
    assert hyp.formed_regime == ""

    engine = ForgettingEngine()
    report = engine.forget(state, now=time.time() + 900.0)
    assert report.hypotheses_context_decayed == 0
    assert state.hypotheses[hyp.hypothesis_id].confidence == 0.5


# ── ForgettingEngine: regime decay wiring on personality ───────────────────


def test_forget_decays_obsolete_regime_quality_on_personality():
    brain = QDEBrain()
    state = brain.get_state(SYM)
    now = time.time()
    state.narrative = MarketNarrative(instrument=SYM, market_regime="TRENDING")
    state.personality = SymbolPersonality(
        instrument=SYM,
        regime_outcome_quality={"TRENDING": 0.5, "RANGE": -0.7},
        regime_last_seen={"TRENDING": now - 1.0, "RANGE": now - 200_000.0},
    )

    engine = ForgettingEngine()
    report = engine.forget(state, now=now)
    assert "RANGE" in report.regimes_decayed
    assert abs(state.personality.regime_outcome_quality["RANGE"]) < 0.7
    assert state.personality.regime_outcome_quality["TRENDING"] == 0.5
    assert state.personality.regime_last_seen["TRENDING"] == now


def test_forget_is_noop_for_regime_decay_without_narrative():
    state = BrainState.new(SYM)
    state.personality.regime_outcome_quality = {"RANGE": -0.4}
    engine = ForgettingEngine()
    report = engine.forget(state, now=time.time())
    assert report.regimes_decayed == []
    assert state.personality.regime_outcome_quality["RANGE"] == -0.4


# ── ForgettingEngine: long-term memory relevance ranking ────────────────────


def test_relevant_long_term_memories_ranks_recent_above_old_of_equal_salience():
    now = time.time()
    recent = LongTermMemoryEntry(instrument=SYM, salience=0.6, created_at=now - 60.0)
    old = LongTermMemoryEntry(instrument=SYM, salience=0.6, created_at=now - 10_000_000.0)

    engine = ForgettingEngine()
    ranked = engine.relevant_long_term_memories([old, recent], now=now)
    assert ranked[0][0] is recent
    assert ranked[0][1] > ranked[1][1]


def test_relevant_long_term_memories_context_match_boosts_rank():
    now = time.time()
    matching = LongTermMemoryEntry(
        instrument=SYM, salience=0.5, tags=["TRENDING"], created_at=now - 60.0
    )
    mismatched = LongTermMemoryEntry(
        instrument=SYM, salience=0.5, tags=["RANGE"], created_at=now - 60.0
    )

    engine = ForgettingEngine()
    ranked = engine.relevant_long_term_memories(
        [mismatched, matching], current_regime="TRENDING", now=now
    )
    assert ranked[0][0] is matching


def test_relevant_long_term_memories_respects_limit():
    now = time.time()
    entries = [LongTermMemoryEntry(instrument=SYM, salience=0.5, created_at=now) for _ in range(5)]
    engine = ForgettingEngine()
    ranked = engine.relevant_long_term_memories(entries, n=2, now=now)
    assert len(ranked) == 2


# ── ForgettingEngine: recency_weight_of convenience wrapper ────────────────


def test_recency_weight_of_matches_engine_math():
    now = time.time()
    engine = ForgettingEngine()
    assert engine.recency_weight_of(now, now=now) == 1.0
    assert 0.0 < engine.recency_weight_of(now - 3600.0, now=now) < 1.0
