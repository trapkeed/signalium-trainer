"""QDE Brain — hypothesis engine / registry / lifecycle tests.

Covers the five required capabilities (create, reinforce, weaken,
invalidate, retire), the OBSERVING → BUILDING → HIGH_CONVICTION →
EXECUTABLE maturity progression, and the "trade execution should emerge
from mature hypotheses" integration point (`executable_hypotheses` /
`mature_direction`).
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_types import BeliefDirection, HypothesisStatus
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.hypothesis_lifecycle import (
    HypothesisMaturity,
    MaturityThresholds,
    compute_maturity,
    hypothesis_age_seconds,
    is_terminal_maturity,
    maturity_rank,
)

SYM = "EURUSD"


def _engine() -> HypothesisEngine:
    return HypothesisEngine(QDEBrain())


# ── Create / reinforce / weaken / invalidate / retire ───────────────────────


def test_create_forms_a_new_hypothesis_with_all_required_fields():
    engine = _engine()
    hyp = engine.create(SYM, "EURUSD breaking resistance", BeliefDirection.BULLISH, confidence=0.4)

    assert hyp.instrument == SYM
    assert hyp.thesis == "EURUSD breaking resistance"
    assert hyp.confidence == 0.4
    assert hyp.supporting_evidence == []
    assert hyp.contradicting_evidence == []
    assert hyp.status == HypothesisStatus.FORMING
    assert hypothesis_age_seconds(hyp) >= 0.0


def test_create_deduplicates_equivalent_active_thesis():
    engine = _engine()
    hyp1 = engine.create(SYM, "same thesis", BeliefDirection.BULLISH, confidence=0.3)
    hyp2 = engine.create(SYM, "same thesis", BeliefDirection.BULLISH, confidence=0.3)

    assert hyp1.hypothesis_id == hyp2.hypothesis_id
    assert hyp2.reinforcement_count >= 1


def test_reinforce_increases_confidence_and_records_evidence():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.3)
    updated = engine.reinforce(hyp.hypothesis_id, "breakout confirmed")

    assert updated is not None
    assert updated.confidence > 0.3
    assert "breakout confirmed" in updated.supporting_evidence


def test_weaken_decreases_confidence_and_records_evidence():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.5)
    updated = engine.weaken(hyp.hypothesis_id, "failed retest")

    assert updated is not None
    assert updated.confidence < 0.5
    assert "failed retest" in updated.contradicting_evidence


def test_invalidate_sets_terminal_status_with_reason():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.5)
    updated = engine.invalidate(hyp.hypothesis_id, "thesis disproven by reversal")

    assert updated is not None
    assert updated.status == HypothesisStatus.INVALIDATED
    assert updated.invalidation_reason == "thesis disproven by reversal"
    assert compute_maturity(updated) == HypothesisMaturity.INVALIDATED


def test_retire_closes_out_without_disproving():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.5)
    retired = engine.retire(hyp.hypothesis_id, "symbol no longer enabled")

    assert retired is not None
    assert retired.status == HypothesisStatus.EXPIRED
    assert "RETIRED" in retired.invalidation_reason
    assert compute_maturity(retired) == HypothesisMaturity.EXPIRED


def test_retire_is_a_no_op_on_already_terminal_hypothesis():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.5)
    engine.invalidate(hyp.hypothesis_id, "disproven")
    result = engine.retire(hyp.hypothesis_id, "attempted double close-out")

    assert result is not None
    assert result.status == HypothesisStatus.INVALIDATED  # unchanged, still terminal


def test_operations_on_unknown_hypothesis_id_return_none():
    engine = _engine()
    assert engine.reinforce("does-not-exist", "x") is None
    assert engine.weaken("does-not-exist", "x") is None
    assert engine.invalidate("does-not-exist", "x") is None
    assert engine.retire("does-not-exist", "x") is None
    assert engine.maturity_of("does-not-exist") is None


# ── Maturity lifecycle ───────────────────────────────────────────────────────


def test_low_confidence_hypothesis_is_observing():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.1)
    assert engine.maturity_of(hyp.hypothesis_id) == HypothesisMaturity.OBSERVING


def test_currently_weakening_hypothesis_cannot_be_high_conviction_or_executable():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.75)
    engine.reinforce(hyp.hypothesis_id, "evidence 1")
    engine.reinforce(hyp.hypothesis_id, "evidence 2")
    engine.weaken(hyp.hypothesis_id, "contradiction appears")

    maturity = engine.maturity_of(hyp.hypothesis_id)
    assert maturity in (HypothesisMaturity.BUILDING, HypothesisMaturity.OBSERVING)


def test_sustained_high_confidence_and_age_reaches_executable():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.5)
    engine.reinforce(hyp.hypothesis_id, "e1")
    engine.reinforce(hyp.hypothesis_id, "e2")
    engine.reinforce(hyp.hypothesis_id, "e3")

    assert hyp.confidence >= 0.8
    # Too young yet, even with high confidence — maturity requires dwell time.
    assert engine.executable_hypotheses(SYM) == []

    hyp.created_at = time.time() - 1000
    views = engine.executable_hypotheses(SYM)
    assert len(views) == 1
    assert views[0].maturity == HypothesisMaturity.EXECUTABLE


def test_confirmed_underlying_hypothesis_is_always_executable():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.3)
    engine._tracker.confirm(engine._brain.get_state(SYM).hypotheses, hyp.hypothesis_id)

    assert compute_maturity(hyp) == HypothesisMaturity.EXECUTABLE


def test_maturity_rank_orders_progression_correctly():
    assert maturity_rank(HypothesisMaturity.OBSERVING) < maturity_rank(HypothesisMaturity.BUILDING)
    assert maturity_rank(HypothesisMaturity.BUILDING) < maturity_rank(
        HypothesisMaturity.HIGH_CONVICTION
    )
    assert maturity_rank(HypothesisMaturity.HIGH_CONVICTION) < maturity_rank(
        HypothesisMaturity.EXECUTABLE
    )


def test_is_terminal_maturity():
    assert is_terminal_maturity(HypothesisMaturity.INVALIDATED)
    assert is_terminal_maturity(HypothesisMaturity.EXPIRED)
    assert not is_terminal_maturity(HypothesisMaturity.EXECUTABLE)
    assert not is_terminal_maturity(HypothesisMaturity.OBSERVING)


def test_custom_thresholds_are_respected():
    engine = HypothesisEngine(
        QDEBrain(),
        thresholds=MaturityThresholds(
            executable_confidence=0.5,
            executable_min_reinforcements=1,
            executable_min_age_seconds=0.0,
        ),
    )
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.4)
    engine.reinforce(hyp.hypothesis_id, "one strong signal")

    assert engine.maturity_of(hyp.hypothesis_id) == HypothesisMaturity.EXECUTABLE


# ── Execution integration ────────────────────────────────────────────────────


def test_mature_direction_is_none_without_an_executable_hypothesis():
    engine = _engine()
    engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.3)
    assert engine.mature_direction(SYM) is None


def test_mature_direction_reflects_most_confident_executable_hypothesis():
    engine = HypothesisEngine(
        QDEBrain(),
        thresholds=MaturityThresholds(
            executable_confidence=0.5,
            executable_min_reinforcements=1,
            executable_min_age_seconds=0.0,
        ),
    )
    hyp = engine.create(SYM, "bullish thesis", BeliefDirection.BULLISH, confidence=0.6)
    engine.reinforce(hyp.hypothesis_id, "confirmation")

    assert engine.mature_direction(SYM) == BeliefDirection.BULLISH


# ── Registry cross-symbol queries ────────────────────────────────────────────


def test_registry_all_and_for_symbol_and_by_maturity():
    engine = _engine()
    engine.create(SYM, "eurusd thesis", BeliefDirection.BULLISH, confidence=0.2)
    engine.create("GBPUSD", "gbpusd thesis", BeliefDirection.BEARISH, confidence=0.2)

    all_views = engine.registry.all()
    assert {v.instrument for v in all_views} == {SYM, "GBPUSD"}

    eur_views = engine.registry.for_symbol(SYM)
    assert len(eur_views) == 1
    assert eur_views[0].instrument == SYM

    observing = engine.registry.by_maturity(HypothesisMaturity.OBSERVING)
    assert len(observing) == 2


def test_registry_get_locates_hypothesis_by_id():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.3)
    view = engine.registry.get(hyp.hypothesis_id)

    assert view is not None
    assert view.hypothesis.hypothesis_id == hyp.hypothesis_id
    assert view.instrument == SYM


def test_registry_active_excludes_terminal_hypotheses():
    engine = _engine()
    hyp1 = engine.create(SYM, "thesis 1", BeliefDirection.BULLISH, confidence=0.3)
    engine.create(SYM, "thesis 2", BeliefDirection.BEARISH, confidence=0.3)
    engine.invalidate(hyp1.hypothesis_id, "disproven")

    active = engine.registry.active(SYM)
    assert len(active) == 1
    assert active[0].hypothesis.thesis == "thesis 2"


# ── Long-term memory / experience integration ────────────────────────────────


def test_reaching_executable_records_an_experience():
    brain = QDEBrain()
    engine = HypothesisEngine(
        brain,
        thresholds=MaturityThresholds(
            executable_confidence=0.5,
            executable_min_reinforcements=1,
            executable_min_age_seconds=0.0,
        ),
    )
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.6)
    engine.reinforce(hyp.hypothesis_id, "confirmation")

    state = brain.get_state(SYM)
    assert state.personality.sample_count >= 1


def test_invalidating_records_a_negative_experience():
    brain = QDEBrain()
    engine = HypothesisEngine(brain)
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.5)
    engine.invalidate(hyp.hypothesis_id, "disproven")

    state = brain.get_state(SYM)
    assert state.personality.sample_count >= 1


def test_age_all_expires_stale_hypotheses_across_symbols():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.3)
    hyp.last_updated_at = time.time() - hyp.max_age_seconds - 10

    expired = engine.age_all()

    assert expired == 1
    assert compute_maturity(hyp) == HypothesisMaturity.EXPIRED


def test_working_memory_records_hypothesis_updates():
    engine = _engine()
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.3)
    engine.reinforce(hyp.hypothesis_id, "evidence")

    state = engine._brain.get_state(SYM)
    kinds = [item.kind for item in state.working_memory]
    assert "HYPOTHESIS_UPDATE" in kinds
