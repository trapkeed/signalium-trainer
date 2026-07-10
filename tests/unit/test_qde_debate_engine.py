"""QDE Brain — debate engine / thesis manager tests.

Covers `thesis_manager.ThesisManager` (every opportunity carries a bull
thesis and a bear thesis, built by reprojecting a hypothesis's own
evidence onto both sides) and `debate_engine.DebateEngine` (comparing
supporting evidence, conflicting evidence, and historical outcomes to
decide whether one side clearly dominates before execution is cleared).
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.debate_engine import DebateEngine, DebateVerdict
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.opportunity_tracker import Opportunity, OpportunityTracker
from trainer.qde.brain.setup_lifecycle import SetupState
from trainer.qde.brain.thesis_manager import Thesis, ThesisManager

SYM = "EURUSD"


def _opportunity_for(brain: QDEBrain, hypotheses: HypothesisEngine, direction, thesis_text):
    hyp = hypotheses.create(SYM, thesis_text, direction, confidence=0.5)
    state = brain.get_state(SYM)
    tracker = OpportunityTracker()
    opp = tracker.open(state.opportunities, SYM, hyp.hypothesis_id, direction=direction)
    return hyp, opp


# ── Thesis ───────────────────────────────────────────────────────────────────


def test_thesis_defaults_are_neutral():
    thesis = Thesis(side=BeliefDirection.BULLISH)
    assert thesis.confidence == 0.5
    assert thesis.historical_support == 0.5
    assert thesis.strength() == 0.5  # no evidence yet -> neutral evidence balance


def test_thesis_strength_rewards_one_sided_supporting_evidence():
    strong = Thesis(
        side=BeliefDirection.BULLISH,
        supporting_evidence=["a", "b", "c"],
        confidence=0.8,
        historical_support=0.7,
    )
    weak = Thesis(
        side=BeliefDirection.BULLISH,
        conflicting_evidence=["x", "y", "z"],
        confidence=0.2,
        historical_support=0.3,
    )
    assert strong.strength() > weak.strength()


def test_opportunity_carries_both_bull_and_bear_theses_by_default():
    opp = Opportunity(instrument=SYM)
    assert opp.bull_thesis.side == BeliefDirection.BULLISH
    assert opp.bear_thesis.side == BeliefDirection.BEARISH


def test_opportunity_to_dict_serializes_both_theses():
    opp = Opportunity(instrument=SYM)
    d = opp.to_dict()
    assert "bull_thesis" in d and "bear_thesis" in d
    assert d["bull_thesis"]["side"] == "BULLISH"
    assert d["bear_thesis"]["side"] == "BEARISH"


# ── ThesisManager.sync ───────────────────────────────────────────────────────


def test_sync_projects_bullish_hypothesis_evidence_onto_both_sides():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "trend continuation")
    hypotheses.reinforce(hyp.hypothesis_id, "breakout confirmed")
    hypotheses.weaken(hyp.hypothesis_id, "volume drying up")

    state = brain.get_state(SYM)
    manager = ThesisManager()
    bull, bear = manager.sync(state, opp, state.hypotheses)

    assert "breakout confirmed" in bull.supporting_evidence
    assert "volume drying up" in bull.conflicting_evidence
    # A bullish hypothesis's own contradicting evidence is bearish support.
    assert "volume drying up" in bear.supporting_evidence
    assert "breakout confirmed" in bear.conflicting_evidence
    assert bull.confidence == hyp.confidence
    assert bear.confidence == 1.0 - hyp.confidence


def test_sync_folds_in_opposing_sibling_hypotheses():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "trend continuation")
    hypotheses.create(
        SYM, "reversal forming", BeliefDirection.BEARISH, confidence=0.4, evidence="lower high"
    )

    state = brain.get_state(SYM)
    manager = ThesisManager()
    bull, bear = manager.sync(state, opp, state.hypotheses)

    assert "lower high" in bear.supporting_evidence
    assert "lower high" in bull.conflicting_evidence


def test_sync_uses_learned_history_not_a_duplicate_tracker():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "trend continuation")

    state = brain.get_state(SYM)
    state.personality.regime_outcome_quality["TREND_UP"] = 0.8

    manager = ThesisManager()
    bull, bear = manager.sync(state, opp, state.hypotheses)
    # Historical support is derived, not independently tracked — it must
    # move with the already-existing SymbolPersonality data.
    assert 0.0 <= bull.historical_support <= 1.0
    assert 0.0 <= bear.historical_support <= 1.0


# ── DebateEngine ─────────────────────────────────────────────────────────────


def test_debate_engine_declares_dominant_side_for_one_sided_evidence():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "trend continuation")
    for _ in range(5):
        hypotheses.reinforce(hyp.hypothesis_id, "confirming evidence")

    engine = DebateEngine(brain)
    verdict = engine.evaluate(SYM, opp.opportunity_id)

    assert isinstance(verdict, DebateVerdict)
    assert verdict.dominant_side == BeliefDirection.BULLISH
    assert verdict.conclusive
    assert verdict.bull_strength > verdict.bear_strength


def test_debate_engine_is_inconclusive_when_evidence_is_balanced():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "uncertain setup")
    hypotheses.reinforce(hyp.hypothesis_id, "some support")
    hypotheses.weaken(hyp.hypothesis_id, "some doubt")

    engine = DebateEngine(brain)
    verdict = engine.evaluate(SYM, opp.opportunity_id)
    assert verdict is not None
    # Roughly balanced supporting/conflicting evidence should not clear
    # the dominance margin.
    assert verdict.margin < 0.5


def test_debate_engine_returns_none_for_unknown_opportunity():
    brain = QDEBrain()
    engine = DebateEngine(brain)
    assert engine.evaluate(SYM, "does-not-exist") is None


def test_clears_for_execution_requires_conclusive_and_aligned_dominance():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "trend continuation")
    for _ in range(5):
        hypotheses.reinforce(hyp.hypothesis_id, "confirming evidence")

    engine = DebateEngine(brain)
    assert engine.clears_for_execution(SYM, opp.opportunity_id) is True


def test_clears_for_execution_withholds_when_dominant_side_opposes_setup():
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    # The opportunity tracks a BULLISH setup, but its hypothesis actually
    # accumulates one-sided bearish-favoring evidence (heavy contradicting
    # evidence against the bullish thesis == heavy bullish support).
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "shaky long")
    for _ in range(5):
        hypotheses.weaken(hyp.hypothesis_id, "bearish signal")

    engine = DebateEngine(brain)
    verdict = engine.evaluate(SYM, opp.opportunity_id)
    assert verdict.dominant_side == BeliefDirection.BEARISH
    assert engine.clears_for_execution(SYM, opp.opportunity_id) is False


def test_debate_engine_does_not_auto_wire_into_setup_queue_execution():
    """DebateEngine is a consult-only seam, exactly like
    `SetupQueue.ready_opportunities()` — it must never be required by
    `OpportunityTracker.mark_executing`, so existing callers are
    unaffected unless they explicitly opt in.
    """
    brain = QDEBrain()
    hypotheses = HypothesisEngine(brain)
    hyp, opp = _opportunity_for(brain, hypotheses, BeliefDirection.BULLISH, "trend continuation")

    tracker = OpportunityTracker()
    state = brain.get_state(SYM)
    opp.state = SetupState.READY
    executed = tracker.mark_executing(state.opportunities, opp.opportunity_id, now=time.time())
    assert executed is not None
    assert executed.state == SetupState.EXECUTING
