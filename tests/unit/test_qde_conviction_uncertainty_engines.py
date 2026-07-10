"""QDE Brain — conviction and uncertainty engine tests.

Covers the confidence/conviction/uncertainty distinction: high-confidence
low-conviction, low-confidence high-conviction, aligned quadrants, and
the "QDE does not understand the market" NO_TRADE-preference gate — plus
their integration points in `HypothesisEngine` and `qde_inference`.
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection, HypothesisStatus
from trainer.qde.brain.conviction_engine import (
    ConvictionProfile,
    classify_profile,
    compute_conviction,
    is_trade_worthy,
)
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.uncertainty_engine import (
    UncertaintyLevel,
    compute_uncertainty,
    does_not_understand_market,
    prefers_no_trade,
)

SYM = "EURUSD"


def _hyp(**overrides) -> Hypothesis:
    defaults = dict(
        instrument=SYM,
        thesis="thesis",
        direction=BeliefDirection.BULLISH,
        confidence=0.5,
        status=HypothesisStatus.ACTIVE,
        created_at=time.time() - 1000.0,
        last_updated_at=time.time(),
        reinforcement_count=0,
        supporting_evidence=[],
        contradicting_evidence=[],
    )
    defaults.update(overrides)
    return Hypothesis(**defaults)


# ── classify_profile ─────────────────────────────────────────────────────────


def test_classify_profile_aligned_high():
    assert classify_profile(0.8, 0.8) == ConvictionProfile.ALIGNED_HIGH


def test_classify_profile_aligned_low():
    assert classify_profile(0.2, 0.2) == ConvictionProfile.ALIGNED_LOW


def test_classify_profile_high_confidence_low_conviction():
    assert classify_profile(0.8, 0.2) == ConvictionProfile.HIGH_CONFIDENCE_LOW_CONVICTION


def test_classify_profile_low_confidence_high_conviction():
    assert classify_profile(0.2, 0.8) == ConvictionProfile.LOW_CONFIDENCE_HIGH_CONVICTION


def test_classify_profile_mixed_for_moderate_values():
    assert classify_profile(0.5, 0.5) == ConvictionProfile.MIXED


# ── compute_conviction ───────────────────────────────────────────────────────


def test_fresh_thin_hypothesis_has_low_conviction_even_with_high_confidence():
    # High confidence, but just formed: no reinforcement, no evidence, no age.
    hyp = _hyp(confidence=0.9, reinforcement_count=0, created_at=time.time())
    score = compute_conviction(hyp)

    assert score.confidence == 0.9
    assert score.conviction < 0.4
    assert score.profile == ConvictionProfile.HIGH_CONFIDENCE_LOW_CONVICTION


def test_heavily_reinforced_hypothesis_has_high_conviction_even_with_moderate_confidence():
    # Lower confidence, but extensively reinforced, one-sided evidence, aged.
    hyp = _hyp(
        confidence=0.35,
        reinforcement_count=6,
        supporting_evidence=["e1", "e2", "e3", "e4"],
        contradicting_evidence=[],
        created_at=time.time() - 600.0,
    )
    score = compute_conviction(hyp)

    assert score.conviction >= 0.6
    assert score.profile == ConvictionProfile.LOW_CONFIDENCE_HIGH_CONVICTION


def test_conviction_never_re_derives_confidence():
    hyp = _hyp(confidence=0.42)
    score = compute_conviction(hyp)
    assert score.confidence == 0.42  # untouched, sourced straight from the hypothesis


def test_is_trade_worthy_true_for_high_conviction_despite_low_confidence():
    hyp = _hyp(
        confidence=0.42,
        reinforcement_count=5,
        supporting_evidence=["a", "b", "c"],
        created_at=time.time() - 600.0,
    )
    score = compute_conviction(hyp)
    assert is_trade_worthy(score, min_conviction=0.55, min_confidence=0.4)


def test_is_trade_worthy_false_below_confidence_floor_regardless_of_conviction():
    hyp = _hyp(
        confidence=0.1,
        reinforcement_count=10,
        supporting_evidence=["a", "b", "c", "d"],
        created_at=time.time() - 1000.0,
    )
    score = compute_conviction(hyp)
    assert not is_trade_worthy(score, min_conviction=0.5, min_confidence=0.4)


# ── compute_uncertainty ──────────────────────────────────────────────────────


def test_calm_market_yields_low_uncertainty():
    score = compute_uncertainty(
        SYM,
        news_risk=0.05,
        volatility_state="NORMAL",
        volatility_quality=0.8,
        liquidity_state="NORMAL",
        liquidity_quality=0.8,
        trend_valid=True,
        microstructure_quality=0.8,
        active_hypotheses=[],
    )
    assert score.level == UncertaintyLevel.LOW
    assert not prefers_no_trade(score)
    assert not does_not_understand_market(score)


def test_high_news_risk_and_bad_structure_yields_high_uncertainty_and_no_trade_preference():
    score = compute_uncertainty(
        SYM,
        news_risk=0.9,
        volatility_state="EXTREME",
        volatility_quality=0.1,
        liquidity_state="THIN",
        liquidity_quality=0.1,
        trend_valid=False,
        microstructure_quality=0.1,
        active_hypotheses=[],
    )
    assert score.level == UncertaintyLevel.HIGH
    assert prefers_no_trade(score)


def test_chaotic_market_with_conflicting_hypotheses_yields_extreme_uncertainty():
    bullish = _hyp(direction=BeliefDirection.BULLISH, confidence=0.7)
    bearish = _hyp(direction=BeliefDirection.BEARISH, confidence=0.7)
    score = compute_uncertainty(
        SYM,
        news_risk=0.9,
        volatility_state="EXTREME",
        volatility_quality=0.1,
        liquidity_state="THIN",
        liquidity_quality=0.1,
        trend_valid=False,
        microstructure_quality=0.1,
        active_hypotheses=[bullish, bearish],
    )
    assert score.level == UncertaintyLevel.EXTREME
    assert prefers_no_trade(score)
    assert does_not_understand_market(score)


def test_conflicting_directional_hypotheses_raise_regime_conflict_signal():
    bullish = _hyp(direction=BeliefDirection.BULLISH, confidence=0.6)
    bearish = _hyp(direction=BeliefDirection.BEARISH, confidence=0.6)
    score = compute_uncertainty(SYM, active_hypotheses=[bullish, bearish])
    assert score.signals.regime_conflict > 0.5


def test_balanced_evidence_on_a_single_hypothesis_raises_evidence_conflict_signal():
    conflicted = _hyp(supporting_evidence=["a", "b"], contradicting_evidence=["c", "d"])
    score = compute_uncertainty(SYM, active_hypotheses=[conflicted])
    assert score.signals.evidence_conflict > 0.5


# ── HypothesisEngine integration ─────────────────────────────────────────────


def test_hypothesis_engine_conviction_for_matches_compute_conviction():
    engine = HypothesisEngine(QDEBrain())
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.5)

    score = engine.conviction_for(hyp.hypothesis_id)
    assert score is not None
    assert score.hypothesis_id == hyp.hypothesis_id


def test_hypothesis_engine_uncertainty_for_unknown_symbol_defaults_to_low():
    engine = HypothesisEngine(QDEBrain())
    score = engine.uncertainty_for("UNKNOWN")
    assert score.level in (UncertaintyLevel.LOW, UncertaintyLevel.MODERATE)


def test_trade_worthy_hypotheses_empty_when_market_uncertainty_extreme():
    engine = HypothesisEngine(QDEBrain())
    engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.9)

    worthy = engine.trade_worthy_hypotheses(
        SYM,
        news_risk=0.95,
        trend_valid=False,
        microstructure_quality=0.05,
    )
    assert worthy == []


def test_trade_worthy_hypotheses_includes_low_confidence_high_conviction_setup():
    engine = HypothesisEngine(QDEBrain())
    hyp = engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.42)
    for i in range(5):
        engine.reinforce(hyp.hypothesis_id, f"evidence {i}")
    # Backdate creation so the persistence/dwell component of conviction matures.
    located = engine._locate(hyp.hypothesis_id)
    assert located is not None
    _, state = located
    state.hypotheses[hyp.hypothesis_id].created_at = time.time() - 600.0

    worthy = engine.trade_worthy_hypotheses(SYM, min_conviction=0.5, min_confidence=0.3)
    ids = [v.hypothesis.hypothesis_id for v in worthy]
    assert hyp.hypothesis_id in ids
