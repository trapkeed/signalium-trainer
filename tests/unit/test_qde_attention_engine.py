"""QDE Brain — attention engine / allocator / opportunity scanner tests.

Covers dynamic per-symbol attention: increases from developing
hypotheses, volatility expansion, liquidity events, news impact,
unusual behavior, and session transitions; a maintained per-symbol
score; gating of expensive reasoning cycles to high-attention symbols
only; and continuous rebalancing as conditions evolve.
"""

from __future__ import annotations

from trainer.lifecycle.lifecycle_models import MarketContext
from trainer.qde.brain.attention_allocator import (
    AttentionScoreAllocator,
    AttentionWeights,
)
from trainer.qde.brain.attention_engine import AttentionEngine
from trainer.qde.brain.brain import QDEBrain
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.opportunity_scanner import OpportunitySignals, scan
from trainer.qde.qde_observer_models import SymbolObservationInput

SYM = "EURUSD"
SYM2 = "GBPUSD"


def _obs(
    instrument: str = SYM,
    volatility_state: str = "NORMAL",
    volatility_quality: float = 0.5,
    liquidity_state: str = "NORMAL",
    liquidity_quality: float = 0.5,
    session: str = "LONDON",
    session_quality: float = 0.5,
    market_regime: str = "RANGE",
    news_risk: float = 0.0,
    momentum: float = 0.0,
    trend_valid: bool = True,
    microstructure_quality: float = 0.5,
) -> SymbolObservationInput:
    return SymbolObservationInput(
        instrument=instrument,
        market=MarketContext(
            session=session, news_risk=news_risk, momentum=momentum, trend_valid=trend_valid
        ),
        volatility_state=volatility_state,
        volatility_quality=volatility_quality,
        liquidity_state=liquidity_state,
        liquidity_quality=liquidity_quality,
        session_quality=session_quality,
        market_regime=market_regime,
        microstructure_quality=microstructure_quality,
    )


# ── Opportunity scanner ──────────────────────────────────────────────────────


def test_scan_first_tick_has_no_delta_signals_but_reads_news_risk_directly():
    signals = scan(_obs(news_risk=0.7))
    assert signals.news_impact == 0.7
    assert signals.volatility_expansion == 0.0
    assert signals.liquidity_event == 0.0
    assert signals.session_transition == 0.0


def test_scan_detects_volatility_expansion():
    prev = _obs(volatility_state="NORMAL")
    cur = _obs(volatility_state="EXTREME")
    signals = scan(cur, prev)
    assert signals.volatility_expansion > 0.5


def test_scan_detects_liquidity_event():
    prev = _obs(liquidity_state="NORMAL", liquidity_quality=0.5)
    cur = _obs(liquidity_state="POOR", liquidity_quality=0.1)
    signals = scan(cur, prev)
    assert signals.liquidity_event > 0.0


def test_scan_detects_unusual_behavior_on_regime_flip():
    prev = _obs(market_regime="TREND_UP")
    cur = _obs(market_regime="TREND_DOWN")
    signals = scan(cur, prev)
    assert signals.unusual_behavior >= 0.6


def test_scan_detects_unusual_behavior_on_invalid_trend():
    signals = scan(_obs(trend_valid=False))
    assert signals.unusual_behavior >= 0.5


def test_scan_detects_session_transition_on_session_change():
    prev = _obs(session="LONDON")
    cur = _obs(session="NEWYORK")
    signals = scan(cur, prev)
    assert signals.session_transition == 1.0


def test_scan_detects_session_transition_on_quality_bucket_crossing():
    prev = _obs(session="LONDON", session_quality=0.8)
    cur = _obs(session="LONDON", session_quality=0.3)
    signals = scan(cur, prev)
    assert signals.session_transition >= 0.6


def test_scan_hypothesis_activity_from_new_hypothesis_and_counts():
    assert scan(_obs(), new_hypothesis_formed=True).hypothesis_activity == 1.0
    assert scan(_obs(), hypothesis_reinforced=True).hypothesis_activity == 0.6
    assert abs(scan(_obs(), active_hypothesis_count=3).hypothesis_activity - 0.6) < 1e-9


# ── Attention allocator ──────────────────────────────────────────────────────


def test_allocator_update_produces_composite_from_weighted_signals():
    allocator = AttentionScoreAllocator()
    signals = OpportunitySignals(volatility_expansion=1.0)
    score = allocator.update(SYM, signals, now=1000.0)
    assert score.composite == AttentionWeights().volatility_expansion
    assert score.instrument == SYM


def test_allocator_decay_reduces_composite_over_time():
    allocator = AttentionScoreAllocator(decay_per_second=0.5)
    allocator.update(SYM, OpportunitySignals(news_impact=1.0), now=1000.0)
    before = allocator.get(SYM).composite
    allocator.decay_all(now=1010.0)
    after = allocator.get(SYM).composite
    assert after < before


def test_allocator_reallocate_normalizes_weights_across_symbols():
    allocator = AttentionScoreAllocator()
    allocator.update(SYM, OpportunitySignals(news_impact=1.0), now=1000.0)
    allocator.update(SYM2, OpportunitySignals(news_impact=0.1), now=1000.0)
    allocations = allocator.reallocate(now=1000.0)
    assert abs(sum(a.weight for a in allocations.values()) - 1.0) < 1e-6
    assert allocations[SYM].weight > allocations[SYM2].weight


def test_allocator_high_attention_ranks_by_weight():
    allocator = AttentionScoreAllocator()
    allocator.update(SYM, OpportunitySignals(news_impact=1.0), now=1000.0)
    allocator.update(SYM2, OpportunitySignals(news_impact=0.05), now=1000.0)
    allocator.reallocate(now=1000.0)
    high = allocator.high_attention(threshold=0.6)
    assert high == [SYM]


def test_allocator_remove_drops_symbol():
    allocator = AttentionScoreAllocator()
    allocator.update(SYM, OpportunitySignals(news_impact=1.0), now=1000.0)
    allocator.remove(SYM)
    assert allocator.get(SYM) is None


# ── Attention engine ─────────────────────────────────────────────────────────


def test_engine_observe_updates_brain_state_attention_salience():
    brain = QDEBrain()
    engine = AttentionEngine(brain)
    engine.observe(_obs(news_risk=0.8), now=1000.0)
    state = brain.get_state(SYM)
    assert state.attention.salience > 0.0


def test_engine_observe_integrates_hypothesis_activity_from_engine():
    brain = QDEBrain()
    hyp_engine = HypothesisEngine(brain)
    engine = AttentionEngine(brain, hypothesis_engine=hyp_engine)

    hyp = hyp_engine.create(SYM, "thesis", BeliefDirection.BULLISH, confidence=0.8)
    for _ in range(4):
        hyp_engine.reinforce(hyp.hypothesis_id, "supporting evidence")

    score = engine.observe(_obs(), now=1000.0)
    assert score.last_signals.hypothesis_activity > 0.0


def test_engine_rebalance_normalizes_weights_across_symbols():
    brain = QDEBrain()
    engine = AttentionEngine(brain)
    engine.observe(_obs(SYM, news_risk=0.9), now=1000.0)
    engine.observe(_obs(SYM2, news_risk=0.1), now=1000.0)

    weights = engine.rebalance(now=1000.0)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights[SYM] > weights[SYM2]
    assert brain.get_state(SYM).attention.weight == weights[SYM]


def test_engine_high_attention_symbols_gates_reasoning_cycle():
    brain = QDEBrain()
    engine = AttentionEngine(brain)
    engine.observe(_obs(SYM, news_risk=1.0), now=1000.0)
    engine.observe(_obs(SYM2, news_risk=0.0), now=1000.0)
    engine.rebalance(now=1000.0)

    high = engine.high_attention_symbols(threshold=0.6)
    assert SYM in high
    assert SYM2 not in high
    assert engine.should_run_expensive_reasoning(SYM, threshold=0.6)
    assert not engine.should_run_expensive_reasoning(SYM2, threshold=0.6)


def test_engine_run_reasoning_cycle_only_reflects_high_attention_symbols():
    brain = QDEBrain()
    engine = AttentionEngine(brain)
    engine.observe(_obs(SYM, news_risk=1.0), now=1000.0)
    engine.observe(_obs(SYM2, news_risk=0.0), now=1000.0)
    engine.rebalance(now=1000.0)

    results = engine.run_reasoning_cycle(threshold=0.6, force=True)
    assert SYM in results
    assert SYM2 not in results


def test_engine_continuously_rebalances_as_conditions_evolve():
    brain = QDEBrain()
    engine = AttentionEngine(brain)
    engine.observe(_obs(SYM, news_risk=1.0), now=1000.0)
    engine.observe(_obs(SYM2, news_risk=0.0), now=1000.0)
    first = engine.rebalance(now=1000.0)

    # Conditions evolve: SYM2 now has a volatility spike, SYM goes quiet.
    engine.observe(
        _obs(SYM2, volatility_state="EXTREME"),
        now=1050.0,
    )
    second = engine.rebalance(now=1100.0)

    assert second[SYM2] > first[SYM2]


def test_engine_drop_removes_symbol_from_tracking():
    brain = QDEBrain()
    engine = AttentionEngine(brain)
    engine.observe(_obs(SYM, news_risk=1.0), now=1000.0)
    engine.drop(SYM)
    assert engine.score_for(SYM) is None
    assert engine.all_scores() == {}
