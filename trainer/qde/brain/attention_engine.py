"""QDE Brain — attention engine (orchestrator).

Ties together `opportunity_scanner` (signal detection) and
`attention_allocator` (scoring + decay + normalization) with the live
`QDEBrain`/`HypothesisEngine`, so attention:

  - increases from developing hypotheses, volatility expansion,
    liquidity events, news impact, unusual behavior, and session
    transitions (the six `OpportunitySignals` components);
  - is maintained as one running score per symbol;
  - continuously rebalances as conditions evolve (decay + reallocation
    each cycle, not a one-shot computation); and
  - gates "expensive reasoning cycles" — this reuses `QDEBrain.reflect`,
    the Brain's existing periodic self-review step, rather than
    inventing a second reasoning mechanism.

This module holds no market-data-fetching or scheduling logic itself;
it is driven by a caller (e.g. `brain_state_manager`) feeding it one
`SymbolObservationInput` per symbol per tick, mirroring how
`HypothesisEngine` is driven externally rather than polling on its own.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from trainer.qde.brain.attention_allocator import (
    DEFAULT_WEIGHTS,
    AttentionScore,
    AttentionScoreAllocator,
    AttentionWeights,
)
from trainer.qde.brain.opportunity_scanner import OpportunitySignals, scan
from trainer.qde.qde_observer_models import SymbolObservationInput

if TYPE_CHECKING:
    from trainer.memory.decision_memory import DecisionMemory
    from trainer.memory.knowledge_base import KnowledgeBase
    from trainer.qde.brain.brain import QDEBrain
    from trainer.qde.brain.brain_models import ReflectionEntry
    from trainer.qde.brain.hypothesis_engine import HypothesisEngine

DEFAULT_HIGH_ATTENTION_THRESHOLD = 0.15


class AttentionEngine:
    """Per-symbol attention scoring driven by market/hypothesis activity,
    with cross-symbol rebalancing and reasoning-cycle gating.
    """

    def __init__(
        self,
        brain: "QDEBrain",
        hypothesis_engine: "HypothesisEngine | None" = None,
        weights: AttentionWeights = DEFAULT_WEIGHTS,
    ) -> None:
        self._brain = brain
        self._hypothesis_engine = hypothesis_engine
        self._allocator = AttentionScoreAllocator(weights=weights)
        self._prior_inputs: dict[str, SymbolObservationInput] = {}

    # ── Ingestion ────────────────────────────────────────────────────────

    def observe(
        self,
        observation: SymbolObservationInput,
        new_hypothesis_formed: bool = False,
        hypothesis_reinforced: bool = False,
        now: float | None = None,
    ) -> AttentionScore:
        """Feed one tick's observation for a symbol into the attention
        system: detects this tick's opportunity signals, folds them into
        that symbol's running score, and syncs the composite salience
        into `BrainState.attention` so it's visible alongside the rest of
        the symbol's internal state.
        """
        instrument = observation.instrument
        previous = self._prior_inputs.get(instrument)

        active_count = 0
        max_confidence = 0.0
        if self._hypothesis_engine is not None:
            views = self._hypothesis_engine.registry.active(instrument)
            active_count = len(views)
            max_confidence = max((v.hypothesis.confidence for v in views), default=0.0)

        signals = scan(
            observation,
            previous,
            active_hypothesis_count=active_count,
            new_hypothesis_formed=new_hypothesis_formed,
            hypothesis_reinforced=hypothesis_reinforced,
        )
        # A high-conviction active hypothesis keeps a symbol interesting
        # even without a fresh delta this tick.
        if max_confidence >= 0.7 and signals.hypothesis_activity < max_confidence:
            signals = OpportunitySignals(
                hypothesis_activity=max_confidence,
                volatility_expansion=signals.volatility_expansion,
                liquidity_event=signals.liquidity_event,
                news_impact=signals.news_impact,
                unusual_behavior=signals.unusual_behavior,
                session_transition=signals.session_transition,
            )

        score = self._allocator.update(instrument, signals, now=now)
        self._prior_inputs[instrument] = observation

        state = self._brain.get_state(instrument)
        state.attention.salience = score.composite
        state.attention.updated_at = score.updated_at

        return score

    # ── Rebalancing ──────────────────────────────────────────────────────

    def rebalance(self, now: float | None = None) -> dict[str, float]:
        """Decay every tracked symbol's residual attention, then
        renormalize weights across all symbols. Call this once per cycle
        (independent of `observe()`) so attention keeps shifting even for
        symbols that didn't tick this round.
        """
        now = now if now is not None else time.time()
        self._allocator.decay_all(now=now)
        allocations = self._allocator.reallocate(now=now)
        for instrument, allocation in allocations.items():
            self._brain.get_state(instrument).attention = allocation
        return {k: v.weight for k, v in allocations.items()}

    # ── Reasoning-cycle gating ───────────────────────────────────────────

    def high_attention_symbols(
        self, threshold: float = DEFAULT_HIGH_ATTENTION_THRESHOLD
    ) -> list[str]:
        """Symbols currently deserving expensive reasoning, ranked by
        normalized attention weight (highest first).
        """
        return self._allocator.high_attention(threshold)

    def should_run_expensive_reasoning(
        self, instrument: str, threshold: float = DEFAULT_HIGH_ATTENTION_THRESHOLD
    ) -> bool:
        return instrument in self.high_attention_symbols(threshold)

    def run_reasoning_cycle(
        self,
        threshold: float = DEFAULT_HIGH_ATTENTION_THRESHOLD,
        decision_memory: "DecisionMemory | None" = None,
        knowledge_base: "KnowledgeBase | None" = None,
        force: bool = False,
    ) -> dict[str, "ReflectionEntry | None"]:
        """Run the Brain's periodic reflection only for symbols that
        currently meet the attention threshold — this is what makes
        reflection an "expensive reasoning cycle" reserved for symbols
        that have earned it, instead of running uniformly for every
        tracked symbol every cycle.
        """
        results: dict[str, "ReflectionEntry | None"] = {}
        for instrument in self.high_attention_symbols(threshold):
            results[instrument] = self._brain.reflect(
                instrument,
                decision_memory=decision_memory,
                knowledge_base=knowledge_base,
                force=force,
            )
        return results

    # ── Introspection ────────────────────────────────────────────────────

    def score_for(self, instrument: str) -> AttentionScore | None:
        return self._allocator.get(instrument)

    def all_scores(self) -> dict[str, AttentionScore]:
        return self._allocator.all()

    def drop(self, instrument: str) -> None:
        """Stop tracking a symbol's attention state (e.g. it was
        disabled/delisted) so it no longer competes for reallocated
        weight or holds stale prior-tick context.
        """
        self._allocator.remove(instrument)
        self._prior_inputs.pop(instrument, None)
