"""QDE Brain — forgetting orchestration.

`ForgettingEngine` is the consult/apply seam that ties `memory_decay_engine`'s
pure math to a live `BrainState`, the same facade convention as
`SetupQueue`/`HypothesisEngine`/`DebateEngine`: a caller invokes it
explicitly (it is **not** auto-wired into `QDEBrain.on_market_update`, so
adopting it is opt-in per the existing convention of that pipeline only
running `WorkingMemory.prune_stale` / `HypothesisTracker.age` /
`LongTermMemory.consolidate` today).

Responsibilities are split so nothing here duplicates existing decay:

  - Working memory pruning is delegated straight to
    `WorkingMemory.prune_stale` — already correct, not re-derived.
  - Hypothesis *binary* expiry is delegated straight to
    `HypothesisTracker.age` — already correct, not re-derived.
  - Hypothesis *gradual* context decay (confidence fading when the market
    has moved to a different regime than the one the hypothesis was formed
    under) is new — via `memory_decay_engine.context_decay`.
  - Long-term memory relevance ranking (recency + context-match weighting
    for entries that are durable and never pruned) is new — via
    `memory_decay_engine.relevance_weight`.
  - Regime decay of `SymbolPersonality.regime_outcome_quality` toward
    neutral for regimes the market is no longer in is new — via
    `memory_decay_engine.regime_decay`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from trainer.qde.brain.brain_models import BrainState, Hypothesis, LongTermMemoryEntry
from trainer.qde.brain.hypotheses import HypothesisTracker
from trainer.qde.brain.memory_decay_engine import (
    context_decay,
    recency_weight,
    regime_decay,
    relevance_weight,
)
from trainer.qde.brain.working_memory import WorkingMemory

# Below this confidence, a context-decayed hypothesis is treated as no
# longer worth carrying forward and is invalidated rather than left to
# linger indefinitely at a near-zero (but never quite zero) confidence.
CONTEXT_DECAY_INVALIDATION_THRESHOLD = 0.05


@dataclass
class ForgettingReport:
    """What one `ForgettingEngine.forget()` pass did, for logging/tests."""

    instrument: str
    working_memory_pruned: int = 0
    hypotheses_expired: int = 0
    hypotheses_context_decayed: int = 0
    hypotheses_invalidated_by_context: int = 0
    regimes_decayed: list[str] = field(default_factory=list)


class ForgettingEngine:
    """Applies recency/relevance/context/regime decay to one symbol's
    `BrainState`. Stateless — like `WorkingMemory`/`LongTermMemory`, all
    state lives on the `BrainState` passed in.
    """

    def __init__(self) -> None:
        self._working_memory = WorkingMemory()
        self._hypotheses = HypothesisTracker()

    def forget(self, state: BrainState, now: float | None = None) -> ForgettingReport:
        """Run one full forgetting pass over `state`: prune decayed working
        memory, expire and context-decay hypotheses, and decay obsolete
        regime knowledge. Returns a report of what changed.
        """
        now = now if now is not None else time.time()
        report = ForgettingReport(instrument=state.instrument)

        report.working_memory_pruned = self._working_memory.prune_stale(
            state.working_memory, now=now
        )
        report.hypotheses_expired = self._hypotheses.age(state.hypotheses, now=now)

        current_regime = state.narrative.market_regime if state.narrative else ""
        decayed, invalidated = self._decay_hypothesis_context(state.hypotheses, current_regime, now)
        report.hypotheses_context_decayed = decayed
        report.hypotheses_invalidated_by_context = invalidated

        if current_regime:
            before = dict(state.personality.regime_outcome_quality)
            state.personality.regime_outcome_quality = regime_decay(
                state.personality.regime_outcome_quality,
                state.personality.regime_last_seen,
                current_regime,
                now,
            )
            report.regimes_decayed = sorted(
                regime
                for regime, quality in before.items()
                if regime != current_regime
                and state.personality.regime_outcome_quality.get(regime) != quality
            )

        return report

    def _decay_hypothesis_context(
        self, hypotheses: dict[str, Hypothesis], current_regime: str, now: float
    ) -> tuple[int, int]:
        decayed_count = 0
        invalidated_count = 0
        for hyp in hypotheses.values():
            if hyp.is_terminal() or not hyp.formed_regime:
                continue
            context_still_valid = hyp.formed_regime == current_regime
            elapsed = max(0.0, now - hyp.last_updated_at)
            new_confidence = context_decay(hyp.confidence, elapsed, context_still_valid)
            if new_confidence == hyp.confidence:
                continue
            hyp.confidence = new_confidence
            decayed_count += 1
            if hyp.confidence < CONTEXT_DECAY_INVALIDATION_THRESHOLD:
                hyp.invalidate(
                    f"context decay — regime moved from {hyp.formed_regime!r} "
                    f"to {current_regime!r}",
                    now=now,
                )
                invalidated_count += 1
        return decayed_count, invalidated_count

    def relevant_long_term_memories(
        self,
        entries: list[LongTermMemoryEntry],
        current_regime: str = "",
        n: int = 10,
        now: float | None = None,
    ) -> list[tuple[LongTermMemoryEntry, float]]:
        """Rank durable (never-pruned) long-term entries by present-moment
        relevance — recency-weighted, and down-weighted (not discarded) if
        the entry's tags don't reference the current regime. Consult-only:
        does not mutate or prune `entries`, since long-term memory is
        deliberately non-decaying storage; this only affects how it's
        *read*, mirroring `DebateEngine.clears_for_execution` being a
        read-only seam over durable state.
        """
        now = now if now is not None else time.time()
        scored = []
        for entry in entries:
            age = max(0.0, now - entry.created_at)
            context_match = not current_regime or current_regime in entry.tags
            weight = relevance_weight(entry.salience, age, context_match)
            scored.append((entry, weight))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:n]

    def recency_weight_of(self, timestamp: float, now: float | None = None) -> float:
        """Convenience wrapper so callers don't need to import
        `memory_decay_engine` directly just to weight a single timestamp.
        """
        now = now if now is not None else time.time()
        return recency_weight(max(0.0, now - timestamp))
