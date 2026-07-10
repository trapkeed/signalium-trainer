"""QDE Brain — setup queue.

`SetupQueue` is the orchestration facade for the opportunity queue,
following the exact composition pattern already established by
`HypothesisEngine` / `attention_allocator.AttentionScoreAllocator`: it
is constructed externally by a caller that already holds a `QDEBrain`
(and, optionally, a `HypothesisEngine`), it owns no duplicate storage
of its own, and it is not wired into `QDEBrain.on_market_update`
automatically — callers that want the queue driven every tick call
`observe()` themselves, the same way `HypothesisEngine`/`AttentionEngine`
are driven externally today.

What this module *adds* that did not exist before:
  * A queue of `Opportunity` records (`BrainState.opportunities`) that
    observes each of a symbol's active hypotheses *over time* — through
    WATCHING -> ARMED -> CONFIRMING -> READY -> EXECUTING -> COOLDOWN —
    rather than treating a single strong tick as tradeable.
  * `ready_opportunities()` — the concrete integration seam for "entry
    should occur only after sufficient evidence accumulates": a
    decision layer (e.g. `qde_inference.py`) can check this before
    treating any hypothesis as immediately actionable. This module
    deliberately does not wire itself into `qde_inference.py` — that
    remains the caller's decision, exactly as `HypothesisEngine`'s own
    docstring describes for `executable_hypotheses()`.
  * `mark_executing()` / `complete()` — the only way an opportunity
    leaves READY/EXECUTING, so the queue's own state can never silently
    resolve an entry itself; that stays an explicit caller action.

Integration points, each reused rather than re-derived:
  * hypotheses    -> `HypothesisEngine` (maturity / conviction / uncertainty)
  * attention     -> `attention_allocator.AttentionScoreAllocator` (or, if
                     none is supplied, the symbol's existing
                     `BrainState.attention.weight`)
  * working memory -> `BrainState.working_memory` via `WorkingMemory.add`
  * market narratives -> `BrainState.narrative` (tone + summary snapshot)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.attention_allocator import AttentionScoreAllocator
from trainer.qde.brain.brain_models import BrainState
from trainer.qde.brain.brain_types import NarrativeTone
from trainer.qde.brain.hypothesis_engine import HypothesisEngine
from trainer.qde.brain.hypothesis_lifecycle import HypothesisMaturity
from trainer.qde.brain.opportunity_tracker import Opportunity, OpportunityTracker
from trainer.qde.brain.setup_lifecycle import (
    DEFAULT_SETUP_THRESHOLDS,
    SetupEvidence,
    SetupState,
    SetupThresholds,
    next_state,
)
from trainer.qde.brain.working_memory import WorkingMemory

if TYPE_CHECKING:
    from trainer.qde.brain.brain import QDEBrain

# Opportunities in these states represent evidence still accumulating —
# they are the ones worth re-observing every tick. COOLDOWN opportunities
# are only touched by `age()` (to decide whether to re-arm or dequeue).
_OBSERVABLE_STATES = frozenset(
    {SetupState.WATCHING, SetupState.ARMED, SetupState.CONFIRMING, SetupState.READY}
)


class SetupQueue:
    """Maintains, per symbol, a queue of developing opportunities that
    must accumulate sustained evidence before QDE will treat them as
    tradeable.
    """

    def __init__(
        self,
        brain: "QDEBrain",
        hypothesis_engine: HypothesisEngine | None = None,
        attention: AttentionScoreAllocator | None = None,
        thresholds: SetupThresholds = DEFAULT_SETUP_THRESHOLDS,
    ) -> None:
        self._brain = brain
        self._hypotheses = hypothesis_engine or HypothesisEngine(brain)
        self._attention = attention
        self._tracker = OpportunityTracker()
        self._thresholds = thresholds
        self._working_memory = WorkingMemory()

    # ── Continuous observation ──────────────────────────────────────────

    def observe(self, symbol: str, now: float | None = None) -> list[Opportunity]:
        """Advance the queue for `symbol` by one tick: open an opportunity
        for any active hypothesis that does not already have one, then
        re-evaluate every non-cooled-down opportunity's evidence and let
        `setup_lifecycle.next_state` decide whether it progresses.

        Must be called repeatedly (e.g. once per market update, the same
        cadence as `QDEBrain.on_market_update`) — a single call can only
        ever move a queue entry one step, by design: entry only after
        evidence accumulates *over time*, not on one strong reading.
        """
        now = now if now is not None else time.time()
        state = self._brain.get_state(symbol)

        for view in self._hypotheses.registry.active(symbol):
            self._tracker.open(
                state.opportunities,
                symbol,
                view.hypothesis.hypothesis_id,
                direction=view.hypothesis.direction,
                now=now,
            )

        attention_weight = self._attention_weight(symbol, state)
        narrative = state.narrative
        narrative_tone = narrative.tone if narrative else NarrativeTone.NEUTRAL
        narrative_summary = narrative.summary if narrative else ""

        touched: list[Opportunity] = []
        for opp in list(state.opportunities.values()):
            if opp.state not in _OBSERVABLE_STATES:
                continue
            maturity = self._hypotheses.maturity_of(opp.hypothesis_id)
            if maturity is None:
                # Hypothesis no longer tracked (e.g. purged) — treat as
                # having nothing left to confirm; let it cool down.
                self._tracker.complete(state.opportunities, opp.opportunity_id, now=now)
                touched.append(opp)
                continue

            evidence = SetupEvidence(
                maturity=maturity,
                conviction=self._hypotheses.conviction_for(opp.hypothesis_id),
                uncertainty=self._hypotheses.uncertainty_for(symbol),
                attention_weight=attention_weight,
                narrative_tone=narrative_tone,
                narrative_snapshot_text=narrative_summary,
            )

            before = opp.state
            self._tracker.advance(
                state.opportunities,
                opp.opportunity_id,
                evidence,
                thresholds=self._thresholds,
                now=now,
            )
            if opp.state != before:
                self._log_transition(state, opp, before, now)
            touched.append(opp)

        self.age(symbol, now=now)
        self._persist(state)
        return touched

    def age(self, symbol: str | None = None, now: float | None = None) -> int:
        """Move eligible COOLDOWN opportunities back to WATCHING (re-arm)
        or drop them once fully stale. Delegates entirely to
        `OpportunityTracker`/`setup_lifecycle` — no reimplemented aging
        logic here.
        """
        now = now if now is not None else time.time()
        symbols = [symbol] if symbol is not None else list(self._brain.all_states().keys())
        pruned = 0
        for sym in symbols:
            state = self._brain.get_state(sym)
            for opp in list(state.opportunities.values()):
                if opp.state != SetupState.COOLDOWN:
                    continue
                maturity = self._hypotheses.maturity_of(opp.hypothesis_id)
                # A hypothesis that no longer exists (e.g. purged by
                # `age_all`) can never be confirmed again — treating it
                # as EXPIRED keeps `next_state` from ever re-arming the
                # opportunity, only pruning it once fully cooled down.
                evidence = SetupEvidence(
                    maturity=maturity if maturity is not None else HypothesisMaturity.EXPIRED,
                    dwell_seconds=opp.dwell_seconds(now),
                )
                proposed = next_state(
                    opp.state,
                    evidence,
                    confirmation_ticks=opp.confirmation_ticks,
                    thresholds=self._thresholds,
                )
                if proposed != opp.state:
                    opp.previous_state = opp.state
                    opp.state = proposed
                    opp.state_entered_at = now
                    opp.last_updated_at = now
            n = self._tracker.prune_stale(state.opportunities, now=now)
            pruned += n
            if n:
                self._persist(state)
        return pruned

    # ── Execution integration seam ───────────────────────────────────────

    def ready_opportunities(self, symbol: str | None = None) -> list[Opportunity]:
        """Opportunities that have accumulated sufficient sustained
        evidence to be considered for entry. This is the seam a decision
        layer should consult instead of acting on a single tick's
        maturity/conviction reading in isolation.
        """
        if symbol is not None:
            state = self._brain.get_state(symbol)
            return self._tracker.in_state(state.opportunities, SetupState.READY)
        ready: list[Opportunity] = []
        for state in self._brain.all_states().values():
            ready.extend(self._tracker.in_state(state.opportunities, SetupState.READY))
        return ready

    def mark_executing(self, symbol: str, opportunity_id: str) -> Opportunity | None:
        state = self._brain.get_state(symbol)
        opp = self._tracker.mark_executing(state.opportunities, opportunity_id)
        if opp is not None:
            self._log_transition(state, opp, SetupState.READY, time.time())
            self._persist(state)
        return opp

    def complete(self, symbol: str, opportunity_id: str) -> Opportunity | None:
        state = self._brain.get_state(symbol)
        prior_state = (
            state.opportunities[opportunity_id].state
            if (opportunity_id in state.opportunities)
            else None
        )
        opp = self._tracker.complete(state.opportunities, opportunity_id)
        if opp is not None and prior_state is not None:
            self._log_transition(state, opp, prior_state, time.time())
            self._persist(state)
        return opp

    # ── Internals ─────────────────────────────────────────────────────────

    def _attention_weight(self, symbol: str, state: BrainState) -> float:
        if self._attention is not None:
            score = self._attention.get(symbol)
            if score is not None:
                return score.weight
        return state.attention.weight

    def _log_transition(
        self, state: BrainState, opp: Opportunity, before: SetupState, now: float
    ) -> None:
        self._working_memory.add(
            state.working_memory,
            instrument=state.instrument,
            kind="OPPORTUNITY_UPDATE",
            content={
                "opportunity_id": opp.opportunity_id,
                "hypothesis_id": opp.hypothesis_id,
                "from_state": before.value,
                "to_state": opp.state.value,
                "confirmation_ticks": opp.confirmation_ticks,
            },
            salience=0.5 if opp.state in (SetupState.READY, SetupState.EXECUTING) else 0.35,
            now=now,
        )

    def _persist(self, state: BrainState) -> None:
        db = getattr(self._brain, "_db", None)
        if db is not None:
            db.upsert_state(state.to_dict())


__all__ = ["SetupQueue"]
