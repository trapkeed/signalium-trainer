"""QDE Brain — opportunity tracker.

Owns the `Opportunity` record (one developing setup being observed
over time) and stateless CRUD/transition operations over a per-symbol
`dict[str, Opportunity]`, mirroring the exact convention
`hypotheses.HypothesisTracker` already established for
`BrainState.hypotheses`: storage lives on `BrainState.opportunities`,
this class only operates on it, it never holds state of its own.

An `Opportunity` is deliberately a thin wrapper that *references* a
hypothesis (`hypothesis_id`) rather than duplicating its confidence/
evidence fields — the hypothesis remains the single source of truth
for "what QDE believes"; the opportunity only tracks "how long, and
how consistently, has this belief been observed to hold up", which is
the piece that did not exist before.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.setup_lifecycle import (
    DEFAULT_SETUP_THRESHOLDS,
    SetupEvidence,
    SetupState,
    SetupThresholds,
    next_state,
)
from trainer.qde.brain.thesis_manager import Thesis

MAX_TRANSITION_HISTORY = 20


@dataclass
class Opportunity:
    """One developing setup being observed over time before QDE will
    consider entering a trade from it.
    """

    opportunity_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    hypothesis_id: str = ""
    direction: BeliefDirection = BeliefDirection.NEUTRAL
    state: SetupState = SetupState.WATCHING
    previous_state: SetupState = SetupState.WATCHING
    confirmation_ticks: int = 0
    narrative_snapshot: str = ""
    last_reason: str = ""
    transition_history: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    state_entered_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    # The competing bull/bear arguments for this opportunity — see
    # `thesis_manager.ThesisManager.sync`, which keeps these up to date.
    # Every opportunity carries both, regardless of which direction its
    # tracked hypothesis favors, so the debate always has two sides.
    bull_thesis: Thesis = field(default_factory=lambda: Thesis(side=BeliefDirection.BULLISH))
    bear_thesis: Thesis = field(default_factory=lambda: Thesis(side=BeliefDirection.BEARISH))

    def dwell_seconds(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, now - self.state_entered_at)

    def is_dequeueable(self, now: float | None = None) -> bool:
        """An opportunity may be safely dropped from the queue once it
        has completed a full COOLDOWN without re-arming.
        """
        return self.state == SetupState.COOLDOWN and self.dwell_seconds(now) >= 0.0

    def to_dict(self) -> dict:
        return {
            "opportunity_id": self.opportunity_id,
            "instrument": self.instrument,
            "hypothesis_id": self.hypothesis_id,
            "direction": self.direction.value,
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "confirmation_ticks": self.confirmation_ticks,
            "narrative_snapshot": self.narrative_snapshot,
            "last_reason": self.last_reason,
            "transition_history": self.transition_history[-MAX_TRANSITION_HISTORY:],
            "created_at": round(self.created_at, 3),
            "state_entered_at": round(self.state_entered_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
            "bull_thesis": self.bull_thesis.to_dict(),
            "bear_thesis": self.bear_thesis.to_dict(),
        }


class OpportunityTracker:
    """Stateless operations over a per-symbol `dict[str, Opportunity]`
    (owned by `BrainState.opportunities`), exactly mirroring the
    `HypothesisTracker` convention.
    """

    def open(
        self,
        opportunities: dict[str, Opportunity],
        instrument: str,
        hypothesis_id: str,
        direction: BeliefDirection = BeliefDirection.NEUTRAL,
        now: float | None = None,
    ) -> Opportunity:
        """Register a new opportunity for `hypothesis_id`, unless one is
        already open (non-COOLDOWN) for that same hypothesis — in which
        case the existing record is returned unchanged, so re-observing
        the same hypothesis never spawns duplicate queue entries.
        """
        now = now if now is not None else time.time()
        for opp in opportunities.values():
            if opp.hypothesis_id == hypothesis_id and opp.state != SetupState.COOLDOWN:
                return opp

        opp = Opportunity(
            instrument=instrument,
            hypothesis_id=hypothesis_id,
            direction=direction,
            created_at=now,
            state_entered_at=now,
            last_updated_at=now,
        )
        opportunities[opp.opportunity_id] = opp
        return opp

    def advance(
        self,
        opportunities: dict[str, Opportunity],
        opportunity_id: str,
        evidence: SetupEvidence,
        thresholds: SetupThresholds = DEFAULT_SETUP_THRESHOLDS,
        now: float | None = None,
    ) -> Opportunity | None:
        """Apply one tick's evidence to an opportunity, transitioning it
        per `setup_lifecycle.next_state` and updating the sustained-
        evidence counter used to gate CONFIRMING -> READY.
        """
        opp = opportunities.get(opportunity_id)
        if opp is None:
            return None
        now = now if now is not None else time.time()

        dwell_evidence = SetupEvidence(
            maturity=evidence.maturity,
            conviction=evidence.conviction,
            uncertainty=evidence.uncertainty,
            attention_weight=evidence.attention_weight,
            narrative_tone=evidence.narrative_tone,
            dwell_seconds=opp.dwell_seconds(now),
        )

        proposed = next_state(
            opp.state,
            dwell_evidence,
            confirmation_ticks=opp.confirmation_ticks,
            thresholds=thresholds,
        )

        if proposed == SetupState.CONFIRMING and opp.state == SetupState.CONFIRMING:
            opp.confirmation_ticks += 1
        elif proposed != opp.state:
            opp.confirmation_ticks = 1 if proposed == SetupState.CONFIRMING else 0

        if evidence.narrative_snapshot_text:
            opp.narrative_snapshot = evidence.narrative_snapshot_text

        if proposed != opp.state:
            opp.previous_state = opp.state
            opp.transition_history.append(
                f"{time.strftime('%H:%M:%S', time.gmtime(now))} "
                f"{opp.state.value} -> {proposed.value}"
            )
            opp.state = proposed
            opp.state_entered_at = now

        opp.last_reason = (
            evidence.uncertainty.explanation if evidence.uncertainty else opp.last_reason
        )
        opp.last_updated_at = now
        return opp

    def mark_executing(
        self, opportunities: dict[str, Opportunity], opportunity_id: str, now: float | None = None
    ) -> Opportunity | None:
        """Explicit, decision-layer-initiated handoff: a READY opportunity
        is being acted on. This never places the order itself — it only
        records that execution is in progress, so a second decision pass
        does not treat the same opportunity as still awaiting entry.
        """
        opp = opportunities.get(opportunity_id)
        if opp is None or opp.state != SetupState.READY:
            return None
        now = now if now is not None else time.time()
        opp.previous_state = opp.state
        opp.state = SetupState.EXECUTING
        opp.state_entered_at = now
        opp.last_updated_at = now
        opp.transition_history.append(
            f"{time.strftime('%H:%M:%S', time.gmtime(now))} READY -> EXECUTING"
        )
        return opp

    def complete(
        self, opportunities: dict[str, Opportunity], opportunity_id: str, now: float | None = None
    ) -> Opportunity | None:
        """Close out an EXECUTING opportunity into COOLDOWN once the
        decision layer has finished acting on it (filled, rejected, or
        expired) — the only way out of EXECUTING besides the orphaned-
        handoff safety net in `setup_lifecycle.next_state`.
        """
        opp = opportunities.get(opportunity_id)
        if opp is None:
            return None
        now = now if now is not None else time.time()
        opp.previous_state = opp.state
        opp.state = SetupState.COOLDOWN
        opp.state_entered_at = now
        opp.last_updated_at = now
        opp.transition_history.append(
            f"{time.strftime('%H:%M:%S', time.gmtime(now))} {opp.previous_state.value} -> COOLDOWN"
        )
        return opp

    def prune_stale(self, opportunities: dict[str, Opportunity], now: float | None = None) -> int:
        """Drop opportunities that have fully cooled down without
        re-arming, bounding queue growth the same way
        `WorkingMemory.prune_stale` bounds working memory.
        """
        now = now if now is not None else time.time()
        before = len(opportunities)
        for opp_id, opp in list(opportunities.items()):
            if opp.state == SetupState.COOLDOWN and opp.dwell_seconds(now) >= 3600.0:
                del opportunities[opp_id]
        return before - len(opportunities)

    def active(self, opportunities: dict[str, Opportunity]) -> list[Opportunity]:
        return [o for o in opportunities.values() if o.state != SetupState.COOLDOWN]

    def in_state(
        self, opportunities: dict[str, Opportunity], state: SetupState
    ) -> list[Opportunity]:
        return [o for o in opportunities.values() if o.state == state]

    def by_hypothesis(
        self, opportunities: dict[str, Opportunity], hypothesis_id: str
    ) -> Opportunity | None:
        for opp in opportunities.values():
            if opp.hypothesis_id == hypothesis_id:
                return opp
        return None


__all__ = ["Opportunity", "OpportunityTracker", "Thesis"]
