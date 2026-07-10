"""QDE Brain — debate engine.

`DebateEngine` is the orchestration facade that *compares* the two
competing arguments `thesis_manager.ThesisManager` maintains on every
`Opportunity` — it never builds or stores theses itself, and it never
places or blocks an order itself, following the exact
external-facade / consult-only-seam convention already established by
`HypothesisEngine` and `SetupQueue`.

What this module adds that did not exist before:
  * `DebateVerdict` — a single comparison of an opportunity's bull vs.
    bear `Thesis.strength()` (each of which already folds in supporting
    evidence, conflicting evidence, and historical outcomes — see
    `thesis_manager.py`), producing a margin and a dominance verdict.
  * `evaluate()` — refreshes both theses via `ThesisManager.sync` and
    returns the verdict, logging it to working memory exactly like
    `SetupQueue._log_transition` logs state changes.
  * `clears_for_execution()` — the concrete integration seam for "final
    execution should occur only when one thesis clearly dominates": a
    decision layer (e.g. before calling `SetupQueue.mark_executing`)
    should consult this. This module deliberately does not wire itself
    into `SetupQueue.mark_executing` — that stays an explicit caller
    decision, the same boundary `SetupQueue.ready_opportunities()`
    already respects for its own callers.

Integration points, each reused rather than re-derived:
  * theses            -> `thesis_manager.ThesisManager` (evidence + confidence
                          + historical support, already computed there)
  * opportunities      -> `BrainState.opportunities` / `opportunity_tracker`
  * hypotheses         -> `BrainState.hypotheses`
  * working memory     -> `BrainState.working_memory` via `WorkingMemory.add`
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.brain.brain_models import BrainState
from trainer.qde.brain.brain_types import BeliefDirection
from trainer.qde.brain.thesis_manager import ThesisManager
from trainer.qde.brain.working_memory import WorkingMemory

if TYPE_CHECKING:
    from trainer.qde.brain.brain import QDEBrain

# How much stronger the winning thesis must be, on the 0-1 `strength()`
# scale, before the debate is considered settled rather than contested.
DEFAULT_DOMINANCE_MARGIN = 0.2


@dataclass
class DebateVerdict:
    """The result of comparing one opportunity's bull and bear theses."""

    instrument: str
    opportunity_id: str
    bull_strength: float
    bear_strength: float
    margin: float
    dominant_side: BeliefDirection | None
    conclusive: bool
    explanation: str = ""
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "opportunity_id": self.opportunity_id,
            "bull_strength": round(self.bull_strength, 4),
            "bear_strength": round(self.bear_strength, 4),
            "margin": round(self.margin, 4),
            "dominant_side": self.dominant_side.value if self.dominant_side else None,
            "conclusive": self.conclusive,
            "explanation": self.explanation,
            "computed_at": round(self.computed_at, 3),
        }


class DebateEngine:
    """Weighs an opportunity's bull thesis against its bear thesis and
    decides whether either one currently, clearly dominates.
    """

    def __init__(
        self,
        brain: "QDEBrain",
        thesis_manager: ThesisManager | None = None,
        dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
    ) -> None:
        self._brain = brain
        self._theses = thesis_manager or ThesisManager()
        self._dominance_margin = dominance_margin
        self._working_memory = WorkingMemory()

    def evaluate(
        self, symbol: str, opportunity_id: str, now: float | None = None
    ) -> DebateVerdict | None:
        """Refresh both theses for `opportunity_id` (via `ThesisManager.
        sync`) and compare them. Returns `None` if the opportunity is not
        tracked — mirroring `OpportunityTracker`'s own "missing id" style.
        """
        now = now if now is not None else time.time()
        state = self._brain.get_state(symbol)
        opp = state.opportunities.get(opportunity_id)
        if opp is None:
            return None

        bull, bear = self._theses.sync(state, opp, state.hypotheses, now=now)
        bull_strength, bear_strength = bull.strength(), bear.strength()
        margin = abs(bull_strength - bear_strength)

        dominant_side: BeliefDirection | None = None
        if bull_strength > bear_strength:
            dominant_side = BeliefDirection.BULLISH
        elif bear_strength > bull_strength:
            dominant_side = BeliefDirection.BEARISH

        conclusive = dominant_side is not None and margin >= self._dominance_margin
        explanation = (
            f"{symbol}: bull {bull_strength:.0%} vs bear {bear_strength:.0%} "
            f"(margin {margin:.0%}) — "
            + (
                f"{dominant_side.value.lower()} thesis clearly dominates"
                if conclusive and dominant_side is not None
                else "no thesis clearly dominates"
            )
        )

        verdict = DebateVerdict(
            instrument=symbol,
            opportunity_id=opportunity_id,
            bull_strength=bull_strength,
            bear_strength=bear_strength,
            margin=margin,
            dominant_side=dominant_side,
            conclusive=conclusive,
            explanation=explanation,
            computed_at=now,
        )
        self._log_verdict(state, verdict, now)
        self._persist(state)
        return verdict

    def clears_for_execution(
        self, symbol: str, opportunity_id: str, now: float | None = None
    ) -> bool:
        """The integration seam for "final execution should occur only
        when one thesis clearly dominates": `True` only when the debate
        is conclusive *and* the dominant side agrees with the direction
        the opportunity itself is tracking — a dominant thesis that
        contradicts the tracked setup is a reason to withhold execution,
        not confirm it.
        """
        state = self._brain.get_state(symbol)
        opp = state.opportunities.get(opportunity_id)
        if opp is None:
            return False
        verdict = self.evaluate(symbol, opportunity_id, now=now)
        if verdict is None or not verdict.conclusive:
            return False
        return verdict.dominant_side == opp.direction

    # ── Internals ─────────────────────────────────────────────────────────

    def _log_verdict(self, state: BrainState, verdict: DebateVerdict, now: float) -> None:
        self._working_memory.add(
            state.working_memory,
            instrument=state.instrument,
            kind="DEBATE_VERDICT",
            content=verdict.to_dict(),
            salience=0.55 if verdict.conclusive else 0.35,
            now=now,
        )

    def _persist(self, state: BrainState) -> None:
        db = getattr(self._brain, "_db", None)
        if db is not None:
            db.upsert_state(state.to_dict())


__all__ = ["DebateEngine", "DebateVerdict", "DEFAULT_DOMINANCE_MARGIN"]
