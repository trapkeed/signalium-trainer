"""QDE Brain — hypothesis tracking.

Owns hypothesis formation, reinforcement, invalidation, and aging. This is
QDE's "developing setups" tracking layer — a hypothesis represents a
standing idea about what a symbol is building toward, distinct from the
Observer's lifecycle state (which tracks readiness-to-enter, not the
underlying thesis for *why* the symbol might be readying).
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection, HypothesisStatus

MAX_ACTIVE_HYPOTHESES_PER_SYMBOL = 8


class HypothesisTracker:
    """Stateless operations over a per-symbol hypothesis dict (owned by
    `BrainState.hypotheses`), mirroring the `WorkingMemory`/`LongTermMemory`
    convention of `QDEBrain` holding the actual state.
    """

    def form(
        self,
        hypotheses: dict[str, Hypothesis],
        instrument: str,
        thesis: str,
        direction: BeliefDirection,
        confidence: float = 0.3,
        now: float | None = None,
    ) -> Hypothesis:
        """Form a new hypothesis, unless an equivalent one (same thesis
        text, still non-terminal) is already active — in which case that
        one is reinforced instead, so re-observing the same setup
        strengthens conviction rather than spawning duplicates.
        """
        now = now if now is not None else time.time()
        for h in hypotheses.values():
            if h.thesis == thesis and not h.is_terminal():
                h.reinforce("re-observed", boost=0.05, now=now)
                return h

        hyp = Hypothesis(
            instrument=instrument,
            thesis=thesis,
            direction=direction,
            confidence=confidence,
            created_at=now,
            last_updated_at=now,
        )
        hypotheses[hyp.hypothesis_id] = hyp

        active = [h for h in hypotheses.values() if not h.is_terminal()]
        if len(active) > MAX_ACTIVE_HYPOTHESES_PER_SYMBOL:
            weakest = min(active, key=lambda h: h.confidence)
            weakest.invalidate("superseded — attention budget exceeded", now=now)

        return hyp

    def reinforce(
        self, hypotheses: dict[str, Hypothesis], hypothesis_id: str, evidence: str
    ) -> Hypothesis | None:
        hyp = hypotheses.get(hypothesis_id)
        if hyp is None:
            return None
        hyp.reinforce(evidence)
        return hyp

    def weaken(
        self, hypotheses: dict[str, Hypothesis], hypothesis_id: str, evidence: str
    ) -> Hypothesis | None:
        hyp = hypotheses.get(hypothesis_id)
        if hyp is None:
            return None
        hyp.weaken(evidence)
        return hyp

    def invalidate(
        self, hypotheses: dict[str, Hypothesis], hypothesis_id: str, reason: str
    ) -> Hypothesis | None:
        hyp = hypotheses.get(hypothesis_id)
        if hyp is None:
            return None
        hyp.invalidate(reason)
        return hyp

    def confirm(self, hypotheses: dict[str, Hypothesis], hypothesis_id: str) -> Hypothesis | None:
        hyp = hypotheses.get(hypothesis_id)
        if hyp is None:
            return None
        hyp.confirm()
        return hyp

    def age(self, hypotheses: dict[str, Hypothesis], now: float | None = None) -> int:
        """Expire hypotheses that have gone stale without resolving.
        Returns the number expired.
        """
        now = now if now is not None else time.time()
        expired = 0
        for h in hypotheses.values():
            if not h.is_terminal() and h.is_stale(now):
                h.status = HypothesisStatus.EXPIRED
                h.last_updated_at = now
                expired += 1
        return expired

    def active(self, hypotheses: dict[str, Hypothesis]) -> list[Hypothesis]:
        return [h for h in hypotheses.values() if not h.is_terminal()]

    def dominant_direction(self, hypotheses: dict[str, Hypothesis]) -> BeliefDirection:
        """Confidence-weighted vote across active hypotheses."""
        active = self.active(hypotheses)
        if not active:
            return BeliefDirection.NEUTRAL

        scores = {BeliefDirection.BULLISH: 0.0, BeliefDirection.BEARISH: 0.0}
        for h in active:
            if h.direction in scores:
                scores[h.direction] += h.confidence

        if scores[BeliefDirection.BULLISH] == scores[BeliefDirection.BEARISH]:
            return BeliefDirection.NEUTRAL
        return max(scores, key=lambda k: scores[k])
