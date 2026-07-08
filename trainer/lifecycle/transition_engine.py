"""Phase 21.5 — Transition Engine.

Continuously evaluates all valid transitions from the current state
and recommends the best next state with rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_models import MarketContext, TransitionResult
from trainer.lifecycle.lifecycle_states import VALID_TRANSITIONS, LifecycleState, is_terminal
from trainer.lifecycle.state_machine import StateMachine


@dataclass
class TransitionRecommendation:
    """The engine's best recommendation for next state."""

    current_state: LifecycleState
    recommended: LifecycleState | None  # None if stay
    confidence: float
    reason: str
    all_candidates: list[TransitionResult] = field(default_factory=list)
    should_stay: bool = False
    stay_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "current_state": self.current_state.value,
            "recommended": self.recommended.value if self.recommended else None,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "should_stay": self.should_stay,
            "stay_reason": self.stay_reason,
            "candidates": [c.to_dict() for c in self.all_candidates],
        }


class TransitionEngine:
    """Pure context provider — evaluates all valid transitions from a given
    state and returns a recommendation for the decision layer (QDE).

    This engine never applies a transition itself; it only ranks candidates
    and returns a `TransitionRecommendation`. The prioritised ordering below
    (emergency exits first, etc.) reflects the engine's advisory ranking
    heuristic, not a binding directive — QDE decides whether to act on it.

    Ranking heuristic:
    1. Emergency exits (news_risk > 0.80, extreme vol)
    2. Negative exits (STOPPED, FORCED_EXIT) if conditions warrant
    3. Forward progression (highest expected_benefit × confidence)
    4. Demotion (WATCHLIST ← PREPARATION) if conditions deteriorate
    5. Stay current state
    """

    def __init__(self):
        self._sm = StateMachine()

    def evaluate(
        self,
        current_state: LifecycleState,
        confidence: float = 0.65,
        context: MarketContext | None = None,
    ) -> TransitionRecommendation:

        if is_terminal(current_state):
            return TransitionRecommendation(
                current_state=current_state,
                recommended=None,
                confidence=1.0,
                reason=f"{current_state.value} is a terminal state",
                should_stay=True,
                stay_reason="Terminal — no further transitions",
            )

        candidates: list[TransitionResult] = []

        for next_state in VALID_TRANSITIONS.get(current_state, frozenset()):
            result = self._sm.validate_transition(current_state, next_state, context, confidence)
            candidates.append(result)

        # Filter for allowed transitions
        allowed = [c for c in candidates if c.allowed]

        # Check emergency first
        if context and context.news_risk > 0.80:
            emerg = next((c for c in allowed if c.to_state == LifecycleState.EMERGENCY_EXIT), None)
            if emerg:
                logger.warning(
                    f"TransitionEngine: EMERGENCY_EXIT recommended "
                    f"news_risk={context.news_risk}"
                )
                return TransitionRecommendation(
                    current_state=current_state,
                    recommended=LifecycleState.EMERGENCY_EXIT,
                    confidence=emerg.confidence,
                    reason="Critical news risk — emergency exit recommended",
                    all_candidates=candidates,
                )

        # Check extreme volatility
        if context and context.volatility > 0.035:
            emerg = next((c for c in allowed if c.to_state == LifecycleState.EMERGENCY_EXIT), None)
            if emerg:
                return TransitionRecommendation(
                    current_state=current_state,
                    recommended=LifecycleState.EMERGENCY_EXIT,
                    confidence=emerg.confidence,
                    reason=f"Extreme volatility {context.volatility:.3f} — emergency exit",
                    all_candidates=candidates,
                )

        # Invaliation: confidence dropped or trend invalid
        if confidence < 0.40 or (
            context
            and not context.trend_valid
            and current_state
            in (
                LifecycleState.WATCHLIST,
                LifecycleState.PREPARATION,
            )
        ):
            inval = next((c for c in candidates if c.to_state == LifecycleState.INVALIDATED), None)
            if inval:
                return TransitionRecommendation(
                    current_state=current_state,
                    recommended=LifecycleState.INVALIDATED,
                    confidence=0.90,
                    reason="Setup invalidated — confidence too low or trend broken",
                    all_candidates=candidates,
                )

        if not allowed:
            # Nothing allowed — stay
            return TransitionRecommendation(
                current_state=current_state,
                recommended=None,
                confidence=confidence,
                reason="No transition conditions met",
                all_candidates=candidates,
                should_stay=True,
                stay_reason="Conditions not yet satisfied for any forward transition",
            )

        # Rank allowed by expected_benefit × confidence (skip negative outcomes for now)
        positive = [
            c
            for c in allowed
            if c.to_state
            not in (
                LifecycleState.STOPPED,
                LifecycleState.FORCED_EXIT,
                LifecycleState.EMERGENCY_EXIT,
                LifecycleState.INVALIDATED,
                LifecycleState.CANCELLED,
                LifecycleState.EXPIRED,
            )
        ]

        if positive:
            best = max(positive, key=lambda c: c.expected_benefit * c.confidence)
        else:
            best = max(allowed, key=lambda c: c.confidence)

        logger.debug(
            f"TransitionEngine: {current_state.value} → {best.to_state.value} "
            f"(conf={best.confidence:.3f} benefit={best.expected_benefit:.3f})"
        )

        return TransitionRecommendation(
            current_state=current_state,
            recommended=best.to_state,
            confidence=best.confidence,
            reason=best.reason,
            all_candidates=candidates,
        )

    def possible_transitions(
        self,
        current_state: LifecycleState,
        confidence: float = 0.65,
        context: MarketContext | None = None,
    ) -> list[TransitionResult]:
        """Return all evaluated transitions (allowed and not) from current state."""
        results = []
        for next_state in VALID_TRANSITIONS.get(current_state, frozenset()):
            results.append(
                self._sm.validate_transition(current_state, next_state, context, confidence)
            )
        return results
