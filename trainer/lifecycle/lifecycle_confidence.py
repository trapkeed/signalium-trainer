"""Phase 21.5 — Lifecycle Confidence Engine.

Computes per-state confidence, expected duration, expected reward/risk,
transition probability, and failure probability.
"""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_models import LifecycleConfidence, MarketContext
from trainer.lifecycle.lifecycle_states import STATE_EXPECTED_DURATION_HOURS, LifecycleState


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class LifecycleConfidenceEngine:
    """Computes confidence and forward projections for any lifecycle state."""

    # Base transition probabilities per state (empirical defaults)
    _BASE_TRANSITION_PROB: dict[LifecycleState, float] = {
        LifecycleState.IDEA: 0.60,
        LifecycleState.WATCHLIST: 0.65,
        LifecycleState.PREPARATION: 0.70,
        LifecycleState.ENTRY_READY: 0.80,
        LifecycleState.PENDING_ENTRY: 0.85,
        LifecycleState.ACTIVE: 0.75,
        LifecycleState.POSITION_BUILDING: 0.70,
        LifecycleState.PROTECTED: 0.85,
        LifecycleState.PARTIAL_EXIT: 0.90,
        LifecycleState.RUNNER: 0.70,
        LifecycleState.TRAILING: 0.80,
        LifecycleState.EXIT_READY: 0.95,
        LifecycleState.CLOSING: 0.99,
        LifecycleState.CLOSED: 1.00,
        LifecycleState.INVALIDATED: 1.00,
        LifecycleState.CANCELLED: 1.00,
        LifecycleState.EXPIRED: 1.00,
        LifecycleState.STOPPED: 1.00,
        LifecycleState.FORCED_EXIT: 1.00,
        LifecycleState.EMERGENCY_EXIT: 1.00,
    }

    # Base failure probability per state
    _BASE_FAILURE_PROB: dict[LifecycleState, float] = {
        LifecycleState.IDEA: 0.40,
        LifecycleState.WATCHLIST: 0.30,
        LifecycleState.PREPARATION: 0.20,
        LifecycleState.ENTRY_READY: 0.15,
        LifecycleState.PENDING_ENTRY: 0.10,
        LifecycleState.ACTIVE: 0.25,
        LifecycleState.POSITION_BUILDING: 0.20,
        LifecycleState.PROTECTED: 0.05,
        LifecycleState.PARTIAL_EXIT: 0.05,
        LifecycleState.RUNNER: 0.20,
        LifecycleState.TRAILING: 0.15,
        LifecycleState.EXIT_READY: 0.02,
        LifecycleState.CLOSING: 0.01,
        LifecycleState.CLOSED: 0.00,
        LifecycleState.INVALIDATED: 0.00,
        LifecycleState.CANCELLED: 0.00,
        LifecycleState.EXPIRED: 0.00,
        LifecycleState.STOPPED: 0.00,
        LifecycleState.FORCED_EXIT: 0.00,
        LifecycleState.EMERGENCY_EXIT: 0.00,
    }

    # Expected reward fraction per state (how much of potential reward remains)
    _BASE_REWARD: dict[LifecycleState, float] = {
        LifecycleState.IDEA: 0.08,
        LifecycleState.WATCHLIST: 0.08,
        LifecycleState.PREPARATION: 0.08,
        LifecycleState.ENTRY_READY: 0.08,
        LifecycleState.PENDING_ENTRY: 0.08,
        LifecycleState.ACTIVE: 0.07,
        LifecycleState.POSITION_BUILDING: 0.09,
        LifecycleState.PROTECTED: 0.06,  # partial already taken
        LifecycleState.PARTIAL_EXIT: 0.04,
        LifecycleState.RUNNER: 0.05,
        LifecycleState.TRAILING: 0.03,
        LifecycleState.EXIT_READY: 0.01,
        LifecycleState.CLOSING: 0.005,
        LifecycleState.CLOSED: 0.00,
        LifecycleState.STOPPED: -0.015,
        LifecycleState.EMERGENCY_EXIT: -0.025,
        LifecycleState.INVALIDATED: 0.00,
        LifecycleState.CANCELLED: 0.00,
        LifecycleState.EXPIRED: 0.00,
        LifecycleState.FORCED_EXIT: -0.01,
    }

    def compute(
        self,
        state: LifecycleState,
        confidence: float,
        context: MarketContext | None = None,
        expected_return: float = 0.08,
        expected_risk: float = 0.10,
    ) -> LifecycleConfidence:

        notes: list[str] = []

        # Base values
        base_trans = self._BASE_TRANSITION_PROB.get(state, 0.65)
        base_fail = self._BASE_FAILURE_PROB.get(state, 0.25)
        base_reward = self._BASE_REWARD.get(state, expected_return)
        exp_dur = STATE_EXPECTED_DURATION_HOURS.get(state, 24.0)

        # Adjust by confidence
        conf_adj = _clamp(confidence)
        trans_prob = _clamp(base_trans * (0.7 + 0.3 * conf_adj))
        fail_prob = _clamp(base_fail * (1.0 + 0.5 * (1.0 - conf_adj)))

        # Adjust by market context
        if context:
            # News risk increases failure probability
            if context.news_risk > 0.50:
                fail_prob = _clamp(fail_prob + context.news_risk * 0.15)
                notes.append(f"Elevated news risk ({context.news_risk:.0%}) increases failure prob")

            # Poor spread → longer duration estimates
            if context.spread > context.atr * 0.25 and context.atr > 0:
                exp_dur *= 1.20
                notes.append("Wide spread may extend time in state")

            # High volatility → shorter expected duration (faster resolution)
            if context.volatility > 0.020:
                exp_dur = max(exp_dur * 0.70, 0.5)
                notes.append("High volatility — faster state resolution expected")

            # Bad session → lower transition probability
            if context.session == "CLOSED":
                trans_prob = _clamp(trans_prob * 0.50)
                notes.append("Closed session — transitions less likely now")

        eff_confidence = _clamp(0.5 * conf_adj + 0.5 * trans_prob)

        exp_risk = (
            expected_risk
            if state
            in (
                LifecycleState.IDEA,
                LifecycleState.WATCHLIST,
                LifecycleState.PREPARATION,
                LifecycleState.ENTRY_READY,
                LifecycleState.PENDING_ENTRY,
                LifecycleState.ACTIVE,
            )
            else expected_risk * 0.5
        )

        logger.debug(
            f"LifecycleConfidence [{state.value}]: conf={eff_confidence:.3f} "
            f"trans={trans_prob:.3f} fail={fail_prob:.3f}"
        )

        return LifecycleConfidence(
            state=state,
            confidence=round(eff_confidence, 4),
            expected_duration_h=round(exp_dur, 2),
            expected_reward=round(base_reward, 4),
            expected_risk=round(exp_risk, 4),
            transition_prob=round(trans_prob, 4),
            failure_prob=round(fail_prob, 4),
            notes=notes,
        )
