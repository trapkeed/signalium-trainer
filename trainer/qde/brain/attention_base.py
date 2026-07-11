"""QDE Brain — attention allocation.

Every enabled symbol is always ingested continuously — attention here is a
*prioritization* signal (what deserves deeper reflection / faster review
cadence / dashboard prominence), never a gate on whether a symbol's market
updates get processed at all.
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain_models import AttentionAllocation


class AttentionAllocator:
    """Computes a normalized attention weight per symbol from salience
    inputs the Brain already has on hand (readiness, volatility quality,
    hypothesis activity, recent experience significance).
    """

    def salience(
        self,
        readiness_score: float,
        volatility_quality: float,
        active_hypothesis_count: int,
        max_hypothesis_confidence: float,
        recent_experience_magnitude: float = 0.0,
    ) -> float:
        """A single 0-1+ salience score — higher means "more deserving of
        QDE's limited attention budget right now".
        """
        hypothesis_signal = min(1.0, active_hypothesis_count * 0.15) * 0.5 + (
            max_hypothesis_confidence * 0.5
        )
        return max(
            0.0,
            0.35 * readiness_score
            + 0.25 * volatility_quality
            + 0.25 * hypothesis_signal
            + 0.15 * min(1.0, recent_experience_magnitude),
        )

    def reallocate(
        self, salience_by_instrument: dict[str, float], now: float | None = None
    ) -> dict[str, AttentionAllocation]:
        """Normalize raw salience scores into weights summing to 1.0 across
        all currently-tracked instruments (softmax-free simple normalization
        — kept linear/interpretable rather than exponential so a single
        very salient symbol can't starve every other symbol's telemetry
        weight to near-zero).
        """
        now = now if now is not None else time.time()
        total = sum(salience_by_instrument.values())
        allocations: dict[str, AttentionAllocation] = {}

        for instrument, salience in salience_by_instrument.items():
            weight = (
                (salience / total) if total > 1e-9 else (1.0 / max(1, len(salience_by_instrument)))
            )
            allocations[instrument] = AttentionAllocation(
                instrument=instrument,
                weight=weight,
                salience=salience,
                reason=self._explain(salience, weight),
                updated_at=now,
            )

        return allocations

    def _explain(self, salience: float, weight: float) -> str:
        if weight >= 0.3:
            return "high-priority: strong readiness/hypothesis activity"
        if weight >= 0.12:
            return "moderate priority"
        return "low priority: quiet/observational"
