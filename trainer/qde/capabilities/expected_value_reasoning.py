"""Phase 28.3 — Capability 3: Expected Value Reasoning.

Calculates the expected value of a potential setup candidate.
All outputs are advisory scores — no trade instructions.

CONTRACT
--------
- Returns EVScore (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from trainer.qde.capabilities.cap_models import (
    EVScore,
    MarketContext,
    MarketSnapshot,
    MemoryContext,
    SetupCandidate,
)


class ExpectedValueReasoningCapability:
    """Computes expected value for a setup candidate.

    EV = (win_probability × expected_gain) − (loss_probability × expected_loss)

    This capability contributes to the QDE's Entry and Trade Duration
    responsibilities by quantifying whether a setup is worth pursuing.
    """

    CAPABILITY_NAME = "expected_value_reasoning"

    # Minimum EV threshold (in account currency units per lot) to flag as viable
    MIN_EV_THRESHOLD = 0.0

    # Minimum risk/reward ratio to consider positive EV reliable
    MIN_RR = 1.5

    def score(
        self,
        snapshot: MarketSnapshot,
        candidate: SetupCandidate,
        risk_assessment: "RiskAssessment | None" = None,  # type: ignore[name-defined]
        memory: MemoryContext | None = None,
        market_context: MarketContext | None = None,
        base_win_probability: float = 0.50,
    ) -> EVScore:
        """Compute expected value for a setup candidate.

        Args:
            snapshot: Current market snapshot.
            candidate: The setup candidate to evaluate.
            risk_assessment: Optional risk output for stop/tp levels.
            memory: Optional historical memory — adjusts win probability.
            market_context: Optional market context for regime adjustment.
            base_win_probability: Starting win probability prior.

        Returns:
            EVScore — advisory only, not a trade instruction.
        """
        win_prob = base_win_probability

        # Adjust win probability from confluence score
        win_prob = self._adjust_for_confluence(win_prob, candidate.confluence_score)

        # Adjust from historical memory if available
        if memory and memory.historical_win_rate_this_setup is not None:
            win_prob = self._blend_with_memory(win_prob, memory)

        # Adjust for regime
        if market_context:
            win_prob = self._adjust_for_regime(win_prob, market_context)

        win_prob = max(0.05, min(0.95, win_prob))
        loss_prob = 1.0 - win_prob

        # Compute expected gain and loss in pip-equivalent units
        stop_dist = 0.0
        tp_dist = 0.0
        if risk_assessment is not None:
            stop_dist = risk_assessment.stop_distance_pips
            tp_dist = risk_assessment.tp_distance_pips or stop_dist * self.MIN_RR
        elif candidate.initial_stop_estimate > 0 and snapshot.mid > 0:
            stop_dist = abs(snapshot.mid - candidate.initial_stop_estimate)
            tp_dist = stop_dist * self.MIN_RR

        # Normalise to per-unit (pips)
        expected_gain = win_prob * tp_dist
        expected_loss = loss_prob * stop_dist
        ev = expected_gain - expected_loss
        ev_per_unit = ev
        rr = tp_dist / stop_dist if stop_dist > 0 else 0.0

        ev_positive = ev > self.MIN_EV_THRESHOLD
        threshold_met = ev_positive and rr >= self.MIN_RR and win_prob >= 0.40

        warnings: list[str] = []
        if rr < self.MIN_RR:
            warnings.append(f"R/R {rr:.2f} below minimum {self.MIN_RR}")
        if win_prob < 0.40:
            warnings.append(f"Win probability {win_prob:.1%} is below 40%")
        if not ev_positive:
            warnings.append("Expected value is negative")

        confidence = min(1.0, candidate.confluence_score * (0.5 + win_prob * 0.5))

        return EVScore(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            ev=ev,
            ev_per_unit=ev_per_unit,
            win_probability=win_prob,
            loss_probability=loss_prob,
            expected_gain=expected_gain,
            expected_loss=expected_loss,
            risk_reward_ratio=rr,
            ev_is_positive=ev_positive,
            ev_threshold_met=threshold_met,
            setup_candidate_id=candidate.setup_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _adjust_for_confluence(self, base: float, confluence: float) -> float:
        """Higher confluence nudges win probability up."""
        # confluence 0.5 → no change; 1.0 → +5%; 0.0 → −5%
        delta = (confluence - 0.5) * 0.10
        return base + delta

    def _blend_with_memory(self, base: float, memory: MemoryContext) -> float:
        """Blend base probability with historical win rate from memory."""
        hist = memory.historical_win_rate_this_setup
        if hist is None:
            return base
        quality = memory.pattern_match_quality
        return base * (1 - quality * 0.4) + hist * (quality * 0.4)

    def _adjust_for_regime(self, base: float, ctx: MarketContext) -> float:
        """Adjust win probability based on regime suitability."""
        from trainer.qde.capabilities.cap_models import RegimeType
        regime_adjustments = {
            RegimeType.TRENDING_BULL: 0.04,
            RegimeType.TRENDING_BEAR: 0.04,
            RegimeType.RANGING: -0.02,
            RegimeType.HIGH_VOLATILITY: -0.06,
            RegimeType.LOW_VOLATILITY: -0.03,
            RegimeType.CRISIS: -0.12,
            RegimeType.RECOVERY: 0.02,
            RegimeType.UNKNOWN: 0.0,
        }
        return base + regime_adjustments.get(ctx.regime, 0.0)
