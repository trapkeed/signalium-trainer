"""Phase 28.3 — Capability 10: Reflection.

Analyses completed trade outcomes and produces insights about whether
the QDE's reasoning was correct, which capabilities were accurate,
and what lessons can be extracted.

CONTRACT
--------
- Returns ReflectionInsight (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from trainer.qde.capabilities.cap_models import (
    ReflectionInsight,
    TradeOutcome,
)


class ReflectionCapability:
    """Reflects on completed trade outcomes to generate learning signals.

    This capability has no QDE responsibility for live market decisions.
    It operates post-trade — it is called after a trade closes, not during
    the live reasoning cycle.

    Its output feeds the Learning capability, which uses it to update priors.
    """

    CAPABILITY_NAME = "reflection"

    # Threshold above which we consider the trade outcome "correct"
    CORRECT_OUTCOME_R_THRESHOLD = 0.0  # breakeven or better

    def reflect(
        self,
        outcome: TradeOutcome,
        predicted_win_probability: float = 0.5,
        predicted_ev: float = 0.0,
    ) -> ReflectionInsight:
        """Analyse a completed trade outcome.

        Args:
            outcome: Completed trade data including entry, exit, P&L.
            predicted_win_probability: What the EV capability predicted.
            predicted_ev: What the EV capability computed.

        Returns:
            ReflectionInsight — advisory only, not a trade instruction.
        """
        was_correct = outcome.pnl_r >= self.CORRECT_OUTCOME_R_THRESHOLD
        actual_ev = outcome.pnl_r  # R-multiple is the realised EV in normalised units

        # EV prediction error (can be positive or negative)
        ev_prediction_error = predicted_ev - actual_ev

        # Score per capability based on trade outcome
        cap_accuracy = self._score_capabilities(outcome, was_correct)

        # Stop placement quality: did the stop get hit prematurely?
        stop_quality = self._assess_stop_quality(outcome)

        # Regime reading quality: was the regime correctly identified?
        regime_quality = self._assess_regime_quality(outcome, was_correct)

        # What confluence score should have been assigned?
        actual_confluence = self._compute_actual_confluence(outcome)

        # What weight adjustments does this outcome suggest?
        weight_adjustments = self._compute_weight_adjustments(outcome, cap_accuracy)

        key_lesson = self._extract_lesson(outcome, was_correct, stop_quality, ev_prediction_error)

        confidence = 0.85  # Reflection is deterministic — high confidence

        return ReflectionInsight(
            capability=self.CAPABILITY_NAME,
            snapshot_id=f"post-trade-{outcome.trade_id}",
            confidence=confidence,
            trade_id=outcome.trade_id,
            reasoning_was_correct=was_correct,
            capability_accuracy=cap_accuracy,
            setup_quality_actual=actual_confluence,
            ev_prediction_error=ev_prediction_error,
            stop_placement_quality=stop_quality,
            regime_reading_quality=regime_quality,
            key_lesson=key_lesson,
            suggested_weight_adjustments=weight_adjustments,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _score_capabilities(
        self,
        outcome: TradeOutcome,
        was_correct: bool,
    ) -> dict[str, float]:
        """Score each capability's contribution accuracy."""
        scores = {}
        prev_scores = outcome.capability_scores_at_entry

        for cap, score in prev_scores.items():
            if was_correct:
                # High-confidence-and-correct = full credit
                # Low-confidence-and-correct = partial credit
                accuracy = 0.5 + score * 0.5
            else:
                # High-confidence-and-wrong = penalty
                accuracy = 0.5 - score * 0.5
            scores[cap] = max(0.0, min(1.0, accuracy))

        return scores

    def _assess_stop_quality(self, outcome: TradeOutcome) -> float:
        """Assess stop placement quality from 0–1."""
        # If the stop was not hit (position closed by TP or discretion), quality is moderate-high
        if outcome.exit_reason != "STOP_LOSS":
            return 0.80
        # Stop was hit — it could be a good stop (loss is small) or bad (loss is large)
        loss_r = abs(outcome.pnl_r)
        if loss_r <= 1.1:
            return 0.70  # stopped out at approximately 1R — acceptable
        elif loss_r <= 1.5:
            return 0.50  # stopped out at 1.5R — stop was too wide or slippage
        else:
            return 0.20  # stopped out at >1.5R — serious stop placement issue

    def _assess_regime_quality(self, outcome: TradeOutcome, was_correct: bool) -> float:
        """Assess regime reading quality based on outcome."""
        if was_correct:
            return 0.85
        # Loss in a regime that's usually profitable = regime mis-read
        high_prob_regimes = {"TRENDING_BULL", "TRENDING_BEAR"}
        if outcome.regime_at_entry in high_prob_regimes:
            return 0.35  # should have won in this regime
        return 0.60  # loss in ranging/volatile regime — partially expected

    def _compute_actual_confluence(self, outcome: TradeOutcome) -> float:
        """What confluence score would have been accurate in hindsight?"""
        if outcome.pnl_r > 2.0:
            return 0.90  # great trade — confluence was underestimated
        elif outcome.pnl_r > 1.0:
            return 0.75
        elif outcome.pnl_r > 0.0:
            return 0.60
        elif outcome.pnl_r > -1.0:
            return 0.40
        else:
            return 0.20  # bad trade — confluence was overestimated

    def _compute_weight_adjustments(
        self,
        outcome: TradeOutcome,
        cap_accuracy: dict[str, float],
    ) -> dict[str, float]:
        """Suggest weight adjustments for each capability based on accuracy."""
        adjustments = {}
        for cap, acc in cap_accuracy.items():
            if acc > 0.70:
                adjustments[cap] = 0.01  # increase trust
            elif acc < 0.40:
                adjustments[cap] = -0.01  # decrease trust
            else:
                adjustments[cap] = 0.0
        return adjustments

    def _extract_lesson(
        self,
        outcome: TradeOutcome,
        was_correct: bool,
        stop_quality: float,
        ev_error: float,
    ) -> str:
        if was_correct and stop_quality > 0.70:
            return "Reasoning and stop placement were sound. Outcome consistent with model."
        if not was_correct and stop_quality < 0.40:
            return f"Stop placement quality was poor ({stop_quality:.0%}). Revisit stop logic for {outcome.setup_type} in {outcome.regime_at_entry} regime."
        if abs(ev_error) > 1.0:
            direction = "over" if ev_error > 0 else "under"
            return f"EV was {direction}estimated by {abs(ev_error):.2f}R for {outcome.setup_type} setup. Recalibrate win probability priors."
        if not was_correct:
            return f"Loss in {outcome.regime_at_entry} on {outcome.setup_type}. Review capability confidence for this regime."
        return "Outcome within expected range."
