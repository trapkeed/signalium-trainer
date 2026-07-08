"""Phase 28.3 — Capability 7: Position Management.

Calculates optimal position sizes based on account equity, risk budget,
volatility, and Kelly criterion.

CONTRACT
--------
- Returns PositionSizeRecommendation (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

import math

from trainer.qde.capabilities.cap_models import (
    EVScore,
    MarketContext,
    MarketSnapshot,
    PositionSizeRecommendation,
    RiskAssessment,
)


class PositionManagementCapability:
    """Computes position size recommendations.

    This capability contributes to the QDE's Position Size and Scaling
    responsibilities by computing safe lot sizes given the current
    risk budget, stop distance, and volatility regime.

    It never allocates capital — only recommends what size would be
    consistent with the configured risk parameters.
    """

    CAPABILITY_NAME = "position_management"

    # Default risk per trade as fraction of equity
    DEFAULT_RISK_PCT = 0.01

    # Maximum risk per trade (hard cap)
    MAX_RISK_PCT = 0.02

    # Kelly fraction cap (use half-Kelly at most)
    KELLY_CAP = 0.5

    # Minimum lot size
    MIN_LOT = 0.01

    # Pip value per lot (standard lot, 5-decimal pair)
    PIP_VALUE_PER_LOT = 10.0  # USD per pip per standard lot (EURUSD approx)

    def compute(
        self,
        snapshot: MarketSnapshot,
        risk_assessment: RiskAssessment,
        ev_score: EVScore | None = None,
        market_context: MarketContext | None = None,
        equity: float = 10000.0,
        risk_pct: float | None = None,
    ) -> PositionSizeRecommendation:
        """Compute the recommended position size.

        Args:
            snapshot: Current market snapshot.
            risk_assessment: Output from RiskReasoningCapability.
            ev_score: Optional EV score for Kelly-adjusted sizing.
            market_context: Optional context for volatility adjustment.
            equity: Account equity in account currency.
            risk_pct: Override for risk per trade fraction.

        Returns:
            PositionSizeRecommendation — advisory only, not a trade instruction.
        """
        warnings: list[str] = []

        base_risk_pct = min(risk_pct or self.DEFAULT_RISK_PCT, self.MAX_RISK_PCT)

        # Volatility adjustment: reduce size in high-volatility environments
        vol_factor = self._volatility_factor(market_context, risk_assessment)
        effective_risk_pct = base_risk_pct * vol_factor

        risk_amount = equity * effective_risk_pct
        stop_pips = risk_assessment.stop_distance_pips

        # Lot size from fixed-risk method
        if stop_pips > 0 and self.PIP_VALUE_PER_LOT > 0:
            lot_fixed = risk_amount / (stop_pips * self.PIP_VALUE_PER_LOT)
        else:
            lot_fixed = self.MIN_LOT
            warnings.append("Stop pips is zero — using minimum lot size")

        # Kelly-adjusted sizing
        kelly_fraction = 0.0
        kelly_adjusted = 0.0
        if ev_score and ev_score.win_probability > 0 and ev_score.risk_reward_ratio > 0:
            p = ev_score.win_probability
            q = 1.0 - p
            b = ev_score.risk_reward_ratio
            kelly_fraction = (p * b - q) / b if b > 0 else 0.0
            kelly_fraction = max(0.0, kelly_fraction)
            kelly_adjusted = min(kelly_fraction, self.KELLY_CAP) * base_risk_pct
            # Blend: 70% fixed-risk, 30% Kelly
            effective_risk_pct = effective_risk_pct * 0.70 + kelly_adjusted * 0.30
            risk_amount = equity * effective_risk_pct
            if stop_pips > 0:
                lot_fixed = risk_amount / (stop_pips * self.PIP_VALUE_PER_LOT)

        # Hard cap
        max_lot = (equity * self.MAX_RISK_PCT) / (max(stop_pips, 1) * self.PIP_VALUE_PER_LOT)
        lot_size = min(max(self.MIN_LOT, round(lot_fixed, 2)), max_lot)

        if lot_size == self.MIN_LOT:
            warnings.append("Position size capped at minimum lot")

        sizing_method = "KELLY" if kelly_fraction > 0 else "FIXED_RISK"
        if market_context and market_context.volatility_state in ("HIGH", "EXTREME"):
            sizing_method = "ATR_SCALED"

        confidence = risk_assessment.confidence * (0.6 + vol_factor * 0.4)

        return PositionSizeRecommendation(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=min(1.0, confidence),
            warnings=warnings,
            lot_size=lot_size,
            risk_amount=round(risk_amount, 2),
            risk_pct=effective_risk_pct,
            max_lot_size=max_lot,
            volatility_adjustment_factor=vol_factor,
            kelly_fraction=kelly_fraction,
            kelly_adjusted=kelly_adjusted,
            sizing_method=sizing_method,
            notes=f"lot={lot_size:.2f} risk={effective_risk_pct:.2%} vol_factor={vol_factor:.2f}",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _volatility_factor(
        self,
        ctx: MarketContext | None,
        risk: RiskAssessment,
    ) -> float:
        """Reduce position size in high-volatility regimes."""
        env = risk.risk_environment
        if env == "EXTREME":
            return 0.40
        if env == "HIGH":
            return 0.65
        if env == "LOW":
            return 1.10   # slight size increase in calm markets
        if ctx and ctx.volatility_state == "EXTREME":
            return 0.40
        if ctx and ctx.volatility_state == "HIGH":
            return 0.65
        return 1.0
