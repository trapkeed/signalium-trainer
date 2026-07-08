"""Phase 28.3 — Capability 4: Risk Reasoning.

Computes stop loss placement, take profit targets, and risk environment
classification for a given setup candidate.

CONTRACT
--------
- Returns RiskAssessment (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from trainer.qde.capabilities.cap_models import (
    Direction,
    MarketContext,
    MarketSnapshot,
    RiskAssessment,
    SetupCandidate,
)


class RiskReasoningCapability:
    """Calculates risk parameters for a potential entry.

    This capability contributes to the QDE's Entry, Position Size, and
    Capital Preservation responsibilities by computing valid stop levels
    and risk environment classification.
    """

    CAPABILITY_NAME = "risk_reasoning"

    # Default ATR multiple for stop placement
    DEFAULT_ATR_STOP_MULT = 1.5

    # Minimum R/R for take profit placement
    MIN_RR = 1.5

    def assess(
        self,
        snapshot: MarketSnapshot,
        candidate: SetupCandidate,
        market_context: MarketContext | None = None,
        risk_per_trade_pct: float = 0.01,
        atr_stop_multiple: float | None = None,
    ) -> RiskAssessment:
        """Calculate risk parameters for the given setup candidate.

        Args:
            snapshot: Current market state.
            candidate: The setup candidate to assess risk for.
            market_context: Optional context for environment classification.
            risk_per_trade_pct: Maximum fraction of equity to risk (default 1%).
            atr_stop_multiple: ATR multiple for stop distance (overrides default).

        Returns:
            RiskAssessment — advisory only, not a trade instruction.
        """
        warnings: list[str] = []
        atr = snapshot.atr_14
        mult = atr_stop_multiple or self._volatility_adjusted_mult(snapshot, market_context)

        # Prefer structural stop from candidate if available
        structural_valid = False
        if candidate.initial_stop_estimate > 0 and atr > 0:
            structural_dist = abs(snapshot.mid - candidate.initial_stop_estimate)
            atr_dist = atr * mult
            # Use the larger of structural and ATR stop for safety
            stop_distance = max(structural_dist, atr_dist * 0.5)
            structural_valid = True
        elif atr > 0:
            stop_distance = atr * mult
        else:
            stop_distance = snapshot.spread_pips * 3 * 0.0001  # fallback: 3× spread
            warnings.append("ATR unavailable — using spread-based stop estimate")

        if candidate.direction == Direction.LONG:
            stop_price = snapshot.mid - stop_distance
            tp_price = snapshot.mid + stop_distance * self.MIN_RR
        elif candidate.direction == Direction.SHORT:
            stop_price = snapshot.mid + stop_distance
            tp_price = snapshot.mid - stop_distance * self.MIN_RR
        else:
            stop_price = 0.0
            tp_price = None

        # Convert price distance to pips (assumes 5-decimal pair: 1 pip = 0.0001)
        pip_value = 0.0001
        stop_pips = stop_distance / pip_value if pip_value > 0 else 0.0
        tp_pips = stop_pips * self.MIN_RR

        risk_env = self._classify_risk_environment(snapshot, market_context)

        # Widen stop in high-risk environments
        if risk_env in ("HIGH", "EXTREME"):
            stop_pips *= 1.25
            tp_pips *= 1.25
            warnings.append(f"Risk environment {risk_env}: stops widened 25%")

        confidence = self._compute_confidence(structural_valid, atr, risk_env)

        return RiskAssessment(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
            stop_distance_pips=stop_pips,
            tp_distance_pips=tp_pips,
            atr_stop_multiple=mult,
            risk_per_trade_pct=risk_per_trade_pct,
            risk_environment=risk_env,
            volatility_adjusted=True,
            structural_stop_valid=structural_valid,
            notes=f"Stop={stop_pips:.1f}pips TP={tp_pips:.1f}pips env={risk_env}",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _volatility_adjusted_mult(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext | None,
    ) -> float:
        if ctx is None:
            return self.DEFAULT_ATR_STOP_MULT
        vol = ctx.volatility_state
        adjustments = {
            "LOW": 1.0,
            "NORMAL": 1.5,
            "HIGH": 2.0,
            "EXTREME": 2.5,
        }
        return adjustments.get(vol, self.DEFAULT_ATR_STOP_MULT)

    def _classify_risk_environment(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext | None,
    ) -> str:
        if ctx is None:
            return "NORMAL"
        vol = ctx.volatility_state
        spread_wide = snap.spread_pips > 3.0
        if vol == "EXTREME" or (vol == "HIGH" and spread_wide):
            return "EXTREME"
        if vol == "HIGH":
            return "HIGH"
        if vol == "LOW" and not spread_wide:
            return "LOW"
        return "NORMAL"

    def _compute_confidence(
        self,
        structural: bool,
        atr: float,
        env: str,
    ) -> float:
        confidence = 0.70
        if structural:
            confidence += 0.15
        if atr == 0:
            confidence -= 0.20
        env_penalty = {"LOW": 0.0, "NORMAL": 0.0, "HIGH": -0.10, "EXTREME": -0.20}
        confidence += env_penalty.get(env, 0.0)
        return max(0.1, min(1.0, confidence))
