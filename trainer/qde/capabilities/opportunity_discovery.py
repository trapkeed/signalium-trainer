"""Phase 28.3 — Capability 2: Opportunity Discovery.

Scans the market snapshot for potential trade setups and returns scored
candidates. Does NOT decide to trade — only identifies and scores possibilities.

CONTRACT
--------
- Returns OpportunityReport (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from trainer.qde.capabilities.cap_models import (
    Direction,
    MarketContext,
    MarketSnapshot,
    OpportunityReport,
    RegimeType,
    SetupCandidate,
    TrendStrength,
)


class OpportunityDiscoveryCapability:
    """Discovers and scores trade setup candidates.

    This capability is a pattern scanner. It identifies setups that MIGHT
    warrant a trade — but the decision to trade belongs entirely to the QDE.

    This capability contributes to the QDE's Entry responsibility by providing
    the raw material (scored candidates) that the QDE synthesises.
    """

    CAPABILITY_NAME = "opportunity_discovery"

    # Minimum confluence score to be surfaced as a candidate
    MIN_CONFLUENCE = 0.35

    # Minimum ATR multiple for a meaningful stop distance
    MIN_STOP_ATR_MULT = 0.8

    def discover(
        self,
        snapshot: MarketSnapshot,
        market_context: MarketContext,
    ) -> OpportunityReport:
        """Discover and score all setup candidates in the current snapshot.

        Args:
            snapshot: Current market state.
            market_context: Output from MarketUnderstandingCapability.

        Returns:
            OpportunityReport — advisory only, no trade instructions.
        """
        candidates: list[SetupCandidate] = []
        warnings: list[str] = []

        # Don't discover opportunities in illiquid or very-wide spread conditions
        if market_context.spread_state == "VERY_WIDE":
            warnings.append("Spread too wide for opportunity discovery")
            return self._empty_report(snapshot, warnings)

        if market_context.volatility_state == "EXTREME":
            warnings.append("Volatility too extreme for reliable setup discovery")
            return self._empty_report(snapshot, warnings)

        # Trend continuation setups
        if market_context.trend_strength in (TrendStrength.STRONG, TrendStrength.MODERATE):
            c = self._build_trend_continuation(snapshot, market_context)
            if c and c.confluence_score >= self.MIN_CONFLUENCE:
                candidates.append(c)

        # Reversal setups at structural extremes
        reversal = self._build_reversal_candidate(snapshot, market_context)
        if reversal and reversal.confluence_score >= self.MIN_CONFLUENCE:
            candidates.append(reversal)

        # Breakout setups from ranging markets
        if market_context.regime == RegimeType.RANGING:
            bo = self._build_breakout_candidate(snapshot, market_context)
            if bo and bo.confluence_score >= self.MIN_CONFLUENCE:
                candidates.append(bo)

        # Sort by confluence score descending
        candidates.sort(key=lambda c: c.confluence_score, reverse=True)
        best = candidates[0] if candidates else None
        tradeable = best is not None and best.confluence_score >= 0.55

        quality = best.confluence_score if best else 0.0
        confidence = min(1.0, quality * market_context.confidence)

        return OpportunityReport(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            candidates=candidates,
            best_candidate=best,
            opportunity_quality=quality,
            market_is_tradeable=tradeable,
        )

    # ── Setup builders ────────────────────────────────────────────────────────

    def _build_trend_continuation(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext,
    ) -> SetupCandidate | None:
        direction = ctx.trend_direction
        signals: list[str] = []
        score = 0.0

        # EMA alignment
        if snap.ema_fast > 0 and snap.ema_slow > 0:
            if direction == Direction.LONG and snap.ema_fast > snap.ema_slow:
                signals.append("EMA_ALIGNED_BULL")
                score += 0.25
            elif direction == Direction.SHORT and snap.ema_fast < snap.ema_slow:
                signals.append("EMA_ALIGNED_BEAR")
                score += 0.25

        # RSI momentum
        rsi = snap.rsi_14
        if direction == Direction.LONG and 45 <= rsi <= 65:
            signals.append("RSI_TREND_ZONE")
            score += 0.20
        elif direction == Direction.SHORT and 35 <= rsi <= 55:
            signals.append("RSI_TREND_ZONE")
            score += 0.20

        # ADX trend strength confirmation
        if snap.adx_14 >= 25:
            signals.append(f"ADX_TREND_CONFIRM:{snap.adx_14:.0f}")
            score += 0.20

        # Volume confirmation
        if snap.volume_ratio > 1.2:
            signals.append("VOLUME_CONFIRM")
            score += 0.15

        # Price at pullback to fast EMA
        if snap.ema_fast > 0 and snap.atr_14 > 0:
            dist = abs(snap.mid - snap.ema_fast)
            if dist < snap.atr_14 * 0.5:
                signals.append("PRICE_AT_EMA_PULLBACK")
                score += 0.20

        if not signals or score < self.MIN_CONFLUENCE:
            return None

        stop_distance = snap.atr_14 * 1.5 if snap.atr_14 > 0 else 0.0
        if direction == Direction.LONG:
            entry_low = snap.mid * (1 - 0.0001)
            entry_high = snap.mid * (1 + 0.0003)
            stop_est = snap.mid - stop_distance
        else:
            entry_low = snap.mid * (1 - 0.0003)
            entry_high = snap.mid * (1 + 0.0001)
            stop_est = snap.mid + stop_distance

        return SetupCandidate(
            direction=direction,
            setup_type="TREND_CONTINUATION",
            confluence_score=min(1.0, score),
            signal_count=len(signals),
            signals=signals,
            ideal_entry_zone=(entry_low, entry_high),
            initial_stop_estimate=stop_est,
            notes=f"Trend continuation {direction.value}; {len(signals)} signals",
        )

    def _build_reversal_candidate(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext,
    ) -> SetupCandidate | None:
        signals: list[str] = []
        score = 0.0
        direction = Direction.FLAT

        rsi = snap.rsi_14

        # Oversold bounce
        if rsi < 30 and ctx.structural_level_below is not None:
            dist_to_support = abs(snap.mid - ctx.structural_level_below)
            if dist_to_support < snap.atr_14 * 0.5:
                signals.append("RSI_OVERSOLD")
                signals.append("PRICE_AT_SUPPORT")
                score += 0.40
                direction = Direction.LONG

        # Overbought reversal
        elif rsi > 70 and ctx.structural_level_above is not None:
            dist_to_resistance = abs(ctx.structural_level_above - snap.mid)
            if dist_to_resistance < snap.atr_14 * 0.5:
                signals.append("RSI_OVERBOUGHT")
                signals.append("PRICE_AT_RESISTANCE")
                score += 0.40
                direction = Direction.SHORT

        if direction == Direction.FLAT:
            return None

        # Bollinger band squeeze near extreme
        if snap.bb_width > 0 and snap.bb_width < 0.5:
            signals.append("BB_SQUEEZE")
            score += 0.15

        if snap.adx_14 < 20:
            signals.append("LOW_ADX_REVERSAL_CONTEXT")
            score += 0.15

        if score < self.MIN_CONFLUENCE or not signals:
            return None

        stop_distance = snap.atr_14 * 1.2 if snap.atr_14 > 0 else 0.0
        if direction == Direction.LONG:
            entry_zone = (snap.mid * (1 - 0.0002), snap.mid * (1 + 0.0002))
            stop_est = snap.mid - stop_distance
        else:
            entry_zone = (snap.mid * (1 - 0.0002), snap.mid * (1 + 0.0002))
            stop_est = snap.mid + stop_distance

        return SetupCandidate(
            direction=direction,
            setup_type="REVERSAL",
            confluence_score=min(1.0, score),
            signal_count=len(signals),
            signals=signals,
            ideal_entry_zone=entry_zone,
            initial_stop_estimate=stop_est,
            notes=f"Reversal {direction.value} at structural level",
        )

    def _build_breakout_candidate(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext,
    ) -> SetupCandidate | None:
        signals: list[str] = []
        score = 0.0
        direction = Direction.FLAT

        # BB squeeze breakout
        if snap.bb_width > 0 and snap.bb_width > 1.5:
            signals.append("BB_EXPANSION")
            score += 0.25
            if snap.mid > snap.ema_slow:
                direction = Direction.LONG
            else:
                direction = Direction.SHORT

        # Volume spike on breakout
        if snap.volume_ratio > 1.8:
            signals.append("VOLUME_BREAKOUT")
            score += 0.25

        # ADX starting to rise from low
        if 15 <= snap.adx_14 <= 25:
            signals.append("ADX_RISING_FROM_LOW")
            score += 0.20

        if direction == Direction.FLAT or score < self.MIN_CONFLUENCE:
            return None

        stop_distance = snap.atr_14 * 1.0 if snap.atr_14 > 0 else 0.0
        entry_zone = (snap.mid * (1 - 0.0001), snap.mid * (1 + 0.0001))
        stop_est = (snap.mid - stop_distance) if direction == Direction.LONG else (snap.mid + stop_distance)

        return SetupCandidate(
            direction=direction,
            setup_type="BREAKOUT",
            confluence_score=min(1.0, score),
            signal_count=len(signals),
            signals=signals,
            ideal_entry_zone=entry_zone,
            initial_stop_estimate=stop_est,
            notes=f"Range breakout {direction.value}",
        )

    def _empty_report(
        self, snapshot: MarketSnapshot, warnings: list[str]
    ) -> OpportunityReport:
        return OpportunityReport(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=0.0,
            warnings=warnings,
            candidates=[],
            best_candidate=None,
            opportunity_quality=0.0,
            market_is_tradeable=False,
        )
