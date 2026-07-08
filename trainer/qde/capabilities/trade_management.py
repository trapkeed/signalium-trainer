"""Phase 28.3 — Capability 6: Trade Management.

Evaluates all open trades and identifies which require management actions:
exit, break-even, partial close, scaling, duration enforcement.
Also detects reversal and continuation signals for open trades.

CONTRACT
--------
- Returns TradeManagementSignals (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from trainer.qde.capabilities.cap_models import (
    Direction,
    MarketContext,
    MarketSnapshot,
    OpenTrade,
    TradeManagementSignals,
)


class TradeManagementCapability:
    """Evaluates open trades and flags management requirements.

    This capability contributes to the QDE's Exit, Scaling, Partial Close,
    Break Even, Trade Duration, Reversal Detection, and Continuation
    Detection responsibilities.

    It identifies WHICH trades need WHAT management — the QDE decides
    whether and how to act on those signals.
    """

    CAPABILITY_NAME = "trade_management"

    # Break-even trigger: trade must be this many R in profit
    BREAK_EVEN_TRIGGER_R = 1.0

    # Partial close trigger: trade is this many R in profit
    PARTIAL_CLOSE_TRIGGER_R = 2.0

    # Scaling trigger: trade is in profit and trend is confirmed
    SCALING_TRIGGER_R = 1.5

    # Maximum trade duration in hours before flagging for review
    MAX_DURATION_HOURS = 24.0

    # RSI thresholds for reversal detection
    RSI_REVERSAL_BULL = 70.0   # Overbought → potential bearish reversal
    RSI_REVERSAL_BEAR = 30.0   # Oversold → potential bullish reversal

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        market_context: MarketContext | None = None,
    ) -> TradeManagementSignals:
        """Evaluate all open trades and return management signal flags.

        Args:
            snapshot: Current market snapshot including open_trades.
            market_context: Optional market context for regime-aware evaluation.

        Returns:
            TradeManagementSignals — advisory only, not trade instructions.
        """
        trades = snapshot.open_trades
        warnings: list[str] = []

        exit_ids: list[str] = []
        be_ids: list[str] = []
        partial_ids: list[str] = []
        scale_ids: list[str] = []
        duration_ids: list[str] = []
        per_trade_notes: dict[str, str] = {}

        for trade in trades:
            notes: list[str] = []

            # Duration check
            hours = trade.duration_s / 3600.0
            if hours > self.MAX_DURATION_HOURS:
                duration_ids.append(trade.trade_id)
                notes.append(f"Duration {hours:.1f}h > max {self.MAX_DURATION_HOURS}h")

            # R-multiple based management
            r = trade.r_multiple

            # Exit if stop is clearly violated (deep negative R)
            if r < -1.2:
                exit_ids.append(trade.trade_id)
                notes.append(f"Deep negative R: {r:.2f}")

            # Break-even: move stop to entry when in profit
            elif r >= self.BREAK_EVEN_TRIGGER_R and trade.trade_id not in exit_ids:
                be_ids.append(trade.trade_id)
                notes.append(f"Break-even eligible: R={r:.2f}")

            # Partial close: lock in profits at 2R
            if r >= self.PARTIAL_CLOSE_TRIGGER_R and trade.trade_id not in exit_ids:
                partial_ids.append(trade.trade_id)
                notes.append(f"Partial close eligible: R={r:.2f}")

            # Scaling: add to winner in strong trend
            if (
                r >= self.SCALING_TRIGGER_R
                and trade.trade_id not in exit_ids
                and market_context is not None
                and market_context.trend_strength.value in ("STRONG", "MODERATE")
            ):
                # Only scale if trade direction aligns with trend
                aligned = (
                    trade.direction == Direction.LONG and market_context.trend_direction == Direction.LONG
                ) or (
                    trade.direction == Direction.SHORT and market_context.trend_direction == Direction.SHORT
                )
                if aligned:
                    scale_ids.append(trade.trade_id)
                    notes.append(f"Scale eligible: R={r:.2f} trend aligned")

            per_trade_notes[trade.trade_id] = " | ".join(notes) if notes else "No action required"

        # Reversal / continuation detection
        reversal, reversal_conf = self._detect_reversal(snapshot, market_context)
        continuation, cont_conf = self._detect_continuation(snapshot, market_context)

        # If reversal is detected against open positions, flag them for exit
        if reversal and reversal_conf > 0.70:
            for trade in trades:
                bearish_reversal = (
                    trade.direction == Direction.LONG
                    and market_context is not None
                    and market_context.trend_direction == Direction.SHORT
                )
                bullish_reversal = (
                    trade.direction == Direction.SHORT
                    and market_context is not None
                    and market_context.trend_direction == Direction.LONG
                )
                if bearish_reversal or bullish_reversal:
                    if trade.trade_id not in exit_ids:
                        exit_ids.append(trade.trade_id)
                        per_trade_notes[trade.trade_id] += " | Reversal detected"

        if snapshot.spread_pips > 5.0:
            warnings.append("Wide spread — management actions may incur high cost")

        confidence = self._compute_confidence(snapshot, market_context)

        return TradeManagementSignals(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            trades_needing_exit=list(set(exit_ids)),
            trades_needing_break_even=list(set(be_ids)),
            trades_needing_partial_close=list(set(partial_ids)),
            trades_eligible_for_scaling=list(set(scale_ids)),
            trades_exceeding_duration=list(set(duration_ids)),
            reversal_detected=reversal,
            continuation_detected=continuation,
            reversal_confidence=reversal_conf,
            continuation_confidence=cont_conf,
            per_trade_notes=per_trade_notes,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_reversal(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext | None,
    ) -> tuple[bool, float]:
        """Detect potential trend reversal signals."""
        signals = 0
        total_weight = 0.0

        # RSI extremes against current trend
        if ctx is not None:
            if ctx.trend_direction == Direction.LONG and snap.rsi_14 > self.RSI_REVERSAL_BULL:
                signals += 1
                total_weight += 0.40
            elif ctx.trend_direction == Direction.SHORT and snap.rsi_14 < self.RSI_REVERSAL_BEAR:
                signals += 1
                total_weight += 0.40

        # ADX declining from strong trend
        if snap.adx_14 < 20 and ctx and ctx.trend_strength.value in ("STRONG", "MODERATE"):
            signals += 1
            total_weight += 0.30

        # Regime transition flagging reversal
        from trainer.qde.capabilities.cap_models import RegimeTransition
        if ctx and ctx.regime_transition == RegimeTransition.REVERSING:
            signals += 1
            total_weight += 0.30

        if signals == 0:
            return False, 0.0
        return True, min(1.0, total_weight)

    def _detect_continuation(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext | None,
    ) -> tuple[bool, float]:
        """Detect trend continuation signals."""
        if ctx is None:
            return False, 0.0

        signals = 0
        total_weight = 0.0

        # ADX rising in strong trend
        if snap.adx_14 > 25:
            signals += 1
            total_weight += 0.35

        # RSI in trend zone (not overbought/oversold)
        if ctx.trend_direction == Direction.LONG and 45 <= snap.rsi_14 <= 65:
            signals += 1
            total_weight += 0.30
        elif ctx.trend_direction == Direction.SHORT and 35 <= snap.rsi_14 <= 55:
            signals += 1
            total_weight += 0.30

        # Volume confirmation
        if snap.volume_ratio > 1.1:
            signals += 1
            total_weight += 0.20

        from trainer.qde.capabilities.cap_models import RegimeTransition
        if ctx.regime_transition in (RegimeTransition.STABLE, RegimeTransition.ACCELERATING):
            signals += 1
            total_weight += 0.15

        if signals < 2:
            return False, 0.0
        return True, min(1.0, total_weight)

    def _compute_confidence(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext | None,
    ) -> float:
        base = 0.75
        if ctx:
            base = ctx.confidence * 0.6 + 0.3
        if snap.spread_pips > 3.0:
            base *= 0.85
        return max(0.2, min(1.0, base))
