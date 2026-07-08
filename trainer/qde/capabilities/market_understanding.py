"""Phase 28.3 — Capability 1: Market Understanding.

Reads raw market data from the MarketSnapshot and synthesises it into a
structured MarketContext.

CONTRACT
--------
- Returns MarketContext (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from trainer.qde.capabilities.cap_models import (
    Direction,
    LiquidityLevel,
    MarketContext,
    MarketSnapshot,
    RegimeTransition,
    RegimeType,
    TrendStrength,
)


class MarketUnderstandingCapability:
    """Synthesises raw market data into a structured market context.

    This capability owns three QDE responsibilities:
    - Market Understanding  — comprehensive market context
    - Regime Understanding  — current regime type and transition
    - Continuation Detection — is the current trend continuing?

    It does not own Entry, Exit, or any trade action responsibility.
    """

    CAPABILITY_NAME = "market_understanding"

    # ADX thresholds for trend strength classification
    ADX_STRONG = 35.0
    ADX_MODERATE = 20.0
    ADX_WEAK = 10.0

    # ATR ratio thresholds for volatility classification
    ATR_EXTREME_MULT = 2.5
    ATR_HIGH_MULT = 1.5
    ATR_LOW_MULT = 0.6

    # Spread thresholds (in pips)
    SPREAD_WIDE_PIPS = 3.0
    SPREAD_VERY_WIDE_PIPS = 8.0

    def __init__(self, atr_volatility_window: int = 50) -> None:
        self._atr_window = atr_volatility_window

    def analyse(self, snapshot: MarketSnapshot) -> MarketContext:
        """Analyse the market snapshot and return a structured context.

        Args:
            snapshot: Complete market state at reasoning time.

        Returns:
            MarketContext — a read-only advisory output. Not a decision.
        """
        trend_dir, trend_str = self._classify_trend(snapshot)
        regime, regime_conf = self._classify_regime(snapshot, trend_dir, trend_str)
        regime_transition = self._classify_regime_transition(snapshot, regime)
        vol_state, vol_pct = self._classify_volatility(snapshot)
        spread_state = self._classify_spread(snapshot)
        session_quality = self._score_session(snapshot)
        s_above, s_below = self._detect_structural_levels(snapshot)

        confidence = self._overall_confidence(
            regime_conf=regime_conf,
            adx=snapshot.adx_14,
            vol_state=vol_state,
            spread_state=spread_state,
        )

        warnings: list[str] = []
        if vol_state in ("HIGH", "EXTREME"):
            warnings.append(f"Elevated volatility: {vol_state}")
        if spread_state in ("WIDE", "VERY_WIDE"):
            warnings.append(f"Spread is {spread_state}")
        if snapshot.liquidity == LiquidityLevel.ILLIQUID:
            warnings.append("Market liquidity is ILLIQUID")

        return MarketContext(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            regime=regime,
            regime_transition=regime_transition,
            regime_confidence=regime_conf,
            trend_direction=trend_dir,
            trend_strength=trend_str,
            volatility_state=vol_state,
            volatility_percentile=vol_pct,
            spread_state=spread_state,
            session_quality=session_quality,
            structural_level_above=s_above,
            structural_level_below=s_below,
            notes=self._build_notes(regime, trend_dir, trend_str, vol_state),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _classify_trend(self, snap: MarketSnapshot) -> tuple[Direction, TrendStrength]:
        if snap.ema_fast == 0.0 and snap.ema_slow == 0.0:
            return Direction.FLAT, TrendStrength.NONE

        ema_bullish = snap.ema_fast > snap.ema_slow
        ema_bearish = snap.ema_fast < snap.ema_slow
        price_above_ema_fast = snap.mid > snap.ema_fast if snap.ema_fast > 0 else None
        adx = snap.adx_14

        if adx >= self.ADX_STRONG:
            strength = TrendStrength.STRONG
        elif adx >= self.ADX_MODERATE:
            strength = TrendStrength.MODERATE
        elif adx >= self.ADX_WEAK:
            strength = TrendStrength.WEAK
        else:
            return Direction.FLAT, TrendStrength.NONE

        if ema_bullish:
            return Direction.LONG, strength
        elif ema_bearish:
            return Direction.SHORT, strength
        return Direction.FLAT, TrendStrength.NONE

    def _classify_regime(
        self,
        snap: MarketSnapshot,
        trend_dir: Direction,
        trend_str: TrendStrength,
    ) -> tuple[RegimeType, float]:
        vol_ratio = snap.atr_14 / (snap.atr_50 + 1e-10) if snap.atr_50 > 0 else 1.0
        adx = snap.adx_14

        # Crisis: extreme volatility + illiquid / very low RSI or very high RSI
        if vol_ratio > self.ATR_EXTREME_MULT or snap.liquidity == LiquidityLevel.ILLIQUID:
            return RegimeType.CRISIS, 0.75

        # High volatility without clear trend direction
        if vol_ratio > self.ATR_HIGH_MULT and trend_str in (TrendStrength.NONE, TrendStrength.WEAK):
            return RegimeType.HIGH_VOLATILITY, 0.80

        # Low volatility
        if vol_ratio < self.ATR_LOW_MULT and adx < self.ADX_WEAK:
            return RegimeType.LOW_VOLATILITY, 0.80

        # Trending
        if trend_str in (TrendStrength.STRONG, TrendStrength.MODERATE):
            conf = 0.85 if trend_str == TrendStrength.STRONG else 0.70
            if trend_dir == Direction.LONG:
                return RegimeType.TRENDING_BULL, conf
            elif trend_dir == Direction.SHORT:
                return RegimeType.TRENDING_BEAR, conf

        # Recovery: moderate ADX with bullish momentum from a low RSI
        if snap.rsi_14 < 40 and trend_dir == Direction.LONG and adx >= self.ADX_WEAK:
            return RegimeType.RECOVERY, 0.60

        # Default to ranging
        return RegimeType.RANGING, 0.65

    def _classify_regime_transition(
        self,
        snap: MarketSnapshot,
        regime: RegimeType,
    ) -> RegimeTransition:
        # Use RSI momentum as a simple proxy for regime transition
        rsi = snap.rsi_14
        if regime in (RegimeType.TRENDING_BULL, RegimeType.TRENDING_BEAR):
            if (regime == RegimeType.TRENDING_BULL and rsi < 45) or (
                regime == RegimeType.TRENDING_BEAR and rsi > 55
            ):
                return RegimeTransition.REVERSING
            if (regime == RegimeType.TRENDING_BULL and rsi > 70) or (
                regime == RegimeType.TRENDING_BEAR and rsi < 30
            ):
                return RegimeTransition.ACCELERATING
        if regime == RegimeType.RANGING:
            if snap.adx_14 > self.ADX_MODERATE:
                return RegimeTransition.ACCELERATING
        return RegimeTransition.STABLE

    def _classify_volatility(self, snap: MarketSnapshot) -> tuple[str, float]:
        if snap.atr_50 == 0:
            return "NORMAL", 50.0
        ratio = snap.atr_14 / snap.atr_50
        if ratio > self.ATR_EXTREME_MULT:
            pct = min(99.0, 80.0 + (ratio - self.ATR_EXTREME_MULT) * 5)
            return "EXTREME", pct
        if ratio > self.ATR_HIGH_MULT:
            pct = 65.0 + (ratio - self.ATR_HIGH_MULT) * 15
            return "HIGH", min(80.0, pct)
        if ratio < self.ATR_LOW_MULT:
            pct = max(5.0, 40.0 - (self.ATR_LOW_MULT - ratio) * 30)
            return "LOW", pct
        return "NORMAL", 50.0

    def _classify_spread(self, snap: MarketSnapshot) -> str:
        if snap.spread_pips >= self.SPREAD_VERY_WIDE_PIPS:
            return "VERY_WIDE"
        if snap.spread_pips >= self.SPREAD_WIDE_PIPS:
            return "WIDE"
        return "NORMAL"

    def _score_session(self, snap: MarketSnapshot) -> float:
        session_scores = {
            "LONDON": 0.90,
            "NEW_YORK": 0.85,
            "OVERLAP": 0.95,  # London/NY overlap — highest liquidity
            "TOKYO": 0.65,
            "SYDNEY": 0.50,
            "": 0.55,
        }
        base = session_scores.get(snap.session, 0.55)
        # Penalise for illiquidity and wide spread
        if snap.liquidity == LiquidityLevel.LOW:
            base *= 0.8
        elif snap.liquidity == LiquidityLevel.ILLIQUID:
            base *= 0.5
        if snap.spread_pips > self.SPREAD_WIDE_PIPS:
            base *= 0.85
        return max(0.0, min(1.0, base))

    def _detect_structural_levels(
        self,
        snap: MarketSnapshot,
    ) -> tuple[float | None, float | None]:
        """Identify nearest recent swing high and swing low from recent_bars."""
        if len(snap.recent_bars) < 5:
            return None, None
        highs = [b.high for b in snap.recent_bars[-20:]]
        lows = [b.low for b in snap.recent_bars[-20:]]
        above = min((h for h in highs if h > snap.mid), default=None)
        below = max((lo for lo in lows if lo < snap.mid), default=None)
        return above, below

    def _overall_confidence(
        self,
        regime_conf: float,
        adx: float,
        vol_state: str,
        spread_state: str,
    ) -> float:
        confidence = regime_conf
        # Penalise for extreme conditions
        if vol_state in ("HIGH", "EXTREME"):
            confidence *= 0.85
        if spread_state in ("WIDE", "VERY_WIDE"):
            confidence *= 0.9
        if adx < self.ADX_WEAK:
            confidence *= 0.9
        return max(0.1, min(1.0, confidence))

    def _build_notes(
        self,
        regime: RegimeType,
        trend: Direction,
        strength: TrendStrength,
        vol: str,
    ) -> str:
        return (
            f"Regime={regime.value} " f"Trend={trend.value}/{strength.value} " f"Volatility={vol}"
        )
