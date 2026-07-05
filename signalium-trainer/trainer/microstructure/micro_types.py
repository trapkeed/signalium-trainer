"""Phase 22 — Market Microstructure types and enums."""

from __future__ import annotations
from enum import Enum


class MarketState(str, Enum):
    TRENDING       = "TRENDING"
    STRONG_TREND   = "STRONG_TREND"
    WEAK_TREND     = "WEAK_TREND"
    RANGE          = "RANGE"
    COMPRESSION    = "COMPRESSION"
    EXPANSION      = "EXPANSION"
    BREAKOUT       = "BREAKOUT"
    FAKE_BREAKOUT  = "FAKE_BREAKOUT"
    REVERSAL       = "REVERSAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY  = "LOW_VOLATILITY"
    TRANSITION     = "TRANSITION"


class MomentumState(str, Enum):
    ACCELERATING  = "ACCELERATING"
    DECELERATING  = "DECELERATING"
    EXHAUSTED     = "EXHAUSTED"
    NEUTRAL       = "NEUTRAL"
    REVERSING     = "REVERSING"


class VolatilityState(str, Enum):
    EXPANDING  = "EXPANDING"
    CONTRACTING = "CONTRACTING"
    SHOCK      = "SHOCK"
    NORMAL     = "NORMAL"
    CLUSTERING = "CLUSTERING"


class LiquidityState(str, Enum):
    DEEP     = "DEEP"
    NORMAL   = "NORMAL"
    THIN     = "THIN"
    VACUUM   = "VACUUM"


class TrendStrength(str, Enum):
    VERY_STRONG = "VERY_STRONG"   # > 0.80
    STRONG      = "STRONG"        # > 0.60
    MODERATE    = "MODERATE"      # > 0.40
    WEAK        = "WEAK"          # > 0.20
    NONE        = "NONE"          # ≤ 0.20


class DataGranularity(str, Enum):
    OHLC      = "OHLC"       # available now
    TICK      = "TICK"       # future
    LEVEL2    = "LEVEL2"     # future
    EXECUTION = "EXECUTION"  # future


# Market state classification priority (highest priority first)
STATE_PRIORITY = [
    MarketState.HIGH_VOLATILITY,
    MarketState.LOW_VOLATILITY,
    MarketState.STRONG_TREND,
    MarketState.BREAKOUT,
    MarketState.FAKE_BREAKOUT,
    MarketState.REVERSAL,
    MarketState.TRENDING,
    MarketState.WEAK_TREND,
    MarketState.COMPRESSION,
    MarketState.EXPANSION,
    MarketState.RANGE,
    MarketState.TRANSITION,
]


def trend_strength_from_score(score: float) -> TrendStrength:
    if score > 0.80: return TrendStrength.VERY_STRONG
    if score > 0.60: return TrendStrength.STRONG
    if score > 0.40: return TrendStrength.MODERATE
    if score > 0.20: return TrendStrength.WEAK
    return TrendStrength.NONE


# Embedding dimensions by sub-component
EMBEDDING_DIMS = {
    "price_action": 8,
    "liquidity":    6,
    "momentum":     6,
    "volatility":   6,
    "efficiency":   6,
}
EMBEDDING_TOTAL = sum(EMBEDDING_DIMS.values())   # 32
