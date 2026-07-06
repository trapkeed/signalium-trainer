"""Phase 24.9 — MIE types and enumerations."""

from __future__ import annotations

from enum import Enum


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    STRONG_TREND = "STRONG_TREND"
    WEAK_TREND = "WEAK_TREND"
    RANGE = "RANGE"
    COMPRESSION = "COMPRESSION"
    EXPANSION = "EXPANSION"
    BREAKOUT = "BREAKOUT"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    REVERSAL = "REVERSAL"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    VOLATILE = "VOLATILE"
    ILLIQUID = "ILLIQUID"
    NEWS_DRIVEN = "NEWS_DRIVEN"
    BLACK_SWAN = "BLACK_SWAN"
    UNKNOWN = "UNKNOWN"


class RiskRegime(str, Enum):
    OPTIMAL = "OPTIMAL"
    LOW_RISK = "LOW_RISK"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH_RISK = "HIGH_RISK"
    EXTREME = "EXTREME"
    FLASH_VOLATILITY = "FLASH_VOLATILITY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    MARKET_STRESS = "MARKET_STRESS"
    NEWS_EVENT = "NEWS_EVENT"
    BLACK_SWAN = "BLACK_SWAN"
    UNKNOWN = "UNKNOWN"


class TradeabilityGrade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    AVOID = "Avoid"


class ExecutionMode(str, Enum):
    MARKET_ORDER = "MARKET_ORDER"
    LIMIT_ORDER = "LIMIT_ORDER"
    WAIT = "WAIT"
    AVOID = "AVOID"


class SessionType(str, Enum):
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"
    OFF_HOURS = "OFF_HOURS"


class NewsImpact(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class MomentumState(str, Enum):
    ACCELERATING = "ACCELERATING"
    STEADY = "STEADY"
    DECELERATING = "DECELERATING"
    REVERSING = "REVERSING"
    FLAT = "FLAT"


class TrendState(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    SIDEWAYS = "SIDEWAYS"
    REVERSING = "REVERSING"


class VolatilityState(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class LiquidityState(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"
    ILLIQUID = "ILLIQUID"


class WorkerTrigger(str, Enum):
    NEW_CANDLE = "NEW_CANDLE"
    ECONOMIC_EVENT = "ECONOMIC_EVENT"
    NEW_PREDICTION = "NEW_PREDICTION"
    MODEL_UPDATE = "MODEL_UPDATE"
    REGIME_TRANSITION = "REGIME_TRANSITION"
    MANUAL_REFRESH = "MANUAL_REFRESH"


# ── Helper functions ───────────────────────────────────────────────────────────


def tradeability_grade(score: float) -> TradeabilityGrade:
    """Map score 0–100 to a TradeabilityGrade."""
    if score >= 85:
        return TradeabilityGrade.A_PLUS
    if score >= 70:
        return TradeabilityGrade.A
    if score >= 55:
        return TradeabilityGrade.B
    if score >= 40:
        return TradeabilityGrade.C
    if score >= 25:
        return TradeabilityGrade.D
    return TradeabilityGrade.AVOID


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# ── Lookup tables ──────────────────────────────────────────────────────────────

RISK_REGIME_MULTIPLIERS: dict[str, float] = {
    RiskRegime.OPTIMAL: 1.20,
    RiskRegime.LOW_RISK: 1.10,
    RiskRegime.NORMAL: 1.00,
    RiskRegime.ELEVATED: 0.80,
    RiskRegime.HIGH_RISK: 0.60,
    RiskRegime.EXTREME: 0.30,
    RiskRegime.FLASH_VOLATILITY: 0.20,
    RiskRegime.LOW_LIQUIDITY: 0.40,
    RiskRegime.MARKET_STRESS: 0.25,
    RiskRegime.NEWS_EVENT: 0.70,
    RiskRegime.BLACK_SWAN: 0.10,
    RiskRegime.UNKNOWN: 0.50,
}

REGIME_TRADEABILITY_BASE: dict[str, float] = {
    MarketRegime.TRENDING: 80.0,
    MarketRegime.STRONG_TREND: 85.0,
    MarketRegime.WEAK_TREND: 65.0,
    MarketRegime.RANGE: 60.0,
    MarketRegime.COMPRESSION: 50.0,
    MarketRegime.EXPANSION: 75.0,
    MarketRegime.BREAKOUT: 78.0,
    MarketRegime.FALSE_BREAKOUT: 30.0,
    MarketRegime.REVERSAL: 55.0,
    MarketRegime.ACCUMULATION: 60.0,
    MarketRegime.DISTRIBUTION: 55.0,
    MarketRegime.VOLATILE: 40.0,
    MarketRegime.ILLIQUID: 20.0,
    MarketRegime.NEWS_DRIVEN: 45.0,
    MarketRegime.BLACK_SWAN: 5.0,
    MarketRegime.UNKNOWN: 35.0,
}

SESSION_QUALITY_SCORES: dict[str, float] = {
    SessionType.ASIAN: 55.0,
    SessionType.LONDON: 85.0,
    SessionType.NEW_YORK: 80.0,
    SessionType.OVERLAP: 95.0,
    SessionType.OFF_HOURS: 25.0,
}

NEWS_IMPACT_PENALTY: dict[str, float] = {
    NewsImpact.HIGH: 30.0,
    NewsImpact.MEDIUM: 15.0,
    NewsImpact.LOW: 5.0,
    NewsImpact.NONE: 0.0,
}
