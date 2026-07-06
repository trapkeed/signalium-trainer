"""Phase 21.6 — Decision types, outcome types, and quality factors."""

from __future__ import annotations
from enum import Enum


class DecisionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"
    WAIT = "WAIT"
    CANCEL = "CANCEL"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    MOVE_STOP = "MOVE_STOP"
    MOVE_TARGET = "MOVE_TARGET"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    EXIT = "EXIT"


class OutcomeType(str, Enum):
    CORRECT_DIRECTION = "CORRECT_DIRECTION"
    INCORRECT_DIRECTION = "INCORRECT_DIRECTION"
    HIT_TP = "HIT_TP"
    HIT_SL = "HIT_SL"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    MISSED_OPPORTUNITY = "MISSED_OPPORTUNITY"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    NO_TRADE = "NO_TRADE"
    UNEXPECTED_MARKET_EVENT = "UNEXPECTED_MARKET_EVENT"


class QualityFactor(str, Enum):
    FORECAST_ACCURACY = "FORECAST_ACCURACY"
    EXECUTION_QUALITY = "EXECUTION_QUALITY"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    LIFECYCLE_MANAGEMENT = "LIFECYCLE_MANAGEMENT"
    PORTFOLIO_IMPACT = "PORTFOLIO_IMPACT"
    CALIBRATION = "CALIBRATION"
    CONFIDENCE_RELIABILITY = "CONFIDENCE_RELIABILITY"
    EXPECTED_VALUE = "EXPECTED_VALUE"
    CONSISTENCY = "CONSISTENCY"
    DECISION_ROBUSTNESS = "DECISION_ROBUSTNESS"


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    UNKNOWN = "UNKNOWN"


class VolatilityRegime(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class DecisionGrade(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    REJECT = "REJECT"


# Quality factor weights (must sum to 1.0)
QUALITY_WEIGHTS = {
    QualityFactor.FORECAST_ACCURACY: 0.20,
    QualityFactor.EXECUTION_QUALITY: 0.15,
    QualityFactor.RISK_MANAGEMENT: 0.15,
    QualityFactor.LIFECYCLE_MANAGEMENT: 0.10,
    QualityFactor.PORTFOLIO_IMPACT: 0.10,
    QualityFactor.CALIBRATION: 0.10,
    QualityFactor.CONFIDENCE_RELIABILITY: 0.08,
    QualityFactor.EXPECTED_VALUE: 0.07,
    QualityFactor.CONSISTENCY: 0.03,
    QualityFactor.DECISION_ROBUSTNESS: 0.02,
}

assert abs(sum(QUALITY_WEIGHTS.values()) - 1.0) < 1e-9

# Grade thresholds
GRADE_THRESHOLDS = {
    DecisionGrade.EXCELLENT: 0.85,
    DecisionGrade.GOOD: 0.70,
    DecisionGrade.ACCEPTABLE: 0.55,
    DecisionGrade.POOR: 0.40,
}


def grade_from_score(score: float) -> DecisionGrade:
    if score >= GRADE_THRESHOLDS[DecisionGrade.EXCELLENT]:
        return DecisionGrade.EXCELLENT
    if score >= GRADE_THRESHOLDS[DecisionGrade.GOOD]:
        return DecisionGrade.GOOD
    if score >= GRADE_THRESHOLDS[DecisionGrade.ACCEPTABLE]:
        return DecisionGrade.ACCEPTABLE
    if score >= GRADE_THRESHOLDS[DecisionGrade.POOR]:
        return DecisionGrade.POOR
    return DecisionGrade.REJECT


# Positive outcome types (used in quality scoring)
POSITIVE_OUTCOMES = frozenset(
    {
        OutcomeType.CORRECT_DIRECTION,
        OutcomeType.HIT_TP,
        OutcomeType.PARTIAL_SUCCESS,
    }
)

NEGATIVE_OUTCOMES = frozenset(
    {
        OutcomeType.INCORRECT_DIRECTION,
        OutcomeType.HIT_SL,
        OutcomeType.UNEXPECTED_MARKET_EVENT,
    }
)
