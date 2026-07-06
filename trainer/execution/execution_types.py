"""Phase 21 — Core execution types shared across all execution modules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

# ── Enumerations ──────────────────────────────────────────────────────────────


class Session(str, Enum):
    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"  # London-NY overlap
    CLOSED = "CLOSED"


class ExecutionGrade(str, Enum):
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    REJECT = "REJECT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    CONDITIONAL = "CONDITIONAL"


class TimingRecommendation(str, Enum):
    EXECUTE_NOW = "EXECUTE_NOW"
    WAIT = "WAIT"
    AVOID = "AVOID"


class SizingMethod(str, Enum):
    FIXED_RISK = "FIXED_RISK"
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"
    KELLY = "KELLY"


class ManagementAction(str, Enum):
    MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"
    TRAIL_STOP = "TRAIL_STOP"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    EXIT_EARLY = "EXIT_EARLY"
    HOLD = "HOLD"


# ── ExecutionForecast ─────────────────────────────────────────────────────────


@dataclass
class ExecutionForecast:
    """Input to the execution platform — a single-instrument forecast with market context."""

    instrument: str
    direction: str  # "BUY" or "SELL"
    confidence: float  # 0-1
    expected_return: float  # fractional (e.g. 0.08 = 8%)
    expected_risk: float  # fractional (e.g. 0.10 = 10%)
    current_price: float  # e.g. 1.0850
    atr: float  # Average True Range in price units
    spread: float  # bid-ask spread in price units
    session: Session = Session.LONDON
    volatility: float = 0.01  # realised vol, fractional
    momentum: float = 0.0  # -1 to +1 (aligned / opposed)
    news_risk: float = 0.0  # 0 = none, 1 = high
    account_size: float = 10_000.0
    currency: str = "USD"

    # Derived helpers ────────────────────────────────────────────────────────
    @property
    def is_buy(self) -> bool:
        return self.direction.upper() == "BUY"

    @property
    def spread_cost_pct(self) -> float:
        """Spread as fraction of current price."""
        if self.current_price < 1e-9:
            return 0.0
        return self.spread / self.current_price

    @property
    def atr_pct(self) -> float:
        """ATR as fraction of current price."""
        if self.current_price < 1e-9:
            return 0.0
        return self.atr / self.current_price

    @property
    def risk_reward_raw(self) -> float:
        """Naive expected_return / expected_risk ratio."""
        if self.expected_risk < 1e-9:
            return 0.0
        return self.expected_return / self.expected_risk

    def to_dict(self) -> Dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "expected_return": round(self.expected_return, 4),
            "expected_risk": round(self.expected_risk, 4),
            "current_price": round(self.current_price, 5),
            "atr": round(self.atr, 5),
            "spread": round(self.spread, 5),
            "session": self.session.value,
            "volatility": round(self.volatility, 4),
            "momentum": round(self.momentum, 4),
            "news_risk": round(self.news_risk, 4),
            "account_size": self.account_size,
        }


# ── BrokerConstraints ─────────────────────────────────────────────────────────


@dataclass
class BrokerConstraints:
    """Configurable broker execution constraints."""

    min_lot_size: float = 0.01
    max_lot_size: float = 100.0
    lot_step: float = 0.01
    min_stop_distance: float = 0.0001  # in price units
    max_spread_to_trade: float = 0.003  # stop trading above this spread
    max_leverage: float = 30.0
    margin_rate: float = 0.033  # 3.3% margin (30:1)
    commission_per_lot: float = 7.0  # USD per lot round trip
    broker_name: str = "default"

    def validate_lot(self, lots: float) -> float:
        """Clip and round lot size to broker constraints."""
        clipped = max(self.min_lot_size, min(self.max_lot_size, lots))
        steps = round(clipped / self.lot_step)
        return round(steps * self.lot_step, 4)

    def spread_ok(self, spread: float) -> bool:
        return spread <= self.max_spread_to_trade

    def to_dict(self) -> Dict:
        return {
            "broker": self.broker_name,
            "min_lot": self.min_lot_size,
            "max_lot": self.max_lot_size,
            "lot_step": self.lot_step,
            "min_stop_distance": self.min_stop_distance,
            "max_spread": self.max_spread_to_trade,
            "max_leverage": self.max_leverage,
            "margin_rate": self.margin_rate,
            "commission_per_lot": self.commission_per_lot,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _grade_from_score(score: float) -> ExecutionGrade:
    if score >= 0.85:
        return ExecutionGrade.EXCELLENT
    if score >= 0.70:
        return ExecutionGrade.GOOD
    if score >= 0.55:
        return ExecutionGrade.ACCEPTABLE
    if score >= 0.40:
        return ExecutionGrade.POOR
    return ExecutionGrade.REJECT


def default_broker() -> BrokerConstraints:
    return BrokerConstraints()


SESSION_QUALITY: Dict[Session, float] = {
    Session.OVERLAP: 1.0,
    Session.LONDON: 0.85,
    Session.NEW_YORK: 0.80,
    Session.TOKYO: 0.55,
    Session.SYDNEY: 0.45,
    Session.CLOSED: 0.10,
}
