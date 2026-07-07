"""Phase 21 — Position Sizer.

Supports: Fixed Risk, Fixed Fractional, Volatility Adjusted, Kelly Fraction.
Enforces maximum daily / weekly / portfolio risk limits.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.execution_types import (
    BrokerConstraints,
    ExecutionForecast,
    SizingMethod,
    _clamp,
    default_broker,
)
from trainer.execution.stop_loss_engine import StopLossRecommendation


@dataclass
class RiskLimits:
    """Portfolio-level risk constraints."""

    max_risk_per_trade_pct: float = 0.02  # 2% account per trade
    max_daily_risk_pct: float = 0.06  # 6% account per day
    max_weekly_risk_pct: float = 0.12  # 12% account per week
    max_portfolio_risk_pct: float = 0.20  # 20% account total open risk
    used_daily_risk_pct: float = 0.0
    used_weekly_risk_pct: float = 0.0
    used_portfolio_risk_pct: float = 0.0

    def available_daily(self) -> float:
        return max(0.0, self.max_daily_risk_pct - self.used_daily_risk_pct)

    def available_weekly(self) -> float:
        return max(0.0, self.max_weekly_risk_pct - self.used_weekly_risk_pct)

    def available_portfolio(self) -> float:
        return max(0.0, self.max_portfolio_risk_pct - self.used_portfolio_risk_pct)

    def effective_max_pct(self) -> float:
        return min(
            self.max_risk_per_trade_pct,
            self.available_daily(),
            self.available_weekly(),
            self.available_portfolio(),
        )


@dataclass
class SizingResult:
    """Output of position sizing calculation."""

    instrument: str
    direction: str
    method: str

    lots: float  # final validated lot size
    units: float  # lots × 100,000 (standard FX convention)
    risk_amount: float  # USD at risk
    risk_pct: float  # as fraction of account
    stop_distance: float  # in price units

    kelly_fraction: float = 0.0  # raw Kelly (informational)
    kelly_lots: float = 0.0  # Kelly-derived lot size

    margin_required: float = 0.0
    commission_cost: float = 0.0

    # Limit flags
    daily_risk_ok: bool = True
    weekly_risk_ok: bool = True
    portfolio_risk_ok: bool = True
    size_reduced: bool = False

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "method": self.method,
            "lots": round(self.lots, 2),
            "units": round(self.units),
            "risk_amount": round(self.risk_amount, 2),
            "risk_pct": round(self.risk_pct, 4),
            "stop_distance": round(self.stop_distance, 5),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "kelly_lots": round(self.kelly_lots, 2),
            "margin_required": round(self.margin_required, 2),
            "commission_cost": round(self.commission_cost, 2),
            "limits": {
                "daily_ok": self.daily_risk_ok,
                "weekly_ok": self.weekly_risk_ok,
                "portfolio_ok": self.portfolio_risk_ok,
                "size_reduced": self.size_reduced,
            },
        }


class PositionSizer:
    """Calculates position size using the requested method.

    All methods ultimately produce a lot size validated against broker
    constraints and portfolio risk limits.
    """

    # Standard FX lot = 100,000 units; pip value ≈ $10/lot for major pairs
    UNITS_PER_LOT = 100_000
    PIP_VALUE_PER_LOT = 10.0  # USD per pip, per standard lot

    def __init__(
        self,
        broker: BrokerConstraints | None = None,
        risk_limits: RiskLimits | None = None,
    ):
        self._broker = broker or default_broker()
        self._limits = risk_limits or RiskLimits()

    def calculate(
        self,
        forecast: ExecutionForecast,
        stop_loss: StopLossRecommendation | None = None,
        method: SizingMethod = SizingMethod.FIXED_RISK,
    ) -> SizingResult:

        entry = forecast.current_price
        stop = (
            stop_loss.initial_stop
            if stop_loss
            else (entry - forecast.atr * 1.5 if forecast.is_buy else entry + forecast.atr * 1.5)
        )
        stop_dist = max(abs(entry - stop), self._broker.min_stop_distance)

        effective_risk_pct = min(
            self._limits.effective_max_pct(),
            0.02,  # hard cap per trade
        )
        risk_amount = forecast.account_size * effective_risk_pct

        # ── Sizing method ────────────────────────────────────────────────────
        if method == SizingMethod.FIXED_RISK:
            lots = self._fixed_risk(risk_amount, stop_dist, entry)

        elif method == SizingMethod.FIXED_FRACTIONAL:
            lots = self._fixed_fractional(forecast.account_size, stop_dist, entry)

        elif method == SizingMethod.VOLATILITY_ADJUSTED:
            lots = self._volatility_adjusted(risk_amount, forecast.volatility, entry)

        else:  # KELLY
            lots = self._kelly(forecast, risk_amount, stop_dist, entry)

        # Validate against broker + risk limits
        raw_lots = lots
        lots = self._broker.validate_lot(lots)

        # Kelly metrics (always compute for info)
        kf, kl = self._kelly_info(forecast, stop_dist, entry)

        # Margin and commission
        margin = round(lots * self.UNITS_PER_LOT * entry * self._broker.margin_rate, 2)
        commiss = round(self._broker.commission_per_lot * lots, 2)

        # Actual risk at final lot size
        actual_risk = round(lots * self.UNITS_PER_LOT * stop_dist, 2)
        actual_pct = round(actual_risk / max(forecast.account_size, 1.0), 4)

        # Limit checks
        d_ok = actual_pct <= self._limits.available_daily()
        w_ok = actual_pct <= self._limits.available_weekly()
        p_ok = actual_pct <= self._limits.available_portfolio()

        size_reduced = lots < raw_lots - 0.001

        logger.debug(
            f"PositionSizer {forecast.instrument} {method.value}: "
            f"lots={lots} risk={actual_pct:.2%}"
        )

        return SizingResult(
            instrument=forecast.instrument,
            direction=forecast.direction,
            method=method.value,
            lots=lots,
            units=round(lots * self.UNITS_PER_LOT),
            risk_amount=actual_risk,
            risk_pct=actual_pct,
            stop_distance=round(stop_dist, 5),
            kelly_fraction=round(kf, 4),
            kelly_lots=round(kl, 2),
            margin_required=margin,
            commission_cost=commiss,
            daily_risk_ok=d_ok,
            weekly_risk_ok=w_ok,
            portfolio_risk_ok=p_ok,
            size_reduced=size_reduced,
        )

    # ── Sizing implementations ────────────────────────────────────────────────

    def _fixed_risk(self, risk_amount: float, stop_dist: float, price: float) -> float:
        """Risk a fixed dollar amount."""
        value_per_lot = self.UNITS_PER_LOT * stop_dist
        if value_per_lot < 1e-9:
            return self._broker.min_lot_size
        return risk_amount / value_per_lot

    def _fixed_fractional(self, account: float, stop_dist: float, price: float) -> float:
        """Fixed fraction of account (1% of account / stop_pct)."""
        stop_pct = stop_dist / max(price, 1e-9)
        if stop_pct < 1e-9:
            return self._broker.min_lot_size
        risk_amount = account * 0.01
        return self._fixed_risk(risk_amount, stop_dist, price)

    def _volatility_adjusted(self, risk_amount: float, vol: float, price: float) -> float:
        """Scale lot size inversely to volatility."""
        base_vol = 0.01
        vol_ratio = base_vol / max(vol, 1e-4)
        base_lots = risk_amount / (self.UNITS_PER_LOT * price * base_vol)
        return base_lots * _clamp(vol_ratio, 0.1, 3.0)

    def _kelly(
        self,
        forecast: ExecutionForecast,
        risk_amount: float,
        stop_dist: float,
        price: float,
    ) -> float:
        kf, _ = self._kelly_info(forecast, stop_dist, price)
        # Half-Kelly for safety
        safe_kf = kf * 0.5
        return self._fixed_risk(risk_amount * safe_kf, stop_dist, price)

    def _kelly_info(
        self,
        forecast: ExecutionForecast,
        stop_dist: float,
        price: float,
    ):
        """Return (kelly_fraction, kelly_lots)."""
        p = forecast.confidence
        q = 1.0 - p
        rr = forecast.risk_reward_raw
        if rr < 1e-9 or q < 1e-9:
            return 0.0, self._broker.min_lot_size
        kf = _clamp(p - q / rr)
        kl = self._fixed_risk(
            forecast.account_size * kf,
            stop_dist,
            price,
        )
        kl = self._broker.validate_lot(kl)
        return kf, kl
