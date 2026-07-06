"""Phase 21 — Order Strategy Engine.

Recommends the optimal order type and parameters for each trade:
  Market, Limit, Stop, Stop-Limit, Conditional.
Also recommends order expiry and reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.entry_zones import EntryZones
from trainer.execution.execution_timing import TimingAssessment
from trainer.execution.execution_types import (
    BrokerConstraints,
    ExecutionForecast,
    OrderType,
    TimingRecommendation,
    _clamp,
    default_broker,
)


@dataclass
class OrderRecommendation:
    """Complete order execution recommendation."""

    instrument: str
    direction: str

    primary_type: OrderType
    alternative_type: OrderType

    # Order parameters
    limit_price: Optional[float] = None  # for LIMIT / STOP_LIMIT
    stop_price: Optional[float] = None  # for STOP / STOP_LIMIT
    entry_price: float = 0.0  # reference entry

    expiry_hours: float = 24.0  # GTC in hours
    expiry_note: str = ""

    reasoning: List[str] = field(default_factory=list)
    fill_confidence: float = 0.0  # 0-1, probability of fill

    def to_dict(self) -> Dict:
        d: Dict = {
            "instrument": self.instrument,
            "direction": self.direction,
            "primary_type": self.primary_type.value,
            "alternative_type": self.alternative_type.value,
            "entry_price": round(self.entry_price, 5),
            "expiry_hours": self.expiry_hours,
            "expiry_note": self.expiry_note,
            "reasoning": self.reasoning,
            "fill_confidence": round(self.fill_confidence, 4),
        }
        if self.limit_price is not None:
            d["limit_price"] = round(self.limit_price, 5)
        if self.stop_price is not None:
            d["stop_price"] = round(self.stop_price, 5)
        return d


class OrderStrategyEngine:
    """Selects the optimal order type based on market conditions.

    Decision logic:
      EXECUTE_NOW + high confidence + spread OK  → MARKET order
      EXECUTE_NOW + lower confidence             → LIMIT at primary entry
      WAIT (price not at zone)                   → LIMIT at primary entry
      AVOID (bad timing)                         → CONDITIONAL (price alert)
      Low fill probability for limit             → STOP order (chase breakout)
    """

    def recommend(
        self,
        forecast: ExecutionForecast,
        zones: Optional[EntryZones] = None,
        timing: Optional[TimingAssessment] = None,
        broker: Optional[BrokerConstraints] = None,
    ) -> OrderRecommendation:

        broker = broker or default_broker()
        entry = forecast.current_price
        sign = 1.0 if forecast.is_buy else -1.0
        atr = forecast.atr
        reasoning: List[str] = []

        pri_entry = zones.primary_entry if zones else entry
        timing_rec = timing.recommendation if timing else TimingRecommendation.WAIT

        # Distance from current price to primary entry
        dist_to_primary = abs(entry - pri_entry)
        at_zone = dist_to_primary <= atr * 0.15

        # Determine primary order type ────────────────────────────────────────
        if timing_rec == TimingRecommendation.AVOID:
            primary = OrderType.CONDITIONAL
            alt = OrderType.LIMIT
            limit_p = round(pri_entry, 5)
            stop_p = None
            expiry = 48.0
            expiry_note = "Good-till-cancelled — activate only when timing improves"
            fill_conf = 0.40
            reasoning.append("Timing AVOID — use conditional order pending improved conditions")

        elif timing_rec == TimingRecommendation.EXECUTE_NOW and at_zone:
            if forecast.confidence >= 0.70 and broker.spread_ok(forecast.spread):
                primary = OrderType.MARKET
                alt = OrderType.LIMIT
                limit_p = round(pri_entry, 5)
                stop_p = None
                expiry = 0.0
                expiry_note = "Immediate execution"
                fill_conf = 0.95
                reasoning.append("Excellent timing + high confidence + tight spread — market order")
            else:
                primary = OrderType.LIMIT
                alt = OrderType.MARKET
                limit_p = round(pri_entry, 5)
                stop_p = None
                expiry = 4.0
                expiry_note = "4-hour limit — cancel if not filled within session"
                fill_conf = 0.75
                reasoning.append("Good timing but lower confidence — limit at primary entry")

        elif timing_rec == TimingRecommendation.WAIT:
            # Check if price is approaching from above (for BUY) or below (for SELL)
            # If price needs to come down to zone, use limit
            if at_zone:
                primary = OrderType.LIMIT
                alt = OrderType.MARKET
                limit_p = round(pri_entry, 5)
                stop_p = None
                expiry = 8.0
                expiry_note = "8-hour session limit"
                fill_conf = 0.65
                reasoning.append(
                    "Near zone but timing WAIT — limit order pending session improvement"
                )
            else:
                # Price far from zone — place limit at primary
                primary = OrderType.LIMIT
                alt = OrderType.STOP_LIMIT
                limit_p = round(pri_entry, 5)
                stop_p = round(pri_entry - sign * atr * 0.10, 5)
                expiry = 24.0
                expiry_note = "24-hour GTC limit"
                fill_conf = 0.55
                reasoning.append(
                    f"Price {round(dist_to_primary, 5)} from zone — "
                    f"limit order at primary entry {round(pri_entry, 5)}"
                )

        else:  # EXECUTE_NOW, not at zone
            primary = OrderType.LIMIT
            alt = OrderType.STOP
            limit_p = round(pri_entry, 5)
            stop_p = round(pri_entry + sign * atr * 0.05, 5)  # breakout stop entry if limit missed
            expiry = 12.0
            expiry_note = "12-hour limit — if not filled, reassess"
            fill_conf = 0.60
            reasoning.append("Execute timing but price away — limit at zone, stop as backup")

        # Breakout consideration: high momentum → prefer STOP order
        aligned_mom = forecast.momentum if forecast.is_buy else -forecast.momentum
        if aligned_mom > 0.60 and primary == OrderType.LIMIT:
            reasoning.append("High momentum — consider STOP order to capture breakout")
            alt = OrderType.STOP

        # Stop-limit safety for STOP orders
        if primary == OrderType.STOP and stop_p is None:
            stop_p = round(entry + sign * atr * 0.10, 5)

        logger.debug(
            f"OrderStrategy {forecast.instrument}: {primary.value} "
            f"limit={limit_p} expiry={expiry}h fill_conf={fill_conf:.2f}"
        )

        return OrderRecommendation(
            instrument=forecast.instrument,
            direction=forecast.direction,
            primary_type=primary,
            alternative_type=alt,
            limit_price=limit_p,
            stop_price=stop_p,
            entry_price=round(pri_entry, 5),
            expiry_hours=expiry,
            expiry_note=expiry_note,
            reasoning=reasoning,
            fill_confidence=round(fill_conf, 4),
        )
