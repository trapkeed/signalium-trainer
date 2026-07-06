"""Phase 21 — Trade Manager.

Recommends active trade management actions based on current trade state:
  Move to Breakeven, Trail Stop, Partial Close, Scale In/Out, Exit Early, Hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.execution_types import ExecutionForecast, ManagementAction, _clamp
from trainer.execution.position_sizer import SizingResult
from trainer.execution.stop_loss_engine import StopLossRecommendation
from trainer.execution.take_profit_engine import TakeProfitRecommendation


@dataclass
class ManagementRule:
    """A single actionable management rule."""

    action: ManagementAction
    trigger: str  # human-readable trigger condition
    detail: str  # specific level / quantity
    priority: int  # 1 = highest

    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "trigger": self.trigger,
            "detail": self.detail,
            "priority": self.priority,
        }


@dataclass
class ManagementPlan:
    """Full trade management recommendation."""

    instrument: str
    direction: str

    primary_action: ManagementAction
    rules: List[ManagementRule] = field(default_factory=list)

    # Breakeven
    breakeven_trigger_price: float = 0.0  # price at which to move stop to BE
    breakeven_stop_price: float = 0.0

    # Trail
    trail_distance: float = 0.0  # ATR multiples to trail by
    trail_step: float = 0.0  # minimum step before moving trail

    # Partial close levels
    partial_close_tp1_pct: float = 0.50  # close 50% at TP1
    partial_close_tp2_pct: float = 0.30  # close 30% at TP2
    runner_pct: float = 0.20  # leave 20% as runner

    # Scale in
    scale_in_price: float = 0.0
    scale_in_lots_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "primary_action": self.primary_action.value,
            "rules": [r.to_dict() for r in self.rules],
            "breakeven": {
                "trigger_price": round(self.breakeven_trigger_price, 5),
                "stop_price": round(self.breakeven_stop_price, 5),
            },
            "trail": {
                "distance": round(self.trail_distance, 5),
                "step": round(self.trail_step, 5),
            },
            "partial_close": {
                "tp1_pct": self.partial_close_tp1_pct,
                "tp2_pct": self.partial_close_tp2_pct,
                "runner_pct": self.runner_pct,
            },
            "scale_in": {
                "price": round(self.scale_in_price, 5),
                "lots_pct": self.scale_in_lots_pct,
            },
        }


class TradeManager:
    """Generates a structured trade management plan.

    Logic:
      - High confidence (>0.75) + positive momentum → scale in available
      - Always recommend partial close at TP1 (50%), TP2 (30%), runner (20%)
      - Breakeven trigger at TP1 level
      - Trail stop after TP1 hit (1× ATR trail)
      - Low confidence or news risk → exit early is primary action
    """

    def plan(
        self,
        forecast: ExecutionForecast,
        sizing: Optional[SizingResult] = None,
        stop_loss: Optional[StopLossRecommendation] = None,
        take_profit: Optional[TakeProfitRecommendation] = None,
    ) -> ManagementPlan:

        entry = forecast.current_price
        sign = 1.0 if forecast.is_buy else -1.0
        atr = forecast.atr

        rules: List[ManagementRule] = []

        # Breakeven trigger: when price reaches TP1
        be_trigger = take_profit.tp1 if take_profit else round(entry + sign * atr, 5)
        be_stop = stop_loss.adaptive_stop if stop_loss else round(entry + sign * atr * 0.05, 5)

        # Trail distance = 1.0 × ATR
        trail_dist = round(atr, 5)
        trail_step = round(atr * 0.25, 5)

        # Scale-in: only if confidence > 0.75 and momentum aligned
        aligned_mom = forecast.momentum if forecast.is_buy else -forecast.momentum
        can_scale_in = forecast.confidence > 0.75 and aligned_mom > 0.2

        # Primary action selection
        if forecast.news_risk > 0.6:
            primary = ManagementAction.EXIT_EARLY
            rules.append(
                ManagementRule(
                    ManagementAction.EXIT_EARLY,
                    "News risk > 60% — close before event",
                    f"news_risk={forecast.news_risk:.0%}",
                    priority=1,
                )
            )
        elif forecast.confidence < 0.55:
            primary = ManagementAction.PARTIAL_CLOSE
            rules.append(
                ManagementRule(
                    ManagementAction.PARTIAL_CLOSE,
                    "Low-confidence setup — take partial at TP1",
                    "50% at TP1, hold remainder only with improved confidence",
                    priority=1,
                )
            )
        elif can_scale_in:
            primary = ManagementAction.SCALE_IN
            rules.append(
                ManagementRule(
                    ManagementAction.SCALE_IN,
                    "High confidence + aligned momentum — add on pullback",
                    f"Add 50% at conservative entry, total risk still within limits",
                    priority=1,
                )
            )
        else:
            primary = ManagementAction.HOLD
            rules.append(
                ManagementRule(
                    ManagementAction.HOLD,
                    "Standard management — hold to TP2, trail to runner",
                    "Follow partial close and trail rules",
                    priority=1,
                )
            )

        # Standard partial-close rules
        if take_profit:
            rules.append(
                ManagementRule(
                    ManagementAction.PARTIAL_CLOSE,
                    f"Partial close 50% at TP1",
                    f"TP1 = {round(take_profit.tp1, 5)}",
                    priority=2,
                )
            )
            rules.append(
                ManagementRule(
                    ManagementAction.PARTIAL_CLOSE,
                    f"Partial close 30% at TP2",
                    f"TP2 = {round(take_profit.tp2, 5)}",
                    priority=3,
                )
            )

        # Breakeven rule
        rules.append(
            ManagementRule(
                ManagementAction.MOVE_TO_BREAKEVEN,
                "Move stop to breakeven when TP1 reached",
                f"Trigger={round(be_trigger, 5)} → Stop={round(be_stop, 5)}",
                priority=4,
            )
        )

        # Trail stop rule
        rules.append(
            ManagementRule(
                ManagementAction.TRAIL_STOP,
                "Trail stop at 1× ATR after TP1",
                f"Trail distance={round(trail_dist, 5)} step={round(trail_step, 5)}",
                priority=5,
            )
        )

        # Scale-in level
        from trainer.execution.entry_zones import EntryZoneEngine

        zones = EntryZoneEngine().calculate(forecast)
        scale_price = zones.conservative_entry
        scale_pct = 0.50 if can_scale_in else 0.0

        logger.debug(
            f"TradeManager {forecast.instrument}: primary={primary.value} "
            f"rules={len(rules)} scale_in={can_scale_in}"
        )

        return ManagementPlan(
            instrument=forecast.instrument,
            direction=forecast.direction,
            primary_action=primary,
            rules=rules,
            breakeven_trigger_price=round(be_trigger, 5),
            breakeven_stop_price=round(be_stop, 5),
            trail_distance=trail_dist,
            trail_step=trail_step,
            partial_close_tp1_pct=0.50,
            partial_close_tp2_pct=0.30,
            runner_pct=0.20,
            scale_in_price=round(scale_price, 5),
            scale_in_lots_pct=scale_pct,
        )
