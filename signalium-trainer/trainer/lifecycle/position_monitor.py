"""Phase 21.5 — Position Monitor.

Generates position management recommendations for open trades:
Hold / Scale In / Scale Out / Reduce Exposure / Move Stop /
Move Target / Exit Immediately / Wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_states import LifecycleState, POSITION_OPEN_STATES
from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics


class ManagementRecommendation(str):
    HOLD              = "HOLD"
    SCALE_IN          = "SCALE_IN"
    SCALE_OUT         = "SCALE_OUT"
    REDUCE_EXPOSURE   = "REDUCE_EXPOSURE"
    MOVE_STOP         = "MOVE_STOP"
    MOVE_TARGET       = "MOVE_TARGET"
    EXIT_IMMEDIATELY  = "EXIT_IMMEDIATELY"
    WAIT              = "WAIT"


@dataclass
class PositionAction:
    """One recommended position management action."""
    action:      str
    priority:    int    # 1 = highest
    trigger:     str
    detail:      str
    urgency:     str    = "NORMAL"   # LOW | NORMAL | HIGH | CRITICAL

    def to_dict(self) -> Dict:
        return {
            "action":   self.action,
            "priority": self.priority,
            "trigger":  self.trigger,
            "detail":   self.detail,
            "urgency":  self.urgency,
        }


@dataclass
class PositionManagementPlan:
    """Complete position management recommendation."""
    instrument:      str
    state:           LifecycleState
    primary:         str                          # ManagementRecommendation
    actions:         List[PositionAction]         = field(default_factory=list)
    risk_level:      str                          = "MEDIUM"
    notes:           List[str]                    = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "instrument":  self.instrument,
            "state":       self.state.value,
            "primary":     self.primary,
            "risk_level":  self.risk_level,
            "actions":     [a.to_dict() for a in self.actions],
            "notes":       self.notes,
        }


class PositionMonitor:
    """Evaluates live position metrics and generates management recommendations."""

    def evaluate(
        self,
        instrument: str,
        state:      LifecycleState,
        metrics:    Optional[PositionMetrics] = None,
        context:    Optional[MarketContext]   = None,
        confidence: float                     = 0.65,
    ) -> PositionManagementPlan:

        actions: List[PositionAction] = []
        notes:   List[str]            = []
        risk_level = "MEDIUM"

        # Default primary
        primary = ManagementRecommendation.HOLD

        # ── Emergency conditions ──────────────────────────────────────────────
        if context and context.news_risk > 0.75:
            primary    = ManagementRecommendation.EXIT_IMMEDIATELY
            risk_level = "CRITICAL"
            actions.append(PositionAction(
                ManagementRecommendation.EXIT_IMMEDIATELY, 1,
                "Critical news risk",
                f"news_risk={context.news_risk:.0%} — close immediately",
                urgency="CRITICAL",
            ))
            notes.append("Critical news risk requires immediate closure")
            return PositionManagementPlan(instrument, state, primary, actions, risk_level, notes)

        if context and context.volatility > 0.035:
            primary    = ManagementRecommendation.REDUCE_EXPOSURE
            risk_level = "HIGH"
            actions.append(PositionAction(
                ManagementRecommendation.REDUCE_EXPOSURE, 1,
                "Extreme volatility",
                f"vol={context.volatility:.3f} — reduce 50% immediately",
                urgency="HIGH",
            ))
            notes.append("Extreme volatility — exposure reduction required")

        # ── State-specific logic ──────────────────────────────────────────────
        if state == LifecycleState.ACTIVE:
            if confidence > 0.75 and context and abs(context.momentum) > 0.3:
                primary = ManagementRecommendation.SCALE_IN
                actions.append(PositionAction(
                    ManagementRecommendation.SCALE_IN, 2,
                    "High confidence + momentum",
                    "Add 50% at conservative entry within risk limits",
                ))
            else:
                actions.append(PositionAction(
                    ManagementRecommendation.HOLD, 2,
                    "Trade progressing normally",
                    "Hold — monitor for TP1 approach",
                ))

        elif state == LifecycleState.PROTECTED:
            actions.append(PositionAction(
                ManagementRecommendation.MOVE_STOP, 2,
                "Trail breakeven stop",
                "Tighten stop as price advances",
            ))

        elif state == LifecycleState.PARTIAL_EXIT:
            actions.append(PositionAction(
                ManagementRecommendation.SCALE_OUT, 2,
                "TP2 approaching",
                "Prepare to close additional 30% at TP2",
            ))

        elif state == LifecycleState.RUNNER:
            actions.append(PositionAction(
                ManagementRecommendation.MOVE_STOP, 2,
                "Trail stop on runner",
                "Follow price with 1× ATR trail",
            ))
            notes.append("Runner active — let price find its own level")

        elif state == LifecycleState.TRAILING:
            actions.append(PositionAction(
                ManagementRecommendation.WAIT, 2,
                "Trail stop managing",
                "No action — trail stop is active",
            ))

        elif state == LifecycleState.EXIT_READY:
            primary = ManagementRecommendation.EXIT_IMMEDIATELY
            actions.append(PositionAction(
                ManagementRecommendation.EXIT_IMMEDIATELY, 1,
                "Exit conditions met",
                "Close remaining position",
                urgency="HIGH",
            ))

        elif state == LifecycleState.POSITION_BUILDING:
            if confidence < 0.60:
                primary = ManagementRecommendation.SCALE_OUT
                actions.append(PositionAction(
                    ManagementRecommendation.SCALE_OUT, 1,
                    "Confidence dropped during build",
                    "Reduce back to original size",
                    urgency="HIGH",
                ))
            else:
                actions.append(PositionAction(
                    ManagementRecommendation.HOLD, 2,
                    "Building phase — hold",
                    "Await full scale-in completion",
                ))

        # ── Metrics-based adjustments ─────────────────────────────────────────
        if metrics:
            if metrics.unrealised_pct < -0.015:
                risk_level = "HIGH"
                actions.append(PositionAction(
                    ManagementRecommendation.MOVE_STOP, 1,
                    "Position drawdown > 1.5%",
                    f"Drawdown={metrics.unrealised_pct:.2%} — review stop level",
                    urgency="HIGH",
                ))
            if metrics.unrealised_pct > 0.020:
                notes.append(f"Good unrealised gain {metrics.unrealised_pct:.2%} — protect it")

        # Confidence-based fallback
        if confidence < 0.50 and state in POSITION_OPEN_STATES:
            if primary == ManagementRecommendation.HOLD:
                primary = ManagementRecommendation.REDUCE_EXPOSURE
                actions.append(PositionAction(
                    ManagementRecommendation.REDUCE_EXPOSURE, 1,
                    "Confidence deteriorated",
                    f"confidence={confidence:.0%} — reduce 30%",
                    urgency="HIGH",
                ))
                risk_level = "HIGH"

        logger.debug(
            f"PositionMonitor {instrument} [{state.value}]: primary={primary} "
            f"risk={risk_level}"
        )

        return PositionManagementPlan(
            instrument = instrument,
            state      = state,
            primary    = primary,
            actions    = actions,
            risk_level = risk_level,
            notes      = notes,
        )
