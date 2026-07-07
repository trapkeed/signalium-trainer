"""Phase 21 — Execution Planner.

Orchestrates all execution engines into a single, coherent ExecutionPlan
for every forecast.

Does NOT place trades. Produces institutional-grade recommendations only.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.execution.entry_quality import EntryQualityEngine, EntryQualityResult
from trainer.execution.entry_zones import EntryZoneEngine, EntryZones
from trainer.execution.execution_simulator import ExecutionSimulator, SimulationResult
from trainer.execution.execution_timing import ExecutionTiming, TimingAssessment
from trainer.execution.execution_types import (
    BrokerConstraints,
    ExecutionForecast,
    SizingMethod,
    default_broker,
)
from trainer.execution.order_strategy import OrderRecommendation, OrderStrategyEngine
from trainer.execution.position_sizer import PositionSizer, RiskLimits, SizingResult
from trainer.execution.risk_engine import RiskEngine, RiskProfile
from trainer.execution.stop_loss_engine import StopLossEngine, StopLossRecommendation
from trainer.execution.take_profit_engine import TakeProfitEngine, TakeProfitRecommendation
from trainer.execution.trade_manager import ManagementPlan, TradeManager
from trainer.execution.trade_scorer import TradeScore, TradeScorer


@dataclass
class ExecutionPlan:
    """Complete institutional execution plan for one instrument."""

    forecast: ExecutionForecast
    zones: EntryZones
    entry_quality: EntryQualityResult
    stop_loss: StopLossRecommendation
    take_profit: TakeProfitRecommendation
    sizing: SizingResult
    management: ManagementPlan
    timing: TimingAssessment
    risk: RiskProfile
    order: OrderRecommendation
    score: TradeScore
    simulation: SimulationResult

    @property
    def instrument(self) -> str:
        return self.forecast.instrument

    @property
    def direction(self) -> str:
        return self.forecast.direction

    def is_executable(self) -> bool:
        """True if timing is not AVOID and entry quality is not REJECT."""
        from trainer.execution.execution_types import ExecutionGrade, TimingRecommendation

        return (
            self.timing.recommendation != TimingRecommendation.AVOID
            and self.entry_quality.grade != ExecutionGrade.REJECT
        )

    def executive_summary(self) -> dict:
        """One-screen institutional summary."""
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "executable": self.is_executable(),
            "timing": self.timing.recommendation.value,
            "entry_grade": self.entry_quality.grade.value,
            "trade_grade": self.score.grade.value,
            "trade_score": round(self.score.overall_score, 4),
            "primary_entry": round(self.zones.primary_entry, 5),
            "initial_stop": round(self.stop_loss.initial_stop, 5),
            "tp1": round(self.take_profit.tp1, 5),
            "tp2": round(self.take_profit.tp2, 5),
            "runner": round(self.take_profit.runner_target, 5),
            "lots": self.sizing.lots,
            "risk_pct": round(self.sizing.risk_pct, 4),
            "risk_reward": round(self.risk.risk_reward_ratio, 2),
            "prob_success": round(self.risk.prob_success, 4),
            "expected_value": round(self.risk.expected_value, 5),
            "order_type": self.order.primary_type.value,
            "management": self.management.primary_action.value,
            "expected_fill": round(self.simulation.expected_fill, 5),
            "slippage": round(self.simulation.expected_slippage, 5),
        }

    def to_dict(self) -> dict:
        return {
            "executive_summary": self.executive_summary(),
            "forecast": self.forecast.to_dict(),
            "entry_zones": self.zones.to_dict(),
            "entry_quality": self.entry_quality.to_dict(),
            "stop_loss": self.stop_loss.to_dict(),
            "take_profit": self.take_profit.to_dict(),
            "sizing": self.sizing.to_dict(),
            "management": self.management.to_dict(),
            "timing": self.timing.to_dict(),
            "risk_profile": self.risk.to_dict(),
            "order_strategy": self.order.to_dict(),
            "trade_score": self.score.to_dict(),
            "simulation": self.simulation.to_dict(),
        }


class ExecutionPlanner:
    """Central orchestrator — converts a forecast into a full ExecutionPlan.

    Usage::

        planner = ExecutionPlanner()
        plan = planner.plan(forecast)
        print(plan.executive_summary())
    """

    def __init__(
        self,
        broker: BrokerConstraints | None = None,
        risk_limits: RiskLimits | None = None,
        sizing_method: SizingMethod = SizingMethod.FIXED_RISK,
    ):
        self._broker = broker or default_broker()
        self._risk_limits = risk_limits or RiskLimits()
        self._sizing_method = sizing_method

        self._zone_engine = EntryZoneEngine()
        self._quality_engine = EntryQualityEngine()
        self._stop_engine = StopLossEngine()
        self._tp_engine = TakeProfitEngine()
        self._sizer = PositionSizer(self._broker, self._risk_limits)
        self._manager = TradeManager()
        self._timer = ExecutionTiming(self._broker)
        self._risk_engine = RiskEngine()
        self._order_engine = OrderStrategyEngine()
        self._scorer = TradeScorer()
        self._simulator = ExecutionSimulator(self._broker)

    def plan(self, forecast: ExecutionForecast) -> ExecutionPlan:
        """Generate a complete institutional execution plan."""

        logger.info(f"ExecutionPlanner: planning {forecast.instrument} {forecast.direction}")

        # Step 1: Entry zones
        zones = self._zone_engine.calculate(forecast)

        # Step 2: Entry quality (uses zones)
        quality = self._quality_engine.evaluate(forecast, zones)

        # Step 3: Stop loss (uses zones)
        stop = self._stop_engine.recommend(forecast, zones)

        # Step 4: Take profit (uses zones + stop)
        tp = self._tp_engine.recommend(forecast, zones, stop)

        # Step 5: Position sizing (uses stop)
        sizing = self._sizer.calculate(forecast, stop, self._sizing_method)

        # Step 6: Trade management (uses sizing + stop + tp)
        management = self._manager.plan(forecast, sizing, stop, tp)

        # Step 7: Execution timing
        timing = self._timer.evaluate(forecast)

        # Step 8: Risk profile (uses stop + tp + sizing)
        risk = self._risk_engine.analyze(forecast, stop, tp, sizing.lots)

        # Step 9: Order strategy (uses zones + timing)
        order = self._order_engine.recommend(forecast, zones, timing, self._broker)

        # Step 10: Trade score (uses quality + risk + timing)
        score = self._scorer.score(forecast, quality, risk, timing)

        # Step 11: Simulation (uses zones + stop + tp)
        simulation = self._simulator.simulate(forecast, zones, stop, tp)

        plan = ExecutionPlan(
            forecast=forecast,
            zones=zones,
            entry_quality=quality,
            stop_loss=stop,
            take_profit=tp,
            sizing=sizing,
            management=management,
            timing=timing,
            risk=risk,
            order=order,
            score=score,
            simulation=simulation,
        )

        logger.info(
            f"ExecutionPlanner: {forecast.instrument} plan complete — "
            f"grade={score.grade.value} executable={plan.is_executable()}"
        )

        return plan

    def plan_batch(self, forecasts: list[ExecutionForecast]) -> list[ExecutionPlan]:
        """Generate plans for a list of forecasts."""
        plans = [self.plan(f) for f in forecasts]
        plans.sort(key=lambda p: p.score.overall_score, reverse=True)
        return plans
