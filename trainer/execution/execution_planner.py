"""Phase 21 — Execution Planner.

Orchestrates all execution engines into a single, coherent ExecutionPlan
for every forecast.

Under the unified QDE architecture (see `docs/qde/unified_qde_architecture.md`),
`ExecutionPlanner` is a pure *executor* of the active QDE model's decision.
Supporting engines (zones, stop/tp, timing, order strategy, scoring,
simulation) still run — they supply execution mechanics (concrete price
levels, fill/timing detail) and act as context fed into the QDE decision —
but entry/exit/scale/size/risk-allocation/management are decided exactly
once, by the QDE, and simply carried out here.

Does NOT place trades. Produces institutional-grade recommendations only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from trainer.lifecycle.lifecycle_models import MarketContext, PositionMetrics
from trainer.memory.decision_memory import DecisionMemory
from trainer.portfolio.portfolio_scorer import PortfolioMetrics

# `trainer.qde` imports `trainer.execution.execution_types` (for the shared
# `ManagementAction` enum), which — because Python must fully run a
# package's `__init__.py` before handing back any submodule — would
# otherwise re-enter this very module while `trainer.execution` is still
# initializing. Importing the QDE pieces lazily (inside functions/methods
# below, not here at module scope) keeps both packages independently
# importable regardless of which one loads first.
if TYPE_CHECKING:
    from trainer.qde.qde_inference import (
        QDEAccountSnapshot,
        QDEDecision,
        QDEDecisionModel,
        QDEInferenceInput,
    )
    from trainer.qde.qde_registry import QDERegistry


@dataclass
class ExecutionPlan:
    """Complete institutional execution plan for one instrument.

    `decision` is the single unified QDE output; every other field is
    supporting execution detail that carries the decision out (or, for
    `sizing`/`management`, is populated directly from it).
    """

    forecast: ExecutionForecast
    decision: QDEDecision
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
        """True iff the QDE decided to enter (or scale in) this instrument.

        Timing/entry-quality grades remain available as informational
        context on the plan, but they no longer independently gate
        execution — the QDE decision already accounted for that context
        when it decided.
        """
        return self.decision.is_entry or self.decision.scale_action.value == "SCALE_IN"

    def executive_summary(self) -> dict:
        """One-screen institutional summary."""
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "executable": self.is_executable(),
            "decision": self.decision.entry_action.value,
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
            "confidence": round(self.decision.confidence, 4),
        }

    def to_dict(self) -> dict:
        return {
            "executive_summary": self.executive_summary(),
            "forecast": self.forecast.to_dict(),
            "decision": self.decision.to_dict(),
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

    The planner asks the QDE registry for the currently active decision
    model and executes exactly what it decides; it does not compute entry,
    exit, scale, sizing, or management decisions itself.
    """

    def __init__(
        self,
        broker: BrokerConstraints | None = None,
        risk_limits: RiskLimits | None = None,
        sizing_method: SizingMethod = SizingMethod.FIXED_RISK,
        qde_registry: "QDERegistry | None" = None,
        decision_model: "QDEDecisionModel | None" = None,
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

        # `decision_model` allows direct injection (tests, or a caller that
        # already resolved a model); otherwise the registry's active-model
        # lookup is consulted on every `plan()` call so a promotion/rollback
        # takes effect immediately without recreating the planner. Imported
        # lazily — see the module-level note above on the qde/execution
        # import cycle.
        self._decision_model = decision_model
        self._qde_registry = qde_registry if decision_model is None else None
        if self._decision_model is None and self._qde_registry is None:
            from trainer.qde.qde_registry import QDERegistry

            self._qde_registry = QDERegistry()

    def _resolve_decision_model(self) -> "QDEDecisionModel":
        if self._decision_model is not None:
            return self._decision_model
        return self._qde_registry.get_active_model()

    def _build_inference_input(
        self,
        forecast: ExecutionForecast,
        zones: EntryZones,
        stop: StopLossRecommendation,
        tp: TakeProfitRecommendation,
        position: PositionMetrics | None,
        portfolio: PortfolioMetrics | None,
        account: "QDEAccountSnapshot | None",
        decision_memory: DecisionMemory | None,
    ) -> "QDEInferenceInput":
        from trainer.qde.qde_inference import (
            QDEAccountSnapshot,
            QDEInferenceInput,
            QDEMemorySnapshot,
            default_portfolio_snapshot,
        )

        market = MarketContext(
            price=forecast.current_price,
            atr=forecast.atr,
            spread=forecast.spread,
            volatility=forecast.volatility,
            momentum=forecast.momentum,
            news_risk=forecast.news_risk,
            session=forecast.session.value,
            trend_valid=True,
        )

        memory_snapshot = QDEMemorySnapshot(instrument=forecast.instrument)
        if decision_memory is not None:
            records = decision_memory.filter(instrument=forecast.instrument, limit=50)
            memory_snapshot = QDEMemorySnapshot.from_records(forecast.instrument, records)

        return QDEInferenceInput(
            forecast=forecast,
            market=market,
            portfolio=portfolio or default_portfolio_snapshot(),
            position=position or PositionMetrics(),
            account=account
            or QDEAccountSnapshot(
                balance=forecast.account_size,
                equity=forecast.account_size,
                margin_available=forecast.account_size,
            ),
            memory=memory_snapshot,
            entry_price_hint=zones.primary_entry,
            stop_price_hint=stop.initial_stop,
            take_profit_hint=tp.tp1,
        )

    def plan(
        self,
        forecast: ExecutionForecast,
        position: PositionMetrics | None = None,
        portfolio: PortfolioMetrics | None = None,
        account: QDEAccountSnapshot | None = None,
        decision_memory: DecisionMemory | None = None,
    ) -> ExecutionPlan:
        """Generate a complete institutional execution plan.

        Args:
            forecast: Raw instrument forecast/context from upstream models.
            position: Live metrics for an existing open position, if any.
            portfolio: Current portfolio-level metrics, if any.
            account: Live account state; defaults to a snapshot derived
                from `forecast.account_size` when omitted.
            decision_memory: Decision history store, used to build the
                memory snapshot handed to the QDE.
        """

        logger.info(f"ExecutionPlanner: planning {forecast.instrument} {forecast.direction}")

        # ── Supporting execution mechanics (context for, and carried out by,
        # the QDE decision — never an independent decision source) ─────────
        zones = self._zone_engine.calculate(forecast)
        quality = self._quality_engine.evaluate(forecast, zones)
        stop = self._stop_engine.recommend(forecast, zones)
        tp = self._tp_engine.recommend(forecast, zones, stop)
        timing = self._timer.evaluate(forecast)

        # ── The single unified decision ──────────────────────────────────
        inference_input = self._build_inference_input(
            forecast, zones, stop, tp, position, portfolio, account, decision_memory
        )
        model = self._resolve_decision_model()
        decision = model.decide(inference_input)

        # S22B: structural (not conventional) guarantees that shadow/
        # challenger decisions can never reach live execution. Two
        # independent checks, since either alone can be defeated:
        #
        # 1. `is_shadow` — set by `ShadowDispatcher` on the challenger side
        #    of a dual-dispatch comparison. Catches a challenger decision
        #    that was explicitly produced via the shadow-testing path and
        #    accidentally routed here.
        # 2. Champion-identity — when this planner resolves its model via
        #    the live `QDERegistry` (the real runtime path, not a
        #    test/caller-injected `decision_model`), re-verify at decide
        #    time that the model that actually produced `decision` is still
        #    the registry's official active/champion version. This catches
        #    a miswired or corrupted active-model pointer that would
        #    otherwise silently produce a non-shadow-flagged decision from
        #    a non-champion model — a case `is_shadow` alone cannot detect,
        #    since only `ShadowDispatcher` ever sets that flag.
        #
        # Raising here (rather than silently skipping) surfaces any misuse
        # immediately instead of producing a confusing partial/empty plan.
        if decision.is_shadow:
            raise RuntimeError(
                f"ExecutionPlanner: refusing to execute a shadow/challenger "
                f"decision (instrument={forecast.instrument!r}, "
                f"model_version_id={decision.model_version_id!r}). Shadow "
                f"decisions are observation-only — see "
                f"trainer.qde.qde_shadow.ShadowDispatcher."
            )
        if self._qde_registry is not None:
            active_version = self._qde_registry.get_active()
            if (
                active_version is not None
                and decision.model_version_id
                and decision.model_version_id != active_version.version_id
            ):
                raise RuntimeError(
                    f"ExecutionPlanner: refusing to execute a decision from "
                    f"model_version_id={decision.model_version_id!r} — the "
                    f"registry's active/champion version is "
                    f"{active_version.version_id!r}. This indicates a "
                    f"miswired or stale active-model pointer; only the "
                    f"champion model may ever reach live execution."
                )

        # ── Execute the decision ─────────────────────────────────────────
        sizing = self._sizer.from_decision(forecast, decision, stop)
        management = self._manager.plan_from_decision(forecast, decision, sizing, stop, tp)

        # ── Remaining execution mechanics, informed by the executed size ──
        risk = self._risk_engine.analyze(forecast, stop, tp, sizing.lots)
        order = self._order_engine.recommend(forecast, zones, timing, self._broker)
        score = self._scorer.score(forecast, quality, risk, timing)
        simulation = self._simulator.simulate(forecast, zones, stop, tp)

        plan = ExecutionPlan(
            forecast=forecast,
            decision=decision,
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
            f"decision={decision.entry_action.value} executable={plan.is_executable()}"
        )

        return plan

    def plan_batch(
        self,
        forecasts: list[ExecutionForecast],
        decision_memory: DecisionMemory | None = None,
    ) -> list[ExecutionPlan]:
        """Generate plans for a list of forecasts."""
        plans = [self.plan(f, decision_memory=decision_memory) for f in forecasts]
        plans.sort(key=lambda p: p.score.overall_score, reverse=True)
        return plans
