"""Phase 24 — ICARI Unified API.

Institutional Capital Allocation & Risk Intelligence Platform.

Endpoints:
1. allocate          — capital allocation decision
2. size_position     — position sizing
3. portfolio_risk    — portfolio VaR/CVaR/metrics
4. capital_health    — composite health assessment
5. risk_budgets      — budget management
6. optimize          — portfolio weight optimization
7. scenario          — scenario stress analysis
8. get_dashboard     — dashboard views
9. system_stats      — platform statistics
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_types import (
    SizingMethod,
    RiskBudgetType,
    DrawdownMode,
    ScenarioType,
    OptimizationObjective,
    AllocationRecommendation,
    AllocationStatus,
)
from trainer.icari.icari_models import (
    AllocationRequest,
    AllocationResult,
    PortfolioRisk,
    CapitalHealth,
    DrawdownEvent,
    OptimizationResult,
    ScenarioResult,
)
from trainer.icari.allocation_engine import AllocationEngine
from trainer.icari.risk_budget_engine import RiskBudgetEngine
from trainer.icari.position_sizing_engine import PositionSizingEngine
from trainer.icari.dynamic_sizer import DynamicSizer
from trainer.icari.capital_efficiency import CapitalEfficiencyEngine
from trainer.icari.portfolio_risk_engine import PortfolioRiskEngine
from trainer.icari.drawdown_manager import DrawdownManager
from trainer.icari.correlation_engine import CorrelationEngine
from trainer.icari.scenario_engine import ScenarioEngine
from trainer.icari.allocation_optimizer import AllocationOptimizer
from trainer.icari.capital_health import CapitalHealthEngine
from trainer.icari.icari_database import IcariDatabase
from trainer.icari.icari_dashboard import IcariDashboard


class IcariAPI:
    """Unified ICARI API — institutional capital allocation & risk intelligence."""

    def __init__(
        self,
        capital: float = 100_000.0,
        db_path: str = ":memory:",
        sizing_method: SizingMethod = SizingMethod.FIXED_RISK,
        soft_dd_limit: float = 0.05,
        hard_dd_limit: float = 0.10,
    ):
        self._capital = capital
        self._db = IcariDatabase(db_path)
        self._dashboard = IcariDashboard()

        # Sub-systems
        self._allocator = AllocationEngine()
        self._budget = RiskBudgetEngine(capital)
        self._sizer = PositionSizingEngine(method=sizing_method)
        self._dynamic_sizer = DynamicSizer(self._sizer, soft_dd_limit, hard_dd_limit)
        self._efficiency = CapitalEfficiencyEngine(capital)
        self._risk_engine = PortfolioRiskEngine()
        self._dd_manager = DrawdownManager(capital, soft_dd_limit, hard_dd_limit)
        self._corr_engine = CorrelationEngine()
        self._scenario = ScenarioEngine(capital)
        self._optimizer = AllocationOptimizer()
        self._health_engine = CapitalHealthEngine()

        # State
        self._latest_risk: Optional[PortfolioRisk] = None
        self._latest_health: Optional[CapitalHealth] = None
        self._opt_results: List[OptimizationResult] = []
        self._scenario_results: List[ScenarioResult] = []

    # ── 1. Allocate ───────────────────────────────────────────────────────────

    def allocate(self, request: AllocationRequest) -> Dict:
        """Run the full allocation pipeline for a single request."""
        # Check drawdown lockdown
        if self._dd_manager.is_locked():
            from trainer.icari.icari_models import AllocationResult

            result = AllocationResult(
                instrument=request.instrument,
                status=AllocationStatus.DENIED,
                allocation_frac=0.0,
                allocation_usd=0.0,
                confidence=0.0,
                rationale="EMERGENCY_LOCKDOWN: trading suspended",
                recommendations=[AllocationRecommendation.ACTIVATE_PRESERVATION],
            )
            self._db.save_allocation(result)
            return result.to_dict()

        result = self._allocator.allocate(request)
        self._db.save_allocation(result)

        # Consume risk budget
        if result.status != AllocationStatus.DENIED:
            self._budget.consume(RiskBudgetType.DAILY, result.allocation_frac * 0.1)
            self._budget.consume(RiskBudgetType.INSTRUMENT, result.allocation_frac)
            self._budget.consume(RiskBudgetType.OPEN_TRADE, result.allocation_frac)

        logger.info(
            f"IcariAPI.allocate: {request.instrument} → {result.status.value} alloc={result.allocation_frac:.4f}"
        )
        return result.to_dict()

    def allocate_batch(self, requests: List[AllocationRequest]) -> List[Dict]:
        return [self.allocate(r) for r in requests]

    # ── 2. Position Sizing ────────────────────────────────────────────────────

    def size_position(
        self,
        request: AllocationRequest,
        method: Optional[SizingMethod] = None,
        use_dynamic: bool = True,
        current_drawdown: float = 0.0,
        session: str = "LONDON",
    ) -> Dict:
        if use_dynamic:
            result = self._dynamic_sizer.size(
                request,
                current_drawdown=current_drawdown,
                session=session,
            )
        else:
            result = self._sizer.size(request, method=method)
        return result.to_dict()

    def compare_sizing_methods(self, request: AllocationRequest) -> Dict:
        """Compare all sizing methods for a single request."""
        return {
            method: self._sizer.size(request, method=SizingMethod(method)).to_dict()
            for method in [m.value for m in SizingMethod if m != SizingMethod.CUSTOM]
        }

    # ── 3. Portfolio Risk ─────────────────────────────────────────────────────

    def portfolio_risk(
        self,
        returns: Dict[str, List[float]],
        weights: Dict[str, float],
    ) -> Dict:
        risk = self._risk_engine.calculate(returns, weights)
        self._latest_risk = risk
        self._db.save_portfolio_risk(risk)
        return risk.to_dict()

    def correlation_matrix(self, returns: Dict[str, List[float]]) -> Dict:
        return self._corr_engine.correlation_matrix(returns)

    def correlation_clusters(self, returns: Dict[str, List[float]]) -> List:
        clusters = self._corr_engine.correlation_clusters(returns)
        return [list(c) for c in clusters]

    # ── 4. Capital Health ─────────────────────────────────────────────────────

    def capital_health(
        self,
        utilisation: float = 0.5,
        current_drawdown: float = 0.0,
        num_instruments: int = 1,
        avg_correlation: float = 0.0,
        recent_returns: Optional[List[float]] = None,
    ) -> Dict:
        pr = self._latest_risk or PortfolioRisk()
        mode = self._dd_manager.mode
        alloc_history = [r.to_dict() for r in self._allocator.history(last_n=20)]

        health = self._health_engine.score(
            portfolio_risk=pr,
            utilisation=utilisation,
            drawdown_mode=mode,
            current_drawdown=current_drawdown,
            num_instruments=num_instruments,
            avg_correlation=avg_correlation,
            recent_returns=recent_returns,
            allocation_results=alloc_history,
        )
        self._latest_health = health
        self._db.save_health(health)
        return health.to_dict()

    # ── 5. Risk Budgets ───────────────────────────────────────────────────────

    def get_budgets(self) -> Dict:
        return self._budget.summary()

    def set_budget(self, budget_type: RiskBudgetType, limit: float, label: str = "") -> Dict:
        budget = self._budget.set_budget(budget_type, limit, label)
        self._db.save_budget(budget)
        return budget.to_dict()

    def check_budget(self, budget_type: RiskBudgetType, amount: float) -> Dict:
        return self._budget.check(budget_type, amount)

    def reset_budget(self, budget_type: RiskBudgetType) -> bool:
        return self._budget.reset(budget_type)

    def reset_all_budgets(self) -> None:
        self._budget.reset_all()

    # ── 6. Optimize ───────────────────────────────────────────────────────────

    def optimize(
        self,
        objective: OptimizationObjective,
        returns: Dict[str, List[float]],
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        result = self._optimizer.optimize(objective, returns, initial_weights)
        self._opt_results.append(result)
        self._db.save_optimization(result)
        return result.to_dict()

    def optimize_all(self, returns: Dict[str, List[float]]) -> Dict:
        results = self._optimizer.compare_objectives(returns)
        for r in results.values():
            self._opt_results.append(r)
            self._db.save_optimization(r)
        return {k: v.to_dict() for k, v in results.items()}

    # ── 7. Scenario ───────────────────────────────────────────────────────────

    def scenario(
        self,
        scenario: ScenarioType,
        weights: Dict[str, float],
        volatilities: Dict[str, float],
        currencies: Optional[Dict[str, str]] = None,
    ) -> Dict:
        result = self._scenario.run(scenario, weights, volatilities, currencies)
        self._scenario_results.append(result)
        self._db.save_scenario(result)
        return result.to_dict()

    def scenario_all(
        self,
        weights: Dict[str, float],
        volatilities: Dict[str, float],
        currencies: Optional[Dict[str, str]] = None,
    ) -> Dict:
        results = self._scenario.run_all(weights, volatilities, currencies)
        for r in results:
            self._scenario_results.append(r)
            self._db.save_scenario(r)
        summary = self._scenario.resilience_summary(results)
        summary["scenarios"] = [r.to_dict() for r in results]
        return summary

    # ── Drawdown ──────────────────────────────────────────────────────────────

    def update_equity(self, equity: float) -> Dict:
        return self._dd_manager.update(equity)

    def drawdown_stats(self) -> Dict:
        return self._dd_manager.stats()

    # ── Capital Efficiency ────────────────────────────────────────────────────

    def capital_efficiency(
        self,
        allocated_usd: float,
        at_risk_usd: float,
        realised_pnl: float,
        unrealised_pnl: float,
        num_open: int,
        num_closed: int = 0,
        period_days: float = 1.0,
    ) -> Dict:
        metrics = self._efficiency.analyse(
            allocated_usd,
            at_risk_usd,
            realised_pnl,
            unrealised_pnl,
            num_open,
            num_closed,
            period_days,
        )
        metrics["efficiency_score"] = self._efficiency.efficiency_score(metrics)
        return metrics

    # ── 8. Dashboard ──────────────────────────────────────────────────────────

    def get_dashboard(self, name: str = "all") -> Dict:
        results = self._allocator.history()
        pr = self._latest_risk or PortfolioRisk()
        budgets = self._budget.all_budgets()
        weights: Dict[str, float] = {}
        rc: Dict[str, float] = {}
        dd_events = self._dd_manager.events()
        health = self._latest_health or CapitalHealth()

        views = {
            "capital_allocation": lambda: self._dashboard.capital_allocation(results),
            "risk_dashboard": lambda: self._dashboard.risk_dashboard(pr, budgets),
            "exposure_dashboard": lambda: self._dashboard.exposure_dashboard(weights, rc),
            "drawdown_dashboard": lambda: self._dashboard.drawdown_dashboard(
                dd_events,
                self._dd_manager.current_drawdown,
                self._dd_manager.mode,
                self._dd_manager.risk_multiplier,
            ),
            "capital_health": lambda: self._dashboard.capital_health_dashboard(health),
            "optimization": lambda: self._dashboard.optimization_dashboard(self._opt_results),
            "scenario_analysis": lambda: self._dashboard.scenario_dashboard(self._scenario_results),
        }

        if name == "all":
            return self._dashboard.all_dashboards(
                results,
                pr,
                budgets,
                weights,
                rc,
                dd_events,
                self._dd_manager.current_drawdown,
                self._dd_manager.mode,
                self._dd_manager.risk_multiplier,
                health,
                self._opt_results,
                self._scenario_results,
            )

        fn = views.get(name)
        return fn() if fn else {"error": f"Unknown dashboard '{name}'"}

    # ── 9. System Stats ───────────────────────────────────────────────────────

    def system_stats(self) -> Dict:
        return {
            "capital": self._capital,
            "database": self._db.stats(),
            "drawdown": self._dd_manager.stats(),
            "budget_summary": self._budget.summary(),
            "allocation_count": len(self._allocator.history()),
            "opt_results": len(self._opt_results),
            "scenario_results": len(self._scenario_results),
        }
