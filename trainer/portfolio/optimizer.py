"""Phase 20 — Portfolio Optimizer.

Supports configurable optimization objectives:
  - Maximum Sharpe
  - Minimum Volatility
  - Maximum Diversification
  - Maximum Risk-Adjusted Return
  - Balanced Objective

Uses pure stdlib arithmetic (no scipy/numpy required).
Implements grid-search + gradient-free optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trainer.portfolio.correlation_engine import CorrelationMatrix
from trainer.portfolio.portfolio_instruments import ModelForecast
from trainer.portfolio.portfolio_scorer import PortfolioMetrics, PortfolioScorer
from trainer.portfolio.risk_budget import RiskBudgetConfig, RiskBudgetEngine


class OptimizationObjective(str, Enum):
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_DIVERSIFICATION = "max_diversification"
    MAX_RISK_ADJUSTED_RETURN = "max_risk_adjusted_return"
    BALANCED = "balanced"


@dataclass
class OptimizationResult:
    objective: str
    optimal_weights: dict[str, float]
    optimal_metrics: PortfolioMetrics
    iterations: int
    converged: bool
    objective_value: float
    weight_history: list[tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "optimal_weights": {k: round(v, 4) for k, v in self.optimal_weights.items()},
            "optimal_metrics": self.optimal_metrics.to_dict(),
            "iterations": self.iterations,
            "converged": self.converged,
            "objective_value": round(self.objective_value, 4),
        }


class PortfolioOptimizer:
    """Optimizes portfolio weights using grid-search approximation."""

    MAX_ITER = 200
    GRID_STEPS = 5
    CONVERGENCE_TOL = 1e-5

    def __init__(
        self,
        risk_free_rate: float = 0.045,
        budget_config: RiskBudgetConfig | None = None,
    ) -> None:
        self._rfr = risk_free_rate
        self._scorer = PortfolioScorer(risk_free_rate=risk_free_rate)
        self._budget_engine = RiskBudgetEngine(budget_config or RiskBudgetConfig())

    def optimize(
        self,
        forecasts: list[ModelForecast],
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        correlation_matrix: CorrelationMatrix | None = None,
        long_only: bool = True,
        max_weight: float = 0.40,
        min_weight: float = 0.01,
    ) -> OptimizationResult:
        active = [f for f in forecasts if f.direction != "NEUTRAL"]
        if not active:
            return self._empty_result(objective)

        n = len(active)
        # Start from equal weights
        weights = {f.instrument: 1.0 / n for f in active}
        best_weights = dict(weights)
        best_obj = self._objective_value(active, weights, objective, correlation_matrix)
        history = [(0, best_obj)]
        converged = False

        for iteration in range(1, self.MAX_ITER + 1):
            improved = False
            for i, fc_i in enumerate(active):
                sym = fc_i.instrument
                current_w = weights[sym]
                best_local = current_w
                best_local_obj = best_obj

                # Try a grid of weight adjustments
                for delta in [-0.10, -0.05, -0.02, +0.02, +0.05, +0.10]:
                    candidate_w = current_w + delta
                    if candidate_w < min_weight or candidate_w > max_weight:
                        continue
                    candidate = dict(weights)
                    candidate[sym] = candidate_w
                    candidate = self._renorm(candidate, sym, min_weight, max_weight)
                    if not self._budget_feasible(candidate, active):
                        continue
                    obj = self._objective_value(active, candidate, objective, correlation_matrix)
                    if obj > best_local_obj + 1e-8:
                        best_local_obj = obj
                        best_local = candidate_w
                        improved = True

                if best_local != current_w:
                    weights[sym] = best_local
                    weights = self._renorm(weights, sym, min_weight, max_weight)
                    best_obj = best_local_obj
                    best_weights = dict(weights)

            history.append((iteration, best_obj))
            if not improved:
                converged = True
                break

        metrics = self._scorer.score(active, best_weights, correlation_matrix)
        return OptimizationResult(
            objective=objective.value,
            optimal_weights=best_weights,
            optimal_metrics=metrics,
            iterations=len(history),
            converged=converged,
            objective_value=best_obj,
            weight_history=history[-20:],
        )

    def compare_objectives(
        self,
        forecasts: list[ModelForecast],
        correlation_matrix: CorrelationMatrix | None = None,
    ) -> dict[str, OptimizationResult]:
        results = {}
        for obj in OptimizationObjective:
            results[obj.value] = self.optimize(forecasts, obj, correlation_matrix)
        return results

    def efficient_frontier(
        self,
        forecasts: list[ModelForecast],
        n_points: int = 10,
        correlation_matrix: CorrelationMatrix | None = None,
    ) -> list[dict]:
        active = [f for f in forecasts if f.direction != "NEUTRAL"]
        if not active:
            return []
        n = len(active)
        frontier = []
        for i in range(n_points):
            # Interpolate from min-vol to max-return allocation
            risk_weight = 1.0 - i / max(n_points - 1, 1)
            return_weight = i / max(n_points - 1, 1)

            weights: dict[str, float] = {}
            for fc in active:
                inv_risk = 1.0 / max(fc.expected_risk, 1e-4)
                ret_score = max(fc.signed_return() * fc.confidence, 1e-6)
                blended = risk_weight * inv_risk + return_weight * ret_score
                weights[fc.instrument] = blended

            total = sum(weights.values())
            norm_w = {k: v / max(total, 1e-9) for k, v in weights.items()}
            metrics = self._scorer.score(active, norm_w, correlation_matrix)
            frontier.append(
                {
                    "point": i,
                    "risk_weight": round(risk_weight, 3),
                    "return_weight": round(return_weight, 3),
                    "expected_return": round(metrics.expected_return, 4),
                    "expected_volatility": round(metrics.expected_volatility, 4),
                    "expected_sharpe": round(metrics.expected_sharpe, 4),
                }
            )
        return frontier

    # ── Objective functions ────────────────────────────────────────────────

    def _objective_value(
        self,
        forecasts: list[ModelForecast],
        weights: dict[str, float],
        objective: OptimizationObjective,
        matrix: CorrelationMatrix | None,
    ) -> float:
        metrics = self._scorer.score(forecasts, weights, matrix)
        if objective == OptimizationObjective.MAX_SHARPE:
            return metrics.expected_sharpe
        if objective == OptimizationObjective.MIN_VOLATILITY:
            return -metrics.expected_volatility
        if objective == OptimizationObjective.MAX_DIVERSIFICATION:
            return metrics.diversification_score
        if objective == OptimizationObjective.MAX_RISK_ADJUSTED_RETURN:
            dd = max(metrics.expected_drawdown, 1e-9)
            return metrics.expected_return / dd
        if objective == OptimizationObjective.BALANCED:
            return (
                0.35 * metrics.expected_sharpe
                + 0.30 * metrics.diversification_score
                + 0.20 * (metrics.expected_return / max(metrics.expected_volatility, 1e-9))
                + 0.15 * metrics.weighted_confidence
            )
        return metrics.expected_sharpe

    @staticmethod
    def _renorm(
        weights: dict[str, float], changed: str, min_w: float, max_w: float
    ) -> dict[str, float]:
        total = sum(weights.values())
        if total < 1e-9:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: max(min_w, min(max_w, v / total)) for k, v in weights.items()}

    def _budget_feasible(self, weights: dict[str, float], forecasts: list[ModelForecast]) -> bool:
        result = self._budget_engine.check(weights, forecasts)
        return result.is_feasible()

    @staticmethod
    def _empty_result(objective: OptimizationObjective) -> OptimizationResult:
        empty_metrics = PortfolioScorer._empty_metrics()
        return OptimizationResult(
            objective=objective.value,
            optimal_weights={},
            optimal_metrics=empty_metrics,
            iterations=0,
            converged=False,
            objective_value=0.0,
        )
