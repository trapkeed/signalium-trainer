"""Phase 20 — Portfolio API.

Unified API entry point for the Portfolio Intelligence Engine.

Endpoints:
  - recommend()         — full portfolio recommendation
  - exposure()          — exposure breakdown
  - correlation()       — correlation matrix
  - optimize()          — portfolio optimization
  - scenario()          — scenario analysis
  - allocation_history()— allocation history
  - conflict_check()    — signal conflict detection
  - regime()            — current regime assessment
  - risk_budget_check() — budget utilisation
  - dashboard_data()    — all visualisation data
  - stats()             — system statistics
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trainer.portfolio.conflict_resolver import ConflictResolver
from trainer.portfolio.correlation_engine import (
    CorrelationEngine,
    CorrelationMethod,
)
from trainer.portfolio.dependency_graph import DependencyGraphBuilder
from trainer.portfolio.exposure_engine import ExposureEngine
from trainer.portfolio.optimizer import OptimizationObjective, PortfolioOptimizer
from trainer.portfolio.portfolio_dashboard import PortfolioDashboard
from trainer.portfolio.portfolio_database import PortfolioDatabase
from trainer.portfolio.portfolio_instruments import (
    InstrumentRegistry,
    ModelForecast,
)
from trainer.portfolio.portfolio_scorer import PortfolioScorer
from trainer.portfolio.position_allocator import AllocationMethod, PositionAllocator
from trainer.portfolio.regime_detector import RegimeDetector
from trainer.portfolio.risk_budget import RiskBudgetConfig, RiskBudgetEngine
from trainer.portfolio.scenario_analysis import ScenarioAnalyser, ScenarioType

try:
    from loguru import logger
except ImportError:
    import types as _types

    _log = _types.ModuleType("loguru")

    class _L:
        def info(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

    _log.logger = _L()
    logger = _log.logger


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(data: Any) -> dict:
    return {"status": "ok", "data": data}


def _err(message: str, exc: Optional[Exception] = None) -> dict:
    logger.error(f"PortfolioAPI error: {message}" + (f" — {exc}" if exc else ""))
    return {"status": "error", "message": message}


class PortfolioApi:
    """Unified entry point for Phase 20 Portfolio Intelligence."""

    def __init__(
        self,
        db_path: str = ":memory:",
        registry: Optional[InstrumentRegistry] = None,
        budget_config: Optional[RiskBudgetConfig] = None,
        risk_free_rate: float = 0.045,
        correlation_window: int = 60,
    ) -> None:
        self._registry = registry or InstrumentRegistry.default()
        self._budget_config = budget_config or RiskBudgetConfig()
        self._rfr = risk_free_rate

        # Engines
        self._corr_engine = CorrelationEngine(window=correlation_window)
        self._exposure_engine = ExposureEngine(registry=self._registry)
        self._dep_graph = DependencyGraphBuilder()
        self._regime_detector = RegimeDetector()
        self._conflict_resolver = ConflictResolver(registry=self._registry)
        self._scorer = PortfolioScorer(risk_free_rate=risk_free_rate)
        self._allocator = PositionAllocator(
            budget_config=self._budget_config, risk_free_rate=risk_free_rate
        )
        self._scenario = ScenarioAnalyser(risk_free_rate=risk_free_rate)
        self._optimizer = PortfolioOptimizer(
            risk_free_rate=risk_free_rate, budget_config=self._budget_config
        )
        self._db = PortfolioDatabase(db_path)
        self._dashboard = PortfolioDashboard(db=self._db)

        # Cache: last scored state
        self._last_forecasts: List[ModelForecast] = []
        self._last_correlation_matrix = None

    # ── Core Recommend ─────────────────────────────────────────────────────

    def recommend(
        self,
        forecasts: List[ModelForecast],
        method: str = "hybrid",
        run_conflict_check: bool = True,
        run_budget_check: bool = True,
        timestamp: Optional[str] = None,
    ) -> dict:
        try:
            ts = timestamp or _now()
            self._last_forecasts = list(forecasts)

            # 1. Conflict resolution
            conflict_report = None
            active_forecasts = forecasts
            if run_conflict_check:
                conflict_report = self._conflict_resolver.resolve(forecasts)
                active_forecasts = conflict_report.adjusted_forecasts

            # 2. Correlation matrix
            for fc in active_forecasts:
                self._corr_engine.add_observation(fc.instrument, fc.signed_return())
            matrix = self._corr_engine.compute_matrix(
                [fc.instrument for fc in active_forecasts],
                method=CorrelationMethod.ROLLING,
                timestamp=ts,
            )
            self._last_correlation_matrix = matrix

            # 3. Regime detection
            for fc in active_forecasts:
                self._regime_detector.add_return(fc.instrument, fc.signed_return())
            regime = self._regime_detector.detect(timestamp=ts)

            # 4. Allocation
            alloc_method = (
                AllocationMethod(method)
                if method in [m.value for m in AllocationMethod]
                else AllocationMethod.HYBRID
            )
            allocation = self._allocator.allocate(active_forecasts, matrix, regime, alloc_method)

            # 5. Exposure
            exposure = self._exposure_engine.compute(active_forecasts, allocation.signed_weights)

            # 6. Portfolio scoring
            metrics = self._scorer.score(active_forecasts, allocation.weights, matrix)

            # 7. Dependency graph
            graph = self._dep_graph.build(matrix, timestamp=ts)

            rec_id = str(uuid.uuid4())
            result = {
                "rec_id": rec_id,
                "timestamp": ts,
                "method": method,
                "instrument_count": len(active_forecasts),
                "regime": regime.to_dict(),
                "allocation": allocation.to_dict(),
                "portfolio_metrics": metrics.to_dict(),
                "exposure": exposure.to_dict(),
                "conflict_report": conflict_report.to_dict() if conflict_report else None,
                "dependency_graph": graph.to_dict(),
                "weights": allocation.signed_weights,
                "expected_return": metrics.expected_return,
                "expected_volatility": metrics.expected_volatility,
                "expected_sharpe": metrics.expected_sharpe,
                "diversification_score": metrics.diversification_score,
            }

            # Persist
            self._db.save_recommendation(
                {
                    "rec_id": rec_id,
                    "timestamp": ts,
                    "instrument_count": len(active_forecasts),
                    "expected_return": metrics.expected_return,
                    "expected_volatility": metrics.expected_volatility,
                    "expected_sharpe": metrics.expected_sharpe,
                    "diversification_score": metrics.diversification_score,
                    "regime": regime.primary_regime.value,
                    "method": method,
                    "weights": allocation.signed_weights,
                    "metrics": metrics.to_dict(),
                }
            )
            self._db.save_allocation(
                {
                    "alloc_id": str(uuid.uuid4()),
                    "timestamp": ts,
                    "method": method,
                    "regime": regime.primary_regime.value,
                    "instrument_count": len(active_forecasts),
                    "expected_return": metrics.expected_return,
                    "expected_sharpe": metrics.expected_sharpe,
                    "budget_feasible": allocation.budget_feasible,
                    "weights": allocation.signed_weights,
                }
            )

            logger.info(
                f"PortfolioApi.recommend: {len(active_forecasts)} instruments, Sharpe={metrics.expected_sharpe:.2f}"
            )
            return _ok(result)

        except Exception as exc:
            return _err(f"recommend failed: {exc}", exc)

    # ── Exposure ───────────────────────────────────────────────────────────

    def exposure(
        self,
        forecasts: List[ModelForecast],
        weights: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        try:
            ts = timestamp or _now()
            report = self._exposure_engine.compute(forecasts, weights)
            exp_id = str(uuid.uuid4())
            self._db.save_exposure(
                {
                    "exp_id": exp_id,
                    "timestamp": ts,
                    "gross_exposure": report.gross_exposure,
                    "net_exposure": report.net_exposure,
                    "long_exposure": report.long_exposure,
                    "short_exposure": report.short_exposure,
                    "diversification_ratio": report.diversification_ratio,
                    "alert_count": len(report.alerts),
                    "by_asset_class": report.by_asset_class,
                    "by_currency": report.by_currency,
                    "by_region": report.by_region,
                    "alerts": [a.to_dict() for a in report.alerts],
                }
            )
            return _ok({"exp_id": exp_id, "exposure": report.to_dict()})
        except Exception as exc:
            return _err(f"exposure failed: {exc}", exc)

    # ── Correlation ────────────────────────────────────────────────────────

    def correlation(
        self,
        observations: Dict[str, float],
        method: str = "rolling",
        instruments: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        try:
            ts = timestamp or _now()
            self._corr_engine.add_observations(observations)
            m = (
                CorrelationMethod(method)
                if method in [cm.value for cm in CorrelationMethod]
                else CorrelationMethod.ROLLING
            )
            matrix = self._corr_engine.compute_matrix(instruments, method=m, timestamp=ts)
            matrix_id = str(uuid.uuid4())
            matrix_dict = matrix.to_dict()
            self._db.save_correlation_matrix(
                matrix_id, ts, method, matrix.instruments, matrix_dict["pairs"]
            )
            return _ok(
                {
                    "matrix_id": matrix_id,
                    "matrix": matrix_dict,
                    "high_correlation_pairs": [
                        {"a": a, "b": b, "correlation": round(c, 4)}
                        for a, b, c in matrix.high_correlation_pairs(0.7)
                    ],
                }
            )
        except Exception as exc:
            return _err(f"correlation failed: {exc}", exc)

    # ── Optimize ───────────────────────────────────────────────────────────

    def optimize(
        self,
        forecasts: List[ModelForecast],
        objective: str = "max_sharpe",
        compare_all: bool = False,
        timestamp: Optional[str] = None,
    ) -> dict:
        try:
            ts = timestamp or _now()
            matrix = self._last_correlation_matrix
            obj = (
                OptimizationObjective(objective)
                if objective in [o.value for o in OptimizationObjective]
                else OptimizationObjective.MAX_SHARPE
            )

            if compare_all:
                results = self._optimizer.compare_objectives(forecasts, matrix)
                best_obj = max(results.keys(), key=lambda k: results[k].objective_value)
                best = results[best_obj]
            else:
                best = self._optimizer.optimize(forecasts, obj, matrix)
                results = {objective: best}

            opt_id = str(uuid.uuid4())
            self._db.save_optimization(
                {
                    "opt_id": opt_id,
                    "timestamp": ts,
                    "objective": best.objective,
                    "expected_sharpe": best.optimal_metrics.expected_sharpe,
                    "expected_return": best.optimal_metrics.expected_return,
                    "expected_volatility": best.optimal_metrics.expected_volatility,
                    "diversification_score": best.optimal_metrics.diversification_score,
                    "iterations": best.iterations,
                    "converged": best.converged,
                    "weights": best.optimal_weights,
                    "metrics": best.optimal_metrics.to_dict(),
                }
            )

            return _ok(
                {
                    "opt_id": opt_id,
                    "best": best.to_dict(),
                    "all_objectives": (
                        {k: v.to_dict() for k, v in results.items()} if compare_all else None
                    ),
                    "efficient_frontier": self._optimizer.efficient_frontier(
                        forecasts, n_points=8, correlation_matrix=matrix
                    ),
                }
            )
        except Exception as exc:
            return _err(f"optimize failed: {exc}", exc)

    # ── Scenario ───────────────────────────────────────────────────────────

    def scenario(
        self,
        forecasts: List[ModelForecast],
        weights: Optional[Dict[str, float]] = None,
        scenarios: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        try:
            ts = timestamp or _now()
            matrix = self._last_correlation_matrix
            if weights is None:
                n = max(len(forecasts), 1)
                weights = {f.instrument: 1.0 / n for f in forecasts}

            sc_types = [
                ScenarioType(s) for s in (scenarios or []) if s in [st.value for st in ScenarioType]
            ] or None
            report = self._scenario.run_all(forecasts, weights, sc_types, matrix)

            # Persist each scenario
            for r in report.results:
                self._db.save_scenario(
                    {
                        "scenario_id": str(uuid.uuid4()),
                        "timestamp": ts,
                        "scenario_type": r.scenario_type,
                        "base_sharpe": r.base_sharpe,
                        "stressed_sharpe": r.stressed_sharpe,
                        "return_impact": r.return_impact,
                        "volatility_impact": r.volatility_impact,
                        "resilience_score": r.resilience_score,
                        "instrument_impacts": r.instrument_impacts,
                    }
                )

            return _ok(report.to_dict())
        except Exception as exc:
            return _err(f"scenario failed: {exc}", exc)

    # ── Conflict Check ─────────────────────────────────────────────────────

    def conflict_check(self, forecasts: List[ModelForecast]) -> dict:
        try:
            report = self._conflict_resolver.resolve(forecasts)
            return _ok(report.to_dict())
        except Exception as exc:
            return _err(f"conflict_check failed: {exc}", exc)

    # ── Regime ────────────────────────────────────────────────────────────

    def regime(
        self,
        observations: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None,
    ) -> dict:
        try:
            ts = timestamp or _now()
            if observations:
                for sym, ret in observations.items():
                    self._regime_detector.add_return(sym, ret)
            assessment = self._regime_detector.detect(timestamp=ts)
            return _ok(assessment.to_dict())
        except Exception as exc:
            return _err(f"regime failed: {exc}", exc)

    # ── Risk Budget ────────────────────────────────────────────────────────

    def risk_budget_check(
        self,
        forecasts: List[ModelForecast],
        weights: Dict[str, float],
    ) -> dict:
        try:
            engine = RiskBudgetEngine(self._budget_config)
            result = engine.check(weights, forecasts)
            return _ok(result.to_dict())
        except Exception as exc:
            return _err(f"risk_budget_check failed: {exc}", exc)

    # ── History ────────────────────────────────────────────────────────────

    def allocation_history(self, limit: int = 20) -> dict:
        try:
            return _ok({"history": self._db.allocation_history(limit)})
        except Exception as exc:
            return _err(f"allocation_history failed: {exc}", exc)

    def recommendations_history(self, limit: int = 20) -> dict:
        try:
            return _ok({"history": self._db.recommendations_history(limit)})
        except Exception as exc:
            return _err(f"recommendations_history failed: {exc}", exc)

    def scenario_history(self, scenario_type: Optional[str] = None, limit: int = 20) -> dict:
        try:
            return _ok({"history": self._db.scenario_history(scenario_type, limit)})
        except Exception as exc:
            return _err(f"scenario_history failed: {exc}", exc)

    # ── Dashboard ──────────────────────────────────────────────────────────

    def dashboard_data(
        self,
        forecasts: List[ModelForecast],
        weights: Optional[Dict[str, float]] = None,
        correlation_matrix=None,
    ) -> dict:
        try:
            matrix = correlation_matrix or self._last_correlation_matrix
            metrics = self._scorer.score(forecasts, weights, matrix)
            exposure = self._exposure_engine.compute(forecasts, weights)
            graph = self._dep_graph.latest_snapshot()
            all_charts = self._dashboard.build_all(
                forecasts, weights, metrics, exposure, matrix, graph
            )
            return _ok(all_charts)
        except Exception as exc:
            return _err(f"dashboard_data failed: {exc}", exc)

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        try:
            return _ok(
                {
                    "database": self._db.stats(),
                    "registry": {
                        "total_instruments": self._registry.total(),
                        "asset_class_summary": self._registry.asset_class_summary(),
                    },
                    "correlation_engine": {
                        "tracked_instruments": len(self._corr_engine.instruments()),
                        "window": self._corr_engine._window,
                    },
                }
            )
        except Exception as exc:
            return _err(f"stats failed: {exc}", exc)

    # ── Registry ───────────────────────────────────────────────────────────

    def registry_info(self) -> dict:
        try:
            return _ok(
                {
                    "total": self._registry.total(),
                    "instruments": self._registry.to_dict(),
                    "asset_class_summary": self._registry.asset_class_summary(),
                }
            )
        except Exception as exc:
            return _err(f"registry_info failed: {exc}", exc)
