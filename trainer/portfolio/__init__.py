"""Signalium Phase 20 — Portfolio Intelligence Engine.

Transforms Signalium from a single-instrument forecasting system into a
portfolio-aware quantitative intelligence platform.

Evaluates all model forecasts collectively before producing final
portfolio recommendations.

Usage::

    from trainer.portfolio import (
        InstrumentRegistry, ModelForecast,
        PortfolioApi,
    )

    api = PortfolioApi(db_path="data/portfolio.db")
    forecasts = [
        ModelForecast("EURUSD", "BUY",  confidence=0.72, expected_return=0.08, expected_risk=0.10),
        ModelForecast("GBPUSD", "BUY",  confidence=0.65, expected_return=0.07, expected_risk=0.12),
        ModelForecast("XAUUSD", "BUY",  confidence=0.80, expected_return=0.12, expected_risk=0.15),
        ModelForecast("NAS100", "SELL", confidence=0.60, expected_return=0.18, expected_risk=0.22),
    ]
    resp = api.recommend(forecasts)
    print(resp["data"]["portfolio_metrics"])
"""

# ── Instruments ────────────────────────────────────────────────────────────────
from trainer.portfolio.portfolio_instruments import (
    AssetClass,
    Region,
    InstrumentInfo,
    ModelForecast,
    InstrumentRegistry,
)

# ── Correlation ────────────────────────────────────────────────────────────────
from trainer.portfolio.correlation_engine import (
    CorrelationMethod,
    CorrelationResult,
    CorrelationMatrix,
    CorrelationEngine,
)

# ── Exposure ───────────────────────────────────────────────────────────────────
from trainer.portfolio.exposure_engine import (
    ExposureAlert,
    ExposureAlertItem,
    ExposureReport,
    ExposureEngine,
)

# ── Dependency Graph ───────────────────────────────────────────────────────────
from trainer.portfolio.dependency_graph import (
    ClusterType,
    GraphNode,
    GraphEdge,
    Cluster,
    DependencyGraphSnapshot,
    DependencyGraphBuilder,
)

# ── Regime ─────────────────────────────────────────────────────────────────────
from trainer.portfolio.regime_detector import (
    MarketRegime,
    RegimeSignal,
    RegimeAssessment,
    RegimeDetector,
)

# ── Risk Budget ────────────────────────────────────────────────────────────────
from trainer.portfolio.risk_budget import (
    BudgetStatus,
    BudgetItem,
    RiskBudgetConfig,
    BudgetCheckResult,
    RiskBudgetEngine,
)

# ── Conflict Resolver ──────────────────────────────────────────────────────────
from trainer.portfolio.conflict_resolver import (
    ConflictType,
    Resolution,
    ConflictDetail,
    ReinforcementDetail,
    ConflictReport,
    ConflictResolver,
)

# ── Portfolio Scorer ───────────────────────────────────────────────────────────
from trainer.portfolio.portfolio_scorer import (
    PortfolioMetrics,
    PortfolioScorer,
)

# ── Position Allocator ─────────────────────────────────────────────────────────
from trainer.portfolio.position_allocator import (
    AllocationMethod,
    AllocationResult,
    PositionAllocator,
)

# ── Scenario Analysis ──────────────────────────────────────────────────────────
from trainer.portfolio.scenario_analysis import (
    ScenarioType,
    ScenarioResult,
    ScenarioReport,
    ScenarioAnalyser,
)

# ── Optimizer ──────────────────────────────────────────────────────────────────
from trainer.portfolio.optimizer import (
    OptimizationObjective,
    OptimizationResult,
    PortfolioOptimizer,
)

# ── Database ───────────────────────────────────────────────────────────────────
from trainer.portfolio.portfolio_database import PortfolioDatabase

# ── API ────────────────────────────────────────────────────────────────────────
from trainer.portfolio.portfolio_api import PortfolioApi

# ── Dashboard ──────────────────────────────────────────────────────────────────
from trainer.portfolio.portfolio_dashboard import PortfolioDashboard

__all__ = [
    # Instruments
    "AssetClass",
    "Region",
    "InstrumentInfo",
    "ModelForecast",
    "InstrumentRegistry",
    # Correlation
    "CorrelationMethod",
    "CorrelationResult",
    "CorrelationMatrix",
    "CorrelationEngine",
    # Exposure
    "ExposureAlert",
    "ExposureAlertItem",
    "ExposureReport",
    "ExposureEngine",
    # Dependency Graph
    "ClusterType",
    "GraphNode",
    "GraphEdge",
    "Cluster",
    "DependencyGraphSnapshot",
    "DependencyGraphBuilder",
    # Regime
    "MarketRegime",
    "RegimeSignal",
    "RegimeAssessment",
    "RegimeDetector",
    # Risk Budget
    "BudgetStatus",
    "BudgetItem",
    "RiskBudgetConfig",
    "BudgetCheckResult",
    "RiskBudgetEngine",
    # Conflict
    "ConflictType",
    "Resolution",
    "ConflictDetail",
    "ReinforcementDetail",
    "ConflictReport",
    "ConflictResolver",
    # Scoring
    "PortfolioMetrics",
    "PortfolioScorer",
    # Allocation
    "AllocationMethod",
    "AllocationResult",
    "PositionAllocator",
    # Scenario
    "ScenarioType",
    "ScenarioResult",
    "ScenarioReport",
    "ScenarioAnalyser",
    # Optimization
    "OptimizationObjective",
    "OptimizationResult",
    "PortfolioOptimizer",
    # Infrastructure
    "PortfolioDatabase",
    "PortfolioApi",
    "PortfolioDashboard",
]
