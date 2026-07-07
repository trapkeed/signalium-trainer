"""Phase 24 — Institutional Capital Allocation & Risk Intelligence (ICARI)."""

from trainer.icari.allocation_engine import AllocationEngine
from trainer.icari.allocation_optimizer import AllocationOptimizer
from trainer.icari.capital_efficiency import CapitalEfficiencyEngine
from trainer.icari.capital_health import CapitalHealthEngine
from trainer.icari.correlation_engine import CorrelationEngine
from trainer.icari.drawdown_manager import DrawdownManager
from trainer.icari.dynamic_sizer import DynamicSizer
from trainer.icari.icari_api import IcariAPI
from trainer.icari.icari_dashboard import IcariDashboard
from trainer.icari.icari_database import IcariDatabase
from trainer.icari.icari_models import (
    AllocationRequest,
    AllocationResult,
    CapitalHealth,
    DrawdownEvent,
    OptimizationResult,
    PortfolioRisk,
    PositionSize,
    RiskBudget,
    ScenarioResult,
)
from trainer.icari.icari_types import (
    DEFAULT_RISK_BUDGETS,
    SCENARIO_FACTORS,
    AllocationRecommendation,
    AllocationStatus,
    CapitalHealthGrade,
    DrawdownMode,
    OptimizationObjective,
    RiskBudgetType,
    ScenarioType,
    SizingMethod,
    health_grade,
)
from trainer.icari.portfolio_risk_engine import PortfolioRiskEngine
from trainer.icari.position_sizing_engine import PositionSizingEngine
from trainer.icari.risk_budget_engine import RiskBudgetEngine
from trainer.icari.scenario_engine import ScenarioEngine

__all__ = [
    # Types
    "SizingMethod",
    "RiskBudgetType",
    "DrawdownMode",
    "ScenarioType",
    "OptimizationObjective",
    "AllocationRecommendation",
    "AllocationStatus",
    "CapitalHealthGrade",
    "health_grade",
    "SCENARIO_FACTORS",
    "DEFAULT_RISK_BUDGETS",
    # Models
    "AllocationRequest",
    "AllocationResult",
    "PositionSize",
    "RiskBudget",
    "PortfolioRisk",
    "CapitalHealth",
    "ScenarioResult",
    "DrawdownEvent",
    "OptimizationResult",
    # Engines
    "AllocationEngine",
    "RiskBudgetEngine",
    "PositionSizingEngine",
    "DynamicSizer",
    "CapitalEfficiencyEngine",
    "PortfolioRiskEngine",
    "DrawdownManager",
    "CorrelationEngine",
    "ScenarioEngine",
    "AllocationOptimizer",
    "CapitalHealthEngine",
    # Infrastructure
    "IcariDatabase",
    "IcariDashboard",
    # Unified API
    "IcariAPI",
]
