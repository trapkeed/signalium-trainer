"""Signalium Phase 21 — Institutional Execution Intelligence Platform.

Converts high-quality forecasts into institutional-grade execution plans.
Produces recommendations only — does NOT place trades.

Usage::

    from trainer.execution import (
        ExecutionForecast,
        Session,
        ExecutionApi,
    )

    api = ExecutionApi(db_path="data/execution.db")
    resp = api.plan({
        "instrument":      "EURUSD",
        "direction":       "BUY",
        "confidence":      0.75,
        "expected_return": 0.08,
        "expected_risk":   0.10,
        "current_price":   1.0850,
        "atr":             0.0012,
        "spread":          0.0001,
        "session":         "LONDON",
        "volatility":      0.009,
        "momentum":        0.3,
        "news_risk":       0.1,
        "account_size":    10000.0,
    })
    print(resp["data"]["executive_summary"])
"""

# ── Entry Quality ──────────────────────────────────────────────────────────────
from trainer.execution.entry_quality import EntryQualityEngine, EntryQualityResult

# ── Entry Zones ────────────────────────────────────────────────────────────────
from trainer.execution.entry_zones import EntryZoneEngine, EntryZones

# ── API ────────────────────────────────────────────────────────────────────────
from trainer.execution.execution_api import ExecutionApi

# ── Dashboard ──────────────────────────────────────────────────────────────────
from trainer.execution.execution_dashboard import ExecutionDashboard

# ── Database ───────────────────────────────────────────────────────────────────
from trainer.execution.execution_database import ExecutionDatabase

# ── Execution Planner ──────────────────────────────────────────────────────────
from trainer.execution.execution_planner import ExecutionPlan, ExecutionPlanner

# ── Execution Simulator ────────────────────────────────────────────────────────
from trainer.execution.execution_simulator import (
    ExecutionSimulator,
    ScenarioOutcome,
    SimulationResult,
)

# ── Execution Timing ───────────────────────────────────────────────────────────
from trainer.execution.execution_timing import ExecutionTiming, TimingAssessment, TimingFactor

# ── Core Types ─────────────────────────────────────────────────────────────────
from trainer.execution.execution_types import (
    SESSION_QUALITY,
    BrokerConstraints,
    ExecutionForecast,
    ExecutionGrade,
    ManagementAction,
    OrderType,
    Session,
    SizingMethod,
    TimingRecommendation,
    default_broker,
)

# ── Order Strategy ─────────────────────────────────────────────────────────────
from trainer.execution.order_strategy import OrderRecommendation, OrderStrategyEngine

# ── Position Sizing ────────────────────────────────────────────────────────────
from trainer.execution.position_sizer import PositionSizer, RiskLimits, SizingResult

# ── Risk Engine ────────────────────────────────────────────────────────────────
from trainer.execution.risk_engine import RiskEngine, RiskProfile

# ── Stop Loss ──────────────────────────────────────────────────────────────────
from trainer.execution.stop_loss_engine import StopLossEngine, StopLossRecommendation

# ── Take Profit ────────────────────────────────────────────────────────────────
from trainer.execution.take_profit_engine import TakeProfitEngine, TakeProfitRecommendation

# ── Trade Management ───────────────────────────────────────────────────────────
from trainer.execution.trade_manager import ManagementPlan, ManagementRule, TradeManager

# ── Trade Scorer ───────────────────────────────────────────────────────────────
from trainer.execution.trade_scorer import TradeScore, TradeScorer

__all__ = [
    # Types
    "Session",
    "ExecutionGrade",
    "OrderType",
    "TimingRecommendation",
    "SizingMethod",
    "ManagementAction",
    "ExecutionForecast",
    "BrokerConstraints",
    "SESSION_QUALITY",
    "default_broker",
    # Zones
    "EntryZones",
    "EntryZoneEngine",
    # Quality
    "EntryQualityResult",
    "EntryQualityEngine",
    # Stop Loss
    "StopLossRecommendation",
    "StopLossEngine",
    # Take Profit
    "TakeProfitRecommendation",
    "TakeProfitEngine",
    # Sizing
    "RiskLimits",
    "SizingResult",
    "PositionSizer",
    # Management
    "ManagementRule",
    "ManagementPlan",
    "TradeManager",
    # Timing
    "TimingFactor",
    "TimingAssessment",
    "ExecutionTiming",
    # Risk
    "RiskProfile",
    "RiskEngine",
    # Order
    "OrderRecommendation",
    "OrderStrategyEngine",
    # Scorer
    "TradeScore",
    "TradeScorer",
    # Simulator
    "ScenarioOutcome",
    "SimulationResult",
    "ExecutionSimulator",
    # Planner
    "ExecutionPlan",
    "ExecutionPlanner",
    # Database
    "ExecutionDatabase",
    # Dashboard
    "ExecutionDashboard",
    # API
    "ExecutionApi",
]
