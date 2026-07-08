"""Phase 24 — ICARI data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.icari.icari_types import (
    AllocationRecommendation,
    AllocationStatus,
    CapitalHealthGrade,
    DrawdownMode,
    OptimizationObjective,
    RiskBudgetType,
    ScenarioType,
    SizingMethod,
)

# ── AllocationRequest ─────────────────────────────────────────────────────────


@dataclass
class AllocationRequest:
    """Input to the capital allocation engine."""

    instrument: str
    direction: str = "LONG"  # LONG | SHORT
    forecast_confidence: float = 0.5  # 0–1
    research_score: float = 0.5  # 0–1
    execution_score: float = 0.5  # 0–1
    expected_value: float = 0.0  # expected $/pip
    expected_return: float = 0.0  # fraction (e.g. 0.02 = 2%)
    expected_drawdown: float = 0.05  # fraction
    portfolio_exposure: float = 0.0  # current exposure fraction
    correlation: float = 0.0  # avg correlation to portfolio
    market_regime: str = "NORMAL"  # NORMAL | TRENDING | RANGING | VOLATILE
    volatility: float = 0.01  # daily vol
    liquidity: float = 1.0  # 0–1 liquidity score
    model_confidence: float = 0.5
    ensemble_confidence: float | None = None  # defaults to model_confidence when unset
    decision_quality: float = 0.5
    atr: float = 0.001  # average true range
    capital: float = 100_000.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.ensemble_confidence is None:
            self.ensemble_confidence = self.model_confidence

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── AllocationResult ──────────────────────────────────────────────────────────


@dataclass
class AllocationResult:
    """Output of the capital allocation engine."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    status: AllocationStatus = AllocationStatus.APPROVED
    allocation_frac: float = 0.0  # fraction of capital to allocate
    allocation_usd: float = 0.0  # dollar allocation
    confidence: float = 0.0  # composite confidence
    rationale: str = ""
    factors: dict[str, float] = field(default_factory=dict)
    recommendations: list[AllocationRecommendation] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "instrument": self.instrument,
            "status": self.status.value,
            "allocation_frac": round(self.allocation_frac, 4),
            "allocation_usd": round(self.allocation_usd, 2),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "recommendations": [r.value for r in self.recommendations],
        }


# ── PositionSize ──────────────────────────────────────────────────────────────


@dataclass
class PositionSize:
    """Result of the position sizing engine."""

    instrument: str
    method: SizingMethod
    units: float = 0.0
    risk_amount: float = 0.0  # $ at risk
    risk_frac: float = 0.0  # fraction of capital at risk
    stop_distance: float = 0.001
    adjusted: bool = False  # True if dynamic adjustments applied
    adjustments: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "method": self.method.value,
            "units": round(self.units, 4),
            "risk_amount": round(self.risk_amount, 2),
            "risk_frac": round(self.risk_frac, 4),
            "stop_distance": round(self.stop_distance, 6),
            "adjusted": self.adjusted,
            "adjustments": {k: round(v, 4) for k, v in self.adjustments.items()},
        }


# ── RiskBudget ────────────────────────────────────────────────────────────────


@dataclass
class RiskBudget:
    """A configurable risk budget."""

    budget_type: RiskBudgetType
    limit: float  # fraction of capital
    used: float = 0.0
    label: str = ""

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.used)

    @property
    def utilisation(self) -> float:
        return self.used / self.limit if self.limit > 0 else 0.0

    @property
    def is_breached(self) -> bool:
        return self.used > self.limit

    def to_dict(self) -> dict:
        return {
            "budget_type": self.budget_type.value,
            "limit": round(self.limit, 4),
            "used": round(self.used, 4),
            "remaining": round(self.remaining, 4),
            "utilisation": round(self.utilisation, 4),
            "breached": self.is_breached,
        }


# ── PortfolioRisk ─────────────────────────────────────────────────────────────


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics."""

    var_95: float = 0.0  # Value at Risk 95%
    var_99: float = 0.0  # Value at Risk 99%
    cvar_95: float = 0.0  # Conditional VaR 95%
    expected_shortfall: float = 0.0
    max_expected_drawdown: float = 0.0
    portfolio_volatility: float = 0.0
    risk_contribution: dict[str, float] = field(default_factory=dict)
    marginal_risk: dict[str, float] = field(default_factory=dict)
    diversification_ratio: float = 1.0
    risk_concentration: float = 0.0  # Herfindahl index

    def to_dict(self) -> dict:
        return {
            "var_95": round(self.var_95, 4),
            "var_99": round(self.var_99, 4),
            "cvar_95": round(self.cvar_95, 4),
            "expected_shortfall": round(self.expected_shortfall, 4),
            "max_expected_drawdown": round(self.max_expected_drawdown, 4),
            "portfolio_volatility": round(self.portfolio_volatility, 4),
            "risk_contribution": {k: round(v, 4) for k, v in self.risk_contribution.items()},
            "marginal_risk": {k: round(v, 4) for k, v in self.marginal_risk.items()},
            "diversification_ratio": round(self.diversification_ratio, 4),
            "risk_concentration": round(self.risk_concentration, 4),
        }


# ── CapitalHealth ─────────────────────────────────────────────────────────────


@dataclass
class CapitalHealth:
    """Composite capital health assessment."""

    efficiency_score: float = 0.0
    risk_score: float = 0.0
    allocation_quality: float = 0.0
    diversification: float = 0.0
    stability: float = 0.0
    utilisation_score: float = 0.0
    overall_score: float = 0.0
    grade: CapitalHealthGrade = CapitalHealthGrade.FAIR
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "efficiency_score": round(self.efficiency_score, 4),
            "risk_score": round(self.risk_score, 4),
            "allocation_quality": round(self.allocation_quality, 4),
            "diversification": round(self.diversification, 4),
            "stability": round(self.stability, 4),
            "utilisation_score": round(self.utilisation_score, 4),
            "overall_score": round(self.overall_score, 4),
            "grade": self.grade.value,
        }


# ── ScenarioResult ────────────────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    """Result of a scenario stress test."""

    scenario: ScenarioType
    portfolio_return: float = 0.0  # stressed return
    portfolio_var: float = 0.0
    max_drawdown: float = 0.0
    resilience_score: float = 0.0  # 0–1
    instrument_impacts: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario.value,
            "portfolio_return": round(self.portfolio_return, 4),
            "portfolio_var": round(self.portfolio_var, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "resilience_score": round(self.resilience_score, 4),
            "instrument_impacts": {k: round(v, 4) for k, v in self.instrument_impacts.items()},
            "recommendations": self.recommendations,
        }


# ── DrawdownEvent ─────────────────────────────────────────────────────────────


@dataclass
class DrawdownEvent:
    """A detected drawdown event."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    peak: float = 0.0
    trough: float = 0.0
    drawdown: float = 0.0  # fraction e.g. 0.05 = 5%
    mode: DrawdownMode = DrawdownMode.NORMAL
    start_ts: float = field(default_factory=time.time)
    end_ts: float | None = None
    recovered: bool = False

    @property
    def duration_seconds(self) -> float | None:
        if self.end_ts:
            return self.end_ts - self.start_ts
        return None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "peak": round(self.peak, 4),
            "trough": round(self.trough, 4),
            "drawdown": round(self.drawdown, 4),
            "mode": self.mode.value,
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3) if self.end_ts else None,
            "recovered": self.recovered,
        }


# ── OptimizationResult ────────────────────────────────────────────────────────


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization run."""

    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE
    weights: dict[str, float] = field(default_factory=dict)  # instrument → weight
    metrics: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    iterations: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "result_id": self.result_id,
            "objective": self.objective.value,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "converged": self.converged,
            "iterations": self.iterations,
        }
