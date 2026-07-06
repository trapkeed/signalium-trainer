"""Phase 24 — ICARI types and enumerations."""

from __future__ import annotations

from enum import Enum


class SizingMethod(str, Enum):
    FIXED_FRACTIONAL = "FIXED_FRACTIONAL"
    FIXED_RISK = "FIXED_RISK"
    ATR_ADJUSTED = "ATR_ADJUSTED"
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"
    EXPECTED_VALUE = "EXPECTED_VALUE"
    CONFIDENCE_ADJUSTED = "CONFIDENCE_ADJUSTED"
    RISK_BUDGET_ADJUSTED = "RISK_BUDGET_ADJUSTED"
    KELLY = "KELLY"
    FRACTIONAL_KELLY = "FRACTIONAL_KELLY"
    CUSTOM = "CUSTOM"


class RiskBudgetType(str, Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    PORTFOLIO = "PORTFOLIO"
    INSTRUMENT = "INSTRUMENT"
    CURRENCY = "CURRENCY"
    COMMODITY = "COMMODITY"
    CRYPTO = "CRYPTO"
    INDEX = "INDEX"
    STRATEGY = "STRATEGY"
    CORRELATED = "CORRELATED"
    OPEN_TRADE = "OPEN_TRADE"


class DrawdownMode(str, Enum):
    NORMAL = "NORMAL"
    SOFT_LIMIT = "SOFT_LIMIT"
    HARD_LIMIT = "HARD_LIMIT"
    PRESERVATION = "PRESERVATION"
    RECOVERY = "RECOVERY"
    EMERGENCY_LOCKDOWN = "EMERGENCY_LOCKDOWN"


class ScenarioType(str, Enum):
    FLASH_CRASH = "FLASH_CRASH"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"
    SPREAD_EXPANSION = "SPREAD_EXPANSION"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    CENTRAL_BANK = "CENTRAL_BANK"
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    STRONG_USD = "STRONG_USD"
    WEAK_USD = "WEAK_USD"
    COMMODITY_SHOCK = "COMMODITY_SHOCK"
    CRYPTO_CRASH = "CRYPTO_CRASH"


class OptimizationObjective(str, Enum):
    MAX_SHARPE = "MAX_SHARPE"
    MAX_SORTINO = "MAX_SORTINO"
    MIN_DRAWDOWN = "MIN_DRAWDOWN"
    MAX_DIVERSIFICATION = "MAX_DIVERSIFICATION"
    RISK_PARITY = "RISK_PARITY"
    BALANCED = "BALANCED"
    CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"
    EXPECTED_UTILITY = "EXPECTED_UTILITY"


class AllocationRecommendation(str, Enum):
    INCREASE_POSITION = "INCREASE_POSITION"
    REDUCE_POSITION = "REDUCE_POSITION"
    DELAY_ENTRY = "DELAY_ENTRY"
    SKIP_TRADE = "SKIP_TRADE"
    REDUCE_PORTFOLIO_RISK = "REDUCE_PORTFOLIO_RISK"
    INCREASE_DIVERSIFICATION = "INCREASE_DIVERSIFICATION"
    REDUCE_CORRELATED = "REDUCE_CORRELATED"
    ACTIVATE_PRESERVATION = "ACTIVATE_PRESERVATION"


class AllocationStatus(str, Enum):
    APPROVED = "APPROVED"
    REDUCED = "REDUCED"
    DENIED = "DENIED"
    PENDING = "PENDING"


class CapitalHealthGrade(str, Enum):
    EXCELLENT = "EXCELLENT"  # ≥ 0.85
    GOOD = "GOOD"  # ≥ 0.70
    FAIR = "FAIR"  # ≥ 0.55
    POOR = "POOR"  # ≥ 0.40
    CRITICAL = "CRITICAL"  # < 0.40


def health_grade(score: float) -> CapitalHealthGrade:
    if score >= 0.85:
        return CapitalHealthGrade.EXCELLENT
    if score >= 0.70:
        return CapitalHealthGrade.GOOD
    if score >= 0.55:
        return CapitalHealthGrade.FAIR
    if score >= 0.40:
        return CapitalHealthGrade.POOR
    return CapitalHealthGrade.CRITICAL


# Scenario stress factors — (return_multiplier, volatility_multiplier, liquidity_multiplier)
SCENARIO_FACTORS: dict = {
    ScenarioType.FLASH_CRASH: (-0.40, 3.0, 0.20),
    ScenarioType.LIQUIDITY_CRISIS: (-0.20, 2.0, 0.10),
    ScenarioType.SPREAD_EXPANSION: (-0.05, 1.5, 0.50),
    ScenarioType.HIGH_VOLATILITY: (-0.10, 2.5, 0.60),
    ScenarioType.CENTRAL_BANK: (0.02, 1.8, 0.80),
    ScenarioType.RISK_ON: (0.05, 0.7, 1.20),
    ScenarioType.RISK_OFF: (-0.08, 1.6, 0.70),
    ScenarioType.STRONG_USD: (-0.03, 1.2, 0.90),
    ScenarioType.WEAK_USD: (0.03, 1.2, 0.90),
    ScenarioType.COMMODITY_SHOCK: (-0.15, 2.0, 0.70),
    ScenarioType.CRYPTO_CRASH: (-0.50, 4.0, 0.30),
}

# Sizing method validation
VALID_SIZING_METHODS = list(SizingMethod)

# Risk budget defaults (fraction of total capital)
DEFAULT_RISK_BUDGETS: dict = {
    RiskBudgetType.DAILY: 0.02,
    RiskBudgetType.WEEKLY: 0.05,
    RiskBudgetType.MONTHLY: 0.10,
    RiskBudgetType.PORTFOLIO: 0.20,
    RiskBudgetType.INSTRUMENT: 0.05,
    RiskBudgetType.CURRENCY: 0.15,
    RiskBudgetType.COMMODITY: 0.10,
    RiskBudgetType.CRYPTO: 0.05,
    RiskBudgetType.INDEX: 0.15,
    RiskBudgetType.STRATEGY: 0.10,
    RiskBudgetType.CORRELATED: 0.20,
    RiskBudgetType.OPEN_TRADE: 0.30,
}
