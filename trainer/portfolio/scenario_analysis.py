"""Phase 20 — Scenario Analysis.

Simulates portfolio performance under different market scenarios:
  - Risk-On / Risk-Off
  - High / Low Inflation
  - Interest Rate Changes
  - Commodity Shocks
  - Volatility Spikes

Estimates portfolio resilience across each scenario.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from trainer.portfolio.portfolio_instruments import ModelForecast
from trainer.portfolio.portfolio_scorer import PortfolioMetrics, PortfolioScorer
from trainer.portfolio.correlation_engine import CorrelationMatrix


class ScenarioType(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    HIGH_INFLATION = "high_inflation"
    LOW_INFLATION = "low_inflation"
    RATE_HIKE = "rate_hike"
    RATE_CUT = "rate_cut"
    COMMODITY_SHOCK_UP = "commodity_shock_up"
    COMMODITY_SHOCK_DOWN = "commodity_shock_down"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLATILITY_CRUSH = "volatility_crush"
    USD_RALLY = "usd_rally"
    USD_SELLOFF = "usd_selloff"
    CREDIT_CRISIS = "credit_crisis"
    LIQUIDITY_SQUEEZE = "liquidity_squeeze"
    CUSTOM = "custom"


# Scenario shocks: how each scenario modifies expected returns by instrument tag
SCENARIO_RETURN_SHOCKS: Dict[str, Dict[str, float]] = {
    ScenarioType.RISK_ON.value: {
        "risk_on": +0.15,
        "safe_haven": -0.10,
        "gold": -0.05,
        "commodity_currency": +0.10,
        "tech": +0.12,
        "crypto": +0.20,
    },
    ScenarioType.RISK_OFF.value: {
        "risk_on": -0.20,
        "safe_haven": +0.15,
        "gold": +0.12,
        "commodity_currency": -0.12,
        "tech": -0.18,
        "crypto": -0.25,
    },
    ScenarioType.HIGH_INFLATION.value: {
        "gold": +0.15,
        "commodity_currency": +0.08,
        "crypto": +0.05,
        "index": -0.08,
        "usd": -0.05,
        "bond": -0.12,
    },
    ScenarioType.LOW_INFLATION.value: {
        "gold": -0.08,
        "commodity_currency": -0.05,
        "index": +0.06,
        "usd": +0.03,
        "bond": +0.08,
    },
    ScenarioType.RATE_HIKE.value: {
        "usd": +0.10,
        "jpy": -0.08,
        "gold": -0.10,
        "index": -0.12,
        "tech": -0.15,
        "crypto": -0.15,
        "bond": -0.18,
    },
    ScenarioType.RATE_CUT.value: {
        "usd": -0.10,
        "gold": +0.08,
        "index": +0.10,
        "tech": +0.12,
        "crypto": +0.10,
        "bond": +0.15,
    },
    ScenarioType.COMMODITY_SHOCK_UP.value: {
        "gold": +0.20,
        "silver": +0.18,
        "commodity_currency": +0.15,
        "aud": +0.10,
        "cad": +0.12,
        "index": -0.05,
    },
    ScenarioType.COMMODITY_SHOCK_DOWN.value: {
        "gold": -0.15,
        "silver": -0.18,
        "commodity_currency": -0.12,
        "aud": -0.08,
        "cad": -0.10,
    },
    ScenarioType.VOLATILITY_SPIKE.value: {
        "risk_on": -0.25,
        "safe_haven": +0.20,
        "gold": +0.15,
        "crypto": -0.35,
        "tech": -0.22,
        "usd": +0.08,
    },
    ScenarioType.VOLATILITY_CRUSH.value: {
        "risk_on": +0.08,
        "safe_haven": -0.05,
        "index": +0.07,
        "crypto": +0.10,
        "tech": +0.08,
    },
    ScenarioType.USD_RALLY.value: {
        "usd": +0.15,
        "eur": -0.12,
        "gbp": -0.10,
        "aud": -0.12,
        "nzd": -0.12,
        "gold": -0.12,
        "commodity_currency": -0.10,
    },
    ScenarioType.USD_SELLOFF.value: {
        "usd": -0.15,
        "eur": +0.10,
        "gbp": +0.09,
        "aud": +0.12,
        "nzd": +0.12,
        "gold": +0.12,
        "commodity_currency": +0.10,
    },
    ScenarioType.CREDIT_CRISIS.value: {
        "risk_on": -0.35,
        "crypto": -0.50,
        "safe_haven": +0.25,
        "gold": +0.20,
        "index": -0.30,
        "tech": -0.35,
    },
    ScenarioType.LIQUIDITY_SQUEEZE.value: {
        "risk_on": -0.20,
        "crypto": -0.40,
        "safe_haven": +0.15,
        "gold": +0.10,
        "usd": +0.18,
        "index": -0.18,
    },
}

# Scenario vol multipliers
SCENARIO_VOL_MULTIPLIERS: Dict[str, float] = {
    ScenarioType.RISK_ON.value: 0.9,
    ScenarioType.RISK_OFF.value: 1.5,
    ScenarioType.HIGH_INFLATION.value: 1.2,
    ScenarioType.LOW_INFLATION.value: 0.9,
    ScenarioType.RATE_HIKE.value: 1.3,
    ScenarioType.RATE_CUT.value: 1.1,
    ScenarioType.COMMODITY_SHOCK_UP.value: 1.4,
    ScenarioType.COMMODITY_SHOCK_DOWN.value: 1.3,
    ScenarioType.VOLATILITY_SPIKE.value: 2.5,
    ScenarioType.VOLATILITY_CRUSH.value: 0.6,
    ScenarioType.USD_RALLY.value: 1.2,
    ScenarioType.USD_SELLOFF.value: 1.2,
    ScenarioType.CREDIT_CRISIS.value: 3.0,
    ScenarioType.LIQUIDITY_SQUEEZE.value: 2.5,
}

# Instrument → tag mapping for scenario shocks
INSTRUMENT_TAGS: Dict[str, List[str]] = {
    "EURUSD": ["eur", "major"],
    "GBPUSD": ["gbp", "major"],
    "USDJPY": ["usd", "jpy", "major", "safe_haven"],
    "USDCHF": ["usd", "chf", "major", "safe_haven"],
    "USDCAD": ["usd", "cad", "major", "commodity_currency"],
    "AUDUSD": ["aud", "major", "commodity_currency", "risk_on"],
    "NZDUSD": ["nzd", "major", "commodity_currency", "risk_on"],
    "EURJPY": ["eur", "jpy", "cross"],
    "GBPJPY": ["gbp", "jpy", "cross"],
    "XAUUSD": ["gold", "commodity", "safe_haven"],
    "XAGUSD": ["silver", "commodity"],
    "NAS100": ["index", "tech", "risk_on"],
    "US30": ["index", "risk_on"],
    "SPX500": ["index", "risk_on"],
    "BTCUSD": ["crypto", "risk_on"],
    "ETHUSD": ["crypto", "risk_on"],
}


@dataclass
class ScenarioResult:
    scenario_type: str
    scenario_name: str
    base_return: float
    stressed_return: float
    base_volatility: float
    stressed_volatility: float
    base_sharpe: float
    stressed_sharpe: float
    return_impact: float
    volatility_impact: float
    sharpe_impact: float
    resilience_score: float
    instrument_impacts: Dict[str, float] = field(default_factory=dict)
    description: str = ""

    def is_resilient(self, min_stressed_sharpe: float = 0.0) -> bool:
        return self.stressed_sharpe >= min_stressed_sharpe

    def to_dict(self) -> dict:
        return {
            "scenario_type": self.scenario_type,
            "scenario_name": self.scenario_name,
            "base_return": round(self.base_return, 4),
            "stressed_return": round(self.stressed_return, 4),
            "base_volatility": round(self.base_volatility, 4),
            "stressed_volatility": round(self.stressed_volatility, 4),
            "base_sharpe": round(self.base_sharpe, 4),
            "stressed_sharpe": round(self.stressed_sharpe, 4),
            "return_impact": round(self.return_impact, 4),
            "volatility_impact": round(self.volatility_impact, 4),
            "sharpe_impact": round(self.sharpe_impact, 4),
            "resilience_score": round(self.resilience_score, 4),
            "description": self.description,
            "instrument_impacts": {k: round(v, 4) for k, v in self.instrument_impacts.items()},
        }


@dataclass
class ScenarioReport:
    base_metrics: PortfolioMetrics
    results: List[ScenarioResult]
    worst_scenario: Optional[ScenarioResult] = None
    best_scenario: Optional[ScenarioResult] = None
    average_resilience: float = 0.0

    def __post_init__(self) -> None:
        if self.results:
            self.worst_scenario = min(self.results, key=lambda r: r.stressed_sharpe)
            self.best_scenario = max(self.results, key=lambda r: r.stressed_sharpe)
            self.average_resilience = sum(r.resilience_score for r in self.results) / len(
                self.results
            )

    def to_dict(self) -> dict:
        return {
            "base_metrics": self.base_metrics.to_dict(),
            "scenarios": [r.to_dict() for r in self.results],
            "worst_scenario": self.worst_scenario.to_dict() if self.worst_scenario else None,
            "best_scenario": self.best_scenario.to_dict() if self.best_scenario else None,
            "average_resilience": round(self.average_resilience, 4),
        }


class ScenarioAnalyser:
    """Simulates portfolio behaviour under macro scenarios."""

    RISK_FREE_RATE = 0.045

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self._rfr = risk_free_rate
        self._scorer = PortfolioScorer(risk_free_rate=risk_free_rate)

    def run_scenario(
        self,
        forecasts: List[ModelForecast],
        weights: Dict[str, float],
        scenario: ScenarioType,
        correlation_matrix: Optional[CorrelationMatrix] = None,
        description: str = "",
    ) -> ScenarioResult:
        base_metrics = self._scorer.score(forecasts, weights, correlation_matrix)
        stressed_forecasts = self._apply_scenario(forecasts, scenario)
        vol_mult = SCENARIO_VOL_MULTIPLIERS.get(scenario.value, 1.0)
        stressed_fc_with_vol = self._scale_volatility(stressed_forecasts, vol_mult)
        stressed_metrics = self._scorer.score(stressed_fc_with_vol, weights, correlation_matrix)

        impact_by_instrument = self._per_instrument_impact(forecasts, stressed_forecasts, weights)
        return_impact = stressed_metrics.expected_return - base_metrics.expected_return
        vol_impact = stressed_metrics.expected_volatility - base_metrics.expected_volatility
        sharpe_impact = stressed_metrics.expected_sharpe - base_metrics.expected_sharpe
        resilience = self._resilience_score(base_metrics, stressed_metrics)

        return ScenarioResult(
            scenario_type=scenario.value,
            scenario_name=scenario.value.replace("_", " ").title(),
            base_return=base_metrics.expected_return,
            stressed_return=stressed_metrics.expected_return,
            base_volatility=base_metrics.expected_volatility,
            stressed_volatility=stressed_metrics.expected_volatility,
            base_sharpe=base_metrics.expected_sharpe,
            stressed_sharpe=stressed_metrics.expected_sharpe,
            return_impact=return_impact,
            volatility_impact=vol_impact,
            sharpe_impact=sharpe_impact,
            resilience_score=resilience,
            instrument_impacts=impact_by_instrument,
            description=description or scenario.value.replace("_", " ").title(),
        )

    def run_all(
        self,
        forecasts: List[ModelForecast],
        weights: Dict[str, float],
        scenarios: Optional[List[ScenarioType]] = None,
        correlation_matrix: Optional[CorrelationMatrix] = None,
    ) -> ScenarioReport:
        base_metrics = self._scorer.score(forecasts, weights, correlation_matrix)
        scenarios = scenarios or [
            ScenarioType.RISK_ON,
            ScenarioType.RISK_OFF,
            ScenarioType.HIGH_INFLATION,
            ScenarioType.LOW_INFLATION,
            ScenarioType.RATE_HIKE,
            ScenarioType.RATE_CUT,
            ScenarioType.VOLATILITY_SPIKE,
            ScenarioType.USD_RALLY,
            ScenarioType.COMMODITY_SHOCK_UP,
            ScenarioType.CREDIT_CRISIS,
        ]
        results = [
            self.run_scenario(forecasts, weights, sc, correlation_matrix) for sc in scenarios
        ]
        return ScenarioReport(base_metrics=base_metrics, results=results)

    def _apply_scenario(
        self,
        forecasts: List[ModelForecast],
        scenario: ScenarioType,
    ) -> List[ModelForecast]:
        shocks = SCENARIO_RETURN_SHOCKS.get(scenario.value, {})
        if not shocks:
            return forecasts

        stressed = []
        for fc in forecasts:
            tags = INSTRUMENT_TAGS.get(fc.instrument, [])
            shock = 0.0
            for tag in tags:
                if tag in shocks:
                    shock += shocks[tag]

            # Apply shock as modification to expected_return
            new_return = fc.expected_return + shock
            stressed.append(
                ModelForecast(
                    instrument=fc.instrument,
                    direction=fc.direction,
                    confidence=fc.confidence,
                    expected_return=max(0.0, new_return),
                    expected_risk=fc.expected_risk,
                    timestamp=fc.timestamp,
                    metadata=dict(fc.metadata),
                )
            )
        return stressed

    @staticmethod
    def _scale_volatility(forecasts: List[ModelForecast], mult: float) -> List[ModelForecast]:
        return [
            ModelForecast(
                instrument=fc.instrument,
                direction=fc.direction,
                confidence=fc.confidence,
                expected_return=fc.expected_return,
                expected_risk=fc.expected_risk * mult,
                timestamp=fc.timestamp,
                metadata=dict(fc.metadata),
            )
            for fc in forecasts
        ]

    @staticmethod
    def _per_instrument_impact(
        base: List[ModelForecast],
        stressed: List[ModelForecast],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        base_map = {f.instrument: f for f in base}
        stressed_map = {f.instrument: f for f in stressed}
        impacts = {}
        for sym in base_map:
            b = base_map[sym]
            s = stressed_map.get(sym, b)
            w = weights.get(sym, 0.0)
            impact = w * (s.expected_return - b.expected_return)
            impacts[sym] = round(impact, 4)
        return impacts

    @staticmethod
    def _resilience_score(base: PortfolioMetrics, stressed: PortfolioMetrics) -> float:
        if base.expected_sharpe <= 0:
            return 0.5
        sharpe_retention = stressed.expected_sharpe / max(base.expected_sharpe, 1e-9)
        return round(min(1.0, max(0.0, (sharpe_retention + 1.0) / 2.0)), 4)
