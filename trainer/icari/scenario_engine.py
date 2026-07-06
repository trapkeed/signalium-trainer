"""Phase 24 — Scenario Engine.

Simulates 11 scenarios:
Flash Crash, Liquidity Crisis, Spread Expansion, High Volatility,
Central Bank, Risk-On, Risk-Off, Strong USD, Weak USD,
Commodity Shock, Crypto Crash.

Estimates portfolio resilience under each scenario.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_types import ScenarioType, SCENARIO_FACTORS
from trainer.icari.icari_models import ScenarioResult

# Currency sensitivity: how each scenario affects different currency pairs
_CCY_SENSITIVITY = {
    ScenarioType.STRONG_USD: {
        "USD": -0.5,
        "EUR": +0.3,
        "GBP": +0.2,
        "JPY": +0.4,
        "AUD": +0.3,
        "NZD": +0.3,
    },
    ScenarioType.WEAK_USD: {
        "USD": +0.5,
        "EUR": -0.3,
        "GBP": -0.2,
        "JPY": -0.4,
        "AUD": -0.3,
        "NZD": -0.3,
    },
    ScenarioType.RISK_OFF: {"JPY": -0.5, "CHF": -0.4, "USD": -0.3, "AUD": +0.5, "NZD": +0.4},
    ScenarioType.RISK_ON: {"AUD": -0.4, "NZD": -0.3, "JPY": +0.4, "CHF": +0.3},
    ScenarioType.CRYPTO_CRASH: {"BTC": -0.8, "ETH": -0.7},
    ScenarioType.COMMODITY_SHOCK: {"OIL": -0.4, "GOLD": -0.2, "SILVER": -0.25},
}


class ScenarioEngine:
    """Stress test the portfolio under 11 macro scenarios."""

    def __init__(self, capital: float = 100_000.0):
        self._capital = capital

    def run(
        self,
        scenario: ScenarioType,
        weights: Dict[str, float],  # instrument → portfolio weight
        volatilities: Dict[str, float],  # instrument → daily vol
        currencies: Optional[Dict[str, str]] = None,  # instrument → base CCY
    ) -> ScenarioResult:
        """Run a single scenario simulation."""
        factors = SCENARIO_FACTORS.get(scenario, (0.0, 1.0, 1.0))
        ret_factor, vol_factor, liq_factor = factors

        # Instrument-level impacts
        instrument_impacts: Dict[str, float] = {}
        for ins, w in weights.items():
            base_vol = volatilities.get(ins, 0.01)
            stressed_vol = base_vol * vol_factor

            # Start with scenario return factor
            impact = ret_factor * w

            # Apply currency sensitivity if available
            if currencies and ins in currencies:
                ccy = currencies[ins]
                ccy_sens = _CCY_SENSITIVITY.get(scenario, {}).get(ccy, 0.0)
                impact += ccy_sens * w * 0.5

            # Liquidity adjustment: illiquid assets hit harder
            if liq_factor < 0.50:
                impact *= 1.0 + (1.0 - liq_factor) * 0.5

            instrument_impacts[ins] = round(impact, 4)

        # Portfolio-level stressed return
        portfolio_return = sum(instrument_impacts.values())

        # Stressed VaR: scale base VaR by vol_factor
        avg_weight = 1.0 / max(len(weights), 1)
        avg_vol = sum(volatilities.get(ins, 0.01) for ins in weights) / max(len(weights), 1)
        portfolio_var = avg_vol * vol_factor * 1.645 * sum(abs(w) for w in weights.values())

        # Max drawdown estimate under scenario
        max_dd = abs(portfolio_return) * 1.5

        # Resilience score (higher = more resilient)
        # 0 = total wipeout, 1 = no impact
        resilience = max(0.0, min(1.0, 1.0 + portfolio_return))

        # Recommendations
        recs = self._recommendations(scenario, portfolio_return, resilience, liq_factor)

        result = ScenarioResult(
            scenario=scenario,
            portfolio_return=round(portfolio_return, 4),
            portfolio_var=round(portfolio_var, 4),
            max_drawdown=round(max_dd, 4),
            resilience_score=round(resilience, 4),
            instrument_impacts=instrument_impacts,
            recommendations=recs,
        )

        logger.debug(
            f"ScenarioEngine: {scenario.value} return={portfolio_return:.4f} resilience={resilience:.4f}"
        )
        return result

    def run_all(
        self,
        weights: Dict[str, float],
        volatilities: Dict[str, float],
        currencies: Optional[Dict[str, str]] = None,
    ) -> List[ScenarioResult]:
        """Run all 11 scenarios and return results sorted by resilience (ascending)."""
        results = [self.run(s, weights, volatilities, currencies) for s in ScenarioType]
        results.sort(key=lambda r: r.resilience_score)
        return results

    def worst_case(
        self,
        weights: Dict[str, float],
        volatilities: Dict[str, float],
        currencies: Optional[Dict[str, str]] = None,
    ) -> ScenarioResult:
        """Return the scenario with the lowest resilience."""
        results = self.run_all(weights, volatilities, currencies)
        return results[0] if results else ScenarioResult(scenario=ScenarioType.FLASH_CRASH)

    def resilience_summary(self, results: List[ScenarioResult]) -> Dict:
        if not results:
            return {}
        avg = sum(r.resilience_score for r in results) / len(results)
        worst = min(results, key=lambda r: r.resilience_score)
        best = max(results, key=lambda r: r.resilience_score)
        return {
            "average_resilience": round(avg, 4),
            "worst_scenario": worst.scenario.value,
            "worst_resilience": round(worst.resilience_score, 4),
            "best_scenario": best.scenario.value,
            "best_resilience": round(best.resilience_score, 4),
            "total_scenarios": len(results),
        }

    @staticmethod
    def _recommendations(
        scenario: ScenarioType,
        port_return: float,
        resilience: float,
        liq_factor: float,
    ) -> List[str]:
        recs = []
        if resilience < 0.70:
            recs.append("Reduce overall portfolio risk before event")
        if liq_factor < 0.50:
            recs.append("Increase cash buffer for liquidity crisis")
        if port_return < -0.10:
            recs.append("Activate capital preservation mode")
        if scenario in (ScenarioType.FLASH_CRASH, ScenarioType.LIQUIDITY_CRISIS):
            recs.append("Pre-position stop losses before potential event")
        if scenario in (ScenarioType.CRYPTO_CRASH, ScenarioType.COMMODITY_SHOCK):
            recs.append("Review cross-asset correlation exposure")
        if not recs:
            recs.append("Portfolio appears resilient to this scenario")
        return recs
