# Scenario Analysis Guide — Phase 20

## Overview

The `ScenarioAnalyser` stress-tests a portfolio under 14 predefined macro
scenarios. It applies return shocks and volatility multipliers to all
instruments to estimate portfolio resilience.

---

## Available Scenarios

| Scenario | Enum | Primary Effect |
|----------|------|----------------|
| Risk-On | `RISK_ON` | Risk assets rally, safe havens sell |
| Risk-Off | `RISK_OFF` | Safe havens rally, risk assets sell |
| High Inflation | `HIGH_INFLATION` | Gold up, USD down, equities down |
| Low Inflation | `LOW_INFLATION` | USD up, bonds up, gold down |
| Rate Hike | `RATE_HIKE` | USD up, equities down, gold down |
| Rate Cut | `RATE_CUT` | USD down, equities up, gold up |
| Commodity Shock Up | `COMMODITY_SHOCK_UP` | Gold/silver/AUD/CAD up |
| Commodity Shock Down | `COMMODITY_SHOCK_DOWN` | Commodities sell off |
| Volatility Spike | `VOLATILITY_SPIKE` | 2.5× vol, risk assets crash |
| Volatility Crush | `VOLATILITY_CRUSH` | 0.6× vol, risk assets rally |
| USD Rally | `USD_RALLY` | USD up, EUR/GBP/AUD/gold down |
| USD Selloff | `USD_SELLOFF` | USD down, all non-USD up |
| Credit Crisis | `CREDIT_CRISIS` | Severe risk-off, crypto -50% |
| Liquidity Squeeze | `LIQUIDITY_SQUEEZE` | USD up, risk assets down, crypto -40% |

---

## Usage

```python
from trainer.portfolio import ScenarioAnalyser, ScenarioType

analyser = ScenarioAnalyser(risk_free_rate=0.045)

# Run one scenario
weights = {f.instrument: 1/len(forecasts) for f in forecasts}
result = analyser.run_scenario(forecasts, weights, ScenarioType.CREDIT_CRISIS)
print(f"Base Sharpe: {result.base_sharpe:.2f}")
print(f"Stressed Sharpe: {result.stressed_sharpe:.2f}")
print(f"Resilience: {result.resilience_score:.2f}")

# Run all scenarios
report = analyser.run_all(forecasts, weights)
print(f"Worst: {report.worst_scenario.scenario_name}")
print(f"Avg resilience: {report.average_resilience:.2f}")
```

---

## Resilience Score

The resilience score measures how well the portfolio's Sharpe ratio
holds up under stress:

```
resilience = (stressed_sharpe / base_sharpe + 1) / 2
```

Range: 0.0 (total collapse) → 1.0 (perfectly maintained).

---

## Via API

```python
from trainer.portfolio import PortfolioApi

api = PortfolioApi()
resp = api.scenario(
    forecasts,
    scenarios=["risk_off", "credit_crisis", "volatility_spike"]
)
report = resp["data"]
print(report["worst_scenario"])
```
