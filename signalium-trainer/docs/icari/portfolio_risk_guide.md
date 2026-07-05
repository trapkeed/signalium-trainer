# Portfolio Risk Guide — Phase 24

## Overview

ICARI computes portfolio risk using historical simulation (no parametric assumptions beyond normal approximation for the parametric VaR helper).

## Metrics

| Metric | Description |
|---|---|
| VaR 95% | 5th percentile of portfolio daily returns (negated) |
| VaR 99% | 1st percentile of portfolio daily returns (negated) |
| CVaR 95% | Mean of losses exceeding VaR 95% |
| Expected Shortfall | Same as CVaR |
| Max Expected Drawdown | 2 × daily_std × √N |
| Portfolio Volatility | std(port_returns) × √252 |
| Risk Contribution | Each instrument's share of total risk |
| Marginal Risk | Covariance of instrument with portfolio ÷ portfolio vol |
| Diversification Ratio | weighted_vol_sum ÷ portfolio_vol |
| Risk Concentration | Herfindahl index of risk contributions |

## Scenario Analysis

```python
from trainer.icari import ScenarioType

# Run one scenario
d = api.scenario(
    ScenarioType.FLASH_CRASH,
    weights      = {"EURUSD": 0.5, "GBPUSD": 0.3, "BTCUSD": 0.2},
    volatilities = {"EURUSD": 0.008, "GBPUSD": 0.009, "BTCUSD": 0.04},
    currencies   = {"EURUSD": "EUR", "GBPUSD": "GBP", "BTCUSD": "USD"},
)
print(f"Flash Crash return: {d['portfolio_return']:.2%}")
print(f"Resilience: {d['resilience_score']:.2f}")

# Run all 11 scenarios
all_scenarios = api.scenario_all(weights, volatilities, currencies)
print(f"Worst: {all_scenarios['worst_scenario']} ({all_scenarios['worst_resilience']:.2f})")
print(f"Average resilience: {all_scenarios['average_resilience']:.2f}")
```

## Portfolio Optimization

```python
from trainer.icari import OptimizationObjective

# Optimize for maximum Sharpe ratio
result = api.optimize(OptimizationObjective.MAX_SHARPE, returns)
print(f"Optimal weights: {result['weights']}")
print(f"Sharpe: {result['metrics']['sharpe']:.2f}")
print(f"Max Drawdown: {result['metrics']['max_drawdown']:.2%}")

# Compare all 8 objectives
all_objectives = api.optimize_all(returns)
for obj, r in all_objectives.items():
    print(f"{obj}: Sharpe={r['metrics']['sharpe']:.2f}")
```

## Capital Health

```python
health = api.capital_health(
    utilisation      = 0.65,
    current_drawdown = 0.03,
    num_instruments  = 8,
    avg_correlation  = 0.35,
    recent_returns   = last_30_days_returns,
)
print(f"Grade: {health['grade']}")
print(f"Overall: {health['overall_score']:.2%}")
print(f"Efficiency: {health['efficiency_score']:.2%}")
print(f"Diversification: {health['diversification']:.2%}")
```

## Interpretation

- **Diversification Ratio > 2**: Good diversification benefit
- **Risk Concentration (HHI) < 0.25**: Acceptable concentration
- **CVaR 95% < 2%**: Conservative risk profile
- **Capital Health Grade GOOD/EXCELLENT**: Portfolio ready for active allocation
