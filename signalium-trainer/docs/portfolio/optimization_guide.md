# Optimization Guide — Phase 20

## Overview

The `PortfolioOptimizer` finds the best allocation weights for a given
optimization objective using a gradient-free grid-search approach (no scipy
required).

---

## Optimization Objectives

| Objective | Enum | Optimizes |
|-----------|------|-----------|
| Maximum Sharpe | `MAX_SHARPE` | (Return − Rf) / Volatility |
| Minimum Volatility | `MIN_VOLATILITY` | Portfolio Volatility |
| Maximum Diversification | `MAX_DIVERSIFICATION` | Diversification Score |
| Maximum Risk-Adjusted Return | `MAX_RISK_ADJUSTED_RETURN` | Return / Max Drawdown |
| Balanced | `BALANCED` | Weighted composite of Sharpe, diversification, risk-adjusted return |

---

## Usage

```python
from trainer.portfolio import PortfolioOptimizer, OptimizationObjective

opt = PortfolioOptimizer(risk_free_rate=0.045)

# Single objective
result = opt.optimize(forecasts, OptimizationObjective.MAX_SHARPE)
print(result.optimal_weights)
print(result.optimal_metrics.expected_sharpe)

# Compare all objectives
all_results = opt.compare_objectives(forecasts)
for obj, res in all_results.items():
    print(f"{obj}: Sharpe={res.optimal_metrics.expected_sharpe:.2f}")

# Efficient frontier
frontier = opt.efficient_frontier(forecasts, n_points=10)
for pt in frontier:
    print(f"Return={pt['expected_return']:.2%}  Vol={pt['expected_volatility']:.2%}")
```

---

## Algorithm

1. Start from equal-weight allocation.
2. For each instrument, try ±2%, ±5%, ±10% weight adjustments.
3. Re-normalise weights to sum to 1.
4. Accept if the objective function improves.
5. Repeat until no improvement (convergence) or `max_iterations` reached.
6. Apply budget constraints via `RiskBudgetEngine.scale_to_budget()`.

---

## OptimizationResult

| Field | Description |
|-------|-------------|
| `objective` | Objective name |
| `optimal_weights` | Dict[instrument → weight] |
| `optimal_metrics` | PortfolioMetrics at optimal weights |
| `iterations` | Iterations used |
| `converged` | Whether the algorithm converged |
| `objective_value` | Final value of the objective function |

---

## Configuration (`conf/portfolio/portfolio.yaml`)

```yaml
optimizer:
  default_objective: max_sharpe
  max_iterations: 200
  max_weight: 0.40
  min_weight: 0.01
  long_only: true
  efficient_frontier_points: 10
```
