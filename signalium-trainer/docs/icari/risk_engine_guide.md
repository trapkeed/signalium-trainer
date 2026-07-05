# Risk Engine Guide — Phase 24

## Portfolio VaR / CVaR

```python
from trainer.icari import IcariAPI

api = IcariAPI(capital=1_000_000.0)

returns = {
    "EURUSD": [0.01, -0.005, 0.008, -0.003, ...],   # 60+ daily returns
    "GBPUSD": [0.009, -0.004, 0.007, ...],
    "USDJPY": [-0.002, 0.003, -0.001, ...],
}
weights = {"EURUSD": 0.40, "GBPUSD": 0.35, "USDJPY": 0.25}

risk = api.portfolio_risk(returns, weights)
print(f"VaR (95%): {risk['var_95']:.2%}")
print(f"CVaR (95%): {risk['cvar_95']:.2%}")
print(f"Portfolio Vol (annual): {risk['portfolio_volatility']:.2%}")
print(f"Diversification Ratio: {risk['div_ratio']:.2f}")
print(f"Risk Concentration (HHI): {risk['concentration']:.4f}")
```

## Drawdown Protection

```python
# Update equity every bar
result = api.update_equity(current_equity)
print(f"DD Mode: {result['mode']}")
print(f"Risk Mult: {result['risk_multiplier']}")

# Check if trading is allowed
stats = api.drawdown_stats()
if not stats['can_trade']:
    print("EMERGENCY LOCKDOWN — no new positions")
```

### Drawdown Modes

| Mode | Threshold | Position Size |
|---|---|---|
| NORMAL | < 5% | 100% |
| SOFT_LIMIT | 5–10% | 75% |
| HARD_LIMIT | 10–20% | 50% |
| PRESERVATION | manual | 25% |
| RECOVERY | improving | 60% |
| EMERGENCY_LOCKDOWN | > 20% | 0% (blocked) |

## Correlation Analysis

```python
# Full correlation matrix
matrix = api.correlation_matrix(returns)

# Find highly correlated clusters (threshold = 0.70)
clusters = api.correlation_clusters(returns)
print(f"Correlation clusters: {clusters}")
```

## Risk Budgets

```python
from trainer.icari import RiskBudgetType

# View all budgets
budgets = api.get_budgets()

# Override a budget
api.set_budget(RiskBudgetType.DAILY, limit=0.015, label="Tighter Daily")

# Check before consuming
check = api.check_budget(RiskBudgetType.INSTRUMENT, amount=0.04)
if check['allowed']:
    # proceed with trade
    pass

# Reset at start of each day
api.reset_budget(RiskBudgetType.DAILY)
api.reset_budget(RiskBudgetType.WEEKLY)   # start of week
```
