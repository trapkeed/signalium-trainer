# Risk Engine Guide

## Purpose

The Risk Engine produces a complete quantitative risk profile for every trade recommendation. It never adjusts forecasting models — it reads their outputs and derives risk statistics.

## RiskProfile Fields

| Field | Description |
|-------|-------------|
| `prob_success` | P(reaching TP2 or better) |
| `prob_failure` | P(stop loss hit) |
| `risk_reward_ratio` | TP2 distance / stop distance |
| `expected_value` | P(win)×reward − P(loss)×risk (price units) |
| `expected_value_pct` | EV as fraction of current price |
| `tail_risk` | 95th percentile adverse move (1.645 × σ × price) |
| `worst_case` | Emergency stop distance |
| `best_case` | Runner target distance |
| `expected_drawdown` | 60% stop + 40% tail risk blended |
| `downside_deviation` | Stop distance × (1 + volatility) |
| `upside_potential` | Runner distance × confidence |

## Probability Model

```
P(success) = take_profit.tp2_hit_prob    [from TakeProfitEngine]
P(failure) = stop_loss.expected_hit_prob [from StopLossEngine]

Expected Value = P(success) × TP2_distance − P(failure) × stop_distance
```

## Tail Risk Calculation

```
σ_price = volatility × current_price
tail_risk = min(1.645 × σ_price, structure_stop_distance)
```

The cap against structure stop prevents unrealistic tail risk estimates when volatility is temporarily elevated.

## Usage

```python
from trainer.execution import RiskEngine, ExecutionForecast, Session

engine = RiskEngine()

risk = engine.analyze(
    forecast   = fc,
    stop_loss  = stop,
    take_profit = tp,
)

print(risk.risk_reward_ratio)   # e.g. 1.5
print(risk.is_positive_ev())    # True / False
print(risk.to_dict())
```

## Risk/Reward Guidelines

| RR Ratio | Assessment |
|----------|-----------|
| < 1.0 | Reject — negative asymmetry |
| 1.0–1.5 | Marginal — only with very high confidence |
| 1.5–2.0 | Acceptable institutional standard |
| 2.0–3.0 | Good — typical professional trade |
| > 3.0 | Excellent — high asymmetry setup |
