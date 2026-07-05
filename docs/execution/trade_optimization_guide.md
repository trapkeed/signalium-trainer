# Trade Optimization Guide

## What Gets Optimized

The Execution Intelligence Platform optimizes five dimensions of every trade:

1. **Entry** — Optimal entry zone, order type, and fill price
2. **Stop** — Best stop placement balancing protection vs. probability
3. **Target** — Structured TP levels with partial close rules
4. **Size** — Risk-adjusted lot size within all constraints
5. **Timing** — Session / spread / volatility / news timing window

## Entry Optimization

### Zone Hierarchy

```
Aggressive Entry  ←─ higher fill probability, less optimal price
Primary Entry     ←─ ideal — default recommendation
Secondary Entry   ←─ secondary pullback confirmation
Conservative Entry ←─ deeper pullback — highest probability but may not fill
Late Entry        ←─ last acceptable level before invalidation
Invalidation      ←─ idea cancelled if price reaches here
```

### Order Type Selection Logic

```
EXECUTE_NOW + confidence ≥ 0.70 + spread OK → MARKET
EXECUTE_NOW + lower confidence              → LIMIT at primary
WAIT + at zone                             → LIMIT with 8h expiry
WAIT + away from zone                      → LIMIT 24h GTC
AVOID                                      → CONDITIONAL (alert only)
High momentum (>0.60)                      → STOP (breakout entry)
```

## Stop Optimization

The system provides 6 stop variants — the trader selects based on style:

| Stop Type | Use Case |
|-----------|----------|
| Initial (1.5× ATR) | Default — balanced risk |
| Adaptive (1.2× ATR) | Intra-trade tightening |
| ATR (2.0× ATR) | More room for noise |
| Structure (invalidation) | Pure technical placement |
| Volatility (2.5× vol-ATR) | Vol-regime adjusted |
| Emergency (3.5× ATR) | Hard maximum |

## Target Optimization

### Partial Close Schedule

```
TP1 (1:1 RR) → Close 50% of position
TP2 (1.5:1)  → Close 30% of position
TP3 (2:1)    → Close remainder to 20% runner
Runner       → Hold to full expected-return target
```

### Best Expectancy Level

Each TP level is evaluated by Expected Value:
```
EV(TPn) = P(TPn_hit) × RR(TPn)
```
The level with the highest EV is flagged as `best_expectancy_level`.

## Management Optimization

### Breakeven Rule
Move stop to breakeven (entry ± spread) when price reaches TP1.

### Trail Stop Rule
After TP1, trail stop at 1× ATR behind price, minimum step 0.25× ATR.

### Scale-In Rule (when applicable)
If confidence > 0.75 and momentum > 0.20 in trade direction:
- Add 50% of original size at conservative entry level
- Total risk remains within portfolio risk limits

## Batch Optimization

```python
from trainer.execution import ExecutionPlanner, ExecutionForecast, Session

planner = ExecutionPlanner()

plans = planner.plan_batch([
    ExecutionForecast("EURUSD", "BUY",  0.75, 0.08, 0.10, 1.0850, 0.0012, 0.0001),
    ExecutionForecast("XAUUSD", "BUY",  0.80, 0.12, 0.15, 1950.0, 3.50,   0.50),
    ExecutionForecast("NAS100", "SELL", 0.65, 0.15, 0.18, 14500., 45.0,   2.00),
])

# Plans sorted by score descending
for p in plans:
    print(p.instrument, p.score.grade.value, p.score.overall_score)
```
