# Position Sizing Guide — Phase 24

## Method Selection

```python
from trainer.icari import IcariAPI, SizingMethod, AllocationRequest

api = IcariAPI(capital=1_000_000.0)
req = AllocationRequest(instrument="EURUSD", ...)

# Static sizing (single method)
result = api.size_position(req, method=SizingMethod.KELLY, use_dynamic=False)
print(f"Kelly units: {result['units']:.2f}")
print(f"Risk fraction: {result['risk_frac']:.2%}")

# Dynamic sizing (recommended)
result = api.size_position(
    req,
    use_dynamic      = True,
    current_drawdown = 0.03,   # current portfolio drawdown
    session          = "LONDON",
)
print(f"Dynamic units: {result['units']:.2f}")
print(f"Adjustments: {result['adjustments']}")

# Compare all methods
comparison = api.compare_sizing_methods(req)
for method, r in comparison.items():
    print(f"{method}: {r['units']:.2f} units (risk={r['risk_frac']:.2%})")
```

## Method Reference

### Kelly Criterion

```
f* = (p × (b+1) - 1) / b

p = forecast_confidence (win probability)
b = |expected_return / expected_drawdown|  (win/loss ratio)
```

Hard cap at 5% risk per trade.

### Fractional Kelly

```
f* = Kelly_f* × kelly_frac (default 0.5)
```

Recommended: `kelly_frac=0.25` for conservative sizing.

### ATR Adjusted

```
stop_distance = ATR × atr_multiplier
units = risk_amount / stop_distance
risk_amount = capital × risk_frac
```

### Volatility Adjusted

```
vol_mult = max(0.25, 1.0 - volatility × 10)
units = base_units × vol_mult
```

At 5% daily vol → 50% size. At 10% daily vol → 25% size.

## Dynamic Adjustment Factors

| Factor | Effect |
|---|---|
| drawdown | 75% at soft limit; 25% at hard limit |
| exposure | 85% at 50–70% exposure; 25% at >90% |
| volatility | reduces above 2% daily vol |
| liquidity | scales linearly 0.1–1.0 |
| correlation | 50% above 0.70 correlation |
| spread | reduces when spread > threshold |
| session | TOKYO=0.80, SYDNEY=0.70, OVERLAP=1.10 |
| historical_perf | 0.5–1.5× based on strategy win rate |
| model_health | 0.3–1.0× based on model health score |
| forecast_quality | 0.2–1.0× based on confidence |

The composite adjustment = geometric_mean(all factors).
