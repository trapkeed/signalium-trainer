# Capital Allocation Guide — Phase 24

## Overview

ICARI transforms Signalium from a forecasting engine into an institutional portfolio management system. It consumes forecasts, execution recommendations, and portfolio analytics but **never modifies forecasting models, executes trades, or connects to brokers**.

## Quick Start

```python
from trainer.icari import IcariAPI, AllocationRequest, SizingMethod

api = IcariAPI(capital=1_000_000.0)

# Create an allocation request from forecast system outputs
req = AllocationRequest(
    instrument          = "EURUSD",
    direction           = "LONG",
    forecast_confidence = 0.82,
    research_score      = 0.78,
    execution_score     = 0.75,
    expected_return     = 0.022,
    expected_drawdown   = 0.048,
    portfolio_exposure  = 0.35,
    correlation         = 0.18,
    market_regime       = "TRENDING",
    volatility          = 0.0075,
    liquidity           = 0.95,
    model_confidence    = 0.80,
    ensemble_confidence = 0.77,
    decision_quality    = 0.79,
    atr                 = 0.00085,
    capital             = 1_000_000.0,
)

result = api.allocate(req)
print(f"Status: {result['status']}")
print(f"Allocation: {result['allocation_frac']*100:.1f}% (${result['allocation_usd']:,.0f})")
```

## Allocation Factors (15)

| Factor | Description |
|---|---|
| `forecast_confidence` | Forecast model confidence (0–1) |
| `research_score` | SAQRA research quality score (0–1) |
| `execution_score` | Execution quality score (0–1) |
| `expected_value` | Expected $/pip from execution system |
| `expected_return` | Expected return fraction |
| `expected_drawdown` | Expected drawdown fraction |
| `portfolio_exposure` | Current portfolio exposure (0–1) |
| `correlation` | Average correlation to existing positions |
| `market_regime` | TRENDING / NORMAL / RANGING / VOLATILE |
| `volatility` | Instrument daily volatility |
| `liquidity` | Liquidity score (0–1) |
| `model_confidence` | Individual model confidence |
| `ensemble_confidence` | Ensemble model confidence |
| `decision_quality` | Historical decision quality score |
| `atr` | Average True Range |

## Regime Multipliers

| Regime | Multiplier |
|---|---|
| TRENDING | 1.10 |
| NORMAL | 1.00 |
| RANGING | 0.85 |
| VOLATILE | 0.65 |

## Allocation Limits

- **Maximum allocation per instrument:** 25% of capital
- **Minimum confidence for approval:** 0.30
- **Exposure penalty:** kicks in above 50% portfolio exposure
- **Emergency lockdown:** activated at 20% portfolio drawdown (no allocations)

## Position Sizing Methods

```python
from trainer.icari import SizingMethod

result = api.size_position(req, method=SizingMethod.KELLY)
result = api.size_position(req, method=SizingMethod.FRACTIONAL_KELLY)
result = api.size_position(req, method=SizingMethod.ATR_ADJUSTED)
result = api.size_position(req, method=SizingMethod.VOLATILITY_ADJUSTED)

# Compare all methods
comparison = api.compare_sizing_methods(req)
```

## Dynamic Sizing

Dynamic sizing applies 10 multipliers to adapt position size to current conditions:

```python
result = api.size_position(
    req,
    use_dynamic      = True,
    current_drawdown = 0.03,
    session          = "LONDON",
)
```

Adjustments: drawdown, exposure, volatility, liquidity, correlation, spread, session, historical_perf, model_health, forecast_quality.
