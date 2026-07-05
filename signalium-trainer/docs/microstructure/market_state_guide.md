# Market State Guide — Phase 22

## Overview

The Market State Classifier automatically classifies current market conditions into one of 12 states. Classification is deterministic and based on microstructure scores derived from OHLC data.

## 12 Market States

| State | Description | Key Indicators |
|---|---|---|
| STRONG_TREND | Clean, persistent directional movement | trend_score > 0.75, directional_persistence > 0.70 |
| TRENDING | Moderate directional movement | trend_score > 0.55 |
| WEAK_TREND | Low-energy directional bias | trend_score > 0.35 |
| RANGE | Sideways, non-directional | trend_score < 0.35, volatility < 0.50 |
| COMPRESSION | Narrowing range, coiling | compression_score > 0.65 |
| EXPANSION | Widening range, impulse | expansion_score > 0.65 |
| BREAKOUT | Large, directional bar breaking structure | breakout_strength > 0.70, expansion_score > 0.65 |
| FAKE_BREAKOUT | Breakout immediately reversed | breakout + reversal_strength > 0.60, persistence < 0.45 |
| REVERSAL | Direction change with strong close | reversal_strength > 0.60, exhaustion > 0.65 |
| HIGH_VOLATILITY | Elevated, disruptive volatility | volatility_score > 0.75 |
| LOW_VOLATILITY | Suppressed volatility | volatility_score < 0.25, compression_score > 0.65 |
| TRANSITION | Mixed signals, no dominant regime | fallback state |

## Classification Priority

States are evaluated in strict priority order:

```
1. HIGH_VOLATILITY     ← checked first (emergency state)
2. LOW_VOLATILITY
3. STRONG_TREND
4. BREAKOUT / FAKE_BREAKOUT
5. REVERSAL
6. TRENDING
7. WEAK_TREND
8. COMPRESSION
9. EXPANSION
10. RANGE
11. TRANSITION         ← fallback (default)
```

## Sub-State Classifications

Each state record includes 3 sub-state classifications:

### Momentum State
| State | Condition |
|---|---|
| EXHAUSTED | momentum_exhaustion ≥ 0.70 |
| ACCELERATING | momentum_acceleration ≥ 0.50 |
| DECELERATING | momentum_deceleration ≥ 0.50 |
| REVERSING | reversal_strength ≥ 0.60 |
| NEUTRAL | none of the above |

### Volatility State
| State | Condition |
|---|---|
| SHOCK | volatility_shock ≥ 0.80 |
| EXPANDING | atr_expansion ≥ 0.60 |
| CONTRACTING | atr_compression ≥ 0.60 |
| CLUSTERING | volatility_clustering ≥ 0.60 |
| NORMAL | none of the above |

### Liquidity State
| State | Condition |
|---|---|
| VACUUM | liquidity_vacuum_prob ≥ 0.70 |
| DEEP | spread_proxy ≤ 0.20, large_candle_freq ≤ 0.15 |
| THIN | spread_proxy ≥ 0.55 |
| NORMAL | none of the above |

## Usage

```python
from trainer.microstructure import MicrostructureAPI

api = MicrostructureAPI()
result = api.analyse(bars, "EURUSD", "H1")

state = result["market_state"]
print(state["state"])             # "STRONG_TREND"
print(state["confidence"])        # 0.82
print(state["momentum_state"])    # "ACCELERATING"
print(state["volatility_state"])  # "EXPANDING"
print(state["liquidity_state"])   # "NORMAL"
print(state["trend_strength"])    # "VERY_STRONG"
print(state["reasoning"])         # ["Trend score 0.82 and persistence 0.88 — STRONG_TREND"]
```

## Market State Timeline Dashboard

```python
timeline = api.get_dashboard("market_state_timeline")
# Returns state counts by day:
# {
#   "view": "MARKET_STATE_TIMELINE",
#   "periods": ["2025-01-01", "2025-01-02", ...],
#   "data": {
#     "2025-01-01": {"TRENDING": 12, "COMPRESSION": 6, ...},
#     "2025-01-02": {"STRONG_TREND": 15, "BREAKOUT": 3, ...},
#   }
# }
```

## Microstructure Heatmaps

```python
heatmap = api.get_dashboard("microstructure_heatmaps")
# Average scores per market state:
# {
#   "STRONG_TREND": {
#     "trend_score":       0.88,
#     "momentum_score":    0.82,
#     "efficiency_score":  0.75,
#     "volatility_score":  0.45,
#     "liquidity_score":   0.72,
#   },
#   "COMPRESSION": {
#     "trend_score":       0.15,
#     "compression_score": 0.82,
#     ...
#   }
# }
```

## Downstream Use in Decision Memory (Phase 21.6)

The market state is captured in the `DecisionSnapshot` as `market_regime`:

```python
from trainer.microstructure import MicrostructureAPI
from trainer.memory import MemoryAPI, DecisionSnapshot, DecisionType, DecisionReasoning

micro_api  = MicrostructureAPI()
memory_api = MemoryAPI()

# Run microstructure analysis first
result = micro_api.analyse(bars, "EURUSD", "H1")

# Feed state into decision snapshot
snap = DecisionSnapshot(
    market_regime = result["market_state"]["state"],     # e.g. "STRONG_TREND"
    volatility_regime = result["market_state"]["volatility_state"],
    forecast_confidence = 0.75,
    ...
)

# Record decision with enriched microstructure context
memory_api.record_decision(DecisionType.BUY, "EURUSD", snap, reasoning)
```
