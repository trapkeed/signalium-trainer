# Microstructure Guide — Phase 22

## Overview

The Market Microstructure Intelligence Engine (MMIE) enriches Signalium's understanding of price behaviour by analysing HOW price moves, not merely WHERE it may move.

The engine operates on a priority-degradation model:
1. **OHLC** — fully operational today
2. **Tick** — interface ready, implementation pending
3. **Level II / DOM** — interface ready, implementation pending
4. **Broker execution** — interface ready, implementation pending

All features degrade gracefully — the system operates at full capacity on OHLC alone.

## Architecture

```
Feature Store
      ↓
Microstructure Layer          ← Phase 22 MMIE
      ↓
Market Memory
      ↓
Forecasting Models
```

The layer enriches market representations without replacing the existing feature pipeline.

## Feature Families

### Price Action (12 features)

Derived from single-bar and multi-bar OHLC analysis:

| Feature | Range | Description |
|---|---|---|
| body_ratio | 0–1 | Body as fraction of total range |
| upper_wick_ratio | 0–1 | Upper wick as fraction of range |
| lower_wick_ratio | 0–1 | Lower wick as fraction of range |
| close_position | 0–1 | Where did price close? 0=low, 1=high |
| open_position | 0–1 | Where did price open? 0=low, 1=high |
| candle_efficiency | 0–1 | Same as body_ratio — directional efficiency |
| directional_efficiency | -1 to +1 | Signed efficiency (+= bullish, -= bearish) |
| compression | 0–1 | 1 - body_ratio — wickiness |
| expansion | 0–1 | = body_ratio — directionality |
| impulse_strength | 0–1 | Size of the directional impulse |
| retracement_quality | 0–1 | Quality of pullback (1=clean shallow, 0=deep choppy) |
| continuation_probability | 0–1 | Estimated probability of continuation |

### Volatility (8 features)

Derived from ATR and range analysis over a configurable window:

| Feature | Description |
|---|---|
| atr_expansion | Current range above average (0–1) |
| atr_compression | Current range below average (0–1) |
| volatility_clustering | Std/mean of ranges — volatility bunching |
| volatility_persistence | Autocorrelation of ranges (lag 1) |
| volatility_shock | Z-score > 2.5 → 1.0 |
| noise_ratio | 1 - avg(body_ratio) — proportion of wicks |
| range_compression | Binary: range < 70% of ATR |
| range_expansion | Binary: range > 130% of ATR |

### Momentum (6 features)

Derived from close-to-close price moves:

| Feature | Description |
|---|---|
| momentum_acceleration | Recent moves larger than early moves |
| momentum_deceleration | Recent moves smaller than early moves |
| momentum_exhaustion | Large ranges but small bodies (wicks dominate) |
| directional_persistence | Fraction of bars moving in the same direction |
| trend_energy | Directional displacement / total displacement |
| impulse_decay | How much has peak impulse decayed? |

### Liquidity Proxies (8 features)

Estimated from OHLC without order book data:

| Feature | Description |
|---|---|
| spread_proxy | Average wick ratio — proxy for bid/ask spread |
| gap_frequency | Fraction of bars with open ≠ prior close |
| large_candle_frequency | Fraction of bars > 1.5× average range |
| compression_zone_score | Fraction of bars < 60% of ATR |
| reaction_zone_score | Fraction of bars closing at extremes |
| breakout_strength | Large + directional + close at extreme |
| reversal_strength | Opposite direction + strong body |
| liquidity_vacuum_prob | Composite vacuum probability |

### Market Efficiency (6 features)

Derived from price displacement mathematics:

| Feature | Description |
|---|---|
| trend_efficiency | Efficiency Ratio: net_move / total_move |
| mean_reversion_probability | Reversal rate across the window |
| randomness_index | 1 - trend_efficiency |
| fractal_dimension | 1.0=trend, 2.0=random walk |
| directional_stability | Consistency of directional bias |
| noise_level | Composite noise measurement |

## Usage

```python
from trainer.microstructure import MicrostructureAPI, OHLCBar
import time

api = MicrostructureAPI(db_path="microstructure.db")

# Create OHLC bars from your data
bars = [
    OHLCBar(timestamp=time.time() - (20-i)*3600,
            open=1.10000 + i*0.00050,
            high=1.10200 + i*0.00050,
            low=1.09800  + i*0.00050,
            close=1.10150 + i*0.00050,
            timeframe="H1", instrument="EURUSD")
    for i in range(20)
]

# Run full analysis
result = api.analyse(bars, instrument="EURUSD", timeframe="H1")

print(result["market_state"]["state"])      # e.g. "STRONG_TREND"
print(result["score"]["overall_score"])     # e.g. 0.74
print(result["embedding"]["dim"])           # 32
```

## Ablation Testing

The MMIE supports ablation testing to measure each feature family's contribution:

```python
# Train with only OHLC baseline features
baseline_result = model.train(features=["ohlc_only"])

# Train with OHLC + microstructure features
micro_result = model.train(features=["ohlc", "microstructure"])

# Train with OHLC + microstructure + embeddings
full_result = model.train(features=["ohlc", "microstructure", "embeddings"])
```
