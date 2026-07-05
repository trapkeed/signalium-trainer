# QuantModelV1 — Input Specification

## Tensor Shape

```
x : torch.Tensor  shape=(batch_size, sequence_length, input_dim)  dtype=float32
```

| Dimension | Default | Description |
|---|---|---|
| `batch_size` | variable | Number of samples per forward pass |
| `sequence_length` | 60 | Historical timesteps per window (configurable) |
| `input_dim` | 128 | Total engineered features per timestep (configurable) |

---

## Feature Groups

All features are pre-computed by the Signalium Feature Store and stored in Parquet format.
The model consumes them as a flat vector of `input_dim` floats per timestep.

### 1. OHLC Window Features
Raw and derived price series values.

| Feature | Description |
|---|---|
| `open`, `high`, `low`, `close` | Raw OHLCV per bar |
| `log_return` | log(close_t / close_{t-1}) |
| `hl_range` | (high - low) / close |
| `body_ratio` | abs(close - open) / (high - low) |

### 2. Technical Indicators
Standard TA features computed over the window.

| Feature | Description |
|---|---|
| `rsi_14`, `rsi_7` | Relative Strength Index |
| `macd`, `macd_signal`, `macd_hist` | MACD and histogram |
| `bb_upper`, `bb_mid`, `bb_lower` | Bollinger Bands (20, 2σ) |
| `ema_9`, `ema_21`, `ema_50` | Exponential moving averages |
| `stoch_k`, `stoch_d` | Stochastic oscillator |
| `atr_14` | Average True Range |
| `adx_14` | Average Directional Index |

### 3. Market Memory Features
Signals derived from price history and regime detection.

| Feature | Description |
|---|---|
| `trend_strength` | Rolling correlation of price with time |
| `mean_reversion_score` | Z-score vs rolling mean |
| `regime_label` | Encoded market regime (trending/ranging) |
| `drawdown` | Running drawdown from recent high |

### 4. Volatility Features

| Feature | Description |
|---|---|
| `realised_vol_5`, `realised_vol_20` | Rolling realised volatility |
| `vol_ratio` | Short/long volatility ratio |
| `vol_zscore` | Volatility Z-score vs baseline |

### 5. Liquidity Features

| Feature | Description |
|---|---|
| `volume`, `volume_ma`, `volume_ratio` | Volume and normalised volume |
| `vwap_delta` | Distance from VWAP |
| `tick_direction` | Tick-level direction (if available) |

### 6. Session & Time Features

| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclical encoding of hour |
| `dow_sin`, `dow_cos` | Cyclical encoding of day of week |
| `session_london`, `session_ny`, `session_tokyo` | Active session one-hots |

### 7. Price Structure Features

| Feature | Description |
|---|---|
| `support_dist`, `resistance_dist` | Distance from nearest S/R level |
| `pivot_type` | Encoded pivot point label |
| `breakout_signal` | Boolean breakout flag |

### 8. Dataset Metadata

| Feature | Description |
|---|---|
| `dataset_version_hash` | Encoded dataset version |
| `feature_version_hash` | Encoded feature pipeline version |

---

## Preprocessing Contract

- All features are expected to be **pre-normalised** by the Feature Store.
- The `InputEncoder` applies `LayerNorm` as a secondary in-model normalization.
- Missing values must be imputed upstream — the model does not handle NaN.
- Features must be ordered consistently across training and inference.

---

## Future Feature Expansion

The `input_dim` is a config parameter. Adding new features requires:
1. Updating the Feature Store pipeline
2. Bumping `input_dim` in `conf/model/quant_v1.yaml`
3. Retraining from scratch (or fine-tuning with the new feature appended as zeros)
