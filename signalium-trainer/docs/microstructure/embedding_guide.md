# Embedding Guide — Phase 22

## Overview

The Microstructure Embedder converts the full feature set into a **32-dimensional dense vector** that can be fed directly into forecasting models as an enriched market representation.

## Architecture

```
MicrostructureFeatures (44 features across 5 families)
        ↓
MicrostructureEmbedder
        ↓
MicrostructureEmbedding (32 dims)
   ├── price_action_state  [8 dims]
   ├── liquidity_state     [6 dims]
   ├── momentum_state      [6 dims]
   ├── volatility_state    [6 dims]
   └── efficiency_state    [6 dims]
```

## Sub-Vector Composition

### Price Action State (dims 0–7)
```
[body_ratio, upper_wick_ratio, lower_wick_ratio, close_position,
 open_position, candle_efficiency, impulse_strength, continuation_probability]
```

### Liquidity State (dims 8–13)
```
[spread_proxy, gap_frequency, breakout_strength, reversal_strength,
 liquidity_vacuum_prob, large_candle_frequency]
```

### Momentum State (dims 14–19)
```
[trend_energy, momentum_acceleration, directional_persistence,
 impulse_decay, momentum_exhaustion, directional_efficiency_norm]
```
Note: `directional_efficiency` is mapped from [-1,+1] → [0,1]

### Volatility State (dims 20–25)
```
[atr_expansion, atr_compression, volatility_clustering,
 volatility_persistence, noise_ratio, volatility_shock]
```

### Efficiency State (dims 26–31)
```
[trend_efficiency, mean_reversion_probability, randomness_index,
 directional_stability, (1-noise_level), (1-fractal_dim_norm)]
```

## Generating Embeddings

```python
from trainer.microstructure import MicrostructureAPI, OHLCBar

api = MicrostructureAPI()

# After running analysis
result = api.analyse(bars, "EURUSD", "H1")
emb = result["embedding"]

# Access the full 32-dim vector
vector = emb["embedding"]   # List[float], all values in [0, 1]

# Access individual sub-vectors
pa_state   = emb["price_action_state"]   # 8 dims
liq_state  = emb["liquidity_state"]      # 6 dims
mom_state  = emb["momentum_state"]       # 6 dims
vol_state  = emb["volatility_state"]     # 6 dims
eff_state  = emb["efficiency_state"]     # 6 dims
```

## Feeding into Forecasting Models

The 32-dim vector can be concatenated with existing model features:

```python
# Example: concatenate with forecast features
import math
forecast_features = [...]           # existing feature vector
micro_embedding   = vector          # 32-dim microstructure state

combined = forecast_features + micro_embedding

# Or pass microstructure vectors separately per-head in a multi-head model:
# Head 1: existing features
# Head 2: microstructure embedding (32 dims)
# Head 3: price action state (8 dims)
# Head 4: volatility state (6 dims)
```

## All Values in [0, 1]

All embedding values are clamped to [0, 1] during construction. No normalisation is required before feeding into models.

## Batch Embedding

```python
from trainer.microstructure import MicrostructureEmbedder, MicrostructureFeatures

embedder = MicrostructureEmbedder()

# Embed a batch of feature objects
features = [...]   # List[MicrostructureFeatures]
embeddings = embedder.batch_embed(features, "EURUSD", "H1")

# Extract just the vectors for model training
vectors = [e.embedding for e in embeddings]
```

## Embedding Explorer Dashboard

```python
dash = api.get_dashboard("embedding_explorer")
# Returns:
# {
#   "view":      "EMBEDDING_EXPLORER",
#   "total":     150,
#   "dim":       32,
#   "dim_means": [0.45, 0.22, ...],  # per-dimension mean across history
#   "sub_means": {
#     "price_action": [...],
#     "liquidity":    [...],
#     "momentum":     [...],
#     "volatility":   [...],
#     "efficiency":   [...],
#   }
# }
```

## Future: Tick Integration

When tick data is available, tick features will augment the embedding:
- `tick_imbalance` can replace or supplement `directional_efficiency_norm`
- `tick_velocity` can replace or supplement `large_candle_frequency`
- `tick_entropy` can replace or supplement `randomness_index`

The 32-dim structure will remain unchanged — tick features will be blended in.
