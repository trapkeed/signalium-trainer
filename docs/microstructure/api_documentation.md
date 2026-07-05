# API Documentation — Phase 22

## MicrostructureAPI

The unified entry point for the Market Microstructure Intelligence Engine.

```python
from trainer.microstructure import MicrostructureAPI
api = MicrostructureAPI(db_path=":memory:")  # or a file path for production
```

---

## Endpoint 1: `analyse()`

Run the full OHLC analysis pipeline.

```python
result = api.analyse(
    bars:       List[OHLCBar],   # required — at least 3 bars recommended
    instrument: str = "",        # e.g. "EURUSD"
    timeframe:  str = "H1",      # e.g. "H1", "D1", "M15"
    ticks:      Optional[List[TickBar]] = None,   # optional tick enrichment
    persist:    bool = True,     # write to database
) -> Dict
```

**Returns:**
```json
{
  "features":    { "body_ratio": 0.72, "trend_efficiency": 0.65, ... },
  "score":       { "trend_score": 0.78, "overall_score": 0.71, ... },
  "market_state": {
    "state":            "STRONG_TREND",
    "confidence":       0.82,
    "momentum_state":   "ACCELERATING",
    "volatility_state": "NORMAL",
    "liquidity_state":  "DEEP",
    "trend_strength":   "VERY_STRONG",
    "reasoning":        ["Trend score 0.82 and persistence 0.88 — STRONG_TREND"]
  },
  "embedding":   { "dim": 32, "embedding": [0.72, 0.15, ...], ... }
}
```

---

## Endpoint 2: `current_state()`

Returns the most recent market state classification.

```python
state = api.current_state() -> Optional[Dict]
```

Returns `None` before any analysis has been run.

---

## Endpoint 3: `scores()`

Returns the most recent microstructure scores.

```python
scores = api.scores(last_n: int = 1) -> List[Dict]
```

**Each score dict:**
```json
{
  "trend_score":        0.78,
  "liquidity_score":    0.65,
  "momentum_score":     0.71,
  "efficiency_score":   0.69,
  "volatility_score":   0.42,
  "compression_score":  0.18,
  "expansion_score":    0.55,
  "overall_score":      0.71
}
```

---

## Endpoint 4: `embedding()`

Returns the most recent 32-dimensional embedding vector.

```python
emb = api.embedding() -> Optional[Dict]
```

**Returns:**
```json
{
  "dim":               32,
  "embedding":         [0.72, 0.15, 0.38, ...],   // 32 values, all in [0, 1]
  "price_action_state": [0.72, 0.15, 0.38, ...],  // 8 dims
  "liquidity_state":   [0.20, 0.05, ...],          // 6 dims
  "momentum_state":    [0.65, 0.40, ...],          // 6 dims
  "volatility_state":  [0.35, 0.10, ...],          // 6 dims
  "efficiency_state":  [0.68, 0.30, ...]           // 6 dims
}
```

---

## Endpoint 5: `historical()`

Returns historical analysis results.

```python
hist = api.historical(
    instrument: Optional[str] = None,
    limit:      int = 100,
) -> Dict
```

---

## Endpoint 6: `get_dashboard()`

Returns one or all dashboard views.

```python
result = api.get_dashboard(name: str = "all") -> Dict
```

**Available names:**
- `"all"` — all 7 views
- `"microstructure_timeline"` — score trends over time
- `"market_state_timeline"` — state distribution over time
- `"momentum_evolution"` — momentum feature time series
- `"volatility_evolution"` — volatility feature time series
- `"liquidity_estimates"` — liquidity feature time series
- `"microstructure_heatmaps"` — avg scores per market state
- `"embedding_explorer"` — embedding dimension statistics

---

## Endpoint 7: `system_stats()`

Returns database and in-memory statistics.

```python
stats = api.system_stats() -> Dict
```

**Returns:**
```json
{
  "in_memory":            { "features": 150, "scores": 150, "states": 150, "embeddings": 150 },
  "database":             { "features": 150, "embeddings": 150, "market_states": 150, "scores": 150 },
  "embedding_dim":        32,
  "tick_available":       false,
  "order_flow_available": false
}
```

---

## Endpoint 8: `order_flow_status()`

Returns a report on what order flow data is currently available.

```python
report = api.order_flow_status() -> Dict
```

**Returns:**
```json
{
  "available":          false,
  "volume_delta":       "pending — requires tick data",
  "bid_ask_imbalance":  "pending — requires Level II data",
  "cumulative_delta":   "pending — requires tick data",
  "footprint":          "pending — requires tick data",
  "order_book":         "pending — requires Level II / DOM data",
  "absorption":         "pending — requires tick + DOM data",
  "iceberg_detection":  "pending — requires tick data",
  "current_source":     "OHLC",
  "degradation":        "graceful — all OHLC features fully functional"
}
```

---

## Lower-Level Components

### MicrostructureScorer

```python
from trainer.microstructure import MicrostructureScorer
scorer = MicrostructureScorer()
score  = scorer.score(features)     # → MicrostructureScore
```

### MarketStateClassifier

```python
from trainer.microstructure import MarketStateClassifier
clf    = MarketStateClassifier()
record = clf.classify(features, score)   # → MarketStateRecord
```

### MicrostructureEmbedder

```python
from trainer.microstructure import MicrostructureEmbedder
emb    = MicrostructureEmbedder()
vector = emb.embed(features, "EURUSD", "H1")   # → MicrostructureEmbedding
batch  = emb.batch_embed([f1, f2, f3], "EURUSD", "H1")
```

### MicrostructureDatabase

```python
from trainer.microstructure import MicrostructureDatabase
db = MicrostructureDatabase("microstructure.db")
db.save_features(features)
db.save_market_state(record, "EURUSD", "H1")
db.state_distribution("EURUSD")     # → {"STRONG_TREND": 45, "COMPRESSION": 12, ...}
db.recent_states("EURUSD", limit=20)
db.stats()
```
