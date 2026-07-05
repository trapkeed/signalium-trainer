# Developer Guide — Phase 22

## Module Inventory

```
trainer/microstructure/
├── micro_types.py          — Enums, constants, dimension config
├── micro_models.py         — Data models (OHLCBar, Features, Score, State, Embedding)
├── price_action.py         — PriceActionAnalyzer
├── volatility_micro.py     — VolatilityMicrostructure
├── momentum_micro.py       — MomentumMicrostructure
├── liquidity_proxy.py      — LiquidityProxy
├── market_efficiency.py    — MarketEfficiency
├── tick_interface.py       — TickInterface + TickBar (OHLC-first, tick-optional)
├── order_flow_interface.py — OrderFlowInterface stubs (future)
├── micro_embeddings.py     — MicrostructureEmbedder
├── state_classifier.py     — MarketStateClassifier
├── micro_scorer.py         — MicrostructureScorer
├── micro_database.py       — MicrostructureDatabase (SQLite, 6 tables)
├── micro_dashboard.py      — MicrostructureDashboard (7 views)
├── micro_api.py            — MicrostructureAPI (unified entry point)
└── __init__.py             — 40+ public exports
```

## Dependency Stack

```
micro_types          ← no dependencies
micro_models         ← micro_types
price_action         ← micro_models
volatility_micro     ← micro_models
momentum_micro       ← micro_models
liquidity_proxy      ← micro_models
market_efficiency    ← micro_models
tick_interface       ← micro_models, micro_types
order_flow_interface ← (no trainer imports)
micro_embeddings     ← micro_models, micro_types
state_classifier     ← micro_models, micro_types
micro_scorer         ← micro_models
micro_database       ← micro_models
micro_dashboard      ← micro_models, micro_types
micro_api            ← everything above
__init__.py          ← micro_api (re-exports all)
```

## Key Design Rules

### 1. No torch, no numpy, no pandas
All modules are pure Python + stdlib + loguru only. This allows the package to be imported anywhere without the ML stack.

### 2. Graceful degradation
Tick and order flow features are always `Optional[float]`. Downstream code checks `features.has_tick_features` before using them.

### 3. OHLC-first
Every analyzer accepts `List[OHLCBar]` and produces meaningful results with 3+ bars. Single-bar analysis is supported for price action only.

### 4. Spec-from-file-location imports in tests
```python
import importlib.util
spec = importlib.util.spec_from_file_location("trainer.microstructure.micro_api", path)
mod  = importlib.util.module_from_spec(spec)
sys.modules["trainer.microstructure.micro_api"] = mod
spec.loader.exec_module(mod)
```
Never use `importlib.import_module("trainer.microstructure.*")` — that triggers torch import.

### 5. SQLite `:memory:` persistence
All databases keep a `_persistent_conn` for `:memory:` mode. The same connection is reused; a new `connect(":memory:")` creates an empty DB.

### 6. Embedding is always 32 dims
`EMBEDDING_TOTAL = 32`. This is checked with an assertion in `MicrostructureEmbedder.embed()`. Do not change sub-dimensions without updating `EMBEDDING_DIMS`.

## Running Tests

```bash
cd signalium-trainer
python3 -m unittest tests/unit/test_microstructure.py -v
```

## Adding a New Analyzer

1. Create `trainer/microstructure/my_analyzer.py`
2. Accept `List[OHLCBar]` and `MicrostructureFeatures` as inputs
3. Write computed values into `features` fields; return `features`
4. Add new fields to `MicrostructureFeatures` in `micro_models.py`
5. Call analyzer in `MicrostructureAPI.analyse()`
6. Export from `__init__.py`

## Integrating Tick Data (future)

When tick data becomes available:

1. Implement the stubs in `TickInterface` (all method signatures are defined)
2. Call `TickInterface.enrich_features(features, ticks)` inside `MicrostructureAPI.analyse()`
3. The pipeline automatically routes tick values into `MicrostructureFeatures.tick_*` fields
4. Downstream scoring and classification can be updated to consume tick-enriched features
5. Override `TickInterface.is_available()` to return `True`

No other code changes are required — the interface is fully wired.

## Integrating Order Flow Data (future)

When Level II or execution data becomes available:

1. Implement the stubs in `OrderFlowInterface`
2. Add `OrderFlowBar` or `OrderBookLevel` objects to the analysis pipeline
3. Blend computed values into the `MicrostructureFeatures` liquidity fields
4. Override `OrderFlowInterface.is_available()` to return `True`

## Adding Embedding Dimensions

The current 32-dim design is rigid (checked at assertion time). To extend:

1. Update `EMBEDDING_DIMS` in `micro_types.py`
2. Update `EMBEDDING_TOTAL`
3. Update the corresponding sub-vector in `MicrostructureEmbedder.embed()`
4. Update assertion in the embedder
5. Bump any downstream model input sizes

## Database Schema

```sql
microstructure_features  (bar_id PK, timestamp, instrument, timeframe, granularity, features_json)
microstructure_embeddings(id, timestamp, instrument, timeframe, embedding, dim)
market_states            (id, timestamp, instrument, timeframe, state, confidence, ...)
microstructure_scores    (id, timestamp, instrument, timeframe, *_score fields)
liquidity_estimates      (id, timestamp, instrument, timeframe, liquidity fields)
momentum_metrics         (id, timestamp, instrument, timeframe, momentum fields)
```

Indexes: instrument, timestamp, state (on market_states).
