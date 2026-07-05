---
name: Phase 22 microstructure package
description: Design rules and gotchas for the trainer/microstructure/ MMIE.
---

## The Package

`trainer/microstructure/` — 15 modules (+ __init__), stdlib + loguru only.
Answers HOW price moves; enriches the feature pipeline without replacing it.
Positioned between Feature Store and Market Memory.

## Module Load Order (for spec_from_file_location tests)

```
micro_types → micro_models → price_action → volatility_micro → momentum_micro →
liquidity_proxy → market_efficiency → tick_interface → order_flow_interface →
micro_embeddings → state_classifier → micro_scorer →
micro_database → micro_dashboard → micro_api
```

## Market States (12)

TRENDING, STRONG_TREND, WEAK_TREND, RANGE, COMPRESSION, EXPANSION,
BREAKOUT, FAKE_BREAKOUT, REVERSAL, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITION

## Embedding Dimensions

32 total = price_action(8) + liquidity(6) + momentum(6) + volatility(6) + efficiency(6)
Asserted at runtime in MicrostructureEmbedder.embed(). Never change sub-dims without updating EMBEDDING_TOTAL.

## MicrostructureScore Fields (8)

trend_score, liquidity_score, momentum_score, efficiency_score,
volatility_score, compression_score, expansion_score, overall_score

## Scorer Weights (must sum to 1.0)

trend=0.25, momentum=0.20, efficiency=0.20, volatility=0.15,
liquidity=0.10, compression=0.05, expansion=0.05

## Database Tables (6)

microstructure_features, microstructure_embeddings, market_states,
microstructure_scores, liquidity_estimates, momentum_metrics

## Dashboard Views (7)

microstructure_timeline, market_state_timeline, momentum_evolution,
volatility_evolution, liquidity_estimates, microstructure_heatmaps, embedding_explorer

## API Endpoints (8)

analyse, current_state, scores, embedding, historical,
get_dashboard, system_stats, order_flow_status

## Critical Gotcha

make_features() test helper in test_microstructure.py must list ALL feature kwargs
explicitly (including candle_efficiency). Python rejects unexpected kwargs → TypeError.
Always add new features to make_features() defaults when adding fields to MicrostructureFeatures.

## Tick / Order Flow

Both interfaces return None when data is unavailable — OHLC features are always the fallback.
TickInterface.is_available() and OrderFlowInterface.is_available() both return False until wired.
No changes needed in the rest of the pipeline when data becomes available — stubs are already called.

## Test Count

- Phase 22 microstructure: 137 tests
- Total across all phases: 761 tests (all green)
