---
name: Regime transition forecasting design
description: Key decisions for trainer/qde/brain/regime/ — infer_target priority, cosine-sim prediction, 6 targets vs 16 MIE regimes.
---

## Rule
`trainer/qde/brain/regime/` adds *sequence memory + forward probability* on top of existing regime work. It does NOT re-classify regimes (that's `MarketRegimeEngine`) and does NOT re-accumulate per-regime outcome quality (that's `SymbolPersonality.regime_outcome_quality`).

## infer_target priority order (order is critical — conflicts exist)
1. BREAKOUT — to-regime is in breakout family
2. VOL_CONTRACTION — from-volatile to non-volatile (beats RANGE_FORMATION)
3. RANGE_FORMATION — to-ranging from non-ranging (beats TREND_EXHAUSTION)
4. TREND_FORMATION — from-non-trending to trending
5. TREND_EXHAUSTION — from-trending to explicit exhaustion regime (REVERSAL, WEAK_TREND)
6. VOL_EXPANSION — from-non-volatile to volatile
7. TREND_EXHAUSTION fallback — from-trending to anything else
8. RANGE_FORMATION fallback

**Why:** `TRENDING → RANGE` is both "trend exit" and "range entry" — RANGE_FORMATION must come before the generic TREND_EXHAUSTION fallback or it always maps to TREND_EXHAUSTION. `VOLATILE → RANGE` is both VOL_CONTRACTION and RANGE_FORMATION — VOL_CONTRACTION must come before RANGE_FORMATION.

## Prediction algorithm (no hardcoded thresholds)
- `RegimeSnapshot` records per-tick observables (ATR, spread, volatility, momentum, vol-rank, trend_valid, microstructure/vol quality).
- `TransitionFeatureVector` (10-dim) computed from N-tick window via OLS slopes + ATR CV + trend-valid ratio.
- For each historical `TransitionRecord`, cosine-similarity between current FV and the record's cached FV, weighted by recency (`exp(-age/τ)`).
- Raw scores per `TransitionTarget` → softmax → blended with uniform prior scaled by confidence.
- Confidence = `min(1, n_transitions / min_samples)`.

## 6 TransitionTargets vs 16 MIE regimes
Targets describe *direction of change* not current state: TREND_FORMATION, TREND_EXHAUSTION, VOL_EXPANSION, VOL_CONTRACTION, BREAKOUT, RANGE_FORMATION.

## Integration interface (5 points)
- `on_observation(instrument, inp)` → ingest tick
- `record_transition(instrument, from, to)` → confirmed change, caches FV immediately
- `context_for_inference()` → 20 `regime_*` keys
- `conviction_modifier()` → ±0.20
- `attention_signal()` → 0-1 (for unusual_behavior boost)
- `risk_appetite_modifier()` → -0.40..+0.20

## How to apply
Wire `on_observation` into `BrainStateManager.tick_symbol` (alongside `LiquidityMapEngine.on_bar`). Wire `record_transition` when consecutive `market_regime` values differ. Merge `context_for_inference()` into `brain_context`.
