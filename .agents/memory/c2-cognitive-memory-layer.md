---
name: C2 cognitive memory layer
description: What was implemented for Phase C2, where it lives, and what was deliberately NOT replaced.
---

## What was built

Three new packages under `trainer/qde/brain/`:

### `memory/` — timeframe-aware structured working memory store
- `WorkingMemoryStore` — one class, multi-symbol, per-symbol per-timeframe candle deques
- Candle limits: M1→500 M5→300 M15→200 H1→150 H4→100
- 7 additional event streams per symbol: volatility, liquidity, trend, failed_breakout, session_transition, spread, correlation
- Public API: `write_memory(instrument, kind, payload, timeframe=)`, `recall(instrument, kind, n, since)`, `candles(instrument, timeframe, n)`
- `to_snapshot(instrument)` / `load_snapshot(instrument, data)` — per-instrument

### `episodic/` — complete market experience records
- `Episode` — all 12 spec fields: symbol, timestamp, features, portfolio, active_hypotheses, action, confidence, reasoning, outcome, trade_result, regime_context, session_context
- `EpisodeFeatures` — 12-dim [0,1] vector for cosine-similarity search
- `EpisodicMemory` — per-symbol bounded FIFO (default 5000), inverted index by session/vol_state/regime/portfolio
- `retrieve_similar(features, instrument, k)` — cosine-sim × `e^(-λ×age_days)` recency; returns `SimilarEpisode` list sorted by combined_weight
- `average_reward()`, `average_confidence()`, `win_rate()` — statistics helpers over results
- Public API: `write_memory(episode)`, `recall(instrument, session=, vol_state=, regime=, ...)`, `retrieve_similar(...)`, `update_outcome(instrument, episode_id, outcome)`
- `to_snapshot(instrument)` / `load_snapshot(instrument, data)`

### `semantic/` — learned market concepts via online centroid clustering
- `SemanticMemory` — threshold-based cosine similarity clustering, no hardcoded rules
- `MarketConcept` — EMA-updated centroid, member count, win_rate, avg_reward; published after min_members (default 3)
- `ConceptKind` — 12 named labels inferred from centroid+outcomes AFTER accumulation (never before)
- `ConceptSeed` — optional caller-supplied label hint; overridden once data-derived kind is inferred
- `_infer_concept_kind()` — reads centroid features + win_rate + avg_reward, assigns UNKNOWN → named kind
- Public API: `write_memory(episode, seed=)`, `recall(feature_vec, n, published_only=True)`, `concepts(published_only=True)`
- `to_snapshot()` / `load_snapshot(data)` — global (not per-instrument)

### `TransitionMemory.transition_statistics()` (added in-place)
- Returns: `target_counts`, `target_probabilities` (uniform-weighted), `target_probabilities_weighted` (e^(-λ×age_days)), `from_regime_probs` (conditional table), `total_transitions`
- Empty history → uniform prior (1/6 each over 6 TransitionTargets)
- All rows in `from_regime_probs` normalized to sum=1

## What was NOT replaced
- `trainer/qde/brain/working_memory.py` — `WorkingMemory` (salience-decaying item buffer) — untouched
- `trainer/qde/brain/long_term_memory.py` — `LongTermMemory` consolidation — untouched
- `trainer/qde/brain/behavioral_memory.py` — `BehavioralMemory` — untouched
- `trainer/qde/brain/memory_decay_engine.py` — decay math — untouched (reused conceptually)
- `trainer/qde/brain/regime/transition_memory.py` — `TransitionMemory` — extended in-place only

## Key design decisions
- `EpisodeFeatures.as_vector()` is 12-dim; all values normalized [0,1] so cosine is well-defined
- Recency decay is `e^(-λ × age_days)` with λ = 1/14 by default (e-fold at 14 days)
- Semantic clustering: cosine sim ≥ 0.80 threshold; below threshold → pending pool; concept forms when ≥ 3 pending vecs are mutually similar
- `WorkingMemoryStore` is separate from `WorkingMemory` — they coexist, answer different questions
- All three new packages have no QDEDecision imports and no cross-package imports from other brain sub-packages

## Files created
- `trainer/qde/brain/memory/__init__.py`
- `trainer/qde/brain/memory/working_memory_store.py`
- `trainer/qde/brain/episodic/__init__.py`
- `trainer/qde/brain/episodic/episodic_memory.py`
- `trainer/qde/brain/semantic/__init__.py`
- `trainer/qde/brain/semantic/semantic_memory.py`

**Why:**
The existing `WorkingMemory` is a priority queue for "what items are still salient", not a structured candle/event buffer for pattern recall.  Episodic and semantic memory had no prior implementation anywhere in the codebase.
