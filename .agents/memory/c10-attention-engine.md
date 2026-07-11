---
name: C10 Attention Engine design
description: How the C10 attention tier/resource/memory package extends the existing attention system without replacing it; where the 4 new signals live; the attention.py → attention_base.py rename.
---

## Package structure after C10

New subpackage `trainer/qde/brain/attention/` adds:
- `attention_types.py` — `AttentionTier` (WATCH/OBSERVE/HIGH_ATTENTION/CRITICAL_ATTENTION), `ResourceBudget`, `TIER_RESOURCES`, `AttentionSnapshot`, `AttentionEvolutionRecord`, `tier_from_composite()`
- `signal_enricher.py` — `ExtendedSignalContext`, `enrich_signals()`, `SignalEnricher`
- `attention_memory.py` — `AttentionMemory` (bounded per-symbol snapshot + transition history)
- `attention_api.py` — `AttentionAPI` wrapping `AttentionEngine` with `calculate_attention()`, `rank_symbols()`, `allocate_resources()`

## Critical rename: attention.py → attention_base.py

`trainer/qde/brain/attention.py` was renamed to `attention_base.py` because Python resolves the new `attention/` package (directory) over the old flat file. Imports in `attention_allocator.py` and `brain.py` were updated accordingly. `AttentionAllocator` is now at `trainer.qde.brain.attention_base`.

**Why:** Python package (dir) takes precedence over module (file) with the same name. The flat file is effectively unreachable once the directory exists.

## Extended OpportunitySignals (backward-compatible)

Added 4 new fields to `OpportunitySignals` (frozen dataclass, all default=0.0):
- `volume_anomaly`, `key_level_proximity`, `spread_change`, `failed_move`

Existing code constructing `OpportunitySignals` with keyword args and omitting these fields still works — the new signals are only populated when the caller uses `enrich_signals(base, ExtendedSignalContext(...))`.

## AttentionWeights.score() — C10 factors included only when weights non-zero

`AttentionWeights` has 4 new weight fields (default=0.0) for the new signals. The existing `DEFAULT_WEIGHTS` is unchanged. New `C10_WEIGHTS` has non-zero weights for all 10 signals. Pass `C10_WEIGHTS` to `AttentionScoreAllocator(weights=C10_WEIGHTS)` to activate them.

## Tier thresholds

WATCH < 0.25, OBSERVE 0.25–0.50, HIGH_ATTENTION 0.50–0.75, CRITICAL_ATTENTION ≥ 0.75

## AttentionEngine.observe() C10 fix

The high-confidence hypothesis override block in `attention_engine.py` now preserves all 10 signal fields (including the 4 new ones) when constructing the replacement `OpportunitySignals`.

## How to apply

- Use `AttentionAPI(engine=existing_engine)` — never construct a second `AttentionEngine`
- Call `calculate_attention(obs, extended_context=ctx)` per tick
- Read `allocate_resources(instrument)` to decide observation/inference cadence
- Call `rebalance()` once per cycle for decay + cross-symbol normalization
- AttentionMemory records tier transitions automatically; read via `tier_transitions()` / `attention_history()`
