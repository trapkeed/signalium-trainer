---
name: Dynamic attention layering
description: Why the multi-signal attention system (attention_engine/attention_allocator/opportunity_scanner) is a layer on top of the existing lightweight AttentionAllocator rather than a replacement, and how it gates reasoning cycles.
---

## Two attention layers, not a duplicate

`trainer/qde/brain/attention.py::AttentionAllocator` already existed and
is wired directly into `brain.py`'s per-tick pipeline
(`QDEBrain.reallocate_attention`) using a narrow salience formula
(readiness + volatility quality + hypothesis count/confidence + recent
experience magnitude). It also owns the actual cross-symbol
normalization math (raw salience → weights summing to 1.0).

The newer `opportunity_scanner.py` / `attention_allocator.py`
(`AttentionScoreAllocator`) / `attention_engine.py` (`AttentionEngine`)
trio adds the richer signal set an explicit feature ask required —
developing hypotheses, volatility expansion, liquidity events, news
impact, unusual behavior, session transitions — each detected as its
own named, independently-decaying component, with continuous
rebalancing (time-decay + re-normalization every cycle, not a one-shot
score).

**Why layered instead of merged:** the two callers have different
lifecycles. `brain.py`'s internal salience is recomputed from
whatever's on `BrainState` every tick as part of its own pipeline and
must stay simple/side-effect-free. The new `AttentionEngine` is driven
externally (like `HypothesisEngine`), needs its own persistent
per-symbol score state with decay, and needs six independently
attributable signal components for observability — cramming that into
the existing narrow allocator would have overloaded its single
responsibility.

**How it avoids duplicating normalization:**
`AttentionScoreAllocator.reallocate()` computes composite scores from
the six signals, then hands them to the *existing*
`attention.AttentionAllocator().reallocate()` for the actual weight
normalization — the softmax-free linear normalization logic exists in
exactly one place. Both allocators end up writing into the same
`BrainState.attention` (`AttentionAllocation`) field, so downstream
consumers only ever look in one place.

## Reasoning-cycle gating

"Only high-attention symbols get expensive reasoning" is implemented by
`AttentionEngine.high_attention_symbols()` / `run_reasoning_cycle()`
calling **the existing** `QDEBrain.reflect()` (already rate-limited,
already the Brain's periodic self-review concept) only for symbols
above a weight threshold — no second reasoning/reflection mechanism was
introduced.

**How to apply:** when a new cross-cutting concern (attention,
priority, urgency, etc.) needs a richer signal set than an existing
narrow implementation provides, prefer building the new signal
detection as its own layer and delegating the parts that already work
correctly (here: cross-symbol normalization) rather than reimplementing
or replacing the existing narrow version outright.
