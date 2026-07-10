---
name: Setup queue opportunity lifecycle
description: A third, distinct state vocabulary (WATCHING/ARMED/CONFIRMING/READY/EXECUTING/COOLDOWN) for observing developing trade setups over time before entry.
---

QDE Brain's opportunity queue (`setup_lifecycle.py`, `opportunity_tracker.py`,
`setup_queue.py`) tracks *multiple developing opportunities per symbol*, each
referencing a hypothesis, progressing WATCHING→ARMED→CONFIRMING→READY→
EXECUTING→COOLDOWN. Entry (READY) requires sustained evidence across several
observed ticks (`confirmation_ticks`), not one strong reading.

**Why:** this is intentionally the *third* lifecycle vocabulary in the
codebase, not a rename of an existing one:
- `qde_observer_types.ObserverState` is per-symbol, singular, and covers the
  in-trade side too (POSITION_ACTIVE/MANAGING/EXITING) — one slot per symbol.
- `hypothesis_lifecycle.HypothesisMaturity` is a pure, memoryless projection
  of one hypothesis's own fields (no evidence-over-time tracking).
- `SetupState` adds the missing piece: multiple concurrent, memoryful
  per-opportunity queues that require sustained (not instantaneous) evidence.

**How to apply:** `SetupQueue` is constructed externally by a caller (like
`HypothesisEngine`/`AttentionEngine`), not embedded in `QDEBrain.on_market_update`.
It reuses `conviction_engine`/`uncertainty_engine`/`hypothesis_lifecycle`
for evidence and never re-derives them. High market uncertainty can only ever
push an opportunity backward/hold it, never advance it toward READY.
`ready_opportunities()`/`mark_executing()`/`complete()` are the integration
seam for a decision layer (e.g. `qde_inference.py`) to consult — this queue
does not wire itself into `qde_inference.py`, matching how `HypothesisEngine.
executable_hypotheses()` is also left as a caller-consulted seam.
