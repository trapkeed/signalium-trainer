---
name: QDE forgetting engine / memory decay engine split
description: Why regime decay and context decay needed new fields (formed_regime, regime_last_seen) rather than reusing existing decay, and how the two new modules divide pure math from orchestration.
---

`memory_decay_engine.py` (pure functions: `recency_weight`, `relevance_weight`,
`context_decay`, `regime_decay`) and `forgetting_engine.py`
(`ForgettingEngine` orchestration facade) were added to
`trainer/qde/brain/` after confirming nearly every subsystem already had its
own decay: working-memory salience decay, attention half-life decay, EMA
trait/regime-quality updates, and binary hypothesis staleness all
pre-existed and are reused as-is (delegated to, never reimplemented).

**The actual gap, confirmed by full-codebase audit before writing code:**
`SymbolPersonality.regime_outcome_quality` is an EMA that only moves when a
regime is *observed again* — nothing pulled a regime's learned quality back
toward neutral while the market moved on to a different regime, so
knowledge from an obsolete regime kept its full original influence forever.
Similarly, hypotheses only expired in a binary way (`is_stale`); nothing
gradually faded confidence just because the regime a hypothesis was formed
under became stale.

**Why two small schema additions were necessary (not violations of "reuse,
don't duplicate"):** neither gap could be implemented without new
timestamp/context data that genuinely didn't exist elsewhere:
- `SymbolPersonality.regime_last_seen: dict[str, float]` — tracks when each
  regime was last the *active* one, separate from `regime_outcome_quality`
  (a value, not a timestamp). `regime_decay()` reads/maintains it.
- `Hypothesis.formed_regime: str` — captures the market regime in effect at
  hypothesis creation (from `MarketNarrative.market_regime`), set in
  `HypothesisTracker.form()`/`HypothesisEngine.create()`. `context_decay()`
  uses it to detect when a hypothesis's originating context no longer
  matches the present one.

**How to apply:** `ForgettingEngine.forget(state, now)` is a consult/apply
seam like `SetupQueue`/`DebateEngine` — NOT auto-wired into
`QDEBrain.on_market_update` (which still only runs `prune_stale`/`age`/
`consolidate`). A caller must invoke it explicitly to get regime decay and
hypothesis context decay. `relevant_long_term_memories()` is read-only
ranking (recency + regime-tag context match) over `LongTermMemoryEntry` —
it never mutates or prunes entries, since long-term memory is documented as
intentionally non-decaying durable storage; only *how it's consulted* is
recency/relevance-weighted, not the storage itself.

Ordering gotcha in tests: `HypothesisTracker.age()` runs before context
decay inside `forget()`, so a hypothesis whose `max_age_seconds` staleness
window has also elapsed gets EXPIRED first and never reaches context decay
— test elapsed times for context decay specifically must stay under
`max_age_seconds` (86400s) to isolate the two mechanisms.
