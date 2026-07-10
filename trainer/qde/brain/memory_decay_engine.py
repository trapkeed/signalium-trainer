"""QDE Brain — generic recency/relevance/context/regime decay math.

This module deliberately does **not** reimplement the decay mechanisms that
already exist elsewhere in the Brain:

  - Working memory salience decay (`WorkingMemoryItem.current_strength`,
    `WorkingMemory.prune_stale`) already makes individual working-memory
    items fade and get pruned — untouched.
  - Attention score decay (`attention_allocator.AttentionScoreAllocator`)
    already cools off per-symbol attention composites — untouched.
  - Hypothesis staleness (`HypothesisTracker.age`, `Hypothesis.is_stale`)
    already expires hypotheses that haven't been touched in
    `max_age_seconds` — untouched.
  - Behavioral/personality trait updates (`behavioral_memory._ema`,
    `SymbolPersonality.update_trait`) already blend new observations via an
    exponential moving average — untouched.

What's missing, and what this module adds, are three things nothing above
covers:

  1. A **generic recency-weighting primitive** usable for anything with a
     timestamp but no decay of its own — most notably
     `LongTermMemoryEntry`, which is explicitly documented as "never
     decays" (durable knowledge), yet when *consulted* for present-moment
     reasoning, an entry from a year ago should count for less than one
     from an hour ago even though it's still worth keeping.
  2. **Context decay** — hypotheses currently only expire in a binary,
     all-or-nothing way (`is_stale`); nothing gradually fades a
     hypothesis's influence just because the regime it was formed under
     has since changed. `Hypothesis.formed_regime` (see `brain_models.py`)
     exists solely to support this.
  3. **Regime decay** — `SymbolPersonality.regime_outcome_quality` is an
     EMA that only ever moves when a regime is *observed again*; nothing
     today pulls a regime's learned quality back toward neutral while it
     goes unobserved, so behavior learned from a regime the market has
     since moved past keeps its full original influence forever. This is
     the literal "market behavior from obsolete regimes should gradually
     lose influence" requirement.

All functions here are pure (no I/O, no mutation of shared state beyond the
dict explicitly passed to `regime_decay`) so `forgetting_engine.py` — the
orchestration seam that actually applies them to a `BrainState` — stays a
thin, easily-testable caller.
"""

from __future__ import annotations

import math

# A memory item's recency weight halves every this many seconds by default.
# Distinct from `working_memory.DEFAULT_WORKING_MEMORY_DECAY_PER_SECOND`
# (per-second salience decay for *transient* items): this is a coarser
# half-life used to discount otherwise-non-decaying durable memories when
# they're consulted, not to prune them.
DEFAULT_RECENCY_HALF_LIFE_SECONDS = 6.0 * 3600.0  # 6 hours

# How much of a hypothesis's confidence survives, per elapsed second, once
# the current regime no longer matches the regime it was formed under.
DEFAULT_CONTEXT_MISMATCH_DECAY_PER_SECOND = 1.0 / 1800.0  # ~e-fold every 30 min

# How much of a regime's learned outcome quality survives, per elapsed
# second, once that regime is no longer the currently-observed one.
DEFAULT_REGIME_DECAY_PER_SECOND = 1.0 / (12.0 * 3600.0)  # ~e-fold every 12 hours

# Regime outcome quality decays *toward* this value, not toward zero —
# "neutral" for a -1..1 quality score is 0.0, i.e. "no longer known to be
# either good or bad for this symbol".
REGIME_NEUTRAL_QUALITY = 0.0


def recency_weight(
    age_seconds: float, half_life_seconds: float = DEFAULT_RECENCY_HALF_LIFE_SECONDS
) -> float:
    """Generic half-life recency weighting, in (0, 1].

    `age_seconds <= 0` returns 1.0 (maximally recent). Used to discount
    otherwise non-decaying memories (e.g. `LongTermMemoryEntry`) so recent
    experience outweighs old experience when both are still "known".
    """
    age_seconds = max(0.0, age_seconds)
    if half_life_seconds <= 0:
        return 1.0 if age_seconds == 0 else 0.0
    return math.pow(0.5, age_seconds / half_life_seconds)


def relevance_weight(
    base_relevance: float,
    age_seconds: float,
    context_match: bool,
    half_life_seconds: float = DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    mismatch_penalty: float = 0.5,
) -> float:
    """Composite relevance score blending recency and context match.

    `base_relevance` is whatever intrinsic score the caller already has
    (salience, confidence, access frequency, ...); this does not replace
    that — it discounts it by how recent the memory is and, if the memory
    carries a context tag (e.g. regime it was formed under), by whether
    that context still matches the present one. A mismatched context
    doesn't zero out the memory (it may still be informative), but it is
    weighted down by `mismatch_penalty`.
    """
    base_relevance = max(0.0, min(1.0, base_relevance))
    weight = base_relevance * recency_weight(age_seconds, half_life_seconds)
    if not context_match:
        weight *= max(0.0, min(1.0, mismatch_penalty))
    return weight


def context_decay(
    confidence: float,
    elapsed_since_context_changed_seconds: float,
    context_still_valid: bool,
    decay_per_second: float = DEFAULT_CONTEXT_MISMATCH_DECAY_PER_SECOND,
) -> float:
    """Gradually fade `confidence` once its originating context is stale.

    While `context_still_valid` is True, confidence is returned unchanged
    — this only kicks in once the context (e.g. a hypothesis's
    `formed_regime`) no longer matches the present one. It is intentionally
    gradual (exponential decay), not the binary all-or-nothing expiry that
    `Hypothesis.is_stale` already provides for pure time-since-touched
    staleness — the two are independent and complementary.
    """
    if context_still_valid:
        return confidence
    elapsed = max(0.0, elapsed_since_context_changed_seconds)
    retained = max(0.0, 1.0 - decay_per_second) ** elapsed
    return confidence * retained


def regime_decay(
    regime_outcome_quality: dict[str, float],
    regime_last_seen: dict[str, float],
    current_regime: str,
    now: float,
    decay_per_second: float = DEFAULT_REGIME_DECAY_PER_SECOND,
) -> dict[str, float]:
    """Pull each regime's learned outcome quality toward neutral, scaled by
    how long it's been since that regime was last the active one.

    `current_regime`'s own entry is left untouched (it's still in effect,
    so its learned quality stays fully influential) and its last-seen
    timestamp is refreshed to `now`. Every *other* regime with a
    `regime_last_seen` entry has its quality decayed toward
    `REGIME_NEUTRAL_QUALITY` proportional to elapsed time — this is what
    makes "market behavior from obsolete regimes gradually lose
    influence" actually happen, since `SymbolPersonality.update_regime_quality`
    on its own never revisits a regime once the market has moved on from it.

    Mutates and returns `regime_last_seen` (marking `current_regime` as
    freshly observed) as well as returning the decayed
    `regime_outcome_quality` dict. `regime_outcome_quality` is returned as a
    new dict; the caller is responsible for assigning it back onto
    `SymbolPersonality.regime_outcome_quality`.
    """
    decayed: dict[str, float] = {}
    for regime, quality in regime_outcome_quality.items():
        if regime == current_regime:
            decayed[regime] = quality
            continue
        last_seen = regime_last_seen.get(regime)
        if last_seen is None:
            # Never tracked before this point — start the clock now rather
            # than assuming it's already stale.
            decayed[regime] = quality
            continue
        elapsed = max(0.0, now - last_seen)
        retained = max(0.0, 1.0 - decay_per_second) ** elapsed
        decayed[regime] = quality * retained + REGIME_NEUTRAL_QUALITY * (1.0 - retained)

    if current_regime:
        regime_last_seen[current_regime] = now

    return decayed
