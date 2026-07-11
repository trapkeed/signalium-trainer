---
name: C12 Thesis Aging design
description: Exponential confidence decay with per-OpportunityClass half-lives, structured event log, aging history, and three public callables. Lives in trainer/qde/brain/thesis/ without touching brain_models.Hypothesis.
---

## Package location

`trainer/qde/brain/thesis/` (new package; `thesis_manager.py` at the flat level is unchanged)

- `thesis_types.py` — all data classes
- `thesis_aging.py` — three public callables + `ThesisAgingEngine`
- `__init__.py` — re-exports everything

## Three public callables (C12 interface)

```python
from trainer.qde.brain.thesis import age_thesis, expire_thesis, update_confidence, ThesisAgingEngine
```

- `age_thesis(hyp, state, now=None) → ThesisAgingResult` — apply exponential decay; auto-expires when confidence < threshold or max_age exceeded
- `expire_thesis(hyp, state, reason="", now=None)` — force expiry; idempotent on already-terminal hypotheses
- `update_confidence(hyp, state, delta, event_type=CONFIRMATION, evidence="", now=None)` — record structured event; CONFIRMATION resets decay clock; auto-expires when confidence < threshold after INVALIDATION

## Design: wraps Hypothesis, does not extend it

`ThesisAgingState` is a per-hypothesis companion dataclass holding what `brain_models.Hypothesis` doesn't have:
- `confirmation_events: list[ThesisEvent]` (bounded, MAX_EVENTS=50)
- `invalidation_events: list[ThesisEvent]` (bounded)
- `aging_history: list[AgingRecord]` (bounded, MAX_HISTORY=200)
- `last_confirmation_at` — decay clock origin; reset by each CONFIRMATION
- `last_decayed_at` — throttle guard (min_decay_interval_seconds)
- `decay_profile: ThesisDecayProfile` — immutable, set at registration

`Hypothesis.confidence`, `hyp.status`, `hyp.last_updated_at` are mutated directly through the existing public interface.

## Exponential decay formula

```python
λ = ln(2) / half_life_seconds
decay_factor(elapsed) = exp(-λ × elapsed)
new_confidence = old_confidence × decay_factor(elapsed_since_last_confirmation)
```

## Per-class decay profiles (DEFAULT_DECAY_PROFILES)

| Class         | Half-life | Max age   |
|---------------|-----------|-----------|
| MICRO_SCALP   | 10 min    | 15 min    |
| SCALP         | 30 min    | 60 min    |
| INTRADAY      | 4 hours   | 8 hours   |
| MULTI_SESSION | 2 days    | 3 days    |
| SWING         | 2 weeks   | 3 weeks   |

Default expiry_threshold = 0.08 (all classes); min_decay_interval = 10s.

## Expiry conditions (any one triggers)

1. `new_confidence < profile.expiry_threshold` after decay
2. `now - created_at >= profile.max_age_seconds`
3. `expire_thesis()` called explicitly
4. `update_confidence()` INVALIDATION drives confidence below threshold

## ThesisAgingEngine

Stateful manager holding `dict[hypothesis_id, ThesisAgingState]`. Mirrors the per-symbol BrainState pattern. Key methods: `register(hyp, opp_class_value)`, `age_all(hypotheses, opp_class_fn=None)`, `drop(hypothesis_id)`, `state_for(hypothesis_id)`.

`age_all` auto-registers unregistered hypotheses (falls back to INTRADAY) so callers don't need to pre-register everything.

## Behavioral guarantees tested

- Repeated confirmations keep confidence high (decay clock resets)
- MICRO_SCALP decays faster than SWING for same elapsed time
- No confirmations → MICRO_SCALP expires within its max_age window
- `HypothesisTracker.age()` (binary staleness) coexists safely — both can run independently

## Note on existing HypothesisTracker.age()

`HypothesisTracker.age()` uses `is_stale()` = `now - last_updated_at > max_age_seconds`. C12 is complementary: it uses `now - last_confirmation_at` (not `last_updated_at`) as the decay clock origin, so a weakened/decayed hypothesis whose `last_updated_at` is recent but has no confirmation still decays under C12.
