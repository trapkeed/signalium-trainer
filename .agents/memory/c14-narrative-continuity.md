---
name: C14 Market Narrative Continuity
description: Multi-horizon narrative tracking (MICRO/SESSION/SHORT_TERM/MACRO), story lineage, cross-session continuity. Package at trainer/qde/brain/narrative/continuity/.
---

## Package location

`trainer/qde/brain/narrative/continuity/` — additive sub-package inside the existing narrative package.

- `continuity_types.py` — all enums, frozen dataclasses, constants
- `continuity_engine.py` — three public callables + ContinuityEngine
- `__init__.py` — re-exports all public symbols

## Three public callables (C14 interface)

```python
from trainer.qde.brain.narrative.continuity import (
    update_narrative,
    retrieve_story,
    evaluate_story_strength,
    ContinuityEngine,
)
```

**`update_narrative(instrument, kind, horizon, *, evidence, boost, is_supporting, session_label, narrative_engine, _state, now) → StoryNode`**

Targeted write: "at this horizon, I have new evidence for this kind." Lighter than `NarrativeAPI.generate_narrative()` (no memory recall, calibration, or tick-level processing). Records in lineage. Session transition detected when `session_label` changes — creates a `SessionSnapshot` and increments `story.session_count`.

**`retrieve_story(instrument, kind=None, horizon=None, *, narrative_engine, _state) → NarrativeStory | list | None`**

Returns the multi-horizon story. `kind=None` returns all stories for the instrument as a list. `horizon` further filters to stories with a node at that horizon. When `narrative_engine` is injected, StoryNodes are refreshed from live `SymbolNarrative` state.

**`evaluate_story_strength(instrument, kind=None, *, narrative_engine, _state, now) → StoryStrength`**

Pure computation. Formula:
```
overall = 0.40 × cross_horizon_alignment
        + 0.30 × mean(horizon_scores)
        + 0.20 × session_continuity_score
        + 0.10 × longevity_bonus
```

## Four time horizons

| Horizon | Range | Examples |
|---------|-------|---------|
| MICRO | 2–6 hours | LIQUIDITY_GRAB, EXHAUSTION |
| SESSION | 1–2 days | BREAKOUT, REVERSAL |
| SHORT_TERM | 1–2 weeks | TREND_CONTINUATION, ACCUMULATION |
| MACRO | 1–3 months | ACCUMULATION, DISTRIBUTION |

No new NarrativeKind values needed — all 7 existing kinds cover the spec examples.

## Key design decisions

**`is_multi_session = session_count >= 1`** (not `> 1`): the first transition means 2 sessions have been seen, which is already multi-session.

**`_cross_horizon_alignment` returns 0.0 for empty nodes** (not 1.0): an empty story has no alignment — this prevents the formula from contributing `0.40 × 1.0` when all nodes are expired.

**Event type priority in `_detect_event_type`**: WEAKENED → CONFIRMED (cross-horizon, regardless of whether this horizon is new) → FORMED → REINFORCED. Cross-horizon confirmation takes priority over FORMED so that updating a new horizon for an already-active kind logs CONFIRMED.

**Incompatible pairs** (from `_INCOMPATIBLE_PAIRS`, same knowledge as `NarrativeConflictDetector`):
- ACCUMULATION ↔ DISTRIBUTION
- TREND_CONTINUATION ↔ REVERSAL
- TREND_CONTINUATION ↔ EXHAUSTION
- BREAKOUT ↔ EXHAUSTION

## Integration with Phase C3

`ContinuityEngine` optionally accepts a `MarketNarrativeEngine`. When injected, StoryNodes are refreshed with live `SymbolNarrative` confidence/maturity via `_refresh_story_from_engine()`. When None, the continuity engine tracks its own lightweight state (mirrors `SymbolNarrative._recompute_maturity` in `_infer_maturity()`).

The continuity package does NOT call `generate_narrative()` or any C3 method — purely additive.

## Session continuity scoring

`session_continuity_score` reads `NarrativeLineage.session_snapshots`: counts snapshots where the primary_kind was active at SESSION or MACRO horizon. Requires ≥2 snapshots to produce a non-zero score (< 2 → 0.0).

`longevity_bonus` scales from 0 at session_count=1 to 1.0 at session_count≥10 (formula: `min(1.0, (session_count - 1) / 9.0)`).

## Narrative package export

C14 symbols exported from `trainer.qde.brain.narrative` package `__init__.py` alongside C3 symbols.
