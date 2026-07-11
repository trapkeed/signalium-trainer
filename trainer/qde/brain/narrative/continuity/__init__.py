"""C14 Market Narrative Continuity — subpackage.

Adds multi-horizon narrative tracking to the QDE narrative layer.

Time horizons
-------------
    MICRO      — 2–6 hours   (intra-session micro-structure)
    SESSION    — 1–2 days    (single to adjacent sessions)
    SHORT_TERM — 1–2 weeks   (swing-trade context)
    MACRO      — 1–3 months  (positional / trend backdrop)

Public callables (spec)
-----------------------
    update_narrative(instrument, kind, horizon, ...)  → StoryNode
    retrieve_story(instrument, kind, horizon, ...)    → NarrativeStory | list | None
    evaluate_story_strength(instrument, kind, ...)    → StoryStrength

Stateful wrapper
----------------
    ContinuityEngine — holds shared state; delegates to the three callables.

Narrative kinds used
--------------------
All seven ``NarrativeKind`` values from the existing narrative package are
valid at any horizon.  Examples from the spec:

    ACCUMULATION      — quiet range build-up (SESSION / MACRO)
    DISTRIBUTION      — supply emerging, breakdown likely (SESSION / MACRO)
    TREND_CONTINUATION — existing trend sustaining (all horizons)
    LIQUIDITY_GRAB    — stop-hunt / wick (MICRO)
    EXHAUSTION        — momentum fading (MICRO / SESSION)

Integration with Phase C3
-------------------------
    ``ContinuityEngine`` optionally accepts a live ``MarketNarrativeEngine``
    (from Phase C3).  When injected, ``retrieve_story()`` refreshes StoryNodes
    with the latest ``SymbolNarrative`` confidence and maturity before returning.
    When not injected, the continuity engine tracks its own lightweight state.

    This package does NOT call ``generate_narrative()`` or any other Phase C3
    method — it is additive and does not duplicate any existing logic.
"""

from trainer.qde.brain.narrative.continuity.continuity_engine import (
    ContinuityEngine,
    evaluate_story_strength,
    retrieve_story,
    update_narrative,
)
from trainer.qde.brain.narrative.continuity.continuity_types import (
    HORIZON_ORDER,
    HORIZON_WINDOWS,
    MAX_LINEAGE_ENTRIES,
    MAX_SESSION_SNAPSHOTS,
    LineageEntry,
    LineageEventType,
    NarrativeHorizon,
    NarrativeLineage,
    NarrativeStory,
    SessionSnapshot,
    StoryNode,
    StoryStrength,
    _INCOMPATIBLE_PAIRS,
)

__all__ = [
    # ── Horizon ──────────────────────────────────────────────────────────
    "NarrativeHorizon",
    "HORIZON_WINDOWS",
    "HORIZON_ORDER",
    "_INCOMPATIBLE_PAIRS",
    # ── Limits ───────────────────────────────────────────────────────────
    "MAX_LINEAGE_ENTRIES",
    "MAX_SESSION_SNAPSHOTS",
    # ── Data types ───────────────────────────────────────────────────────
    "StoryNode",
    "LineageEventType",
    "LineageEntry",
    "NarrativeLineage",
    "SessionSnapshot",
    "NarrativeStory",
    "StoryStrength",
    # ── Public callables (spec) ───────────────────────────────────────────
    "update_narrative",
    "retrieve_story",
    "evaluate_story_strength",
    # ── Stateful wrapper ──────────────────────────────────────────────────
    "ContinuityEngine",
]
