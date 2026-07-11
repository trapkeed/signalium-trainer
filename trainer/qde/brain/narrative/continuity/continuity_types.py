"""C14 Market Narrative Continuity — type definitions.

Four time horizons organize the narrative hierarchy:

    MICRO      — 2–6 hours   (intra-session micro-structure)
    SESSION    — 1–2 days    (single to adjacent sessions)
    SHORT_TERM — 1–2 weeks   (swing-trade context)
    MACRO      — 1–3 months  (positional / trend backdrop)

The key insight: the QDE should read market behavior at *all four horizons
simultaneously*, and check that the current micro event is coherent with
the larger SESSION and MACRO narratives — the "narrative stack" idea.

Examples of narrative kinds mapped to each horizon:

    MICRO      — LIQUIDITY_GRAB (stop-hunt wick), EXHAUSTION (local fade)
    SESSION    — BREAKOUT, REVERSAL (intraday turning points)
    SHORT_TERM — TREND_CONTINUATION, ACCUMULATION (swing build-up)
    MACRO      — ACCUMULATION, DISTRIBUTION, TREND_CONTINUATION (positional)

Data model
----------
    StoryNode         — frozen snapshot of one horizon's narrative state.
    LineageEntry      — one auditable event in narrative history.
    LineageEventType  — FORMED / REINFORCED / WEAKENED / CONFIRMED / EXPIRED /
                        SESSION_NEW.
    SessionSnapshot   — full four-horizon state at a session boundary.
    NarrativeLineage  — ordered history of LineageEntry + SessionSnapshot.
    NarrativeStory    — the aggregated multi-horizon narrative object.
    StoryStrength     — evaluation result from evaluate_story_strength().
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from trainer.qde.brain.narrative.narrative_store import NarrativeKind, NarrativeMaturity


# ── Time Horizons ─────────────────────────────────────────────────────────────


class NarrativeHorizon(str, Enum):
    """Time-scale at which a market narrative operates."""

    MICRO      = "MICRO"       # 2–6 hours   (intra-session structure)
    SESSION    = "SESSION"     # 1–2 days    (single / adjacent session)
    SHORT_TERM = "SHORT_TERM"  # 1–2 weeks   (swing perspective)
    MACRO      = "MACRO"       # 1–3 months  (positional / trend backdrop)


# Canonical expected age range per horizon (seconds)
HORIZON_WINDOWS: dict[NarrativeHorizon, tuple[float, float]] = {
    NarrativeHorizon.MICRO:      (2  * 3_600,  6  * 3_600),    # 2–6 h
    NarrativeHorizon.SESSION:    (1  * 86_400, 2  * 86_400),   # 1–2 d
    NarrativeHorizon.SHORT_TERM: (7  * 86_400, 14 * 86_400),   # 1–2 wk
    NarrativeHorizon.MACRO:      (30 * 86_400, 90 * 86_400),   # 1–3 mo
}

# Canonical iteration order — most granular first
HORIZON_ORDER: list[NarrativeHorizon] = [
    NarrativeHorizon.MICRO,
    NarrativeHorizon.SESSION,
    NarrativeHorizon.SHORT_TERM,
    NarrativeHorizon.MACRO,
]

# NarrativeKind pairs that are structurally contradictory.
# Reuses the same semantic knowledge as NarrativeConflictDetector (avoiding
# re-importing the detector's private tables — we only need a superset test
# for cross-horizon alignment; the per-tick confidence-weighted logic lives
# in the conflict detector).
_INCOMPATIBLE_PAIRS: frozenset[frozenset] = frozenset({
    frozenset({NarrativeKind.ACCUMULATION,       NarrativeKind.DISTRIBUTION}),
    frozenset({NarrativeKind.TREND_CONTINUATION, NarrativeKind.REVERSAL}),
    frozenset({NarrativeKind.TREND_CONTINUATION, NarrativeKind.EXHAUSTION}),
    frozenset({NarrativeKind.BREAKOUT,           NarrativeKind.EXHAUSTION}),
})

MAX_LINEAGE_ENTRIES  = 200  # rolling cap per (instrument, kind) story
MAX_SESSION_SNAPSHOTS = 90  # rolling cap — roughly 3 months of daily sessions


# ── Story Node ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoryNode:
    """Immutable snapshot of one horizon's narrative state.

    A ``StoryNode`` is produced by each ``update_narrative()`` call and
    replaces the previous node in ``NarrativeStory.nodes[horizon]``.
    Immutability means the lineage is never accidentally mutated in place.
    """

    horizon:           NarrativeHorizon
    kind:              NarrativeKind
    narrative_id:      str
    confidence:        float          # 0-1
    maturity:          str            # NarrativeMaturity.value
    supporting_count:  int
    conflicting_count: int
    evidence_balance:  float          # supporting / total (0.5 when no evidence)
    formed_at:         float
    last_updated_at:   float
    summary:           str            # human-readable snapshot for logs/dashboards

    # ── Computed helpers ──────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self.maturity != NarrativeMaturity.EXPIRED.value

    def is_established(self) -> bool:
        return self.maturity == NarrativeMaturity.ESTABLISHED.value

    def is_fading(self) -> bool:
        return self.maturity == NarrativeMaturity.FADING.value

    def update_count(self) -> int:
        return self.supporting_count + self.conflicting_count

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "horizon":           self.horizon.value,
            "kind":              self.kind.value,
            "narrative_id":      self.narrative_id,
            "confidence":        round(self.confidence, 4),
            "maturity":          self.maturity,
            "supporting_count":  self.supporting_count,
            "conflicting_count": self.conflicting_count,
            "evidence_balance":  round(self.evidence_balance, 4),
            "formed_at":         round(self.formed_at, 3),
            "last_updated_at":   round(self.last_updated_at, 3),
            "summary":           self.summary,
        }


# ── Lineage Event ─────────────────────────────────────────────────────────────


class LineageEventType(str, Enum):
    """Types of events that can appear in NarrativeLineage."""

    FORMED      = "FORMED"       # narrative first detected at this horizon
    REINFORCED  = "REINFORCED"   # new supporting evidence added
    WEAKENED    = "WEAKENED"     # conflicting evidence registered
    CONFIRMED   = "CONFIRMED"    # same kind active at ≥2 horizons simultaneously
    EXPIRED     = "EXPIRED"      # narrative aged out or explicitly closed
    SESSION_NEW = "SESSION_NEW"  # new trading session started (lineage marker)


@dataclass(frozen=True)
class LineageEntry:
    """One auditable event in the narrative history of a (instrument, kind) story.

    Entries are stored most-recent-first in ``NarrativeLineage.entries``
    and capped at ``MAX_LINEAGE_ENTRIES``.
    """

    entry_id:          str               # short UUID for external reference
    horizon:           NarrativeHorizon
    kind:              NarrativeKind
    event_type:        LineageEventType
    evidence:          str               # free-text evidence note
    confidence_before: float
    confidence_after:  float
    session_label:     str               # e.g. "2024-01-15_LONDON"
    occurred_at:       float

    def to_dict(self) -> dict:
        return {
            "entry_id":          self.entry_id,
            "horizon":           self.horizon.value,
            "kind":              self.kind.value,
            "event_type":        self.event_type.value,
            "evidence":          self.evidence,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after":  round(self.confidence_after, 4),
            "session_label":     self.session_label,
            "occurred_at":       round(self.occurred_at, 3),
        }


# ── Session Snapshot ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionSnapshot:
    """Four-horizon narrative state frozen at a session boundary.

    Stored in ``NarrativeLineage.session_snapshots`` to let the system
    connect today's market behavior with previous sessions:

        continuity_engine.evaluate_story_strength(instrument)
        → session_continuity_score derived from these snapshots.

    A session boundary is detected whenever ``session_label`` changes
    between ``update_narrative()`` calls.
    """

    session_label:      str                      # e.g. "2024-01-15_LONDON"
    recorded_at:        float
    nodes:              dict[str, StoryNode]     # horizon.value → StoryNode
    dominant_kind:      str | None               # NarrativeKind.value or None
    overall_confidence: float                    # mean confidence across active nodes

    def to_dict(self) -> dict:
        return {
            "session_label":      self.session_label,
            "recorded_at":        round(self.recorded_at, 3),
            "nodes":              {h: n.to_dict() for h, n in self.nodes.items()},
            "dominant_kind":      self.dominant_kind,
            "overall_confidence": round(self.overall_confidence, 4),
        }


# ── Narrative Lineage ─────────────────────────────────────────────────────────


@dataclass
class NarrativeLineage:
    """Complete auditable history for one (instrument, kind) story.

    ``entries``           — most-recent-first, capped at MAX_LINEAGE_ENTRIES.
    ``session_snapshots`` — appended chronologically, capped at MAX_SESSION_SNAPSHOTS.
    """

    instrument:        str
    kind:              NarrativeKind
    entries:           list[LineageEntry]    = field(default_factory=list)
    session_snapshots: list[SessionSnapshot] = field(default_factory=list)

    # ── Mutation ──────────────────────────────────────────────────────────

    def append_entry(self, entry: LineageEntry) -> None:
        self.entries.insert(0, entry)
        if len(self.entries) > MAX_LINEAGE_ENTRIES:
            self.entries = self.entries[:MAX_LINEAGE_ENTRIES]

    def append_snapshot(self, snapshot: SessionSnapshot) -> None:
        self.session_snapshots.append(snapshot)
        if len(self.session_snapshots) > MAX_SESSION_SNAPSHOTS:
            self.session_snapshots = self.session_snapshots[-MAX_SESSION_SNAPSHOTS:]

    # ── Queries ───────────────────────────────────────────────────────────

    def last_session_label(self) -> str | None:
        """Session label of the most recent lineage entry, or None."""
        for e in self.entries:
            if e.session_label:
                return e.session_label
        return None

    def distinct_session_count(self) -> int:
        """Number of distinct session labels seen across all entries."""
        seen: set[str] = set()
        for e in self.entries:
            if e.session_label:
                seen.add(e.session_label)
        return len(seen)

    def entries_for_horizon(self, horizon: NarrativeHorizon) -> list[LineageEntry]:
        return [e for e in self.entries if e.horizon == horizon]

    def last_event_of_type(self, event_type: LineageEventType) -> LineageEntry | None:
        for e in self.entries:
            if e.event_type == event_type:
                return e
        return None

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "instrument":         self.instrument,
            "kind":               self.kind.value,
            "entry_count":        len(self.entries),
            "session_count":      self.distinct_session_count(),
            "entries":            [e.to_dict() for e in self.entries[:20]],
            "session_snapshots":  [s.to_dict() for s in self.session_snapshots[-10:]],
        }


# ── Narrative Story ───────────────────────────────────────────────────────────


@dataclass
class NarrativeStory:
    """Multi-horizon narrative for one (instrument, primary_kind) pair.

    ``nodes`` holds the current ``StoryNode`` for each horizon that has
    received at least one update.  Horizons not yet seen are absent from
    the dict (never None — callers should use ``.get()``).

    ``lineage`` holds the full auditable history.

    The story accumulates evidence across sessions: ``session_count`` grows
    each time a new session label is detected, and ``is_multi_session``
    becomes True after the second session.  This is what enables the QDE to
    connect today's behaviour with previous sessions.
    """

    story_id:         str
    instrument:       str
    primary_kind:     NarrativeKind
    nodes:            dict[str, StoryNode]  = field(default_factory=dict)
    lineage:          NarrativeLineage      = field(
        default_factory=lambda: NarrativeLineage(
            instrument="", kind=NarrativeKind.TREND_CONTINUATION
        )
    )
    formed_at:        float                 = field(default_factory=time.time)
    last_updated_at:  float                 = field(default_factory=time.time)
    session_count:    int                   = 0
    is_multi_session: bool                  = False

    # ── Helpers ───────────────────────────────────────────────────────────

    def active_nodes(self) -> list[StoryNode]:
        """Nodes whose maturity is not EXPIRED."""
        return [n for n in self.nodes.values() if n.is_active()]

    def established_nodes(self) -> list[StoryNode]:
        return [n for n in self.nodes.values() if n.is_established()]

    def horizon_count(self) -> int:
        """Number of active horizons."""
        return len(self.active_nodes())

    def mean_confidence(self) -> float:
        active = self.active_nodes()
        if not active:
            return 0.0
        return sum(n.confidence for n in active) / len(active)

    def age_seconds(self, now: float | None = None) -> float:
        t = now if now is not None else time.time()
        return max(0.0, t - self.formed_at)

    def node_at(self, horizon: NarrativeHorizon) -> StoryNode | None:
        return self.nodes.get(horizon.value)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "story_id":         self.story_id,
            "instrument":       self.instrument,
            "primary_kind":     self.primary_kind.value,
            "nodes":            {h: n.to_dict() for h, n in self.nodes.items()},
            "active_horizons":  [n.horizon.value for n in self.active_nodes()],
            "formed_at":        round(self.formed_at, 3),
            "last_updated_at":  round(self.last_updated_at, 3),
            "session_count":    self.session_count,
            "is_multi_session": self.is_multi_session,
            "mean_confidence":  round(self.mean_confidence(), 4),
            "horizon_count":    self.horizon_count(),
        }


# ── Story Strength ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoryStrength:
    """Output of ``evaluate_story_strength()``.

    ``overall_score`` is the headline signal (0-1).  All component scores
    are also 0-1 and individually interpretable.

    Formula
    -------
    overall_score = (
        0.40 × cross_horizon_alignment    # do all horizons agree?
      + 0.30 × mean(horizon_scores)       # quality of individual horizons
      + 0.20 × session_continuity_score   # does story persist across sessions?
      + 0.10 × longevity_bonus            # bonus for multi-session persistence
    )
    """

    instrument:               str
    kind:                     NarrativeKind | None  # None → composite across all kinds
    overall_score:            float                  # headline 0-1
    horizon_scores:           dict[str, float]       # horizon.value → 0-1
    cross_horizon_alignment:  float                  # 0-1
    session_continuity_score: float                  # 0-1
    longevity_bonus:          float                  # 0-1
    active_horizon_count:     int
    dominant_kind:            NarrativeKind | None
    is_multi_session:         bool
    reasoning:                str                    # compact log string

    def to_dict(self) -> dict:
        return {
            "instrument":               self.instrument,
            "kind":                     self.kind.value if self.kind else None,
            "overall_score":            round(self.overall_score, 4),
            "horizon_scores":           {h: round(v, 4) for h, v in self.horizon_scores.items()},
            "cross_horizon_alignment":  round(self.cross_horizon_alignment, 4),
            "session_continuity_score": round(self.session_continuity_score, 4),
            "longevity_bonus":          round(self.longevity_bonus, 4),
            "active_horizon_count":     self.active_horizon_count,
            "dominant_kind":            self.dominant_kind.value if self.dominant_kind else None,
            "is_multi_session":         self.is_multi_session,
            "reasoning":                self.reasoning,
        }


# ── Public API list ───────────────────────────────────────────────────────────

__all__ = [
    # Horizon
    "NarrativeHorizon",
    "HORIZON_WINDOWS",
    "HORIZON_ORDER",
    "_INCOMPATIBLE_PAIRS",
    # Limits
    "MAX_LINEAGE_ENTRIES",
    "MAX_SESSION_SNAPSHOTS",
    # Story node
    "StoryNode",
    # Lineage
    "LineageEventType",
    "LineageEntry",
    "NarrativeLineage",
    "SessionSnapshot",
    # Story
    "NarrativeStory",
    # Evaluation
    "StoryStrength",
]
