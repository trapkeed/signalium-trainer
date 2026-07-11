"""C14 Market Narrative Continuity — engine and public callables.

Three public callables (spec):

    update_narrative(instrument, kind, horizon, ...)  → StoryNode
    retrieve_story(instrument, kind, horizon, ...)    → NarrativeStory | list | None
    evaluate_story_strength(instrument, kind, ...)    → StoryStrength

``ContinuityEngine`` — stateful wrapper (one instance per process or symbol-set).

Design
------
The engine is *additive* on top of the existing MarketNarrativeEngine (C3):

  When a ``MarketNarrativeEngine`` is injected, ``update_narrative()`` reads
  the live ``SymbolNarrative`` state from it to populate StoryNodes with
  current confidence and maturity.  When None, the continuity engine tracks
  its own lightweight confidence/maturity state per StoryNode.

  ``evaluate_story_strength()`` is a pure computation; it reuses the
  ``_INCOMPATIBLE_PAIRS`` set from continuity_types (derived from the same
  semantic knowledge as ``NarrativeConflictDetector``) for cross-horizon
  alignment — no reimplementation of the conflict-detection logic.

Session continuity
------------------
A session transition is detected when ``session_label`` changes between
``update_narrative()`` calls for the same story.  On transition:
  1. A ``SessionSnapshot`` of all current nodes is appended to the lineage.
  2. ``story.session_count`` is incremented.
  3. ``story.is_multi_session`` is set once ``session_count > 1``.

This is how the QDE connects today's market behaviour with previous sessions:
the lineage stores what each horizon's narrative looked like at the boundary
of every prior session, so ``evaluate_story_strength()`` can measure whether
the same story has been active across multiple days or weeks.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.narrative.continuity.continuity_types import (
    _INCOMPATIBLE_PAIRS,
    LineageEntry,
    LineageEventType,
    NarrativeHorizon,
    NarrativeLineage,
    NarrativeStory,
    SessionSnapshot,
    StoryNode,
    StoryStrength,
)
from trainer.qde.brain.narrative.narrative_store import (
    NarrativeKind,
    NarrativeMaturity,
    SymbolNarrative,
)

if TYPE_CHECKING:
    from trainer.qde.brain.narrative.market_narrative_engine import MarketNarrativeEngine


# ── Internal helpers ──────────────────────────────────────────────────────────


def _node_from_symbol_narrative(
    horizon: NarrativeHorizon,
    sn: SymbolNarrative,
) -> StoryNode:
    """Build a StoryNode snapshot from a live SymbolNarrative."""
    total = len(sn.supporting_evidence) + len(sn.conflicting_evidence)
    balance = (len(sn.supporting_evidence) / total) if total > 0 else 0.5
    summary = (
        f"{sn.kind.value} @ {sn.maturity.value} "
        f"(conf={sn.confidence:.0%}, reinf={sn.reinforcement_count})"
    )
    return StoryNode(
        horizon=horizon,
        kind=sn.kind,
        narrative_id=sn.narrative_id,
        confidence=sn.confidence,
        maturity=sn.maturity.value,
        supporting_count=len(sn.supporting_evidence),
        conflicting_count=len(sn.conflicting_evidence),
        evidence_balance=balance,
        formed_at=sn.formed_at,
        last_updated_at=sn.last_updated_at,
        summary=summary,
    )


def _infer_maturity(
    confidence: float,
    evidence_balance: float,
    update_count: int,
) -> str:
    """Mirror of SymbolNarrative._recompute_maturity for standalone nodes.

    Kept in sync with the source to avoid maturity divergence when the
    continuity engine runs without a live MarketNarrativeEngine.
    """
    if evidence_balance < 0.35 and update_count >= 2:
        return NarrativeMaturity.FADING.value
    if confidence >= 0.70 and update_count >= 3 and evidence_balance >= 0.60:
        return NarrativeMaturity.ESTABLISHED.value
    if confidence >= 0.45 and update_count >= 1:
        return NarrativeMaturity.DEVELOPING.value
    return NarrativeMaturity.FORMING.value


def _build_node_standalone(
    *,
    horizon: NarrativeHorizon,
    kind: NarrativeKind,
    existing_node: StoryNode | None,
    boost: float,
    is_supporting: bool,
    now: float,
) -> StoryNode:
    """Build / update a StoryNode when no live engine is available."""
    if existing_node is None:
        raw_conf = 0.30 + (boost if is_supporting else -boost)
        confidence = max(0.0, min(1.0, raw_conf))
        supporting = 1 if is_supporting else 0
        conflicting = 0 if is_supporting else 1
        total = supporting + conflicting
        balance = (supporting / total) if total > 0 else 0.5
        narrative_id = str(uuid.uuid4())[:12]
        formed_at = now
    else:
        delta = boost if is_supporting else -boost
        confidence = max(0.0, min(1.0, existing_node.confidence + delta))
        supporting = existing_node.supporting_count + (1 if is_supporting else 0)
        conflicting = existing_node.conflicting_count + (0 if is_supporting else 1)
        total = supporting + conflicting
        balance = (supporting / total) if total > 0 else 0.5
        narrative_id = existing_node.narrative_id
        formed_at = existing_node.formed_at

    maturity = _infer_maturity(confidence, balance, supporting + conflicting)
    summary = f"{kind.value} @ {maturity} (conf={confidence:.0%})"

    return StoryNode(
        horizon=horizon,
        kind=kind,
        narrative_id=narrative_id,
        confidence=confidence,
        maturity=maturity,
        supporting_count=supporting,
        conflicting_count=conflicting,
        evidence_balance=balance,
        formed_at=formed_at,
        last_updated_at=now,
        summary=summary,
    )


def _make_lineage_entry(
    *,
    horizon: NarrativeHorizon,
    kind: NarrativeKind,
    event_type: LineageEventType,
    evidence: str,
    confidence_before: float,
    confidence_after: float,
    session_label: str,
    now: float,
) -> LineageEntry:
    return LineageEntry(
        entry_id=str(uuid.uuid4())[:8],
        horizon=horizon,
        kind=kind,
        event_type=event_type,
        evidence=evidence,
        confidence_before=round(confidence_before, 4),
        confidence_after=round(confidence_after, 4),
        session_label=session_label,
        occurred_at=now,
    )


def _make_story(instrument: str, kind: NarrativeKind, now: float) -> NarrativeStory:
    return NarrativeStory(
        story_id=str(uuid.uuid4())[:12],
        instrument=instrument,
        primary_kind=kind,
        nodes={},
        lineage=NarrativeLineage(instrument=instrument, kind=kind),
        formed_at=now,
        last_updated_at=now,
        session_count=0,
        is_multi_session=False,
    )


def _detect_event_type(
    existing_node: StoryNode | None,
    is_supporting: bool,
    story: NarrativeStory,
    horizon: NarrativeHorizon,
    kind: NarrativeKind,
) -> LineageEventType:
    """Classify the lineage event type for this update.

    Priority order:
    1. WEAKENED — conflicting evidence always wins the label.
    2. CONFIRMED — cross-horizon confirmation takes precedence over FORMED/REINFORCED:
       when the same kind is already active at another horizon, arriving at a
       *new* horizon is a multi-timeframe alignment event (stronger signal than
       simply forming or reinforcing at a single horizon).
    3. FORMED — first supporting evidence at this horizon.
    4. REINFORCED — repeat supporting evidence at an already-seen horizon.
    """
    if not is_supporting:
        return LineageEventType.WEAKENED
    # Cross-horizon confirmation: same kind already active at another horizon
    other_active = [
        n
        for h_val, n in story.nodes.items()
        if h_val != horizon.value and n.is_active() and n.kind == kind
    ]
    if other_active:
        return LineageEventType.CONFIRMED
    if existing_node is None:
        return LineageEventType.FORMED
    return LineageEventType.REINFORCED


def _session_snapshot(story: NarrativeStory, session_label: str, now: float) -> SessionSnapshot:
    """Snapshot current story node state at a session boundary."""
    snap_nodes = dict(story.nodes)
    active_confs = [n.confidence for n in story.active_nodes()]
    dominant_kind: str | None = None
    if story.active_nodes():
        dominant_kind = max(story.active_nodes(), key=lambda n: n.confidence).kind.value
    return SessionSnapshot(
        session_label=session_label,
        recorded_at=now,
        nodes=snap_nodes,
        dominant_kind=dominant_kind,
        overall_confidence=(sum(active_confs) / len(active_confs) if active_confs else 0.0),
    )


# ── Scoring helpers ───────────────────────────────────────────────────────────


def _horizon_score(node: StoryNode) -> float:
    """Per-horizon quality score 0-1."""
    if not node.is_active():
        return 0.0
    score = node.confidence
    if node.is_established():
        score = min(1.0, score * 1.15)
    if node.evidence_balance < 0.40:
        score *= 0.70
    return round(min(1.0, score), 4)


def _cross_horizon_alignment(nodes: list[StoryNode]) -> float:
    """Measure agreement across active horizons (0-1).

    Penalises each incompatible pair of narrative kinds.
    1.0 = no contradictions; 0.0 = every pair contradicts every other.
    Returns 0.0 when there are no active nodes (no story = no alignment).
    """
    active = [n for n in nodes if n.is_active()]
    if len(active) == 0:
        return 0.0
    if len(active) == 1:
        return 1.0

    total_pairs = 0
    incompatible_pairs = 0
    for i, na in enumerate(active):
        for nb in active[i + 1 :]:
            total_pairs += 1
            if frozenset({na.kind, nb.kind}) in _INCOMPATIBLE_PAIRS:
                incompatible_pairs += 1

    return round(1.0 - (incompatible_pairs / total_pairs), 4)


def _session_continuity_score(story: NarrativeStory) -> float:
    """How consistently this narrative's primary_kind persisted across sessions.

    Reads from ``NarrativeLineage.session_snapshots``.  Each snapshot that
    carried an active SESSION or MACRO node for story.primary_kind earns one
    continuity point.  Score = continuity_points / snapshot_count.

    Returns 0.0 when fewer than 2 snapshots exist (no cross-session evidence).
    """
    snapshots = story.lineage.session_snapshots
    if len(snapshots) < 2:
        return 0.0

    kind_val = story.primary_kind.value
    present_count = 0
    for snap in snapshots:
        # Prefer SESSION then MACRO horizons for continuity anchoring
        node = snap.nodes.get(NarrativeHorizon.SESSION.value) or snap.nodes.get(
            NarrativeHorizon.MACRO.value
        )
        if node is not None and node.kind.value == kind_val and node.is_active():
            present_count += 1

    return round(present_count / len(snapshots), 4)


def _longevity_bonus(story: NarrativeStory) -> float:
    """Bonus (0-1) for multi-session persistence.

    Scales linearly from 0 at 1 session to 1.0 at 10+ sessions.
    """
    if story.session_count < 2:
        return 0.0
    return round(min(1.0, (story.session_count - 1) / 9.0), 4)


def _refresh_story_from_engine(
    story: NarrativeStory,
    engine: "MarketNarrativeEngine",
) -> None:
    """Re-read SymbolNarrative states from the live engine into StoryNodes.

    Called by retrieve_story() when a narrative_engine is injected.
    Errors are swallowed so callers always get the best available state.
    """
    for horizon_val in list(story.nodes):
        try:
            horizon = NarrativeHorizon(horizon_val)
            sn = engine.narrative_of_kind(story.instrument, story.primary_kind)
            if sn is not None and sn.is_active():
                story.nodes[horizon_val] = _node_from_symbol_narrative(horizon, sn)
        except Exception:
            pass  # leave existing node unchanged on error


# ── Public callable: update_narrative ────────────────────────────────────────


def update_narrative(
    instrument: str,
    kind: NarrativeKind,
    horizon: NarrativeHorizon,
    *,
    evidence: str = "",
    boost: float = 0.12,
    is_supporting: bool = True,
    session_label: str = "",
    narrative_engine: "MarketNarrativeEngine | None" = None,
    _state: dict | None = None,
    now: float | None = None,
) -> StoryNode:
    """Record a narrative update at a specific time horizon.

    This is the primary write entry-point for C14.  It is intentionally
    lighter than ``NarrativeAPI.generate_narrative`` (which does a full tick
    update with memory recall, calibration, and probability recompute):
    ``update_narrative`` makes a targeted "at this horizon, I have new
    evidence for this kind" statement and records it in the lineage.

    Parameters
    ----------
    instrument:
        Symbol being updated (e.g. "EURUSD", "GOLD").
    kind:
        Which narrative kind is being asserted at this horizon.
    horizon:
        Time-scale of this update (MICRO / SESSION / SHORT_TERM / MACRO).
    evidence:
        Human-readable evidence note (e.g. "RSI divergence at HTF supply").
    boost:
        Magnitude of the confidence change.  Applied positively when
        ``is_supporting=True``, negatively when False.
    is_supporting:
        True → reinforcing evidence; False → conflicting / weakening.
    session_label:
        Current session identifier (e.g. "2024-01-15_LONDON").  Changes
        to this string trigger session-transition handling: the current node
        state is snapshotted and ``story.session_count`` is incremented.
    narrative_engine:
        Optional live ``MarketNarrativeEngine``.  When supplied, the
        ``SymbolNarrative`` state is read from it to populate the StoryNode.
        When None, the continuity engine manages its own lightweight state.
    _state:
        Mutable state dict shared with the other callables.  Pass the same
        dict to all three functions for consistent state.  ``ContinuityEngine``
        handles this automatically.
    now:
        Override current time (for replay / testing).

    Returns
    -------
    StoryNode — the freshly-built node for this (horizon, kind) update.
    """
    now = now if now is not None else time.time()
    state = _state if _state is not None else {}

    # ── Retrieve or create the story ──────────────────────────────────────
    story_key = f"{instrument}:{kind.value}"
    story: NarrativeStory = state.get(story_key) or _make_story(instrument, kind, now)

    # ── Session-transition detection ──────────────────────────────────────
    last_session = story.lineage.last_session_label()
    is_new_session = (
        bool(session_label) and last_session is not None and session_label != last_session
    )
    if is_new_session:
        snap = _session_snapshot(story, last_session or "", now)
        story.lineage.append_snapshot(snap)
        story.session_count += 1
        story.is_multi_session = story.session_count >= 1
        # Record the session boundary in the lineage
        boundary_entry = _make_lineage_entry(
            horizon=horizon,
            kind=kind,
            event_type=LineageEventType.SESSION_NEW,
            evidence=f"session transition: {last_session} → {session_label}",
            confidence_before=story.nodes.get(
                horizon.value,
                StoryNode(
                    horizon=horizon,
                    kind=kind,
                    narrative_id="",
                    confidence=0.0,
                    maturity=NarrativeMaturity.FORMING.value,
                    supporting_count=0,
                    conflicting_count=0,
                    evidence_balance=0.5,
                    formed_at=now,
                    last_updated_at=now,
                    summary="",
                ),
            ).confidence,
            confidence_after=story.nodes.get(
                horizon.value,
                StoryNode(
                    horizon=horizon,
                    kind=kind,
                    narrative_id="",
                    confidence=0.0,
                    maturity=NarrativeMaturity.FORMING.value,
                    supporting_count=0,
                    conflicting_count=0,
                    evidence_balance=0.5,
                    formed_at=now,
                    last_updated_at=now,
                    summary="",
                ),
            ).confidence,
            session_label=session_label,
            now=now,
        )
        story.lineage.append_entry(boundary_entry)
        logger.debug(
            f"[Continuity] {instrument} {kind.value}: session transition "
            f"{last_session!r} → {session_label!r} — snapshot saved "
            f"(session_count={story.session_count})"
        )

    # ── Build updated StoryNode ───────────────────────────────────────────
    existing_node = story.nodes.get(horizon.value)
    confidence_before = existing_node.confidence if existing_node else 0.30

    if narrative_engine is not None:
        try:
            sn = narrative_engine.narrative_of_kind(instrument, kind)
        except Exception:
            sn = None
        if sn is not None and sn.is_active():
            node = _node_from_symbol_narrative(horizon, sn)
        else:
            node = _build_node_standalone(
                horizon=horizon,
                kind=kind,
                existing_node=existing_node,
                boost=boost,
                is_supporting=is_supporting,
                now=now,
            )
    else:
        node = _build_node_standalone(
            horizon=horizon,
            kind=kind,
            existing_node=existing_node,
            boost=boost,
            is_supporting=is_supporting,
            now=now,
        )

    # ── Record lineage entry ──────────────────────────────────────────────
    event_type = _detect_event_type(existing_node, is_supporting, story, horizon, kind)
    entry = _make_lineage_entry(
        horizon=horizon,
        kind=kind,
        event_type=event_type,
        evidence=evidence,
        confidence_before=confidence_before,
        confidence_after=node.confidence,
        session_label=session_label,
        now=now,
    )
    story.lineage.append_entry(entry)

    # ── Persist ───────────────────────────────────────────────────────────
    story.nodes[horizon.value] = node
    story.last_updated_at = now
    state[story_key] = story

    return node


# ── Public callable: retrieve_story ──────────────────────────────────────────


def retrieve_story(
    instrument: str,
    kind: NarrativeKind | None = None,
    horizon: NarrativeHorizon | None = None,
    *,
    narrative_engine: "MarketNarrativeEngine | None" = None,
    _state: dict | None = None,
) -> "NarrativeStory | list[NarrativeStory] | None":
    """Retrieve the narrative story (or stories) for an instrument.

    Parameters
    ----------
    instrument:
        Symbol to retrieve.
    kind:
        When given, returns the single ``NarrativeStory`` for that kind,
        or None if it does not exist yet.
        When None, returns a list of all stories for the instrument.
    horizon:
        When given alongside ``kind``, only returns the story when it has
        an active node at that horizon (None otherwise).
        When given without ``kind``, filters the returned list to stories
        that have an active node at that horizon.
    narrative_engine:
        Optional live ``MarketNarrativeEngine``.  When supplied, StoryNodes
        are refreshed with the latest ``SymbolNarrative`` state before the
        story is returned.
    _state:
        Shared mutable state dict (managed by ContinuityEngine).

    Returns
    -------
    NarrativeStory         — when ``kind`` is given and the story exists.
    list[NarrativeStory]   — when ``kind`` is None.
    None                   — when ``kind`` is given but no story exists yet,
                             or the story has no node at the requested horizon.
    """
    state = _state if _state is not None else {}

    if kind is not None:
        story_key = f"{instrument}:{kind.value}"
        story = state.get(story_key)
        if story is None:
            return None
        if narrative_engine is not None:
            _refresh_story_from_engine(story, narrative_engine)
        if horizon is not None and horizon.value not in story.nodes:
            return None
        return story
    else:
        prefix = f"{instrument}:"
        stories = [v for k, v in state.items() if k.startswith(prefix)]
        if narrative_engine is not None:
            for s in stories:
                _refresh_story_from_engine(s, narrative_engine)
        if horizon is not None:
            stories = [s for s in stories if horizon.value in s.nodes]
        return stories


# ── Public callable: evaluate_story_strength ─────────────────────────────────


def evaluate_story_strength(
    instrument: str,
    kind: NarrativeKind | None = None,
    *,
    narrative_engine: "MarketNarrativeEngine | None" = None,
    _state: dict | None = None,
    now: float | None = None,
) -> StoryStrength:
    """Evaluate the strength of the narrative story across time horizons.

    Parameters
    ----------
    instrument:
        Symbol to evaluate.
    kind:
        When given, evaluates the specific story for that ``NarrativeKind``.
        When None, computes a composite across all active stories for the
        instrument.
    narrative_engine:
        Optional live engine for refreshed node state.
    _state:
        Shared mutable state dict.
    now:
        Override current time.

    Returns
    -------
    StoryStrength with overall_score (0-1) and per-component breakdown.

    Score formula
    -------------
    overall = 0.40 × cross_horizon_alignment
            + 0.30 × mean(horizon_scores)
            + 0.20 × session_continuity_score
            + 0.10 × longevity_bonus

    Cross-horizon alignment: 1.0 when all active horizons tell a compatible
    story; penalised proportionally for each incompatible pair.

    Session continuity: fraction of stored ``SessionSnapshot``s where the
    primary narrative kind was active at the SESSION or MACRO horizon.
    Requires ≥2 snapshots to produce a non-zero score.

    Longevity bonus: scales linearly from 0 at 1 session to 1.0 at 10+
    sessions.
    """
    now = now if now is not None else time.time()
    state = _state if _state is not None else {}

    # ── Collect relevant stories ──────────────────────────────────────────
    if kind is not None:
        s = retrieve_story(
            instrument,
            kind,
            narrative_engine=narrative_engine,
            _state=state,
        )
        if s is None:
            stories: list[NarrativeStory] = []
        elif isinstance(s, list):
            stories = s
        else:
            stories = [s]
    else:
        result = retrieve_story(
            instrument,
            narrative_engine=narrative_engine,
            _state=state,
        )
        stories = result if isinstance(result, list) else []

    if not stories:
        return StoryStrength(
            instrument=instrument,
            kind=kind,
            overall_score=0.0,
            horizon_scores={},
            cross_horizon_alignment=0.0,
            session_continuity_score=0.0,
            longevity_bonus=0.0,
            active_horizon_count=0,
            dominant_kind=None,
            is_multi_session=False,
            reasoning="No active stories for this instrument/kind.",
        )

    # ── Aggregate across all stories ──────────────────────────────────────
    all_nodes: list[StoryNode] = []
    horizon_scores: dict[str, float] = {}
    session_conts: list[float] = []
    longevity_vals: list[float] = []
    is_multi = False
    best_conf = -1.0
    dominant_kind: NarrativeKind | None = None

    for story in stories:
        active = story.active_nodes()
        all_nodes.extend(active)
        for node in active:
            h_score = _horizon_score(node)
            existing = horizon_scores.get(node.horizon.value, 0.0)
            horizon_scores[node.horizon.value] = max(existing, h_score)
            if node.confidence > best_conf:
                best_conf = node.confidence
                dominant_kind = node.kind
        session_conts.append(_session_continuity_score(story))
        longevity_vals.append(_longevity_bonus(story))
        if story.is_multi_session:
            is_multi = True

    alignment = _cross_horizon_alignment(all_nodes)
    mean_h = (sum(horizon_scores.values()) / len(horizon_scores)) if horizon_scores else 0.0
    session_c = sum(session_conts) / len(session_conts) if session_conts else 0.0
    longevity = sum(longevity_vals) / len(longevity_vals) if longevity_vals else 0.0

    overall = round(
        min(1.0, max(0.0, 0.40 * alignment + 0.30 * mean_h + 0.20 * session_c + 0.10 * longevity)),
        4,
    )

    reasoning = (
        f"alignment={alignment:.0%} | mean_horizon={mean_h:.0%} | "
        f"session_continuity={session_c:.0%} | longevity={longevity:.0%} | "
        f"active_horizons={len(horizon_scores)} | multi_session={is_multi}"
    )

    return StoryStrength(
        instrument=instrument,
        kind=kind,
        overall_score=overall,
        horizon_scores=horizon_scores,
        cross_horizon_alignment=alignment,
        session_continuity_score=session_c,
        longevity_bonus=longevity,
        active_horizon_count=len(horizon_scores),
        dominant_kind=dominant_kind,
        is_multi_session=is_multi,
        reasoning=reasoning,
    )


# ── ContinuityEngine (stateful wrapper) ───────────────────────────────────────


class ContinuityEngine:
    """Stateful wrapper for the three C14 public callables.

    One instance per process (or per symbol set).  Holds a shared
    ``_state`` dict so all three callables operate on consistent data.

    Parameters
    ----------
    narrative_engine:
        Optional reference to the shared ``MarketNarrativeEngine``.
        When supplied, StoryNodes are refreshed with live ``SymbolNarrative``
        state on every ``retrieve_story()`` or ``evaluate_story_strength()``
        call.  When None, the continuity engine manages its own lightweight
        confidence/maturity tracking per StoryNode.

    Typical usage
    -------------
    engine = ContinuityEngine(narrative_engine=brain.narrative_engine)

    # Every time you observe evidence at a specific horizon:
    node = engine.update_narrative(
        "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION,
        evidence="price compressing below resistance, volume declining",
        session_label="2024-01-15_LONDON",
    )

    # Read the full multi-horizon story:
    story = engine.retrieve_story("EURUSD", NarrativeKind.ACCUMULATION)

    # Score how strong / aligned the story is across time horizons:
    strength = engine.evaluate_story_strength("EURUSD", NarrativeKind.ACCUMULATION)
    print(f"story strength: {strength.overall_score:.0%}")
    """

    def __init__(
        self,
        narrative_engine: "MarketNarrativeEngine | None" = None,
    ) -> None:
        self._engine = narrative_engine
        self._state: dict = {}

    # ── Three public callables ────────────────────────────────────────────

    def update_narrative(
        self,
        instrument: str,
        kind: NarrativeKind,
        horizon: NarrativeHorizon,
        *,
        evidence: str = "",
        boost: float = 0.12,
        is_supporting: bool = True,
        session_label: str = "",
        now: float | None = None,
    ) -> StoryNode:
        """Record a narrative update at the given horizon and store in lineage.

        See module-level ``update_narrative`` for full parameter docs.
        """
        return update_narrative(
            instrument=instrument,
            kind=kind,
            horizon=horizon,
            evidence=evidence,
            boost=boost,
            is_supporting=is_supporting,
            session_label=session_label,
            narrative_engine=self._engine,
            _state=self._state,
            now=now,
        )

    def retrieve_story(
        self,
        instrument: str,
        kind: NarrativeKind | None = None,
        horizon: NarrativeHorizon | None = None,
    ) -> "NarrativeStory | list[NarrativeStory] | None":
        """Return the narrative story for an instrument (optionally filtered).

        See module-level ``retrieve_story`` for full parameter docs.
        """
        return retrieve_story(
            instrument=instrument,
            kind=kind,
            horizon=horizon,
            narrative_engine=self._engine,
            _state=self._state,
        )

    def evaluate_story_strength(
        self,
        instrument: str,
        kind: NarrativeKind | None = None,
        *,
        now: float | None = None,
    ) -> StoryStrength:
        """Evaluate story strength across horizons.

        See module-level ``evaluate_story_strength`` for full parameter docs.
        """
        return evaluate_story_strength(
            instrument=instrument,
            kind=kind,
            narrative_engine=self._engine,
            _state=self._state,
            now=now,
        )

    # ── Convenience methods ───────────────────────────────────────────────

    def all_instruments(self) -> list[str]:
        """All instruments with at least one active story."""
        seen: set[str] = set()
        for key in self._state:
            seen.add(key.split(":")[0])
        return sorted(seen)

    def stories_for(self, instrument: str) -> list[NarrativeStory]:
        """All stories for one instrument (empty list if none)."""
        result = self.retrieve_story(instrument)
        return result if isinstance(result, list) else []

    def active_kinds_for(self, instrument: str) -> list[NarrativeKind]:
        """All NarrativeKind values with active stories for an instrument."""
        return [s.primary_kind for s in self.stories_for(instrument) if s.active_nodes()]

    def summary(self, instrument: str) -> dict:
        """Compact summary dict for logging or dashboards."""
        stories = self.stories_for(instrument)
        strength = self.evaluate_story_strength(instrument)
        return {
            "instrument": instrument,
            "story_count": len(stories),
            "multi_session": any(s.is_multi_session for s in stories),
            "overall_strength": strength.overall_score,
            "dominant_kind": (strength.dominant_kind.value if strength.dominant_kind else None),
            "active_horizons": strength.active_horizon_count,
            "stories": [s.to_dict() for s in stories],
        }

    def to_snapshot(self, instrument: str) -> dict:
        """Full serialisable snapshot for persistence / debugging."""
        return {
            "instrument": instrument,
            "stories": [s.to_dict() for s in self.stories_for(instrument)],
        }


__all__ = [
    # Public callables
    "update_narrative",
    "retrieve_story",
    "evaluate_story_strength",
    # Stateful wrapper
    "ContinuityEngine",
]
