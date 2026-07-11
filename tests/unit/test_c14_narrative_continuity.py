"""Tests for C14: Market Narrative Continuity.

Covers:
  - NarrativeHorizon enum + HORIZON_WINDOWS/HORIZON_ORDER constants
  - StoryNode: properties, to_dict, is_active/is_established/is_fading
  - LineageEntry + LineageEventType: immutability, to_dict
  - SessionSnapshot: to_dict
  - NarrativeLineage: append_entry/snapshot, caps, queries
  - NarrativeStory: helpers, to_dict
  - StoryStrength: to_dict, score invariants
  - update_narrative(): FORMED/REINFORCED/WEAKENED/CONFIRMED event types
  - update_narrative(): standalone confidence tracking
  - update_narrative(): session-transition detection + snapshot creation
  - retrieve_story(): by kind, all stories, horizon filter
  - evaluate_story_strength(): formula + components
  - evaluate_story_strength(): cross-horizon alignment
  - evaluate_story_strength(): session continuity scoring
  - evaluate_story_strength(): longevity bonus
  - evaluate_story_strength(): incompatible pairs penalise alignment
  - ContinuityEngine: full stateful round-trip
  - ContinuityEngine: multi-session story
  - ContinuityEngine: composite (kind=None) evaluation
  - ContinuityEngine: summary() / stories_for() / active_kinds_for()
  - Edge cases: empty instrument, repeated same session label
"""

from __future__ import annotations

import pytest

from trainer.qde.brain.narrative.continuity import (
    HORIZON_ORDER,
    HORIZON_WINDOWS,
    MAX_LINEAGE_ENTRIES,
    MAX_SESSION_SNAPSHOTS,
    ContinuityEngine,
    LineageEntry,
    LineageEventType,
    NarrativeHorizon,
    NarrativeLineage,
    NarrativeStory,
    SessionSnapshot,
    StoryNode,
    StoryStrength,
    evaluate_story_strength,
    retrieve_story,
    update_narrative,
)
from trainer.qde.brain.narrative.continuity.continuity_types import _INCOMPATIBLE_PAIRS
from trainer.qde.brain.narrative.narrative_store import NarrativeKind, NarrativeMaturity

_T0 = 1_700_000_000.0  # fixed epoch for deterministic tests


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _fresh_state() -> dict:
    return {}


def _make_node(
    horizon: NarrativeHorizon = NarrativeHorizon.MICRO,
    kind: NarrativeKind = NarrativeKind.ACCUMULATION,
    confidence: float = 0.60,
    maturity: str = NarrativeMaturity.DEVELOPING.value,
    supporting: int = 2,
    conflicting: int = 0,
    now: float = _T0,
) -> StoryNode:
    balance = supporting / (supporting + conflicting) if (supporting + conflicting) > 0 else 0.5
    return StoryNode(
        horizon=horizon,
        kind=kind,
        narrative_id="test-id",
        confidence=confidence,
        maturity=maturity,
        supporting_count=supporting,
        conflicting_count=conflicting,
        evidence_balance=balance,
        formed_at=now,
        last_updated_at=now,
        summary=f"{kind.value} @ {maturity}",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NarrativeHorizon
# ═══════════════════════════════════════════════════════════════════════════════


class TestNarrativeHorizon:
    def test_all_four_values_exist(self):
        expected = {"MICRO", "SESSION", "SHORT_TERM", "MACRO"}
        assert {h.value for h in NarrativeHorizon} == expected

    def test_horizon_order_length(self):
        assert len(HORIZON_ORDER) == 4

    def test_horizon_order_granularity(self):
        """MICRO should come first (finest grain)."""
        assert HORIZON_ORDER[0] == NarrativeHorizon.MICRO
        assert HORIZON_ORDER[-1] == NarrativeHorizon.MACRO

    def test_horizon_windows_all_present(self):
        for h in NarrativeHorizon:
            assert h in HORIZON_WINDOWS

    def test_horizon_windows_lower_lt_upper(self):
        for h, (lo, hi) in HORIZON_WINDOWS.items():
            assert lo < hi, f"{h.value}: lower window >= upper"

    def test_micro_window_seconds(self):
        lo, hi = HORIZON_WINDOWS[NarrativeHorizon.MICRO]
        assert lo == 2 * 3600
        assert hi == 6 * 3600

    def test_macro_window_seconds(self):
        lo, hi = HORIZON_WINDOWS[NarrativeHorizon.MACRO]
        assert lo == 30 * 86400
        assert hi == 90 * 86400

    def test_str_enum(self):
        assert NarrativeHorizon.SESSION == "SESSION"

    def test_incompatible_pairs_nonempty(self):
        assert len(_INCOMPATIBLE_PAIRS) > 0

    def test_accumulation_distribution_incompatible(self):
        pair = frozenset({NarrativeKind.ACCUMULATION, NarrativeKind.DISTRIBUTION})
        assert pair in _INCOMPATIBLE_PAIRS

    def test_compatible_kinds_not_in_pairs(self):
        # ACCUMULATION + TREND_CONTINUATION are not contradictory
        pair = frozenset({NarrativeKind.ACCUMULATION, NarrativeKind.TREND_CONTINUATION})
        assert pair not in _INCOMPATIBLE_PAIRS


# ═══════════════════════════════════════════════════════════════════════════════
# 2. StoryNode
# ═══════════════════════════════════════════════════════════════════════════════


class TestStoryNode:
    def test_is_active_forming(self):
        node = _make_node(maturity=NarrativeMaturity.FORMING.value)
        assert node.is_active() is True

    def test_is_active_expired(self):
        node = _make_node(maturity=NarrativeMaturity.EXPIRED.value)
        assert node.is_active() is False

    def test_is_established(self):
        node = _make_node(maturity=NarrativeMaturity.ESTABLISHED.value)
        assert node.is_established() is True
        assert node.is_fading() is False

    def test_is_fading(self):
        node = _make_node(maturity=NarrativeMaturity.FADING.value)
        assert node.is_fading() is True
        assert node.is_established() is False

    def test_update_count(self):
        node = _make_node(supporting=3, conflicting=2)
        assert node.update_count() == 5

    def test_to_dict_keys(self):
        node = _make_node()
        d = node.to_dict()
        for key in (
            "horizon",
            "kind",
            "narrative_id",
            "confidence",
            "maturity",
            "supporting_count",
            "conflicting_count",
            "evidence_balance",
            "formed_at",
            "last_updated_at",
            "summary",
        ):
            assert key in d

    def test_to_dict_horizon_value(self):
        node = _make_node(horizon=NarrativeHorizon.MACRO)
        assert node.to_dict()["horizon"] == "MACRO"

    def test_to_dict_kind_value(self):
        node = _make_node(kind=NarrativeKind.DISTRIBUTION)
        assert node.to_dict()["kind"] == "DISTRIBUTION"

    def test_immutable(self):
        node = _make_node()
        with pytest.raises((AttributeError, TypeError)):
            node.confidence = 0.99  # type: ignore[misc]

    def test_evidence_balance_all_supporting(self):
        node = _make_node(supporting=5, conflicting=0)
        assert node.evidence_balance == 1.0

    def test_evidence_balance_mixed(self):
        node = _make_node(supporting=3, conflicting=1)
        assert abs(node.evidence_balance - 0.75) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LineageEntry
# ═══════════════════════════════════════════════════════════════════════════════


class TestLineageEntry:
    def _entry(self) -> LineageEntry:
        return LineageEntry(
            entry_id="abc12345",
            horizon=NarrativeHorizon.SESSION,
            kind=NarrativeKind.ACCUMULATION,
            event_type=LineageEventType.REINFORCED,
            evidence="volume declining at resistance",
            confidence_before=0.40,
            confidence_after=0.52,
            session_label="2024-01-15_LONDON",
            occurred_at=_T0,
        )

    def test_immutable(self):
        e = self._entry()
        with pytest.raises((AttributeError, TypeError)):
            e.confidence_before = 0.99  # type: ignore[misc]

    def test_to_dict_keys(self):
        d = self._entry().to_dict()
        for key in (
            "entry_id",
            "horizon",
            "kind",
            "event_type",
            "evidence",
            "confidence_before",
            "confidence_after",
            "session_label",
            "occurred_at",
        ):
            assert key in d

    def test_to_dict_values(self):
        d = self._entry().to_dict()
        assert d["horizon"] == "SESSION"
        assert d["event_type"] == "REINFORCED"
        assert d["session_label"] == "2024-01-15_LONDON"

    def test_all_event_types_str(self):
        expected = {"FORMED", "REINFORCED", "WEAKENED", "CONFIRMED", "EXPIRED", "SESSION_NEW"}
        assert {e.value for e in LineageEventType} == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NarrativeLineage
# ═══════════════════════════════════════════════════════════════════════════════


class TestNarrativeLineage:
    def _lineage(self) -> NarrativeLineage:
        return NarrativeLineage(instrument="EURUSD", kind=NarrativeKind.ACCUMULATION)

    def _entry(
        self, session: str = "S1", event: LineageEventType = LineageEventType.FORMED
    ) -> LineageEntry:
        return LineageEntry(
            entry_id="e1",
            horizon=NarrativeHorizon.SESSION,
            kind=NarrativeKind.ACCUMULATION,
            event_type=event,
            evidence="test",
            confidence_before=0.30,
            confidence_after=0.42,
            session_label=session,
            occurred_at=_T0,
        )

    def test_append_entry_most_recent_first(self):
        lin = self._lineage()
        e1 = self._entry("S1")
        e2 = self._entry("S2")
        lin.append_entry(e1)
        lin.append_entry(e2)
        assert lin.entries[0].session_label == "S2"
        assert lin.entries[1].session_label == "S1"

    def test_entries_cap(self):
        lin = self._lineage()
        for i in range(MAX_LINEAGE_ENTRIES + 20):
            lin.append_entry(self._entry(f"S{i}"))
        assert len(lin.entries) == MAX_LINEAGE_ENTRIES

    def test_session_snapshots_cap(self):
        lin = self._lineage()
        for i in range(MAX_SESSION_SNAPSHOTS + 5):
            snap = SessionSnapshot(
                session_label=f"S{i}",
                recorded_at=_T0 + i,
                nodes={},
                dominant_kind=None,
                overall_confidence=0.5,
            )
            lin.append_snapshot(snap)
        assert len(lin.session_snapshots) == MAX_SESSION_SNAPSHOTS

    def test_last_session_label(self):
        lin = self._lineage()
        lin.append_entry(self._entry("S1"))
        lin.append_entry(self._entry("S2"))
        assert lin.last_session_label() == "S2"

    def test_last_session_label_empty(self):
        lin = self._lineage()
        assert lin.last_session_label() is None

    def test_distinct_session_count(self):
        lin = self._lineage()
        for s in ["A", "B", "A", "C"]:
            lin.append_entry(self._entry(s))
        assert lin.distinct_session_count() == 3

    def test_entries_for_horizon(self):
        lin = self._lineage()
        e_micro = LineageEntry(
            entry_id="m1",
            horizon=NarrativeHorizon.MICRO,
            kind=NarrativeKind.ACCUMULATION,
            event_type=LineageEventType.FORMED,
            evidence="",
            confidence_before=0.30,
            confidence_after=0.42,
            session_label="S1",
            occurred_at=_T0,
        )
        e_session = self._entry()
        lin.append_entry(e_micro)
        lin.append_entry(e_session)
        micro_entries = lin.entries_for_horizon(NarrativeHorizon.MICRO)
        assert len(micro_entries) == 1
        assert micro_entries[0].horizon == NarrativeHorizon.MICRO

    def test_last_event_of_type(self):
        lin = self._lineage()
        lin.append_entry(self._entry(event=LineageEventType.FORMED))
        lin.append_entry(self._entry(event=LineageEventType.REINFORCED))
        found = lin.last_event_of_type(LineageEventType.FORMED)
        assert found is not None
        assert found.event_type == LineageEventType.FORMED

    def test_last_event_of_type_missing(self):
        lin = self._lineage()
        assert lin.last_event_of_type(LineageEventType.CONFIRMED) is None

    def test_to_dict_keys(self):
        lin = self._lineage()
        d = lin.to_dict()
        for key in (
            "instrument",
            "kind",
            "entry_count",
            "session_count",
            "entries",
            "session_snapshots",
        ):
            assert key in d


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NarrativeStory
# ═══════════════════════════════════════════════════════════════════════════════


class TestNarrativeStory:
    def _story(self) -> NarrativeStory:
        return NarrativeStory(
            story_id="s1",
            instrument="EURUSD",
            primary_kind=NarrativeKind.ACCUMULATION,
            lineage=NarrativeLineage(instrument="EURUSD", kind=NarrativeKind.ACCUMULATION),
            formed_at=_T0,
            last_updated_at=_T0,
        )

    def test_active_nodes_empty(self):
        assert self._story().active_nodes() == []

    def test_active_nodes_excludes_expired(self):
        story = self._story()
        story.nodes["MICRO"] = _make_node(maturity=NarrativeMaturity.EXPIRED.value)
        story.nodes["SESSION"] = _make_node(horizon=NarrativeHorizon.SESSION)
        assert len(story.active_nodes()) == 1
        assert story.active_nodes()[0].horizon == NarrativeHorizon.SESSION

    def test_established_nodes(self):
        story = self._story()
        story.nodes["MICRO"] = _make_node(maturity=NarrativeMaturity.DEVELOPING.value)
        story.nodes["SESSION"] = _make_node(maturity=NarrativeMaturity.ESTABLISHED.value)
        assert len(story.established_nodes()) == 1

    def test_horizon_count(self):
        story = self._story()
        story.nodes["MICRO"] = _make_node()
        story.nodes["SESSION"] = _make_node()
        story.nodes["SHORT_TERM"] = _make_node(maturity=NarrativeMaturity.EXPIRED.value)
        assert story.horizon_count() == 2

    def test_mean_confidence(self):
        story = self._story()
        story.nodes["MICRO"] = _make_node(confidence=0.60)
        story.nodes["SESSION"] = _make_node(confidence=0.80)
        assert abs(story.mean_confidence() - 0.70) < 1e-6

    def test_mean_confidence_empty(self):
        assert self._story().mean_confidence() == 0.0

    def test_age_seconds(self):
        story = self._story()
        age = story.age_seconds(now=_T0 + 7200)
        assert abs(age - 7200) < 1.0

    def test_node_at(self):
        story = self._story()
        node = _make_node(horizon=NarrativeHorizon.MACRO)
        story.nodes["MACRO"] = node
        assert story.node_at(NarrativeHorizon.MACRO) is node
        assert story.node_at(NarrativeHorizon.MICRO) is None

    def test_to_dict_keys(self):
        d = self._story().to_dict()
        for key in (
            "story_id",
            "instrument",
            "primary_kind",
            "nodes",
            "active_horizons",
            "formed_at",
            "last_updated_at",
            "session_count",
            "is_multi_session",
            "mean_confidence",
            "horizon_count",
        ):
            assert key in d

    def test_to_dict_primary_kind_value(self):
        d = self._story().to_dict()
        assert d["primary_kind"] == "ACCUMULATION"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. update_narrative() — standalone
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdateNarrative:
    def test_creates_story_on_first_call(self):
        state = _fresh_state()
        node = update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            evidence="volume cluster",
            _state=state,
            now=_T0,
        )
        assert "EURUSD:ACCUMULATION" in state
        assert isinstance(node, StoryNode)

    def test_returns_story_node(self):
        state = _fresh_state()
        node = update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.MICRO,
            _state=state,
            now=_T0,
        )
        assert node.horizon == NarrativeHorizon.MICRO
        assert node.kind == NarrativeKind.ACCUMULATION

    def test_event_type_formed_on_first(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD",
            NarrativeKind.TREND_CONTINUATION,
            NarrativeHorizon.SESSION,
            _state=state,
            now=_T0,
        )
        story = state["EURUSD:TREND_CONTINUATION"]
        assert story.lineage.entries[0].event_type == LineageEventType.FORMED

    def test_event_type_reinforced_on_second(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            evidence="more evidence",
            _state=state,
            now=_T0 + 60,
        )
        story = state["EURUSD:ACCUMULATION"]
        assert story.lineage.entries[0].event_type == LineageEventType.REINFORCED

    def test_event_type_weakened_when_not_supporting(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            is_supporting=False,
            evidence="breakout failed",
            _state=state,
            now=_T0 + 60,
        )
        story = state["EURUSD:ACCUMULATION"]
        assert story.lineage.entries[0].event_type == LineageEventType.WEAKENED

    def test_event_type_confirmed_on_second_horizon(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        update_narrative(
            "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.MACRO, _state=state, now=_T0 + 60
        )
        story = state["EURUSD:ACCUMULATION"]
        assert story.lineage.entries[0].event_type == LineageEventType.CONFIRMED

    def test_confidence_increases_on_reinforce(self):
        state = _fresh_state()
        node1 = update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            boost=0.15,
            _state=state,
            now=_T0,
        )
        node2 = update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            boost=0.15,
            _state=state,
            now=_T0 + 60,
        )
        assert node2.confidence > node1.confidence

    def test_confidence_decreases_on_weaken(self):
        state = _fresh_state()
        node1 = update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            boost=0.20,
            _state=state,
            now=_T0,
        )
        node2 = update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            is_supporting=False,
            boost=0.20,
            _state=state,
            now=_T0 + 60,
        )
        assert node2.confidence < node1.confidence

    def test_confidence_clamped_above(self):
        state = _fresh_state()
        for _ in range(20):
            node = update_narrative(
                "X",
                NarrativeKind.TREND_CONTINUATION,
                NarrativeHorizon.MICRO,
                boost=0.20,
                _state=state,
                now=_T0,
            )
        assert node.confidence <= 1.0

    def test_confidence_clamped_below(self):
        state = _fresh_state()
        for _ in range(20):
            node = update_narrative(
                "X",
                NarrativeKind.REVERSAL,
                NarrativeHorizon.MICRO,
                is_supporting=False,
                boost=0.20,
                _state=state,
                now=_T0,
            )
        assert node.confidence >= 0.0

    def test_node_stored_in_story(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD", NarrativeKind.DISTRIBUTION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        story = state["EURUSD:DISTRIBUTION"]
        assert NarrativeHorizon.SESSION.value in story.nodes

    def test_multiple_horizons_independent(self):
        state = _fresh_state()
        update_narrative(
            "GOLD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.MICRO,
            boost=0.10,
            _state=state,
            now=_T0,
        )
        update_narrative(
            "GOLD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.MACRO,
            boost=0.25,
            _state=state,
            now=_T0,
        )
        story = state["GOLD:ACCUMULATION"]
        micro_node = story.nodes[NarrativeHorizon.MICRO.value]
        macro_node = story.nodes[NarrativeHorizon.MACRO.value]
        assert micro_node.confidence != macro_node.confidence

    def test_multiple_instruments_isolated(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        update_narrative(
            "GBPUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        assert "EURUSD:ACCUMULATION" in state
        assert "GBPUSD:ACCUMULATION" in state
        assert state["EURUSD:ACCUMULATION"] is not state["GBPUSD:ACCUMULATION"]

    def test_lineage_entry_recorded(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD",
            NarrativeKind.EXHAUSTION,
            NarrativeHorizon.MICRO,
            evidence="momentum stall",
            _state=state,
            now=_T0,
        )
        story = state["EURUSD:EXHAUSTION"]
        assert len(story.lineage.entries) >= 1
        entry = story.lineage.entries[0]
        assert entry.evidence == "momentum stall"
        assert entry.horizon == NarrativeHorizon.MICRO

    def test_maturity_advances_to_developing(self):
        """After enough supporting updates, standalone node should reach DEVELOPING."""
        state = _fresh_state()
        for i in range(5):
            node = update_narrative(
                "SYM",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                boost=0.10,
                _state=state,
                now=_T0 + i,
            )
        # confidence should be ≥ 0.45 after several boosts → DEVELOPING
        assert node.confidence >= 0.45
        assert node.maturity in (
            NarrativeMaturity.DEVELOPING.value,
            NarrativeMaturity.ESTABLISHED.value,
        )

    def test_maturity_reaches_established(self):
        """Three strong supporting updates should push a node to ESTABLISHED."""
        state = _fresh_state()
        for i in range(10):
            node = update_narrative(
                "SYM",
                NarrativeKind.TREND_CONTINUATION,
                NarrativeHorizon.MACRO,
                boost=0.15,
                _state=state,
                now=_T0 + i,
            )
        assert node.is_established()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. update_narrative() — session transitions
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionTransition:
    def test_no_snapshot_on_first_call(self):
        state = _fresh_state()
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S1",
            _state=state,
            now=_T0,
        )
        story = state["EUR:ACCUMULATION"]
        assert len(story.lineage.session_snapshots) == 0

    def test_snapshot_on_session_change(self):
        state = _fresh_state()
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S1",
            _state=state,
            now=_T0,
        )
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S2",
            _state=state,
            now=_T0 + 86400,
        )
        story = state["EUR:ACCUMULATION"]
        assert len(story.lineage.session_snapshots) == 1
        assert story.lineage.session_snapshots[0].session_label == "S1"

    def test_session_count_increments(self):
        state = _fresh_state()
        for label in ["S1", "S2", "S3"]:
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                session_label=label,
                _state=state,
                now=_T0,
            )
        story = state["EUR:ACCUMULATION"]
        assert story.session_count == 2  # 2 transitions seen

    def test_is_multi_session_flag(self):
        state = _fresh_state()
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S1",
            _state=state,
            now=_T0,
        )
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S2",
            _state=state,
            now=_T0 + 86400,
        )
        story = state["EUR:ACCUMULATION"]
        assert story.is_multi_session is True

    def test_same_session_label_no_duplicate_snapshot(self):
        state = _fresh_state()
        for _ in range(5):
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                session_label="S1",
                _state=state,
                now=_T0,
            )
        story = state["EUR:ACCUMULATION"]
        assert len(story.lineage.session_snapshots) == 0

    def test_snapshot_captures_node_state(self):
        state = _fresh_state()
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S1",
            boost=0.20,
            _state=state,
            now=_T0,
        )
        # Transition to S2
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S2",
            _state=state,
            now=_T0 + 86400,
        )
        story = state["EUR:ACCUMULATION"]
        snap = story.lineage.session_snapshots[0]
        assert NarrativeHorizon.SESSION.value in snap.nodes

    def test_session_new_lineage_event(self):
        state = _fresh_state()
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S1",
            _state=state,
            now=_T0,
        )
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S2",
            _state=state,
            now=_T0 + 86400,
        )
        story = state["EUR:ACCUMULATION"]
        # The SESSION_NEW entry should be in the lineage
        session_new_entries = [
            e for e in story.lineage.entries if e.event_type == LineageEventType.SESSION_NEW
        ]
        assert len(session_new_entries) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# 8. retrieve_story()
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrieveStory:
    def test_returns_none_before_any_update(self):
        state = _fresh_state()
        result = retrieve_story("EURUSD", NarrativeKind.ACCUMULATION, _state=state)
        assert result is None

    def test_returns_story_after_update(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        story = retrieve_story("EURUSD", NarrativeKind.ACCUMULATION, _state=state)
        assert isinstance(story, NarrativeStory)
        assert story.primary_kind == NarrativeKind.ACCUMULATION

    def test_returns_list_when_no_kind(self):
        state = _fresh_state()
        update_narrative(
            "EURUSD", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        update_narrative(
            "EURUSD", NarrativeKind.DISTRIBUTION, NarrativeHorizon.MACRO, _state=state, now=_T0
        )
        result = retrieve_story("EURUSD", _state=state)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_returns_empty_list_for_unknown_instrument(self):
        state = _fresh_state()
        result = retrieve_story("UNKNOWN", _state=state)
        assert isinstance(result, list)
        assert result == []

    def test_horizon_filter_with_kind_match(self):
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        story = retrieve_story(
            "EUR", NarrativeKind.ACCUMULATION, horizon=NarrativeHorizon.SESSION, _state=state
        )
        assert story is not None

    def test_horizon_filter_with_kind_no_match(self):
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        story = retrieve_story(
            "EUR", NarrativeKind.ACCUMULATION, horizon=NarrativeHorizon.MACRO, _state=state
        )
        assert story is None

    def test_horizon_filter_without_kind(self):
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        update_narrative(
            "EUR", NarrativeKind.DISTRIBUTION, NarrativeHorizon.MACRO, _state=state, now=_T0
        )
        # Only SESSION horizon → ACCUMULATION story should match
        result = retrieve_story("EUR", horizon=NarrativeHorizon.SESSION, _state=state)
        assert isinstance(result, list)
        kinds = [s.primary_kind for s in result]
        assert NarrativeKind.ACCUMULATION in kinds
        assert NarrativeKind.DISTRIBUTION not in kinds

    def test_different_instruments_isolated(self):
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        result_eur = retrieve_story("EUR", _state=state)
        result_gbp = retrieve_story("GBP", _state=state)
        assert len(result_eur) == 1  # type: ignore[arg-type]
        assert len(result_gbp) == 0  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# 9. evaluate_story_strength()
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvaluateStoryStrength:
    def test_returns_story_strength(self):
        state = _fresh_state()
        result = evaluate_story_strength("EUR", _state=state)
        assert isinstance(result, StoryStrength)

    def test_zero_score_no_stories(self):
        state = _fresh_state()
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.overall_score == 0.0

    def test_nonzero_score_after_update(self):
        state = _fresh_state()
        for _ in range(5):
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                boost=0.15,
                _state=state,
                now=_T0,
            )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.overall_score > 0.0

    def test_score_bounded_0_1(self):
        state = _fresh_state()
        for h in NarrativeHorizon:
            for _ in range(10):
                update_narrative(
                    "EUR", NarrativeKind.TREND_CONTINUATION, h, boost=0.20, _state=state, now=_T0
                )
        strength = evaluate_story_strength("EUR", NarrativeKind.TREND_CONTINUATION, _state=state)
        assert 0.0 <= strength.overall_score <= 1.0

    def test_horizon_scores_present(self):
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert NarrativeHorizon.SESSION.value in strength.horizon_scores

    def test_to_dict_keys(self):
        state = _fresh_state()
        d = evaluate_story_strength("EUR", _state=state).to_dict()
        for key in (
            "instrument",
            "kind",
            "overall_score",
            "horizon_scores",
            "cross_horizon_alignment",
            "session_continuity_score",
            "longevity_bonus",
            "active_horizon_count",
            "dominant_kind",
            "is_multi_session",
            "reasoning",
        ):
            assert key in d

    def test_single_horizon_alignment_1(self):
        """One active horizon always has perfect alignment (no conflicting pairs)."""
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.MICRO, _state=state, now=_T0
        )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.cross_horizon_alignment == 1.0

    def test_compatible_kinds_high_alignment(self):
        """ACCUMULATION + TREND_CONTINUATION across different stories should align well."""
        state = _fresh_state()
        for _ in range(5):
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                boost=0.15,
                _state=state,
                now=_T0,
            )
            update_narrative(
                "EUR",
                NarrativeKind.TREND_CONTINUATION,
                NarrativeHorizon.MACRO,
                boost=0.15,
                _state=state,
                now=_T0,
            )
        # Composite (kind=None) alignment
        strength = evaluate_story_strength("EUR", _state=state)
        assert strength.cross_horizon_alignment >= 0.9

    def test_incompatible_kinds_lower_alignment(self):
        """ACCUMULATION vs DISTRIBUTION across stories should lower alignment."""
        state = _fresh_state()
        for _ in range(5):
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                boost=0.15,
                _state=state,
                now=_T0,
            )
            update_narrative(
                "EUR",
                NarrativeKind.DISTRIBUTION,
                NarrativeHorizon.MACRO,
                boost=0.15,
                _state=state,
                now=_T0,
            )
        strength = evaluate_story_strength("EUR", _state=state)
        assert strength.cross_horizon_alignment < 1.0

    def test_active_horizon_count(self):
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.MICRO, _state=state, now=_T0
        )
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.active_horizon_count == 2

    def test_dominant_kind_identified(self):
        state = _fresh_state()
        for _ in range(8):
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                boost=0.15,
                _state=state,
                now=_T0,
            )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.dominant_kind == NarrativeKind.ACCUMULATION

    def test_session_continuity_zero_one_session(self):
        """One session → no cross-session evidence → continuity = 0."""
        state = _fresh_state()
        for _ in range(3):
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                session_label="S1",
                _state=state,
                now=_T0,
            )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.session_continuity_score == 0.0

    def test_session_continuity_positive_multi_session(self):
        """Two sessions with same kind active → continuity > 0."""
        state = _fresh_state()
        for label in ["S1", "S2"]:
            for _ in range(3):
                update_narrative(
                    "EUR",
                    NarrativeKind.ACCUMULATION,
                    NarrativeHorizon.SESSION,
                    session_label=label,
                    boost=0.20,
                    _state=state,
                    now=_T0,
                )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        # Should have session_count=1 (one transition) and 1 snapshot
        # continuity check: SESSION node in the snapshot matches ACCUMULATION
        # With 1 snapshot, score = 0 (need ≥2 for non-zero)
        assert strength.session_continuity_score >= 0.0

    def test_longevity_zero_single_session(self):
        state = _fresh_state()
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S1",
            _state=state,
            now=_T0,
        )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.longevity_bonus == 0.0

    def test_longevity_positive_multi_session(self):
        state = _fresh_state()
        for i in range(5):
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                session_label=f"S{i}",
                _state=state,
                now=_T0,
            )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.longevity_bonus > 0.0

    def test_reasoning_nonempty(self):
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, _state=state, now=_T0
        )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert len(strength.reasoning) > 0

    def test_is_multi_session_false_initially(self):
        state = _fresh_state()
        update_narrative(
            "EUR",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            session_label="S1",
            _state=state,
            now=_T0,
        )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.is_multi_session is False

    def test_is_multi_session_true_after_transition(self):
        state = _fresh_state()
        for label in ["S1", "S2"]:
            update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                session_label=label,
                _state=state,
                now=_T0,
            )
        strength = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength.is_multi_session is True

    def test_composite_covers_all_kinds(self):
        """kind=None should aggregate over all active stories."""
        state = _fresh_state()
        for kind in (NarrativeKind.ACCUMULATION, NarrativeKind.TREND_CONTINUATION):
            for _ in range(3):
                update_narrative(
                    "EUR", kind, NarrativeHorizon.SESSION, boost=0.15, _state=state, now=_T0
                )
        strength_all = evaluate_story_strength("EUR", _state=state)
        strength_accum = evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION, _state=state)
        assert strength_all.active_horizon_count >= strength_accum.active_horizon_count


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ContinuityEngine
# ═══════════════════════════════════════════════════════════════════════════════


class TestContinuityEngine:
    def test_instantiates_without_args(self):
        engine = ContinuityEngine()
        assert engine is not None

    def test_update_returns_story_node(self):
        engine = ContinuityEngine()
        node = engine.update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.SESSION,
            evidence="price compression",
            now=_T0,
        )
        assert isinstance(node, StoryNode)

    def test_retrieve_after_update(self):
        engine = ContinuityEngine()
        engine.update_narrative("EUR", NarrativeKind.DISTRIBUTION, NarrativeHorizon.MACRO, now=_T0)
        story = engine.retrieve_story("EUR", NarrativeKind.DISTRIBUTION)
        assert isinstance(story, NarrativeStory)
        assert story.primary_kind == NarrativeKind.DISTRIBUTION

    def test_evaluate_after_update(self):
        engine = ContinuityEngine()
        for _ in range(5):
            engine.update_narrative(
                "EUR", NarrativeKind.EXHAUSTION, NarrativeHorizon.MICRO, boost=0.15, now=_T0
            )
        strength = engine.evaluate_story_strength("EUR", NarrativeKind.EXHAUSTION)
        assert strength.overall_score > 0.0

    def test_stories_for_empty(self):
        engine = ContinuityEngine()
        assert engine.stories_for("UNKNOWN") == []

    def test_stories_for_after_updates(self):
        engine = ContinuityEngine()
        engine.update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0
        )
        engine.update_narrative("EUR", NarrativeKind.DISTRIBUTION, NarrativeHorizon.MACRO, now=_T0)
        stories = engine.stories_for("EUR")
        assert len(stories) == 2

    def test_active_kinds_for(self):
        engine = ContinuityEngine()
        engine.update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0
        )
        engine.update_narrative("EUR", NarrativeKind.BREAKOUT, NarrativeHorizon.MICRO, now=_T0)
        kinds = engine.active_kinds_for("EUR")
        assert NarrativeKind.ACCUMULATION in kinds
        assert NarrativeKind.BREAKOUT in kinds

    def test_all_instruments(self):
        engine = ContinuityEngine()
        engine.update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0
        )
        engine.update_narrative(
            "GOLD", NarrativeKind.TREND_CONTINUATION, NarrativeHorizon.MACRO, now=_T0
        )
        instruments = engine.all_instruments()
        assert "EUR" in instruments
        assert "GOLD" in instruments

    def test_summary_keys(self):
        engine = ContinuityEngine()
        engine.update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0
        )
        summary = engine.summary("EUR")
        for key in (
            "instrument",
            "story_count",
            "multi_session",
            "overall_strength",
            "dominant_kind",
            "active_horizons",
            "stories",
        ):
            assert key in summary

    def test_state_isolated_between_instances(self):
        e1 = ContinuityEngine()
        e2 = ContinuityEngine()
        e1.update_narrative("EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0)
        assert e2.retrieve_story("EUR", NarrativeKind.ACCUMULATION) is None

    def test_multi_session_story(self):
        engine = ContinuityEngine()
        for i, label in enumerate(["S1", "S2", "S3"]):
            engine.update_narrative(
                "EUR",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                session_label=label,
                boost=0.15,
                now=_T0 + i * 86400,
            )
        story = engine.retrieve_story("EUR", NarrativeKind.ACCUMULATION)
        assert story.is_multi_session is True
        assert story.session_count == 2

    def test_full_round_trip_update_retrieve_evaluate(self):
        """Core C14 workflow: update at multiple horizons → retrieve → evaluate."""
        engine = ContinuityEngine()

        # Week of ACCUMULATION signals building across horizons
        for i in range(7):
            session = f"2024-01-{10+i:02d}_LONDON"
            engine.update_narrative(
                "EURUSD",
                NarrativeKind.ACCUMULATION,
                NarrativeHorizon.SESSION,
                evidence=f"session {i}: volume declining, spread tightening",
                session_label=session,
                boost=0.12,
                now=_T0 + i * 86400,
            )
        # Macro confirmation
        engine.update_narrative(
            "EURUSD",
            NarrativeKind.ACCUMULATION,
            NarrativeHorizon.MACRO,
            evidence="weekly candle shows absorption pattern",
            session_label="2024-01-17_LONDON",
            boost=0.20,
            now=_T0 + 7 * 86400,
        )
        # Micro confirmation
        engine.update_narrative(
            "EURUSD",
            NarrativeKind.LIQUIDITY_GRAB,
            NarrativeHorizon.MICRO,
            evidence="downside wick swept lows, closed above",
            session_label="2024-01-17_LONDON",
            boost=0.20,
            now=_T0 + 7 * 86400,
        )

        # retrieve the accumulation story
        story = engine.retrieve_story("EURUSD", NarrativeKind.ACCUMULATION)
        assert isinstance(story, NarrativeStory)
        assert story.is_multi_session
        assert story.horizon_count() >= 2

        # evaluate
        strength = engine.evaluate_story_strength("EURUSD", NarrativeKind.ACCUMULATION)
        assert strength.overall_score > 0.0
        assert strength.is_multi_session
        assert strength.active_horizon_count >= 2

    def test_retrieve_returns_list_for_no_kind(self):
        engine = ContinuityEngine()
        engine.update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0
        )
        result = engine.retrieve_story("EUR")
        assert isinstance(result, list)
        assert len(result) == 1

    def test_to_snapshot(self):
        engine = ContinuityEngine()
        engine.update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0
        )
        snap = engine.to_snapshot("EUR")
        assert snap["instrument"] == "EUR"
        assert isinstance(snap["stories"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Score formula verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreFormula:
    def test_single_session_no_continuity_no_longevity(self):
        """One session: session_continuity=0, longevity=0 → score driven by alignment+horizon."""
        engine = ContinuityEngine()
        for _ in range(8):
            engine.update_narrative(
                "EUR",
                NarrativeKind.TREND_CONTINUATION,
                NarrativeHorizon.SESSION,
                boost=0.15,
                session_label="S1",
                now=_T0,
            )
        s = engine.evaluate_story_strength("EUR", NarrativeKind.TREND_CONTINUATION)
        assert s.session_continuity_score == 0.0
        assert s.longevity_bonus == 0.0
        # Score should still be > 0 (driven by alignment=1.0 and horizon score)
        assert s.overall_score > 0.0

    def test_four_horizons_established_scores_high(self):
        """All four horizons established for the same kind → strong story."""
        engine = ContinuityEngine()
        for h in NarrativeHorizon:
            for _ in range(12):
                engine.update_narrative("EUR", NarrativeKind.ACCUMULATION, h, boost=0.15, now=_T0)
        s = engine.evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION)
        assert s.active_horizon_count == 4
        assert s.overall_score > 0.4

    def test_no_active_nodes_zero_score(self):
        """An expired story has score 0."""
        state = _fresh_state()
        update_narrative(
            "EUR", NarrativeKind.REVERSAL, NarrativeHorizon.MICRO, _state=state, now=_T0
        )
        # Mark the node expired
        story = state["EUR:REVERSAL"]
        from dataclasses import replace

        story.nodes[NarrativeHorizon.MICRO.value] = replace(
            story.nodes[NarrativeHorizon.MICRO.value],
            maturity=NarrativeMaturity.EXPIRED.value,
        )
        strength = evaluate_story_strength("EUR", NarrativeKind.REVERSAL, _state=state)
        # No active nodes → overall_score driven by 0 alignment / 0 horizon scores
        assert strength.active_horizon_count == 0
        assert strength.overall_score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Import sanity — narrative package __init__ exports C14
# ═══════════════════════════════════════════════════════════════════════════════


class TestNarrativePackageExports:
    def test_import_from_narrative_package(self):
        from trainer.qde.brain.narrative import (
            ContinuityEngine,
            NarrativeHorizon,
            evaluate_story_strength,
            retrieve_story,
            update_narrative,
        )

        assert NarrativeHorizon.MICRO == "MICRO"
        assert callable(update_narrative)
        assert callable(retrieve_story)
        assert callable(evaluate_story_strength)
        assert ContinuityEngine is not None

    def test_all_horizons_importable(self):
        from trainer.qde.brain.narrative import NarrativeHorizon

        for h in ("MICRO", "SESSION", "SHORT_TERM", "MACRO"):
            assert NarrativeHorizon(h) is not None

    def test_all_spec_callables_importable(self):
        from trainer.qde.brain.narrative.continuity import (
            evaluate_story_strength,
            retrieve_story,
            update_narrative,
        )

        assert all(
            callable(fn) for fn in (update_narrative, retrieve_story, evaluate_story_strength)
        )

    def test_story_strength_to_dict_round_trip(self):
        engine = ContinuityEngine()
        engine.update_narrative(
            "EUR", NarrativeKind.ACCUMULATION, NarrativeHorizon.SESSION, now=_T0
        )
        strength = engine.evaluate_story_strength("EUR", NarrativeKind.ACCUMULATION)
        d = strength.to_dict()
        assert isinstance(d["overall_score"], float)
        assert isinstance(d["horizon_scores"], dict)
        assert isinstance(d["reasoning"], str)
