"""C10 — Attention evolution memory.

Stores a bounded, per-symbol history of ``AttentionSnapshot`` objects and
``AttentionEvolutionRecord`` tier-transition events so the QDE can answer
questions like:

  * "When did EURUSD enter CRITICAL_ATTENTION?"
  * "How long has GBPUSD been in HIGH_ATTENTION?"
  * "What was the dominant signal when XAUUSD last escalated?"
  * "Show me EURUSD's attention evolution over the last hour."

Design constraints
------------------
* Standalone — no import from ``AttentionEngine`` or ``AttentionScoreAllocator``.
  Accepts finished ``AttentionSnapshot`` objects from the ``AttentionAPI``.
* Thread-safe via per-instrument collections.Deque (GIL-safe appends).
  No explicit lock needed for single-writer/multiple-reader patterns.
* Bounded: configurable per-instrument snapshot depth (default 500) and
  transition depth (default 100).  Oldest entries are evicted automatically.
* Only writes a tier-transition record when the tier *changes* — keeps the
  evolution log compact even for symbols ticking every second.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from trainer.qde.brain.attention.attention_types import (
    AttentionEvolutionRecord,
    AttentionSnapshot,
    AttentionTier,
    tier_from_composite,
)

DEFAULT_SNAPSHOT_DEPTH = 500
DEFAULT_TRANSITION_DEPTH = 100


# ─────────────────────────────────────────────────────────────────────────────
# Per-symbol state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _SymbolAttentionState:
    snapshots:   deque     # deque[AttentionSnapshot]
    transitions: deque     # deque[AttentionEvolutionRecord]
    last_tier:   AttentionTier | None = None


# ─────────────────────────────────────────────────────────────────────────────
# AttentionMemory
# ─────────────────────────────────────────────────────────────────────────────


class AttentionMemory:
    """Bounded per-symbol history of attention snapshots and tier transitions.

    Usage::

        memory = AttentionMemory()
        snapshot = attention_api.calculate_attention(observation)
        event = memory.record(snapshot)   # returns transition event or None
    """

    def __init__(
        self,
        max_snapshots: int = DEFAULT_SNAPSHOT_DEPTH,
        max_transitions: int = DEFAULT_TRANSITION_DEPTH,
    ) -> None:
        self._max_snapshots = max(1, max_snapshots)
        self._max_transitions = max(1, max_transitions)
        self._states: dict[str, _SymbolAttentionState] = {}

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, snapshot: AttentionSnapshot) -> AttentionEvolutionRecord | None:
        """Store a snapshot; return a tier-transition record if the tier changed.

        A tier-transition record is produced on:
          * The very first snapshot for a symbol (``from_tier=None``).
          * Any subsequent snapshot where ``tier != last_tier``.

        Returns ``None`` when the tier is unchanged (most ticks).
        """
        instrument = snapshot.instrument
        state = self._states.setdefault(
            instrument,
            _SymbolAttentionState(
                snapshots=deque(maxlen=self._max_snapshots),
                transitions=deque(maxlen=self._max_transitions),
            ),
        )

        state.snapshots.append(snapshot)

        # Detect tier transition.
        if state.last_tier is None or state.last_tier != snapshot.tier:
            dominant = _dominant_signal(snapshot.signals)
            record = AttentionEvolutionRecord(
                instrument=instrument,
                from_tier=state.last_tier,
                to_tier=snapshot.tier,
                composite_at_transition=snapshot.composite,
                transitioned_at=snapshot.captured_at,
                dominant_signal=dominant,
            )
            state.transitions.append(record)
            state.last_tier = snapshot.tier
            return record

        return None

    # ── Queries ───────────────────────────────────────────────────────────────

    def history(self, instrument: str, n: int = 50) -> list[AttentionSnapshot]:
        """Most recent ``n`` snapshots for ``instrument``, newest last."""
        state = self._states.get(instrument)
        if state is None:
            return []
        snapshots = list(state.snapshots)
        return snapshots[-n:] if n < len(snapshots) else snapshots

    def tier_transitions(
        self, instrument: str, n: int = 20
    ) -> list[AttentionEvolutionRecord]:
        """Most recent ``n`` tier-change events for ``instrument``, newest last."""
        state = self._states.get(instrument)
        if state is None:
            return []
        transitions = list(state.transitions)
        return transitions[-n:] if n < len(transitions) else transitions

    def current_tier(self, instrument: str) -> AttentionTier | None:
        """The most recently recorded tier for ``instrument``; ``None`` if unseen."""
        state = self._states.get(instrument)
        return state.last_tier if state else None

    def current_snapshot(self, instrument: str) -> AttentionSnapshot | None:
        """The most recently stored snapshot for ``instrument``."""
        state = self._states.get(instrument)
        if not state or not state.snapshots:
            return None
        return state.snapshots[-1]

    def time_in_current_tier(self, instrument: str, now: float | None = None) -> float:
        """Seconds the symbol has been in its current tier.

        Returns 0.0 if no transitions have been recorded yet.
        """
        now = now if now is not None else time.time()
        state = self._states.get(instrument)
        if not state or not state.transitions:
            return 0.0
        last_transition = state.transitions[-1]
        return max(0.0, now - last_transition.transitioned_at)

    def evolution_summary(self, instrument: str) -> dict:
        """Compact diagnostics dict for one symbol's attention evolution."""
        state = self._states.get(instrument)
        if state is None:
            return {"instrument": instrument, "tracked": False}

        last_snapshot = state.snapshots[-1] if state.snapshots else None
        last_transition = state.transitions[-1] if state.transitions else None
        total_snapshots = len(state.snapshots)
        total_transitions = len(state.transitions)

        tier_counts: dict[str, int] = {}
        for snap in state.snapshots:
            tier_counts[snap.tier.value] = tier_counts.get(snap.tier.value, 0) + 1

        return {
            "instrument": instrument,
            "tracked": True,
            "current_tier": state.last_tier.value if state.last_tier else None,
            "snapshots_stored": total_snapshots,
            "tier_transitions": total_transitions,
            "tier_distribution": tier_counts,
            "last_snapshot": last_snapshot.to_dict() if last_snapshot else None,
            "last_transition": last_transition.to_dict() if last_transition else None,
        }

    def all_instruments(self) -> list[str]:
        """All instruments currently tracked in this memory store."""
        return sorted(self._states)

    def drop(self, instrument: str) -> None:
        """Remove all stored state for ``instrument``.

        Called when a symbol is disabled/delisted.
        """
        self._states.pop(instrument, None)

    def clear(self) -> None:
        """Remove all state.  Primarily useful in tests."""
        self._states.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _dominant_signal(signals: dict) -> str:
    """Return the name of the signal with the highest value in ``signals``.

    Used to annotate tier-transition records with the factor that most
    contributed to the escalation.  Returns ``""`` if ``signals`` is empty.
    """
    if not signals:
        return ""
    return max(signals, key=lambda k: signals.get(k, 0.0))


__all__ = [
    "AttentionMemory",
    "DEFAULT_SNAPSHOT_DEPTH",
    "DEFAULT_TRANSITION_DEPTH",
]
