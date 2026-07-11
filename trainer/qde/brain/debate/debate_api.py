"""QDE Brain — debate API (Phase C5).

Top-level façade for the internal debate engine, exposing the three
spec-required methods:

    ``create_debate(instrument, opportunity_id, now)``
        Orchestrates a full debate cycle:
          1. Evaluates bull vs. bear via ``DebateEngine.evaluate()``
             (reads synced theses from the opportunity).
          2. Computes conviction via ``compute_conviction()`` (delegates).
          3. Measures debate quality via ``measure_quality()``.
          4. Checks for overconfidence via ``OverconfidenceTracker``.
          5. Checks for ignored evidence via ``IgnoredEvidenceTracker``.
          6. Builds and returns a ``DebateRecord``.

    ``calculate_conviction(hypothesis, now)``
        Thin, explicit delegate to ``compute_conviction()``.  Exists so
        callers have one stable import point that won't change if the
        underlying function signature evolves.

    ``store_debate(record, now)``
        Persists a ``DebateRecord`` to C2 episodic memory (via
        ``DebateMemoryBridge``) and logs to the working-memory snapshot
        that ``DebateEngine`` already maintains.

What this module does NOT do
----------------------------
* Never reimplements ``ThesisManager.sync()``, ``DebateEngine.evaluate()``,
  or ``compute_conviction()`` — those remain the single source of truth.
* Never imports or references ``QDEDecision``.
* Never places or blocks trades.

Dependency direction
--------------------
``DebateAPI`` → ``DebateEngine`` (for evaluate / working-memory log)
             → ``HypothesisEngine`` (to fetch the hypothesis for conviction)
             → ``OverconfidenceTracker``, ``IgnoredEvidenceTracker``,
               ``DebateMemoryBridge`` (all owned by the API)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.conviction_engine import ConvictionScore, compute_conviction
from trainer.qde.brain.debate.debate_quality import DebateQualityScore, measure_quality
from trainer.qde.brain.debate.debate_store import (
    DebateMemoryBridge,
    DebateRecord,
    make_debate_record,
)
from trainer.qde.brain.debate.ignored_evidence_tracker import IgnoredEvidenceTracker
from trainer.qde.brain.debate.overconfidence_tracker import OverconfidenceTracker
from trainer.qde.brain.debate_engine import DebateEngine, DebateVerdict

if TYPE_CHECKING:
    from trainer.qde.brain.hypothesis_engine import HypothesisEngine


# -------------------------------------------------------------------
# DebateAPI
# -------------------------------------------------------------------


class DebateAPI:
    """Phase C5 public API: ``create_debate``, ``calculate_conviction``,
    ``store_debate``.

    One instance is shared across all instruments.

    Parameters
    ----------
    debate_engine:
        The existing ``DebateEngine`` instance (required — all verdict
        evaluation delegates to it).
    hypothesis_engine:
        The existing ``HypothesisEngine`` (used to locate the ``Hypothesis``
        object for conviction calculation).
    overconfidence_tracker:
        ``OverconfidenceTracker`` instance.  None → a fresh one is created.
    ignored_evidence_tracker:
        ``IgnoredEvidenceTracker`` instance.  None → a fresh one is created.
    memory_bridge:
        ``DebateMemoryBridge`` instance.  None → a bridge with no C2
        episodic backend is created (local deque still works).
    """

    def __init__(
        self,
        debate_engine: DebateEngine,
        hypothesis_engine: "HypothesisEngine",
        overconfidence_tracker: OverconfidenceTracker | None = None,
        ignored_evidence_tracker: IgnoredEvidenceTracker | None = None,
        memory_bridge: DebateMemoryBridge | None = None,
    ) -> None:
        self._engine = debate_engine
        self._hyp_engine = hypothesis_engine
        self._oc_tracker = overconfidence_tracker or OverconfidenceTracker()
        self._ie_tracker = ignored_evidence_tracker or IgnoredEvidenceTracker()
        self._bridge = memory_bridge or DebateMemoryBridge()

    # ── Primary API ───────────────────────────────────────────────────

    def create_debate(
        self,
        instrument: str,
        opportunity_id: str,
        now: float | None = None,
    ) -> DebateRecord | None:
        """Run a complete debate cycle for one opportunity and return the
        result as a ``DebateRecord``.

        Returns None if the opportunity is not tracked (same convention
        as ``DebateEngine.evaluate()`` — missing id → None, not error).

        Steps (each delegates; nothing is reimplemented):
          1. ``DebateEngine.evaluate()`` → ``DebateVerdict``
             (also syncs theses on the opportunity via ThesisManager).
          2. Locate ``Hypothesis`` via ``HypothesisEngine._locate()``.
          3. ``compute_conviction(hypothesis)`` → ``ConvictionScore``.
          4. ``measure_quality(verdict, conviction, bull, bear)``
             → ``DebateQualityScore``.
          5. ``OverconfidenceTracker.check_and_record()``.
          6. ``IgnoredEvidenceTracker.check_and_record()``.
          7. ``make_debate_record(...)`` → ``DebateRecord``.
        """
        now = now if now is not None else time.time()

        # ── Step 1: evaluate (also syncs theses) ─────────────────────
        verdict: DebateVerdict | None = self._engine.evaluate(instrument, opportunity_id, now=now)
        if verdict is None:
            logger.debug(
                f"[DebateAPI] create_debate: opportunity {opportunity_id} "
                f"not found on {instrument}"
            )
            return None

        # ── Step 2: locate hypothesis ─────────────────────────────────
        brain = self._engine._brain
        state = brain.get_state(instrument)
        opp = state.opportunities.get(opportunity_id)
        if opp is None:
            return None

        hyp: Hypothesis | None = state.hypotheses.get(opp.hypothesis_id)

        # ── Step 3: conviction ────────────────────────────────────────
        conviction: ConvictionScore
        if hyp is not None:
            conviction = compute_conviction(hyp, now=now)
        else:
            # Hypothesis has been removed from state (e.g. aged out) —
            # build a minimal zero-conviction score so the debate can
            # still be recorded without crashing.
            from trainer.qde.brain.conviction_engine import (
                ConvictionComponents,
                ConvictionProfile,
            )

            conviction = ConvictionScore(
                hypothesis_id=opp.hypothesis_id,
                instrument=instrument,
                confidence=0.0,
                conviction=0.0,
                components=ConvictionComponents(),
                profile=ConvictionProfile.ALIGNED_LOW,
                explanation="Hypothesis not found at debate time.",
                computed_at=now,
            )

        # ── Step 4: quality ───────────────────────────────────────────
        quality: DebateQualityScore = measure_quality(
            verdict=verdict,
            conviction=conviction,
            bull_thesis=opp.bull_thesis,
            bear_thesis=opp.bear_thesis,
            debate_id="",  # filled in below after record creation
            now=now,
        )

        # ── Step 5: overconfidence ────────────────────────────────────
        oc_event = self._oc_tracker.check_and_record(
            verdict=verdict,
            conviction=conviction,
            bull_thesis=opp.bull_thesis,
            bear_thesis=opp.bear_thesis,
            now=now,
        )

        # ── Step 6: ignored evidence ──────────────────────────────────
        ie_record = self._ie_tracker.check_and_record(
            verdict=verdict,
            conviction=conviction,
            bull_thesis=opp.bull_thesis,
            bear_thesis=opp.bear_thesis,
            debate_quality=quality.overall_score,
            now=now,
        )

        # ── Step 7: assemble DebateRecord ─────────────────────────────
        record = make_debate_record(
            verdict=verdict,
            conviction=conviction,
            quality=quality,
            bull_thesis_dict=opp.bull_thesis.to_dict(),
            bear_thesis_dict=opp.bear_thesis.to_dict(),
            overconfidence_event=oc_event,
            ignored_evidence=ie_record,
            now=now,
        )
        # Backfill the debate_id into the quality score dict
        record.quality["debate_id"] = record.debate_id

        logger.debug(
            f"[DebateAPI] create_debate {instrument}/{opportunity_id}: "
            f"conclusive={verdict.conclusive} "
            f"quality={quality.grade} "
            f"conviction={conviction.conviction:.2f} "
            f"oc={'Y' if oc_event else 'N'} "
            f"ie={'Y' if ie_record else 'N'}"
        )
        return record

    def calculate_conviction(
        self,
        hypothesis: Hypothesis,
        now: float | None = None,
    ) -> ConvictionScore:
        """Thin delegate to ``compute_conviction()``.

        This is the stable entry-point callers should use so they have
        one import location for conviction calculation regardless of
        whether the underlying function is reorganised in the future.
        """
        return compute_conviction(hypothesis, now=now)

    def store_debate(
        self,
        record: DebateRecord,
        now: float | None = None,
    ) -> str:
        """Persist a ``DebateRecord`` to episodic memory (if available)
        and the local in-process deque.

        Returns the episode_id on successful episodic storage, or
        ``record.debate_id`` when only the local deque was used.
        """
        return self._bridge.store_debate(record, now=now)

    # ── Convenience helpers ───────────────────────────────────────────

    def create_and_store(
        self,
        instrument: str,
        opportunity_id: str,
        now: float | None = None,
    ) -> DebateRecord | None:
        """Create a debate and immediately persist it.

        Convenience combination of ``create_debate`` + ``store_debate``
        for the common call-site pattern.
        """
        record = self.create_debate(instrument, opportunity_id, now=now)
        if record is not None:
            self.store_debate(record, now=now)
        return record

    def overconfidence_rate(self, instrument: str, window: int = 50) -> float:
        """Fraction of recent debates that triggered an overconfidence event."""
        return self._oc_tracker.overconfidence_rate(instrument, window=window)

    def common_ignored_signals(self, instrument: str, top_n: int = 10) -> dict[str, int]:
        """Most frequently ignored evidence strings."""
        return self._ie_tracker.common_ignored_signals(instrument, top_n=top_n)

    def recent_debates(self, instrument: str, n: int = 20) -> list[DebateRecord]:
        """Recent stored debates from the local deque, newest first."""
        return self._bridge.recent_debates(instrument, n=n)

    def overconfidence_events(
        self,
        instrument: str,
        n: int = 20,
    ) -> list[dict]:
        """Recent overconfidence events as dicts, newest first."""
        return [e.to_dict() for e in self._oc_tracker.get_events(instrument, n=n)]

    def ignored_evidence_records(
        self,
        instrument: str,
        n: int = 20,
    ) -> list[dict]:
        """Recent ignored-evidence records as dicts, newest first."""
        return [r.to_dict() for r in self._ie_tracker.get_records(instrument, n=n)]

    def resolve_overconfidence(
        self,
        event_id: str,
        outcome: str,
        now: float | None = None,
    ) -> bool:
        """Mark an overconfidence event as resolved with a trade outcome."""
        return self._oc_tracker.resolve_event(event_id, outcome, now=now)

    # ── Persistence ───────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        """Serialise all stateful sub-components."""
        return {
            "overconfidence_tracker": self._oc_tracker.to_snapshot(),
            "ignored_evidence_tracker": self._ie_tracker.to_snapshot(),
            "memory_bridge": self._bridge.to_snapshot(),
        }

    def load_snapshot(self, data: dict) -> None:
        """Restore previously-serialised state."""
        if "overconfidence_tracker" in data:
            self._oc_tracker.load_snapshot(data["overconfidence_tracker"])
        if "ignored_evidence_tracker" in data:
            self._ie_tracker.load_snapshot(data["ignored_evidence_tracker"])
        if "memory_bridge" in data:
            self._bridge.load_snapshot(data["memory_bridge"])

    # ── Sub-component access ──────────────────────────────────────────

    @property
    def overconfidence_tracker(self) -> OverconfidenceTracker:
        return self._oc_tracker

    @property
    def ignored_evidence_tracker(self) -> IgnoredEvidenceTracker:
        return self._ie_tracker

    @property
    def memory_bridge(self) -> DebateMemoryBridge:
        return self._bridge


__all__ = ["DebateAPI"]
