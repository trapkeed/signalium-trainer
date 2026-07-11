"""QDE Brain — ignored evidence tracker (Phase C5).

Tracks evidence that was available during a debate but not given enough
weight to actually affect the verdict.

"Ignored evidence" definition
-------------------------------
When the dominant thesis has non-empty ``conflicting_evidence`` AND
conviction is above ``IGNORED_EVIDENCE_MIN_CONVICTION``, the system is
effectively overriding its own contradictory signals.  Each item in
``dominant_thesis.conflicting_evidence`` is an "ignored signal" — it
was collected and stored on the thesis by ``ThesisManager.sync()`` but
did not cause conviction to drop.

This matters because:

1. **Pattern learning** — repeated ignored signals that precede bad
   trades indicate a blind spot.  Surfacing them to the reflection
   engine and the failure tracker allows the brain to learn "when you
   see evidence X and still proceed, outcomes are poor".

2. **Meta-awareness** — ``IgnoredEvidenceTracker.common_ignored_signals``
   shows which evidence strings are chronically dismissed, which informs
   hypothesis formation upstream.

Design constraints
------------------
* Read-only against ``Thesis`` / ``DebateVerdict`` / ``ConvictionScore``.
* No storage of its own ``Hypothesis`` mutation or trade placement.
* No ``QDEDecision`` imports.
* ``to_snapshot()`` / ``load_snapshot()`` dict pattern.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.conviction_engine import ConvictionScore
from trainer.qde.brain.debate_engine import DebateVerdict
from trainer.qde.brain.thesis_manager import Thesis

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

IGNORED_EVIDENCE_MIN_CONVICTION = 0.55  # conviction floor for "should have cared"
MAX_RECORDS_PER_INSTRUMENT = 200
MAX_IGNORED_PER_RECORD = 20  # cap how many signals we store per record


# -------------------------------------------------------------------
# Record dataclass
# -------------------------------------------------------------------


@dataclass
class IgnoredEvidenceRecord:
    """One instance of ignored contradicting evidence.

    ``ignored_signals``
        The conflicting_evidence items from the dominant thesis that
        were present but not weighted heavily enough to lower conviction
        below the detection threshold.

    ``conviction_at_time``
        Conviction score at the time of detection.

    ``debate_margin``
        The winning margin of the debate — shows how "confident" the
        system was while ignoring these signals.

    ``dominant_side``
        Which thesis (BULLISH / BEARISH) was the winner — so the
        ignored signals are the *counter-evidence against the winner*.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    hypothesis_id: str = ""
    instrument: str = ""
    opportunity_id: str = ""
    dominant_side: str = "NEUTRAL"
    ignored_signals: list[str] = field(default_factory=list)
    conviction_at_time: float = 0.0
    confidence_at_time: float = 0.0
    debate_margin: float = 0.0
    debate_quality: float = 0.0  # overall_score from DebateQualityScore if available
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "hypothesis_id": self.hypothesis_id,
            "instrument": self.instrument,
            "opportunity_id": self.opportunity_id,
            "dominant_side": self.dominant_side,
            "ignored_signals": self.ignored_signals[:MAX_IGNORED_PER_RECORD],
            "conviction_at_time": round(self.conviction_at_time, 4),
            "confidence_at_time": round(self.confidence_at_time, 4),
            "debate_margin": round(self.debate_margin, 4),
            "debate_quality": round(self.debate_quality, 4),
            "created_at": round(self.created_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IgnoredEvidenceRecord":
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            hypothesis_id=d.get("hypothesis_id", ""),
            instrument=d.get("instrument", ""),
            opportunity_id=d.get("opportunity_id", ""),
            dominant_side=d.get("dominant_side", "NEUTRAL"),
            ignored_signals=d.get("ignored_signals", [])[:MAX_IGNORED_PER_RECORD],
            conviction_at_time=d.get("conviction_at_time", 0.0),
            confidence_at_time=d.get("confidence_at_time", 0.0),
            debate_margin=d.get("debate_margin", 0.0),
            debate_quality=d.get("debate_quality", 0.0),
            created_at=d.get("created_at", 0.0),
        )


# -------------------------------------------------------------------
# Tracker
# -------------------------------------------------------------------


class IgnoredEvidenceTracker:
    """Detect, record, and surface chronically ignored evidence signals.

    Usage::

        tracker = IgnoredEvidenceTracker()

        rec = tracker.check_and_record(verdict, conviction, bull_thesis, bear_thesis)
        if rec:
            print(f"Ignored {len(rec.ignored_signals)} signals on {rec.instrument}")

        commons = tracker.common_ignored_signals("EURUSD", top_n=5)
        # {"divergence detected": 12, "volume divergence": 7, ...}
    """

    def __init__(
        self,
        min_conviction: float = IGNORED_EVIDENCE_MIN_CONVICTION,
        max_records: int = MAX_RECORDS_PER_INSTRUMENT,
    ) -> None:
        self._min_conviction = min_conviction
        self._max_records = max_records
        # instrument → deque of records
        self._records: dict[str, deque[IgnoredEvidenceRecord]] = defaultdict(
            lambda: deque(maxlen=self._max_records)
        )

    # ── Primary API ───────────────────────────────────────────────────

    def check_and_record(
        self,
        verdict: DebateVerdict,
        conviction: ConvictionScore,
        bull_thesis: Thesis,
        bear_thesis: Thesis,
        debate_quality: float = 0.0,
        now: float | None = None,
    ) -> IgnoredEvidenceRecord | None:
        """Detect ignored evidence and record it if present.

        Returns a record if conviction is high enough and the dominant
        thesis has conflicting evidence items that were effectively
        overridden.  Returns None when there is nothing to record
        (no dominant side, conviction too low, or no conflicting
        evidence on the winning side).
        """
        now = now if now is not None else time.time()

        if conviction.conviction < self._min_conviction:
            return None
        if verdict.dominant_side is None:
            return None

        # Extract the dominant thesis
        from trainer.qde.brain.brain_types import BeliefDirection

        dominant = bull_thesis if verdict.dominant_side == BeliefDirection.BULLISH else bear_thesis

        ignored = list(dominant.conflicting_evidence)[:MAX_IGNORED_PER_RECORD]
        if not ignored:
            return None

        rec = IgnoredEvidenceRecord(
            hypothesis_id=conviction.hypothesis_id,
            instrument=verdict.instrument,
            opportunity_id=verdict.opportunity_id,
            dominant_side=verdict.dominant_side.value,
            ignored_signals=ignored,
            conviction_at_time=conviction.conviction,
            confidence_at_time=conviction.confidence,
            debate_margin=verdict.margin,
            debate_quality=debate_quality,
            created_at=now,
        )
        self._records[verdict.instrument].append(rec)
        logger.debug(
            f"[IgnoredEvidenceTracker] {len(ignored)} ignored signal(s) on "
            f"{verdict.instrument} conviction={conviction.conviction:.2f}"
        )
        return rec

    # ── Query ─────────────────────────────────────────────────────────

    def get_records(
        self,
        instrument: str,
        n: int = 20,
    ) -> list[IgnoredEvidenceRecord]:
        """Most-recent records for ``instrument``, newest first."""
        all_recs = list(self._records.get(instrument, []))
        all_recs.reverse()
        return all_recs[:n]

    def common_ignored_signals(
        self,
        instrument: str,
        top_n: int = 10,
    ) -> dict[str, int]:
        """Most frequently ignored evidence strings for ``instrument``.

        Returns a dict of ``{signal_text: count}`` sorted by count
        descending, capped at ``top_n`` entries.
        """
        counter: Counter[str] = Counter()
        for rec in self._records.get(instrument, []):
            counter.update(rec.ignored_signals)
        return dict(counter.most_common(top_n))

    def ignored_rate(
        self,
        instrument: str,
        window: int = 50,
    ) -> float:
        """Fraction of recent records that contained at least one ignored
        signal.  Returns 0.0 if no records exist (not an error).
        """
        recs = list(self._records.get(instrument, []))
        if not recs:
            return 0.0
        sample = recs[-window:]
        with_signals = sum(1 for r in sample if r.ignored_signals)
        return with_signals / max(1, len(sample))

    def conviction_at_time_stats(self, instrument: str) -> dict:
        """Basic statistics on conviction-at-time for all records."""
        vals = [r.conviction_at_time for r in self._records.get(instrument, [])]
        if not vals:
            return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
        return {
            "count": len(vals),
            "mean": round(sum(vals) / len(vals), 4),
            "max": round(max(vals), 4),
            "min": round(min(vals), 4),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "records": {instr: [r.to_dict() for r in deq] for instr, deq in self._records.items()}
        }

    def load_snapshot(self, data: dict) -> None:
        self._records.clear()
        for instr, recs in data.get("records", {}).items():
            for d in recs:
                try:
                    rec = IgnoredEvidenceRecord.from_dict(d)
                    self._records[instr].append(rec)
                except Exception:
                    pass


__all__ = [
    "IgnoredEvidenceRecord",
    "IgnoredEvidenceTracker",
    "IGNORED_EVIDENCE_MIN_CONVICTION",
]
