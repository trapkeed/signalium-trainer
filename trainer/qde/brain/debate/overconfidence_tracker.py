"""QDE Brain — overconfidence tracker (Phase C5).

Detects and records **overconfidence events**: moments when the QDE's
belief is stronger than the quality of the underlying debate justifies.

Three trigger conditions, each with a distinct diagnostic value:

``LOW_MARGIN``
    Conviction ≥ ``HIGH_CONVICTION_THRESHOLD`` but the bull/bear debate
    margin is below ``DEFAULT_DOMINANCE_MARGIN``.  The brain is very
    confident even though neither thesis clearly dominates — a sign that
    conviction is running ahead of evidence.

``POOR_EVIDENCE_BALANCE``
    Conviction ≥ ``HIGH_CONVICTION_THRESHOLD`` but the dominant thesis
    has more *conflicting* evidence than *supporting* evidence — the
    winning argument is opposed by its own evidence and the brain is
    ignoring that.

``CONVICTION_CONFIDENCE_SPLIT``
    The ``ConvictionProfile`` is ``HIGH_CONFIDENCE_LOW_CONVICTION`` —
    the model's confidence estimate is high but the setup itself is
    poorly formed (shallow reinforcement, few evidence items, freshly
    created).  High confidence in a thin setup is a classic
    overconfidence pattern.

Design constraints
------------------
* Only reads ``DebateVerdict`` and ``ConvictionScore`` — never reads or
  writes ``Hypothesis`` fields, never places or blocks trades.
* ``to_snapshot()`` / ``load_snapshot()`` for all stateful objects.
* No ``QDEDecision`` imports.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.conviction_engine import ConvictionProfile, ConvictionScore
from trainer.qde.brain.debate_engine import DEFAULT_DOMINANCE_MARGIN, DebateVerdict
from trainer.qde.brain.thesis_manager import Thesis

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

HIGH_CONVICTION_THRESHOLD = 0.70  # conviction score above which an event can trigger
POOR_BALANCE_THRESHOLD = 0.50  # evidence balance below this → "poor"
MAX_EVENTS_PER_INSTRUMENT = 200
MAX_EVENTS_PER_INSTRUMENT_DEQUE = 200


# -------------------------------------------------------------------
# Event dataclass
# -------------------------------------------------------------------


@dataclass
class OverconfidenceEvent:
    """One recorded overconfidence episode.

    ``trigger``
        One of "LOW_MARGIN", "POOR_EVIDENCE_BALANCE",
        "CONVICTION_CONFIDENCE_SPLIT".

    ``resolved``
        Set to True (with ``resolution_outcome``) when the trade that
        followed this debate has a known outcome.  The resolution step
        is optional — the tracker records events even if resolution
        never arrives (e.g. no trade was taken).
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    hypothesis_id: str = ""
    instrument: str = ""
    opportunity_id: str = ""
    trigger: str = "LOW_MARGIN"
    conviction_at_event: float = 0.0
    confidence_at_event: float = 0.0
    debate_margin: float = 0.0
    bull_strength: float = 0.0
    bear_strength: float = 0.0
    evidence_balance: float = 0.0  # (dominant_supporting - dominant_conflicting) normalised
    profile: str = ConvictionProfile.MIXED.value
    resolved: bool = False
    resolution_outcome: str | None = None  # "WIN", "LOSS", "SKIPPED", etc.
    created_at: float = field(default_factory=time.time)
    resolved_at: float | None = None

    def resolve(self, outcome: str, now: float | None = None) -> None:
        if self.resolved:
            return
        self.resolved = True
        self.resolution_outcome = outcome
        self.resolved_at = now if now is not None else time.time()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "hypothesis_id": self.hypothesis_id,
            "instrument": self.instrument,
            "opportunity_id": self.opportunity_id,
            "trigger": self.trigger,
            "conviction_at_event": round(self.conviction_at_event, 4),
            "confidence_at_event": round(self.confidence_at_event, 4),
            "debate_margin": round(self.debate_margin, 4),
            "bull_strength": round(self.bull_strength, 4),
            "bear_strength": round(self.bear_strength, 4),
            "evidence_balance": round(self.evidence_balance, 4),
            "profile": self.profile,
            "resolved": self.resolved,
            "resolution_outcome": self.resolution_outcome,
            "created_at": round(self.created_at, 3),
            "resolved_at": round(self.resolved_at, 3) if self.resolved_at else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OverconfidenceEvent":
        evt = cls(
            event_id=d.get("event_id", str(uuid.uuid4())[:12]),
            hypothesis_id=d.get("hypothesis_id", ""),
            instrument=d.get("instrument", ""),
            opportunity_id=d.get("opportunity_id", ""),
            trigger=d.get("trigger", "LOW_MARGIN"),
            conviction_at_event=d.get("conviction_at_event", 0.0),
            confidence_at_event=d.get("confidence_at_event", 0.0),
            debate_margin=d.get("debate_margin", 0.0),
            bull_strength=d.get("bull_strength", 0.0),
            bear_strength=d.get("bear_strength", 0.0),
            evidence_balance=d.get("evidence_balance", 0.0),
            profile=d.get("profile", ConvictionProfile.MIXED.value),
            resolved=d.get("resolved", False),
            resolution_outcome=d.get("resolution_outcome"),
            created_at=d.get("created_at", 0.0),
            resolved_at=d.get("resolved_at"),
        )
        return evt


# -------------------------------------------------------------------
# Tracker
# -------------------------------------------------------------------


def _dominant_evidence_balance(
    verdict: DebateVerdict,
    bull_thesis: Thesis,
    bear_thesis: Thesis,
) -> float:
    """Normalised evidence balance for the dominant thesis.
    Positive = more supporting than conflicting; negative = the opposite.
    Returns 0.0 if there is no dominant side.
    """
    if verdict.dominant_side is None:
        return 0.0
    from trainer.qde.brain.brain_types import BeliefDirection

    thesis = bull_thesis if verdict.dominant_side == BeliefDirection.BULLISH else bear_thesis
    sup = len(thesis.supporting_evidence)
    conf = len(thesis.conflicting_evidence)
    total = sup + conf
    if total == 0:
        return 0.0
    return (sup - conf) / total  # range [-1, +1]


class OverconfidenceTracker:
    """Record overconfidence events and maintain per-instrument statistics.

    Usage::

        tracker = OverconfidenceTracker()

        event = tracker.check_and_record(verdict, conviction, bull_thesis, bear_thesis, now)
        if event:
            print(f"Overconfidence: {event.trigger} for {event.instrument}")

        tracker.resolve_event("evt-id", "LOSS", now)
        rate = tracker.overconfidence_rate("EURUSD")
    """

    def __init__(
        self,
        high_conviction_threshold: float = HIGH_CONVICTION_THRESHOLD,
        dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
        max_events: int = MAX_EVENTS_PER_INSTRUMENT_DEQUE,
    ) -> None:
        self._threshold = high_conviction_threshold
        self._dom_margin = dominance_margin
        self._max_events = max_events
        # instrument → deque of events (most recent at right)
        self._events: dict[str, deque[OverconfidenceEvent]] = defaultdict(
            lambda: deque(maxlen=self._max_events)
        )
        # event_id → event (for resolve look-up)
        self._index: dict[str, OverconfidenceEvent] = {}

    # ── Primary API ───────────────────────────────────────────────────

    def check_and_record(
        self,
        verdict: DebateVerdict,
        conviction: ConvictionScore,
        bull_thesis: Thesis,
        bear_thesis: Thesis,
        now: float | None = None,
    ) -> OverconfidenceEvent | None:
        """Evaluate the current debate for overconfidence signals and,
        if found, record the first triggered event.

        Returns the recorded event or None (no overconfidence detected).
        """
        now = now if now is not None else time.time()
        trigger = self._classify(verdict, conviction, bull_thesis, bear_thesis)
        if trigger is None:
            return None

        ev_balance = _dominant_evidence_balance(verdict, bull_thesis, bear_thesis)
        event = OverconfidenceEvent(
            hypothesis_id=conviction.hypothesis_id,
            instrument=verdict.instrument,
            opportunity_id=verdict.opportunity_id,
            trigger=trigger,
            conviction_at_event=conviction.conviction,
            confidence_at_event=conviction.confidence,
            debate_margin=verdict.margin,
            bull_strength=verdict.bull_strength,
            bear_strength=verdict.bear_strength,
            evidence_balance=ev_balance,
            profile=conviction.profile.value,
            created_at=now,
        )
        self._record(event)
        logger.debug(
            f"[OverconfidenceTracker] {trigger} on {verdict.instrument} "
            f"opp={verdict.opportunity_id} conviction={conviction.conviction:.2f} "
            f"margin={verdict.margin:.2f}"
        )
        return event

    def resolve_event(
        self,
        event_id: str,
        outcome: str,
        now: float | None = None,
    ) -> bool:
        """Mark an event as resolved with a known trade outcome."""
        event = self._index.get(event_id)
        if event is None:
            return False
        event.resolve(outcome, now=now)
        return True

    # ── Query ─────────────────────────────────────────────────────────

    def get_events(
        self,
        instrument: str,
        n: int = 20,
        resolved: bool | None = None,
    ) -> list[OverconfidenceEvent]:
        """Return up to ``n`` events for ``instrument``, newest first.

        ``resolved=True`` → only resolved events.
        ``resolved=False`` → only open events.
        ``resolved=None`` → all events.
        """
        all_evts = list(self._events.get(instrument, []))
        all_evts.reverse()  # newest first
        if resolved is not None:
            all_evts = [e for e in all_evts if e.resolved == resolved]
        return all_evts[:n]

    def overconfidence_rate(self, instrument: str, window: int = 50) -> float:
        """Fraction of recent debates that triggered an overconfidence event.

        ``window`` is the look-back event count.  Returns 0.0 if no events
        have been recorded (not an error — the system may not have had
        enough activity yet).
        """
        evts = list(self._events.get(instrument, []))
        if not evts:
            return 0.0
        sample = evts[-window:]
        return len(sample) / max(1, window)  # events / total debates approximated

    def trigger_breakdown(self, instrument: str) -> dict[str, int]:
        """Count of events by trigger type for ``instrument``."""
        counts: dict[str, int] = {}
        for evt in self._events.get(instrument, []):
            counts[evt.trigger] = counts.get(evt.trigger, 0) + 1
        return counts

    def resolution_summary(self, instrument: str) -> dict:
        """Summary of resolved event outcomes for ``instrument``."""
        resolved = [e for e in self._events.get(instrument, []) if e.resolved]
        outcomes: dict[str, int] = {}
        for evt in resolved:
            key = evt.resolution_outcome or "UNKNOWN"
            outcomes[key] = outcomes.get(key, 0) + 1
        return {
            "total_resolved": len(resolved),
            "outcome_counts": outcomes,
        }

    # ── Persistence ───────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "events": {instr: [e.to_dict() for e in deq] for instr, deq in self._events.items()}
        }

    def load_snapshot(self, data: dict) -> None:
        self._events.clear()
        self._index.clear()
        for instr, evts in data.get("events", {}).items():
            for d in evts:
                try:
                    evt = OverconfidenceEvent.from_dict(d)
                    self._events[instr].append(evt)
                    self._index[evt.event_id] = evt
                except Exception:
                    pass

    # ── Internal ──────────────────────────────────────────────────────

    def _classify(
        self,
        verdict: DebateVerdict,
        conviction: ConvictionScore,
        bull_thesis: Thesis,
        bear_thesis: Thesis,
    ) -> str | None:
        """Return the first matching trigger or None."""
        is_high_conviction = conviction.conviction >= self._threshold

        # 1. High conviction despite inconclusive margin
        if is_high_conviction and verdict.margin < self._dom_margin:
            return "LOW_MARGIN"

        # 2. High conviction with poor dominant-thesis evidence balance
        if is_high_conviction:
            ev_balance = _dominant_evidence_balance(verdict, bull_thesis, bear_thesis)
            if ev_balance < 0.0:  # more conflicting than supporting
                return "POOR_EVIDENCE_BALANCE"

        # 3. Conviction/confidence profile divergence
        if conviction.profile == ConvictionProfile.HIGH_CONFIDENCE_LOW_CONVICTION:
            return "CONVICTION_CONFIDENCE_SPLIT"

        return None

    def _record(self, event: OverconfidenceEvent) -> None:
        self._events[event.instrument].append(event)
        self._index[event.event_id] = event


__all__ = [
    "OverconfidenceEvent",
    "OverconfidenceTracker",
    "HIGH_CONVICTION_THRESHOLD",
]
