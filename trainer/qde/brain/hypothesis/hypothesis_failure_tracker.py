"""QDE Brain — hypothesis failure tracker.

Records what actually happened when a hypothesis was invalidated, enabling
the system to learn from its failures rather than simply discarding them.

For each invalidated hypothesis the tracker captures:

  * ``failure_mode``         — WHY it failed (contradicted, regime shift,
                              aged out, superseded, or unknown).
  * ``market_direction_observed`` — what direction price actually moved
                              (the "ground truth" after the hypothesis was wrong).
  * ``winning_scenario_thesis`` — which competing scenario (if any) played out.
  * ``lesson``               — a derived natural-language lesson that can be
                              stored in long-term memory or surfaced to the
                              operator.

These records let the system:
  1. Compute per-instrument / per-direction failure rates.
  2. Identify the most common failure modes.
  3. Detect systematic biases (e.g. "BULLISH hypotheses in TREND_DOWN regimes
     fail 80% of the time — stop generating them").
  4. Generate lessons that feed back into hypothesis seed confidence adjustment
     via ``HypothesisMemoryBridge``.

No duplicate storage
--------------------
The existing ``HypothesisEngine._check_milestone`` already records
``HYPOTHESIS_INVALIDATED`` events to ``LongTermMemory`` via
``QDEBrain.record_experience``.  This module is complementary — it stores
*richer* failure records (failure mode, market direction, winning thesis) in
its own bounded deques so they can be queried independently of the main
long-term memory, without creating a parallel training loop.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Literal

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection

# ── Taxonomy ──────────────────────────────────────────────────────────────────

FailureMode = Literal[
    "CONTRADICTED",  # Direct contradicting evidence exceeded support
    "REGIME_SHIFT",  # Market regime changed, invalidating the thesis
    "AGED_OUT",  # Hypothesis expired without resolving
    "SUPERSEDED",  # A competing scenario was promoted / confirmed
    "STOP_HIT",  # Price hit stop-loss level that disproves the thesis
    "UNKNOWN",  # Caller did not provide enough context to classify
]

_MAX_RECORDS_PER_KEY = 200  # per (instrument, direction) history cap
_GLOBAL_MAX = 1000  # total records cap


# ── Record ────────────────────────────────────────────────────────────────────


@dataclass
class HypothesisFailureRecord:
    """Structured record of one hypothesis invalidation event.

    ``failure_mode``
        Categorical reason the hypothesis failed (see ``FailureMode``).

    ``market_direction_observed``
        Direction price actually moved after invalidation, if known.
        None = caller did not provide (e.g. hypothesis was aged out rather
        than explicitly disproven by a price move).

    ``winning_scenario_thesis``
        If the hypothesis was part of a ``HypothesisSet``, the thesis of the
        scenario that turned out to be correct (i.e. was promoted/confirmed).

    ``lesson``
        Auto-generated natural-language lesson derived from the failure.
        Suitable for storage in long-term memory or operator inspection.
    """

    failure_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    hypothesis_id: str = ""
    set_id: str | None = None
    instrument: str = ""
    thesis: str = ""
    direction: BeliefDirection = BeliefDirection.NEUTRAL
    formed_regime: str = ""
    confidence_at_failure: float = 0.0
    reinforcement_count: int = 0
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    invalidation_reason: str = ""
    failure_mode: FailureMode = "UNKNOWN"
    failed_at: float = field(default_factory=time.time)
    market_direction_observed: BeliefDirection | None = None
    winning_scenario_thesis: str | None = None
    lesson: str = ""

    def to_dict(self) -> dict:
        return {
            "failure_id": self.failure_id,
            "hypothesis_id": self.hypothesis_id,
            "set_id": self.set_id,
            "instrument": self.instrument,
            "thesis": self.thesis,
            "direction": self.direction.value,
            "formed_regime": self.formed_regime,
            "confidence_at_failure": round(self.confidence_at_failure, 4),
            "reinforcement_count": self.reinforcement_count,
            "supporting_evidence": self.supporting_evidence[-5:],
            "contradicting_evidence": self.contradicting_evidence[-5:],
            "invalidation_reason": self.invalidation_reason,
            "failure_mode": self.failure_mode,
            "failed_at": round(self.failed_at, 3),
            "market_direction_observed": (
                self.market_direction_observed.value if self.market_direction_observed else None
            ),
            "winning_scenario_thesis": self.winning_scenario_thesis,
            "lesson": self.lesson,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HypothesisFailureRecord:
        mdir = d.get("market_direction_observed")
        return cls(
            failure_id=d.get("failure_id", str(uuid.uuid4())[:12]),
            hypothesis_id=d.get("hypothesis_id", ""),
            set_id=d.get("set_id"),
            instrument=d.get("instrument", ""),
            thesis=d.get("thesis", ""),
            direction=BeliefDirection(d.get("direction", "NEUTRAL")),
            formed_regime=d.get("formed_regime", ""),
            confidence_at_failure=d.get("confidence_at_failure", 0.0),
            reinforcement_count=d.get("reinforcement_count", 0),
            supporting_evidence=d.get("supporting_evidence", []),
            contradicting_evidence=d.get("contradicting_evidence", []),
            invalidation_reason=d.get("invalidation_reason", ""),
            failure_mode=d.get("failure_mode", "UNKNOWN"),
            failed_at=d.get("failed_at", 0.0),
            market_direction_observed=BeliefDirection(mdir) if mdir else None,
            winning_scenario_thesis=d.get("winning_scenario_thesis"),
            lesson=d.get("lesson", ""),
        )


# ── Lesson derivation ─────────────────────────────────────────────────────────


def _derive_lesson(
    hyp: Hypothesis,
    failure_mode: FailureMode,
    market_direction: BeliefDirection | None,
    winning_thesis: str | None,
) -> str:
    """Derive a natural-language lesson from a failure record."""
    direction_label = hyp.direction.value.title()
    regime_clause = f" in {hyp.formed_regime} regime" if hyp.formed_regime else ""
    winning_clause = f" — '{winning_thesis}' played out instead" if winning_thesis else ""
    market_clause = (
        f" — market went {market_direction.value.lower()}"
        if market_direction and market_direction != hyp.direction
        else ""
    )

    mode_templates: dict[str, str] = {
        "CONTRADICTED": (
            f"{direction_label} thesis{regime_clause} was directly contradicted "
            f"after {hyp.reinforcement_count} reinforcement(s){market_clause}{winning_clause}. "
            f"Consider lower initial confidence for similar setups."
        ),
        "REGIME_SHIFT": (
            f"{direction_label} thesis{regime_clause} became invalid after a regime change. "
            f"Hypothesis confidence should decay faster when regime transitions are possible."
        ),
        "AGED_OUT": (
            f"{direction_label} thesis{regime_clause} never resolved within its time limit "
            f"(confidence {hyp.confidence:.0%}, {hyp.reinforcement_count} reinforcement(s)). "
            f"Setup did not materialize — entry timing may have been too early."
        ),
        "SUPERSEDED": (
            f"{direction_label} thesis{regime_clause} was superseded by a stronger competing "
            f"scenario{winning_clause}. "
            f"Probability allocation to this direction should be reduced in similar conditions."
        ),
        "STOP_HIT": (
            f"{direction_label} thesis{regime_clause} was stopped out{market_clause}. "
            f"Risk/reward or entry timing requires review for similar setups."
        ),
        "UNKNOWN": (
            f"{direction_label} thesis{regime_clause} was invalidated "
            f"(confidence {hyp.confidence:.0%}){market_clause}{winning_clause}."
        ),
    }
    return mode_templates.get(failure_mode, mode_templates["UNKNOWN"])


def _classify_failure_mode(
    reason: str,
    market_direction: BeliefDirection | None,
    hyp_direction: BeliefDirection,
) -> FailureMode:
    """Heuristic classification of failure mode from reason string."""
    reason_lower = reason.lower()
    if "regime" in reason_lower or "regime_shift" in reason_lower:
        return "REGIME_SHIFT"
    if "superseded" in reason_lower or "promoted" in reason_lower or "competing" in reason_lower:
        return "SUPERSEDED"
    if "stop" in reason_lower or "stopped" in reason_lower:
        return "STOP_HIT"
    if "expired" in reason_lower or "stale" in reason_lower or "aged" in reason_lower:
        return "AGED_OUT"
    if (
        market_direction
        and market_direction != hyp_direction
        and market_direction != BeliefDirection.NEUTRAL
    ):
        return "CONTRADICTED"
    return "UNKNOWN"


# ── Tracker ───────────────────────────────────────────────────────────────────


class HypothesisFailureTracker:
    """Records and queries hypothesis failure history.

    Maintains per-(instrument, direction) failure deques so queries like
    "what fraction of BULLISH hypotheses on EURUSD failed in the past N days"
    are O(N) rather than requiring a full-table scan.

    Also computes per-instrument failure rates and dominant failure modes,
    which ``HypothesisMemoryBridge`` uses to adjust seed confidence.
    """

    def __init__(
        self,
        max_per_key: int = _MAX_RECORDS_PER_KEY,
    ) -> None:
        self._max = max_per_key
        # (instrument, direction.value) → deque[HypothesisFailureRecord]
        self._records: dict[tuple[str, str], deque[HypothesisFailureRecord]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )
        self._by_id: dict[str, HypothesisFailureRecord] = {}

    # ── Write ─────────────────────────────────────────────────────────────

    def record_failure(
        self,
        hypothesis: Hypothesis,
        set_id: str | None = None,
        market_direction: BeliefDirection | None = None,
        winning_scenario_thesis: str | None = None,
        invalidation_reason: str = "",
        failure_mode: FailureMode | None = None,
        now: float | None = None,
    ) -> HypothesisFailureRecord:
        """Record a hypothesis failure.

        Idempotent: if a record for ``hypothesis.hypothesis_id`` already
        exists, return it without creating a duplicate.
        """
        now = now if now is not None else time.time()

        existing = self._by_id.get(hypothesis.hypothesis_id)
        if existing:
            return existing

        reason = invalidation_reason or hypothesis.invalidation_reason or ""
        mode = failure_mode or _classify_failure_mode(
            reason, market_direction, hypothesis.direction
        )
        lesson = _derive_lesson(hypothesis, mode, market_direction, winning_scenario_thesis)

        rec = HypothesisFailureRecord(
            hypothesis_id=hypothesis.hypothesis_id,
            set_id=set_id,
            instrument=hypothesis.instrument,
            thesis=hypothesis.thesis,
            direction=hypothesis.direction,
            formed_regime=hypothesis.formed_regime,
            confidence_at_failure=hypothesis.confidence,
            reinforcement_count=hypothesis.reinforcement_count,
            supporting_evidence=list(hypothesis.supporting_evidence[-5:]),
            contradicting_evidence=list(hypothesis.contradicting_evidence[-5:]),
            invalidation_reason=reason,
            failure_mode=mode,
            failed_at=now,
            market_direction_observed=market_direction,
            winning_scenario_thesis=winning_scenario_thesis,
            lesson=lesson,
        )

        key = (hypothesis.instrument, hypothesis.direction.value)
        self._records[key].append(rec)
        self._by_id[rec.failure_id] = rec
        self._by_id[hypothesis.hypothesis_id] = rec

        logger.debug(
            f"[FailureTracker] recorded {rec.failure_mode} failure for "
            f"'{rec.thesis[:40]}' on {rec.instrument}"
        )
        return rec

    # ── Read ──────────────────────────────────────────────────────────────

    def get_failures(
        self,
        instrument: str,
        direction: BeliefDirection | None = None,
        n: int = 20,
    ) -> list[HypothesisFailureRecord]:
        """Most recent failure records for ``instrument``, newest-last."""
        if direction is not None:
            key = (instrument, direction.value)
            recs = list(self._records[key])
        else:
            recs = []
            for d in BeliefDirection:
                recs.extend(self._records[(instrument, d.value)])
            recs.sort(key=lambda r: r.failed_at)
        return recs[-n:]

    def failure_rate(
        self,
        instrument: str,
        direction: BeliefDirection | None = None,
    ) -> float:
        """Fraction of hypotheses that failed (vs zero = no failures on record).

        Since we only track failures, this returns the count-based density:
        failures per instrument (not a ratio, because we don't separately
        track successes here — that lives in ``HypothesisFailureTracker``'s
        complement, ``NarrativeOutcomeTracker``).

        Use this as a relative signal: higher = more failure history for this
        direction/instrument.
        """
        recs = self.get_failures(instrument, direction, n=1000)
        return len(recs)

    def common_failure_modes(
        self,
        instrument: str,
        n: int = 50,
    ) -> dict[str, int]:
        """Failure mode → count for the most recent ``n`` failures."""
        recs = self.get_failures(instrument, n=n)
        counts: dict[str, int] = defaultdict(int)
        for r in recs:
            counts[r.failure_mode] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def dominant_failure_mode(self, instrument: str) -> FailureMode:
        """Most common failure mode for ``instrument`` (or 'UNKNOWN')."""
        modes = self.common_failure_modes(instrument, n=100)
        return max(modes, key=lambda k: modes[k]) if modes else "UNKNOWN"

    def lesson_summary(self, instrument: str, n: int = 3) -> list[str]:
        """Top ``n`` most recent unique lessons for ``instrument``."""
        recs = self.get_failures(instrument, n=50)
        seen = set()
        result = []
        for r in reversed(recs):
            if r.lesson and r.lesson not in seen:
                seen.add(r.lesson)
                result.append(r.lesson)
                if len(result) >= n:
                    break
        return result

    def direction_bias_summary(self, instrument: str) -> dict[str, float]:
        """Relative failure count by direction (BULLISH / BEARISH / NEUTRAL).

        Higher count = more failures for that direction.  Callers can use
        this to reduce seed confidence for repeatedly-failing directions.
        """
        result = {}
        for d in BeliefDirection:
            result[d.value] = float(len(self._records[(instrument, d.value)]))
        return result

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        all_recs: list[dict] = []
        for q in self._records.values():
            for r in q:
                all_recs.append(r.to_dict())
        return {"records": all_recs}

    def load_snapshot(self, data: dict) -> None:
        self._records.clear()
        self._by_id.clear()
        for d in data.get("records", []):
            try:
                rec = HypothesisFailureRecord.from_dict(d)
                key = (rec.instrument, rec.direction.value)
                if key not in self._records:
                    self._records[key] = deque(maxlen=self._max)
                self._records[key].append(rec)
                self._by_id[rec.failure_id] = rec
                self._by_id[rec.hypothesis_id] = rec
            except Exception:
                pass


__all__ = [
    "FailureMode",
    "HypothesisFailureRecord",
    "HypothesisFailureTracker",
]
