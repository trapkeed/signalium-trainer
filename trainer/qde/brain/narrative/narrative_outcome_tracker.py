"""QDE Brain — narrative outcome tracker.

Records what actually happened after a ``SymbolNarrative`` reached
``ESTABLISHED`` maturity, enabling the system to learn whether its narrative
conclusions were correct and to score narrative quality over time.

Lifecycle of an outcome record
---------------------------------
1. A narrative reaches ``NarrativeMaturity.ESTABLISHED``.
2. The caller (``NarrativeAPI.store_narrative``) calls
   ``record_establishment(narrative_id, kind, instrument, direction, ...)``
   to open an outcome record.
3. When the outcome is known — typically after the narrative has expired
   and enough bars have passed — the caller calls
   ``record_outcome(narrative_id, direction_correct, reward_normalized)``
   to resolve the record.
4. ``quality_score(instrument, kind)`` returns a rolling summary of how
   accurate past narratives of that kind have been for that instrument.

Quality score
-------------
``NarrativeQualityScore`` captures:

  * ``win_rate``          — fraction of resolved records where
                           ``direction_correct=True`` (rolling window).
  * ``avg_reward``        — EMA of ``reward_normalized`` (signed −1..+1).
  * ``calibration_bias``  — signed bias = mean(direction_correct − 0.5).
                           Positive means the narrative over-estimated
                           accuracy; negative means under-estimated.
  * ``sample_count``      — number of resolved records used.

The ``NarrativeOutcomeTracker`` is intentionally stateful so quality
evolves incrementally — each new resolved record updates a rolling EMA
rather than recomputing from scratch (consistent with the rest of the
brain's learning layer).

Persistence
-----------
``to_snapshot()`` / ``load_snapshot()`` via the standard dict convention.
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

from trainer.qde.brain.narrative.narrative_store import NarrativeKind

# ── Hyper-parameters ──────────────────────────────────────────────────────────

_EMA_ALPHA = 0.15  # smoothing for win_rate / avg_reward EMA
_MAX_RECORDS_PER_KEY = 500  # per (instrument, kind) history depth
_GLOBAL_MAX_RECORDS = 2000  # global cap across all records


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class NarrativeOutcomeRecord:
    """Outcome data for one ``SymbolNarrative`` that reached ESTABLISHED.

    ``direction``           — +1 bullish, −1 bearish, 0 neutral at time of
                              establishment (from ``net_bias`` or narrative kind).
    ``price_at_establishment`` — mid price at the moment of ESTABLISHED
                              (0.0 if not available from the caller).
    ``resolved``            — True once ``record_outcome`` has been called.
    ``direction_correct``   — True if actual price moved in the predicted direction.
    ``reward_normalized``   — Signed −1..+1 normalized outcome
                              (e.g. reward_atr / some_normalization_constant).
    ``resolution_bars``     — Bars until resolution (informational).
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    narrative_id: str = ""
    kind: NarrativeKind = NarrativeKind.TREND_CONTINUATION
    instrument: str = ""
    direction: int = 0  # +1 / -1 / 0
    price_at_establishment: float = 0.0
    established_at: float = field(default_factory=time.time)
    resolved: bool = False
    direction_correct: bool = False
    reward_normalized: float = 0.0
    resolution_bars: int = 0
    resolved_at: float = 0.0

    def resolve(
        self,
        direction_correct: bool,
        reward_normalized: float,
        resolution_bars: int = 0,
        now: float | None = None,
    ) -> None:
        self.resolved = True
        self.direction_correct = direction_correct
        self.reward_normalized = max(-1.0, min(1.0, reward_normalized))
        self.resolution_bars = resolution_bars
        self.resolved_at = now if now is not None else time.time()

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "narrative_id": self.narrative_id,
            "kind": self.kind.value,
            "instrument": self.instrument,
            "direction": self.direction,
            "price_at_establishment": round(self.price_at_establishment, 6),
            "established_at": round(self.established_at, 3),
            "resolved": self.resolved,
            "direction_correct": self.direction_correct,
            "reward_normalized": round(self.reward_normalized, 4),
            "resolution_bars": self.resolution_bars,
            "resolved_at": round(self.resolved_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> NarrativeOutcomeRecord:
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            narrative_id=d.get("narrative_id", ""),
            kind=NarrativeKind(d.get("kind", NarrativeKind.TREND_CONTINUATION.value)),
            instrument=d.get("instrument", ""),
            direction=d.get("direction", 0),
            price_at_establishment=d.get("price_at_establishment", 0.0),
            established_at=d.get("established_at", 0.0),
            resolved=d.get("resolved", False),
            direction_correct=d.get("direction_correct", False),
            reward_normalized=d.get("reward_normalized", 0.0),
            resolution_bars=d.get("resolution_bars", 0),
            resolved_at=d.get("resolved_at", 0.0),
        )


@dataclass
class NarrativeQualityScore:
    """Rolling quality summary for one (instrument, kind) or (global, kind) pair.

    ``win_rate``         — EMA fraction of resolved records where direction_correct.
    ``avg_reward``       — EMA average normalized reward.
    ``calibration_bias`` — mean(direction_correct − 0.5); positive = over-confident.
    ``sample_count``     — total resolved records contributing to this score.
    ``reliability``      — composite 0-1: win_rate * (1 − |calibration_bias| * 2).
                           Higher = both accurate and well-calibrated.
    """

    kind: NarrativeKind
    instrument: str = ""  # "" = global across all instruments
    win_rate: float = 0.50
    avg_reward: float = 0.0
    calibration_bias: float = 0.0
    sample_count: int = 0
    reliability: float = 0.50

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "instrument": self.instrument,
            "win_rate": round(self.win_rate, 4),
            "avg_reward": round(self.avg_reward, 4),
            "calibration_bias": round(self.calibration_bias, 4),
            "sample_count": self.sample_count,
            "reliability": round(self.reliability, 4),
        }


# ── Tracker ───────────────────────────────────────────────────────────────────


class NarrativeOutcomeTracker:
    """Records, resolves, and summarizes narrative outcomes.

    Per-symbol, per-kind history is maintained as bounded deques.
    Rolling statistics (win_rate, avg_reward) are EMA-updated on each
    ``record_outcome`` call so cost is O(1) per resolution.
    """

    def __init__(
        self,
        ema_alpha: float = _EMA_ALPHA,
        max_per_key: int = _MAX_RECORDS_PER_KEY,
        global_max: int = _GLOBAL_MAX_RECORDS,
    ) -> None:
        self._alpha = ema_alpha
        self._max = max_per_key
        self._global_max = global_max

        # (instrument, kind.value) → deque[NarrativeOutcomeRecord]
        self._records: dict[tuple[str, str], deque[NarrativeOutcomeRecord]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )
        # narrative_id → NarrativeOutcomeRecord (for fast lookup by ID)
        self._by_id: dict[str, NarrativeOutcomeRecord] = {}

        # Per (instrument, kind) EMA state
        # _ema_state[(instrument, kind.value)] = {"win": 0.5, "reward": 0.0, "n": 0, "bias_sum": 0.0}
        self._ema: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"win": 0.5, "reward": 0.0, "n": 0, "bias_sum": 0.0}
        )
        # Global per-kind EMA state (key = ("", kind.value))
        # Same structure, instrument=""

    # ── Write ─────────────────────────────────────────────────────────────

    def record_establishment(
        self,
        narrative_id: str,
        kind: NarrativeKind,
        instrument: str,
        direction: int = 0,
        price_at_establishment: float = 0.0,
        now: float | None = None,
    ) -> NarrativeOutcomeRecord:
        """Open a new outcome record when a narrative reaches ESTABLISHED.

        Idempotent: if a record for ``narrative_id`` already exists, returns it
        without creating a duplicate.
        """
        now = now if now is not None else time.time()

        # Idempotency
        if narrative_id in self._by_id:
            return self._by_id[narrative_id]

        rec = NarrativeOutcomeRecord(
            narrative_id=narrative_id,
            kind=kind,
            instrument=instrument,
            direction=direction,
            price_at_establishment=price_at_establishment,
            established_at=now,
        )
        key = (instrument, kind.value)
        self._records[key].append(rec)
        self._by_id[rec.record_id] = rec
        self._by_id[narrative_id] = rec  # also index by narrative_id

        logger.debug(
            f"[OutcomeTracker] opened record {rec.record_id} "
            f"{instrument}/{kind.value} direction={direction}"
        )
        return rec

    def record_outcome(
        self,
        narrative_id: str,
        direction_correct: bool,
        reward_normalized: float,
        resolution_bars: int = 0,
        now: float | None = None,
    ) -> bool:
        """Resolve an open outcome record.

        Returns True if the record was found and resolved, False if not found
        or already resolved.
        """
        now = now if now is not None else time.time()
        rec = self._by_id.get(narrative_id)
        if rec is None or rec.resolved:
            return False

        rec.resolve(direction_correct, reward_normalized, resolution_bars, now)

        # Update per-instrument EMA
        key = (rec.instrument, rec.kind.value)
        self._update_ema(key, direction_correct, reward_normalized)

        # Update global EMA
        global_key = ("", rec.kind.value)
        self._update_ema(global_key, direction_correct, reward_normalized)

        logger.debug(
            f"[OutcomeTracker] resolved {rec.record_id} "
            f"correct={direction_correct} reward={reward_normalized:.3f}"
        )
        return True

    # ── Read ──────────────────────────────────────────────────────────────

    def quality_score(
        self,
        instrument: str,
        kind: NarrativeKind,
    ) -> NarrativeQualityScore:
        """Quality score for a specific instrument + kind pair.

        Returns a default score (win_rate=0.5) when no resolved records exist.
        """
        key = (instrument, kind.value)
        ema = self._ema[key]
        n = ema["n"]
        win = ema["win"]
        rw = ema["reward"]
        bias = ema["bias_sum"] / max(1, n)
        rel = win * max(0.0, 1.0 - abs(bias) * 2.0)
        return NarrativeQualityScore(
            kind=kind,
            instrument=instrument,
            win_rate=round(win, 4),
            avg_reward=round(rw, 4),
            calibration_bias=round(bias, 4),
            sample_count=n,
            reliability=round(min(1.0, max(0.0, rel)), 4),
        )

    def quality_score_global(self, kind: NarrativeKind) -> NarrativeQualityScore:
        """Global quality score across all instruments for ``kind``."""
        return self.quality_score("", kind)

    def all_quality_scores(
        self,
        instrument: str = "",
    ) -> dict[str, NarrativeQualityScore]:
        """Quality scores for all 7 kinds for one instrument (or global)."""
        return {k.value: self.quality_score(instrument, k) for k in NarrativeKind}

    def pending_records(self, instrument: str, kind: NarrativeKind) -> list[NarrativeOutcomeRecord]:
        """Unresolved records for ``instrument``/``kind``."""
        key = (instrument, kind.value)
        return [r for r in self._records[key] if not r.resolved]

    def resolved_records(
        self,
        instrument: str,
        kind: NarrativeKind,
        n: int = 20,
    ) -> list[NarrativeOutcomeRecord]:
        """Most recent resolved records (newest last)."""
        key = (instrument, kind.value)
        return [r for r in self._records[key] if r.resolved][-n:]

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        records_out: list[dict] = []
        for q in self._records.values():
            for r in q:
                records_out.append(r.to_dict())
        return {
            "records": records_out,
            "ema": {f"{k[0]}|{k[1]}": v for k, v in self._ema.items()},
        }

    def load_snapshot(self, data: dict) -> None:
        self._records.clear()
        self._by_id.clear()
        self._ema.clear()

        for d in data.get("records", []):
            try:
                rec = NarrativeOutcomeRecord.from_dict(d)
                key = (rec.instrument, rec.kind.value)
                if key not in self._records:
                    self._records[key] = deque(maxlen=self._max)
                self._records[key].append(rec)
                self._by_id[rec.record_id] = rec
                self._by_id[rec.narrative_id] = rec
            except Exception:
                pass

        for composite_key, v in data.get("ema", {}).items():
            try:
                parts = composite_key.split("|", 1)
                k = (parts[0], parts[1])
                self._ema[k] = {
                    "win": float(v.get("win", 0.5)),
                    "reward": float(v.get("reward", 0.0)),
                    "n": int(v.get("n", 0)),
                    "bias_sum": float(v.get("bias_sum", 0.0)),
                }
            except Exception:
                pass

    # ── Internal ──────────────────────────────────────────────────────────

    def _update_ema(self, key: tuple[str, str], direction_correct: bool, reward: float) -> None:
        """EMA-update the quality state for ``key``."""
        state = self._ema[key]
        a = self._alpha
        state["win"] = (1 - a) * state["win"] + a * (1.0 if direction_correct else 0.0)
        state["reward"] = (1 - a) * state["reward"] + a * reward
        state["n"] += 1
        state["bias_sum"] += (1.0 if direction_correct else 0.0) - 0.5


__all__ = [
    "NarrativeOutcomeRecord",
    "NarrativeQualityScore",
    "NarrativeOutcomeTracker",
]
