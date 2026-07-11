"""QDE Brain — debate record and memory bridge (Phase C5).

``DebateRecord``
    A complete, immutable-in-practice snapshot of one finished debate:
    verdict, conviction, quality score, overconfidence event (if any),
    and ignored-evidence record (if any).  This is the unit that is
    persisted to episodic memory and used by the reflection engine.

``DebateMemoryBridge``
    Connects the debate package to C2 episodic memory (soft import —
    works with or without the C2 layer, identical pattern to
    ``NarrativeMemoryBridge`` and ``HypothesisMemoryBridge``).

    ``store_debate(record, now)``
        Write the debate as an ``Episode`` to ``EpisodicMemory``.

    ``recall_similar_debates(instrument, features, k, now)``
        Retrieve the ``k`` most-similar past debate episodes and
        reconstruct their ``DebateRecord`` from episode reasoning tags.

Design constraints
------------------
* ``DebateRecord`` is a pure data container — no business logic.
* ``DebateMemoryBridge`` never reimplements ``EpisodicMemory`` or
  ``ThesisManager`` logic.
* All C2 imports are soft (``try/except ImportError``).
* No ``QDEDecision`` imports.
* ``to_dict()`` / ``from_dict()`` for ``DebateRecord``; snapshot pattern
  for ``DebateMemoryBridge``.
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

from trainer.qde.brain.conviction_engine import ConvictionScore
from trainer.qde.brain.debate.debate_quality import DebateQualityScore
from trainer.qde.brain.debate.ignored_evidence_tracker import IgnoredEvidenceRecord
from trainer.qde.brain.debate.overconfidence_tracker import OverconfidenceEvent
from trainer.qde.brain.debate_engine import DebateVerdict

# C2 soft imports
try:
    from trainer.qde.brain.episodic.episodic_memory import (
        Episode,
        EpisodeFeatures,
        EpisodeOutcome,
        EpisodePortfolioState,
    )

    _HAS_EPISODIC = True
except ImportError:
    _HAS_EPISODIC = False

MAX_STORED_RECORDS = 500  # in-process deque cap (episodic memory is the durable store)


# -------------------------------------------------------------------
# DebateRecord
# -------------------------------------------------------------------


@dataclass
class DebateRecord:
    """Complete snapshot of one finished debate.

    All sub-fields are stored as ``to_dict()`` copies rather than live
    references so the record is self-contained and safe to serialise.

    ``overconfidence_event`` and ``ignored_evidence`` are ``None`` when
    no event was detected in the corresponding check.
    """

    debate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""
    opportunity_id: str = ""
    hypothesis_id: str = ""

    # Sub-component snapshots stored as dicts for clean serialisation
    verdict: dict = field(default_factory=dict)
    conviction: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    bull_thesis: dict = field(default_factory=dict)
    bear_thesis: dict = field(default_factory=dict)
    overconfidence_event: dict | None = None
    ignored_evidence: dict | None = None

    created_at: float = field(default_factory=time.time)

    # Convenience accessors (derived from stored dicts — no extra storage)
    @property
    def conclusive(self) -> bool:
        return bool(self.verdict.get("conclusive", False))

    @property
    def margin(self) -> float:
        return float(self.verdict.get("margin", 0.0))

    @property
    def quality_grade(self) -> str:
        return str(self.quality.get("grade", "LOW"))

    @property
    def overall_quality(self) -> float:
        return float(self.quality.get("overall_score", 0.0))

    @property
    def conviction_score(self) -> float:
        return float(self.conviction.get("conviction", 0.0))

    @property
    def has_overconfidence(self) -> bool:
        return self.overconfidence_event is not None

    @property
    def has_ignored_evidence(self) -> bool:
        return self.ignored_evidence is not None

    def to_dict(self) -> dict:
        return {
            "debate_id": self.debate_id,
            "instrument": self.instrument,
            "opportunity_id": self.opportunity_id,
            "hypothesis_id": self.hypothesis_id,
            "verdict": self.verdict,
            "conviction": self.conviction,
            "quality": self.quality,
            "bull_thesis": self.bull_thesis,
            "bear_thesis": self.bear_thesis,
            "overconfidence_event": self.overconfidence_event,
            "ignored_evidence": self.ignored_evidence,
            "created_at": round(self.created_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DebateRecord":
        return cls(
            debate_id=d.get("debate_id", str(uuid.uuid4())[:12]),
            instrument=d.get("instrument", ""),
            opportunity_id=d.get("opportunity_id", ""),
            hypothesis_id=d.get("hypothesis_id", ""),
            verdict=d.get("verdict", {}),
            conviction=d.get("conviction", {}),
            quality=d.get("quality", {}),
            bull_thesis=d.get("bull_thesis", {}),
            bear_thesis=d.get("bear_thesis", {}),
            overconfidence_event=d.get("overconfidence_event"),
            ignored_evidence=d.get("ignored_evidence"),
            created_at=d.get("created_at", 0.0),
        )


def make_debate_record(
    verdict: DebateVerdict,
    conviction: ConvictionScore,
    quality: DebateQualityScore,
    bull_thesis_dict: dict,
    bear_thesis_dict: dict,
    overconfidence_event: OverconfidenceEvent | None = None,
    ignored_evidence: IgnoredEvidenceRecord | None = None,
    debate_id: str = "",
    now: float | None = None,
) -> DebateRecord:
    """Factory: build a ``DebateRecord`` from the live objects produced
    by the various C5 engines.  Snapshots all inputs at call time.
    """
    now = now if now is not None else time.time()
    did = debate_id or str(uuid.uuid4())[:12]
    return DebateRecord(
        debate_id=did,
        instrument=verdict.instrument,
        opportunity_id=verdict.opportunity_id,
        hypothesis_id=conviction.hypothesis_id,
        verdict=verdict.to_dict(),
        conviction=conviction.to_dict(),
        quality=quality.to_dict(),
        bull_thesis=bull_thesis_dict,
        bear_thesis=bear_thesis_dict,
        overconfidence_event=overconfidence_event.to_dict() if overconfidence_event else None,
        ignored_evidence=ignored_evidence.to_dict() if ignored_evidence else None,
        created_at=now,
    )


# -------------------------------------------------------------------
# DebateMemoryBridge
# -------------------------------------------------------------------

# Feature dimensions to encode a debate for cosine-similarity recall.
# Indices: bull_strength, bear_strength, margin, conviction, confidence,
#          quality, evidence_density, evidence_balance, conviction_alignment,
#          timeliness, has_overconfidence, has_ignored
_FEAT_DIM = 12


def _debate_to_features(record: DebateRecord) -> dict:
    """Convert a ``DebateRecord`` to the 12-dim feature dict consumed
    by ``EpisodeFeatures.from_dict()``.  Keys match ``EpisodeFeatures``
    field names where possible; the rest map to the closest semantically
    appropriate field.
    """
    q = record.quality
    v = record.verdict
    c = record.conviction
    return {
        "volatility": float(v.get("bull_strength", 0.5)),  # proxy: bull pressure
        "momentum": float(v.get("bear_strength", 0.5)) * 2.0 - 1.0,  # [-1,1]
        "trend_strength": float(v.get("margin", 0.0)),
        "range_compression": float(q.get("evidence_density", 0.5)),
        "absorption": float(q.get("evidence_balance", 0.5)),
        "volume_balance": float(c.get("conviction", 0.5)),
        "session_quality": float(q.get("overall_score", 0.5)),
        "microstructure": float(q.get("conviction_alignment", 0.5)),
        "regime_numeric": float(q.get("timeliness", 0.5)),
        "hypothesis_conf": float(c.get("confidence", 0.5)),
        "attention_weight": 1.0 if record.has_overconfidence else 0.0,
        "spread_norm": 1.0 if record.has_ignored_evidence else 0.0,
    }


class DebateMemoryBridge:
    """Persist finished debates to C2 episodic memory (soft dependency).

    All C2 imports are wrapped in ``try/except ImportError`` so the
    debate package works even when the episodic package is not installed.

    In-process deque
    ----------------
    An in-memory deque also stores ``DebateRecord`` objects so recent
    debates can be recalled without needing the episodic index.  This
    is the fallback when episodic memory is not available.

    Usage::

        bridge = DebateMemoryBridge(episodic_memory=emem)
        episode_id = bridge.store_debate(record, now=now)

        recent = bridge.recent_debates("EURUSD", n=10)
    """

    def __init__(
        self,
        episodic_memory: object | None = None,
        max_in_process: int = MAX_STORED_RECORDS,
    ) -> None:
        self._episodic = episodic_memory
        self._max_in_process = max_in_process
        # instrument → deque of DebateRecord
        self._local: dict[str, deque[DebateRecord]] = defaultdict(
            lambda: deque(maxlen=self._max_in_process)
        )

    # ── Primary API ───────────────────────────────────────────────────

    def store_debate(
        self,
        record: DebateRecord,
        now: float | None = None,
    ) -> str:
        """Persist a ``DebateRecord`` to episodic memory and the local
        deque.

        Returns the episode_id if stored in episodic memory, or the
        ``record.debate_id`` if only the local deque was used.
        """
        now = now if now is not None else time.time()
        self._local[record.instrument].append(record)

        if not _HAS_EPISODIC or self._episodic is None:
            logger.debug(f"[DebateMemoryBridge] stored locally (no episodic): {record.debate_id}")
            return record.debate_id

        return self._store_episodic(record, now)

    def recent_debates(
        self,
        instrument: str,
        n: int = 20,
    ) -> list[DebateRecord]:
        """Return the ``n`` most-recent debates for ``instrument`` from
        the local in-process deque, newest first.
        """
        recs = list(self._local.get(instrument, []))
        recs.reverse()
        return recs[:n]

    def debate_count(self, instrument: str) -> int:
        return len(self._local.get(instrument, []))

    # ── Persistence (snapshot of local deque only) ────────────────────

    def to_snapshot(self) -> dict:
        return {
            "records": {instr: [r.to_dict() for r in deq] for instr, deq in self._local.items()}
        }

    def load_snapshot(self, data: dict) -> None:
        self._local.clear()
        for instr, recs in data.get("records", {}).items():
            for d in recs:
                try:
                    rec = DebateRecord.from_dict(d)
                    self._local[instr].append(rec)
                except Exception:
                    pass

    # ── Internal ──────────────────────────────────────────────────────

    def _store_episodic(self, record: DebateRecord, now: float) -> str:
        """Write the debate as an Episode.  Returns episode_id or ""."""
        try:
            feat_dict = _debate_to_features(record)
            feat = EpisodeFeatures.from_dict(feat_dict)  # type: ignore[name-defined]
            outcome = EpisodeOutcome(  # type: ignore[name-defined]
                direction_correct=record.conclusive,
                reward_atr=record.margin,
                resolved=True,
            )
            ep = Episode(  # type: ignore[name-defined]
                instrument=record.instrument,
                timestamp=now,
                features=feat,
                portfolio=EpisodePortfolioState(),  # type: ignore[name-defined]
                action="DEBATE",
                confidence=record.conviction_score,
                outcome=outcome,
                trade_result_atr=record.margin,
                reasoning=(
                    f"debate:{record.debate_id}:"
                    f"quality={record.quality_grade}:"
                    f"oc={'Y' if record.has_overconfidence else 'N'}:"
                    f"ie={'Y' if record.has_ignored_evidence else 'N'}"
                ),
            )
            self._episodic.write_memory(ep, now=now)  # type: ignore[union-attr]
            logger.debug(
                f"[DebateMemoryBridge] stored episode {ep.episode_id} "
                f"for debate {record.debate_id} on {record.instrument}"
            )
            return ep.episode_id
        except Exception as exc:
            logger.warning(f"[DebateMemoryBridge] episodic store failed: {exc}")
            return record.debate_id


__all__ = [
    "DebateRecord",
    "DebateMemoryBridge",
    "make_debate_record",
]
