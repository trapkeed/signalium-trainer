"""QDE Brain — debate quality measurement (Phase C5).

``DebateQualityScore`` answers: "How well-formed was this debate?"

A well-formed debate has:
- Dense evidence on both sides (both bull and bear theses are supported
  by more than one or two evidence items).
- Balanced coverage (neither side is starved of evidence — the system
  actually considered the other direction).
- A clear margin (one thesis genuinely dominates, reducing ambiguity).
- Conviction alignment (the computed conviction score tracks the
  winning thesis strength rather than diverging from it).
- Timely theses (both sides were updated recently, not stale).

This module only reads existing ``Thesis`` / ``DebateVerdict`` /
``ConvictionScore`` objects — it introduces no new tracked state.

Design constraints
------------------
* Pure functions only; the stateful tracker lives in ``DebateAPI``.
* No ``QDEDecision`` imports.
* Never reimplements conviction, thesis strength, or margin logic.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.qde.brain.conviction_engine import ConvictionScore
from trainer.qde.brain.debate_engine import DEFAULT_DOMINANCE_MARGIN, DebateVerdict
from trainer.qde.brain.thesis_manager import Thesis

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# How stale (seconds) a thesis must be before timeliness degrades to 0
_TIMELINESS_DECAY_SECONDS = 300.0  # 5-minute thesis freshness window

# Minimum evidence items per thesis for full density credit
_DENSITY_TARGET = 6

# Weights for overall_score
_W_DENSITY = 0.20
_W_BALANCE = 0.20
_W_MARGIN = 0.25
_W_ALIGNMENT = 0.20
_W_TIMELINESS = 0.15


# -------------------------------------------------------------------
# Score dataclass
# -------------------------------------------------------------------


@dataclass
class DebateQualityScore:
    """Quality assessment of one debate at one point in time.

    ``evidence_density``
        Average evidence count per thesis, normalised to [0, 1].
        Low density = one or both theses are weakly supported.

    ``evidence_balance``
        How evenly evidence is distributed between the two sides.
        1.0 = perfectly symmetric; 0.0 = all evidence on one side.

    ``margin_clarity``
        How decisive the winning margin is.  1.0 = margin ≥ the
        ``DEFAULT_DOMINANCE_MARGIN`` threshold; 0.0 = no margin.

    ``conviction_alignment``
        How closely the conviction score tracks the dominant thesis
        strength.  Divergence indicates the brain over- or under-
        weights the mechanistic evidence read.

    ``timeliness``
        1.0 if both theses were updated within the freshness window;
        decays linearly to 0 as the oldest update approaches
        ``_TIMELINESS_DECAY_SECONDS``.

    ``overall_score``
        Weighted composite of the five dimensions.

    ``grade``
        "HIGH" (≥ 0.70), "MEDIUM" (≥ 0.45), or "LOW".
    """

    score_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    debate_id: str = ""
    instrument: str = ""
    opportunity_id: str = ""
    evidence_density: float = 0.0
    evidence_balance: float = 0.0
    margin_clarity: float = 0.0
    conviction_alignment: float = 0.0
    timeliness: float = 0.0
    overall_score: float = 0.0
    grade: str = "LOW"
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "score_id": self.score_id,
            "debate_id": self.debate_id,
            "instrument": self.instrument,
            "opportunity_id": self.opportunity_id,
            "evidence_density": round(self.evidence_density, 4),
            "evidence_balance": round(self.evidence_balance, 4),
            "margin_clarity": round(self.margin_clarity, 4),
            "conviction_alignment": round(self.conviction_alignment, 4),
            "timeliness": round(self.timeliness, 4),
            "overall_score": round(self.overall_score, 4),
            "grade": self.grade,
            "computed_at": round(self.computed_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DebateQualityScore":
        return cls(
            score_id=d.get("score_id", str(uuid.uuid4())[:12]),
            debate_id=d.get("debate_id", ""),
            instrument=d.get("instrument", ""),
            opportunity_id=d.get("opportunity_id", ""),
            evidence_density=d.get("evidence_density", 0.0),
            evidence_balance=d.get("evidence_balance", 0.0),
            margin_clarity=d.get("margin_clarity", 0.0),
            conviction_alignment=d.get("conviction_alignment", 0.0),
            timeliness=d.get("timeliness", 0.0),
            overall_score=d.get("overall_score", 0.0),
            grade=d.get("grade", "LOW"),
            computed_at=d.get("computed_at", 0.0),
        )


# -------------------------------------------------------------------
# Pure measurement function
# -------------------------------------------------------------------


def measure_quality(
    verdict: DebateVerdict,
    conviction: ConvictionScore,
    bull_thesis: Thesis,
    bear_thesis: Thesis,
    debate_id: str = "",
    now: float | None = None,
    decay_seconds: float = _TIMELINESS_DECAY_SECONDS,
    density_target: int = _DENSITY_TARGET,
    dominance_margin: float = DEFAULT_DOMINANCE_MARGIN,
) -> DebateQualityScore:
    """Measure the quality of a completed debate.

    All inputs are read-only; nothing is mutated.

    Parameters
    ----------
    verdict:
        The ``DebateVerdict`` produced by ``DebateEngine.evaluate()``.
    conviction:
        The ``ConvictionScore`` for the hypothesis backing the
        opportunity.
    bull_thesis / bear_thesis:
        The two ``Thesis`` objects from ``ThesisManager.sync()`` (read
        from ``opportunity.bull_thesis`` / ``opportunity.bear_thesis``
        after the sync).
    debate_id:
        Identifier for the ``DebateRecord`` this score belongs to.
    now:
        Override current time.
    """
    now = now if now is not None else time.time()

    # ── Evidence density ──────────────────────────────────────────────
    bull_ev = len(bull_thesis.supporting_evidence) + len(bull_thesis.conflicting_evidence)
    bear_ev = len(bear_thesis.supporting_evidence) + len(bear_thesis.conflicting_evidence)
    avg_ev = (bull_ev + bear_ev) / 2.0
    evidence_density = min(1.0, avg_ev / max(1, density_target))

    # ── Evidence balance ──────────────────────────────────────────────
    total_ev = bull_ev + bear_ev
    if total_ev > 0:
        evidence_balance = 1.0 - abs(bull_ev - bear_ev) / total_ev
    else:
        evidence_balance = 0.0

    # ── Margin clarity ────────────────────────────────────────────────
    margin_clarity = min(1.0, verdict.margin / max(1e-9, dominance_margin))

    # ── Conviction alignment ──────────────────────────────────────────
    # The dominant thesis strength is what conviction "should" track.
    dominant_strength = max(verdict.bull_strength, verdict.bear_strength)
    conviction_alignment = max(0.0, 1.0 - abs(conviction.conviction - dominant_strength))

    # ── Timeliness ────────────────────────────────────────────────────
    oldest_update = min(bull_thesis.updated_at, bear_thesis.updated_at)
    age = max(0.0, now - oldest_update)
    timeliness = max(0.0, 1.0 - age / max(1e-9, decay_seconds))

    # ── Composite ─────────────────────────────────────────────────────
    overall_score = max(
        0.0,
        min(
            1.0,
            _W_DENSITY * evidence_density
            + _W_BALANCE * evidence_balance
            + _W_MARGIN * margin_clarity
            + _W_ALIGNMENT * conviction_alignment
            + _W_TIMELINESS * timeliness,
        ),
    )

    if overall_score >= 0.70:
        grade = "HIGH"
    elif overall_score >= 0.45:
        grade = "MEDIUM"
    else:
        grade = "LOW"

    return DebateQualityScore(
        debate_id=debate_id,
        instrument=verdict.instrument,
        opportunity_id=verdict.opportunity_id,
        evidence_density=evidence_density,
        evidence_balance=evidence_balance,
        margin_clarity=margin_clarity,
        conviction_alignment=conviction_alignment,
        timeliness=timeliness,
        overall_score=overall_score,
        grade=grade,
        computed_at=now,
    )


__all__ = [
    "DebateQualityScore",
    "measure_quality",
]
