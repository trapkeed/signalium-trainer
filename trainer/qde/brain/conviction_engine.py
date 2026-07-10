"""QDE Brain — conviction engine.

Distinguishes **confidence** from **conviction**:

  - confidence: the probability a prediction is correct. This already
    exists as `Hypothesis.confidence` / `QDEOutputHeads.confidence` —
    a model/evidence-weighted estimate, owned elsewhere. This module
    never re-derives it.
  - conviction: the strength of belief in the *setup* — how well-formed
    and battle-tested the underlying thesis is, independent of the raw
    probability estimate. A thesis can be highly probable but freshly
    formed (thin track record), or less probable on paper but backed by
    sustained reinforcement and one-sided evidence.

Conviction is computed purely from signals already tracked on a
`Hypothesis` (reinforcement count, evidence balance, age, maturity) via
`hypothesis_lifecycle` — it introduces no new tracked fields on
`Hypothesis` itself and no new storage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from trainer.qde.brain.brain_models import Hypothesis
from trainer.qde.brain.hypothesis_lifecycle import (
    DEFAULT_MATURITY_THRESHOLDS,
    HypothesisMaturity,
    MaturityThresholds,
    compute_maturity,
    hypothesis_age_seconds,
    maturity_rank,
)

DEFAULT_HIGH_THRESHOLD = 0.6
DEFAULT_LOW_THRESHOLD = 0.4


class ConvictionProfile(str, Enum):
    """Where a hypothesis sits in the confidence x conviction space."""

    ALIGNED_HIGH = "ALIGNED_HIGH"  # high confidence, high conviction
    ALIGNED_LOW = "ALIGNED_LOW"  # low confidence, low conviction
    HIGH_CONFIDENCE_LOW_CONVICTION = "HIGH_CONFIDENCE_LOW_CONVICTION"
    LOW_CONFIDENCE_HIGH_CONVICTION = "LOW_CONFIDENCE_HIGH_CONVICTION"
    MIXED = "MIXED"  # neither clearly high nor clearly low on either axis


@dataclass(frozen=True)
class ConvictionComponents:
    """Raw 0-1 contributors to the composite conviction score."""

    reinforcement_strength: float = 0.0
    evidence_strength: float = 0.0
    maturity_strength: float = 0.0
    persistence_strength: float = 0.0

    def to_dict(self) -> dict:
        return {
            "reinforcement_strength": round(self.reinforcement_strength, 4),
            "evidence_strength": round(self.evidence_strength, 4),
            "maturity_strength": round(self.maturity_strength, 4),
            "persistence_strength": round(self.persistence_strength, 4),
        }


@dataclass
class ConvictionScore:
    """Conviction reading for one hypothesis at one point in time."""

    hypothesis_id: str
    instrument: str
    confidence: float
    conviction: float
    components: ConvictionComponents = field(default_factory=ConvictionComponents)
    profile: ConvictionProfile = ConvictionProfile.MIXED
    explanation: str = ""
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "instrument": self.instrument,
            "confidence": round(self.confidence, 4),
            "conviction": round(self.conviction, 4),
            "components": self.components.to_dict(),
            "profile": self.profile.value,
            "explanation": self.explanation,
            "computed_at": round(self.computed_at, 3),
        }


def classify_profile(
    confidence: float,
    conviction: float,
    high: float = DEFAULT_HIGH_THRESHOLD,
    low: float = DEFAULT_LOW_THRESHOLD,
) -> ConvictionProfile:
    """Classify where (confidence, conviction) sits, explicitly surfacing
    the divergent quadrants the QDE must be able to reason about:
    high-confidence/low-conviction and low-confidence/high-conviction.
    """
    high_confidence = confidence >= high
    low_confidence = confidence < low
    high_conviction = conviction >= high
    low_conviction = conviction < low

    if high_confidence and high_conviction:
        return ConvictionProfile.ALIGNED_HIGH
    if low_confidence and low_conviction:
        return ConvictionProfile.ALIGNED_LOW
    if high_confidence and low_conviction:
        return ConvictionProfile.HIGH_CONFIDENCE_LOW_CONVICTION
    if low_confidence and high_conviction:
        return ConvictionProfile.LOW_CONFIDENCE_HIGH_CONVICTION
    return ConvictionProfile.MIXED


def compute_conviction(
    hypothesis: Hypothesis,
    now: float | None = None,
    thresholds: MaturityThresholds = DEFAULT_MATURITY_THRESHOLDS,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
) -> ConvictionScore:
    """Derive a conviction score for a hypothesis — how strongly-formed
    the *setup* is — independent of its raw confidence value.

    Pure function of existing `Hypothesis` fields; never mutates the
    hypothesis and never touches storage.
    """
    now = now if now is not None else time.time()

    reinforcement_strength = min(
        1.0, hypothesis.reinforcement_count / max(1, thresholds.executable_min_reinforcements)
    )

    supporting = len(hypothesis.supporting_evidence)
    contradicting = len(hypothesis.contradicting_evidence)
    total_evidence = supporting + contradicting
    evidence_strength = (supporting / total_evidence) if total_evidence > 0 else 0.0

    maturity = compute_maturity(hypothesis, now=now, thresholds=thresholds)
    maturity_strength = max(0.0, maturity_rank(maturity) / 3.0)

    age = hypothesis_age_seconds(hypothesis, now=now)
    persistence_strength = min(1.0, age / max(1e-9, thresholds.executable_min_age_seconds))

    conviction = max(
        0.0,
        min(
            1.0,
            0.30 * reinforcement_strength
            + 0.30 * evidence_strength
            + 0.25 * maturity_strength
            + 0.15 * persistence_strength,
        ),
    )

    components = ConvictionComponents(
        reinforcement_strength=reinforcement_strength,
        evidence_strength=evidence_strength,
        maturity_strength=maturity_strength,
        persistence_strength=persistence_strength,
    )
    profile = classify_profile(
        hypothesis.confidence, conviction, high=high_threshold, low=low_threshold
    )

    explanation = (
        f"{hypothesis.instrument}: confidence={hypothesis.confidence:.0%}, "
        f"conviction={conviction:.0%} ({profile.value.replace('_', ' ').lower()}); "
        f"{hypothesis.reinforcement_count} reinforcement(s), "
        f"{supporting} supporting / {contradicting} contradicting evidence."
    )

    return ConvictionScore(
        hypothesis_id=hypothesis.hypothesis_id,
        instrument=hypothesis.instrument,
        confidence=hypothesis.confidence,
        conviction=conviction,
        components=components,
        profile=profile,
        explanation=explanation,
        computed_at=now,
    )


def is_trade_worthy(
    score: ConvictionScore,
    min_conviction: float = DEFAULT_HIGH_THRESHOLD,
    min_confidence: float = DEFAULT_LOW_THRESHOLD,
) -> bool:
    """A hypothesis may justify a trade either through high confidence
    (the usual path, already gated upstream by `HypothesisMaturity`) or
    through high conviction alone — this is what lets a **low-confidence,
    high-conviction** setup still be trade-worthy, which pure
    confidence-thresholding could never recognize.

    A floor on confidence still applies (`min_confidence`) so conviction
    alone can never justify a trade the evidence actively contradicts.
    """
    if score.confidence < min_confidence:
        return False
    return score.conviction >= min_conviction or score.profile == ConvictionProfile.ALIGNED_HIGH


__all__ = [
    "ConvictionProfile",
    "ConvictionComponents",
    "ConvictionScore",
    "classify_profile",
    "compute_conviction",
    "is_trade_worthy",
    "HypothesisMaturity",
]
