"""QDE Brain — thesis manager.

`ThesisManager` maintains the two competing arguments — bull and bear —
that every `opportunity_tracker.Opportunity` carries, so QDE always
reasons about a setup by weighing both sides rather than only tracking
whichever direction its underlying hypothesis happens to favor.

This module deliberately does not duplicate evidence storage or
confidence scoring that already exists:

  * `Hypothesis.supporting_evidence` / `contradicting_evidence`
    (`brain_models.py`) remain the source of truth for the *tracked*
    hypothesis's own evidence. `ThesisManager.sync()` only *reprojects*
    that evidence onto the two-sided `Thesis` view (a hypothesis's
    contradicting evidence for its own direction is, definitionally,
    supporting evidence for the opposite direction) — it never asks a
    caller to record evidence twice.
  * `Hypothesis.confidence` remains the only confidence estimate for the
    tracked hypothesis; `Thesis.confidence` derives from it (or its
    complement) rather than introducing a second, independently-tracked
    number.
  * "Historical outcomes" reuses `brain_models.SymbolPersonality.
    regime_outcome_quality` (already accumulated by
    `long_term_memory.LongTermMemory.record_experience`) and
    `personality_profiles.PersonalityProfile.trend_persistence` /
    `reversal_tendency` (already learned by `behavioral_memory.
    BehavioralMemory`) as the historical-track-record signal — this
    module adds no parallel outcome-tracking store of its own.

What this module *adds* that did not exist before: the `Thesis`
dataclass itself (a named, two-sided argument with its own supporting/
conflicting evidence, confidence, and historical-support reading) and
the projection logic that keeps both sides of every opportunity
up to date as hypotheses evolve. `debate_engine.DebateEngine` is the
layer that *compares* the two theses this module produces; this module
never compares them itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_models import BrainState, Hypothesis
from trainer.qde.brain.brain_types import BeliefDirection

if TYPE_CHECKING:
    from trainer.qde.brain.opportunity_tracker import Opportunity

# Bounds evidence-list growth the same way
# `opportunity_tracker.MAX_TRANSITION_HISTORY` bounds transition history.
MAX_EVIDENCE_ITEMS = 20

# Weights for `Thesis.strength()` — evidence balance is the primary
# signal (it's the most direct read on "supporting vs. conflicting"),
# confidence and historical support each contribute a supporting share.
EVIDENCE_WEIGHT = 0.4
CONFIDENCE_WEIGHT = 0.35
HISTORICAL_WEIGHT = 0.25


@dataclass
class Thesis:
    """One side of the debate — a directional argument for `side`,
    backed by its own supporting/conflicting evidence, a confidence
    reading, and a historical-support reading.
    """

    side: BeliefDirection = BeliefDirection.NEUTRAL
    argument: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    historical_support: float = 0.5
    updated_at: float = field(default_factory=time.time)

    def strength(self) -> float:
        """Composite 0-1 read on how compelling this side of the debate
        currently is: evidence balance (supporting vs. conflicting),
        confidence in the underlying claim, and historical track record.
        """
        total_evidence = len(self.supporting_evidence) + len(self.conflicting_evidence)
        evidence_balance = (
            len(self.supporting_evidence) / total_evidence if total_evidence > 0 else 0.5
        )
        return max(
            0.0,
            min(
                1.0,
                EVIDENCE_WEIGHT * evidence_balance
                + CONFIDENCE_WEIGHT * self.confidence
                + HISTORICAL_WEIGHT * self.historical_support,
            ),
        )

    def to_dict(self) -> dict:
        return {
            "side": self.side.value,
            "argument": self.argument,
            "supporting_evidence": self.supporting_evidence[-MAX_EVIDENCE_ITEMS:],
            "conflicting_evidence": self.conflicting_evidence[-MAX_EVIDENCE_ITEMS:],
            "confidence": round(self.confidence, 4),
            "historical_support": round(self.historical_support, 4),
            "strength": round(self.strength(), 4),
            "updated_at": round(self.updated_at, 3),
        }


def _history_baseline(regime_outcome_quality: dict[str, float]) -> float:
    """Symbol-wide historical track record, 0-1, derived from
    `SymbolPersonality.regime_outcome_quality` (-1..1 average, already
    accumulated elsewhere) — 0.5 (neutral) when nothing has been
    recorded yet.
    """
    if not regime_outcome_quality:
        return 0.5
    avg = sum(regime_outcome_quality.values()) / len(regime_outcome_quality)
    return max(0.0, min(1.0, (avg + 1.0) / 2.0))


def _historical_support(
    state: BrainState, side: BeliefDirection, opportunity_direction: BeliefDirection
) -> float:
    """Historical-outcomes reading for one side of the debate: the
    symbol-wide baseline, blended (in proportion to learned trust) with
    the behavioral profile's continuation-vs-reversal skew — the side
    that agrees with the tracked opportunity's own direction is scored
    against learned trend persistence, the opposing side against learned
    reversal tendency. Never recomputes either signal; both are read
    as-is from already-learned data.
    """
    base = _history_baseline(state.personality.regime_outcome_quality)
    profile = state.behavioral_profile
    trust = profile.trust()
    is_continuation_side = side == opportunity_direction
    skew = profile.trend_persistence if is_continuation_side else profile.reversal_tendency
    return max(0.0, min(1.0, (1 - trust) * base + trust * skew))


class ThesisManager:
    """Keeps an opportunity's `bull_thesis` / `bear_thesis` in sync with
    its tracked hypothesis and the symbol's broader hypothesis set and
    learned history. Stateless — operates on the `Thesis` objects owned
    by `Opportunity.bull_thesis` / `Opportunity.bear_thesis`, the same
    storage-ownership convention as `OpportunityTracker`/`HypothesisTracker`.
    """

    def sync(
        self,
        state: BrainState,
        opportunity: "Opportunity",
        hypotheses: dict[str, Hypothesis],
        now: float | None = None,
    ) -> tuple[Thesis, Thesis]:
        """Recompute both sides of the debate for `opportunity` from its
        tracked hypothesis's own evidence (reprojected onto both sides)
        plus any opposing-direction sibling hypotheses on the same
        symbol, and refresh each side's historical-support reading.
        Mutates and returns `(opportunity.bull_thesis, opportunity.bear_thesis)`.
        """
        now = now if now is not None else time.time()
        bull, bear = opportunity.bull_thesis, opportunity.bear_thesis
        bull.side, bear.side = BeliefDirection.BULLISH, BeliefDirection.BEARISH
        bull.supporting_evidence, bull.conflicting_evidence = [], []
        bear.supporting_evidence, bear.conflicting_evidence = [], []

        tracked = hypotheses.get(opportunity.hypothesis_id)
        if tracked is not None and tracked.direction in (
            BeliefDirection.BULLISH,
            BeliefDirection.BEARISH,
        ):
            aligned, opposing = (
                (bull, bear) if tracked.direction == BeliefDirection.BULLISH else (bear, bull)
            )
            aligned.argument = tracked.thesis
            aligned.supporting_evidence = list(tracked.supporting_evidence)
            aligned.conflicting_evidence = list(tracked.contradicting_evidence)
            # A hypothesis's own contradicting evidence for its direction
            # is, by definition, supporting evidence for the other side.
            opposing.supporting_evidence = list(tracked.contradicting_evidence)
            opposing.conflicting_evidence = list(tracked.supporting_evidence)
            aligned.confidence = tracked.confidence
            opposing.confidence = max(0.0, 1.0 - tracked.confidence)

        for other in hypotheses.values():
            if other.hypothesis_id == opportunity.hypothesis_id:
                continue
            if other.direction == BeliefDirection.BULLISH:
                bull.supporting_evidence.extend(other.supporting_evidence)
                bear.conflicting_evidence.extend(other.supporting_evidence)
            elif other.direction == BeliefDirection.BEARISH:
                bear.supporting_evidence.extend(other.supporting_evidence)
                bull.conflicting_evidence.extend(other.supporting_evidence)

        bull.supporting_evidence = bull.supporting_evidence[-MAX_EVIDENCE_ITEMS:]
        bull.conflicting_evidence = bull.conflicting_evidence[-MAX_EVIDENCE_ITEMS:]
        bear.supporting_evidence = bear.supporting_evidence[-MAX_EVIDENCE_ITEMS:]
        bear.conflicting_evidence = bear.conflicting_evidence[-MAX_EVIDENCE_ITEMS:]

        bull.historical_support = _historical_support(
            state, BeliefDirection.BULLISH, opportunity.direction
        )
        bear.historical_support = _historical_support(
            state, BeliefDirection.BEARISH, opportunity.direction
        )
        bull.updated_at = now
        bear.updated_at = now
        return bull, bear


__all__ = ["Thesis", "ThesisManager", "MAX_EVIDENCE_ITEMS"]
