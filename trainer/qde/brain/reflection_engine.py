"""QDE Brain — trade reflection engine.

Per-trade post-mortem analysis, distinct from `reflection.ReflectionEngine`
(which periodically synthesizes an `insight` + `lessons` across *many*
decisions in a rolling window). `TradeReflectionEngine` instead analyzes
*one* just-completed decision in depth — the analysis the QDE should perform
"after every completed trade":

  - assumptions: what the decision was betting on (`primary_reasons` /
    `alternatives_considered`, already captured on `DecisionReasoning`).
  - reasoning quality: reused from `DecisionScore` (already computed by the
    decision-scoring pipeline) rather than re-derived here.
  - ignored evidence: `contradicting_factors` that were on record at
    decision time but didn't stop (or sufficiently discount) the trade.
  - false evidence: `primary_reasons` / `supporting_factors` that, in
    hindsight, did not hold up against the realized outcome.
  - surprise magnitude: delegated to `surprise_engine.compute_surprise`.

This module reads `DecisionRecord`/`DecisionReasoning`/`DecisionScore`
(already produced by `trainer.memory`) — it never re-implements decision
scoring, outcome tracking, or aggregate self-evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trainer.qde.brain.surprise_engine import SurpriseAssessment, compute_surprise

if TYPE_CHECKING:
    from trainer.memory.decision_models import DecisionRecord


@dataclass
class TradeReflection:
    """Post-mortem analysis of one completed decision."""

    decision_id: str
    instrument: str
    assumptions: list[str] = field(default_factory=list)
    reasoning_quality: float | None = None  # 0-1, from DecisionScore.overall if present
    reasoning_quality_notes: list[str] = field(default_factory=list)
    ignored_evidence: list[str] = field(default_factory=list)
    false_evidence: list[str] = field(default_factory=list)
    surprise: SurpriseAssessment | None = None
    lessons: list[str] = field(default_factory=list)
    was_win: bool | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "instrument": self.instrument,
            "assumptions": self.assumptions,
            "reasoning_quality": (
                round(self.reasoning_quality, 4) if self.reasoning_quality is not None else None
            ),
            "reasoning_quality_notes": self.reasoning_quality_notes,
            "ignored_evidence": self.ignored_evidence,
            "false_evidence": self.false_evidence,
            "surprise": self.surprise.to_dict() if self.surprise else None,
            "lessons": self.lessons,
            "was_win": self.was_win,
            "created_at": round(self.created_at, 3),
        }


class TradeReflectionEngine:
    """Stateless per-trade analyzer — no storage of its own, the same
    convention as `WorkingMemory`/`LongTermMemory`; callers own persistence
    of the returned `TradeReflection` (e.g. via `KnowledgeBase`, the same
    sink `reflection.ReflectionEngine` already writes to).
    """

    def reflect_on_trade(self, record: "DecisionRecord") -> TradeReflection | None:
        """Analyze one completed decision. Returns `None` if the decision
        has no recorded outcome yet — there is nothing to reflect on.
        """
        outcome = record.outcome
        if outcome is None:
            return None

        reasoning = record.reasoning
        was_win = bool(outcome.actual_direction_correct) or outcome.hit_tp

        assumptions = list(reasoning.primary_reasons)
        if reasoning.alternatives_considered:
            assumptions.append(
                "alternatives considered and rejected: "
                + "; ".join(reasoning.alternatives_considered)
            )

        reasoning_quality = record.score.overall if record.score is not None else None
        reasoning_quality_notes = list(record.score.notes) if record.score is not None else []

        # Ignored evidence: contradicting factors were on record but the
        # trade proceeded anyway. Always worth surfacing on a loss; on a
        # win it's a "got away with it" note rather than a lesson driver.
        ignored_evidence = list(reasoning.contradicting_factors) if not was_win else []

        # False evidence: the reasons/supporting factors that were relied
        # on but did not hold up given how the trade actually resolved.
        false_evidence = (
            list(reasoning.primary_reasons) + list(reasoning.supporting_factors)
            if (outcome.actual_direction_correct is False)
            else []
        )

        surprise = compute_surprise(record)

        lessons: list[str] = []
        if not was_win and ignored_evidence:
            lessons.append(
                f"{record.instrument}: contradicting evidence "
                f"({'; '.join(ignored_evidence)}) was present but not weighted heavily enough."
            )
        if false_evidence:
            lessons.append(
                f"{record.instrument}: the reasoning behind this decision "
                f"({'; '.join(false_evidence)}) did not hold up against the outcome."
            )
        if surprise is not None and surprise.magnitude >= 0.6:
            lessons.append(
                f"{record.instrument}: outcome was highly surprising ({surprise.level.value}) — "
                f"{surprise.explanation}."
            )
        if reasoning_quality is not None and reasoning_quality < 0.4 and was_win:
            lessons.append(
                f"{record.instrument}: won despite low-quality reasoning "
                f"(score={reasoning_quality:.2f}) — treat as luck, not confirmation."
            )
        if not lessons:
            lessons.append(
                f"{record.instrument}: outcome broadly matched the reasoning behind the decision."
            )

        return TradeReflection(
            decision_id=record.decision_id,
            instrument=record.instrument,
            assumptions=assumptions,
            reasoning_quality=reasoning_quality,
            reasoning_quality_notes=reasoning_quality_notes,
            ignored_evidence=ignored_evidence,
            false_evidence=false_evidence,
            surprise=surprise,
            lessons=lessons,
            was_win=was_win,
        )
