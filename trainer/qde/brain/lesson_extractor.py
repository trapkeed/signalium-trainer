"""QDE Brain — lesson extractor.

Turns a `TradeReflection` (+ its `SurpriseAssessment`) into concrete updates
across the Brain's existing learning surfaces:

  - symbol personalities: via `LongTermMemory.record_experience` (reused,
    not duplicated) — an `ExperienceRecord` is derived from the reflection.
  - conviction calibration / uncertainty estimates: via `BrainState.calibration`
    (`CalibrationProfile.nudge`), the durable bias-correction layer.
  - future hypotheses: a `LESSON` item is written into working memory (via
    the existing `WorkingMemory.add`) so hypothesis formation has this
    lesson in its context window, and, if the reflection implicates an
    active hypothesis directly, that hypothesis's `contradicting_evidence`
    is updated so future evidence weighing already accounts for it.

This module owns no state of its own and never re-implements personality
learning, decay, or hypothesis lifecycle transitions — it only routes a
reflection's findings to the modules that already own each concern.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trainer.qde.brain.brain_models import ExperienceRecord
from trainer.qde.brain.long_term_memory import LongTermMemory
from trainer.qde.brain.working_memory import WorkingMemory

if TYPE_CHECKING:
    from trainer.qde.brain.brain_models import BrainState
    from trainer.qde.brain.reflection_engine import TradeReflection

LESSON_WORKING_MEMORY_KIND = "LESSON"
CALIBRATION_ALPHA = 0.15


@dataclass
class LessonApplication:
    """What a `LessonExtractor.apply` call actually changed, for
    observability/testing — the extractor doesn't just act silently.
    """

    instrument: str
    experience: ExperienceRecord
    conviction_bias_delta: float = 0.0
    uncertainty_bias_delta: float = 0.0
    hypotheses_updated: list[str] = field(default_factory=list)
    lessons_recorded: list[str] = field(default_factory=list)
    applied_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "experience": self.experience.to_dict(),
            "conviction_bias_delta": round(self.conviction_bias_delta, 4),
            "uncertainty_bias_delta": round(self.uncertainty_bias_delta, 4),
            "hypotheses_updated": self.hypotheses_updated,
            "lessons_recorded": self.lessons_recorded,
            "applied_at": round(self.applied_at, 3),
        }


class LessonExtractor:
    """Stateless — operates on the `BrainState` passed in, same convention
    as `WorkingMemory`/`LongTermMemory`.
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory | None = None,
        working_memory: WorkingMemory | None = None,
    ) -> None:
        self._long_term_memory = long_term_memory or LongTermMemory()
        self._working_memory = working_memory or WorkingMemory()

    def apply(
        self,
        state: "BrainState",
        reflection: "TradeReflection",
        regime: str = "",
        now: float | None = None,
    ) -> LessonApplication:
        now = now if now is not None else time.time()

        outcome_quality = self._outcome_quality(reflection)
        tags = self._derive_tags(reflection)

        experience = ExperienceRecord(
            instrument=reflection.instrument,
            kind="TRADE_REFLECTION",
            description="; ".join(reflection.lessons) or "Trade reviewed — no notable lesson.",
            outcome_quality=outcome_quality,
            regime=regime,
            tags=tags,
        )
        self._long_term_memory.record_experience(state.personality, experience)

        conviction_delta = 0.0
        uncertainty_delta = 0.0
        if reflection.surprise is not None:
            conviction_delta, uncertainty_delta = self._calibration_deltas(reflection)
            if conviction_delta != 0.0:
                state.calibration.nudge("conviction_bias", conviction_delta, CALIBRATION_ALPHA)
            if uncertainty_delta != 0.0:
                state.calibration.nudge("uncertainty_bias", uncertainty_delta, CALIBRATION_ALPHA)
            state.calibration.instrument = state.calibration.instrument or reflection.instrument

        hypotheses_updated: list[str] = []
        if not reflection.was_win:
            for hyp in state.hypotheses.values():
                if hyp.instrument != reflection.instrument or hyp.is_terminal():
                    continue
                for note in reflection.ignored_evidence + reflection.false_evidence:
                    if note and note not in hyp.contradicting_evidence:
                        hyp.contradicting_evidence.append(note)
                if reflection.ignored_evidence or reflection.false_evidence:
                    hypotheses_updated.append(hyp.hypothesis_id)

        self._working_memory.add(
            state.working_memory,
            instrument=reflection.instrument,
            kind=LESSON_WORKING_MEMORY_KIND,
            content={
                "decision_id": reflection.decision_id,
                "lessons": reflection.lessons,
                "assumptions": reflection.assumptions,
                "surprise": reflection.surprise.to_dict() if reflection.surprise else None,
            },
            salience=0.6 if reflection.was_win else 0.8,
            now=now,
        )

        return LessonApplication(
            instrument=reflection.instrument,
            experience=experience,
            conviction_bias_delta=conviction_delta,
            uncertainty_bias_delta=uncertainty_delta,
            hypotheses_updated=hypotheses_updated,
            lessons_recorded=reflection.lessons,
        )

    def _outcome_quality(self, reflection: "TradeReflection") -> float:
        if reflection.was_win is None:
            return 0.0
        quality = 1.0 if reflection.was_win else -1.0
        if reflection.surprise is not None:
            # A surprising win is less reliable evidence of good process;
            # a surprising loss is a stronger signal something was missed.
            if reflection.was_win:
                quality *= 1.0 - 0.5 * reflection.surprise.magnitude
            else:
                quality *= 1.0 + 0.5 * reflection.surprise.magnitude
        return max(-1.0, min(1.0, quality))

    def _derive_tags(self, reflection: "TradeReflection") -> list[str]:
        tags = []
        if reflection.ignored_evidence:
            tags.append("IGNORED_EVIDENCE")
        if reflection.false_evidence:
            tags.append("FALSE_EVIDENCE")
        if reflection.surprise is not None and reflection.surprise.magnitude >= 0.6:
            tags.append("HIGH_SURPRISE")
        return tags

    def _calibration_deltas(self, reflection: "TradeReflection") -> tuple[float, float]:
        """Derive small signed nudges to conviction/uncertainty bias from
        how surprising this trade was. A high-surprise loss implies both
        conviction and uncertainty were mis-estimated (too confident, not
        wary enough); a low-surprise outcome implies the current estimates
        are well-calibrated and nudges toward zero drift.
        """
        surprise = reflection.surprise
        if surprise is None:
            return 0.0, 0.0

        if reflection.was_win:
            # Winning despite a surprising path — don't over-correct.
            return 0.0, surprise.magnitude * 0.3

        # Losing surprisingly: conviction should have been lower (negative
        # delta target pulls the bias down) and uncertainty should have
        # been read as higher (positive delta target pulls the bias up).
        conviction_target = -surprise.magnitude
        uncertainty_target = surprise.magnitude
        return conviction_target, uncertainty_target
