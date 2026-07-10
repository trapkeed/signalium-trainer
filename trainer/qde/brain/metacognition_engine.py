"""QDE Brain — metacognition engine.

Continuously answers "how is QDE actually doing on this symbol right now",
in numeric form, from the existing decision/personality infrastructure —
never a parallel ledger:

  - Performance and calibration come from `trainer.memory.DecisionMemory`
    records (the same source `self_evaluation.SelfEvaluationEngine` and
    `reflection.ReflectionEngine` already read).
  - Regime weakness comes from `SymbolPersonality.regime_outcome_quality`
    (already maintained by `long_term_memory.LongTermMemory`), never
    recomputed here.
  - Confidence calibration is folded together with `CalibrationProfile`
    (the bias-correction layer `lesson_extractor.LessonExtractor` already
    writes to) — this is that profile's first real reader, closing the
    loop between "a lesson nudged the bias" and "behavior actually shifts".

This module is deliberately distinct from two existing, similarly-named
things:

  - `self_evaluation.SelfEvaluationEngine` produces a periodic, *textual*
    institutional report (strengths/weaknesses/grade) for human/dashboard
    consumption. `MetacognitionEngine` produces a compact, *numeric*
    signal meant to be recomputed on every tick and fed straight into
    `self_awareness_engine.SelfAwarenessEngine` — it has no prose.
  - `reflection.ReflectionEngine` is a periodic (hourly, rate-limited)
    narrative synthesis of *what happened*. `MetacognitionEngine` is the
    always-on "how well am I currently calibrated" gauge that drives
    *behavioral* adjustment, not narrative.

`MetacognitionEngine` itself never touches trade sizing, entry gating, or
any other live decision path — it only produces a `MetacognitiveAssessment`
for `self_awareness_engine` (or a dashboard) to act on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trainer.memory.decision_types import NEGATIVE_OUTCOMES, POSITIVE_OUTCOMES

if TYPE_CHECKING:
    from trainer.memory.decision_memory import DecisionMemory
    from trainer.qde.brain.brain_models import CalibrationProfile, SymbolPersonality

DEFAULT_LOOKBACK_LIMIT = 50
DEFAULT_RECENT_WINDOW = 15
MIN_SAMPLES_FOR_SIGNAL = 5  # below this, noise dominates — stay neutral

OVERCONFIDENCE_CONF_THRESHOLD = 0.75
OVERCONFIDENCE_LOSS_RATIO = 0.35  # baseline acceptable loss rate at high confidence
WEAK_REGIME_QUALITY = 0.4  # regime_outcome_quality below this counts as a weakness


@dataclass
class MetacognitiveAssessment:
    """A single tick's numeric self-evaluation for one symbol."""

    instrument: str
    sample_count: int = 0
    recent_win_rate: float | None = None
    baseline_win_rate: float | None = None
    performance_trend: float = 0.0  # recent - baseline, signed; negative = deteriorating
    calibration_error: float = 0.0  # |mean forecast confidence - actual win rate|, 0..1
    overconfidence_score: float = 0.0  # 0..1, 0 = no evidence of overconfidence
    weak_regimes: list[str] = field(default_factory=list)
    symbol_weakness: float = 0.0  # 0..1, higher = this symbol is currently a weak spot
    caution_level: float = 0.0  # 0 = confident/neutral, 1 = maximally cautious; composite signal
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "sample_count": self.sample_count,
            "recent_win_rate": self.recent_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "performance_trend": round(self.performance_trend, 4),
            "calibration_error": round(self.calibration_error, 4),
            "overconfidence_score": round(self.overconfidence_score, 4),
            "weak_regimes": list(self.weak_regimes),
            "symbol_weakness": round(self.symbol_weakness, 4),
            "caution_level": round(self.caution_level, 4),
            "generated_at": round(self.generated_at, 3),
        }


class MetacognitionEngine:
    """Computes a `MetacognitiveAssessment` from existing decision/personality
    state. Pure read + score — never mutates any of its inputs.
    """

    def assess(
        self,
        instrument: str,
        decision_memory: "DecisionMemory | None",
        personality: "SymbolPersonality | None" = None,
        calibration: "CalibrationProfile | None" = None,
        recent_window: int = DEFAULT_RECENT_WINDOW,
        lookback_limit: int = DEFAULT_LOOKBACK_LIMIT,
        now: float | None = None,
    ) -> MetacognitiveAssessment:
        now = now if now is not None else time.time()
        assessment = MetacognitiveAssessment(instrument=instrument, generated_at=now)

        records = []
        if decision_memory is not None:
            records = decision_memory.filter(instrument=instrument, limit=lookback_limit)

        scored = [r for r in records if r.has_outcome]
        assessment.sample_count = len(scored)

        if personality is not None:
            assessment.weak_regimes = sorted(
                regime
                for regime, quality in personality.regime_outcome_quality.items()
                if quality < WEAK_REGIME_QUALITY
            )

        if len(scored) < MIN_SAMPLES_FOR_SIGNAL:
            # Not enough completed trades to trust a performance signal yet —
            # stay neutral rather than reacting to noise.
            return assessment

        scored_sorted = sorted(scored, key=lambda r: r.timestamp)
        recent = scored_sorted[-recent_window:]
        baseline = scored_sorted[:-recent_window] or scored_sorted

        def _win_rate(rs: list) -> float:
            return sum(1 for r in rs if r.outcome.outcome_type in POSITIVE_OUTCOMES) / len(rs)

        assessment.recent_win_rate = round(_win_rate(recent), 4)
        assessment.baseline_win_rate = round(_win_rate(baseline), 4)
        assessment.performance_trend = round(
            assessment.recent_win_rate - assessment.baseline_win_rate, 4
        )

        confidences = [r.snapshot.forecast_confidence for r in recent if r.snapshot]
        if confidences:
            mean_conf = sum(confidences) / len(confidences)
            assessment.calibration_error = round(abs(mean_conf - assessment.recent_win_rate), 4)

        if calibration is not None:
            # A persistently overstated confidence_bias (negative) means raw
            # confidence has been running hot relative to reality — compound
            # that into the calibration-error signal rather than ignoring the
            # already-learned correction.
            assessment.calibration_error = round(
                min(
                    1.0, assessment.calibration_error + max(0.0, -calibration.confidence_bias) * 0.5
                ),
                4,
            )

        hi_conf = [
            r
            for r in recent
            if r.snapshot and r.snapshot.forecast_confidence >= OVERCONFIDENCE_CONF_THRESHOLD
        ]
        if hi_conf:
            hi_loss_rate = sum(
                1 for r in hi_conf if r.outcome.outcome_type in NEGATIVE_OUTCOMES
            ) / len(hi_conf)
            if hi_loss_rate > OVERCONFIDENCE_LOSS_RATIO:
                assessment.overconfidence_score = round(
                    min(
                        1.0,
                        (hi_loss_rate - OVERCONFIDENCE_LOSS_RATIO)
                        / (1.0 - OVERCONFIDENCE_LOSS_RATIO),
                    ),
                    4,
                )

        shortfall = max(0.0, 0.5 - assessment.recent_win_rate) * 2.0  # 0..1
        trend_penalty = max(0.0, -assessment.performance_trend) * 2.0  # 0..1
        assessment.symbol_weakness = round(min(1.0, 0.6 * shortfall + 0.4 * trend_penalty), 4)

        assessment.caution_level = round(
            min(
                1.0,
                max(
                    0.0,
                    0.45 * shortfall
                    + 0.25 * assessment.calibration_error
                    + 0.30 * assessment.overconfidence_score,
                ),
            ),
            4,
        )
        return assessment
