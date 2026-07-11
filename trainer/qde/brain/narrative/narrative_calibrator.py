"""QDE Brain — narrative confidence recalibration.

Adjusts the initial confidence of new ``NarrativeSeed`` objects and the
ongoing confidence of ``SymbolNarrative`` objects based on the historical
accuracy of each ``NarrativeKind`` for a specific instrument (or globally).

Why recalibration matters
--------------------------
The ``NarrativeBuilder`` detects patterns from structural rules (trend
validity, vol rank, readiness score, etc.) and assigns initial confidence
values that are sensible priors — but they may systematically over- or
under-estimate accuracy in a specific market or instrument.

For example:
  * If REVERSAL narratives for EURUSD have historically resolved correctly
    only 35% of the time, their initial confidence should be scaled down
    so the engine is naturally more skeptical of reversals on EURUSD.
  * If ACCUMULATION narratives have a 72% accuracy, their confidence can
    be scaled slightly up to reflect the pattern's demonstrated reliability.

The calibration is conservative: adjustments are bounded to ±30% of the
original confidence and only applied when enough samples exist (default 5).
Below the minimum sample threshold the calibration is a no-op (neutral
factor = 1.0) so the system never incorrectly penalises a kind it has not
observed enough.

How calibration factors are computed
--------------------------------------
Given a ``NarrativeQualityScore`` for a (instrument, kind) pair::

    factor = 0.5 + quality.reliability
                   ^ this gives 0.5..1.5 range (from reliability ∈ [0, 1])

Then clamped to [0.70, 1.30] to prevent extreme overcorrection.

When ``reliability = 0.5`` (no data, or 50% win with zero bias), factor = 1.0
(neutral, no adjustment).  This is the behaviour when there is no outcome
history.

``NarrativeCalibrator`` is stateless — it holds no per-symbol or per-kind
history itself.  The history lives in ``NarrativeOutcomeTracker``; the
calibrator simply converts a ``NarrativeQualityScore`` into a scaling factor
and applies it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from trainer.qde.brain.narrative.narrative_builder import NarrativeSeed
from trainer.qde.brain.narrative.narrative_outcome_tracker import (
    NarrativeQualityScore,
)
from trainer.qde.brain.narrative.narrative_store import (
    NarrativeKind,
    NarrativeMaturity,
    SymbolNarrative,
)

# ── Configuration ─────────────────────────────────────────────────────────────

_MIN_SAMPLES_FOR_ADJUSTMENT = 5  # below this, factor = 1.0 (no adjustment)
_FACTOR_FLOOR = 0.70  # never scale confidence below 70% of original
_FACTOR_CEIL = 1.30  # never scale confidence above 130% of original
_MIN_CONFIDENCE_FLOOR = 0.05  # floor on calibrated confidence
_MAX_CONFIDENCE_CEIL = 0.92  # ceiling on calibrated confidence


# ── Output type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationResult:
    """The output of one calibration call.

    ``original_confidence``   — confidence before calibration.
    ``calibrated_confidence`` — confidence after applying the factor.
    ``factor``                — the scaling factor applied (0.70–1.30, or 1.0 if
                                insufficient data).
    ``data_driven``           — True if the factor was computed from real history
                                (``sample_count >= min_samples``).
    ``quality_score``         — the ``NarrativeQualityScore`` used.
    """

    kind: NarrativeKind
    original_confidence: float
    calibrated_confidence: float
    factor: float
    data_driven: bool
    quality_score: NarrativeQualityScore | None = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "original_confidence": round(self.original_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "factor": round(self.factor, 4),
            "data_driven": self.data_driven,
            "quality_score": self.quality_score.to_dict() if self.quality_score else None,
        }


# ── Calibrator ────────────────────────────────────────────────────────────────


class NarrativeCalibrator:
    """Apply historical accuracy to adjust narrative confidence.

    Stateless: no per-symbol state is held here.  Call with a
    ``NarrativeQualityScore`` retrieved from ``NarrativeOutcomeTracker``.

    Usage::

        # At seed-creation time:
        quality = outcome_tracker.quality_score(instrument, seed.kind)
        calibrated_seed = calibrator.calibrate_seed(seed, quality)

        # For an already-active narrative:
        quality = outcome_tracker.quality_score(instrument, narrative.kind)
        new_conf = calibrator.recalibrate_confidence(narrative.confidence,
                                                      narrative.kind, quality)
    """

    def __init__(
        self,
        min_samples: int = _MIN_SAMPLES_FOR_ADJUSTMENT,
        factor_floor: float = _FACTOR_FLOOR,
        factor_ceil: float = _FACTOR_CEIL,
        conf_floor: float = _MIN_CONFIDENCE_FLOOR,
        conf_ceil: float = _MAX_CONFIDENCE_CEIL,
    ) -> None:
        self._min_samples = min_samples
        self._floor = factor_floor
        self._ceil = factor_ceil
        self._conf_floor = conf_floor
        self._conf_ceil = conf_ceil

    # ── Primary calibration entry points ─────────────────────────────────

    def calibrate_seed(
        self,
        seed: NarrativeSeed,
        quality: NarrativeQualityScore,
    ) -> NarrativeSeed:
        """Return a shallow-copy of ``seed`` with adjusted ``initial_confidence``.

        The original seed is never mutated.
        """
        result = self.compute_factor(seed.kind, quality)
        new_conf = self._apply(seed.initial_confidence, result.factor)
        # Build a modified copy — NarrativeSeed is a dataclass
        new_seed = copy.copy(seed)
        (
            object.__setattr__(new_seed, "initial_confidence", new_conf)
            if hasattr(new_seed, "__dataclass_fields__")
            else None
        )
        # Dataclasses are not frozen, so direct attribute set works
        new_seed.initial_confidence = new_conf
        return new_seed

    def recalibrate_confidence(
        self,
        confidence: float,
        kind: NarrativeKind,
        quality: NarrativeQualityScore,
    ) -> float:
        """Return a recalibrated confidence value.

        Typically called when an established narrative needs a quality-adjusted
        confidence for probability distribution computation.  Does NOT mutate
        the narrative in place — the caller decides whether to apply.
        """
        result = self.compute_factor(kind, quality)
        return self._apply(confidence, result.factor)

    def calibrate_narrative(
        self,
        narrative: SymbolNarrative,
        quality: NarrativeQualityScore,
        apply: bool = False,
    ) -> CalibrationResult:
        """Compute (and optionally apply) calibration to a live narrative.

        Parameters
        ----------
        narrative:
            The narrative to calibrate.  Never mutated unless ``apply=True``.
        quality:
            Quality score from ``NarrativeOutcomeTracker``.
        apply:
            If True, directly update ``narrative.confidence`` and
            ``narrative.last_updated_at`` in place.

        Returns
        -------
        CalibrationResult
            Full calibration details.  Caller can inspect ``factor``,
            ``data_driven``, and ``calibrated_confidence`` before deciding
            whether to apply.
        """
        result = self.compute_factor(narrative.kind, quality)
        new_conf = self._apply(narrative.confidence, result.factor)
        cr = CalibrationResult(
            kind=narrative.kind,
            original_confidence=narrative.confidence,
            calibrated_confidence=new_conf,
            factor=result.factor,
            data_driven=result.data_driven,
            quality_score=quality,
        )
        if apply and narrative.maturity != NarrativeMaturity.EXPIRED:
            narrative.confidence = new_conf
        return cr

    # ── Factor computation ────────────────────────────────────────────────

    def compute_factor(
        self,
        kind: NarrativeKind,
        quality: NarrativeQualityScore,
    ) -> CalibrationResult:
        """Compute the scaling factor for ``kind`` given ``quality``.

        The factor is 1.0 (neutral) when there is insufficient history.
        When data-driven, it is::

            factor = clamp(0.5 + quality.reliability, floor, ceil)

        ``quality.reliability`` ∈ [0, 1] maps to factor ∈ [0.5, 1.5],
        then clamped to [_FACTOR_FLOOR, _FACTOR_CEIL].

        A reliability of 0.5 (well-calibrated 50% accuracy) → factor = 1.0.
        A reliability of 0.8 → factor = 1.30 (upper bound).
        A reliability of 0.1 → factor = 0.70 (lower bound).
        """
        if quality.sample_count < self._min_samples:
            return CalibrationResult(
                kind=kind,
                original_confidence=0.0,
                calibrated_confidence=0.0,
                factor=1.0,
                data_driven=False,
                quality_score=quality,
            )

        raw_factor = 0.5 + quality.reliability
        factor = max(self._floor, min(self._ceil, raw_factor))

        return CalibrationResult(
            kind=kind,
            original_confidence=0.0,
            calibrated_confidence=0.0,
            factor=round(factor, 4),
            data_driven=True,
            quality_score=quality,
        )

    def factor_for_kind(
        self,
        kind: NarrativeKind,
        quality_by_kind: dict[str, NarrativeQualityScore],
    ) -> float:
        """Convenience wrapper: get the factor from a {kind.value: score} dict."""
        q = quality_by_kind.get(kind.value)
        if q is None:
            return 1.0
        return self.compute_factor(kind, q).factor

    # ── Internal ──────────────────────────────────────────────────────────

    def _apply(self, confidence: float, factor: float) -> float:
        scaled = confidence * factor
        return round(max(self._conf_floor, min(self._conf_ceil, scaled)), 4)


__all__ = [
    "CalibrationResult",
    "NarrativeCalibrator",
]
