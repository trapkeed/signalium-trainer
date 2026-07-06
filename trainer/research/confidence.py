"""Phase 19.2 — Research Score Confidence.

Generates an independent confidence score for the Research Score itself.

This reflects how much trust we should place in the composite Research
Score, based on:

  - Sample size (number of validation folds / trade samples)
  - Validation coverage (fraction of regimes covered)
  - Market diversity (number of instruments / timeframes)
  - Regime diversity (number of distinct regimes tested)
  - Statistical confidence (e.g. p-value from fold variance)

The confidence score is in [0.0, 1.0].

Usage::

    inputs = ConfidenceInputs(
        n_validation_folds=12,
        n_trade_samples=10_000,
        regime_coverage=0.85,
        instrument_count=5,
        timeframe_count=3,
        p_value=0.02,
    )
    calc = ConfidenceCalculator()
    result = calc.compute(inputs)
    print(f"Confidence: {result.score:.0%} ({result.level})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trainer.research.categories import _clamp, _linear, _weighted_avg

# ---------------------------------------------------------------------------
# ConfidenceInputs
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceInputs:
    """Inputs that determine how much to trust the Research Score."""

    # Validation scale
    n_validation_folds: int = 0  # 12+ → high confidence
    n_trade_samples: int = 0  # 10000+ → high confidence

    # Coverage
    regime_coverage: float = 0.0  # 0–1 (fraction of known regimes covered)
    instrument_count: int = 0  # ≥ 5 → high confidence
    timeframe_count: int = 0  # ≥ 3 → high confidence

    # Statistical
    p_value: float = 1.0  # from hypothesis test (lower → more confident)
    fold_score_std: float = 1.0  # std dev of fold scores (lower → more confident)

    # Data quality
    data_quality_score: float = 0.0  # 0–1
    label_quality_score: float = 0.0  # 0–1

    # Environment reproducibility
    reproducibility_score: float = 0.0  # 0–1


# ---------------------------------------------------------------------------
# ConfidenceResult
# ---------------------------------------------------------------------------


@dataclass
class ConfidenceResult:
    score: float  # 0.0–1.0
    sub_scores: dict[str, float] = field(default_factory=dict)

    @property
    def level(self) -> str:
        if self.score >= 0.90:
            return "Very High"
        if self.score >= 0.75:
            return "High"
        if self.score >= 0.55:
            return "Moderate"
        if self.score >= 0.35:
            return "Low"
        return "Very Low"

    @property
    def percentage(self) -> float:
        return round(self.score * 100.0, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "percentage": self.percentage,
            "level": self.level,
            "sub_scores": self.sub_scores,
        }


# ---------------------------------------------------------------------------
# ConfidenceCalculator
# ---------------------------------------------------------------------------


class ConfidenceCalculator:
    """Compute an independent confidence score for the Research Score."""

    def compute(self, inputs: ConfidenceInputs) -> ConfidenceResult:
        sub: dict[str, float] = {}

        # Sample size: 6 folds → 50, 12+ → 100
        sub["fold_coverage"] = _clamp(inputs.n_validation_folds / 12.0 * 100.0)

        # Trade samples: 5000 → 50, 20000+ → 100
        sub["sample_size"] = _linear(float(inputs.n_trade_samples), 1000.0, 20000.0)

        # Regime coverage
        sub["regime_coverage"] = _clamp(inputs.regime_coverage * 100.0)

        # Instrument diversity: 3 → 60, 5 → 100
        sub["instrument_diversity"] = _clamp(min(inputs.instrument_count, 5) / 5.0 * 100.0)

        # Timeframe diversity: 2 → 67, 3+ → 100
        sub["timeframe_diversity"] = _clamp(min(inputs.timeframe_count, 3) / 3.0 * 100.0)

        # Statistical confidence: p ≤ 0.01 → 100, p ≥ 0.10 → 0
        if inputs.p_value <= 0.01:
            sub["statistical_confidence"] = 100.0
        elif inputs.p_value >= 0.10:
            sub["statistical_confidence"] = 0.0
        else:
            sub["statistical_confidence"] = _clamp((0.10 - inputs.p_value) / 0.09 * 100.0)

        # Fold score std: 0 → 100, 0.10+ → 0
        sub["fold_consistency"] = _clamp(max(0.0, (0.10 - inputs.fold_score_std) / 0.10 * 100.0))

        # Data quality
        sub["data_quality"] = _clamp(inputs.data_quality_score * 100.0)
        sub["label_quality"] = _clamp(inputs.label_quality_score * 100.0)

        # Reproducibility
        sub["reproducibility"] = _clamp(inputs.reproducibility_score * 100.0)

        composite_100 = _weighted_avg(
            [
                (sub["fold_coverage"], 0.15),
                (sub["sample_size"], 0.15),
                (sub["regime_coverage"], 0.15),
                (sub["instrument_diversity"], 0.10),
                (sub["timeframe_diversity"], 0.08),
                (sub["statistical_confidence"], 0.12),
                (sub["fold_consistency"], 0.10),
                (sub["data_quality"], 0.07),
                (sub["label_quality"], 0.05),
                (sub["reproducibility"], 0.03),
            ]
        )

        confidence = _clamp(composite_100) / 100.0
        return ConfidenceResult(score=confidence, sub_scores=sub)
