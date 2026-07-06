"""Phase 19.2 — IRSE Scoring Weights.

Configurable weight system for the Institutional Research Scoring Engine.

Default weighting (must sum to 1.0):

  Validation Quality   25%
  Trading Performance  25%
  Robustness           15%
  Calibration          10%
  Generalization       10%
  Efficiency            5%
  Explainability        5%
  Operational Readiness 5%

Weights can be loaded from a dict or YAML file without modifying code.

Usage::

    weights = ScoringWeights.default()
    weights = ScoringWeights.from_dict({"validation_quality": 0.30, ...})
    weights.validate()
    print(weights.as_dict())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Category names (used as dict keys and DB column names)
# ---------------------------------------------------------------------------

CATEGORY_VALIDATION_QUALITY = "validation_quality"
CATEGORY_TRADING_PERFORMANCE = "trading_performance"
CATEGORY_ROBUSTNESS = "robustness"
CATEGORY_CALIBRATION = "calibration"
CATEGORY_GENERALIZATION = "generalization"
CATEGORY_EFFICIENCY = "efficiency"
CATEGORY_EXPLAINABILITY = "explainability"
CATEGORY_OPERATIONAL_READINESS = "operational_readiness"

ALL_CATEGORIES = [
    CATEGORY_VALIDATION_QUALITY,
    CATEGORY_TRADING_PERFORMANCE,
    CATEGORY_ROBUSTNESS,
    CATEGORY_CALIBRATION,
    CATEGORY_GENERALIZATION,
    CATEGORY_EFFICIENCY,
    CATEGORY_EXPLAINABILITY,
    CATEGORY_OPERATIONAL_READINESS,
]


# ---------------------------------------------------------------------------
# ScoringWeights
# ---------------------------------------------------------------------------


@dataclass
class ScoringWeights:
    """Configurable weights for the 8 IRSE scoring categories.

    Weights must be non-negative and sum to 1.0 (within 1e-6 tolerance).
    """

    validation_quality: float = 0.25
    trading_performance: float = 0.25
    robustness: float = 0.15
    calibration: float = 0.10
    generalization: float = 0.10
    efficiency: float = 0.05
    explainability: float = 0.05
    operational_readiness: float = 0.05

    def validate(self) -> None:
        """Raise ValueError if weights are invalid."""
        weights = self.as_dict()
        for name, w in weights.items():
            if w < 0:
                raise ValueError(f"Weight {name!r} is negative: {w}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0 but sum to {total:.8f}. " f"Adjust: {weights}"
            )

    def as_dict(self) -> Dict[str, float]:
        return {
            CATEGORY_VALIDATION_QUALITY: self.validation_quality,
            CATEGORY_TRADING_PERFORMANCE: self.trading_performance,
            CATEGORY_ROBUSTNESS: self.robustness,
            CATEGORY_CALIBRATION: self.calibration,
            CATEGORY_GENERALIZATION: self.generalization,
            CATEGORY_EFFICIENCY: self.efficiency,
            CATEGORY_EXPLAINABILITY: self.explainability,
            CATEGORY_OPERATIONAL_READINESS: self.operational_readiness,
        }

    def weight_for(self, category: str) -> float:
        return self.as_dict().get(category, 0.0)

    @classmethod
    def default(cls) -> "ScoringWeights":
        """Return the default IRSE weight configuration."""
        return cls()

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ScoringWeights":
        """Create weights from a dict. Missing keys use default values."""
        defaults = cls()
        return cls(
            validation_quality=d.get(CATEGORY_VALIDATION_QUALITY, defaults.validation_quality),
            trading_performance=d.get(CATEGORY_TRADING_PERFORMANCE, defaults.trading_performance),
            robustness=d.get(CATEGORY_ROBUSTNESS, defaults.robustness),
            calibration=d.get(CATEGORY_CALIBRATION, defaults.calibration),
            generalization=d.get(CATEGORY_GENERALIZATION, defaults.generalization),
            efficiency=d.get(CATEGORY_EFFICIENCY, defaults.efficiency),
            explainability=d.get(CATEGORY_EXPLAINABILITY, defaults.explainability),
            operational_readiness=d.get(
                CATEGORY_OPERATIONAL_READINESS, defaults.operational_readiness
            ),
        )

    @classmethod
    def from_yaml_section(cls, section: Dict[str, Any]) -> "ScoringWeights":
        """Create from a parsed YAML section.

        Expected format::

            weights:
              validation_quality: 0.25
              trading_performance: 0.25
              ...
        """
        return cls.from_dict({k: float(v) for k, v in section.items()})

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2)

    def normalise(self) -> "ScoringWeights":
        """Return a copy where weights are normalised to sum exactly to 1.0."""
        d = self.as_dict()
        total = sum(d.values())
        if total <= 0:
            raise ValueError("Sum of weights is zero — cannot normalise.")
        return ScoringWeights.from_dict({k: v / total for k, v in d.items()})

    def __str__(self) -> str:
        lines = ["ScoringWeights:"]
        for cat, w in self.as_dict().items():
            lines.append(f"  {cat:<24} {w*100:5.1f}%")
        return "\n".join(lines)
