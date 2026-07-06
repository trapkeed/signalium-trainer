"""Phase 19.2 — IRSE Grade System.

Assigns human-readable grades to composite Research Scores.

Default grade thresholds (configurable):

  95–100   Institutional Elite
  90–94    Institutional
  85–89    Production Ready
  80–84    Candidate
  70–79    Experimental
  < 70     Research Only

Usage::

    gs = GradeSystem.default()
    grade = gs.assign(92.5)
    print(grade.label)        # "Institutional"
    print(grade.description)  # "..."
    print(grade.promotion_eligible)  # True

    custom = GradeSystem.from_dict([
        {"min_score": 90, "label": "Elite", "description": "...", "promotion_eligible": True},
        ...
    ])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------


@dataclass
class Grade:
    """A single grade level with its score range and metadata."""

    label: str
    min_score: float
    max_score: float
    description: str = ""
    promotion_eligible: bool = False
    deployment_eligible: bool = False
    color: str = "gray"

    def contains(self, score: float) -> bool:
        return self.min_score <= score <= self.max_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "description": self.description,
            "promotion_eligible": self.promotion_eligible,
            "deployment_eligible": self.deployment_eligible,
            "color": self.color,
        }


# ---------------------------------------------------------------------------
# GradeResult
# ---------------------------------------------------------------------------


@dataclass
class GradeResult:
    """Result of assigning a grade to a specific score."""

    score: float
    grade: Grade
    percentile: Optional[float] = None

    @property
    def label(self) -> str:
        return self.grade.label

    @property
    def promotion_eligible(self) -> bool:
        return self.grade.promotion_eligible

    @property
    def deployment_eligible(self) -> bool:
        return self.grade.deployment_eligible

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "grade_label": self.grade.label,
            "grade_min_score": self.grade.min_score,
            "grade_max_score": self.grade.max_score,
            "grade_description": self.grade.description,
            "promotion_eligible": self.promotion_eligible,
            "deployment_eligible": self.deployment_eligible,
            "color": self.grade.color,
            "percentile": self.percentile,
        }


# ---------------------------------------------------------------------------
# Default grade definitions
# ---------------------------------------------------------------------------


_DEFAULT_GRADES: List[Dict[str, Any]] = [
    {
        "label": "Institutional Elite",
        "min_score": 95.0,
        "max_score": 100.0,
        "description": (
            "Exceptional model meeting the highest institutional standards. "
            "Suitable for full capital allocation."
        ),
        "promotion_eligible": True,
        "deployment_eligible": True,
        "color": "gold",
    },
    {
        "label": "Institutional",
        "min_score": 90.0,
        "max_score": 94.999,
        "description": (
            "Institutional-grade model with strong validated performance. "
            "Eligible for production deployment."
        ),
        "promotion_eligible": True,
        "deployment_eligible": True,
        "color": "green",
    },
    {
        "label": "Production Ready",
        "min_score": 85.0,
        "max_score": 89.999,
        "description": (
            "Production-ready model with solid metrics. "
            "Eligible for limited deployment with monitoring."
        ),
        "promotion_eligible": True,
        "deployment_eligible": True,
        "color": "blue",
    },
    {
        "label": "Candidate",
        "min_score": 80.0,
        "max_score": 84.999,
        "description": (
            "Strong candidate model requiring additional validation "
            "before production deployment."
        ),
        "promotion_eligible": True,
        "deployment_eligible": False,
        "color": "cyan",
    },
    {
        "label": "Experimental",
        "min_score": 70.0,
        "max_score": 79.999,
        "description": (
            "Experimental model showing promise. "
            "Not ready for production; further research required."
        ),
        "promotion_eligible": False,
        "deployment_eligible": False,
        "color": "yellow",
    },
    {
        "label": "Research Only",
        "min_score": 0.0,
        "max_score": 69.999,
        "description": ("Below production threshold. " "Research and development only."),
        "promotion_eligible": False,
        "deployment_eligible": False,
        "color": "red",
    },
]


# ---------------------------------------------------------------------------
# GradeSystem
# ---------------------------------------------------------------------------


class GradeSystem:
    """Assigns grades to Research Scores based on configurable thresholds."""

    def __init__(self, grades: List[Grade]) -> None:
        # Sort descending by min_score so we check highest thresholds first
        self._grades = sorted(grades, key=lambda g: g.min_score, reverse=True)

    @classmethod
    def default(cls) -> "GradeSystem":
        """Return the default IRSE grade system."""
        return cls._from_dicts(_DEFAULT_GRADES)

    @classmethod
    def from_dict(cls, grade_dicts: List[Dict[str, Any]]) -> "GradeSystem":
        """Build from a list of grade definition dicts."""
        return cls._from_dicts(grade_dicts)

    @classmethod
    def _from_dicts(cls, dicts: List[Dict[str, Any]]) -> "GradeSystem":
        grades = [
            Grade(
                label=d["label"],
                min_score=float(d["min_score"]),
                max_score=float(d.get("max_score", 100.0)),
                description=d.get("description", ""),
                promotion_eligible=bool(d.get("promotion_eligible", False)),
                deployment_eligible=bool(d.get("deployment_eligible", False)),
                color=d.get("color", "gray"),
            )
            for d in dicts
        ]
        return cls(grades)

    def assign(self, score: float) -> GradeResult:
        """Assign a grade to a research score.  Raises ValueError if score < 0 or > 100."""
        if score < 0 or score > 100:
            raise ValueError(f"Score must be in [0, 100] but got {score}")
        for grade in self._grades:
            if score >= grade.min_score:
                return GradeResult(score=score, grade=grade)
        # Fallback — should not happen with default thresholds covering 0–100
        return GradeResult(score=score, grade=self._grades[-1])

    def promotion_threshold(self) -> float:
        """Return the minimum score required for promotion eligibility."""
        for grade in reversed(self._grades):
            if grade.promotion_eligible:
                return grade.min_score
        return 100.0

    def deployment_threshold(self) -> float:
        """Return the minimum score required for deployment eligibility."""
        for grade in reversed(self._grades):
            if grade.deployment_eligible:
                return grade.min_score
        return 100.0

    def all_grades(self) -> List[Dict[str, Any]]:
        return [g.to_dict() for g in sorted(self._grades, key=lambda g: g.min_score)]

    def grade_for_label(self, label: str) -> Optional[Grade]:
        for g in self._grades:
            if g.label == label:
                return g
        return None

    def summary(self) -> str:
        lines = ["IRSE Grade System:"]
        for g in sorted(self._grades, key=lambda g: g.min_score, reverse=True):
            promo = "✓ promotion" if g.promotion_eligible else "  research"
            lines.append(f"  {g.min_score:5.1f}+ → {g.label:<22} [{promo}]")
        return "\n".join(lines)
