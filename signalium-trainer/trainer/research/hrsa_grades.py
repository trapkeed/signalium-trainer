"""Phase 19.3 — HRSA Grade Engine.

Extended grade engine that assigns multiple grade types:

  Institutional Grade  — overall quality for capital allocation
  Research Grade       — quality of the research methodology
  Promotion Recommendation — GO / REVIEW / HOLD / NO
  Risk Rating          — LOW / MEDIUM / HIGH / VERY_HIGH
  Future Potential     — STRONG / MODERATE / LIMITED / DECLINING

All grade systems are configurable. Organisations can define custom grades
by passing lists of GradeThreshold dicts or calling define_custom_grade().

Usage::

    engine = HRSAGradeEngine.default()

    institutional = engine.assign_institutional(87.5)
    research      = engine.assign_research(82.0, confidence=0.85)
    promotion     = engine.assign_promotion(87.5, institutional, category_scores)
    risk          = engine.assign_risk(category_scores)
    potential     = engine.assign_future_potential(trend_slope=1.2, score=87.5)

    full = engine.full_assessment(
        composite=87.5,
        confidence=0.85,
        category_scores=category_scores,
        trend_slope=1.2,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# GradeThreshold
# ---------------------------------------------------------------------------


@dataclass
class GradeThreshold:
    label: str
    min_score: float
    max_score: float = 100.0
    description: str = ""
    color: str = "gray"
    rank: int = 0  # 0 = best; higher = worse

    def contains(self, score: float) -> bool:
        return self.min_score <= score <= self.max_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "description": self.description,
            "color": self.color,
            "rank": self.rank,
        }


# ---------------------------------------------------------------------------
# Default grade tables
# ---------------------------------------------------------------------------

_INSTITUTIONAL_GRADES = [
    GradeThreshold("Institutional Elite", 95.0, 100.0,
                   "Highest institutional standard. Full capital allocation eligible.", "gold", 0),
    GradeThreshold("Institutional",       90.0, 94.999,
                   "Institutional grade. Production deployment eligible.", "green", 1),
    GradeThreshold("Production Ready",    85.0, 89.999,
                   "Production ready. Limited deployment with monitoring.", "blue", 2),
    GradeThreshold("Candidate",           80.0, 84.999,
                   "Strong candidate. Additional validation required.", "cyan", 3),
    GradeThreshold("Experimental",        70.0, 79.999,
                   "Experimental. Not production ready.", "yellow", 4),
    GradeThreshold("Research Only",        0.0, 69.999,
                   "Research phase only.", "red", 5),
    GradeThreshold("Retired",             -1.0, -0.001,
                   "Retired or invalidated.", "gray", 6),
]

_RESEARCH_GRADES = [
    GradeThreshold("Exceptional Research",  90.0, 100.0, "Exceptional methodology.", "gold", 0),
    GradeThreshold("High Quality Research", 80.0, 89.999, "High quality research.", "green", 1),
    GradeThreshold("Standard Research",     65.0, 79.999, "Meets research standards.", "blue", 2),
    GradeThreshold("Preliminary Research",  50.0, 64.999, "Preliminary findings.", "yellow", 3),
    GradeThreshold("Insufficient Research",  0.0, 49.999, "Insufficient methodology.", "red", 4),
]


# ---------------------------------------------------------------------------
# Specialised grade labels
# ---------------------------------------------------------------------------

class PromotionRecommendation:
    GO     = "GO"
    REVIEW = "REVIEW"
    HOLD   = "HOLD"
    NO     = "NO"


class RiskRating:
    LOW       = "LOW"
    MEDIUM    = "MEDIUM"
    HIGH      = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class FuturePotential:
    STRONG    = "STRONG"
    MODERATE  = "MODERATE"
    LIMITED   = "LIMITED"
    DECLINING = "DECLINING"


# ---------------------------------------------------------------------------
# GradeAssessment — full grade result for one score
# ---------------------------------------------------------------------------


@dataclass
class GradeAssessment:
    composite: float
    confidence: float

    institutional_grade: str = ""
    research_grade: str = ""
    promotion_recommendation: str = ""
    risk_rating: str = ""
    future_potential: str = ""

    # Individual grade descriptions
    institutional_description: str = ""
    research_description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite": self.composite,
            "confidence": self.confidence,
            "institutional_grade": self.institutional_grade,
            "research_grade": self.research_grade,
            "promotion_recommendation": self.promotion_recommendation,
            "risk_rating": self.risk_rating,
            "future_potential": self.future_potential,
            "institutional_description": self.institutional_description,
            "research_description": self.research_description,
        }


# ---------------------------------------------------------------------------
# HRSAGradeEngine
# ---------------------------------------------------------------------------


class HRSAGradeEngine:
    """Assigns multiple grade types to a hierarchical research score."""

    def __init__(
        self,
        institutional_grades: Optional[List[GradeThreshold]] = None,
        research_grades: Optional[List[GradeThreshold]] = None,
    ) -> None:
        self._inst = sorted(
            institutional_grades or _INSTITUTIONAL_GRADES,
            key=lambda g: g.min_score, reverse=True
        )
        self._res = sorted(
            research_grades or _RESEARCH_GRADES,
            key=lambda g: g.min_score, reverse=True
        )
        self._custom_grades: List[GradeThreshold] = []

    @classmethod
    def default(cls) -> "HRSAGradeEngine":
        return cls()

    def define_custom_grade(
        self,
        name: str,
        min_score: float,
        max_score: float = 100.0,
        description: str = "",
        color: str = "gray",
    ) -> None:
        """Add a custom grade threshold to the institutional grade system."""
        self._inst.append(GradeThreshold(
            name, min_score, max_score, description, color,
            rank=len(self._inst),
        ))
        self._inst.sort(key=lambda g: g.min_score, reverse=True)
        self._custom_grades.append(self._inst[0])

    # ------------------------------------------------------------------
    # Grade assignments
    # ------------------------------------------------------------------

    def assign_institutional(self, score: float) -> GradeThreshold:
        if score < 0:
            score = -1.0
        for g in self._inst:
            if score >= g.min_score:
                return g
        return self._inst[-1]

    def assign_research(
        self, score: float, confidence: float = 1.0
    ) -> GradeThreshold:
        """Research grade adjusts the score by confidence."""
        adjusted = score * (0.70 + 0.30 * confidence)
        for g in self._res:
            if adjusted >= g.min_score:
                return g
        return self._res[-1]

    def assign_promotion(
        self,
        score: float,
        institutional: GradeThreshold,
        category_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """Determine promotion recommendation."""
        if score >= 85.0 and institutional.rank <= 2:
            # Check for critical weaknesses
            if category_scores:
                weaknesses = [
                    cat for cat, s in category_scores.items()
                    if s < 50.0 and cat in (
                        "trading_performance", "robustness", "risk_management"
                    )
                ]
                if weaknesses:
                    return PromotionRecommendation.REVIEW
            return PromotionRecommendation.GO
        if score >= 80.0:
            return PromotionRecommendation.REVIEW
        if score >= 70.0:
            return PromotionRecommendation.HOLD
        return PromotionRecommendation.NO

    def assign_risk(
        self, category_scores: Optional[Dict[str, float]] = None
    ) -> str:
        """Compute risk rating from risk-sensitive category scores."""
        if not category_scores:
            return RiskRating.MEDIUM
        risk_cats = ["robustness", "risk_management", "model_stability", "calibration"]
        risk_scores = [
            category_scores.get(cat, 50.0)
            for cat in risk_cats
            if cat in category_scores
        ]
        if not risk_scores:
            return RiskRating.MEDIUM
        avg_risk = sum(risk_scores) / len(risk_scores)
        if avg_risk >= 80.0:
            return RiskRating.LOW
        if avg_risk >= 65.0:
            return RiskRating.MEDIUM
        if avg_risk >= 45.0:
            return RiskRating.HIGH
        return RiskRating.VERY_HIGH

    def assign_future_potential(
        self,
        trend_slope: float = 0.0,
        score: float = 0.0,
        confidence: float = 0.5,
    ) -> str:
        """Determine future potential from trend slope and current score."""
        if trend_slope > 1.5 and score >= 70:
            return FuturePotential.STRONG
        if trend_slope > 0.5 or (score >= 80 and confidence >= 0.70):
            return FuturePotential.MODERATE
        if trend_slope < -1.0 or score < 50:
            return FuturePotential.DECLINING
        return FuturePotential.LIMITED

    # ------------------------------------------------------------------
    # Full assessment
    # ------------------------------------------------------------------

    def full_assessment(
        self,
        composite: float,
        confidence: float = 0.5,
        category_scores: Optional[Dict[str, float]] = None,
        trend_slope: float = 0.0,
    ) -> GradeAssessment:
        inst = self.assign_institutional(composite)
        res  = self.assign_research(composite, confidence)
        promo = self.assign_promotion(composite, inst, category_scores)
        risk  = self.assign_risk(category_scores)
        future = self.assign_future_potential(trend_slope, composite, confidence)

        return GradeAssessment(
            composite=composite,
            confidence=confidence,
            institutional_grade=inst.label,
            research_grade=res.label,
            promotion_recommendation=promo,
            risk_rating=risk,
            future_potential=future,
            institutional_description=inst.description,
            research_description=res.description,
        )

    def all_institutional_grades(self) -> List[Dict[str, Any]]:
        return [g.to_dict() for g in sorted(self._inst, key=lambda g: g.min_score)]

    def all_research_grades(self) -> List[Dict[str, Any]]:
        return [g.to_dict() for g in sorted(self._res, key=lambda g: g.min_score)]
