"""Phase 19.2 — IRSE Leaderboard.

Leaderboard ranked by Research Score with multi-axis filtering.

Ranking axes:
  - Research Score (composite)
  - Validation Score
  - Trading Score
  - Efficiency Score
  - Robustness Score
  - Calibration Score
  - Overall Grade

Filters:
  - Instrument
  - Timeframe
  - Backbone
  - Architecture
  - Dataset version
  - Feature version
  - Model family

Usage::

    lb = IRSELeaderboard()
    lb.add(research_score)
    lb.add(research_score_2)

    top_10 = lb.top(10)
    by_trading = lb.rank_by("trading_performance")
    filtered = lb.filter(instrument="ES", timeframe="1h")
    report = lb.leaderboard_report()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# LeaderboardEntry
# ---------------------------------------------------------------------------


@dataclass
class IRSELeaderboardEntry:
    """One row in the IRSE leaderboard."""

    rank: int = 0
    score_id: str = ""
    experiment_id: str = ""
    model_id: str = ""
    name: str = ""
    composite: float = 0.0
    grade_label: str = ""
    promotion_eligible: bool = False
    deployment_eligible: bool = False
    confidence_score: float = 0.0

    # Per-category scores for secondary ranking
    validation_quality_score: float = 0.0
    trading_performance_score: float = 0.0
    robustness_score: float = 0.0
    calibration_score: float = 0.0
    generalization_score: float = 0.0
    efficiency_score: float = 0.0
    explainability_score: float = 0.0
    operational_readiness_score: float = 0.0

    # Metadata
    instrument: str = ""
    timeframe: str = ""
    backbone: str = ""
    architecture: str = ""
    dataset_version: str = ""
    feature_version: str = ""
    model_family: str = ""
    tags: List[str] = field(default_factory=list)

    scored_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score_id": self.score_id,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "name": self.name,
            "composite": round(self.composite, 2),
            "grade_label": self.grade_label,
            "promotion_eligible": self.promotion_eligible,
            "deployment_eligible": self.deployment_eligible,
            "confidence_score": round(self.confidence_score, 4),
            "validation_quality_score": round(self.validation_quality_score, 2),
            "trading_performance_score": round(self.trading_performance_score, 2),
            "robustness_score": round(self.robustness_score, 2),
            "calibration_score": round(self.calibration_score, 2),
            "generalization_score": round(self.generalization_score, 2),
            "efficiency_score": round(self.efficiency_score, 2),
            "explainability_score": round(self.explainability_score, 2),
            "operational_readiness_score": round(self.operational_readiness_score, 2),
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "backbone": self.backbone,
            "architecture": self.architecture,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_family": self.model_family,
            "tags": self.tags,
            "scored_at": self.scored_at,
        }

    @classmethod
    def from_research_score(cls, rs: Any) -> "IRSELeaderboardEntry":
        """Build from a ResearchScore object."""

        def _cat_score(cat: str) -> float:
            for cs in rs.category_scores:
                if cs.category == cat:
                    return cs.score
            return 0.0

        return cls(
            score_id=rs.score_id,
            experiment_id=rs.experiment_id,
            model_id=rs.model_id,
            name=rs.name,
            composite=rs.composite,
            grade_label=rs.grade_label,
            promotion_eligible=rs.promotion_eligible,
            deployment_eligible=rs.deployment_eligible,
            confidence_score=rs.confidence_score(),
            validation_quality_score=_cat_score("validation_quality"),
            trading_performance_score=_cat_score("trading_performance"),
            robustness_score=_cat_score("robustness"),
            calibration_score=_cat_score("calibration"),
            generalization_score=_cat_score("generalization"),
            efficiency_score=_cat_score("efficiency"),
            explainability_score=_cat_score("explainability"),
            operational_readiness_score=_cat_score("operational_readiness"),
            instrument=rs.instrument,
            timeframe=rs.timeframe,
            backbone=rs.backbone,
            architecture=rs.architecture,
            dataset_version=rs.dataset_version,
            feature_version=rs.feature_version,
            model_family=rs.model_family,
            tags=rs.tags,
            scored_at=rs.scored_at,
        )


# ---------------------------------------------------------------------------
# Ranking keys
# ---------------------------------------------------------------------------

_RANKING_KEYS: Dict[str, str] = {
    "research_score": "composite",
    "validation_score": "validation_quality_score",
    "trading_score": "trading_performance_score",
    "robustness_score": "robustness_score",
    "efficiency_score": "efficiency_score",
    "calibration_score": "calibration_score",
    "generalization_score": "generalization_score",
    "explainability_score": "explainability_score",
    "operational_score": "operational_readiness_score",
    "confidence": "confidence_score",
}

VALID_RANKING_AXES = list(_RANKING_KEYS)


# ---------------------------------------------------------------------------
# IRSELeaderboard
# ---------------------------------------------------------------------------


class IRSELeaderboard:
    """Research Score leaderboard with multi-axis ranking and filtering."""

    def __init__(self) -> None:
        self._entries: List[IRSELeaderboardEntry] = []

    # ------------------------------------------------------------------
    # Adding entries
    # ------------------------------------------------------------------

    def add(self, research_score: Any) -> IRSELeaderboardEntry:
        """Add a ResearchScore to the leaderboard."""
        entry = IRSELeaderboardEntry.from_research_score(research_score)
        # Remove existing entry for same experiment (update)
        self._entries = [e for e in self._entries if e.experiment_id != entry.experiment_id]
        self._entries.append(entry)
        return entry

    def add_entry(self, entry: IRSELeaderboardEntry) -> None:
        """Add a pre-built entry."""
        self._entries = [e for e in self._entries if e.experiment_id != entry.experiment_id]
        self._entries.append(entry)

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _sorted(
        self, axis: str = "research_score", descending: bool = True
    ) -> List[IRSELeaderboardEntry]:
        attr = _RANKING_KEYS.get(axis, "composite")
        ranked = sorted(
            self._entries,
            key=lambda e: getattr(e, attr, 0.0),
            reverse=descending,
        )
        for i, entry in enumerate(ranked, 1):
            entry.rank = i
        return ranked

    def top(self, n: int = 10, axis: str = "research_score") -> List[Dict[str, Any]]:
        """Return top-n entries ranked by axis."""
        return [e.to_dict() for e in self._sorted(axis)[:n]]

    def rank_by(self, axis: str) -> List[Dict[str, Any]]:
        """Return all entries ranked by the given axis."""
        if axis not in _RANKING_KEYS:
            raise ValueError(f"Unknown ranking axis: {axis!r}. " f"Available: {VALID_RANKING_AXES}")
        return [e.to_dict() for e in self._sorted(axis)]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        *,
        instrument: Optional[str] = None,
        timeframe: Optional[str] = None,
        backbone: Optional[str] = None,
        architecture: Optional[str] = None,
        dataset_version: Optional[str] = None,
        feature_version: Optional[str] = None,
        model_family: Optional[str] = None,
        grade_label: Optional[str] = None,
        promotion_eligible: Optional[bool] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> "IRSELeaderboard":
        """Return a new leaderboard containing only matching entries."""
        entries = list(self._entries)
        if instrument:
            entries = [e for e in entries if e.instrument == instrument]
        if timeframe:
            entries = [e for e in entries if e.timeframe == timeframe]
        if backbone:
            entries = [e for e in entries if e.backbone == backbone]
        if architecture:
            entries = [e for e in entries if e.architecture == architecture]
        if dataset_version:
            entries = [e for e in entries if e.dataset_version == dataset_version]
        if feature_version:
            entries = [e for e in entries if e.feature_version == feature_version]
        if model_family:
            entries = [e for e in entries if e.model_family == model_family]
        if grade_label:
            entries = [e for e in entries if e.grade_label == grade_label]
        if promotion_eligible is not None:
            entries = [e for e in entries if e.promotion_eligible == promotion_eligible]
        if min_score is not None:
            entries = [e for e in entries if e.composite >= min_score]
        if max_score is not None:
            entries = [e for e in entries if e.composite <= max_score]

        sub_lb = IRSELeaderboard()
        for e in entries:
            sub_lb.add_entry(e)
        return sub_lb

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best(self, axis: str = "research_score") -> Optional[Dict[str, Any]]:
        ranked = self._sorted(axis)
        return ranked[0].to_dict() if ranked else None

    def get_by_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        for e in self._entries:
            if e.experiment_id == experiment_id:
                return e.to_dict()
        return None

    def promotion_candidates(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._sorted("research_score") if e.promotion_eligible]

    def grade_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entries:
            counts[e.grade_label] = counts.get(e.grade_label, 0) + 1
        return counts

    def category_leaders(self) -> Dict[str, Optional[str]]:
        """Return the experiment_id of the leader in each category."""
        leaders: Dict[str, Optional[str]] = {}
        for axis in VALID_RANKING_AXES:
            ranked = self._sorted(axis)
            leaders[axis] = ranked[0].experiment_id if ranked else None
        return leaders

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def leaderboard_report(self, top_n: int = 20, axis: str = "research_score") -> Dict[str, Any]:
        ranked = self._sorted(axis)
        scores = [e.composite for e in ranked]
        return {
            "total_entries": len(ranked),
            "ranking_axis": axis,
            "top_entries": [e.to_dict() for e in ranked[:top_n]],
            "promotion_candidates": len([e for e in ranked if e.promotion_eligible]),
            "deployment_candidates": len([e for e in ranked if e.deployment_eligible]),
            "grade_distribution": self.grade_distribution(),
            "score_range": {
                "min": round(min(scores), 2) if scores else 0.0,
                "max": round(max(scores), 2) if scores else 0.0,
                "mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
            },
            "category_leaders": self.category_leaders(),
        }

    def all_entries(self) -> List[IRSELeaderboardEntry]:
        return list(self._entries)
