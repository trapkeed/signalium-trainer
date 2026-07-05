"""Phase 19.2 — Score Trend Analysis.

Tracks Research Score evolution over time and detects:

  - Score history per experiment/model
  - Improvement trend (linear regression over score history)
  - Regression detection (score dropped from recent high)
  - Champion timeline (sequence of champion model changes)
  - Score distribution across all experiments

Usage::

    tracker = ScoreTrendTracker()
    tracker.record("exp_001", score=78.5, name="baseline_v1", model_family="transformer")
    tracker.record("exp_002", score=82.1, name="transformer_v2")
    tracker.record_champion_change("exp_002", "exp_001", score=82.1)

    trend = tracker.trend_report()
    regression = tracker.detect_regressions(window=5, threshold=5.0)
    champion_tl = tracker.champion_timeline()
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ScoreSnapshot
# ---------------------------------------------------------------------------


@dataclass
class ScoreSnapshot:
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    model_id: str = ""
    name: str = ""
    composite: float = 0.0
    grade_label: str = ""
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    category_scores: Dict[str, float] = field(default_factory=dict)
    model_family: str = ""
    backbone: str = ""
    is_champion: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "name": self.name,
            "composite": self.composite,
            "grade_label": self.grade_label,
            "recorded_at": self.recorded_at,
            "category_scores": self.category_scores,
            "model_family": self.model_family,
            "backbone": self.backbone,
            "is_champion": self.is_champion,
        }


# ---------------------------------------------------------------------------
# ChampionChange
# ---------------------------------------------------------------------------


@dataclass
class ChampionChange:
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    new_champion_id: str = ""
    old_champion_id: str = ""
    new_champion_score: float = 0.0
    old_champion_score: float = 0.0
    score_improvement: float = 0.0
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "new_champion_id": self.new_champion_id,
            "old_champion_id": self.old_champion_id,
            "new_champion_score": self.new_champion_score,
            "old_champion_score": self.old_champion_score,
            "score_improvement": self.score_improvement,
            "recorded_at": self.recorded_at,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# RegressionAlert
# ---------------------------------------------------------------------------


@dataclass
class RegressionAlert:
    experiment_id: str
    name: str
    current_score: float
    peak_score: float
    drop: float
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "current_score": self.current_score,
            "peak_score": self.peak_score,
            "drop": round(self.drop, 2),
            "severity": "high" if self.drop >= 10 else "medium" if self.drop >= 5 else "low",
            "detected_at": self.detected_at,
        }


# ---------------------------------------------------------------------------
# ScoreTrendTracker
# ---------------------------------------------------------------------------


class ScoreTrendTracker:
    """Tracks Research Score history and detects trends and regressions."""

    def __init__(self) -> None:
        self._history: List[ScoreSnapshot] = []
        self._champion_changes: List[ChampionChange] = []
        self._current_champion_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        experiment_id: str,
        score: float,
        *,
        model_id: str = "",
        name: str = "",
        grade_label: str = "",
        category_scores: Optional[Dict[str, float]] = None,
        model_family: str = "",
        backbone: str = "",
        is_champion: bool = False,
    ) -> ScoreSnapshot:
        snap = ScoreSnapshot(
            experiment_id=experiment_id,
            model_id=model_id,
            name=name,
            composite=score,
            grade_label=grade_label,
            category_scores=category_scores or {},
            model_family=model_family,
            backbone=backbone,
            is_champion=is_champion,
        )
        self._history.append(snap)
        return snap

    def record_from_research_score(self, rs: Any) -> ScoreSnapshot:
        """Record a snapshot directly from a ResearchScore object."""
        cat_scores = {cs.category: cs.score for cs in rs.category_scores}
        return self.record(
            rs.experiment_id,
            rs.composite,
            model_id=rs.model_id,
            name=rs.name,
            grade_label=rs.grade_label,
            category_scores=cat_scores,
            model_family=rs.model_family,
            backbone=rs.backbone,
        )

    def record_champion_change(
        self,
        new_champion_id: str,
        old_champion_id: str = "",
        new_score: float = 0.0,
        old_score: float = 0.0,
        reason: str = "",
    ) -> None:
        change = ChampionChange(
            new_champion_id=new_champion_id,
            old_champion_id=old_champion_id,
            new_champion_score=new_score,
            old_champion_score=old_score,
            score_improvement=round(new_score - old_score, 3),
            reason=reason,
        )
        self._champion_changes.append(change)
        self._current_champion_id = new_champion_id

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def score_history(
        self, experiment_id: Optional[str] = None
    ) -> List[ScoreSnapshot]:
        if experiment_id:
            return [s for s in self._history if s.experiment_id == experiment_id]
        return sorted(self._history, key=lambda s: s.recorded_at)

    def improvement_trend(self, window: int = 10) -> Dict[str, Any]:
        """Compute linear regression slope over the last `window` scores."""
        snaps = self.score_history()[-window:]
        if len(snaps) < 2:
            return {"slope": 0.0, "trend": "insufficient_data", "n": len(snaps)}
        scores = [s.composite for s in snaps]
        n = len(scores)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(scores) / n
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores))
        den = sum((xi - x_mean) ** 2 for xi in x)
        slope = num / den if den != 0 else 0.0
        trend = "improving" if slope > 0.5 else "declining" if slope < -0.5 else "stable"
        return {
            "slope": round(slope, 4),
            "trend": trend,
            "n": n,
            "score_min": min(scores),
            "score_max": max(scores),
            "score_mean": round(sum(scores) / n, 2),
            "score_latest": scores[-1],
        }

    def detect_regressions(
        self, window: int = 5, threshold: float = 5.0
    ) -> List[RegressionAlert]:
        """Detect experiments whose score dropped more than `threshold` points
        from their peak within the last `window` snapshots."""
        alerts: List[RegressionAlert] = []
        # Group by experiment
        by_exp: Dict[str, List[ScoreSnapshot]] = {}
        for snap in self._history:
            by_exp.setdefault(snap.experiment_id, []).append(snap)

        for exp_id, snaps in by_exp.items():
            if len(snaps) < 2:
                continue
            recent = sorted(snaps, key=lambda s: s.recorded_at)[-window:]
            if len(recent) < 2:
                continue
            peak = max(s.composite for s in recent)
            latest = recent[-1].composite
            drop = peak - latest
            if drop >= threshold:
                alerts.append(RegressionAlert(
                    experiment_id=exp_id,
                    name=recent[-1].name,
                    current_score=latest,
                    peak_score=peak,
                    drop=drop,
                ))
        return alerts

    def champion_timeline(self) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self._champion_changes]

    def current_champion_id(self) -> Optional[str]:
        return self._current_champion_id

    def score_distribution(self) -> Dict[str, Any]:
        scores = [s.composite for s in self._history]
        if not scores:
            return {"n": 0}
        n = len(scores)
        sorted_scores = sorted(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = variance ** 0.5

        def percentile(p: float) -> float:
            idx = p / 100.0 * (n - 1)
            lo, hi = int(idx), min(int(idx) + 1, n - 1)
            return sorted_scores[lo] + (idx - lo) * (sorted_scores[hi] - sorted_scores[lo])

        grade_counts: Dict[str, int] = {}
        for s in self._history:
            grade_counts[s.grade_label] = grade_counts.get(s.grade_label, 0) + 1

        return {
            "n": n,
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "p25": round(percentile(25), 2),
            "p50": round(percentile(50), 2),
            "p75": round(percentile(75), 2),
            "p90": round(percentile(90), 2),
            "grade_distribution": grade_counts,
        }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def trend_report(self) -> Dict[str, Any]:
        return {
            "total_scored_experiments": len(self._history),
            "champion_changes": len(self._champion_changes),
            "current_champion_id": self._current_champion_id,
            "improvement_trend": self.improvement_trend(),
            "score_distribution": self.score_distribution(),
            "champion_timeline": self.champion_timeline(),
            "recent_regressions": [
                a.to_dict() for a in self.detect_regressions()
            ],
        }

    def load_snapshot(self, snap: ScoreSnapshot) -> None:
        self._history.append(snap)

    def all_snapshots(self) -> List[ScoreSnapshot]:
        return list(self._history)
