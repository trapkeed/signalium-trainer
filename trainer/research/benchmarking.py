"""Phase 19.2 — IRSE Benchmarking.

Compare any model's Research Score against:

  - Current champion
  - Historical champions
  - Baseline models
  - Experimental models

Generates detailed score breakdowns showing exactly which
categories drive differences.

Usage::

    bm = IRSEBenchmarker()
    bm.register("exp_001", research_score_obj, role="baseline")
    bm.register("exp_002", research_score_obj, role="champion")
    bm.register("exp_003", research_score_obj, role="candidate")

    result = bm.compare("exp_003", against="exp_002")
    report = bm.benchmark_report("exp_003")
    full = bm.full_benchmark_report()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Model roles
# ---------------------------------------------------------------------------


class ModelRole:
    CHAMPION = "champion"
    BASELINE = "baseline"
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    HISTORICAL = "historical"
    ALL = [CHAMPION, BASELINE, CANDIDATE, EXPERIMENTAL, HISTORICAL]


# ---------------------------------------------------------------------------
# BenchmarkRecord — lightweight indexed version of a ResearchScore
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRecord:
    experiment_id: str
    model_id: str
    name: str
    role: str
    composite: float
    grade_label: str
    promotion_eligible: bool
    category_scores: Dict[str, float]  # category → score
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_research_score(cls, rs: Any, role: str = ModelRole.EXPERIMENTAL) -> "BenchmarkRecord":
        return cls(
            experiment_id=rs.experiment_id,
            model_id=rs.model_id,
            name=rs.name,
            role=role,
            composite=rs.composite,
            grade_label=rs.grade_label,
            promotion_eligible=rs.promotion_eligible,
            category_scores={cs.category: cs.score for cs in rs.category_scores},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "name": self.name,
            "role": self.role,
            "composite": round(self.composite, 2),
            "grade_label": self.grade_label,
            "promotion_eligible": self.promotion_eligible,
            "category_scores": {k: round(v, 2) for k, v in self.category_scores.items()},
            "registered_at": self.registered_at,
        }


# ---------------------------------------------------------------------------
# BenchmarkComparison
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkComparison:
    """Result of comparing two models."""

    comparison_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    subject_name: str = ""
    reference_id: str = ""
    reference_name: str = ""
    reference_role: str = ""
    compared_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Composite deltas
    subject_composite: float = 0.0
    reference_composite: float = 0.0
    composite_delta: float = 0.0
    composite_improved: bool = False

    # Per-category deltas
    category_deltas: Dict[str, float] = field(default_factory=dict)

    # Strengths and weaknesses
    stronger_categories: List[str] = field(default_factory=list)
    weaker_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "subject": {
                "id": self.subject_id,
                "name": self.subject_name,
                "score": self.subject_composite,
            },
            "reference": {
                "id": self.reference_id,
                "name": self.reference_name,
                "role": self.reference_role,
                "score": self.reference_composite,
            },
            "composite_delta": round(self.composite_delta, 3),
            "composite_improved": self.composite_improved,
            "category_deltas": {k: round(v, 3) for k, v in self.category_deltas.items()},
            "stronger_categories": self.stronger_categories,
            "weaker_categories": self.weaker_categories,
            "compared_at": self.compared_at,
        }

    def summary(self) -> str:
        direction = "+" if self.composite_improved else ""
        lines = [
            f"Benchmark: {self.subject_name} vs {self.reference_name} [{self.reference_role}]",
            f"  Composite delta: {direction}{self.composite_delta:+.2f}",
            f"  Stronger categories: {', '.join(self.stronger_categories) or 'none'}",
            f"  Weaker categories:   {', '.join(self.weaker_categories) or 'none'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IRSEBenchmarker
# ---------------------------------------------------------------------------


class IRSEBenchmarker:
    """IRSE benchmark comparison engine."""

    def __init__(self) -> None:
        self._records: Dict[str, BenchmarkRecord] = {}
        self._champion_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        experiment_id: str,
        research_score: Any,
        role: str = ModelRole.EXPERIMENTAL,
    ) -> BenchmarkRecord:
        """Register a model's ResearchScore for benchmarking."""
        record = BenchmarkRecord.from_research_score(research_score, role)
        self._records[experiment_id] = record
        if role == ModelRole.CHAMPION:
            self._champion_id = experiment_id
        return record

    def register_record(self, record: BenchmarkRecord) -> None:
        """Register a pre-built BenchmarkRecord."""
        self._records[record.experiment_id] = record
        if record.role == ModelRole.CHAMPION:
            self._champion_id = record.experiment_id

    def set_champion(self, experiment_id: str) -> None:
        if experiment_id not in self._records:
            raise KeyError(f"Experiment not registered: {experiment_id}")
        self._records[experiment_id].role = ModelRole.CHAMPION
        self._champion_id = experiment_id

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        subject_id: str,
        against: Optional[str] = None,
    ) -> BenchmarkComparison:
        """Compare subject experiment against a reference (default: champion)."""
        subject = self._records.get(subject_id)
        if subject is None:
            raise KeyError(f"Subject not registered: {subject_id}")

        reference_id = against or self._champion_id
        if reference_id is None:
            raise ValueError("No reference specified and no champion registered.")
        reference = self._records.get(reference_id)
        if reference is None:
            raise KeyError(f"Reference not registered: {reference_id}")

        delta = subject.composite - reference.composite
        cat_deltas: Dict[str, float] = {}
        stronger: List[str] = []
        weaker: List[str] = []

        all_cats = set(subject.category_scores) | set(reference.category_scores)
        for cat in sorted(all_cats):
            sub_s = subject.category_scores.get(cat, 0.0)
            ref_s = reference.category_scores.get(cat, 0.0)
            d = sub_s - ref_s
            cat_deltas[cat] = d
            if d >= 2.0:
                stronger.append(cat)
            elif d <= -2.0:
                weaker.append(cat)

        return BenchmarkComparison(
            subject_id=subject_id,
            subject_name=subject.name,
            reference_id=reference_id,
            reference_name=reference.name,
            reference_role=reference.role,
            subject_composite=subject.composite,
            reference_composite=reference.composite,
            composite_delta=round(delta, 3),
            composite_improved=delta > 0,
            category_deltas=cat_deltas,
            stronger_categories=stronger,
            weaker_categories=weaker,
        )

    def compare_all_against(self, reference_id: str) -> List[Dict[str, Any]]:
        """Compare every registered model against the reference."""
        results: List[Dict[str, Any]] = []
        for exp_id in self._records:
            if exp_id == reference_id:
                continue
            try:
                comp = self.compare(exp_id, against=reference_id)
                results.append(comp.to_dict())
            except (KeyError, ValueError):
                pass
        return sorted(results, key=lambda d: d["composite_delta"], reverse=True)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def benchmark_report(self, subject_id: str) -> Dict[str, Any]:
        """Full benchmark report for one experiment."""
        subject = self._records.get(subject_id)
        if subject is None:
            return {"error": f"Not registered: {subject_id}"}

        comparisons: List[Dict[str, Any]] = []
        for ref_id, ref_rec in self._records.items():
            if ref_id == subject_id:
                continue
            try:
                c = self.compare(subject_id, against=ref_id)
                comparisons.append(c.to_dict())
            except Exception:
                pass

        champion_comparison: Optional[Dict[str, Any]] = None
        if self._champion_id and self._champion_id != subject_id:
            try:
                champion_comparison = self.compare(subject_id, against=self._champion_id).to_dict()
            except Exception:
                pass

        return {
            "subject": subject.to_dict(),
            "champion_id": self._champion_id,
            "champion_comparison": champion_comparison,
            "all_comparisons": comparisons,
            "total_registered_models": len(self._records),
        }

    def full_benchmark_report(self) -> Dict[str, Any]:
        """Summary benchmark report for all registered models."""
        sorted_records = sorted(
            self._records.values(),
            key=lambda r: r.composite,
            reverse=True,
        )
        role_counts: Dict[str, int] = {}
        for rec in self._records.values():
            role_counts[rec.role] = role_counts.get(rec.role, 0) + 1

        composites = [r.composite for r in sorted_records]
        return {
            "total_registered": len(self._records),
            "champion_id": self._champion_id,
            "role_distribution": role_counts,
            "ranking": [r.to_dict() for r in sorted_records],
            "score_range": {
                "min": round(min(composites), 2) if composites else 0.0,
                "max": round(max(composites), 2) if composites else 0.0,
                "mean": round(sum(composites) / len(composites), 2) if composites else 0.0,
            },
        }

    def by_role(self, role: str) -> List[BenchmarkRecord]:
        return [r for r in self._records.values() if r.role == role]

    def champion_record(self) -> Optional[BenchmarkRecord]:
        if self._champion_id:
            return self._records.get(self._champion_id)
        return None
