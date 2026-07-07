"""Phase 19.3 — HRSA Benchmark Engine.

Compares a model's HierarchicalScore against multiple reference points:

  Champion                — current production champion
  Previous Champion       — last displaced champion
  Historical Average      — mean of all recorded champions
  Current Generation      — models in current training generation
  Previous Generation     — models from prior training generation
  Baseline Models         — fixed institutional reference points
  Public Benchmarks       — externally published benchmarks
  Internal Benchmarks     — firm-specific historical targets

Usage::

    engine = HRSABenchmarkEngine()
    engine.register("champion_v5", score_obj, role="champion")
    engine.register("baseline_2025", score_obj, role="baseline",
                    generation="2025")

    suite = engine.compare("candidate_v6", score_obj)
    print(suite.vs_champion())

    report = engine.full_report()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# BenchmarkRole constants
# ---------------------------------------------------------------------------


class BenchmarkRole:
    CHAMPION = "champion"
    PREV_CHAMPION = "previous_champion"
    BASELINE = "baseline"
    CURRENT_GEN = "current_generation"
    PREV_GEN = "previous_generation"
    PUBLIC = "public_benchmark"
    INTERNAL = "internal_benchmark"
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    ALL = [
        CHAMPION,
        PREV_CHAMPION,
        BASELINE,
        CURRENT_GEN,
        PREV_GEN,
        PUBLIC,
        INTERNAL,
        CANDIDATE,
        EXPERIMENTAL,
    ]


# ---------------------------------------------------------------------------
# BenchmarkRecord
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRecord:
    experiment_id: str
    name: str
    role: str
    composite: float
    confidence: float
    institutional_grade: str
    category_scores: dict[str, float]  # category → score (0–100)
    generation: str = ""
    registered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_hierarchical_score(cls, hs: Any, role: str, generation: str = "") -> BenchmarkRecord:
        return cls(
            experiment_id=hs.experiment_id,
            name=hs.name,
            role=role,
            composite=hs.level3_composite,
            confidence=hs.confidence,
            institutional_grade=hs.institutional_grade,
            category_scores=hs.category_scores_dict(),
            generation=generation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "role": self.role,
            "composite": round(self.composite, 2),
            "confidence": round(self.confidence, 4),
            "institutional_grade": self.institutional_grade,
            "category_scores": {k: round(v, 2) for k, v in self.category_scores.items()},
            "generation": self.generation,
            "registered_at": self.registered_at,
        }


# Backward-compatible alias used by tests and external importers
HRSABenchmarkRecord = BenchmarkRecord


# ---------------------------------------------------------------------------
# BenchmarkDelta — one pairwise comparison
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkDelta:
    subject_id: str
    subject_name: str
    reference_id: str
    reference_name: str
    reference_role: str
    subject_score: float
    reference_score: float
    composite_delta: float
    improved: bool
    category_deltas: dict[str, float]
    stronger: list[str]  # categories where subject > reference + 2
    weaker: list[str]  # categories where subject < reference - 2
    compared_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": {
                "id": self.subject_id,
                "name": self.subject_name,
                "score": round(self.subject_score, 2),
            },
            "reference": {
                "id": self.reference_id,
                "name": self.reference_name,
                "role": self.reference_role,
                "score": round(self.reference_score, 2),
            },
            "composite_delta": round(self.composite_delta, 3),
            "improved": self.improved,
            "category_deltas": {k: round(v, 3) for k, v in self.category_deltas.items()},
            "stronger_categories": self.stronger,
            "weaker_categories": self.weaker,
            "compared_at": self.compared_at,
        }

    def summary(self) -> str:
        sign = "+" if self.improved else ""
        return (
            f"{self.subject_name} vs {self.reference_name} [{self.reference_role}]: "
            f"{sign}{self.composite_delta:+.2f} pts"
        )


# ---------------------------------------------------------------------------
# BenchmarkSuite — all comparisons for one subject
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSuite:
    subject_id: str
    subject_name: str
    subject_score: float
    deltas: list[BenchmarkDelta] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def vs_champion(self) -> BenchmarkDelta | None:
        for d in self.deltas:
            if d.reference_role == BenchmarkRole.CHAMPION:
                return d
        return None

    def vs_baseline(self) -> BenchmarkDelta | None:
        for d in self.deltas:
            if d.reference_role == BenchmarkRole.BASELINE:
                return d
        return None

    def beats_champion(self) -> bool:
        d = self.vs_champion()
        return d is not None and d.improved

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "subject_name": self.subject_name,
            "subject_score": round(self.subject_score, 2),
            "beats_champion": self.beats_champion(),
            "comparisons": [d.to_dict() for d in self.deltas],
            "total_compared": len(self.deltas),
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# HRSABenchmarkEngine
# ---------------------------------------------------------------------------


class HRSABenchmarkEngine:
    """Benchmark comparison engine for hierarchical research scores."""

    def __init__(self) -> None:
        self._records: dict[str, BenchmarkRecord] = {}
        self._champion_id: str | None = None
        self._prev_champion_id: str | None = None
        self._historical_composites: list[float] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        experiment_id: str,
        hierarchical_score: Any,
        role: str = BenchmarkRole.EXPERIMENTAL,
        generation: str = "",
    ) -> BenchmarkRecord:
        """Register a HierarchicalScore for benchmarking."""
        rec = BenchmarkRecord.from_hierarchical_score(
            hierarchical_score, role=role, generation=generation
        )
        self._records[experiment_id] = rec

        if role == BenchmarkRole.CHAMPION:
            if self._champion_id and self._champion_id != experiment_id:
                # Demote current champion to previous_champion
                if self._champion_id in self._records:
                    self._records[self._champion_id].role = BenchmarkRole.PREV_CHAMPION
                    self._prev_champion_id = self._champion_id
            self._champion_id = experiment_id
            self._historical_composites.append(rec.composite)

        return rec

    def register_raw(self, record: BenchmarkRecord) -> None:
        self._records[record.experiment_id] = record
        if record.role == BenchmarkRole.CHAMPION:
            self._champion_id = record.experiment_id

    def set_champion(self, experiment_id: str) -> None:
        if experiment_id not in self._records:
            raise KeyError(f"Not registered: {experiment_id}")
        if self._champion_id and self._champion_id != experiment_id:
            if self._champion_id in self._records:
                self._records[self._champion_id].role = BenchmarkRole.PREV_CHAMPION
                self._prev_champion_id = self._champion_id
        self._records[experiment_id].role = BenchmarkRole.CHAMPION
        self._champion_id = experiment_id
        self._historical_composites.append(self._records[experiment_id].composite)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def _make_delta(self, subject: BenchmarkRecord, reference: BenchmarkRecord) -> BenchmarkDelta:
        delta = subject.composite - reference.composite
        cat_deltas: dict[str, float] = {}
        stronger: list[str] = []
        weaker: list[str] = []
        all_cats = set(subject.category_scores) | set(reference.category_scores)
        for cat in sorted(all_cats):
            s = subject.category_scores.get(cat, 0.0)
            r = reference.category_scores.get(cat, 0.0)
            d = s - r
            cat_deltas[cat] = round(d, 3)
            if d >= 2.0:
                stronger.append(cat)
            elif d <= -2.0:
                weaker.append(cat)

        return BenchmarkDelta(
            subject_id=subject.experiment_id,
            subject_name=subject.name,
            reference_id=reference.experiment_id,
            reference_name=reference.name,
            reference_role=reference.role,
            subject_score=subject.composite,
            reference_score=reference.composite,
            composite_delta=round(delta, 3),
            improved=delta > 0,
            category_deltas=cat_deltas,
            stronger=stronger,
            weaker=weaker,
        )

    def compare(
        self,
        subject_id: str,
        hierarchical_score: Any,
        roles: list[str] | None = None,
    ) -> BenchmarkSuite:
        """Compare subject against all registered references (optionally filtered by role)."""
        subject_rec = BenchmarkRecord.from_hierarchical_score(
            hierarchical_score, role=BenchmarkRole.CANDIDATE
        )
        roles_filter = set(roles) if roles else None
        deltas: list[BenchmarkDelta] = []

        # Historical average as a virtual record
        if self._historical_composites and (
            roles_filter is None or "historical_average" in roles_filter
        ):
            avg = sum(self._historical_composites) / len(self._historical_composites)
            avg_cat_scores = {}
            if self._champion_id and self._champion_id in self._records:
                avg_cat_scores = {
                    k: v * 0.95 for k, v in self._records[self._champion_id].category_scores.items()
                }
            hist_rec = BenchmarkRecord(
                experiment_id="historical_avg",
                name="Historical Average",
                role="historical_average",
                composite=avg,
                confidence=0.8,
                institutional_grade="",
                category_scores=avg_cat_scores,
            )
            deltas.append(self._make_delta(subject_rec, hist_rec))

        for ref_id, ref_rec in self._records.items():
            if ref_id == subject_id:
                continue
            if roles_filter and ref_rec.role not in roles_filter:
                continue
            deltas.append(self._make_delta(subject_rec, ref_rec))

        return BenchmarkSuite(
            subject_id=subject_id,
            subject_name=hierarchical_score.name or subject_id,
            subject_score=hierarchical_score.level3_composite,
            deltas=deltas,
        )

    def compare_ids(self, subject_id: str, reference_id: str) -> BenchmarkDelta | None:
        """Compare two registered experiments."""
        sub = self._records.get(subject_id)
        ref = self._records.get(reference_id)
        if sub is None or ref is None:
            return None
        return self._make_delta(sub, ref)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def champion_record(self) -> BenchmarkRecord | None:
        if self._champion_id:
            return self._records.get(self._champion_id)
        return None

    def by_role(self, role: str) -> list[BenchmarkRecord]:
        return [r for r in self._records.values() if r.role == role]

    def full_report(self) -> dict[str, Any]:
        sorted_recs = sorted(self._records.values(), key=lambda r: r.composite, reverse=True)
        composites = [r.composite for r in sorted_recs]
        hist_avg = (
            sum(self._historical_composites) / len(self._historical_composites)
            if self._historical_composites
            else None
        )
        role_counts: dict[str, int] = {}
        for r in self._records.values():
            role_counts[r.role] = role_counts.get(r.role, 0) + 1

        return {
            "total_registered": len(self._records),
            "champion_id": self._champion_id,
            "previous_champion_id": self._prev_champion_id,
            "historical_average": round(hist_avg, 2) if hist_avg else None,
            "role_distribution": role_counts,
            "ranking": [r.to_dict() for r in sorted_recs],
            "score_range": {
                "min": round(min(composites), 2) if composites else 0.0,
                "max": round(max(composites), 2) if composites else 0.0,
                "mean": round(sum(composites) / len(composites), 2) if composites else 0.0,
            },
        }

    def all_records(self) -> list[BenchmarkRecord]:
        return list(self._records.values())
