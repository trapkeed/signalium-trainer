"""Phase 19 — Dataset Evolution Tracker.

Tracks every dataset version including:

  - Version
  - Coverage (date range)
  - Instrument count
  - Timeframe coverage
  - Market regime coverage
  - Feature schema
  - Label schema
  - Data quality score
  - Completeness

Stores full history and generates evolution reports.

Usage::

    det = DatasetEvolutionTracker()
    det.register(DatasetRecord(
        version="v4.0",
        n_instruments=45, n_samples=2_500_000,
        date_start="2018-01-01", date_end="2026-06-30",
        data_quality_score=0.97,
    ))
    report = det.evolution_report()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Dataset record
# ---------------------------------------------------------------------------


@dataclass
class DatasetRecord:
    """Complete metadata record for one dataset version.

    Attributes:
        dataset_id: Unique UUID.
        version: Dataset version string (e.g. "v4.0").
        description: Free-form description.
        date_start: Earliest data date (ISO-8601).
        date_end: Latest data date (ISO-8601).
        n_instruments: Number of instruments covered.
        n_samples: Total training samples.
        n_features: Number of features included.
        n_labels: Number of label classes.
        timeframes: List of covered timeframes (e.g. ["1H", "4H", "1D"]).
        instruments: List of covered instruments.
        regimes_covered: Market regimes present in the dataset.
        feature_schema_version: Feature engineering pipeline version.
        label_schema_version: Label generation pipeline version.
        data_quality_score: Composite data quality score (0-1).
        completeness: Fraction of expected rows present (0-1).
        status: active / deprecated / retired.
        parent_version: Previous dataset version (if iterative).
        changes_from_parent: Description of changes from parent version.
        registered_at: When the dataset was registered.
        metadata: Arbitrary extra metadata.
    """

    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = ""
    description: str = ""
    date_start: str = ""
    date_end: str = ""
    n_instruments: int = 0
    n_samples: int = 0
    n_features: int = 0
    n_labels: int = 3
    timeframes: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    regimes_covered: List[str] = field(default_factory=list)
    feature_schema_version: str = ""
    label_schema_version: str = ""
    data_quality_score: float = 0.0
    completeness: float = 1.0
    status: str = "active"  # active / deprecated / retired
    parent_version: Optional[str] = None
    changes_from_parent: str = ""
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def date_range_years(self) -> float:
        """Approximate number of years of data covered."""
        if not (self.date_start and self.date_end):
            return 0.0
        try:
            start = datetime.fromisoformat(self.date_start[:10])
            end = datetime.fromisoformat(self.date_end[:10])
            return (end - start).days / 365.25
        except (ValueError, AttributeError):
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "description": self.description,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "n_instruments": self.n_instruments,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_labels": self.n_labels,
            "timeframes": self.timeframes,
            "instruments": self.instruments,
            "regimes_covered": self.regimes_covered,
            "feature_schema_version": self.feature_schema_version,
            "label_schema_version": self.label_schema_version,
            "data_quality_score": self.data_quality_score,
            "completeness": self.completeness,
            "status": self.status,
            "is_active": self.is_active,
            "date_range_years": round(self.date_range_years, 2),
            "parent_version": self.parent_version,
            "changes_from_parent": self.changes_from_parent,
            "registered_at": self.registered_at,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Dataset quality change
# ---------------------------------------------------------------------------


@dataclass
class DatasetQualityChange:
    """Recorded change in dataset quality between versions."""

    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_version: str = ""
    to_version: str = ""
    quality_delta: float = 0.0
    completeness_delta: float = 0.0
    instrument_delta: int = 0
    sample_delta: int = 0
    feature_delta: int = 0
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "quality_delta": self.quality_delta,
            "completeness_delta": self.completeness_delta,
            "instrument_delta": self.instrument_delta,
            "sample_delta": self.sample_delta,
            "feature_delta": self.feature_delta,
            "description": self.description,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Dataset evolution tracker
# ---------------------------------------------------------------------------


class DatasetEvolutionTracker:
    """Tracks all dataset versions and their evolution over time.

    Stores full history, computes quality changes, and generates
    evolution reports for research review.
    """

    def __init__(self) -> None:
        self._datasets: Dict[str, DatasetRecord] = {}
        self._registration_order: List[str] = []
        self._changes: List[DatasetQualityChange] = []

    def register(self, record: DatasetRecord) -> str:
        """Register a new dataset version. Returns dataset_id."""
        self._datasets[record.version] = record
        self._registration_order.append(record.version)

        if record.parent_version and record.parent_version in self._datasets:
            parent = self._datasets[record.parent_version]
            change = DatasetQualityChange(
                from_version=record.parent_version,
                to_version=record.version,
                quality_delta=round(record.data_quality_score - parent.data_quality_score, 4),
                completeness_delta=round(record.completeness - parent.completeness, 4),
                instrument_delta=record.n_instruments - parent.n_instruments,
                sample_delta=record.n_samples - parent.n_samples,
                feature_delta=record.n_features - parent.n_features,
                description=record.changes_from_parent,
            )
            self._changes.append(change)
        return record.dataset_id

    def get(self, version: str) -> Optional[DatasetRecord]:
        return self._datasets.get(version)

    def get_or_raise(self, version: str) -> DatasetRecord:
        r = self._datasets.get(version)
        if r is None:
            raise KeyError(f"Dataset version {version!r} not found.")
        return r

    def deprecate(self, version: str, reason: str = "") -> bool:
        r = self._datasets.get(version)
        if r is None:
            return False
        r.status = "deprecated"
        r.metadata["deprecation_reason"] = reason
        return True

    def retire(self, version: str, reason: str = "") -> bool:
        r = self._datasets.get(version)
        if r is None:
            return False
        r.status = "retired"
        r.metadata["retirement_reason"] = reason
        return True

    def active_datasets(self) -> List[DatasetRecord]:
        return [d for d in self._datasets.values() if d.is_active]

    def latest(self) -> Optional[DatasetRecord]:
        if not self._registration_order:
            return None
        return self._datasets[self._registration_order[-1]]

    def evolution_report(self) -> Dict[str, Any]:
        """Generate a full dataset evolution report."""
        active = self.active_datasets()
        versions = [v for v in self._registration_order]
        quality_timeline = [
            {
                "version": v,
                "quality_score": self._datasets[v].data_quality_score,
                "n_samples": self._datasets[v].n_samples,
                "n_instruments": self._datasets[v].n_instruments,
                "n_features": self._datasets[v].n_features,
                "completeness": self._datasets[v].completeness,
                "registered_at": self._datasets[v].registered_at,
            }
            for v in versions
            if v in self._datasets
        ]
        best = (
            max(self._datasets.values(), key=lambda d: d.data_quality_score)
            if self._datasets
            else None
        )
        return {
            "report_type": "dataset_evolution_report",
            "total_versions": len(self._datasets),
            "active_versions": len(active),
            "latest_version": self.latest().version if self.latest() else None,
            "best_quality_version": best.version if best else None,
            "quality_timeline": quality_timeline,
            "changes": [c.to_dict() for c in self._changes],
            "summary": self.summary(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def compare(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare two dataset versions."""
        a = self.get_or_raise(version_a)
        b = self.get_or_raise(version_b)
        return {
            "version_a": version_a,
            "version_b": version_b,
            "quality_delta": round(b.data_quality_score - a.data_quality_score, 4),
            "completeness_delta": round(b.completeness - a.completeness, 4),
            "instrument_delta": b.n_instruments - a.n_instruments,
            "sample_delta": b.n_samples - a.n_samples,
            "feature_delta": b.n_features - a.n_features,
            "new_timeframes": list(set(b.timeframes) - set(a.timeframes)),
            "removed_timeframes": list(set(a.timeframes) - set(b.timeframes)),
            "new_regimes": list(set(b.regimes_covered) - set(a.regimes_covered)),
            "removed_regimes": list(set(a.regimes_covered) - set(b.regimes_covered)),
        }

    def summary(self) -> Dict[str, Any]:
        if not self._datasets:
            return {"total_versions": 0}
        quality_scores = [d.data_quality_score for d in self._datasets.values()]
        avg_q = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        return {
            "total_versions": len(self._datasets),
            "active_versions": len(self.active_datasets()),
            "avg_quality_score": round(avg_q, 4),
            "best_quality": round(max(quality_scores, default=0.0), 4),
            "total_changes": len(self._changes),
        }
