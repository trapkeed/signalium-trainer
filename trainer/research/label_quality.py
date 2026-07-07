"""Phase 19 — Label Quality Monitor.

Continuously evaluates label quality:

  - Noise measurement
  - Consistency tracking
  - Class imbalance monitoring
  - Prediction difficulty estimation
  - Label drift detection

Generates quality reports for research review.
No automated actions — advisory only.

Usage::

    lqm = LabelQualityMonitor()
    lqm.record_batch("labels_v3.2", labels=["buy","sell","no_trade","buy",...],
                      confidences=[0.78, 0.55, 0.91, 0.62, ...])
    report = lqm.get_quality_report("labels_v3.2")
"""

from __future__ import annotations

import math
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Label batch record
# ---------------------------------------------------------------------------


@dataclass
class LabelBatch:
    """One batch of labels submitted for quality evaluation."""

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label_version: str = ""
    labels: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    predicted: list[str] = field(default_factory=list)
    dataset_version: str = ""
    n_samples: int = 0
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if not self.n_samples:
            self.n_samples = len(self.labels)

    @property
    def class_distribution(self) -> dict[str, float]:
        n = len(self.labels)
        if n == 0:
            return {}
        c = Counter(self.labels)
        return {k: v / n for k, v in c.items()}

    @property
    def noise_estimate(self) -> float:
        """Low confidence → high estimated noise."""
        if not self.confidences:
            return 0.0
        avg_conf = sum(self.confidences) / len(self.confidences)
        return round(max(0.0, 1.0 - avg_conf), 4)

    @property
    def consistency(self) -> float:
        """Label+prediction agreement rate where both are available."""
        if not self.predicted or len(self.predicted) != len(self.labels):
            return 1.0
        n = len(self.labels)
        matches = sum(1 for l, p in zip(self.labels, self.predicted) if l == p)
        return round(matches / n, 4)

    @property
    def entropy(self) -> float:
        """Shannon entropy of class distribution (difficulty proxy)."""
        dist = self.class_distribution
        e = 0.0
        for p in dist.values():
            if p > 0:
                e -= p * math.log2(p)
        return round(e, 4)

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of most common to least common class."""
        dist = self.class_distribution
        if not dist:
            return 1.0
        vals = list(dist.values())
        if min(vals) == 0:
            return float("inf")
        return round(max(vals) / min(vals), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "label_version": self.label_version,
            "n_samples": self.n_samples,
            "class_distribution": self.class_distribution,
            "noise_estimate": self.noise_estimate,
            "consistency": self.consistency,
            "entropy": self.entropy,
            "imbalance_ratio": self.imbalance_ratio,
            "dataset_version": self.dataset_version,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Label quality report
# ---------------------------------------------------------------------------


@dataclass
class LabelQualityReport:
    """Aggregated label quality report for a label version."""

    label_version: str = ""
    n_batches: int = 0
    n_samples_total: int = 0
    avg_noise: float = 0.0
    avg_consistency: float = 1.0
    avg_entropy: float = 0.0
    avg_imbalance_ratio: float = 1.0
    class_distribution: dict[str, float] = field(default_factory=dict)
    drift_from_baseline: float = 0.0
    quality_score: float = 1.0  # 0-1, higher is better
    quality_grade: str = "A"  # A / B / C / D / F
    issues: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_version": self.label_version,
            "n_batches": self.n_batches,
            "n_samples_total": self.n_samples_total,
            "avg_noise": self.avg_noise,
            "avg_consistency": self.avg_consistency,
            "avg_entropy": self.avg_entropy,
            "avg_imbalance_ratio": self.avg_imbalance_ratio,
            "class_distribution": self.class_distribution,
            "drift_from_baseline": self.drift_from_baseline,
            "quality_score": self.quality_score,
            "quality_grade": self.quality_grade,
            "issues": self.issues,
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Label quality monitor
# ---------------------------------------------------------------------------


class LabelQualityMonitor:
    """Continuously monitors label quality across all label versions.

    Args:
        batch_window: Rolling batch window per label version.
        noise_threshold: Noise above this → quality issue.
        consistency_threshold: Consistency below this → quality issue.
        imbalance_threshold: Imbalance ratio above this → quality issue.
        drift_threshold: Distribution drift above this → quality issue.
    """

    def __init__(
        self,
        batch_window: int = 50,
        noise_threshold: float = 0.25,
        consistency_threshold: float = 0.70,
        imbalance_threshold: float = 5.0,
        drift_threshold: float = 0.15,
    ) -> None:
        self.batch_window = batch_window
        self.noise_threshold = noise_threshold
        self.consistency_threshold = consistency_threshold
        self.imbalance_threshold = imbalance_threshold
        self.drift_threshold = drift_threshold
        self._batches: dict[str, deque[LabelBatch]] = {}
        self._all_batches: dict[str, list[LabelBatch]] = {}
        self._baselines: dict[str, dict[str, float]] = {}

    def record_batch(
        self,
        label_version: str,
        labels: list[str] | None = None,
        confidences: list[float] | None = None,
        predicted: list[str] | None = None,
        dataset_version: str = "",
        notes: str = "",
    ) -> LabelBatch:
        """Record one label batch for quality evaluation."""
        batch = LabelBatch(
            label_version=label_version,
            labels=labels or [],
            confidences=confidences or [],
            predicted=predicted or [],
            dataset_version=dataset_version,
            notes=notes,
        )
        if label_version not in self._batches:
            self._batches[label_version] = deque(maxlen=self.batch_window)
            self._all_batches[label_version] = []
        self._batches[label_version].append(batch)
        self._all_batches[label_version].append(batch)

        if label_version not in self._baselines and len(self._batches[label_version]) >= 5:
            self._baselines[label_version] = batch.class_distribution
        return batch

    def set_baseline(self, label_version: str, distribution: dict[str, float]) -> None:
        self._baselines[label_version] = distribution

    def get_quality_report(self, label_version: str) -> LabelQualityReport:
        """Generate a quality report for a label version."""
        batches = list(self._batches.get(label_version, []))
        n = len(batches)
        if n == 0:
            return LabelQualityReport(label_version=label_version)

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        avg_noise = _mean([b.noise_estimate for b in batches])
        avg_cons = _mean([b.consistency for b in batches])
        avg_ent = _mean([b.entropy for b in batches])
        avg_imb = _mean([b.imbalance_ratio for b in batches if b.imbalance_ratio < float("inf")])
        total_samples = sum(b.n_samples for b in batches)

        combined_counts: dict[str, float] = {}
        for b in batches:
            for cls, frac in b.class_distribution.items():
                combined_counts[cls] = combined_counts.get(cls, 0.0) + frac
        n_b = len(batches)
        avg_dist = {k: v / n_b for k, v in combined_counts.items()}

        baseline = self._baselines.get(label_version, {})
        drift = 0.0
        if baseline:
            all_keys = set(avg_dist) | set(baseline)
            eps = 1e-9
            for k in all_keys:
                p = max(avg_dist.get(k, 0.0), eps)
                q = max(baseline.get(k, 0.0), eps)
                drift += abs(p - q)
            drift /= max(len(all_keys), 1)

        quality_score = (
            (1.0 - avg_noise) * 0.30
            + avg_cons * 0.30
            + (1.0 - min(drift, 1.0)) * 0.20
            + (1.0 - min((avg_imb - 1.0) / 10.0, 1.0)) * 0.20
        )
        quality_score = round(min(max(quality_score, 0.0), 1.0), 4)
        if quality_score >= 0.85:
            grade = "A"
        elif quality_score >= 0.70:
            grade = "B"
        elif quality_score >= 0.55:
            grade = "C"
        elif quality_score >= 0.40:
            grade = "D"
        else:
            grade = "F"

        issues = []
        if avg_noise > self.noise_threshold:
            issues.append(f"High noise: {avg_noise:.3f} > {self.noise_threshold}")
        if avg_cons < self.consistency_threshold:
            issues.append(f"Low consistency: {avg_cons:.3f} < {self.consistency_threshold}")
        if avg_imb > self.imbalance_threshold:
            issues.append(f"High imbalance ratio: {avg_imb:.2f} > {self.imbalance_threshold}")
        if drift > self.drift_threshold:
            issues.append(f"Label drift: {drift:.3f} > {self.drift_threshold}")

        return LabelQualityReport(
            label_version=label_version,
            n_batches=n,
            n_samples_total=total_samples,
            avg_noise=round(avg_noise, 4),
            avg_consistency=round(avg_cons, 4),
            avg_entropy=round(avg_ent, 4),
            avg_imbalance_ratio=round(avg_imb, 4),
            class_distribution=avg_dist,
            drift_from_baseline=round(drift, 4),
            quality_score=quality_score,
            quality_grade=grade,
            issues=issues,
        )

    def all_reports(self) -> dict[str, LabelQualityReport]:
        return {v: self.get_quality_report(v) for v in self._batches}

    def low_quality_versions(self, min_grade: str = "C") -> list[str]:
        """Return label versions below a quality grade threshold."""
        grades = ["A", "B", "C", "D", "F"]
        min_idx = grades.index(min_grade)
        result = []
        for version, report in self.all_reports().items():
            report_idx = grades.index(report.quality_grade)
            if report_idx > min_idx:
                result.append(version)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "total_versions": len(self._batches),
            "total_batches": sum(len(v) for v in self._all_batches.values()),
            "low_quality_versions": self.low_quality_versions(),
        }
