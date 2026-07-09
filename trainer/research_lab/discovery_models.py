"""S20 — Discovery pipeline configuration and summary models.

Lightweight dataclasses shared across ExperimentCreator,
FeatureDiscoveryPipeline, and LabelDiscoveryEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiscoveryQuery:
    """Input parameters for a discovery run."""

    model_version: str | None = None
    """If set, restrict telemetry queries to this model version."""

    since: float | None = None
    """Unix timestamp lower bound for telemetry queries."""

    until: float | None = None
    """Unix timestamp upper bound for telemetry queries."""

    symbol: str | None = None
    """If set, restrict telemetry queries to this symbol."""

    min_failure_severity: float = 0.3
    """Minimum FailureReport severity to include in experiment proposals."""

    max_proposed_per_run: int = 5
    """Maximum new experiment proposals to create in a single run."""

    min_feature_samples: int = 5
    """Minimum sample count for a discovered feature to be persisted."""

    min_label_samples: int = 3
    """Minimum trade count to run a label type and persist the dataset."""

    limit_per_stream: int = 5000
    """Maximum telemetry events pulled per stream per run."""


@dataclass
class DiscoverySummary:
    """Return value from a full discovery run (all three pipelines)."""

    experiments_proposed: int = 0
    experiments_skipped_duplicate: int = 0
    features_discovered: int = 0
    label_datasets_created: int = 0
    errors: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    feature_ids: list[str] = field(default_factory=list)
    label_dataset_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiments_proposed": self.experiments_proposed,
            "experiments_skipped_duplicate": self.experiments_skipped_duplicate,
            "features_discovered": self.features_discovered,
            "label_datasets_created": self.label_datasets_created,
            "errors": self.errors,
            "experiment_ids": self.experiment_ids,
            "feature_ids": self.feature_ids,
            "label_dataset_ids": self.label_dataset_ids,
        }
