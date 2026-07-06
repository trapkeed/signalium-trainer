"""Phase 19.1 — Experiment Differencing.

Compare any two experiments, models, datasets, or feature sets and
automatically identify what changed.

Classes:

  ExperimentDiff   — full diff between two LineageNodes
  DatasetDiff      — diff between two DatasetSnapshots
  FeatureDiff      — diff between two FeatureSnapshots
  ModelDiff        — diff between two ModelSnapshots
  ValidationDiff   — diff between two ValidationSnapshots
  ExperimentDiffer — high-level differ that uses a LineageTree

Usage::

    differ = ExperimentDiffer(tree)
    diff = differ.compare_experiments(node_id_a, node_id_b)
    print(diff.summary())

    # Or compare snapshots directly
    ds_diff = DatasetDiff.compare(ds_a, ds_b)
    ms_diff = ModelDiff.compare(ms_a, ms_b)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trainer.research.snapshots import (
    DatasetSnapshot,
    FeatureSnapshot,
    ModelSnapshot,
    ValidationSnapshot,
)

# ---------------------------------------------------------------------------
# Primitive diff helpers
# ---------------------------------------------------------------------------


def _scalar_diff(a: Any, b: Any, name: str, threshold: float = 0.0) -> Optional[Dict[str, Any]]:
    """Return a diff entry for a scalar field if it changed meaningfully."""
    if isinstance(a, float) and isinstance(b, float):
        if abs(b - a) <= threshold:
            return None
        return {"field": name, "before": a, "after": b, "delta": round(b - a, 8)}
    if a != b:
        return {"field": name, "before": a, "after": b}
    return None


# ---------------------------------------------------------------------------
# DatasetDiff
# ---------------------------------------------------------------------------


@dataclass
class DatasetDiff:
    snapshot_a_id: str
    snapshot_b_id: str
    changes: Dict[str, Any] = field(default_factory=dict)
    is_identical: bool = False

    @classmethod
    def compare(cls, a: DatasetSnapshot, b: DatasetSnapshot) -> "DatasetDiff":
        raw = a.diff(b)
        return cls(
            snapshot_a_id=a.snapshot_id,
            snapshot_b_id=b.snapshot_id,
            changes=raw,
            is_identical=len(raw) == 0,
        )

    def summary(self) -> str:
        if self.is_identical:
            return "Datasets are identical."
        lines = ["Dataset differences:"]
        for k, v in self.changes.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_a_id": self.snapshot_a_id,
            "snapshot_b_id": self.snapshot_b_id,
            "is_identical": self.is_identical,
            "change_count": len(self.changes),
            "changes": self.changes,
        }


# ---------------------------------------------------------------------------
# FeatureDiff
# ---------------------------------------------------------------------------


@dataclass
class FeatureDiff:
    snapshot_a_id: str
    snapshot_b_id: str
    changes: Dict[str, Any] = field(default_factory=dict)
    is_identical: bool = False

    @classmethod
    def compare(cls, a: FeatureSnapshot, b: FeatureSnapshot) -> "FeatureDiff":
        raw = a.diff(b)
        return cls(
            snapshot_a_id=a.snapshot_id,
            snapshot_b_id=b.snapshot_id,
            changes=raw,
            is_identical=len(raw) == 0,
        )

    def summary(self) -> str:
        if self.is_identical:
            return "Feature sets are identical."
        lines = ["Feature differences:"]
        added = self.changes.get("features_added", [])
        removed = self.changes.get("features_removed", [])
        imp = self.changes.get("importance_changes", {})
        if added:
            lines.append(f"  Added features:   {added}")
        if removed:
            lines.append(f"  Removed features: {removed}")
        if imp:
            lines.append(f"  Importance shifts in {len(imp)} features")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_a_id": self.snapshot_a_id,
            "snapshot_b_id": self.snapshot_b_id,
            "is_identical": self.is_identical,
            "change_count": len(self.changes),
            "changes": self.changes,
        }


# ---------------------------------------------------------------------------
# ModelDiff
# ---------------------------------------------------------------------------


@dataclass
class ModelDiff:
    snapshot_a_id: str
    snapshot_b_id: str
    changes: Dict[str, Any] = field(default_factory=dict)
    is_identical: bool = False

    @classmethod
    def compare(cls, a: ModelSnapshot, b: ModelSnapshot) -> "ModelDiff":
        raw = a.diff(b)
        return cls(
            snapshot_a_id=a.snapshot_id,
            snapshot_b_id=b.snapshot_id,
            changes=raw,
            is_identical=len(raw) == 0,
        )

    def architecture_changed(self) -> bool:
        return "architecture" in self.changes or "backbone" in self.changes

    def hyperparameter_changed(self) -> bool:
        return any(
            k in self.changes for k in ("learning_rate", "dropout", "optimizer", "scheduler")
        )

    def summary(self) -> str:
        if self.is_identical:
            return "Models are identical."
        lines = ["Model differences:"]
        for k, v in self.changes.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_a_id": self.snapshot_a_id,
            "snapshot_b_id": self.snapshot_b_id,
            "is_identical": self.is_identical,
            "architecture_changed": self.architecture_changed(),
            "hyperparameter_changed": self.hyperparameter_changed(),
            "change_count": len(self.changes),
            "changes": self.changes,
        }


# ---------------------------------------------------------------------------
# ValidationDiff
# ---------------------------------------------------------------------------


@dataclass
class ValidationDiff:
    snapshot_a_id: str
    snapshot_b_id: str
    changes: Dict[str, Any] = field(default_factory=dict)
    is_identical: bool = False
    performance_improved: bool = False

    @classmethod
    def compare(cls, a: ValidationSnapshot, b: ValidationSnapshot) -> "ValidationDiff":
        raw = a.diff(b)
        improved = False
        acc_delta = raw.get("walk_forward_accuracy", {}).get("delta", 0.0)
        sharpe_delta = raw.get("sharpe_ratio", {}).get("delta", 0.0)
        if acc_delta > 0 or sharpe_delta > 0:
            improved = True
        return cls(
            snapshot_a_id=a.snapshot_id,
            snapshot_b_id=b.snapshot_id,
            changes=raw,
            is_identical=len(raw) == 0,
            performance_improved=improved,
        )

    def summary(self) -> str:
        if self.is_identical:
            return "Validation results are identical."
        lines = ["Validation differences:"]
        for k, v in self.changes.items():
            lines.append(f"  {k}: {v}")
        direction = "improved" if self.performance_improved else "degraded or unchanged"
        lines.append(f"  Overall performance: {direction}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_a_id": self.snapshot_a_id,
            "snapshot_b_id": self.snapshot_b_id,
            "is_identical": self.is_identical,
            "performance_improved": self.performance_improved,
            "change_count": len(self.changes),
            "changes": self.changes,
        }


# ---------------------------------------------------------------------------
# ExperimentDiff — full diff of two lineage nodes
# ---------------------------------------------------------------------------


@dataclass
class ExperimentDiff:
    """Complete comparison between two experiment lineage nodes."""

    node_id_a: str
    node_id_b: str
    node_name_a: str
    node_name_b: str
    compared_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    environment_diff: Dict[str, Any] = field(default_factory=dict)
    dataset_diff: Optional[DatasetDiff] = None
    feature_diff: Optional[FeatureDiff] = None
    model_diff: Optional[ModelDiff] = None
    validation_diff: Optional[ValidationDiff] = None

    cost_diff_usd: float = 0.0
    duration_diff_seconds: float = 0.0

    def total_changes(self) -> int:
        total = len(self.environment_diff)
        if self.dataset_diff:
            total += len(self.dataset_diff.changes)
        if self.feature_diff:
            total += len(self.feature_diff.changes)
        if self.model_diff:
            total += len(self.model_diff.changes)
        if self.validation_diff:
            total += len(self.validation_diff.changes)
        return total

    def performance_improved(self) -> bool:
        if self.validation_diff:
            return self.validation_diff.performance_improved
        return False

    def summary(self) -> str:
        lines = [
            f"=== Experiment Diff: {self.node_name_a} → {self.node_name_b} ===",
            f"Compared at: {self.compared_at}",
            f"Total changes: {self.total_changes()}",
            "",
        ]
        if self.environment_diff:
            lines.append("Environment changes:")
            for k, v in self.environment_diff.items():
                lines.append(f"  {k}: {v}")
            lines.append("")
        if self.dataset_diff and not self.dataset_diff.is_identical:
            lines.append(self.dataset_diff.summary())
            lines.append("")
        if self.feature_diff and not self.feature_diff.is_identical:
            lines.append(self.feature_diff.summary())
            lines.append("")
        if self.model_diff and not self.model_diff.is_identical:
            lines.append(self.model_diff.summary())
            lines.append("")
        if self.validation_diff and not self.validation_diff.is_identical:
            lines.append(self.validation_diff.summary())
            lines.append("")
        lines.append(
            f"Performance: {'improved' if self.performance_improved() else 'unchanged/degraded'}"
        )
        lines.append(f"Cost delta:  ${self.cost_diff_usd:+.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id_a": self.node_id_a,
            "node_id_b": self.node_id_b,
            "node_name_a": self.node_name_a,
            "node_name_b": self.node_name_b,
            "compared_at": self.compared_at,
            "total_changes": self.total_changes(),
            "performance_improved": self.performance_improved(),
            "cost_diff_usd": self.cost_diff_usd,
            "duration_diff_seconds": self.duration_diff_seconds,
            "environment_diff": self.environment_diff,
            "dataset_diff": self.dataset_diff.to_dict() if self.dataset_diff else None,
            "feature_diff": self.feature_diff.to_dict() if self.feature_diff else None,
            "model_diff": self.model_diff.to_dict() if self.model_diff else None,
            "validation_diff": (self.validation_diff.to_dict() if self.validation_diff else None),
        }


# ---------------------------------------------------------------------------
# ExperimentDiffer — high-level comparison using a LineageTree
# ---------------------------------------------------------------------------


class ExperimentDiffer:
    """Produce ExperimentDiff objects for any two nodes in a LineageTree."""

    def __init__(self, tree: Any) -> None:
        self._tree = tree

    def compare_experiments(self, node_id_a: str, node_id_b: str) -> ExperimentDiff:
        """Compare two experiment lineage nodes and return a full diff."""
        na = self._tree.get(node_id_a)
        nb = self._tree.get(node_id_b)
        if na is None:
            raise KeyError(f"Node not found: {node_id_a}")
        if nb is None:
            raise KeyError(f"Node not found: {node_id_b}")

        # Environment diff
        env_diff: Dict[str, Any] = {}
        if na.environment and nb.environment:
            ea, eb = na.environment.to_dict(), nb.environment.to_dict()
            for field_name in (
                "git_commit_hash",
                "git_branch",
                "python_version",
                "pytorch_version",
                "cuda_version",
                "os_name",
                "gpu_model",
                "cpu_model",
                "random_seed",
            ):
                va, vb = ea.get(field_name), eb.get(field_name)
                if va != vb:
                    env_diff[field_name] = {"before": va, "after": vb}

        # Dataset diff
        ds_diff: Optional[DatasetDiff] = None
        if na.dataset_snapshot and nb.dataset_snapshot:
            ds_diff = DatasetDiff.compare(na.dataset_snapshot, nb.dataset_snapshot)

        # Feature diff
        fs_diff: Optional[FeatureDiff] = None
        if na.feature_snapshot and nb.feature_snapshot:
            fs_diff = FeatureDiff.compare(na.feature_snapshot, nb.feature_snapshot)

        # Model diff
        ms_diff: Optional[ModelDiff] = None
        if na.model_snapshot and nb.model_snapshot:
            ms_diff = ModelDiff.compare(na.model_snapshot, nb.model_snapshot)

        # Validation diff
        vs_diff: Optional[ValidationDiff] = None
        if na.validation_snapshot and nb.validation_snapshot:
            vs_diff = ValidationDiff.compare(na.validation_snapshot, nb.validation_snapshot)

        # Cost / duration
        cost_diff = nb.total_cost_usd - na.total_cost_usd
        dur_a = na.environment.training_duration_seconds if na.environment else 0.0
        dur_b = nb.environment.training_duration_seconds if nb.environment else 0.0

        return ExperimentDiff(
            node_id_a=node_id_a,
            node_id_b=node_id_b,
            node_name_a=na.name,
            node_name_b=nb.name,
            environment_diff=env_diff,
            dataset_diff=ds_diff,
            feature_diff=fs_diff,
            model_diff=ms_diff,
            validation_diff=vs_diff,
            cost_diff_usd=round(cost_diff, 6),
            duration_diff_seconds=round(dur_b - dur_a, 2),
        )

    def compare_models(self, node_id_a: str, node_id_b: str) -> Optional[ModelDiff]:
        na = self._tree.get(node_id_a)
        nb = self._tree.get(node_id_b)
        if na and nb and na.model_snapshot and nb.model_snapshot:
            return ModelDiff.compare(na.model_snapshot, nb.model_snapshot)
        return None

    def compare_datasets(self, node_id_a: str, node_id_b: str) -> Optional[DatasetDiff]:
        na = self._tree.get(node_id_a)
        nb = self._tree.get(node_id_b)
        if na and nb and na.dataset_snapshot and nb.dataset_snapshot:
            return DatasetDiff.compare(na.dataset_snapshot, nb.dataset_snapshot)
        return None

    def compare_features(self, node_id_a: str, node_id_b: str) -> Optional[FeatureDiff]:
        na = self._tree.get(node_id_a)
        nb = self._tree.get(node_id_b)
        if na and nb and na.feature_snapshot and nb.feature_snapshot:
            return FeatureDiff.compare(na.feature_snapshot, nb.feature_snapshot)
        return None
