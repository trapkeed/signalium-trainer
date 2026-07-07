"""Phase 19.1 — Experiment Snapshots.

Point-in-time snapshots of dataset, feature set, model, and validation
state — embedded in each experiment's lineage record so any historical
experiment can be reproduced with identical inputs.

Classes:

  DatasetSnapshot    — version, hash, instruments, timeframes, quality
  FeatureSnapshot    — definitions, parameters, importance, statistics
  ModelSnapshot      — architecture, backbone, optimizer, loss, hashes
  ValidationSnapshot — walk-forward results, calibration, trading metrics

Usage::

    ds = DatasetSnapshot(
        dataset_version="v2.3.1",
        dataset_hash="sha256:abc123",
        instruments=["ES", "NQ", "CL"],
        coverage_start="2018-01-01",
        coverage_end="2025-12-31",
        sample_count=1_250_000,
        data_quality_score=0.97,
    )
    ms = ModelSnapshot.from_dict(checkpoint_dict)
    diff = ds.diff(other_ds)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# DatasetSnapshot
# ---------------------------------------------------------------------------


@dataclass
class DatasetSnapshot:
    """Immutable record of a dataset's state at a point in time."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Identity
    dataset_version: str = ""
    dataset_hash: str = ""

    # Coverage
    instruments: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    coverage_start: str = ""
    coverage_end: str = ""
    sample_count: int = 0

    # Feature / Label versioning
    feature_version: str = ""
    feature_hash: str = ""
    label_version: str = ""
    label_hash: str = ""
    normalization_version: str = ""
    scaling_version: str = ""

    # Quality
    data_quality_score: float = 0.0

    # Arbitrary metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def diff(self, other: DatasetSnapshot) -> dict[str, Any]:
        """Return field-level differences between this and another snapshot."""
        changes: dict[str, Any] = {}
        for attr in (
            "dataset_version",
            "dataset_hash",
            "feature_version",
            "feature_hash",
            "label_version",
            "label_hash",
            "normalization_version",
            "scaling_version",
            "sample_count",
            "data_quality_score",
            "coverage_start",
            "coverage_end",
        ):
            a, b = getattr(self, attr), getattr(other, attr)
            if a != b:
                changes[attr] = {"before": a, "after": b}
        added_instruments = set(other.instruments) - set(self.instruments)
        removed_instruments = set(self.instruments) - set(other.instruments)
        if added_instruments or removed_instruments:
            changes["instruments"] = {
                "added": sorted(added_instruments),
                "removed": sorted(removed_instruments),
            }
        added_tf = set(other.timeframes) - set(self.timeframes)
        removed_tf = set(self.timeframes) - set(other.timeframes)
        if added_tf or removed_tf:
            changes["timeframes"] = {
                "added": sorted(added_tf),
                "removed": sorted(removed_tf),
            }
        return changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "dataset_version": self.dataset_version,
            "dataset_hash": self.dataset_hash,
            "instruments": self.instruments,
            "timeframes": self.timeframes,
            "coverage_start": self.coverage_start,
            "coverage_end": self.coverage_end,
            "sample_count": self.sample_count,
            "feature_version": self.feature_version,
            "feature_hash": self.feature_hash,
            "label_version": self.label_version,
            "label_hash": self.label_hash,
            "normalization_version": self.normalization_version,
            "scaling_version": self.scaling_version,
            "data_quality_score": self.data_quality_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetSnapshot:
        return cls(
            snapshot_id=d.get("snapshot_id", str(uuid.uuid4())),
            captured_at=d.get("captured_at", datetime.now(UTC).isoformat()),
            dataset_version=d.get("dataset_version", ""),
            dataset_hash=d.get("dataset_hash", ""),
            instruments=d.get("instruments", []),
            timeframes=d.get("timeframes", []),
            coverage_start=d.get("coverage_start", ""),
            coverage_end=d.get("coverage_end", ""),
            sample_count=d.get("sample_count", 0),
            feature_version=d.get("feature_version", ""),
            feature_hash=d.get("feature_hash", ""),
            label_version=d.get("label_version", ""),
            label_hash=d.get("label_hash", ""),
            normalization_version=d.get("normalization_version", ""),
            scaling_version=d.get("scaling_version", ""),
            data_quality_score=d.get("data_quality_score", 0.0),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# FeatureSnapshot
# ---------------------------------------------------------------------------


@dataclass
class FeatureSnapshot:
    """Immutable record of the feature engineering state."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    feature_count: int = 0
    feature_names: list[str] = field(default_factory=list)
    feature_definitions: dict[str, Any] = field(default_factory=dict)
    feature_parameters: dict[str, Any] = field(default_factory=dict)

    # Performance / quality metrics per feature
    feature_importance: dict[str, float] = field(default_factory=dict)
    feature_statistics: dict[str, Any] = field(default_factory=dict)
    feature_distributions: dict[str, Any] = field(default_factory=dict)
    feature_correlations: dict[str, float] = field(default_factory=dict)

    # Lifecycle state
    feature_drift_status: dict[str, str] = field(default_factory=dict)
    feature_lifecycle_status: dict[str, str] = field(default_factory=dict)

    # Version hash
    feature_set_hash: str = ""

    def diff(self, other: FeatureSnapshot) -> dict[str, Any]:
        """Return differences in feature sets."""
        changes: dict[str, Any] = {}
        added = set(other.feature_names) - set(self.feature_names)
        removed = set(self.feature_names) - set(other.feature_names)
        if added:
            changes["features_added"] = sorted(added)
        if removed:
            changes["features_removed"] = sorted(removed)
        importance_changes: dict[str, Any] = {}
        for name in set(self.feature_importance) & set(other.feature_importance):
            delta = other.feature_importance[name] - self.feature_importance[name]
            if abs(delta) > 0.005:
                importance_changes[name] = {
                    "before": self.feature_importance[name],
                    "after": other.feature_importance[name],
                    "delta": round(delta, 6),
                }
        if importance_changes:
            changes["importance_changes"] = importance_changes
        if self.feature_set_hash != other.feature_set_hash:
            changes["feature_set_hash"] = {
                "before": self.feature_set_hash,
                "after": other.feature_set_hash,
            }
        return changes

    def top_features(self, n: int = 10) -> list[str]:
        """Return top-n feature names by importance."""
        return sorted(
            self.feature_importance,
            key=lambda k: self.feature_importance[k],
            reverse=True,
        )[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "feature_count": self.feature_count,
            "feature_names": self.feature_names,
            "feature_definitions": self.feature_definitions,
            "feature_parameters": self.feature_parameters,
            "feature_importance": self.feature_importance,
            "feature_statistics": self.feature_statistics,
            "feature_distributions": self.feature_distributions,
            "feature_correlations": self.feature_correlations,
            "feature_drift_status": self.feature_drift_status,
            "feature_lifecycle_status": self.feature_lifecycle_status,
            "feature_set_hash": self.feature_set_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureSnapshot:
        return cls(
            snapshot_id=d.get("snapshot_id", str(uuid.uuid4())),
            captured_at=d.get("captured_at", datetime.now(UTC).isoformat()),
            feature_count=d.get("feature_count", 0),
            feature_names=list(d.get("feature_names", [])),
            feature_definitions=dict(d.get("feature_definitions", {})),
            feature_parameters=dict(d.get("feature_parameters", {})),
            feature_importance=dict(d.get("feature_importance", {})),
            feature_statistics=dict(d.get("feature_statistics", {})),
            feature_distributions=dict(d.get("feature_distributions", {})),
            feature_correlations=dict(d.get("feature_correlations", {})),
            feature_drift_status=dict(d.get("feature_drift_status", {})),
            feature_lifecycle_status=dict(d.get("feature_lifecycle_status", {})),
            feature_set_hash=d.get("feature_set_hash", ""),
        )


# ---------------------------------------------------------------------------
# ModelSnapshot
# ---------------------------------------------------------------------------


@dataclass
class ModelSnapshot:
    """Immutable record of a model's architecture and training configuration."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Architecture
    architecture: str = ""
    backbone: str = ""
    parameter_count: int = 0
    hidden_layers: int = 0
    embedding_sizes: list[int] = field(default_factory=list)
    attention_heads: int = 0
    fusion_strategy: str = ""
    prediction_heads: list[str] = field(default_factory=list)

    # Training configuration
    loss_functions: list[str] = field(default_factory=list)
    optimizer: str = ""
    learning_rate: float = 0.0
    scheduler: str = ""
    dropout: float = 0.0
    regularization: str = ""

    # Artifact hashes
    checkpoint_hash: str = ""
    onnx_hash: str = ""
    torchscript_hash: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def diff(self, other: ModelSnapshot) -> dict[str, Any]:
        """Return differences in model configuration."""
        changes: dict[str, Any] = {}
        for attr in (
            "architecture",
            "backbone",
            "parameter_count",
            "hidden_layers",
            "attention_heads",
            "fusion_strategy",
            "optimizer",
            "learning_rate",
            "scheduler",
            "dropout",
            "regularization",
            "checkpoint_hash",
        ):
            a, b = getattr(self, attr), getattr(other, attr)
            if a != b:
                changes[attr] = {"before": a, "after": b}
        if self.prediction_heads != other.prediction_heads:
            changes["prediction_heads"] = {
                "before": self.prediction_heads,
                "after": other.prediction_heads,
            }
        if self.loss_functions != other.loss_functions:
            changes["loss_functions"] = {
                "before": self.loss_functions,
                "after": other.loss_functions,
            }
        return changes

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "architecture": self.architecture,
            "backbone": self.backbone,
            "parameter_count": self.parameter_count,
            "hidden_layers": self.hidden_layers,
            "embedding_sizes": self.embedding_sizes,
            "attention_heads": self.attention_heads,
            "fusion_strategy": self.fusion_strategy,
            "prediction_heads": self.prediction_heads,
            "loss_functions": self.loss_functions,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler,
            "dropout": self.dropout,
            "regularization": self.regularization,
            "checkpoint_hash": self.checkpoint_hash,
            "onnx_hash": self.onnx_hash,
            "torchscript_hash": self.torchscript_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSnapshot:
        return cls(
            snapshot_id=d.get("snapshot_id", str(uuid.uuid4())),
            captured_at=d.get("captured_at", datetime.now(UTC).isoformat()),
            architecture=d.get("architecture", ""),
            backbone=d.get("backbone", ""),
            parameter_count=d.get("parameter_count", 0),
            hidden_layers=d.get("hidden_layers", 0),
            embedding_sizes=d.get("embedding_sizes", []),
            attention_heads=d.get("attention_heads", 0),
            fusion_strategy=d.get("fusion_strategy", ""),
            prediction_heads=d.get("prediction_heads", []),
            loss_functions=d.get("loss_functions", []),
            optimizer=d.get("optimizer", ""),
            learning_rate=d.get("learning_rate", 0.0),
            scheduler=d.get("scheduler", ""),
            dropout=d.get("dropout", 0.0),
            regularization=d.get("regularization", ""),
            checkpoint_hash=d.get("checkpoint_hash", ""),
            onnx_hash=d.get("onnx_hash", ""),
            torchscript_hash=d.get("torchscript_hash", ""),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# ValidationSnapshot
# ---------------------------------------------------------------------------


@dataclass
class ValidationSnapshot:
    """Immutable record of a validation run's results."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Walk-forward results
    walk_forward_accuracy: float = 0.0
    walk_forward_f1_macro: float = 0.0
    walk_forward_n_folds: int = 0
    walk_forward_results: list[dict[str, Any]] = field(default_factory=list)

    # Calibration
    calibration_ece: float = 0.0
    calibration_mce: float = 0.0
    calibration_brier: float = 0.0
    calibration_reliability: dict[str, Any] = field(default_factory=dict)

    # Trading metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0

    # Stress tests / Monte Carlo
    stress_test_results: dict[str, Any] = field(default_factory=dict)
    monte_carlo_var_95: float = 0.0
    monte_carlo_var_99: float = 0.0

    # Failure analysis
    failure_modes: list[str] = field(default_factory=list)
    regime_performance: dict[str, float] = field(default_factory=dict)

    # Promotion
    promotion_decision: str = ""
    promotion_rationale: str = ""

    def diff(self, other: ValidationSnapshot) -> dict[str, Any]:
        """Return differences in validation results."""
        changes: dict[str, Any] = {}
        for attr in (
            "walk_forward_accuracy",
            "walk_forward_f1_macro",
            "calibration_ece",
            "calibration_brier",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "monte_carlo_var_95",
            "monte_carlo_var_99",
        ):
            a, b = getattr(self, attr), getattr(other, attr)
            if abs(a - b) > 1e-6:
                changes[attr] = {
                    "before": round(a, 6),
                    "after": round(b, 6),
                    "delta": round(b - a, 6),
                }
        if self.promotion_decision != other.promotion_decision:
            changes["promotion_decision"] = {
                "before": self.promotion_decision,
                "after": other.promotion_decision,
            }
        return changes

    def summary(self) -> dict[str, Any]:
        return {
            "walk_forward_accuracy": self.walk_forward_accuracy,
            "walk_forward_f1_macro": self.walk_forward_f1_macro,
            "calibration_ece": self.calibration_ece,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "promotion_decision": self.promotion_decision,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "walk_forward_accuracy": self.walk_forward_accuracy,
            "walk_forward_f1_macro": self.walk_forward_f1_macro,
            "walk_forward_n_folds": self.walk_forward_n_folds,
            "walk_forward_results": self.walk_forward_results,
            "calibration_ece": self.calibration_ece,
            "calibration_mce": self.calibration_mce,
            "calibration_brier": self.calibration_brier,
            "calibration_reliability": self.calibration_reliability,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "stress_test_results": self.stress_test_results,
            "monte_carlo_var_95": self.monte_carlo_var_95,
            "monte_carlo_var_99": self.monte_carlo_var_99,
            "failure_modes": self.failure_modes,
            "regime_performance": self.regime_performance,
            "promotion_decision": self.promotion_decision,
            "promotion_rationale": self.promotion_rationale,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ValidationSnapshot:
        return cls(
            snapshot_id=d.get("snapshot_id", str(uuid.uuid4())),
            captured_at=d.get("captured_at", datetime.now(UTC).isoformat()),
            walk_forward_accuracy=d.get("walk_forward_accuracy", 0.0),
            walk_forward_f1_macro=d.get("walk_forward_f1_macro", 0.0),
            walk_forward_n_folds=d.get("walk_forward_n_folds", 0),
            walk_forward_results=d.get("walk_forward_results", []),
            calibration_ece=d.get("calibration_ece", 0.0),
            calibration_mce=d.get("calibration_mce", 0.0),
            calibration_brier=d.get("calibration_brier", 0.0),
            calibration_reliability=d.get("calibration_reliability", {}),
            sharpe_ratio=d.get("sharpe_ratio", 0.0),
            sortino_ratio=d.get("sortino_ratio", 0.0),
            max_drawdown=d.get("max_drawdown", 0.0),
            win_rate=d.get("win_rate", 0.0),
            profit_factor=d.get("profit_factor", 0.0),
            calmar_ratio=d.get("calmar_ratio", 0.0),
            stress_test_results=d.get("stress_test_results", {}),
            monte_carlo_var_95=d.get("monte_carlo_var_95", 0.0),
            monte_carlo_var_99=d.get("monte_carlo_var_99", 0.0),
            failure_modes=d.get("failure_modes", []),
            regime_performance=d.get("regime_performance", {}),
            promotion_decision=d.get("promotion_decision", ""),
            promotion_rationale=d.get("promotion_rationale", ""),
        )
