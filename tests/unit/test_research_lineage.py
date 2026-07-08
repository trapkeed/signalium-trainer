"""Unit tests for Phase 19.1 — Experiment Lineage and Reproducibility.

Covers:
  EnvironmentSnapshot  (reproducibility.py)
  DatasetSnapshot      (snapshots.py)
  FeatureSnapshot      (snapshots.py)
  ModelSnapshot        (snapshots.py)
  ValidationSnapshot   (snapshots.py)
  LineageNode          (lineage.py)
  LineageTree          (lineage.py)
  DatasetDiff / FeatureDiff / ModelDiff / ValidationDiff  (differencing.py)
  ExperimentDiffer     (differencing.py)
  KnowledgeGraph       (knowledge_graph.py)
  ResearchTimeline     (timeline.py)
  CostTracker          (cost_tracker.py)
  DashboardBuilder     (dashboard.py)
  ResearchExporter     (export.py)
  LineageDatabase      (lineage_database.py)
  LineageAPI           (lineage_api.py)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from trainer.research.cost_tracker import CostTracker
from trainer.research.dashboard import DashboardBuilder
from trainer.research.differencing import DatasetDiff, ExperimentDiffer, ModelDiff, ValidationDiff
from trainer.research.export import ResearchExporter
from trainer.research.knowledge_graph import EdgeType, KnowledgeGraph, NodeType
from trainer.research.lineage import ExperimentStatus, LineageNode, LineageTree
from trainer.research.lineage_api import LineageAPI
from trainer.research.lineage_database import LineageDatabase
from trainer.research.reproducibility import EnvironmentSnapshot
from trainer.research.snapshots import (
    DatasetSnapshot,
    FeatureSnapshot,
    ModelSnapshot,
    ValidationSnapshot,
)
from trainer.research.timeline import ResearchTimeline, TimelineEventType

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def env_snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        git_commit_hash="abc1234def5678",
        git_branch="feature/test",
        repo_version="v1.2.3",
        python_version="3.11.0 (main)",
        pytorch_version="2.3.0",
        cuda_version="12.1",
        random_seed=42,
        os_name="Linux",
        os_version="5.15.0",
        cpu_model="Intel Xeon",
        ram_gb=64.0,
        gpu_model="NVIDIA A100",
        docker_image="signalium:v1.2.3",
        env_vars={"CUDA_VISIBLE_DEVICES": "0,1"},
        training_duration_seconds=7200.0,
        training_cost_estimate_usd=4.00,
    )


@pytest.fixture()
def dataset_snapshot() -> DatasetSnapshot:
    return DatasetSnapshot(
        dataset_version="v2.3.1",
        dataset_hash="sha256:abc123",
        instruments=["ES", "NQ", "CL"],
        timeframes=["1h", "4h"],
        coverage_start="2018-01-01",
        coverage_end="2025-12-31",
        sample_count=1_250_000,
        feature_version="v1.5",
        feature_hash="sha256:feat123",
        label_version="v2.0",
        label_hash="sha256:label456",
        normalization_version="v1.0",
        scaling_version="v1.0",
        data_quality_score=0.97,
    )


@pytest.fixture()
def feature_snapshot() -> FeatureSnapshot:
    return FeatureSnapshot(
        feature_count=3,
        feature_names=["rsi_14", "macd_12_26", "atr_14"],
        feature_importance={"rsi_14": 0.35, "macd_12_26": 0.40, "atr_14": 0.25},
        feature_drift_status={"rsi_14": "stable", "macd_12_26": "stable"},
        feature_lifecycle_status={"rsi_14": "active", "macd_12_26": "active", "atr_14": "active"},
        feature_set_hash="sha256:feat_set_001",
    )


@pytest.fixture()
def model_snapshot() -> ModelSnapshot:
    return ModelSnapshot(
        architecture="transformer",
        backbone="transformer_4h",
        parameter_count=2_500_000,
        hidden_layers=6,
        embedding_sizes=[128, 256],
        attention_heads=8,
        fusion_strategy="concat",
        prediction_heads=["direction", "magnitude"],
        loss_functions=["focal_loss", "calibration_loss"],
        optimizer="adamw",
        learning_rate=0.0003,
        scheduler="cosine_annealing",
        dropout=0.1,
        regularization="l2",
        checkpoint_hash="sha256:ckpt_001",
        onnx_hash="sha256:onnx_001",
        torchscript_hash="sha256:ts_001",
    )


@pytest.fixture()
def validation_snapshot() -> ValidationSnapshot:
    return ValidationSnapshot(
        walk_forward_accuracy=0.692,
        walk_forward_f1_macro=0.681,
        walk_forward_n_folds=12,
        calibration_ece=0.028,
        calibration_mce=0.045,
        calibration_brier=0.189,
        sharpe_ratio=1.82,
        sortino_ratio=2.43,
        max_drawdown=0.067,
        win_rate=0.573,
        profit_factor=1.68,
        calmar_ratio=2.71,
        monte_carlo_var_95=0.023,
        monte_carlo_var_99=0.041,
        failure_modes=["low_volume_chop"],
        regime_performance={"trending": 0.74, "ranging": 0.61},
        promotion_decision="promote",
        promotion_rationale="Exceeds all thresholds.",
    )


@pytest.fixture()
def populated_tree(
    env_snapshot: EnvironmentSnapshot,
    dataset_snapshot: DatasetSnapshot,
    feature_snapshot: FeatureSnapshot,
    model_snapshot: ModelSnapshot,
    validation_snapshot: ValidationSnapshot,
) -> LineageTree:
    tree = LineageTree()
    root_id = tree.register(
        "baseline_v1",
        researcher="alice",
        purpose="Baseline experiment",
        branch="main",
        project="q4_2026",
        environment=env_snapshot,
    )
    tree.attach_dataset_snapshot(root_id, dataset_snapshot)
    tree.attach_feature_snapshot(root_id, feature_snapshot)
    tree.attach_model_snapshot(root_id, model_snapshot)
    tree.attach_validation_snapshot(root_id, validation_snapshot)
    tree.complete(root_id, ExperimentStatus.SUCCEEDED)

    child_id = tree.register(
        "transformer_v2",
        researcher="bob",
        purpose="Test v2 transformer",
        branch="feature/v2",
        project="q4_2026",
        parent_id=root_id,
    )
    return tree


@pytest.fixture()
def timeline() -> ResearchTimeline:
    return ResearchTimeline()


@pytest.fixture()
def cost_tracker() -> CostTracker:
    return CostTracker(gpu_rate_per_hour=2.00, cpu_rate_per_hour=0.05)


@pytest.fixture()
def kg() -> KnowledgeGraph:
    return KnowledgeGraph()


# ===========================================================================
# EnvironmentSnapshot
# ===========================================================================


class TestEnvironmentSnapshot:
    def test_capture_returns_snapshot(self) -> None:
        snap = EnvironmentSnapshot.capture(random_seed=99)
        assert snap.snapshot_id
        assert snap.python_version

    def test_reproducibility_score_full(self, env_snapshot: EnvironmentSnapshot) -> None:
        score = env_snapshot.reproducibility_score()
        assert 0.0 <= score <= 1.0
        assert score >= 0.75  # git + python + pytorch + seed + os

    def test_reproducibility_score_empty(self) -> None:
        snap = EnvironmentSnapshot()
        assert snap.reproducibility_score() == 0.0

    def test_round_trip(self, env_snapshot: EnvironmentSnapshot) -> None:
        d = env_snapshot.to_dict()
        restored = EnvironmentSnapshot.from_dict(d)
        assert restored.snapshot_id == env_snapshot.snapshot_id
        assert restored.git_commit_hash == env_snapshot.git_commit_hash
        assert restored.pytorch_version == env_snapshot.pytorch_version
        assert restored.random_seed == 42
        assert restored.ram_gb == pytest.approx(64.0)

    def test_with_training_metrics(self, env_snapshot: EnvironmentSnapshot) -> None:
        updated = env_snapshot.with_training_metrics(3600.0, cost_estimate_usd=2.00)
        assert updated.training_duration_seconds == pytest.approx(3600.0)
        assert updated.training_cost_estimate_usd == pytest.approx(2.00)
        # original unchanged
        assert env_snapshot.training_duration_seconds == pytest.approx(7200.0)

    def test_to_report_contains_score(self, env_snapshot: EnvironmentSnapshot) -> None:
        report = env_snapshot.to_report()
        assert "Repro Score" in report
        assert "Git Commit" in report


# ===========================================================================
# DatasetSnapshot
# ===========================================================================


class TestDatasetSnapshot:
    def test_round_trip(self, dataset_snapshot: DatasetSnapshot) -> None:
        d = dataset_snapshot.to_dict()
        restored = DatasetSnapshot.from_dict(d)
        assert restored.dataset_version == "v2.3.1"
        assert restored.sample_count == 1_250_000
        assert restored.instruments == ["ES", "NQ", "CL"]

    def test_diff_no_change(self, dataset_snapshot: DatasetSnapshot) -> None:
        other = DatasetSnapshot.from_dict(dataset_snapshot.to_dict())
        diff = dataset_snapshot.diff(other)
        assert diff == {}

    def test_diff_detects_instrument_change(self, dataset_snapshot: DatasetSnapshot) -> None:
        other = DatasetSnapshot.from_dict(dataset_snapshot.to_dict())
        other.instruments = ["ES", "NQ", "GC"]  # CL → GC
        diff = dataset_snapshot.diff(other)
        assert "instruments" in diff
        assert "GC" in diff["instruments"]["added"]
        assert "CL" in diff["instruments"]["removed"]

    def test_diff_detects_version_change(self, dataset_snapshot: DatasetSnapshot) -> None:
        other = DatasetSnapshot.from_dict(dataset_snapshot.to_dict())
        other.dataset_version = "v2.4.0"
        diff = dataset_snapshot.diff(other)
        assert "dataset_version" in diff


# ===========================================================================
# FeatureSnapshot
# ===========================================================================


class TestFeatureSnapshot:
    def test_round_trip(self, feature_snapshot: FeatureSnapshot) -> None:
        d = feature_snapshot.to_dict()
        restored = FeatureSnapshot.from_dict(d)
        assert restored.feature_count == 3
        assert "rsi_14" in restored.feature_names
        assert restored.feature_importance["macd_12_26"] == pytest.approx(0.40)

    def test_top_features(self, feature_snapshot: FeatureSnapshot) -> None:
        top = feature_snapshot.top_features(2)
        assert top[0] == "macd_12_26"  # highest importance 0.40
        assert top[1] == "rsi_14"

    def test_diff_adds_and_removes(self, feature_snapshot: FeatureSnapshot) -> None:
        other = FeatureSnapshot.from_dict(feature_snapshot.to_dict())
        other.feature_names = ["rsi_14", "macd_12_26", "new_feature"]
        diff = feature_snapshot.diff(other)
        assert "features_added" in diff
        assert "new_feature" in diff["features_added"]
        assert "features_removed" in diff
        assert "atr_14" in diff["features_removed"]

    def test_diff_detects_importance_shift(self, feature_snapshot: FeatureSnapshot) -> None:
        other = FeatureSnapshot.from_dict(feature_snapshot.to_dict())
        other.feature_importance["rsi_14"] = 0.15  # dropped from 0.35
        diff = feature_snapshot.diff(other)
        assert "importance_changes" in diff
        assert "rsi_14" in diff["importance_changes"]


# ===========================================================================
# ModelSnapshot
# ===========================================================================


class TestModelSnapshot:
    def test_round_trip(self, model_snapshot: ModelSnapshot) -> None:
        d = model_snapshot.to_dict()
        restored = ModelSnapshot.from_dict(d)
        assert restored.architecture == "transformer"
        assert restored.parameter_count == 2_500_000
        assert restored.learning_rate == pytest.approx(0.0003)

    def test_diff_no_change(self, model_snapshot: ModelSnapshot) -> None:
        other = ModelSnapshot.from_dict(model_snapshot.to_dict())
        diff = model_snapshot.diff(other)
        assert diff == {}

    def test_diff_detects_lr_change(self, model_snapshot: ModelSnapshot) -> None:
        other = ModelSnapshot.from_dict(model_snapshot.to_dict())
        other.learning_rate = 0.001
        diff = model_snapshot.diff(other)
        assert "learning_rate" in diff

    def test_diff_detects_backbone_change(self, model_snapshot: ModelSnapshot) -> None:
        other = ModelSnapshot.from_dict(model_snapshot.to_dict())
        other.backbone = "tcn_256"
        diff = model_snapshot.diff(other)
        assert "backbone" in diff


# ===========================================================================
# ValidationSnapshot
# ===========================================================================


class TestValidationSnapshot:
    def test_round_trip(self, validation_snapshot: ValidationSnapshot) -> None:
        d = validation_snapshot.to_dict()
        restored = ValidationSnapshot.from_dict(d)
        assert restored.walk_forward_accuracy == pytest.approx(0.692)
        assert restored.sharpe_ratio == pytest.approx(1.82)
        assert restored.promotion_decision == "promote"

    def test_summary(self, validation_snapshot: ValidationSnapshot) -> None:
        s = validation_snapshot.summary()
        assert "walk_forward_accuracy" in s
        assert s["sharpe_ratio"] == pytest.approx(1.82)

    def test_diff_detects_accuracy_improvement(
        self, validation_snapshot: ValidationSnapshot
    ) -> None:
        other = ValidationSnapshot.from_dict(validation_snapshot.to_dict())
        other.walk_forward_accuracy = 0.710
        vd = ValidationDiff.compare(validation_snapshot, other)
        assert vd.performance_improved
        assert "walk_forward_accuracy" in vd.changes


# ===========================================================================
# LineageNode
# ===========================================================================


class TestLineageNode:
    def test_defaults(self) -> None:
        node = LineageNode()
        assert node.node_id
        assert node.status == ExperimentStatus.PLANNED

    def test_total_cost(self) -> None:
        node = LineageNode(
            training_cost_usd=10.0,
            storage_cost_usd=1.5,
            validation_cost_usd=0.5,
        )
        assert node.total_cost_usd == pytest.approx(12.0)

    def test_reproducibility_score_no_env(self) -> None:
        node = LineageNode()
        assert node.reproducibility_score() == 0.0

    def test_reproducibility_score_with_env(self, env_snapshot: EnvironmentSnapshot) -> None:
        node = LineageNode(environment=env_snapshot)
        assert node.reproducibility_score() >= 0.75

    def test_round_trip(
        self,
        env_snapshot: EnvironmentSnapshot,
        dataset_snapshot: DatasetSnapshot,
    ) -> None:
        node = LineageNode(
            name="test_exp",
            researcher="alice",
            branch="main",
            environment=env_snapshot,
            dataset_snapshot=dataset_snapshot,
        )
        d = node.to_dict()
        restored = LineageNode.from_dict(d)
        assert restored.name == "test_exp"
        assert restored.researcher == "alice"
        assert restored.environment is not None
        assert restored.environment.git_commit_hash == env_snapshot.git_commit_hash
        assert restored.dataset_snapshot is not None
        assert restored.dataset_snapshot.dataset_version == "v2.3.1"


# ===========================================================================
# LineageTree
# ===========================================================================


class TestLineageTree:
    def test_register_creates_node(self) -> None:
        tree = LineageTree()
        nid = tree.register("exp_01", researcher="alice")
        assert nid
        assert tree.get(nid) is not None

    def test_parent_child_relationship(self) -> None:
        tree = LineageTree()
        parent_id = tree.register("parent")
        child_id = tree.register("child", parent_id=parent_id)
        assert tree.get(child_id).parent_id == parent_id
        assert child_id in tree.get(parent_id).child_ids

    def test_ancestors(self, populated_tree: LineageTree) -> None:
        nodes = populated_tree.all_nodes()
        child = next(n for n in nodes if n.name == "transformer_v2")
        ancestors = populated_tree.ancestors(child.node_id)
        assert len(ancestors) == 1
        assert ancestors[0].name == "baseline_v1"

    def test_descendants(self, populated_tree: LineageTree) -> None:
        root = populated_tree.roots()[0]
        descs = populated_tree.descendants(root.node_id)
        assert len(descs) == 1

    def test_roots(self, populated_tree: LineageTree) -> None:
        roots = populated_tree.roots()
        assert len(roots) == 1
        assert roots[0].name == "baseline_v1"

    def test_by_branch(self, populated_tree: LineageTree) -> None:
        feature_nodes = populated_tree.by_branch("feature/v2")
        assert len(feature_nodes) == 1

    def test_by_project(self, populated_tree: LineageTree) -> None:
        q4_nodes = populated_tree.by_project("q4_2026")
        assert len(q4_nodes) == 2

    def test_by_status(self, populated_tree: LineageTree) -> None:
        succeeded = populated_tree.by_status(ExperimentStatus.SUCCEEDED)
        assert len(succeeded) == 1

    def test_lineage_report(self, populated_tree: LineageTree) -> None:
        root = populated_tree.roots()[0]
        report = populated_tree.lineage_report(root.node_id)
        assert report["ancestor_count"] == 0
        assert report["descendant_count"] == 1
        assert 0.0 < report["reproducibility_score"] <= 1.0
        assert report["reproducibility_grade"] in ("A", "B", "C", "D", "F")

    def test_tree_summary(self, populated_tree: LineageTree) -> None:
        summary = populated_tree.tree_summary()
        assert summary["total_experiments"] == 2
        assert summary["root_experiments"] == 1
        assert "q4_2026" in summary["projects"]

    def test_complete_invalid_status_raises(self) -> None:
        tree = LineageTree()
        nid = tree.register("exp")
        with pytest.raises(ValueError):
            tree.complete(nid, status="invalid_status")

    def test_get_missing_raises_in_internal(self) -> None:
        tree = LineageTree()
        with pytest.raises(KeyError):
            tree._get("nonexistent")

    def test_attach_cost(self, populated_tree: LineageTree) -> None:
        root = populated_tree.roots()[0]
        populated_tree.record_cost(
            root.node_id,
            gpu_hours=4.5,
            training_cost_usd=9.0,
            storage_cost_usd=0.5,
        )
        node = populated_tree.get(root.node_id)
        assert node.gpu_hours == pytest.approx(4.5)
        assert node.total_cost_usd == pytest.approx(9.5)


# ===========================================================================
# Differencing
# ===========================================================================


class TestDifferencing:
    def test_dataset_diff_identical(self, dataset_snapshot: DatasetSnapshot) -> None:
        other = DatasetSnapshot.from_dict(dataset_snapshot.to_dict())
        diff = DatasetDiff.compare(dataset_snapshot, other)
        assert diff.is_identical

    def test_dataset_diff_changed(self, dataset_snapshot: DatasetSnapshot) -> None:
        other = DatasetSnapshot.from_dict(dataset_snapshot.to_dict())
        other.sample_count = 1_500_000
        diff = DatasetDiff.compare(dataset_snapshot, other)
        assert not diff.is_identical
        assert "sample_count" in diff.changes

    def test_model_diff_arch_changed(self, model_snapshot: ModelSnapshot) -> None:
        other = ModelSnapshot.from_dict(model_snapshot.to_dict())
        other.architecture = "tcn"
        other.backbone = "tcn_256"
        diff = ModelDiff.compare(model_snapshot, other)
        assert diff.architecture_changed()

    def test_model_diff_hyperparam_changed(self, model_snapshot: ModelSnapshot) -> None:
        other = ModelSnapshot.from_dict(model_snapshot.to_dict())
        other.learning_rate = 0.001
        diff = ModelDiff.compare(model_snapshot, other)
        assert diff.hyperparameter_changed()

    def test_validation_diff_improved(self, validation_snapshot: ValidationSnapshot) -> None:
        better = ValidationSnapshot.from_dict(validation_snapshot.to_dict())
        better.walk_forward_accuracy = 0.720
        diff = ValidationDiff.compare(validation_snapshot, better)
        assert diff.performance_improved
        assert not diff.is_identical

    def test_experiment_differ_full(self, populated_tree: LineageTree) -> None:
        nodes = populated_tree.all_nodes()
        root = next(n for n in nodes if n.name == "baseline_v1")
        child = next(n for n in nodes if n.name == "transformer_v2")
        differ = ExperimentDiffer(populated_tree)
        diff = differ.compare_experiments(root.node_id, child.node_id)
        assert diff.node_id_a == root.node_id
        assert diff.node_id_b == child.node_id
        # Child has no snapshots so diffs should be None
        assert diff.dataset_diff is None

    def test_differ_missing_node_raises(self, populated_tree: LineageTree) -> None:
        differ = ExperimentDiffer(populated_tree)
        with pytest.raises(KeyError):
            differ.compare_experiments("bad_id", "also_bad")


# ===========================================================================
# KnowledgeGraph
# ===========================================================================


class TestKnowledgeGraph:
    def test_add_node_and_edge(self) -> None:
        kg = KnowledgeGraph()
        exp_id = kg.add_node(NodeType.EXPERIMENT, "exp_01")
        ds_id = kg.add_node(NodeType.DATASET, "ds_v1")
        eid = kg.add_edge(exp_id, ds_id, EdgeType.USES_DATASET)
        assert kg.get_node(exp_id) is not None
        assert kg.get_edge(eid) is not None

    def test_neighbours_out(self) -> None:
        kg = KnowledgeGraph()
        exp_id = kg.add_node(NodeType.EXPERIMENT, "exp_01")
        ds_id = kg.add_node(NodeType.DATASET, "ds_v1")
        kg.add_edge(exp_id, ds_id, EdgeType.USES_DATASET)
        neighbours = kg.neighbours(exp_id, direction="out")
        assert len(neighbours) == 1
        assert neighbours[0].node_id == ds_id

    def test_shortest_path(self) -> None:
        kg = KnowledgeGraph()
        a = kg.add_node(NodeType.EXPERIMENT, "a")
        b = kg.add_node(NodeType.EXPERIMENT, "b")
        c = kg.add_node(NodeType.EXPERIMENT, "c")
        kg.add_edge(a, b, EdgeType.PARENT_OF)
        kg.add_edge(b, c, EdgeType.PARENT_OF)
        path = kg.shortest_path(a, c)
        assert path == [a, b, c]

    def test_shortest_path_no_connection(self) -> None:
        kg = KnowledgeGraph()
        a = kg.add_node(NodeType.EXPERIMENT, "a")
        b = kg.add_node(NodeType.EXPERIMENT, "b")
        assert kg.shortest_path(a, b) == []

    def test_edge_missing_node_raises(self) -> None:
        kg = KnowledgeGraph()
        kg.add_node(NodeType.EXPERIMENT, "a", node_id="a")
        with pytest.raises(KeyError):
            kg.add_edge("a", "nonexistent", EdgeType.PARENT_OF)

    def test_nodes_by_type(self) -> None:
        kg = KnowledgeGraph()
        kg.add_node(NodeType.EXPERIMENT, "e1")
        kg.add_node(NodeType.EXPERIMENT, "e2")
        kg.add_node(NodeType.DATASET, "d1")
        assert len(kg.nodes_by_type(NodeType.EXPERIMENT)) == 2
        assert len(kg.nodes_by_type(NodeType.DATASET)) == 1

    def test_round_trip(self) -> None:
        kg = KnowledgeGraph()
        a = kg.add_node(NodeType.EXPERIMENT, "exp_a", {"researcher": "alice"})
        b = kg.add_node(NodeType.MODEL, "model_b")
        kg.add_edge(a, b, EdgeType.HAS_MODEL)
        d = kg.to_dict()
        restored = KnowledgeGraph.from_dict(d)
        assert len(restored.nodes_by_type(NodeType.EXPERIMENT)) == 1
        assert len(restored.nodes_by_type(NodeType.MODEL)) == 1

    def test_graph_report(self, kg: KnowledgeGraph) -> None:
        kg.add_node(NodeType.CHAMPION_MODEL, "champ1")
        report = kg.graph_report()
        assert report["champion_models"] == 1
        assert report["node_count"] >= 1


# ===========================================================================
# ResearchTimeline
# ===========================================================================


class TestResearchTimeline:
    def test_record_event(self) -> None:
        tl = ResearchTimeline()
        eid = tl.record(TimelineEventType.EXPERIMENT_STARTED, entity_id="e1", description="started")
        assert eid
        assert tl.total_events() == 1

    def test_by_type(self) -> None:
        tl = ResearchTimeline()
        tl.record_experiment_started("e1", "exp_01")
        tl.record_experiment_started("e2", "exp_02")
        tl.record_dataset_updated("ds_v2", 1_000_000)
        started = tl.by_type(TimelineEventType.EXPERIMENT_STARTED)
        assert len(started) == 2

    def test_by_entity(self) -> None:
        tl = ResearchTimeline()
        tl.record_experiment_started("e1", "exp_01")
        tl.record_experiment_completed("e1", "exp_01", "succeeded", accuracy=0.69)
        events = tl.by_entity("e1")
        assert len(events) == 2

    def test_experiment_history(self) -> None:
        tl = ResearchTimeline()
        tl.record_experiment_started("e1", "exp_01", "alice")
        tl.record_experiment_completed("e1", "exp_01", "succeeded", 0.72)
        history = tl.experiment_history()
        assert len(history) == 1
        assert history[0]["status"] == "succeeded"
        assert history[0]["accuracy"] == pytest.approx(0.72)

    def test_recent(self) -> None:
        tl = ResearchTimeline()
        for i in range(25):
            tl.record(TimelineEventType.COST_RECORDED, entity_id=f"e{i}")
        assert len(tl.recent(10)) == 10

    def test_champion_history(self) -> None:
        tl = ResearchTimeline()
        tl.record_champion_changed("old_champ", "new_champ", "better sharpe")
        hist = tl.champion_history()
        assert len(hist) == 1

    def test_to_markdown(self) -> None:
        tl = ResearchTimeline()
        tl.record_experiment_started("e1", "exp_01")
        md = tl.to_markdown()
        assert "# Research Timeline" in md
        assert "experiment_started" in md


# ===========================================================================
# CostTracker
# ===========================================================================


class TestCostTracker:
    def test_record_computes_cost(self) -> None:
        ct = CostTracker(gpu_rate_per_hour=2.00, cpu_rate_per_hour=0.05)
        cost = ct.record("e1", gpu_hours=4.0, cpu_hours=8.0)
        assert cost.training_cost_usd == pytest.approx(8.00)
        assert cost.cpu_cost_usd == pytest.approx(0.40)
        assert cost.total_cost_usd == pytest.approx(8.40)

    def test_estimate_from_duration(self) -> None:
        ct = CostTracker()
        cost = ct.estimate_from_duration("e1", training_seconds=7200.0, gpu_count=1)
        assert cost.gpu_hours == pytest.approx(2.0)
        assert cost.training_cost_usd == pytest.approx(4.00)

    def test_total_cost(self) -> None:
        ct = CostTracker(gpu_rate_per_hour=1.00, cpu_rate_per_hour=0.10)
        ct.record("e1", gpu_hours=2.0)
        ct.record("e2", gpu_hours=3.0)
        assert ct.total_cost_usd() == pytest.approx(5.00)

    def test_cost_by_project(self) -> None:
        ct = CostTracker(gpu_rate_per_hour=1.00, cpu_rate_per_hour=0.0)
        ct.record("e1", project="proj_a", gpu_hours=2.0)
        ct.record("e2", project="proj_b", gpu_hours=3.0)
        by_proj = ct.cost_by_project()
        assert by_proj["proj_a"] == pytest.approx(2.00)
        assert by_proj["proj_b"] == pytest.approx(3.00)

    def test_most_expensive(self) -> None:
        ct = CostTracker(gpu_rate_per_hour=1.00, cpu_rate_per_hour=0.0)
        for i in range(10):
            ct.record(f"e{i}", gpu_hours=float(i))
        top = ct.most_expensive(3)
        assert top[0].gpu_hours == pytest.approx(9.0)

    def test_cost_by_category(self) -> None:
        ct = CostTracker(gpu_rate_per_hour=2.00, storage_rate_per_gb_month=0.023)
        ct.record("e1", gpu_hours=1.0, storage_gb_months=10.0)
        cats = ct.cost_by_category()
        assert cats["training"] == pytest.approx(2.00)
        assert cats["storage"] == pytest.approx(0.23)

    def test_get_by_entity(self) -> None:
        ct = CostTracker()
        ct.record("exp_xyz", gpu_hours=1.0)
        found = ct.get("exp_xyz")
        assert found is not None
        assert found.entity_id == "exp_xyz"

    def test_get_missing_returns_none(self) -> None:
        ct = CostTracker()
        assert ct.get("missing") is None


# ===========================================================================
# DashboardBuilder
# ===========================================================================


class TestDashboardBuilder:
    @pytest.fixture()
    def builder(
        self,
        populated_tree: LineageTree,
        timeline: ResearchTimeline,
        cost_tracker: CostTracker,
        kg: KnowledgeGraph,
    ) -> DashboardBuilder:
        timeline.record_experiment_started("e1", "exp_01")
        cost_tracker.record("e1", gpu_hours=2.0, project="q4_2026")
        return DashboardBuilder(populated_tree, timeline, cost_tracker, kg)

    def test_build_research_timeline(self, builder: DashboardBuilder) -> None:
        payload = builder.build("research_timeline")
        assert payload["dashboard_type"] == "research_timeline"
        assert "total_events" in payload["data"]

    def test_build_knowledge_graph(self, builder: DashboardBuilder) -> None:
        payload = builder.build("knowledge_graph")
        assert "nodes" in payload["data"]
        assert "edges" in payload["data"]

    def test_build_experiment_tree(self, builder: DashboardBuilder) -> None:
        payload = builder.build("experiment_tree")
        assert "tree" in payload["data"]
        assert len(payload["data"]["tree"]) >= 1

    def test_build_training_cost(self, builder: DashboardBuilder) -> None:
        payload = builder.build("training_cost")
        assert "total_cost_usd" in payload["data"]
        assert "cost_by_category" in payload["data"]

    def test_build_invalid_raises(self, builder: DashboardBuilder) -> None:
        with pytest.raises(ValueError):
            builder.build("nonexistent_dashboard")

    def test_build_all_returns_all_types(self, builder: DashboardBuilder) -> None:
        all_db = builder.build_all()
        assert "dashboards" in all_db
        for dt in DashboardBuilder.DASHBOARD_TYPES:
            assert dt in all_db["dashboards"]


# ===========================================================================
# ResearchExporter
# ===========================================================================


class TestResearchExporter:
    @pytest.fixture()
    def exporter(
        self,
        populated_tree: LineageTree,
        timeline: ResearchTimeline,
        cost_tracker: CostTracker,
        kg: KnowledgeGraph,
    ) -> ResearchExporter:
        return ResearchExporter(populated_tree, timeline, cost_tracker, kg)

    def test_experiment_card(self, exporter: ResearchExporter, populated_tree: LineageTree) -> None:
        root = populated_tree.roots()[0]
        card = exporter.experiment_card(root.node_id)
        assert card["card_type"] == "experiment_card"
        assert card["name"] == "baseline_v1"
        assert card["researcher"] == "alice"

    def test_experiment_card_markdown(
        self, exporter: ResearchExporter, populated_tree: LineageTree
    ) -> None:
        root = populated_tree.roots()[0]
        md = exporter.experiment_card_markdown(root.node_id)
        assert "# Experiment Card" in md
        assert "baseline_v1" in md

    def test_lineage_report(self, exporter: ResearchExporter, populated_tree: LineageTree) -> None:
        root = populated_tree.roots()[0]
        report = exporter.lineage_report(root.node_id)
        assert report["report_type"] == "lineage_report"
        assert len(report["descendants"]) == 1

    def test_research_report(self, exporter: ResearchExporter) -> None:
        report = exporter.research_report()
        assert report["report_type"] == "research_report"
        assert "lineage_tree" in report
        assert "cost" in report

    def test_experiment_tree_export(self, exporter: ResearchExporter) -> None:
        tree_exp = exporter.experiment_tree()
        assert tree_exp["export_type"] == "experiment_tree"
        assert "nodes" in tree_exp

    def test_knowledge_graph_export(self, exporter: ResearchExporter) -> None:
        kg_exp = exporter.knowledge_graph_export()
        assert kg_exp["export_type"] == "knowledge_graph"

    def test_missing_node_returns_error(self, exporter: ResearchExporter) -> None:
        card = exporter.experiment_card("nonexistent_id")
        assert "error" in card


# ===========================================================================
# LineageDatabase
# ===========================================================================


class TestLineageDatabase:
    def test_save_and_load_node(self) -> None:
        db = LineageDatabase(":memory:")
        node = LineageNode(name="test_node", researcher="alice", branch="main")
        db.save_node(node.to_dict())
        loaded = db.load_all_nodes()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "test_node"
        assert loaded[0]["researcher"] == "alice"

    def test_get_node(self) -> None:
        db = LineageDatabase(":memory:")
        node = LineageNode(name="exp_01")
        db.save_node(node.to_dict())
        found = db.get_node(node.node_id)
        assert found is not None
        assert found["name"] == "exp_01"

    def test_save_and_load_environment(self, env_snapshot: EnvironmentSnapshot) -> None:
        db = LineageDatabase(":memory:")
        node = LineageNode(name="exp_01")
        db.save_node(node.to_dict())
        db.save_environment(node.node_id, env_snapshot.to_dict())
        loaded = db.load_environment(node.node_id)
        assert loaded is not None
        assert loaded["git_commit_hash"] == env_snapshot.git_commit_hash

    def test_save_and_load_cost(self) -> None:
        db = LineageDatabase(":memory:")
        ct = CostTracker(gpu_rate_per_hour=2.00, cpu_rate_per_hour=0.05)
        cost = ct.record("exp_01", gpu_hours=3.0)
        db.save_cost(cost.to_dict())
        loaded = db.load_all_costs()
        assert len(loaded) == 1
        assert loaded[0]["entity_id"] == "exp_01"

    def test_save_and_load_timeline_events(self) -> None:
        db = LineageDatabase(":memory:")
        tl = ResearchTimeline()
        tl.record_experiment_started("e1", "exp_01")
        for event in tl.all_events():
            db.save_timeline_event(event.to_dict())
        loaded = db.load_all_timeline_events()
        assert len(loaded) == 1
        assert loaded[0]["event_type"] == TimelineEventType.EXPERIMENT_STARTED

    def test_save_and_load_knowledge_graph(self) -> None:
        db = LineageDatabase(":memory:")
        kg = KnowledgeGraph()
        kg.add_node(NodeType.EXPERIMENT, "exp_01")
        db.save_knowledge_graph(kg.to_dict())
        loaded = db.load_knowledge_graph()
        assert loaded is not None
        assert loaded["node_count"] == 1

    def test_stats(self) -> None:
        db = LineageDatabase(":memory:")
        node = LineageNode(name="exp_01")
        db.save_node(node.to_dict())
        stats = db.stats()
        assert stats["lineage_nodes"] == 1

    def test_file_based_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "lineage.db")
            db = LineageDatabase(path)
            node = LineageNode(name="file_test")
            db.save_node(node.to_dict())
            db.close()
            db2 = LineageDatabase(path)
            loaded = db2.load_all_nodes()
            assert len(loaded) == 1
            db2.close()


# ===========================================================================
# LineageAPI
# ===========================================================================


class TestLineageAPI:
    @pytest.fixture()
    def api(self) -> LineageAPI:
        tree = LineageTree()
        tl = ResearchTimeline()
        ct = CostTracker()
        kg = KnowledgeGraph()
        return LineageAPI(tree, tl, ct, kg)

    def test_register_experiment(self, api: LineageAPI) -> None:
        resp = api.register_experiment(
            "exp_01",
            researcher="alice",
            purpose="test",
            capture_environment=False,
        )
        assert resp["status"] == "ok"
        assert "node_id" in resp["data"]

    def test_start_and_complete(self, api: LineageAPI) -> None:
        resp = api.register_experiment("exp_01", capture_environment=False)
        node_id = resp["data"]["node_id"]
        start_resp = api.start_experiment(node_id)
        assert start_resp["status"] == "ok"
        complete_resp = api.complete_experiment(node_id, status="succeeded", accuracy=0.71)
        assert complete_resp["status"] == "ok"

    def test_get_experiment(self, api: LineageAPI) -> None:
        resp = api.register_experiment("exp_02", capture_environment=False)
        node_id = resp["data"]["node_id"]
        get_resp = api.get_experiment(node_id)
        assert get_resp["status"] == "ok"
        assert get_resp["data"]["name"] == "exp_02"

    def test_get_tree_summary(self, api: LineageAPI) -> None:
        api.register_experiment("e1", capture_environment=False)
        api.register_experiment("e2", capture_environment=False)
        resp = api.get_tree_summary()
        assert resp["status"] == "ok"
        assert resp["data"]["total_experiments"] == 2

    def test_attach_dataset_snapshot(
        self, api: LineageAPI, dataset_snapshot: DatasetSnapshot
    ) -> None:
        resp = api.register_experiment("e1", capture_environment=False)
        node_id = resp["data"]["node_id"]
        attach_resp = api.attach_dataset_snapshot(node_id, dataset_snapshot.to_dict())
        assert attach_resp["status"] == "ok"

    def test_record_cost(self, api: LineageAPI) -> None:
        resp = api.register_experiment("e1", capture_environment=False)
        node_id = resp["data"]["node_id"]
        cost_resp = api.record_cost(node_id, gpu_hours=2.0, project="proj_a")
        assert cost_resp["status"] == "ok"
        assert cost_resp["data"]["gpu_hours"] == pytest.approx(2.0)

    def test_compare_experiments_no_snapshots(self, api: LineageAPI) -> None:
        r1 = api.register_experiment("e1", capture_environment=False)
        r2 = api.register_experiment("e2", capture_environment=False)
        resp = api.compare_experiments(r1["data"]["node_id"], r2["data"]["node_id"])
        assert resp["status"] == "ok"
        assert resp["data"]["total_changes"] == 0  # no snapshots, nothing to diff

    def test_get_dashboard(self, api: LineageAPI) -> None:
        resp = api.get_dashboard("training_cost")
        assert resp["status"] == "ok"
        assert resp["data"]["dashboard_type"] == "training_cost"

    def test_get_all_dashboards(self, api: LineageAPI) -> None:
        resp = api.get_all_dashboards()
        assert resp["status"] == "ok"
        assert "dashboards" in resp["data"]

    def test_export_experiment_card(self, api: LineageAPI) -> None:
        r = api.register_experiment("e1", capture_environment=False)
        node_id = r["data"]["node_id"]
        resp = api.export_experiment_card(node_id)
        assert resp["status"] == "ok"

    def test_export_experiment_card_markdown(self, api: LineageAPI) -> None:
        r = api.register_experiment("e1", capture_environment=False)
        node_id = r["data"]["node_id"]
        resp = api.export_experiment_card(node_id, as_markdown=True)
        assert resp["status"] == "ok"
        assert "markdown" in resp["data"]

    def test_get_cost_report(self, api: LineageAPI) -> None:
        resp = api.get_cost_report()
        assert resp["status"] == "ok"
        assert "total_cost_usd" in resp["data"]

    def test_get_stats(self, api: LineageAPI) -> None:
        api.register_experiment("e1", capture_environment=False)
        resp = api.get_stats()
        assert resp["status"] == "ok"
        assert resp["data"]["lineage_tree"]["total_experiments"] == 1

    def test_error_response_for_missing_node(self, api: LineageAPI) -> None:
        resp = api.get_experiment("totally_missing_id")
        assert resp["status"] == "error"

    def test_add_graph_node_and_edge(self, api: LineageAPI) -> None:
        n1 = api.add_graph_node(NodeType.EXPERIMENT, "exp_a")
        n2 = api.add_graph_node(NodeType.MODEL, "model_b")
        assert n1["status"] == "ok"
        nid1 = n1["data"]["node_id"]
        nid2 = n2["data"]["node_id"]
        edge_resp = api.add_graph_edge(nid1, nid2, EdgeType.HAS_MODEL)
        assert edge_resp["status"] == "ok"

    def test_get_timeline_report(self, api: LineageAPI) -> None:
        api.register_experiment("e1", capture_environment=False)
        resp = api.get_timeline_report()
        assert resp["status"] == "ok"
        assert resp["data"]["total_events"] >= 1
