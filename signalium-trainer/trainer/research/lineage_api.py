"""Phase 19.1 — Lineage API.

Structured endpoint layer for the Experiment Lineage and
Reproducibility subsystems.

All responses use the standard envelope::

    {
      "status": "ok" | "error",
      "code": "...",
      "message": "...",
      "data": { ... },
      "timestamp": "2026-07-05T12:00:00Z"
    }

Endpoint groups:

  Lineage:         register_experiment, start_experiment, complete_experiment,
                   get_experiment, get_lineage_report, get_tree_summary,
                   get_ancestors, get_descendants, get_by_branch,
                   get_by_project, get_by_researcher, get_by_status

  Snapshots:       attach_environment, attach_dataset_snapshot,
                   attach_feature_snapshot, attach_model_snapshot,
                   attach_validation_snapshot, record_cost

  Differencing:    compare_experiments, compare_models,
                   compare_datasets, compare_features

  KnowledgeGraph:  get_graph_report, add_graph_node, add_graph_edge,
                   get_node_neighbours, get_model_lineage,
                   get_shortest_path, get_graph_export

  Timeline:        get_timeline_report, get_experiment_history,
                   get_champion_history, get_dataset_history,
                   get_recent_events, record_timeline_event

  Cost:            get_cost_report, get_experiment_cost,
                   record_cost_estimate

  Dashboard:       get_dashboard, get_all_dashboards

  Export:          export_experiment_card, export_lineage_report,
                   export_research_report, export_experiment_tree,
                   export_knowledge_graph
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.research.cost_tracker import CostTracker
from trainer.research.dashboard import DashboardBuilder
from trainer.research.differencing import ExperimentDiffer
from trainer.research.export import ResearchExporter
from trainer.research.knowledge_graph import KnowledgeGraph
from trainer.research.lineage import LineageTree
from trainer.research.reproducibility import EnvironmentSnapshot
from trainer.research.snapshots import (
    DatasetSnapshot,
    FeatureSnapshot,
    ModelSnapshot,
    ValidationSnapshot,
)
from trainer.research.timeline import ResearchTimeline, TimelineEventType


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


def _ok(code: str, data: Any, message: str = "ok") -> Dict[str, Any]:
    return {
        "status": "ok",
        "code": code,
        "message": message,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _err(code: str, message: str, detail: str = "") -> Dict[str, Any]:
    return {
        "status": "error",
        "code": code,
        "message": message,
        "detail": detail,
        "data": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _wrap(code: str, fn: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    try:
        return _ok(code, fn(*args, **kwargs))
    except KeyError as exc:
        return _err(f"{code}.not_found", str(exc))
    except ValueError as exc:
        return _err(f"{code}.invalid", str(exc))
    except Exception:
        detail = traceback.format_exc()
        logger.error(f"[LineageAPI] {code} error:\n{detail}")
        return _err(f"{code}.internal_error", "internal error", detail)


# ---------------------------------------------------------------------------
# LineageAPI
# ---------------------------------------------------------------------------


class LineageAPI:
    """Structured API for all Phase 19.1 lineage, reproducibility, and
    knowledge graph functionality."""

    def __init__(
        self,
        tree: LineageTree,
        timeline: ResearchTimeline,
        cost_tracker: CostTracker,
        knowledge_graph: KnowledgeGraph,
    ) -> None:
        self._tree = tree
        self._tl = timeline
        self._costs = cost_tracker
        self._kg = knowledge_graph
        self._differ = ExperimentDiffer(tree)
        self._dashboard = DashboardBuilder(tree, timeline, cost_tracker, knowledge_graph)
        self._exporter = ResearchExporter(tree, timeline, cost_tracker, knowledge_graph, self._differ)

    # ------------------------------------------------------------------
    # Lineage: experiment CRUD
    # ------------------------------------------------------------------

    def register_experiment(
        self,
        name: str,
        *,
        researcher: str = "system",
        purpose: str = "",
        branch: str = "main",
        project: str = "",
        parent_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        capture_environment: bool = True,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            env = EnvironmentSnapshot.capture(random_seed) if capture_environment else None
            node_id = self._tree.register(
                name,
                researcher=researcher,
                purpose=purpose,
                branch=branch,
                project=project,
                parent_id=parent_id,
                environment=env,
                tags=tags,
                notes=notes,
            )
            self._tl.record_experiment_started(node_id, name, researcher)
            # Register in knowledge graph
            self._kg.register_experiment_node(
                node_id, name,
                {"researcher": researcher, "branch": branch, "project": project},
            )
            if parent_id and self._kg.get_node(parent_id):
                try:
                    from trainer.research.knowledge_graph import EdgeType
                    self._kg.add_edge(parent_id, node_id, EdgeType.PARENT_OF)
                except Exception:
                    pass
            return {"node_id": node_id, "name": name}
        return _wrap("lineage.register_experiment", _do)

    def start_experiment(self, node_id: str) -> Dict[str, Any]:
        return _wrap("lineage.start_experiment", self._tree.start, node_id)

    def complete_experiment(
        self,
        node_id: str,
        status: str = "succeeded",
        accuracy: Optional[float] = None,
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            self._tree.complete(node_id, status)
            node = self._tree.get(node_id)
            name = node.name if node else node_id
            self._tl.record_experiment_completed(node_id, name, status, accuracy)
            return {"node_id": node_id, "status": status}
        return _wrap("lineage.complete_experiment", _do)

    def get_experiment(self, node_id: str) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            node = self._tree.get(node_id)
            if node is None:
                raise KeyError(node_id)
            return node.to_dict()
        return _wrap("lineage.get_experiment", _do)

    def get_lineage_report(self, node_id: str) -> Dict[str, Any]:
        return _wrap("lineage.get_lineage_report", self._tree.lineage_report, node_id)

    def get_tree_summary(self) -> Dict[str, Any]:
        return _wrap("lineage.get_tree_summary", self._tree.tree_summary)

    def get_ancestors(self, node_id: str) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [n.to_dict() for n in self._tree.ancestors(node_id)]
        return _wrap("lineage.get_ancestors", _do)

    def get_descendants(self, node_id: str) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [n.to_dict() for n in self._tree.descendants(node_id)]
        return _wrap("lineage.get_descendants", _do)

    def get_by_branch(self, branch: str) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [n.to_dict() for n in self._tree.by_branch(branch)]
        return _wrap("lineage.get_by_branch", _do)

    def get_by_project(self, project: str) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [n.to_dict() for n in self._tree.by_project(project)]
        return _wrap("lineage.get_by_project", _do)

    def get_by_status(self, status: str) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [n.to_dict() for n in self._tree.by_status(status)]
        return _wrap("lineage.get_by_status", _do)

    def get_by_researcher(self, researcher: str) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [n.to_dict() for n in self._tree.by_researcher(researcher)]
        return _wrap("lineage.get_by_researcher", _do)

    # ------------------------------------------------------------------
    # Snapshot attachment
    # ------------------------------------------------------------------

    def attach_environment(
        self, node_id: str, env_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            env = EnvironmentSnapshot.from_dict(env_dict)
            self._tree.attach_environment(node_id, env)
            return {"node_id": node_id, "snapshot_id": env.snapshot_id}
        return _wrap("lineage.attach_environment", _do)

    def attach_dataset_snapshot(
        self, node_id: str, snapshot_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            ds = DatasetSnapshot.from_dict(snapshot_dict)
            self._tree.attach_dataset_snapshot(node_id, ds)
            ds_node_id = self._kg.register_dataset_node(
                ds.dataset_version, ds.dataset_hash,
                {"sample_count": ds.sample_count, "quality": ds.data_quality_score},
            )
            if self._kg.get_node(node_id):
                from trainer.research.knowledge_graph import EdgeType
                try:
                    self._kg.add_edge(node_id, ds_node_id, EdgeType.USES_DATASET)
                except Exception:
                    pass
            return {"node_id": node_id, "snapshot_id": ds.snapshot_id}
        return _wrap("lineage.attach_dataset_snapshot", _do)

    def attach_feature_snapshot(
        self, node_id: str, snapshot_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            fs = FeatureSnapshot.from_dict(snapshot_dict)
            self._tree.attach_feature_snapshot(node_id, fs)
            for feat_name in fs.feature_names[:50]:
                feat_nid = self._kg.register_feature_node(feat_name)
                if self._kg.get_node(node_id):
                    from trainer.research.knowledge_graph import EdgeType
                    try:
                        self._kg.add_edge(node_id, feat_nid, EdgeType.USES_FEATURE)
                    except Exception:
                        pass
            return {"node_id": node_id, "snapshot_id": fs.snapshot_id}
        return _wrap("lineage.attach_feature_snapshot", _do)

    def attach_model_snapshot(
        self, node_id: str, snapshot_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            ms = ModelSnapshot.from_dict(snapshot_dict)
            self._tree.attach_model_snapshot(node_id, ms)
            mdl_nid = self._kg.register_model_node(
                node_id, ms.checkpoint_hash,
                {"architecture": ms.architecture, "backbone": ms.backbone},
            )
            if self._kg.get_node(node_id):
                from trainer.research.knowledge_graph import EdgeType
                try:
                    self._kg.add_edge(node_id, mdl_nid, EdgeType.HAS_MODEL)
                except Exception:
                    pass
            return {"node_id": node_id, "snapshot_id": ms.snapshot_id}
        return _wrap("lineage.attach_model_snapshot", _do)

    def attach_validation_snapshot(
        self, node_id: str, snapshot_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            vs = ValidationSnapshot.from_dict(snapshot_dict)
            self._tree.attach_validation_snapshot(node_id, vs)
            self._tl.record_validation_completed(
                node_id,
                accuracy=vs.walk_forward_accuracy,
                sharpe=vs.sharpe_ratio,
                promotion_decision=vs.promotion_decision,
            )
            return {"node_id": node_id, "snapshot_id": vs.snapshot_id}
        return _wrap("lineage.attach_validation_snapshot", _do)

    def record_cost(
        self,
        entity_id: str,
        *,
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
        storage_gb_months: float = 0.0,
        validation_gpu_hours: float = 0.0,
        inference_hours: float = 0.0,
        project: str = "",
        branch: str = "",
        researcher: str = "system",
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            cost = self._costs.record(
                entity_id,
                project=project,
                branch=branch,
                researcher=researcher,
                gpu_hours=gpu_hours,
                cpu_hours=cpu_hours,
                storage_gb_months=storage_gb_months,
                validation_gpu_hours=validation_gpu_hours,
                inference_hours=inference_hours,
            )
            node = self._tree.get(entity_id)
            if node:
                self._tree.record_cost(
                    entity_id,
                    gpu_hours=gpu_hours,
                    cpu_hours=cpu_hours,
                    training_cost_usd=cost.training_cost_usd,
                    storage_cost_usd=cost.storage_cost_usd,
                    validation_cost_usd=cost.validation_cost_usd,
                )
            self._tl.record_cost(entity_id, cost.total_cost_usd, gpu_hours)
            return cost.to_dict()
        return _wrap("lineage.record_cost", _do)

    # ------------------------------------------------------------------
    # Differencing
    # ------------------------------------------------------------------

    def compare_experiments(
        self, node_id_a: str, node_id_b: str
    ) -> Dict[str, Any]:
        return _wrap(
            "lineage.compare_experiments",
            lambda: self._differ.compare_experiments(node_id_a, node_id_b).to_dict(),
        )

    def compare_models(
        self, node_id_a: str, node_id_b: str
    ) -> Dict[str, Any]:
        def _do() -> Any:
            diff = self._differ.compare_models(node_id_a, node_id_b)
            return diff.to_dict() if diff else {"message": "One or both nodes missing model snapshot"}
        return _wrap("lineage.compare_models", _do)

    def compare_datasets(
        self, node_id_a: str, node_id_b: str
    ) -> Dict[str, Any]:
        def _do() -> Any:
            diff = self._differ.compare_datasets(node_id_a, node_id_b)
            return diff.to_dict() if diff else {"message": "One or both nodes missing dataset snapshot"}
        return _wrap("lineage.compare_datasets", _do)

    def compare_features(
        self, node_id_a: str, node_id_b: str
    ) -> Dict[str, Any]:
        def _do() -> Any:
            diff = self._differ.compare_features(node_id_a, node_id_b)
            return diff.to_dict() if diff else {"message": "One or both nodes missing feature snapshot"}
        return _wrap("lineage.compare_features", _do)

    # ------------------------------------------------------------------
    # Knowledge Graph
    # ------------------------------------------------------------------

    def get_graph_report(self) -> Dict[str, Any]:
        return _wrap("kg.get_graph_report", self._kg.graph_report)

    def add_graph_node(
        self,
        node_type: str,
        label: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            nid = self._kg.add_node(node_type, label, attributes)
            return {"node_id": nid}
        return _wrap("kg.add_graph_node", _do)

    def add_graph_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            eid = self._kg.add_edge(source_id, target_id, edge_type, attributes)
            return {"edge_id": eid}
        return _wrap("kg.add_graph_edge", _do)

    def get_node_neighbours(
        self,
        node_id: str,
        direction: str = "out",
        edge_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [
                n.to_dict()
                for n in self._kg.neighbours(node_id, direction, edge_type)
            ]
        return _wrap("kg.get_node_neighbours", _do)

    def get_model_lineage(self, model_node_id: str) -> Dict[str, Any]:
        return _wrap("kg.get_model_lineage", self._kg.model_lineage, model_node_id)

    def get_shortest_path(
        self, source_id: str, target_id: str
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            path = self._kg.shortest_path(source_id, target_id)
            return {"path": path, "length": len(path)}
        return _wrap("kg.get_shortest_path", _do)

    def get_graph_export(self) -> Dict[str, Any]:
        return _wrap("kg.get_graph_export", self._kg.to_dict)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def get_timeline_report(self) -> Dict[str, Any]:
        return _wrap("timeline.get_report", self._tl.timeline_report)

    def get_experiment_history(self) -> Dict[str, Any]:
        return _wrap("timeline.experiment_history", self._tl.experiment_history)

    def get_champion_history(self) -> Dict[str, Any]:
        return _wrap("timeline.champion_history", self._tl.champion_history)

    def get_dataset_history(self) -> Dict[str, Any]:
        return _wrap("timeline.dataset_history", self._tl.dataset_history)

    def get_recent_events(self, n: int = 20) -> Dict[str, Any]:
        def _do() -> List[Dict[str, Any]]:
            return [e.to_dict() for e in self._tl.recent(n)]
        return _wrap("timeline.get_recent_events", _do)

    def record_timeline_event(
        self,
        event_type: str,
        entity_id: str = "",
        entity_type: str = "",
        description: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        researcher: str = "system",
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            eid = self._tl.record(
                event_type,
                entity_id=entity_id,
                entity_type=entity_type,
                description=description,
                attributes=attributes,
                researcher=researcher,
            )
            return {"event_id": eid}
        return _wrap("timeline.record_event", _do)

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def get_cost_report(self) -> Dict[str, Any]:
        return _wrap("cost.get_report", self._costs.cost_report)

    def get_experiment_cost(self, entity_id: str) -> Dict[str, Any]:
        def _do() -> Any:
            cost = self._costs.get(entity_id)
            return cost.to_dict() if cost else {"message": f"No cost record for {entity_id}"}
        return _wrap("cost.get_experiment_cost", _do)

    def record_cost_estimate(
        self,
        entity_id: str,
        training_seconds: float,
        *,
        gpu_count: int = 1,
        cpu_cores: int = 8,
        storage_gb: float = 5.0,
        project: str = "",
        branch: str = "",
        researcher: str = "system",
    ) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            cost = self._costs.estimate_from_duration(
                entity_id,
                training_seconds,
                gpu_count=gpu_count,
                cpu_cores=cpu_cores,
                storage_gb=storage_gb,
                project=project,
                branch=branch,
                researcher=researcher,
            )
            return cost.to_dict()
        return _wrap("cost.record_cost_estimate", _do)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_dashboard(self, dashboard_type: str) -> Dict[str, Any]:
        return _wrap("dashboard.get", self._dashboard.build, dashboard_type)

    def get_all_dashboards(self) -> Dict[str, Any]:
        return _wrap("dashboard.get_all", self._dashboard.build_all)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_experiment_card(
        self, node_id: str, as_markdown: bool = False
    ) -> Dict[str, Any]:
        def _do() -> Any:
            if as_markdown:
                return {"markdown": self._exporter.experiment_card_markdown(node_id)}
            return self._exporter.experiment_card(node_id)
        return _wrap("export.experiment_card", _do)

    def export_lineage_report(
        self, node_id: str, as_markdown: bool = False
    ) -> Dict[str, Any]:
        def _do() -> Any:
            if as_markdown:
                return {"markdown": self._exporter.lineage_report_markdown(node_id)}
            return self._exporter.lineage_report(node_id)
        return _wrap("export.lineage_report", _do)

    def export_research_report(self, as_markdown: bool = False) -> Dict[str, Any]:
        def _do() -> Any:
            if as_markdown:
                return {"markdown": self._exporter.research_report_markdown()}
            return self._exporter.research_report()
        return _wrap("export.research_report", _do)

    def export_experiment_tree(self) -> Dict[str, Any]:
        return _wrap("export.experiment_tree", self._exporter.experiment_tree)

    def export_knowledge_graph(self) -> Dict[str, Any]:
        return _wrap("export.knowledge_graph", self._exporter.knowledge_graph_export)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            return {
                "lineage_tree": self._tree.tree_summary(),
                "timeline_events": self._tl.total_events(),
                "cost_records": len(self._costs.all_costs()),
                "knowledge_graph": self._kg.graph_report(),
                "total_cost_usd": self._costs.total_cost_usd(),
                "total_gpu_hours": self._costs.total_gpu_hours(),
            }
        return _wrap("lineage.get_stats", _do)
