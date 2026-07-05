"""Phase 19.1 — Research Dashboard Builder.

Generates pure-data dashboard payloads (dicts) for seven research
dashboards.  No matplotlib / plotly / torch dependencies.

Dashboard types:

  1. research_timeline    — chronological events chart
  2. knowledge_graph      — node/edge data for graph visualisation
  3. experiment_tree      — hierarchical experiment tree
  4. dataset_history      — dataset version evolution
  5. champion_evolution   — champion model performance over time
  6. training_cost        — cost breakdown charts
  7. feature_evolution    — feature importance and lifecycle evolution

Usage::

    builder = DashboardBuilder(tree, timeline, cost_tracker, kg)
    payload = builder.build("training_cost")
    all_payloads = builder.build_all()
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# DashboardBuilder
# ---------------------------------------------------------------------------


class DashboardBuilder:
    """Generates all seven research dashboard data payloads."""

    DASHBOARD_TYPES = [
        "research_timeline",
        "knowledge_graph",
        "experiment_tree",
        "dataset_history",
        "champion_evolution",
        "training_cost",
        "feature_evolution",
    ]

    def __init__(
        self,
        lineage_tree: Any,
        timeline: Any,
        cost_tracker: Any,
        knowledge_graph: Any,
    ) -> None:
        self._tree = lineage_tree
        self._tl = timeline
        self._costs = cost_tracker
        self._kg = knowledge_graph

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def build(self, dashboard_type: str) -> Dict[str, Any]:
        """Build a single dashboard payload by type."""
        builders = {
            "research_timeline":  self._build_timeline,
            "knowledge_graph":    self._build_knowledge_graph,
            "experiment_tree":    self._build_experiment_tree,
            "dataset_history":    self._build_dataset_history,
            "champion_evolution": self._build_champion_evolution,
            "training_cost":      self._build_training_cost,
            "feature_evolution":  self._build_feature_evolution,
        }
        fn = builders.get(dashboard_type)
        if fn is None:
            raise ValueError(
                f"Unknown dashboard type: {dashboard_type!r}. "
                f"Available: {list(builders)}"
            )
        return {
            "dashboard_type": dashboard_type,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": fn(),
        }

    def build_all(self) -> Dict[str, Any]:
        """Build all seven dashboards and return them as a single payload."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dashboards": {dt: self.build(dt) for dt in self.DASHBOARD_TYPES},
        }

    # ------------------------------------------------------------------
    # 1. Research Timeline
    # ------------------------------------------------------------------

    def _build_timeline(self) -> Dict[str, Any]:
        report = self._tl.timeline_report()
        events = self._tl.all_events()

        # Aggregate events per day for a bar/area chart
        daily: Dict[str, int] = {}
        for e in events:
            day = e.recorded_at[:10]
            daily[day] = daily.get(day, 0) + 1

        type_series: Dict[str, List[Dict[str, Any]]] = {}
        for e in events:
            type_series.setdefault(e.event_type, []).append(
                {"timestamp": e.recorded_at, "entity_id": e.entity_id,
                 "description": e.description}
            )

        return {
            "total_events": report["total_events"],
            "event_type_distribution": report["event_type_distribution"],
            "daily_event_counts": dict(sorted(daily.items())),
            "series_by_type": type_series,
            "experiment_history": report["experiment_history"],
            "champion_history": report["champion_history"],
            "dataset_history": report["dataset_history"],
            "deployment_history": report["deployment_history"],
            "architecture_history": report["architecture_history"],
        }

    # ------------------------------------------------------------------
    # 2. Knowledge Graph
    # ------------------------------------------------------------------

    def _build_knowledge_graph(self) -> Dict[str, Any]:
        kg_dict = self._kg.to_dict()
        report = self._kg.graph_report()

        # Build adjacency summary for frontend graph layout
        adjacency: List[Dict[str, Any]] = []
        for edge in kg_dict["edges"]:
            adjacency.append({
                "source": edge["source_id"],
                "target": edge["target_id"],
                "type": edge["edge_type"],
            })

        return {
            "node_count": report["node_count"],
            "edge_count": report["edge_count"],
            "node_type_distribution": report["node_type_distribution"],
            "edge_type_distribution": report["edge_type_distribution"],
            "nodes": kg_dict["nodes"],
            "edges": kg_dict["edges"],
            "adjacency": adjacency,
            "champion_models": report["champion_models"],
            "retired_models": report["retired_models"],
        }

    # ------------------------------------------------------------------
    # 3. Experiment Tree
    # ------------------------------------------------------------------

    def _build_experiment_tree(self) -> Dict[str, Any]:
        summary = self._tree.tree_summary()
        nodes = self._tree.all_nodes()

        # Build hierarchical tree structure
        def _node_to_tree(node: Any) -> Dict[str, Any]:
            return {
                "node_id": node.node_id,
                "name": node.name,
                "status": node.status,
                "branch": node.branch,
                "project": node.project,
                "researcher": node.researcher,
                "created_at": node.created_at,
                "completed_at": node.completed_at,
                "reproducibility_score": node.reproducibility_score(),
                "total_cost_usd": node.total_cost_usd,
                "children": [
                    _node_to_tree(child_node)
                    for cid in node.child_ids
                    if (child_node := self._tree.get(cid)) is not None
                ],
            }

        roots = self._tree.roots()
        tree_data = [_node_to_tree(r) for r in roots]

        return {
            "summary": summary,
            "tree": tree_data,
            "flat_nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "parent_id": n.parent_id,
                    "status": n.status,
                    "branch": n.branch,
                    "project": n.project,
                }
                for n in nodes
            ],
        }

    # ------------------------------------------------------------------
    # 4. Dataset History
    # ------------------------------------------------------------------

    def _build_dataset_history(self) -> Dict[str, Any]:
        ds_events = self._tl.dataset_history()

        # Extract unique dataset versions from lineage nodes
        dataset_snapshots: List[Dict[str, Any]] = []
        seen: set = set()
        for node in self._tree.all_nodes():
            if node.dataset_snapshot:
                ds = node.dataset_snapshot
                key = ds.dataset_version
                if key not in seen:
                    seen.add(key)
                    dataset_snapshots.append({
                        "version": ds.dataset_version,
                        "hash": ds.dataset_hash[:16] if ds.dataset_hash else "",
                        "sample_count": ds.sample_count,
                        "instrument_count": len(ds.instruments),
                        "quality_score": ds.data_quality_score,
                        "coverage_start": ds.coverage_start,
                        "coverage_end": ds.coverage_end,
                    })

        return {
            "dataset_update_events": ds_events,
            "dataset_snapshots": dataset_snapshots,
            "total_dataset_versions": len(dataset_snapshots),
        }

    # ------------------------------------------------------------------
    # 5. Champion Evolution
    # ------------------------------------------------------------------

    def _build_champion_evolution(self) -> Dict[str, Any]:
        champion_events = self._tl.champion_history()

        # Extract validation metrics from lineage nodes that succeeded
        performance_series: List[Dict[str, Any]] = []
        for node in sorted(self._tree.all_nodes(), key=lambda n: n.created_at):
            if node.validation_snapshot and node.status == "succeeded":
                vs = node.validation_snapshot
                performance_series.append({
                    "experiment_id": node.node_id,
                    "name": node.name,
                    "created_at": node.created_at,
                    "accuracy": vs.walk_forward_accuracy,
                    "f1_macro": vs.walk_forward_f1_macro,
                    "sharpe_ratio": vs.sharpe_ratio,
                    "max_drawdown": vs.max_drawdown,
                    "win_rate": vs.win_rate,
                    "promotion_decision": vs.promotion_decision,
                })

        return {
            "champion_change_events": champion_events,
            "performance_series": performance_series,
            "total_champion_changes": len(champion_events),
        }

    # ------------------------------------------------------------------
    # 6. Training Cost
    # ------------------------------------------------------------------

    def _build_training_cost(self) -> Dict[str, Any]:
        report = self._costs.cost_report()

        # Cost over time series
        cost_series: List[Dict[str, Any]] = [
            {
                "entity_id": c.entity_id,
                "recorded_at": c.recorded_at,
                "total_cost_usd": c.total_cost_usd,
                "gpu_hours": c.gpu_hours,
                "project": c.project,
                "researcher": c.researcher,
            }
            for c in sorted(self._costs.all_costs(), key=lambda c: c.recorded_at)
        ]

        # Cumulative cost
        cumulative = 0.0
        for entry in cost_series:
            cumulative += entry["total_cost_usd"]
            entry["cumulative_cost_usd"] = round(cumulative, 4)

        return {
            "total_cost_usd": report["total_cost_usd"],
            "total_gpu_hours": report["total_gpu_hours"],
            "total_cpu_hours": report["total_cpu_hours"],
            "cost_by_category": report["cost_by_category"],
            "cost_by_project": report["cost_by_project"],
            "cost_by_researcher": report["cost_by_researcher"],
            "most_expensive_experiments": report["most_expensive_experiments"],
            "cost_series": cost_series,
            "rates": report["rates"],
        }

    # ------------------------------------------------------------------
    # 7. Feature Evolution
    # ------------------------------------------------------------------

    def _build_feature_evolution(self) -> Dict[str, Any]:
        # Collect feature snapshots across all lineage nodes
        feature_timeline: List[Dict[str, Any]] = []
        all_feature_names: set = set()

        for node in sorted(self._tree.all_nodes(), key=lambda n: n.created_at):
            if node.feature_snapshot:
                fs = node.feature_snapshot
                all_feature_names.update(fs.feature_names)
                feature_timeline.append({
                    "experiment_id": node.node_id,
                    "experiment_name": node.name,
                    "created_at": node.created_at,
                    "feature_count": fs.feature_count,
                    "feature_set_hash": fs.feature_set_hash,
                    "top_features": fs.top_features(5),
                    "lifecycle_status_counts": _count_values(
                        fs.feature_lifecycle_status
                    ),
                    "drift_status_counts": _count_values(fs.feature_drift_status),
                })

        return {
            "total_unique_features": len(all_feature_names),
            "all_feature_names": sorted(all_feature_names),
            "feature_timeline": feature_timeline,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_values(d: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for v in d.values():
        counts[str(v)] = counts.get(str(v), 0) + 1
    return counts
