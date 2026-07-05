"""Phase 19.1 — Research Knowledge Graph.

A directed graph linking every research entity:

  Node types:   Experiment, Dataset, Model, Feature, ValidationRun,
                Deployment, ChampionModel, RetiredModel
  Edge types:   USES_DATASET, HAS_MODEL, USES_FEATURE, VALIDATED_BY,
                DEPLOYED_AS, PROMOTED_TO, RETIRED, PARENT_OF,
                DERIVED_FROM, COMPARED_WITH

Allows full lineage navigation, ancestor/descendant queries,
and "what used dataset X" style queries.

Usage::

    kg = KnowledgeGraph()
    exp_id = kg.add_node("experiment", "exp_001", {"researcher": "alice"})
    ds_id  = kg.add_node("dataset",    "ds_v2.1", {"sample_count": 1_200_000})
    kg.add_edge(exp_id, ds_id, "USES_DATASET")
    neighbours = kg.neighbours(exp_id)
    path = kg.shortest_path(exp_id, ds_id)
    report = kg.graph_report()
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Node and Edge constants
# ---------------------------------------------------------------------------


class NodeType:
    EXPERIMENT      = "experiment"
    DATASET         = "dataset"
    MODEL           = "model"
    FEATURE         = "feature"
    VALIDATION_RUN  = "validation_run"
    DEPLOYMENT      = "deployment"
    CHAMPION_MODEL  = "champion_model"
    RETIRED_MODEL   = "retired_model"

    ALL = [
        EXPERIMENT, DATASET, MODEL, FEATURE, VALIDATION_RUN,
        DEPLOYMENT, CHAMPION_MODEL, RETIRED_MODEL,
    ]


class EdgeType:
    USES_DATASET    = "USES_DATASET"
    HAS_MODEL       = "HAS_MODEL"
    USES_FEATURE    = "USES_FEATURE"
    VALIDATED_BY    = "VALIDATED_BY"
    DEPLOYED_AS     = "DEPLOYED_AS"
    PROMOTED_TO     = "PROMOTED_TO"
    RETIRED         = "RETIRED"
    PARENT_OF       = "PARENT_OF"
    DERIVED_FROM    = "DERIVED_FROM"
    COMPARED_WITH   = "COMPARED_WITH"

    ALL = [
        USES_DATASET, HAS_MODEL, USES_FEATURE, VALIDATED_BY,
        DEPLOYED_AS, PROMOTED_TO, RETIRED, PARENT_OF,
        DERIVED_FROM, COMPARED_WITH,
    ]


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    label: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# GraphEdge
# ---------------------------------------------------------------------------


@dataclass
class GraphEdge:
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    """Directed knowledge graph of all Signalium research entities."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        # adjacency: source → list[edge_id]
        self._out_edges: Dict[str, List[str]] = {}
        # reverse adjacency: target → list[edge_id]
        self._in_edges: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_type: str,
        label: str,
        attributes: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """Add or update a node, returning its node_id."""
        nid = node_id or str(uuid.uuid4())
        node = GraphNode(
            node_id=nid,
            node_type=node_type,
            label=label,
            attributes=attributes or {},
        )
        self._nodes[nid] = node
        self._out_edges.setdefault(nid, [])
        self._in_edges.setdefault(nid, [])
        return nid

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def update_node_attributes(
        self, node_id: str, attributes: Dict[str, Any]
    ) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.attributes.update(attributes)

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a directed edge, returning its edge_id."""
        if source_id not in self._nodes:
            raise KeyError(f"Source node not found: {source_id}")
        if target_id not in self._nodes:
            raise KeyError(f"Target node not found: {target_id}")
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            attributes=attributes or {},
        )
        self._edges[edge.edge_id] = edge
        self._out_edges[source_id].append(edge.edge_id)
        self._in_edges.setdefault(target_id, []).append(edge.edge_id)
        return edge.edge_id

    def get_edge(self, edge_id: str) -> Optional[GraphEdge]:
        return self._edges.get(edge_id)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def neighbours(
        self,
        node_id: str,
        direction: str = "out",
        edge_type: Optional[str] = None,
    ) -> List[GraphNode]:
        """Return adjacent nodes.  direction: 'out', 'in', or 'both'."""
        results: List[GraphNode] = []
        edge_ids: List[str] = []
        if direction in ("out", "both"):
            edge_ids.extend(self._out_edges.get(node_id, []))
        if direction in ("in", "both"):
            edge_ids.extend(self._in_edges.get(node_id, []))
        seen: Set[str] = set()
        for eid in edge_ids:
            edge = self._edges.get(eid)
            if edge is None:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            other_id = (
                edge.target_id if edge.source_id == node_id else edge.source_id
            )
            if other_id not in seen:
                seen.add(other_id)
                node = self._nodes.get(other_id)
                if node:
                    results.append(node)
        return results

    def reachable(
        self, start_id: str, direction: str = "out"
    ) -> List[GraphNode]:
        """Return all reachable nodes from start_id via BFS."""
        visited: Set[str] = {start_id}
        queue: deque = deque([start_id])
        result: List[GraphNode] = []
        while queue:
            nid = queue.popleft()
            for neighbour in self.neighbours(nid, direction=direction):
                if neighbour.node_id not in visited:
                    visited.add(neighbour.node_id)
                    result.append(neighbour)
                    queue.append(neighbour.node_id)
        return result

    def shortest_path(
        self, source_id: str, target_id: str
    ) -> List[str]:
        """Return the shortest path (node IDs) between two nodes; [] if none."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return []
        visited: Set[str] = {source_id}
        queue: deque = deque([[source_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == target_id:
                return path
            for neighbour in self.neighbours(current, direction="both"):
                nid = neighbour.node_id
                if nid not in visited:
                    visited.add(nid)
                    queue.append(path + [nid])
        return []

    # ------------------------------------------------------------------
    # Typed queries
    # ------------------------------------------------------------------

    def nodes_by_type(self, node_type: str) -> List[GraphNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def experiments_using_dataset(self, dataset_node_id: str) -> List[GraphNode]:
        return [
            n for n in self.neighbours(dataset_node_id, direction="in",
                                        edge_type=EdgeType.USES_DATASET)
            if n.node_type == NodeType.EXPERIMENT
        ]

    def experiments_using_feature(self, feature_node_id: str) -> List[GraphNode]:
        return [
            n for n in self.neighbours(feature_node_id, direction="in",
                                        edge_type=EdgeType.USES_FEATURE)
            if n.node_type == NodeType.EXPERIMENT
        ]

    def model_lineage(self, model_node_id: str) -> Dict[str, Any]:
        """Return full lineage for a model: parents, experiments, validation."""
        parents = self.neighbours(model_node_id, direction="in",
                                   edge_type=EdgeType.PARENT_OF)
        experiments = [
            n for n in self.neighbours(model_node_id, direction="in",
                                        edge_type=EdgeType.HAS_MODEL)
        ]
        validation_runs = self.neighbours(
            model_node_id, direction="out", edge_type=EdgeType.VALIDATED_BY
        )
        deployments = self.neighbours(
            model_node_id, direction="out", edge_type=EdgeType.DEPLOYED_AS
        )
        return {
            "model_node_id": model_node_id,
            "parent_models": [n.to_dict() for n in parents],
            "experiments": [n.to_dict() for n in experiments],
            "validation_runs": [n.to_dict() for n in validation_runs],
            "deployments": [n.to_dict() for n in deployments],
        }

    # ------------------------------------------------------------------
    # From LineageTree / snapshots helpers
    # ------------------------------------------------------------------

    def register_experiment_node(
        self, node_id: str, name: str, attributes: Dict[str, Any]
    ) -> str:
        return self.add_node(
            NodeType.EXPERIMENT, name, attributes, node_id=node_id
        )

    def register_dataset_node(
        self, dataset_version: str, dataset_hash: str, attributes: Dict[str, Any]
    ) -> str:
        nid = f"ds:{dataset_version}:{dataset_hash[:8]}"
        if nid not in self._nodes:
            self.add_node(
                NodeType.DATASET,
                dataset_version,
                {"hash": dataset_hash, **attributes},
                node_id=nid,
            )
        return nid

    def register_model_node(
        self, model_id: str, checkpoint_hash: str, attributes: Dict[str, Any]
    ) -> str:
        nid = f"mdl:{model_id}:{checkpoint_hash[:8]}"
        if nid not in self._nodes:
            self.add_node(
                NodeType.MODEL,
                model_id,
                {"checkpoint_hash": checkpoint_hash, **attributes},
                node_id=nid,
            )
        return nid

    def register_feature_node(self, feature_name: str) -> str:
        nid = f"feat:{feature_name}"
        if nid not in self._nodes:
            self.add_node(NodeType.FEATURE, feature_name, {}, node_id=nid)
        return nid

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges.values()],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeGraph":
        kg = cls()
        for nd in d.get("nodes", []):
            node = GraphNode(
                node_id=nd["node_id"],
                node_type=nd["node_type"],
                label=nd["label"],
                attributes=nd.get("attributes", {}),
                created_at=nd.get("created_at", datetime.now(timezone.utc).isoformat()),
            )
            kg._nodes[node.node_id] = node
            kg._out_edges.setdefault(node.node_id, [])
            kg._in_edges.setdefault(node.node_id, [])
        for ed in d.get("edges", []):
            edge = GraphEdge(
                edge_id=ed["edge_id"],
                source_id=ed["source_id"],
                target_id=ed["target_id"],
                edge_type=ed["edge_type"],
                attributes=ed.get("attributes", {}),
                created_at=ed.get("created_at", datetime.now(timezone.utc).isoformat()),
            )
            kg._edges[edge.edge_id] = edge
            kg._out_edges.setdefault(edge.source_id, []).append(edge.edge_id)
            kg._in_edges.setdefault(edge.target_id, []).append(edge.edge_id)
        return kg

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def graph_report(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = {}
        for n in self._nodes.values():
            type_counts[n.node_type] = type_counts.get(n.node_type, 0) + 1
        edge_type_counts: Dict[str, int] = {}
        for e in self._edges.values():
            edge_type_counts[e.edge_type] = edge_type_counts.get(e.edge_type, 0) + 1
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_type_distribution": type_counts,
            "edge_type_distribution": edge_type_counts,
            "champion_models": len(self.nodes_by_type(NodeType.CHAMPION_MODEL)),
            "retired_models": len(self.nodes_by_type(NodeType.RETIRED_MODEL)),
            "deployments": len(self.nodes_by_type(NodeType.DEPLOYMENT)),
        }
