"""Phase 19.1 — Experiment Lineage Tracking.

Provides full lineage tracking for every experiment:

  - Unique Experiment ID
  - Parent / Child Experiments
  - Research Branch and Project
  - Researcher and Purpose
  - Status and Notes
  - Full EnvironmentSnapshot (git, python, pytorch, cuda, os, gpu, cpu)
  - DatasetSnapshot, FeatureSnapshot, ModelSnapshot, ValidationSnapshot
  - Cost information

The LineageTree allows full ancestor/descendant navigation and
displays the research family tree for any experiment.

Usage::

    tree = LineageTree()
    node_id = tree.register(
        name="transformer_4h_v2",
        researcher="alice",
        purpose="Test 4-head attention on hourly bar data",
        branch="feature/attention-sweep",
        project="q4-2026-architecture",
        parent_id=parent_node_id,
        environment=EnvironmentSnapshot.capture(random_seed=42),
    )
    tree.attach_dataset_snapshot(node_id, ds)
    tree.attach_model_snapshot(node_id, ms)
    tree.attach_validation_snapshot(node_id, vs)
    tree.complete(node_id, status="succeeded")
    ancestors = tree.ancestors(node_id)
    lineage_report = tree.lineage_report(node_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.research.reproducibility import EnvironmentSnapshot
from trainer.research.snapshots import (
    DatasetSnapshot,
    FeatureSnapshot,
    ModelSnapshot,
    ValidationSnapshot,
)

# ---------------------------------------------------------------------------
# Experiment status constants
# ---------------------------------------------------------------------------


class ExperimentStatus:
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"
    ARCHIVED = "archived"

    ALL = [PLANNED, RUNNING, SUCCEEDED, FAILED, ABANDONED, ARCHIVED]


# ---------------------------------------------------------------------------
# LineageNode — one node in the experiment family tree
# ---------------------------------------------------------------------------


@dataclass
class LineageNode:
    """A single experiment in the research lineage graph."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    researcher: str = "system"
    purpose: str = ""

    # Tree relationships
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)

    # Research context
    branch: str = "main"
    project: str = ""

    # Lifecycle
    status: ExperimentStatus = field(default_factory=lambda: ExperimentStatus.PLANNED)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    # Reproducibility & snapshots
    environment: Optional[EnvironmentSnapshot] = None
    dataset_snapshot: Optional[DatasetSnapshot] = None
    feature_snapshot: Optional[FeatureSnapshot] = None
    model_snapshot: Optional[ModelSnapshot] = None
    validation_snapshot: Optional[ValidationSnapshot] = None

    # Cost
    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    training_cost_usd: float = 0.0
    storage_cost_usd: float = 0.0
    validation_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self.training_cost_usd + self.storage_cost_usd + self.validation_cost_usd

    def reproducibility_score(self) -> float:
        if self.environment is None:
            return 0.0
        base = self.environment.reproducibility_score()
        bonus = 0.0
        if self.dataset_snapshot:
            bonus += 0.05
        if self.feature_snapshot:
            bonus += 0.05
        if self.model_snapshot and self.model_snapshot.checkpoint_hash:
            bonus += 0.10
        if self.validation_snapshot:
            bonus += 0.05
        return min(1.0, base + bonus)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "researcher": self.researcher,
            "purpose": self.purpose,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "branch": self.branch,
            "project": self.project,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
            "tags": self.tags,
            "environment": self.environment.to_dict() if self.environment else None,
            "dataset_snapshot": (
                self.dataset_snapshot.to_dict() if self.dataset_snapshot else None
            ),
            "feature_snapshot": (
                self.feature_snapshot.to_dict() if self.feature_snapshot else None
            ),
            "model_snapshot": (self.model_snapshot.to_dict() if self.model_snapshot else None),
            "validation_snapshot": (
                self.validation_snapshot.to_dict() if self.validation_snapshot else None
            ),
            "gpu_hours": self.gpu_hours,
            "cpu_hours": self.cpu_hours,
            "training_cost_usd": self.training_cost_usd,
            "storage_cost_usd": self.storage_cost_usd,
            "validation_cost_usd": self.validation_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LineageNode":
        env_d = d.get("environment")
        ds_d = d.get("dataset_snapshot")
        fs_d = d.get("feature_snapshot")
        ms_d = d.get("model_snapshot")
        vs_d = d.get("validation_snapshot")
        return cls(
            node_id=d.get("node_id", str(uuid.uuid4())),
            name=d.get("name", ""),
            researcher=d.get("researcher", "system"),
            purpose=d.get("purpose", ""),
            parent_id=d.get("parent_id"),
            child_ids=d.get("child_ids", []),
            branch=d.get("branch", "main"),
            project=d.get("project", ""),
            status=d.get("status", ExperimentStatus.PLANNED),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            completed_at=d.get("completed_at"),
            notes=d.get("notes", ""),
            tags=d.get("tags", []),
            environment=EnvironmentSnapshot.from_dict(env_d) if env_d else None,
            dataset_snapshot=DatasetSnapshot.from_dict(ds_d) if ds_d else None,
            feature_snapshot=FeatureSnapshot.from_dict(fs_d) if fs_d else None,
            model_snapshot=ModelSnapshot.from_dict(ms_d) if ms_d else None,
            validation_snapshot=ValidationSnapshot.from_dict(vs_d) if vs_d else None,
            gpu_hours=d.get("gpu_hours", 0.0),
            cpu_hours=d.get("cpu_hours", 0.0),
            training_cost_usd=d.get("training_cost_usd", 0.0),
            storage_cost_usd=d.get("storage_cost_usd", 0.0),
            validation_cost_usd=d.get("validation_cost_usd", 0.0),
        )


# ---------------------------------------------------------------------------
# LineageTree
# ---------------------------------------------------------------------------


class LineageTree:
    """In-memory experiment lineage graph.

    Persisted via the ResearchLineageDatabase when needed.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, LineageNode] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        researcher: str = "system",
        purpose: str = "",
        branch: str = "main",
        project: str = "",
        parent_id: Optional[str] = None,
        environment: Optional[EnvironmentSnapshot] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> str:
        """Register a new experiment node and return its node_id."""
        node = LineageNode(
            name=name,
            researcher=researcher,
            purpose=purpose,
            branch=branch,
            project=project,
            parent_id=parent_id,
            environment=environment,
            tags=tags or [],
            notes=notes,
            status=ExperimentStatus.PLANNED,
        )
        self._nodes[node.node_id] = node
        if parent_id and parent_id in self._nodes:
            self._nodes[parent_id].child_ids.append(node.node_id)
        logger.debug(f"[LineageTree] registered {node.node_id} ({name})")
        return node.node_id

    def start(self, node_id: str) -> None:
        """Mark an experiment as running."""
        node = self._get(node_id)
        node.status = ExperimentStatus.RUNNING

    def complete(self, node_id: str, status: str = ExperimentStatus.SUCCEEDED) -> None:
        """Mark an experiment as completed."""
        node = self._get(node_id)
        if status not in ExperimentStatus.ALL:
            raise ValueError(f"Unknown status: {status}")
        node.status = status
        node.completed_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Snapshot attachment
    # ------------------------------------------------------------------

    def attach_environment(self, node_id: str, env: EnvironmentSnapshot) -> None:
        self._get(node_id).environment = env

    def attach_dataset_snapshot(self, node_id: str, snapshot: DatasetSnapshot) -> None:
        self._get(node_id).dataset_snapshot = snapshot

    def attach_feature_snapshot(self, node_id: str, snapshot: FeatureSnapshot) -> None:
        self._get(node_id).feature_snapshot = snapshot

    def attach_model_snapshot(self, node_id: str, snapshot: ModelSnapshot) -> None:
        self._get(node_id).model_snapshot = snapshot

    def attach_validation_snapshot(self, node_id: str, snapshot: ValidationSnapshot) -> None:
        self._get(node_id).validation_snapshot = snapshot

    # ------------------------------------------------------------------
    # Cost recording
    # ------------------------------------------------------------------

    def record_cost(
        self,
        node_id: str,
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
        training_cost_usd: float = 0.0,
        storage_cost_usd: float = 0.0,
        validation_cost_usd: float = 0.0,
    ) -> None:
        node = self._get(node_id)
        node.gpu_hours = gpu_hours
        node.cpu_hours = cpu_hours
        node.training_cost_usd = training_cost_usd
        node.storage_cost_usd = storage_cost_usd
        node.validation_cost_usd = validation_cost_usd

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def get(self, node_id: str) -> Optional[LineageNode]:
        return self._nodes.get(node_id)

    def ancestors(self, node_id: str) -> List[LineageNode]:
        """Return all ancestor nodes from parent up to root."""
        result: List[LineageNode] = []
        current = self._nodes.get(node_id)
        seen: set = set()
        while current and current.parent_id and current.parent_id not in seen:
            seen.add(current.node_id)
            parent = self._nodes.get(current.parent_id)
            if parent:
                result.append(parent)
            current = parent
        return result

    def descendants(self, node_id: str) -> List[LineageNode]:
        """Return all descendant nodes (BFS)."""
        result: List[LineageNode] = []
        queue = list(self._nodes.get(node_id, LineageNode()).child_ids)
        seen: set = {node_id}
        while queue:
            cid = queue.pop(0)
            if cid in seen:
                continue
            seen.add(cid)
            child = self._nodes.get(cid)
            if child:
                result.append(child)
                queue.extend(child.child_ids)
        return result

    def roots(self) -> List[LineageNode]:
        """Return nodes with no parent."""
        return [n for n in self._nodes.values() if n.parent_id is None]

    def by_branch(self, branch: str) -> List[LineageNode]:
        return [n for n in self._nodes.values() if n.branch == branch]

    def by_project(self, project: str) -> List[LineageNode]:
        return [n for n in self._nodes.values() if n.project == project]

    def by_status(self, status: str) -> List[LineageNode]:
        return [n for n in self._nodes.values() if n.status == status]

    def by_researcher(self, researcher: str) -> List[LineageNode]:
        return [n for n in self._nodes.values() if n.researcher == researcher]

    # ------------------------------------------------------------------
    # Summary and reports
    # ------------------------------------------------------------------

    def lineage_report(self, node_id: str) -> Dict[str, Any]:
        """Return a full lineage report for a single experiment."""
        node = self._get(node_id)
        ancestors = self.ancestors(node_id)
        descendants = self.descendants(node_id)
        repro_score = node.reproducibility_score()
        return {
            "node": node.to_dict(),
            "ancestor_count": len(ancestors),
            "ancestor_ids": [a.node_id for a in ancestors],
            "descendant_count": len(descendants),
            "descendant_ids": [d.node_id for d in descendants],
            "lineage_depth": len(ancestors),
            "reproducibility_score": repro_score,
            "reproducibility_grade": _grade(repro_score),
            "total_cost_usd": node.total_cost_usd,
            "branch": node.branch,
            "project": node.project,
        }

    def tree_summary(self) -> Dict[str, Any]:
        """Return a high-level summary of the full lineage tree."""
        nodes = list(self._nodes.values())
        total_cost = sum(n.total_cost_usd for n in nodes)
        avg_repro = sum(n.reproducibility_score() for n in nodes) / len(nodes) if nodes else 0.0
        status_counts: Dict[str, int] = {}
        for n in nodes:
            status_counts[n.status] = status_counts.get(n.status, 0) + 1
        projects = sorted({n.project for n in nodes if n.project})
        branches = sorted({n.branch for n in nodes if n.branch})
        researchers = sorted({n.researcher for n in nodes if n.researcher})
        return {
            "total_experiments": len(nodes),
            "root_experiments": len(self.roots()),
            "status_distribution": status_counts,
            "projects": projects,
            "branches": branches,
            "researchers": researchers,
            "total_cost_usd": round(total_cost, 4),
            "average_reproducibility_score": round(avg_repro, 4),
            "reproducibility_grade": _grade(avg_repro),
        }

    def load_node(self, node: LineageNode) -> None:
        """Load a pre-built node into the tree (used by persistence layer)."""
        self._nodes[node.node_id] = node
        if node.parent_id and node.parent_id in self._nodes:
            parent = self._nodes[node.parent_id]
            if node.node_id not in parent.child_ids:
                parent.child_ids.append(node.node_id)

    def all_nodes(self) -> List[LineageNode]:
        return list(self._nodes.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, node_id: str) -> LineageNode:
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"LineageNode not found: {node_id}")
        return node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grade(score: float) -> str:
    if score >= 0.90:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.35:
        return "D"
    return "F"
