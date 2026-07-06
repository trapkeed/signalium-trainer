"""Phase 21.6 — Decision Lineage.

Every decision is traceable through:
Decision → Forecast → Model → Dataset → Feature Set → Training Run →
Research Experiment → Git Commit
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class LineageNode:
    """One node in the decision lineage graph."""

    node_id: str
    node_type: str  # DECISION | FORECAST | MODEL | DATASET | FEATURE_SET |
    # TRAINING_RUN | EXPERIMENT | GIT_COMMIT
    label: str
    version: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    parent_ids: List[str] = field(default_factory=list)  # upstream nodes

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "version": self.version,
            "created_at": round(self.created_at, 3),
            "metadata": self.metadata,
            "parent_ids": self.parent_ids,
        }


@dataclass
class LineageRecord:
    """Complete lineage for one decision."""

    lineage_id: str
    decision_id: str
    nodes: List[LineageNode] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def depth(self) -> int:
        return len(self.nodes)

    @property
    def root(self) -> Optional[LineageNode]:
        """The oldest node (git commit or experiment)."""
        return self.nodes[-1] if self.nodes else None

    @property
    def leaf(self) -> Optional[LineageNode]:
        """The newest node (the decision itself)."""
        return self.nodes[0] if self.nodes else None

    def to_dict(self) -> Dict:
        return {
            "lineage_id": self.lineage_id,
            "decision_id": self.decision_id,
            "depth": self.depth,
            "created_at": round(self.created_at, 3),
            "nodes": [n.to_dict() for n in self.nodes],
        }


class DecisionLineage:
    """Builds and queries decision lineage graphs."""

    def __init__(self):
        self._lineages: Dict[str, LineageRecord] = {}  # lineage_id → record
        self._by_decision: Dict[str, str] = {}  # decision_id → lineage_id

    def create_lineage(
        self,
        decision_id: str,
        model_version: str = "",
        dataset_name: str = "",
        feature_set: str = "",
        training_run: str = "",
        experiment: str = "",
        git_commit: str = "",
        forecast_id: str = "",
    ) -> LineageRecord:
        """Build a full lineage record for a decision."""

        lid = str(uuid.uuid4())[:12]
        nodes: List[LineageNode] = []

        # Build nodes from decision (leaf) → git commit (root)
        # 1. Decision
        nodes.append(
            LineageNode(
                node_id=decision_id or str(uuid.uuid4())[:8],
                node_type="DECISION",
                label=f"Decision {decision_id[:8] if decision_id else 'unknown'}",
                parent_ids=[forecast_id] if forecast_id else [],
            )
        )
        # 2. Forecast
        fid = forecast_id or str(uuid.uuid4())[:8]
        nodes.append(
            LineageNode(
                node_id=fid,
                node_type="FORECAST",
                label=f"Forecast {fid[:8]}",
                parent_ids=[model_version] if model_version else [],
            )
        )
        # 3. Model
        if model_version:
            nodes.append(
                LineageNode(
                    node_id=model_version,
                    node_type="MODEL",
                    label=f"Model {model_version}",
                    version=model_version,
                    parent_ids=[dataset_name] if dataset_name else [],
                )
            )
        # 4. Dataset
        if dataset_name:
            nodes.append(
                LineageNode(
                    node_id=dataset_name,
                    node_type="DATASET",
                    label=f"Dataset {dataset_name}",
                    parent_ids=[feature_set] if feature_set else [],
                )
            )
        # 5. Feature Set
        if feature_set:
            nodes.append(
                LineageNode(
                    node_id=feature_set,
                    node_type="FEATURE_SET",
                    label=f"Features {feature_set}",
                    parent_ids=[training_run] if training_run else [],
                )
            )
        # 6. Training Run
        if training_run:
            nodes.append(
                LineageNode(
                    node_id=training_run,
                    node_type="TRAINING_RUN",
                    label=f"Training {training_run}",
                    parent_ids=[experiment] if experiment else [],
                )
            )
        # 7. Experiment
        if experiment:
            nodes.append(
                LineageNode(
                    node_id=experiment,
                    node_type="EXPERIMENT",
                    label=f"Experiment {experiment}",
                    parent_ids=[git_commit] if git_commit else [],
                )
            )
        # 8. Git commit
        if git_commit:
            nodes.append(
                LineageNode(
                    node_id=git_commit,
                    node_type="GIT_COMMIT",
                    label=f"Commit {git_commit[:8]}",
                    version=git_commit,
                )
            )

        record = LineageRecord(
            lineage_id=lid,
            decision_id=decision_id,
            nodes=nodes,
        )
        self._lineages[lid] = record
        self._by_decision[decision_id] = lid

        logger.debug(
            f"DecisionLineage: created lineage {lid[:8]} "
            f"depth={record.depth} for decision {decision_id[:8]}"
        )
        return record

    def get_by_decision(self, decision_id: str) -> Optional[LineageRecord]:
        lid = self._by_decision.get(decision_id)
        return self._lineages.get(lid) if lid else None

    def get(self, lineage_id: str) -> Optional[LineageRecord]:
        return self._lineages.get(lineage_id)

    def all_lineages(self) -> List[LineageRecord]:
        return list(self._lineages.values())

    def decisions_by_model(self, model_version: str) -> List[str]:
        """Return decision_ids that used a specific model version."""
        result = []
        for record in self._lineages.values():
            for node in record.nodes:
                if node.node_type == "MODEL" and node.version == model_version:
                    result.append(record.decision_id)
                    break
        return result

    def decisions_by_experiment(self, experiment: str) -> List[str]:
        result = []
        for record in self._lineages.values():
            for node in record.nodes:
                if node.node_type == "EXPERIMENT" and node.label == f"Experiment {experiment}":
                    result.append(record.decision_id)
                    break
        return result

    def stats(self) -> Dict:
        n = len(self._lineages)
        avg_depth = sum(r.depth for r in self._lineages.values()) / n if n else 0.0
        return {
            "total_lineages": n,
            "avg_depth": round(avg_depth, 2),
        }
