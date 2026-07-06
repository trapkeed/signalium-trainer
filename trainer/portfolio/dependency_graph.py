"""Phase 20 — Dependency Graph.

Represents instrument relationships, correlation clusters, risk clusters,
and market clusters. Updated continuously from new correlation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trainer.portfolio.correlation_engine import CorrelationMatrix


class ClusterType(str, Enum):
    CORRELATION = "correlation"
    RISK = "risk"
    MARKET = "market"
    CURRENCY = "currency"


@dataclass
class GraphNode:
    instrument: str
    cluster_id: str | None = None
    centrality: float = 0.0
    risk_weight: float = 1.0
    asset_class: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "cluster_id": self.cluster_id,
            "centrality": round(self.centrality, 4),
            "risk_weight": round(self.risk_weight, 4),
            "asset_class": self.asset_class,
            "tags": self.tags,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    correlation: float
    edge_type: str = "correlation"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 4),
            "correlation": round(self.correlation, 4),
            "edge_type": self.edge_type,
        }


@dataclass
class Cluster:
    cluster_id: str
    cluster_type: ClusterType
    members: list[str]
    centroid_instrument: str
    average_internal_correlation: float
    risk_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "cluster_type": self.cluster_type.value,
            "members": self.members,
            "centroid_instrument": self.centroid_instrument,
            "average_internal_correlation": round(self.average_internal_correlation, 4),
            "risk_score": round(self.risk_score, 4),
        }


@dataclass
class DependencyGraphSnapshot:
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]
    clusters: list[Cluster]
    timestamp: str = ""

    def node_list(self) -> list[dict]:
        return [n.to_dict() for n in self.nodes.values()]

    def edge_list(self) -> list[dict]:
        return [e.to_dict() for e in self.edges]

    def cluster_list(self) -> list[dict]:
        return [c.to_dict() for c in self.clusters]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "nodes": self.node_list(),
            "edges": self.edge_list(),
            "clusters": self.cluster_list(),
            "stats": {
                "n_nodes": len(self.nodes),
                "n_edges": len(self.edges),
                "n_clusters": len(self.clusters),
            },
        }

    def cluster_for(self, instrument: str) -> Cluster | None:
        for c in self.clusters:
            if instrument in c.members:
                return c
        return None

    def neighbors(self, instrument: str, min_weight: float = 0.5) -> list[str]:
        result = []
        for e in self.edges:
            if e.source == instrument and abs(e.weight) >= min_weight:
                result.append(e.target)
            elif e.target == instrument and abs(e.weight) >= min_weight:
                result.append(e.source)
        return result


class DependencyGraphBuilder:
    """Builds and updates the live instrument dependency graph."""

    EDGE_THRESHOLD = 0.3
    CLUSTER_THRESHOLD = 0.6

    def __init__(self, edge_threshold: float = 0.3, cluster_threshold: float = 0.6) -> None:
        self.edge_threshold = edge_threshold
        self.cluster_threshold = cluster_threshold
        self._snapshots: list[DependencyGraphSnapshot] = []

    def build(self, matrix: CorrelationMatrix, timestamp: str = "") -> DependencyGraphSnapshot:
        nodes = self._build_nodes(matrix)
        edges = self._build_edges(matrix)
        self._compute_centrality(nodes, edges)
        clusters = self._build_clusters(matrix, nodes)
        self._assign_clusters(nodes, clusters)

        snapshot = DependencyGraphSnapshot(
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            timestamp=timestamp,
        )
        self._snapshots.append(snapshot)
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]
        return snapshot

    def _build_nodes(self, matrix: CorrelationMatrix) -> dict[str, GraphNode]:
        return {sym: GraphNode(instrument=sym) for sym in matrix.instruments}

    def _build_edges(self, matrix: CorrelationMatrix) -> list[GraphEdge]:
        edges = []
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(matrix.instruments):
            for j, b in enumerate(matrix.instruments):
                if a == b:
                    continue
                key = tuple(sorted([a, b]))
                if key in seen:
                    continue
                seen.add(key)
                corr = matrix.get(a, b)
                if abs(corr) >= self.edge_threshold:
                    edges.append(
                        GraphEdge(
                            source=a,
                            target=b,
                            weight=abs(corr),
                            correlation=corr,
                        )
                    )
        return edges

    @staticmethod
    def _compute_centrality(nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> None:
        degree: dict[str, int] = {sym: 0 for sym in nodes}
        strength: dict[str, float] = {sym: 0.0 for sym in nodes}
        for e in edges:
            degree[e.source] = degree.get(e.source, 0) + 1
            degree[e.target] = degree.get(e.target, 0) + 1
            strength[e.source] = strength.get(e.source, 0.0) + abs(e.weight)
            strength[e.target] = strength.get(e.target, 0.0) + abs(e.weight)
        max_strength = max(strength.values()) if strength else 1.0
        for sym, node in nodes.items():
            node.centrality = round(strength.get(sym, 0.0) / max(max_strength, 1e-9), 4)

    def _build_clusters(
        self, matrix: CorrelationMatrix, nodes: dict[str, GraphNode]
    ) -> list[Cluster]:
        symbols = matrix.instruments
        n = len(symbols)
        if n == 0:
            return []

        # Greedy single-link clustering by correlation
        assigned: dict[str, str] = {}
        clusters: dict[str, list[str]] = {}
        cluster_idx = 0

        for i in range(n):
            a = symbols[i]
            if a in assigned:
                continue
            cid = f"C{cluster_idx}"
            cluster_idx += 1
            assigned[a] = cid
            clusters[cid] = [a]
            for j in range(i + 1, n):
                b = symbols[j]
                if b in assigned:
                    continue
                corr = matrix.get(a, b)
                if abs(corr) >= self.cluster_threshold:
                    assigned[b] = cid
                    clusters[cid].append(b)

        result = []
        for cid, members in clusters.items():
            centroid = self._centroid(members, matrix)
            avg_corr = self._avg_internal_correlation(members, matrix)
            result.append(
                Cluster(
                    cluster_id=cid,
                    cluster_type=ClusterType.CORRELATION,
                    members=members,
                    centroid_instrument=centroid,
                    average_internal_correlation=avg_corr,
                )
            )
        return result

    @staticmethod
    def _centroid(members: list[str], matrix: CorrelationMatrix) -> str:
        if len(members) == 1:
            return members[0]
        best = members[0]
        best_sum = -1.0
        for a in members:
            s = sum(abs(matrix.get(a, b)) for b in members if b != a)
            if s > best_sum:
                best_sum = s
                best = a
        return best

    @staticmethod
    def _avg_internal_correlation(members: list[str], matrix: CorrelationMatrix) -> float:
        if len(members) < 2:
            return 1.0
        pairs = 0
        total = 0.0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                total += abs(matrix.get(members[i], members[j]))
                pairs += 1
        return round(total / max(pairs, 1), 4)

    @staticmethod
    def _assign_clusters(nodes: dict[str, GraphNode], clusters: list[Cluster]) -> None:
        for cluster in clusters:
            for sym in cluster.members:
                if sym in nodes:
                    nodes[sym].cluster_id = cluster.cluster_id

    def latest_snapshot(self) -> DependencyGraphSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def history(self, n: int = 10) -> list[DependencyGraphSnapshot]:
        return self._snapshots[-n:]

    def cluster_evolution(self) -> list[dict]:
        return [{"timestamp": s.timestamp, "n_clusters": len(s.clusters)} for s in self._snapshots]
