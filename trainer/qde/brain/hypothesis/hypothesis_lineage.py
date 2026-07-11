"""QDE Brain — hypothesis lineage.

Tracks the parent/child relationships between hypotheses over time so the
system can reason about *how a belief evolved*:

  Root hypothesis (formed from narrative seed):
    "H1 uptrend is strong — continuation likely"

  Spawned competing sub-hypotheses (Scenario A / B / C):
    A: "clean breakout above 1.0950"       ← most probable, reinforced
    B: "fake breakout, reversal follows"   ← gaining evidence
    C: "deep retracement to 1.0860"        ← low probability, aging

  Resolution:
    A was PROMOTED → CONFIRMED
    B was INVALIDATED (no reversal)
    C was AGED_OUT

The lineage tree lets the failure tracker attribute the right lesson to the
right generational context, and lets the memory bridge reconstruct the full
reasoning path from root to resolution when storing episodes.

Design rules
------------
* ``HypothesisLineage`` is an in-memory index — no persistent storage by
  itself.  ``to_snapshot()`` / ``load_snapshot()`` provide the dict-based
  persistence pattern consistent with the rest of the brain package.
* ``LineageNode`` never holds a reference to the live ``Hypothesis`` object
  (to avoid stale references after hypotheses are mutated or garbage-
  collected).  It stores the ``hypothesis_id`` and the thesis string captured
  at registration time.
* Cycles are prevented: ``spawn_child`` refuses to create a parent/child
  relationship that would make a node its own ancestor.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.brain.brain_types import BeliefDirection

# Resolution states a lineage node can reach
ResolutionState = str   # "ACTIVE" | "INVALIDATED" | "CONFIRMED" | "PROMOTED" | "EXPIRED" | "RETIRED"


# ── Node ──────────────────────────────────────────────────────────────────────


@dataclass
class LineageNode:
    """One node in the hypothesis lineage tree.

    ``generation``
        0 = root hypothesis (no parent).
        1 = first-generation child (spawned from a root).
        2 = grandchild, etc.

    ``resolution``
        How the hypothesis eventually concluded.  "ACTIVE" while it is
        still live.

    ``probability_at_resolution``
        The scenario probability at the time of resolution (0 if not part
        of a set, or if the hypothesis resolved before being added to a set).
    """

    hypothesis_id:             str
    thesis:                    str
    direction:                 BeliefDirection
    parent_id:                 str | None
    children:                  list[str]              = field(default_factory=list)
    set_id:                    str | None             = None
    generation:                int                    = 0
    created_at:                float                  = field(default_factory=time.time)
    resolved_at:               float | None           = None
    resolution:                ResolutionState        = "ACTIVE"
    probability_at_resolution: float                  = 0.0
    confidence_at_resolution:  float                  = 0.0

    def is_terminal(self) -> bool:
        return self.resolution != "ACTIVE"

    def resolve(
        self,
        resolution:                ResolutionState,
        probability_at_resolution: float = 0.0,
        confidence_at_resolution:  float = 0.0,
        now:                       float | None = None,
    ) -> None:
        if self.is_terminal():
            return
        self.resolution                = resolution
        self.resolved_at               = now if now is not None else time.time()
        self.probability_at_resolution = probability_at_resolution
        self.confidence_at_resolution  = confidence_at_resolution

    def to_dict(self) -> dict:
        return {
            "hypothesis_id":             self.hypothesis_id,
            "thesis":                    self.thesis,
            "direction":                 self.direction.value,
            "parent_id":                 self.parent_id,
            "children":                  list(self.children),
            "set_id":                    self.set_id,
            "generation":                self.generation,
            "created_at":                round(self.created_at, 3),
            "resolved_at":               round(self.resolved_at, 3) if self.resolved_at else None,
            "resolution":                self.resolution,
            "probability_at_resolution": round(self.probability_at_resolution, 4),
            "confidence_at_resolution":  round(self.confidence_at_resolution, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> LineageNode:
        return cls(
            hypothesis_id=d["hypothesis_id"],
            thesis=d.get("thesis", ""),
            direction=BeliefDirection(d.get("direction", "NEUTRAL")),
            parent_id=d.get("parent_id"),
            children=d.get("children", []),
            set_id=d.get("set_id"),
            generation=d.get("generation", 0),
            created_at=d.get("created_at", 0.0),
            resolved_at=d.get("resolved_at"),
            resolution=d.get("resolution", "ACTIVE"),
            probability_at_resolution=d.get("probability_at_resolution", 0.0),
            confidence_at_resolution=d.get("confidence_at_resolution", 0.0),
        )


# ── Lineage ───────────────────────────────────────────────────────────────────


class HypothesisLineage:
    """In-memory parent/child lineage tree for hypotheses.

    All hypotheses from all instruments are tracked in a single flat index
    (by ``hypothesis_id``), with parent/child links maintained bidirectionally.
    This mirrors how ``HypothesisRegistry`` works: one instance, all symbols.

    Usage::

        lineage = HypothesisLineage()

        # When a root hypothesis is created:
        root = lineage.register("hyp-001", "trend continuation", BULLISH, set_id="set-A")

        # When competing sub-hypotheses are spawned from the root:
        child_a = lineage.spawn_child("hyp-001", "hyp-002", "clean breakout", BULLISH, set_id="set-A")
        child_b = lineage.spawn_child("hyp-001", "hyp-003", "fake breakout → reversal", BEARISH, set_id="set-A")

        # On resolution:
        lineage.resolve("hyp-002", "PROMOTED", probability_at_resolution=0.72)
        lineage.resolve("hyp-003", "INVALIDATED")

        # Inspect:
        path = lineage.lineage_path("hyp-002")   # [root_node, child_a_node]
        siblings = lineage.siblings("hyp-002")   # [child_b_node]
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        # instrument → list of root hypothesis_ids for that instrument
        self._roots_by_instrument: dict[str, list[str]] = defaultdict(list)

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        hypothesis_id: str,
        thesis:        str,
        direction:     BeliefDirection,
        parent_id:     str | None   = None,
        set_id:        str | None   = None,
        instrument:    str          = "",
        now:           float | None = None,
    ) -> LineageNode:
        """Register a hypothesis as a lineage node.

        Idempotent: if the hypothesis is already registered, return the
        existing node without modification.

        If ``parent_id`` is given, the node is a child of that parent
        (generation = parent.generation + 1).  The parent must already be
        registered or a root node is created regardless.
        """
        now = now if now is not None else time.time()

        if hypothesis_id in self._nodes:
            return self._nodes[hypothesis_id]

        generation = 0
        if parent_id and parent_id in self._nodes:
            generation = self._nodes[parent_id].generation + 1

        node = LineageNode(
            hypothesis_id=hypothesis_id,
            thesis=thesis,
            direction=direction,
            parent_id=parent_id,
            set_id=set_id,
            generation=generation,
            created_at=now,
        )
        self._nodes[hypothesis_id] = node

        # Wire parent → child
        if parent_id and parent_id in self._nodes:
            parent = self._nodes[parent_id]
            if hypothesis_id not in parent.children:
                parent.children.append(hypothesis_id)

        # Track roots per instrument
        if parent_id is None and instrument:
            self._roots_by_instrument[instrument].append(hypothesis_id)

        logger.debug(
            f"[Lineage] registered {'root' if not parent_id else 'child'} "
            f"node {hypothesis_id} gen={generation}"
        )
        return node

    def spawn_child(
        self,
        parent_id:      str,
        child_id:       str,
        child_thesis:   str,
        child_direction: BeliefDirection,
        set_id:         str | None   = None,
        now:            float | None = None,
    ) -> LineageNode | None:
        """Register ``child_id`` as a child of ``parent_id``.

        Returns None if:
        - ``parent_id`` is not registered.
        - The child–parent relationship would create a cycle.
        """
        if parent_id not in self._nodes:
            logger.debug(f"[Lineage] spawn_child: parent {parent_id} not found")
            return None
        if self._would_create_cycle(parent_id, child_id):
            logger.warning(f"[Lineage] spawn_child: cycle detected {parent_id}→{child_id}")
            return None
        return self.register(
            child_id, child_thesis, child_direction,
            parent_id=parent_id, set_id=set_id, now=now,
        )

    # ── Resolution ────────────────────────────────────────────────────────

    def resolve(
        self,
        hypothesis_id:             str,
        resolution:                ResolutionState,
        probability_at_resolution: float       = 0.0,
        confidence_at_resolution:  float       = 0.0,
        now:                       float | None = None,
    ) -> bool:
        """Mark a hypothesis as resolved.

        Returns True if the node was found and updated (or was already
        terminal), False if the node is not registered.
        """
        node = self._nodes.get(hypothesis_id)
        if node is None:
            return False
        node.resolve(
            resolution,
            probability_at_resolution=probability_at_resolution,
            confidence_at_resolution=confidence_at_resolution,
            now=now,
        )
        return True

    # ── Query ─────────────────────────────────────────────────────────────

    def get(self, hypothesis_id: str) -> LineageNode | None:
        return self._nodes.get(hypothesis_id)

    def lineage_path(self, hypothesis_id: str) -> list[LineageNode]:
        """Return the path from the root ancestor down to this node.

        The list is ordered root → leaf.  Returns an empty list if the
        node is not registered.
        """
        node = self._nodes.get(hypothesis_id)
        if node is None:
            return []
        path = [node]
        visited = {hypothesis_id}
        while path[-1].parent_id is not None:
            parent_id = path[-1].parent_id
            if parent_id in visited or parent_id not in self._nodes:
                break
            visited.add(parent_id)
            path.append(self._nodes[parent_id])
        path.reverse()
        return path

    def children(self, hypothesis_id: str) -> list[LineageNode]:
        """Direct children of ``hypothesis_id``."""
        node = self._nodes.get(hypothesis_id)
        if node is None:
            return []
        return [self._nodes[c] for c in node.children if c in self._nodes]

    def siblings(self, hypothesis_id: str) -> list[LineageNode]:
        """Nodes with the same parent as ``hypothesis_id`` (excluding self)."""
        node = self._nodes.get(hypothesis_id)
        if node is None or node.parent_id is None:
            return []
        parent = self._nodes.get(node.parent_id)
        if parent is None:
            return []
        return [
            self._nodes[c] for c in parent.children
            if c != hypothesis_id and c in self._nodes
        ]

    def roots(self, instrument: str = "") -> list[LineageNode]:
        """All root nodes (generation 0), optionally filtered by instrument."""
        if instrument:
            ids = self._roots_by_instrument.get(instrument, [])
            return [self._nodes[i] for i in ids if i in self._nodes]
        return [n for n in self._nodes.values() if n.parent_id is None]

    def active_nodes(self, instrument: str = "") -> list[LineageNode]:
        """All nodes that have not yet resolved."""
        nodes = [n for n in self._nodes.values() if not n.is_terminal()]
        if instrument:
            # filter by set_id prefix or registration order — we check parent lineage
            root_ids = set(self._roots_by_instrument.get(instrument, []))
            result   = []
            for n in nodes:
                path = self.lineage_path(n.hypothesis_id)
                if path and path[0].hypothesis_id in root_ids:
                    result.append(n)
            return result
        return nodes

    def depth(self, hypothesis_id: str) -> int:
        """Depth of the node (0 = root, 1 = child, etc.)."""
        node = self._nodes.get(hypothesis_id)
        return node.generation if node else 0

    # ── Persistence ───────────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "roots_by_instrument": {
                k: list(v) for k, v in self._roots_by_instrument.items()
            },
        }

    def load_snapshot(self, data: dict) -> None:
        self._nodes.clear()
        self._roots_by_instrument.clear()
        for d in data.get("nodes", []):
            try:
                node = LineageNode.from_dict(d)
                self._nodes[node.hypothesis_id] = node
            except Exception:
                pass
        for instr, ids in data.get("roots_by_instrument", {}).items():
            self._roots_by_instrument[instr] = list(ids)

    # ── Internal ──────────────────────────────────────────────────────────

    def _would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        """Return True if making child_id a child of parent_id would create a cycle."""
        # Walk up from parent; if we reach child_id, it's a cycle
        visited = set()
        current = parent_id
        while current is not None:
            if current == child_id:
                return True
            if current in visited:
                break
            visited.add(current)
            node = self._nodes.get(current)
            current = node.parent_id if node else None
        return False


__all__ = [
    "LineageNode",
    "HypothesisLineage",
    "ResolutionState",
]
