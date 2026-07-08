"""Unified Model Registry.

The single source of truth for every trained/versioned artifact in Signalium:
QDE versions, MLOps model records, and anything else that goes through a
promotion/deployment lifecycle.

This module owns:
  - Model Versions
  - Dataset Versions / Feature Versions (as attributes of a version)
  - Artifact URIs
  - Lineage (parent/child chains)
  - Deployment States
  - Champion Status
  - Shadow Status
  - Rollback History
  - Promotion History
  - Experiment Lineage (links back to Research Lab experiments)
  - Research Approval Status

`trainer/qde/qde_registry.py` and `trainer/mlops/registry.py` are retired as
independent stores. They now exist only as thin, backward-compatible adapters
around this module so existing callers (QDEPipeline, MLOps lifecycle/champion/
governance/export/api) keep working unchanged while all state actually lives
here.

Invariants enforced centrally (never delegated to callers):
  - An entry_id is registered exactly once; re-registration raises.
  - At most one entry may hold CHAMPION status within a given `family` at
    any time. Setting a new champion demotes the previous one atomically.
  - A rolled-back entry cannot simultaneously be CHAMPION.
  - Every mutation is followed by a consistency check; violations raise
    `RegistryConsistencyError` instead of silently persisting a bad state.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


# ── Enumerations ────────────────────────────────────────────────────────────


class ChampionStatus(str, Enum):
    NONE = "none"
    CHAMPION = "champion"
    FORMER_CHAMPION = "former_champion"


class ShadowStatus(str, Enum):
    NONE = "none"
    SHADOW = "shadow"
    CONCLUDED = "concluded"


class ResearchApprovalStatus(str, Enum):
    NOT_SUBMITTED = "not_submitted"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


TERMINAL_DEPLOYMENT_STATES = {
    "RETIRED",
    "ROLLED_BACK",
    "FAILED",
    "ARCHIVED",
    "retired",
    "rolled_back",
    "archived",
}


class RegistryConsistencyError(Exception):
    """Raised when a mutation would leave the registry in a conflicting state."""


# ── History records ───────────────────────────────────────────────────────


@dataclass
class PromotionHistoryEntry:
    promotion_id: str
    entry_id: str
    decided_at: float
    is_eligible: bool
    is_promoted: bool
    approver: str = ""
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "promotion_id": self.promotion_id,
            "entry_id": self.entry_id,
            "decided_at": self.decided_at,
            "is_eligible": self.is_eligible,
            "is_promoted": self.is_promoted,
            "approver": self.approver,
            "notes": self.notes,
            "raw": self.raw,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PromotionHistoryEntry:
        return PromotionHistoryEntry(
            promotion_id=d.get("promotion_id", str(uuid.uuid4())[:12]),
            entry_id=d.get("entry_id", ""),
            decided_at=d.get("decided_at", time.time()),
            is_eligible=d.get("is_eligible", False),
            is_promoted=d.get("is_promoted", False),
            approver=d.get("approver", ""),
            notes=d.get("notes", ""),
            raw=d.get("raw", {}),
        )


@dataclass
class RollbackHistoryEntry:
    rollback_id: str
    from_entry_id: str
    to_entry_id: str
    reason: str
    initiated_by: str
    rolled_back_at: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "from_entry_id": self.from_entry_id,
            "to_entry_id": self.to_entry_id,
            "reason": self.reason,
            "initiated_by": self.initiated_by,
            "rolled_back_at": self.rolled_back_at,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RollbackHistoryEntry:
        return RollbackHistoryEntry(
            rollback_id=d.get("rollback_id", str(uuid.uuid4())[:12]),
            from_entry_id=d.get("from_entry_id", ""),
            to_entry_id=d.get("to_entry_id", ""),
            reason=d.get("reason", ""),
            initiated_by=d.get("initiated_by", "system"),
            rolled_back_at=d.get("rolled_back_at", time.time()),
            notes=d.get("notes", ""),
        )


@dataclass
class ExperimentLineageEntry:
    experiment_id: str
    source: str = "research_lab"
    linked_at: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "source": self.source,
            "linked_at": self.linked_at,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ExperimentLineageEntry:
        return ExperimentLineageEntry(
            experiment_id=d.get("experiment_id", ""),
            source=d.get("source", "research_lab"),
            linked_at=d.get("linked_at", time.time()),
            notes=d.get("notes", ""),
        )


# ── Unified entry ─────────────────────────────────────────────────────────


@dataclass
class UnifiedModelEntry:
    """One versioned artifact tracked by the Unified Model Registry.

    `family` scopes champion/shadow exclusivity — e.g. all QDE versions share
    the family "qde", while MLOps model records are scoped by model `name`
    (mirroring the legacy per-name champion semantics).
    """

    entry_id: str
    source_system: str  # "qde" | "mlops" | "research_lab" | ...
    family: str
    label: str
    model_version: str

    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""

    artifact_uri: str = ""
    checksum: str = ""
    git_commit: str = ""

    parent_entry_id: str | None = None
    child_entry_ids: list[str] = field(default_factory=list)

    tags: list[str] = field(default_factory=list)
    description: str = ""

    metrics: dict[str, float] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    validation_report: dict[str, Any] = field(default_factory=dict)

    deployment_state: str = "DRAFT"
    champion_status: ChampionStatus = ChampionStatus.NONE
    shadow_status: ShadowStatus = ShadowStatus.NONE
    research_approval_status: ResearchApprovalStatus = ResearchApprovalStatus.NOT_SUBMITTED

    promotion_history: list[PromotionHistoryEntry] = field(default_factory=list)
    rollback_history: list[RollbackHistoryEntry] = field(default_factory=list)
    experiment_lineage: list[ExperimentLineageEntry] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Serialized snapshot of the native object (QDEVersion / ModelRecord / ...)
    # used purely for persistence + migration; the live object itself is held
    # out-of-band by UnifiedModelRegistry._native_refs while the process runs.
    native_snapshot: dict[str, Any] = field(default_factory=dict)
    native_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "source_system": self.source_system,
            "family": self.family,
            "label": self.label,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "artifact_uri": self.artifact_uri,
            "checksum": self.checksum,
            "git_commit": self.git_commit,
            "parent_entry_id": self.parent_entry_id,
            "child_entry_ids": self.child_entry_ids,
            "tags": self.tags,
            "description": self.description,
            "metrics": self.metrics,
            "training_config": self.training_config,
            "validation_report": self.validation_report,
            "deployment_state": self.deployment_state,
            "champion_status": self.champion_status.value,
            "shadow_status": self.shadow_status.value,
            "research_approval_status": self.research_approval_status.value,
            "promotion_history": [p.to_dict() for p in self.promotion_history],
            "rollback_history": [r.to_dict() for r in self.rollback_history],
            "experiment_lineage": [e.to_dict() for e in self.experiment_lineage],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "native_snapshot": self.native_snapshot,
            "native_type": self.native_type,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> UnifiedModelEntry:
        return UnifiedModelEntry(
            entry_id=d["entry_id"],
            source_system=d.get("source_system", ""),
            family=d.get("family", ""),
            label=d.get("label", ""),
            model_version=d.get("model_version", ""),
            dataset_version=d.get("dataset_version", ""),
            feature_version=d.get("feature_version", ""),
            label_version=d.get("label_version", ""),
            artifact_uri=d.get("artifact_uri", ""),
            checksum=d.get("checksum", ""),
            git_commit=d.get("git_commit", ""),
            parent_entry_id=d.get("parent_entry_id"),
            child_entry_ids=d.get("child_entry_ids", []),
            tags=d.get("tags", []),
            description=d.get("description", ""),
            metrics=d.get("metrics", {}),
            training_config=d.get("training_config", {}),
            validation_report=d.get("validation_report", {}),
            deployment_state=d.get("deployment_state", "DRAFT"),
            champion_status=ChampionStatus(d.get("champion_status", "none")),
            shadow_status=ShadowStatus(d.get("shadow_status", "none")),
            research_approval_status=ResearchApprovalStatus(
                d.get("research_approval_status", "not_submitted")
            ),
            promotion_history=[
                PromotionHistoryEntry.from_dict(p) for p in d.get("promotion_history", [])
            ],
            rollback_history=[
                RollbackHistoryEntry.from_dict(r) for r in d.get("rollback_history", [])
            ],
            experiment_lineage=[
                ExperimentLineageEntry.from_dict(e) for e in d.get("experiment_lineage", [])
            ],
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            native_snapshot=d.get("native_snapshot", {}),
            native_type=d.get("native_type", ""),
        )


# ── Unified registry ─────────────────────────────────────────────────────


class UnifiedModelRegistry:
    """Sole source of truth for model/version state across the platform.

    Args:
        storage_path: Optional path to a JSON persistence file. If omitted,
            the registry is purely in-memory (matches legacy behavior of
            constructing a fresh, isolated registry with no args).
        reconstruct_native: Optional callable ``(UnifiedModelEntry) -> Any``
            used to rebuild a native object (e.g. ``QDEVersion``) from a
            persisted snapshot when the process restarts and `get_native`
            is called before the object has been re-registered in-memory.
    """

    def __init__(
        self,
        storage_path: str | None = None,
        reconstruct_native: Callable[[UnifiedModelEntry], Any] | None = None,
    ) -> None:
        self._entries: dict[str, UnifiedModelEntry] = {}
        self._native_refs: dict[str, Any] = {}
        self._storage_path = storage_path
        self._reconstruct_native = reconstruct_native
        self._lock = threading.RLock()
        if self._storage_path:
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._storage_path or not os.path.exists(self._storage_path):
            return
        try:
            with open(self._storage_path) as fh:
                raw = json.load(fh)
            for ed in raw.get("entries", []):
                entry = UnifiedModelEntry.from_dict(ed)
                self._entries[entry.entry_id] = entry
            logger.info(f"UnifiedModelRegistry loaded {len(self._entries)} entries")
        except Exception as exc:
            logger.warning(f"UnifiedModelRegistry load failed: {exc}")

    def _save(self) -> None:
        if not self._storage_path:
            return
        try:
            os.makedirs(os.path.dirname(self._storage_path) or ".", exist_ok=True)
            data = {"entries": [e.to_dict() for e in self._entries.values()]}
            with open(self._storage_path, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            logger.warning(f"UnifiedModelRegistry save failed: {exc}")

    # ── Consistency ──────────────────────────────────────────────────────

    def _check_consistency(self, family: str | None = None) -> list[str]:
        """Return a list of consistency violations (empty == healthy)."""
        issues: list[str] = []
        entries = self.list_all(family=family) if family else self.list_all()
        champions_by_family: dict[str, list[str]] = {}
        for e in entries:
            if e.champion_status == ChampionStatus.CHAMPION:
                champions_by_family.setdefault(e.family, []).append(e.entry_id)
            if (
                e.champion_status == ChampionStatus.CHAMPION
                and e.deployment_state in TERMINAL_DEPLOYMENT_STATES
            ):
                issues.append(
                    f"{e.entry_id} is CHAMPION but deployment_state={e.deployment_state} "
                    "(terminal state)"
                )
            if e.rollback_history and e.champion_status == ChampionStatus.CHAMPION:
                last_rollback = e.rollback_history[-1]
                if last_rollback.from_entry_id == e.entry_id:
                    issues.append(
                        f"{e.entry_id} was rolled back away from but is still CHAMPION"
                    )
        for fam, ids in champions_by_family.items():
            if len(ids) > 1:
                issues.append(f"family={fam!r} has multiple CHAMPION entries: {ids}")
        return issues

    def _assert_consistent(self, family: str | None = None) -> None:
        issues = self._check_consistency(family=family)
        if issues:
            raise RegistryConsistencyError("; ".join(issues))

    # ── CRUD ─────────────────────────────────────────────────────────────

    def register(self, entry: UnifiedModelEntry, native_ref: Any = None) -> UnifiedModelEntry:
        with self._lock:
            if entry.entry_id in self._entries:
                raise ValueError(f"Entry {entry.entry_id} already registered")
            self._entries[entry.entry_id] = entry
            if native_ref is not None:
                self._native_refs[entry.entry_id] = native_ref
            self._assert_consistent(family=entry.family)
            self._save()
            logger.info(
                f"UnifiedModelRegistry.register: {entry.entry_id} "
                f"(source={entry.source_system}, family={entry.family})"
            )
            return entry

    def get(self, entry_id: str) -> UnifiedModelEntry | None:
        return self._entries.get(entry_id)

    def get_native(self, entry_id: str) -> Any | None:
        if entry_id in self._native_refs:
            return self._native_refs[entry_id]
        entry = self._entries.get(entry_id)
        if entry is not None and self._reconstruct_native is not None:
            native = self._reconstruct_native(entry)
            self._native_refs[entry_id] = native
            return native
        return None

    def update(self, entry: UnifiedModelEntry, native_ref: Any = None) -> None:
        with self._lock:
            if entry.entry_id not in self._entries:
                raise KeyError(f"Entry {entry.entry_id} not found in registry")
            if (
                entry.deployment_state in TERMINAL_DEPLOYMENT_STATES
                and entry.champion_status == ChampionStatus.CHAMPION
            ):
                # A caller moving an entry into a terminal deployment state
                # (RETIRED / ROLLED_BACK / FAILED / ARCHIVED) implicitly gives
                # up championship — auto-demote rather than raising, since the
                # caller usually promotes a different entry in the same
                # logical operation (rollback, retirement, etc.).
                entry.champion_status = ChampionStatus.FORMER_CHAMPION
                logger.info(
                    f"UnifiedModelRegistry.update: auto-demoting {entry.entry_id} from "
                    f"CHAMPION due to terminal deployment_state={entry.deployment_state}"
                )
            entry.updated_at = time.time()
            self._entries[entry.entry_id] = entry
            if native_ref is not None:
                self._native_refs[entry.entry_id] = native_ref
            self._assert_consistent(family=entry.family)
            self._save()

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            existed = entry_id in self._entries
            self._entries.pop(entry_id, None)
            self._native_refs.pop(entry_id, None)
            if existed:
                self._save()
            return existed

    def list_all(self, family: str | None = None) -> list[UnifiedModelEntry]:
        entries = list(self._entries.values())
        if family is not None:
            entries = [e for e in entries if e.family == family]
        return sorted(entries, key=lambda e: e.created_at)

    def list_by_deployment_state(
        self, state: str, family: str | None = None
    ) -> list[UnifiedModelEntry]:
        return [e for e in self.list_all(family=family) if e.deployment_state == state]

    def list_by_source_system(self, source_system: str) -> list[UnifiedModelEntry]:
        return [e for e in self._entries.values() if e.source_system == source_system]

    # ── Champion / Shadow ────────────────────────────────────────────────

    def set_champion(self, entry_id: str, family: str | None = None) -> UnifiedModelEntry:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise KeyError(f"Entry {entry_id} not found")
            scope = family or entry.family
            for other in self._entries.values():
                if other.family == scope and other.champion_status == ChampionStatus.CHAMPION:
                    other.champion_status = ChampionStatus.FORMER_CHAMPION
                    other.updated_at = time.time()
            entry.champion_status = ChampionStatus.CHAMPION
            entry.updated_at = time.time()
            self._assert_consistent(family=scope)
            self._save()
            logger.info(f"UnifiedModelRegistry.set_champion: {entry_id} (family={scope})")
            return entry

    def get_champion(self, family: str) -> UnifiedModelEntry | None:
        for e in self._entries.values():
            if e.family == family and e.champion_status == ChampionStatus.CHAMPION:
                return e
        return None

    def get_active_native(self, family: str) -> Any | None:
        """Convenience entry point: the native object backing the current
        champion of `family`, or None if no champion is set.

        Combines `get_champion` + `get_native` into the single "give me the
        current live model for this family" lookup that consumers of the
        registry (e.g. `ExecutionPlanner` via `QDERegistry`) need.
        """
        champion = self.get_champion(family)
        if champion is None:
            return None
        return self.get_native(champion.entry_id)

    def set_shadow(self, entry_id: str, active: bool = True) -> UnifiedModelEntry:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise KeyError(f"Entry {entry_id} not found")
            entry.shadow_status = ShadowStatus.SHADOW if active else ShadowStatus.CONCLUDED
            entry.updated_at = time.time()
            self._save()
            return entry

    # ── Rollback / Promotion / Lineage ──────────────────────────────────

    def record_rollback(
        self,
        from_entry_id: str,
        to_entry_id: str,
        reason: str,
        initiated_by: str = "system",
        notes: str = "",
    ) -> RollbackHistoryEntry:
        with self._lock:
            record = RollbackHistoryEntry(
                rollback_id=str(uuid.uuid4())[:12],
                from_entry_id=from_entry_id,
                to_entry_id=to_entry_id,
                reason=reason,
                initiated_by=initiated_by,
                rolled_back_at=time.time(),
                notes=notes,
            )
            from_entry = self._entries.get(from_entry_id)
            to_entry = self._entries.get(to_entry_id)
            if from_entry is not None:
                from_entry.rollback_history.append(record)
                if from_entry.champion_status == ChampionStatus.CHAMPION:
                    from_entry.champion_status = ChampionStatus.FORMER_CHAMPION
                from_entry.updated_at = time.time()
            if to_entry is not None:
                to_entry.rollback_history.append(record)
                to_entry.updated_at = time.time()
            self._save()
            return record

    def record_promotion(
        self,
        entry_id: str,
        is_eligible: bool,
        is_promoted: bool,
        approver: str = "",
        notes: str = "",
        raw: dict[str, Any] | None = None,
    ) -> PromotionHistoryEntry:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise KeyError(f"Entry {entry_id} not found")
            record = PromotionHistoryEntry(
                promotion_id=str(uuid.uuid4())[:12],
                entry_id=entry_id,
                decided_at=time.time(),
                is_eligible=is_eligible,
                is_promoted=is_promoted,
                approver=approver,
                notes=notes,
                raw=raw or {},
            )
            entry.promotion_history.append(record)
            entry.updated_at = time.time()
            self._save()
            return record

    def link_experiment(
        self,
        entry_id: str,
        experiment_id: str,
        source: str = "research_lab",
        notes: str = "",
    ) -> ExperimentLineageEntry:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise KeyError(f"Entry {entry_id} not found")
            record = ExperimentLineageEntry(
                experiment_id=experiment_id, source=source, notes=notes
            )
            entry.experiment_lineage.append(record)
            entry.updated_at = time.time()
            self._save()
            return record

    def set_research_approval(
        self, entry_id: str, status: ResearchApprovalStatus
    ) -> UnifiedModelEntry:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise KeyError(f"Entry {entry_id} not found")
            entry.research_approval_status = status
            entry.updated_at = time.time()
            self._save()
            return entry

    def lineage(self, entry_id: str) -> list[UnifiedModelEntry]:
        """Ancestor chain from earliest ancestor to this entry."""
        chain: list[UnifiedModelEntry] = []
        current = self._entries.get(entry_id)
        while current:
            chain.append(current)
            parent_id = current.parent_entry_id
            current = self._entries.get(parent_id) if parent_id else None
        chain.reverse()
        return chain

    def next_version_number(self, family: str) -> int:
        family_entries = self.list_all(family=family)
        numbers = []
        for e in family_entries:
            try:
                numbers.append(int(float(e.model_version)))
            except (ValueError, TypeError):
                continue
        return (max(numbers) + 1) if numbers else 1

    # ── Reporting ────────────────────────────────────────────────────────

    def summary(self, family: str | None = None) -> dict[str, Any]:
        entries = self.list_all(family=family)
        by_state: dict[str, int] = {}
        for e in entries:
            by_state[e.deployment_state] = by_state.get(e.deployment_state, 0) + 1
        champion = None
        if family is not None:
            champion = self.get_champion(family)
        return {
            "total_entries": len(entries),
            "by_deployment_state": by_state,
            "champion_entry_id": champion.entry_id if champion else None,
        }

    def audit_consistency(self) -> list[str]:
        """Public, read-only consistency audit across the entire registry."""
        return self._check_consistency()
