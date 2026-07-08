"""QDE Version Registry — retired as an independent store.

This module used to own its own persisted dict of `QDEVersion` objects. It is
now a thin, backward-compatible adapter over
`trainer.registry.unified_registry.UnifiedModelRegistry`, which is the sole
source of truth for model versions, deployment state, champion status,
rollback history, and lineage across the platform.

All existing callers (`QDEPipeline`, `QDEPromotionEngine`, `QDERollback`,
`QDEServer`, `QDEAPI`, `QDEABTesting`) keep working unchanged against this
class's public interface.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.qde.qde_inference import HeuristicQDEDecisionModel, QDEDecisionModel
from trainer.qde.qde_models import QDEStage, QDEVersion
from trainer.registry.unified_registry import (
    ChampionStatus,
    UnifiedModelEntry,
    UnifiedModelRegistry,
)

_QDE_FAMILY = "qde"

# In-process map of version_id -> registered decision-model implementation.
# Populated via `QDERegistry.register_decision_model` once a real trained
# artifact is wired up; falls back to `HeuristicQDEDecisionModel` otherwise.
_DECISION_MODELS: dict[str, QDEDecisionModel] = {}


def _qde_version_to_entry(version: QDEVersion) -> UnifiedModelEntry:
    return UnifiedModelEntry(
        entry_id=version.version_id,
        source_system="qde",
        family=_QDE_FAMILY,
        label=version.label,
        model_version=str(version.version_number),
        artifact_uri=version.artifact_uri,
        checksum=version.checksum,
        parent_entry_id=version.parent_version_id,
        tags=list(version.tags),
        description=version.description,
        metrics=dict(version.metrics),
        deployment_state=version.stage.value,
        champion_status=(
            ChampionStatus.CHAMPION if version.stage == QDEStage.ACTIVE else ChampionStatus.NONE
        ),
        created_at=version.created_at,
        updated_at=version.updated_at,
        native_snapshot=version.to_dict(),
        native_type="QDEVersion",
    )


def _entry_to_qde_version(entry: UnifiedModelEntry) -> QDEVersion:
    snap = entry.native_snapshot
    return QDEVersion(
        version_id=snap["version_id"],
        version_number=snap["version_number"],
        label=snap["label"],
        stage=QDEStage(snap["stage"]),
        created_at=snap["created_at"],
        updated_at=snap["updated_at"],
        parent_version_id=snap.get("parent_version_id"),
        tags=snap.get("tags", []),
        description=snap.get("description", ""),
        metrics=snap.get("metrics", {}),
        stage_timestamps=snap.get("stage_timestamps", {}),
        deployed_at=snap.get("deployed_at"),
        retired_at=snap.get("retired_at"),
        artifact_uri=snap.get("artifact_uri", ""),
        checksum=snap.get("checksum", ""),
    )


class QDERegistry:
    """Central registry for all QDE versions.

    Deprecated as an independent store: all state now lives in
    `UnifiedModelRegistry`. This class only translates QDE-shaped calls into
    unified-registry operations and reconstructs `QDEVersion` objects for
    callers that expect them.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        self._storage_path = storage_path or os.environ.get(
            "QDE_REGISTRY_PATH", "/tmp/qde_registry.json"
        )
        self._unified = UnifiedModelRegistry(
            storage_path=self._storage_path,
            reconstruct_native=_entry_to_qde_version,
        )

    # ── CRUD ─────────────────────────────────────────────────────────────

    def register(self, version: QDEVersion) -> QDEVersion:
        entry = _qde_version_to_entry(version)
        self._unified.register(entry, native_ref=version)
        logger.info(f"QDERegistry.register: {version.version_id} ({version.label})")
        return version

    def get(self, version_id: str) -> QDEVersion | None:
        native = self._unified.get_native(version_id)
        return native

    def get_by_number(self, version_number: int) -> QDEVersion | None:
        for entry in self._unified.list_all(family=_QDE_FAMILY):
            if entry.model_version == str(version_number):
                return self._unified.get_native(entry.entry_id)
        return None

    def update(self, version: QDEVersion) -> None:
        entry = self._unified.get(version.version_id)
        if entry is None:
            raise KeyError(f"Version {version.version_id} not found in registry")
        updated_entry = _qde_version_to_entry(version)
        updated_entry.champion_status = entry.champion_status
        updated_entry.promotion_history = entry.promotion_history
        updated_entry.rollback_history = entry.rollback_history
        updated_entry.experiment_lineage = entry.experiment_lineage
        updated_entry.research_approval_status = entry.research_approval_status
        self._unified.update(updated_entry, native_ref=version)

    def list_all(self) -> list[QDEVersion]:
        return [
            self._unified.get_native(e.entry_id)
            for e in sorted(
                self._unified.list_all(family=_QDE_FAMILY),
                key=lambda e: int(e.model_version),
            )
        ]

    def list_by_stage(self, stage: QDEStage) -> list[QDEVersion]:
        return [v for v in self.list_all() if v.stage == stage]

    # ── Active Version ────────────────────────────────────────────────────

    def get_active(self) -> QDEVersion | None:
        champion = self._unified.get_champion(_QDE_FAMILY)
        return self._unified.get_native(champion.entry_id) if champion else None

    def register_decision_model(self, version_id: str, model: QDEDecisionModel) -> None:
        """Attach an invocable `QDEDecisionModel` to a specific QDE version.

        Call this once a real trained artifact is available for a version;
        until then, `get_active_model()` falls back to a heuristic baseline.
        """
        model.version_id = version_id
        _DECISION_MODELS[version_id] = model

    def get_active_model(self) -> QDEDecisionModel:
        """The single live decision-making model entry point.

        Resolves the current active/champion QDE version (via the unified
        registry's champion tracking) and returns its registered decision
        model, falling back to `HeuristicQDEDecisionModel` if none has been
        registered yet. This is the one place `ExecutionPlanner` asks "what
        model makes the decision right now?" — it never re-derives the
        decision itself.
        """
        active = self.get_active()
        if active is None:
            return HeuristicQDEDecisionModel(version=None)
        model = _DECISION_MODELS.get(active.version_id)
        if model is None:
            model = HeuristicQDEDecisionModel(version=active)
            _DECISION_MODELS[active.version_id] = model
        return model

    def set_active(self, version_id: str) -> QDEVersion:
        version = self.get(version_id)
        if not version:
            raise KeyError(f"Version {version_id} not found")

        previous = self.get_active()

        # Ordering matters for the unified registry's consistency invariants:
        # an entry may never simultaneously be CHAMPION and in a terminal
        # deployment_state. `version` may currently hold a terminal state
        # (e.g. RETIRED, if it's being re-activated after a rollback), so we
        # must move it to ACTIVE *before* granting champion status. Likewise
        # `previous` must be demoted from CHAMPION (via `set_champion` below)
        # before it is moved into a terminal state.
        now = __import__("time").time()
        version.stage = QDEStage.ACTIVE
        version.deployed_at = version.deployed_at or now
        version.updated_at = now
        self.update(version)

        self._unified.set_champion(version_id, family=_QDE_FAMILY)

        if previous and previous.version_id != version_id:
            previous.stage = QDEStage.RETIRED
            previous.retired_at = previous.updated_at = __import__("time").time()
            self.update(previous)

        logger.info(f"QDERegistry.set_active: {version_id}")
        return version

    def retire(self, version_id: str) -> QDEVersion:
        version = self.get(version_id)
        if not version:
            raise KeyError(f"Version {version_id} not found")
        import time as _time

        version.stage = QDEStage.RETIRED
        version.retired_at = _time.time()
        version.updated_at = version.retired_at
        self.update(version)
        return version

    # ── Helpers ───────────────────────────────────────────────────────────

    def next_version_number(self) -> int:
        return self._unified.next_version_number(_QDE_FAMILY)

    def lineage(self, version_id: str) -> list[QDEVersion]:
        return [
            self._unified.get_native(e.entry_id) for e in self._unified.lineage(version_id)
        ]

    def summary(self) -> dict[str, Any]:
        active = self.get_active()
        by_stage: dict[str, int] = {}
        for v in self.list_all():
            by_stage[v.stage.value] = by_stage.get(v.stage.value, 0) + 1
        return {
            "total_versions": len(self._unified.list_all(family=_QDE_FAMILY)),
            "active_version_id": active.version_id if active else None,
            "active_version_number": active.version_number if active else None,
            "by_stage": by_stage,
        }
