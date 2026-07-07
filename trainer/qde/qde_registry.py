"""Phase 27.6 — QDE Version Registry.

In-memory + persistent store for all QDE versions.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.qde.qde_models import QDEStage, QDEVersion


class QDERegistry:
    """Central registry for all QDE versions.

    Maintains:
      - An ordered list of all versions by version_number.
      - A pointer to the currently ACTIVE (champion) version.
      - A persistence layer (JSON file) so state survives restarts.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        self._versions: dict[str, QDEVersion] = {}
        self._active_version_id: str | None = None
        self._storage_path = storage_path or os.environ.get(
            "QDE_REGISTRY_PATH", "/tmp/qde_registry.json"
        )
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._storage_path):
            return
        try:
            with open(self._storage_path) as fh:
                raw = json.load(fh)
            for vd in raw.get("versions", []):
                v = QDEVersion(
                    version_id=vd["version_id"],
                    version_number=vd["version_number"],
                    label=vd["label"],
                    stage=QDEStage(vd["stage"]),
                    created_at=vd["created_at"],
                    updated_at=vd["updated_at"],
                    parent_version_id=vd.get("parent_version_id"),
                    tags=vd.get("tags", []),
                    description=vd.get("description", ""),
                    metrics=vd.get("metrics", {}),
                    stage_timestamps=vd.get("stage_timestamps", {}),
                    deployed_at=vd.get("deployed_at"),
                    retired_at=vd.get("retired_at"),
                    artifact_uri=vd.get("artifact_uri", ""),
                    checksum=vd.get("checksum", ""),
                )
                self._versions[v.version_id] = v
            self._active_version_id = raw.get("active_version_id")
            logger.info(f"QDERegistry loaded {len(self._versions)} versions")
        except Exception as exc:
            logger.warning(f"QDERegistry load failed: {exc}")

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._storage_path) or ".", exist_ok=True)
            data: dict[str, Any] = {
                "active_version_id": self._active_version_id,
                "versions": [v.to_dict() for v in self._versions.values()],
            }
            with open(self._storage_path, "w") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            logger.warning(f"QDERegistry save failed: {exc}")

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def register(self, version: QDEVersion) -> QDEVersion:
        """Register a new QDE version."""
        if version.version_id in self._versions:
            raise ValueError(f"Version {version.version_id} already registered")
        self._versions[version.version_id] = version
        self._save()
        logger.info(f"QDERegistry.register: {version.version_id} ({version.label})")
        return version

    def get(self, version_id: str) -> QDEVersion | None:
        return self._versions.get(version_id)

    def get_by_number(self, version_number: int) -> QDEVersion | None:
        for v in self._versions.values():
            if v.version_number == version_number:
                return v
        return None

    def update(self, version: QDEVersion) -> None:
        if version.version_id not in self._versions:
            raise KeyError(f"Version {version.version_id} not found in registry")
        self._versions[version.version_id] = version
        self._save()

    def list_all(self) -> list[QDEVersion]:
        return sorted(self._versions.values(), key=lambda v: v.version_number)

    def list_by_stage(self, stage: QDEStage) -> list[QDEVersion]:
        return [v for v in self.list_all() if v.stage == stage]

    # ── Active Version ────────────────────────────────────────────────────────

    def get_active(self) -> QDEVersion | None:
        if self._active_version_id:
            return self._versions.get(self._active_version_id)
        return None

    def set_active(self, version_id: str) -> QDEVersion:
        version = self._versions.get(version_id)
        if not version:
            raise KeyError(f"Version {version_id} not found")

        # Retire the previous active version
        if self._active_version_id and self._active_version_id != version_id:
            prev = self._versions.get(self._active_version_id)
            if prev and prev.stage == QDEStage.ACTIVE:
                prev.stage = QDEStage.RETIRED
                prev.retired_at = time.time()
                prev.updated_at = prev.retired_at

        version.stage = QDEStage.ACTIVE
        version.deployed_at = version.deployed_at or time.time()
        version.updated_at = time.time()
        self._active_version_id = version_id
        self._save()
        logger.info(f"QDERegistry.set_active: {version_id}")
        return version

    def retire(self, version_id: str) -> QDEVersion:
        version = self._versions.get(version_id)
        if not version:
            raise KeyError(f"Version {version_id} not found")
        version.stage = QDEStage.RETIRED
        version.retired_at = time.time()
        version.updated_at = version.retired_at
        self._save()
        return version

    # ── Helpers ───────────────────────────────────────────────────────────────

    def next_version_number(self) -> int:
        if not self._versions:
            return 1
        return max(v.version_number for v in self._versions.values()) + 1

    def lineage(self, version_id: str) -> list[QDEVersion]:
        """Return the ancestry chain from earliest ancestor to this version."""
        chain: list[QDEVersion] = []
        current = self._versions.get(version_id)
        while current:
            chain.append(current)
            parent_id = current.parent_version_id
            current = self._versions.get(parent_id) if parent_id else None
        chain.reverse()
        return chain

    def summary(self) -> dict[str, Any]:
        active = self.get_active()
        by_stage: dict[str, int] = {}
        for v in self._versions.values():
            by_stage[v.stage.value] = by_stage.get(v.stage.value, 0) + 1
        return {
            "total_versions": len(self._versions),
            "active_version_id": self._active_version_id,
            "active_version_number": active.version_number if active else None,
            "by_stage": by_stage,
        }
