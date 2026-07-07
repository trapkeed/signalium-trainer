"""Phase 27.6 — QDE Rollback Manager.

Supports immediate rollback to any previous QDE version.
Maintains a full audit trail of every rollback operation.
"""

from __future__ import annotations

from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.qde.qde_models import QDERollbackRecord, QDEStage, QDEVersion
from trainer.qde.qde_registry import QDERegistry


class QDERollbackManager:
    """Manages rollback operations between QDE versions.

    A rollback:
      1. Marks the current ACTIVE version as ROLLED_BACK.
      2. Restores the target version to ACTIVE status.
      3. Records an immutable ``QDERollbackRecord`` for the audit trail.

    Any RETIRED or DEPLOYMENT_CANDIDATE version is eligible as a rollback target.
    """

    _ELIGIBLE_TARGET_STAGES = {
        QDEStage.RETIRED,
        QDEStage.DEPLOYMENT_CANDIDATE,
        QDEStage.SHADOW_TESTING,
        QDEStage.ACTIVE,  # No-op but allowed
    }

    def __init__(self, registry: QDERegistry) -> None:
        self._registry = registry
        self._history: list[QDERollbackRecord] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def rollback_to(
        self,
        target_version_id: str,
        reason: str,
        initiated_by: str = "system",
        notes: str = "",
    ) -> QDERollbackRecord:
        """Immediately roll back to a specific target version.

        Args:
            target_version_id: The version to restore as active.
            reason: Human-readable reason for the rollback.
            initiated_by: Actor that triggered the rollback (user / system).
            notes: Optional additional context.

        Returns:
            A ``QDERollbackRecord`` capturing the operation.

        Raises:
            KeyError: If the target version does not exist.
            ValueError: If the target version is not in an eligible stage.
        """
        target = self._registry.get(target_version_id)
        if target is None:
            raise KeyError(f"Rollback target {target_version_id!r} not found")
        if target.stage not in self._ELIGIBLE_TARGET_STAGES:
            raise ValueError(
                f"Cannot roll back to version {target_version_id!r} "
                f"in stage {target.stage.value}. "
                f"Eligible stages: {[s.value for s in self._ELIGIBLE_TARGET_STAGES]}"
            )

        current_active = self._registry.get_active()
        from_version_id = current_active.version_id if current_active else "NONE"

        # Mark current active as ROLLED_BACK
        if current_active and current_active.version_id != target_version_id:
            current_active.stage = QDEStage.ROLLED_BACK
            current_active.updated_at = __import__("time").time()
            self._registry.update(current_active)

        # Restore the target as ACTIVE
        self._registry.set_active(target_version_id)

        record = QDERollbackRecord.create(
            from_version_id=from_version_id,
            to_version_id=target_version_id,
            reason=reason,
            initiated_by=initiated_by,
            notes=notes,
        )
        self._history.append(record)
        logger.warning(
            f"QDERollback: {from_version_id} → {target_version_id} "
            f"by {initiated_by!r} | {reason}"
        )
        return record

    def rollback_to_previous(
        self,
        reason: str,
        initiated_by: str = "system",
        notes: str = "",
    ) -> QDERollbackRecord:
        """Roll back to the most recent previously active (RETIRED) version.

        Raises:
            RuntimeError: If no previous version is available.
        """
        active = self._registry.get_active()
        all_versions = self._registry.list_all()

        retired = [
            v for v in all_versions
            if v.stage in (QDEStage.RETIRED, QDEStage.ROLLED_BACK)
            and (active is None or v.version_number < active.version_number)
        ]
        if not retired:
            raise RuntimeError("No previous version available for rollback")

        # Most recent retired version by version_number
        target = max(retired, key=lambda v: v.version_number)
        return self.rollback_to(
            target_version_id=target.version_id,
            reason=reason,
            initiated_by=initiated_by,
            notes=notes,
        )

    def rollback_history(self) -> list[dict[str, Any]]:
        """Return the full audit trail of rollback operations."""
        return [r.to_dict() for r in self._history]

    def eligible_targets(self) -> list[dict[str, Any]]:
        """List all versions that are currently eligible as rollback targets."""
        all_versions = self._registry.list_all()
        return [
            v.to_dict()
            for v in all_versions
            if v.stage in self._ELIGIBLE_TARGET_STAGES
        ]
