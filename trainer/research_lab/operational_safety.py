"""S30 — Operational Safety Switches.

Explicit, queryable master switches for the operational-safety scope that
was previously only reachable procedurally (e.g. via withheld approvals or
drawdown-triggered EMERGENCY_LOCKDOWN). This module adds no new business
logic for rollback itself — ``rollback_shortcut`` delegates directly to the
existing ``DeploymentManager``/``QDERollbackManager`` — it only adds the
missing explicit on/off primitives:

  - emergency_disable / clear_emergency_disable  — halts all trading activity
  - pause_model / resume_model                    — per-model-version pause
  - inference_kill_switch / restore_inference     — severs inference itself
    (distinct from pause: pause stops a specific model from being used for
    new decisions, the kill switch stops the inference layer from running
    at all, regardless of which model would have been used)
  - freeze_deployments / unfreeze_deployments      — blocks all stage
    transitions (SHADOW/CANDIDATE/ACTIVE/rollback) regardless of approval
  - rollback_shortcut                              — one-call rollback to
    previous version, bypassing manual lookup of rollback targets
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class SafetyEvent:
    action: str
    actor: str
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "actor": self.actor,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class DeploymentsFrozenError(Exception):
    """Raised when a deployment/rollback operation is attempted while frozen."""


class OperationalSafetySwitches:
    """In-process master switches for the operational-safety scope.

    All state here is intentionally in-memory + audit-logged, mirroring the
    existing ``QDERollbackManager._history`` pattern — persistence, if
    needed, is the caller's responsibility (e.g. wrap with a DB-backed
    subclass), keeping this primitive dependency-free and safe to construct
    anywhere (including inside a startup self-check).
    """

    def __init__(
        self,
        *,
        deployment_manager: Any | None = None,
        qde_rollback_manager: Any | None = None,
    ) -> None:
        self._deployment_manager = deployment_manager
        self._qde_rollback_manager = qde_rollback_manager
        self._emergency_disabled = False
        self._inference_killed = False
        self._deployments_frozen = False
        self._paused_model_versions: set[str] = set()
        self._history: list[SafetyEvent] = []

    # ── Emergency disable (halts all trading activity) ──────────────────────

    def emergency_disable(self, reason: str, actor: str = "system") -> None:
        self._emergency_disabled = True
        self._record("EMERGENCY_DISABLE", actor, reason)
        logger.critical(f"OperationalSafety: EMERGENCY DISABLE engaged by {actor!r} — {reason}")

    def clear_emergency_disable(self, actor: str = "system", reason: str = "") -> None:
        self._emergency_disabled = False
        self._record("CLEAR_EMERGENCY_DISABLE", actor, reason)

    # ── Model pause (per-version) ───────────────────────────────────────────

    def pause_model(self, model_version: str, reason: str, actor: str = "system") -> None:
        self._paused_model_versions.add(model_version)
        self._record("PAUSE_MODEL", actor, f"{model_version}: {reason}")

    def resume_model(self, model_version: str, actor: str = "system", reason: str = "") -> None:
        self._paused_model_versions.discard(model_version)
        self._record("RESUME_MODEL", actor, f"{model_version}: {reason}")

    def is_model_paused(self, model_version: str) -> bool:
        return model_version in self._paused_model_versions

    # ── Inference kill switch (severs inference regardless of model) ───────

    def inference_kill_switch(self, reason: str, actor: str = "system") -> None:
        self._inference_killed = True
        self._record("INFERENCE_KILL_SWITCH", actor, reason)
        logger.critical(f"OperationalSafety: INFERENCE KILL SWITCH engaged by {actor!r} — {reason}")

    def restore_inference(self, actor: str = "system", reason: str = "") -> None:
        self._inference_killed = False
        self._record("RESTORE_INFERENCE", actor, reason)

    # ── Deployment freeze (master toggle over all stage transitions) ───────

    def freeze_deployments(self, reason: str, actor: str = "system") -> None:
        self._deployments_frozen = True
        self._record("FREEZE_DEPLOYMENTS", actor, reason)
        logger.warning(f"OperationalSafety: deployment freeze engaged by {actor!r} — {reason}")

    def unfreeze_deployments(self, actor: str = "system", reason: str = "") -> None:
        self._deployments_frozen = False
        self._record("UNFREEZE_DEPLOYMENTS", actor, reason)

    def assert_not_frozen(self) -> None:
        if self._deployments_frozen:
            raise DeploymentsFrozenError(
                "deployments are frozen; unfreeze before promoting/rolling back"
            )

    # ── Rollback shortcut (delegates to existing rollback authorities) ─────

    def rollback_shortcut(self, reason: str, actor: str = "system") -> dict[str, Any]:
        """One-call rollback to the previous version. Bypasses target lookup
        but never bypasses the freeze switch or the underlying approval/
        rollback authority itself.
        """
        self.assert_not_frozen()
        self._record("ROLLBACK_SHORTCUT", actor, reason)
        if self._qde_rollback_manager is not None:
            record = self._qde_rollback_manager.rollback_to_previous(
                reason=reason, initiated_by=actor
            )
            return record.to_dict() if hasattr(record, "to_dict") else record
        if self._deployment_manager is not None:
            return self._deployment_manager.request_rollback(
                model_version="latest_retired", reason=reason, requested_by=actor
            )
        raise RuntimeError(
            "no rollback authority configured (qde_rollback_manager or deployment_manager)"
        )

    # ── State + audit ────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        return {
            "emergency_disabled": self._emergency_disabled,
            "inference_killed": self._inference_killed,
            "deployments_frozen": self._deployments_frozen,
            "paused_model_versions": sorted(self._paused_model_versions),
        }

    def can_serve_inference(self, model_version: str | None = None) -> bool:
        if self._emergency_disabled or self._inference_killed:
            return False
        if model_version is not None and self.is_model_paused(model_version):
            return False
        return True

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._history[-limit:]]

    def _record(self, action: str, actor: str, reason: str) -> None:
        self._history.append(SafetyEvent(action=action, actor=actor, reason=reason))
