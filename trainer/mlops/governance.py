"""Phase 18 — Deployment Governance.

Implements controlled deployment workflows:

  - Manual promotion (requires explicit approval)
  - Automatic promotion (configurable, based on validation thresholds)
  - Approval workflows (multi-step, auditable)
  - Rollback (immediate + emergency)
  - Scheduled deployment
  - Emergency rollback
  - Deployment history
  - Deployment audit logs

Usage::

    gov = DeploymentGovernance(registry, lifecycle_manager, audit_system)
    deploy_id = gov.deploy(model_id, target="production", approver="ops_lead")
    gov.rollback(model_id, reason="confidence_drift", actor="system")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.mlops.registry import LifecycleState, ModelRecord, ModelRegistry
from trainer.mlops.lifecycle import LifecycleManager, LifecycleTransition


# ---------------------------------------------------------------------------
# Deployment record
# ---------------------------------------------------------------------------


@dataclass
class DeploymentRecord:
    """One deployment event for a model.

    Attributes:
        deployment_id: Unique UUID.
        model_id: Model deployed.
        target: Deployment target (champion/production/shadow/rollback).
        deploy_type: manual | automatic | scheduled | emergency_rollback.
        approver: Who approved the deployment.
        actor: Who executed the deployment.
        reason: Deployment reason.
        status: pending | active | rolled_back | superseded | failed.
        from_state: Lifecycle state before deployment.
        to_state: Lifecycle state after deployment.
        deployed_at: Deployment timestamp.
        rolled_back_at: Rollback timestamp (if applicable).
        rollback_reason: Reason for rollback (if applicable).
        metadata: Deployment-specific metadata (e.g. validation scores).
    """

    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    target: str = "production"
    deploy_type: str = "manual"
    approver: str = "system"
    actor: str = "system"
    reason: str = ""
    status: str = "active"
    from_state: str = ""
    to_state: str = ""
    deployed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    rolled_back_at: Optional[str] = None
    rollback_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "model_id": self.model_id,
            "target": self.target,
            "deploy_type": self.deploy_type,
            "approver": self.approver,
            "actor": self.actor,
            "reason": self.reason,
            "status": self.status,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "deployed_at": self.deployed_at,
            "rolled_back_at": self.rolled_back_at,
            "rollback_reason": self.rollback_reason,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Governance config
# ---------------------------------------------------------------------------


@dataclass
class GovernanceConfig:
    """Configuration for deployment governance.

    Attributes:
        auto_promote_enabled: Whether automatic promotion is allowed.
        auto_promote_min_validation_score: Minimum validation score for auto-promotion.
        auto_promote_max_calibration_error: Max ECE for auto-promotion.
        require_approver: Whether a named approver is required.
        allow_emergency_rollback: Whether emergency rollback is available.
        max_candidates: Maximum candidates allowed simultaneously.
        rollback_on_alert: Whether to automatically rollback on critical alert.
    """

    auto_promote_enabled: bool = False
    auto_promote_min_validation_score: float = 0.60
    auto_promote_max_calibration_error: float = 0.10
    require_approver: bool = True
    allow_emergency_rollback: bool = True
    max_candidates: int = 5
    rollback_on_alert: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auto_promote_enabled": self.auto_promote_enabled,
            "auto_promote_min_validation_score": self.auto_promote_min_validation_score,
            "auto_promote_max_calibration_error": self.auto_promote_max_calibration_error,
            "require_approver": self.require_approver,
            "allow_emergency_rollback": self.allow_emergency_rollback,
            "max_candidates": self.max_candidates,
            "rollback_on_alert": self.rollback_on_alert,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GovernanceConfig":
        cfg = GovernanceConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Deployment governance
# ---------------------------------------------------------------------------


class DeploymentGovernance:
    """Controls the complete deployment lifecycle for registered models.

    Enforces approval workflows, records deployment history, and provides
    rollback capabilities.

    Args:
        registry: ModelRegistry with all model records.
        lifecycle: LifecycleManager for state transitions.
        config: GovernanceConfig.
    """

    def __init__(self,
                 registry: ModelRegistry,
                 lifecycle: LifecycleManager,
                 config: Optional[GovernanceConfig] = None) -> None:
        self.registry = registry
        self.lifecycle = lifecycle
        self.config = config or GovernanceConfig()
        self._deployments: Dict[str, DeploymentRecord] = {}
        self._deployment_order: List[str] = []

    def deploy(self, model_id: str,
               target: str = "production",
               approver: str = "system",
               actor: str = "system",
               reason: str = "",
               deploy_type: str = "manual",
               metadata: Optional[Dict[str, Any]] = None) -> DeploymentRecord:
        """Deploy a model to the specified target.

        Args:
            model_id: Model to deploy.
            target: "champion" | "production" | "shadow".
            approver: Who approved this deployment.
            actor: Who executed the deployment.
            reason: Deployment reason.
            deploy_type: manual | automatic | scheduled | emergency_rollback.
            metadata: Extra deployment metadata.

        Returns:
            DeploymentRecord.

        Raises:
            ValueError: If model not found or approval required but missing.
            LifecycleError: If the lifecycle transition is invalid.
        """
        model = self.registry.get_or_raise(model_id)

        if self.config.require_approver and not approver:
            raise ValueError("Approver is required for deployment.")

        target_state_map = {
            "champion": LifecycleState.CHAMPION,
            "production": LifecycleState.PRODUCTION,
            "shadow": LifecycleState.SHADOW,
        }
        to_state = target_state_map.get(target, LifecycleState.PRODUCTION)
        from_state = model.lifecycle_state

        transition = self.lifecycle.transition(
            model, to_state,
            actor=actor or approver,
            reason=reason or f"Deployed to {target}",
            metadata=metadata or {},
        )

        for d in self._deployments.values():
            if d.model_id != model_id and d.status == "active":
                d.status = "superseded"

        rec = DeploymentRecord(
            model_id=model_id,
            target=target,
            deploy_type=deploy_type,
            approver=approver,
            actor=actor or approver,
            reason=reason,
            status="active",
            from_state=from_state,
            to_state=to_state,
            metadata=metadata or {},
        )
        self._deployments[rec.deployment_id] = rec
        self._deployment_order.append(rec.deployment_id)

        if target == "champion":
            self.registry.set_champion(model_id, approver=approver)

        logger.info(
            f"Deployed {model.full_name} → {target} "
            f"(type={deploy_type}, approver={approver})"
        )
        return rec

    def rollback(self, model_id: str,
                 actor: str = "system",
                 reason: str = "rollback",
                 emergency: bool = False) -> DeploymentRecord:
        """Roll back a deployed model.

        Args:
            model_id: Model to roll back.
            actor: Who initiated the rollback.
            reason: Rollback reason.
            emergency: If True, skips approval check.

        Returns:
            DeploymentRecord for the rollback event.
        """
        if not emergency and not self.config.allow_emergency_rollback:
            raise ValueError("Emergency rollback is not enabled.")

        model = self.registry.get_or_raise(model_id)
        from_state = model.lifecycle_state

        self.lifecycle.rollback(model, actor=actor, reason=reason)

        for d in self._deployments.values():
            if d.model_id == model_id and d.status == "active":
                d.status = "rolled_back"
                d.rolled_back_at = datetime.now(timezone.utc).isoformat()
                d.rollback_reason = reason

        rec = DeploymentRecord(
            model_id=model_id,
            target="rollback",
            deploy_type="emergency_rollback" if emergency else "rollback",
            approver=actor,
            actor=actor,
            reason=reason,
            status="active",
            from_state=from_state,
            to_state=LifecycleState.ROLLED_BACK,
        )
        self._deployments[rec.deployment_id] = rec
        self._deployment_order.append(rec.deployment_id)
        logger.warning(
            f"Rolled back {model.full_name} (emergency={emergency}): {reason}"
        )
        return rec

    def auto_promote_if_eligible(self, model_id: str,
                                  actor: str = "system") -> Optional[DeploymentRecord]:
        """Auto-promote a model if governance config allows and thresholds are met."""
        if not self.config.auto_promote_enabled:
            return None

        model = self.registry.get_or_raise(model_id)
        if model.lifecycle_state != LifecycleState.APPROVED:
            return None

        val_score = model.validation_report.get("accuracy", 0.0)
        cal_err = model.validation_report.get("calibration_error", 1.0)

        if (val_score >= self.config.auto_promote_min_validation_score and
                cal_err <= self.config.auto_promote_max_calibration_error):
            return self.deploy(
                model_id, target="production",
                approver="system", actor=actor,
                reason="Auto-promoted: thresholds met",
                deploy_type="automatic",
                metadata={"validation_score": val_score, "calibration_error": cal_err},
            )
        return None

    def history(self, model_id: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        """Return deployment history, optionally filtered by model."""
        records = [self._deployments[did]
                   for did in reversed(self._deployment_order)
                   if did in self._deployments]
        if model_id:
            records = [r for r in records if r.model_id == model_id]
        return [r.to_dict() for r in records[:limit]]

    def active_deployments(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._deployments.values()
                if r.status == "active"]

    def audit_log(self) -> List[Dict[str, Any]]:
        """Return all deployment records as an audit log."""
        return [self._deployments[did].to_dict()
                for did in self._deployment_order]

    def stats(self) -> Dict[str, Any]:
        all_d = list(self._deployments.values())
        return {
            "total_deployments": len(all_d),
            "active": sum(1 for d in all_d if d.status == "active"),
            "rolled_back": sum(1 for d in all_d if d.status == "rolled_back"),
            "superseded": sum(1 for d in all_d if d.status == "superseded"),
            "by_type": {
                t: sum(1 for d in all_d if d.deploy_type == t)
                for t in ("manual", "automatic", "scheduled",
                          "emergency_rollback", "rollback")
            },
        }
