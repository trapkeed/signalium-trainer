"""S20 — Human Approval Gate data models.

Every consequential Research Lab action is gated behind an explicit admin
approval. This module defines the data shapes; the enforcement logic lives in
ApprovalGate (approval_gate.py).

Gate types
----------
EXPERIMENT_LAUNCH — before an experiment is started (PROPOSED → PENDING)
TRAINING_START    — before a training run begins
PROMOTION         — before a model is promoted to champion / production
DEPLOYMENT        — before any environment deployment
ROLLBACK          — before a version rollback is executed

Lifecycle
---------
PENDING → APPROVED (admin approves)
PENDING → REJECTED (admin rejects)
PENDING → EXPIRED  (TTL elapsed without decision)
APPROVED → CONSUMED (the gated action atomically consumed the approval)

Relationship to existing models
--------------------------------
- `ApprovalRecord` in mlops/registry.py: per-model approval history; used as
  audit trail on individual ModelRecord objects. Not the same layer.
- `LifecycleAuthorization` in rl_models.py: machine-to-machine enforcement gate
  for QDE version lifecycle transitions. The ApprovalGate operates at the human
  admin layer above it; an approved PROMOTION gate request typically results in
  a LifecycleAuthorization being created downstream.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ── Enumerations ──────────────────────────────────────────────────────────────


class ApprovalGateType(str, Enum):
    EXPERIMENT_LAUNCH = "EXPERIMENT_LAUNCH"
    TRAINING_START = "TRAINING_START"
    PROMOTION = "PROMOTION"
    DEPLOYMENT = "DEPLOYMENT"
    ROLLBACK = "ROLLBACK"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CONSUMED = "CONSUMED"  # atomically consumed after the gated action executes


# ── Exceptions ────────────────────────────────────────────────────────────────


class ApprovalRequiredError(Exception):
    """Raised when a gated action is attempted without a valid approval.

    Attributes:
        gate_type: The gate that blocked the action.
        resource_id: The resource for which approval was sought.
        detail: Machine-readable detail dict for structured API errors.
    """

    def __init__(
        self,
        gate_type: str,
        resource_id: str,
        reason: str = "No valid approval found",
    ) -> None:
        self.gate_type = gate_type
        self.resource_id = resource_id
        self.reason = reason
        self.detail = {
            "error": "approval_required",
            "gate_type": gate_type,
            "resource_id": resource_id,
            "reason": reason,
        }
        super().__init__(
            f"ApprovalRequired: gate={gate_type!r} resource={resource_id!r} — {reason}"
        )


class ApprovalAlreadyResolved(Exception):
    """Raised when an approval action is attempted on a non-PENDING request."""

    def __init__(self, request_id: str, current_status: str) -> None:
        self.request_id = request_id
        self.current_status = current_status
        super().__init__(
            f"Approval request {request_id!r} is already {current_status!r} — cannot modify"
        )


# ── Approval audit entry ──────────────────────────────────────────────────────


@dataclass
class ApprovalAuditEntry:
    """One event in the permanent approval audit log."""

    event: str  # "created" | "approved" | "rejected" | "expired" | "consumed"
    actor: str
    timestamp: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "actor": self.actor,
            "timestamp": round(self.timestamp, 3),
            "reason": self.reason,
            "metadata": self.metadata,
        }


# ── ApprovalRequest ───────────────────────────────────────────────────────────


# Default TTL for PENDING requests: 24 hours
DEFAULT_TTL_SECONDS: float = 86_400.0


@dataclass
class ApprovalRequest:
    """A pending or resolved admin approval request.

    Lifecycle: PENDING → APPROVED/REJECTED/EXPIRED → (if APPROVED) CONSUMED.
    """

    request_id: str
    gate_type: str  # ApprovalGateType.value
    resource_id: str  # e.g. experiment_id, model_version
    resource_type: str  # e.g. "experiment", "qde_version"
    requested_by: str
    action_summary: str  # human-readable: "Launch experiment exp-abc for v2.1"
    requestor_context: dict[str, Any]
    status: str  # ApprovalStatus.value
    created_at: float
    expires_at: float  # PENDING requests expire after this epoch time
    updated_at: float

    resolved_by: str = ""
    resolved_at: float | None = None
    resolution_reason: str = ""
    consumed_at: float | None = None
    consumed_by: str = ""

    # Permanent audit trail (append-only list of ApprovalAuditEntry.to_dict())
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    # ── Factories ──────────────────────────────────────────────────────────────

    @classmethod
    def new(
        cls,
        gate_type: ApprovalGateType,
        resource_id: str,
        resource_type: str,
        requested_by: str,
        action_summary: str,
        requestor_context: dict[str, Any] | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> ApprovalRequest:
        now = time.time()
        req = cls(
            request_id=f"apr-{str(uuid.uuid4())[:12]}",
            gate_type=gate_type.value,
            resource_id=resource_id,
            resource_type=resource_type,
            requested_by=requested_by,
            action_summary=action_summary,
            requestor_context=requestor_context or {},
            status=ApprovalStatus.PENDING.value,
            created_at=now,
            expires_at=now + ttl_seconds,
            updated_at=now,
        )
        req.audit_log.append(
            ApprovalAuditEntry(
                event="created",
                actor=requested_by,
                timestamp=now,
                reason=action_summary,
            ).to_dict()
        )
        return req

    # ── State transitions ─────────────────────────────────────────────────────

    def is_expired(self) -> bool:
        """True if the request is PENDING and has passed its expiry time."""
        return self.status == ApprovalStatus.PENDING.value and time.time() > self.expires_at

    def approve(self, approved_by: str, reason: str = "") -> None:
        """Admin approves the request (PENDING → APPROVED)."""
        now = time.time()
        if self.status != ApprovalStatus.PENDING.value:
            raise ApprovalAlreadyResolved(self.request_id, self.status)
        if self.is_expired():
            self._expire()
            raise ApprovalAlreadyResolved(self.request_id, self.status)
        self.status = ApprovalStatus.APPROVED.value
        self.resolved_by = approved_by
        self.resolved_at = now
        self.resolution_reason = reason
        self.updated_at = now
        self.audit_log.append(
            ApprovalAuditEntry(
                event="approved", actor=approved_by, timestamp=now, reason=reason
            ).to_dict()
        )

    def reject(self, rejected_by: str, reason: str) -> None:
        """Admin rejects the request (PENDING → REJECTED)."""
        now = time.time()
        if self.status != ApprovalStatus.PENDING.value:
            raise ApprovalAlreadyResolved(self.request_id, self.status)
        if self.is_expired():
            self._expire()
            raise ApprovalAlreadyResolved(self.request_id, self.status)
        self.status = ApprovalStatus.REJECTED.value
        self.resolved_by = rejected_by
        self.resolved_at = now
        self.resolution_reason = reason
        self.updated_at = now
        self.audit_log.append(
            ApprovalAuditEntry(
                event="rejected", actor=rejected_by, timestamp=now, reason=reason
            ).to_dict()
        )

    def _expire(self) -> None:
        """Mark PENDING request as EXPIRED (called internally)."""
        now = time.time()
        self.status = ApprovalStatus.EXPIRED.value
        self.updated_at = now
        self.audit_log.append(
            ApprovalAuditEntry(
                event="expired",
                actor="system",
                timestamp=now,
                reason="TTL elapsed without admin decision",
            ).to_dict()
        )

    def consume(self, consumed_by: str = "system") -> None:
        """Atomically consume an APPROVED request (APPROVED → CONSUMED).

        Called by the gate enforcement layer immediately before the gated action
        executes. Must only be called once per approval.
        """
        now = time.time()
        if self.status != ApprovalStatus.APPROVED.value:
            raise ApprovalAlreadyResolved(self.request_id, self.status)
        self.status = ApprovalStatus.CONSUMED.value
        self.consumed_at = now
        self.consumed_by = consumed_by
        self.updated_at = now
        self.audit_log.append(
            ApprovalAuditEntry(
                event="consumed", actor=consumed_by, timestamp=now, reason="Gated action executed"
            ).to_dict()
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "gate_type": self.gate_type,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "requested_by": self.requested_by,
            "action_summary": self.action_summary,
            "requestor_context": self.requestor_context,
            "status": self.status,
            "created_at": round(self.created_at, 3),
            "expires_at": round(self.expires_at, 3),
            "updated_at": round(self.updated_at, 3),
            "resolved_by": self.resolved_by,
            "resolved_at": round(self.resolved_at, 3) if self.resolved_at else None,
            "resolution_reason": self.resolution_reason,
            "consumed_at": round(self.consumed_at, 3) if self.consumed_at else None,
            "consumed_by": self.consumed_by,
            "audit_log": self.audit_log,
        }
