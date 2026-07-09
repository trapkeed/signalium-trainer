"""Phase S27 — Multi-role deployment approval workflow.

The existing single-actor `ApprovalRequest`/`ApprovalGate`
(`approval_models.py` / `approval_gate.py`) records one approver's decision.
S27 requires deploying a QDE version — the sole decision maker for live
trading — to institutional dual-control: SuperAdmin *and*
DeploymentApprover must both sign off (mandatory), with Founder,
QuantResearcher, and MLResearcher able to add an optional review, before the
workflow as a whole is considered approved.

This module is a thin layer *above* the single-actor gate: it does not
replace `ApprovalGate`, it aggregates multiple structured reviews into one
workflow record that becomes the immutable record of who signed off on a
given QDE version's deployment.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from trainer.research_lab.sso import Role

MANDATORY_APPROVER_ROLES: frozenset[Role] = frozenset({
    Role.SUPER_ADMIN,
    Role.DEPLOYMENT_APPROVER,
})

OPTIONAL_REVIEWER_ROLES: frozenset[Role] = frozenset({
    Role.FOUNDER,
    Role.QUANT_RESEARCHER,
    Role.ML_RESEARCHER,
})

# Admin is treated as an alias for SuperAdmin's mandatory-approver seat —
# both hold Capability.UNRESTRICTED in sso.py's ROLE_CAPABILITIES.
_SUPER_ADMIN_ALIASES: frozenset[Role] = frozenset({Role.SUPER_ADMIN, Role.ADMIN})


class ReviewDecision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class DeploymentApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class DeploymentReview:
    """One structured reviewer decision on a deployment approval workflow."""

    reviewer: str
    role: Role
    decision: ReviewDecision
    certification_result: str = ""
    notes: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer": self.reviewer,
            "role": self.role.value,
            "decision": self.decision.value,
            "certification_result": self.certification_result,
            "notes": self.notes,
            "timestamp": round(self.timestamp, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentReview:
        return cls(
            reviewer=data["reviewer"],
            role=Role(data["role"]),
            decision=ReviewDecision(data["decision"]),
            certification_result=data.get("certification_result", ""),
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class DeploymentApprovalWorkflow:
    """Aggregates every reviewer decision for one QDE-version deployment.

    Approved only once both mandatory roles (SuperAdmin/Admin and
    DeploymentApprover) have independently recorded APPROVE. A REJECT from
    *any* reviewer — mandatory or optional — rejects the whole workflow;
    institutional governance treats a single objection as a hard stop, not
    something a majority can override.
    """

    workflow_id: str
    qde_version: str
    certification_report_id: str
    created_at: float
    updated_at: float
    status: DeploymentApprovalStatus = DeploymentApprovalStatus.PENDING
    reviews: list[DeploymentReview] = field(default_factory=list)
    requested_by: str = ""
    notes: str = ""

    @classmethod
    def new(
        cls,
        qde_version: str,
        certification_report_id: str = "",
        requested_by: str = "",
        notes: str = "",
    ) -> DeploymentApprovalWorkflow:
        now = time.time()
        return cls(
            workflow_id=f"depapp-{str(uuid.uuid4())[:12]}",
            qde_version=qde_version,
            certification_report_id=certification_report_id,
            created_at=now,
            updated_at=now,
            requested_by=requested_by,
            notes=notes,
        )

    def _mandatory_roles_covered(self) -> set[Role]:
        covered: set[Role] = set()
        for review in self.reviews:
            if review.decision != ReviewDecision.APPROVE:
                continue
            if review.role in _SUPER_ADMIN_ALIASES:
                covered.add(Role.SUPER_ADMIN)
            elif review.role in MANDATORY_APPROVER_ROLES:
                covered.add(review.role)
        return covered

    def add_review(
        self,
        reviewer: str,
        role: Role,
        decision: ReviewDecision,
        certification_result: str = "",
        notes: str = "",
    ) -> DeploymentReview:
        """Record a structured review and recompute workflow status.

        Each (reviewer, role) pair may only review once; a repeat call from
        the same reviewer/role replaces their prior review rather than
        appending a duplicate, so the workflow always reflects each
        participant's *latest* decision without losing the audit trail of
        who reviewed (the replaced review is dropped, not hidden — this
        mirrors the single-actor gate's re-review semantics).
        """
        self.reviews = [
            r for r in self.reviews if not (r.reviewer == reviewer and r.role == role)
        ]
        review = DeploymentReview(
            reviewer=reviewer,
            role=role,
            decision=decision,
            certification_result=certification_result,
            notes=notes,
        )
        self.reviews.append(review)
        self.updated_at = time.time()
        self._recompute_status()
        return review

    def _recompute_status(self) -> None:
        if any(r.decision == ReviewDecision.REJECT for r in self.reviews):
            self.status = DeploymentApprovalStatus.REJECTED
            return
        mandatory_covered = self._mandatory_roles_covered()
        if MANDATORY_APPROVER_ROLES.issubset(mandatory_covered):
            self.status = DeploymentApprovalStatus.APPROVED
        else:
            self.status = DeploymentApprovalStatus.PENDING

    def missing_mandatory_roles(self) -> list[Role]:
        covered = self._mandatory_roles_covered()
        return [role for role in MANDATORY_APPROVER_ROLES if role not in covered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "qde_version": self.qde_version,
            "certification_report_id": self.certification_report_id,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "status": self.status.value,
            "reviews": [r.to_dict() for r in self.reviews],
            "requested_by": self.requested_by,
            "notes": self.notes,
            "missing_mandatory_roles": [r.value for r in self.missing_mandatory_roles()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentApprovalWorkflow:
        wf = cls(
            workflow_id=data["workflow_id"],
            qde_version=data["qde_version"],
            certification_report_id=data.get("certification_report_id", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            status=DeploymentApprovalStatus(data.get("status", "PENDING")),
            requested_by=data.get("requested_by", ""),
            notes=data.get("notes", ""),
        )
        wf.reviews = [DeploymentReview.from_dict(r) for r in data.get("reviews", [])]
        return wf
