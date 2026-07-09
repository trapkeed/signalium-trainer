"""S20 Task #13 — Deployment Approval, Signalium Notification + Rollback.

Implements the full deployment lifecycle for Research Lab model versions:

  SHADOW → CANDIDATE → ACTIVE → RETIRED

Each stage transition requires an ``ApprovalRequest`` (PROMOTION gate for
CANDIDATE, DEPLOYMENT gate for ACTIVE).  Only one model version may be in
ACTIVE state at any time; activating a new version atomically retires the
previous one.

After an ACTIVE promotion is approved, the manager sends a deployment
notification to Signalium containing the weights download URL, deployment
manifest, metadata, and version information.  Signalium handles the hot-swap.

Rollback is a first-class operation: ``request_rollback`` creates a ROLLBACK
``ApprovalRequest``; ``apply_rollback`` restores the most recent RETIRED version
to ACTIVE and re-notifies Signalium with its previously stored artifact
references.

All stage transitions, Signalium notifications, and rollback events are
permanently logged as immutable ``DeploymentEvent`` records.

Dependency direction: this module lives entirely within ``trainer.research_lab``
and never imports from ``trainer.qde``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.approval_models import ApprovalGateType
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.signing import ArtifactSigner, SignatureResult, SigningKeyError


# ── Valid stage transition map ─────────────────────────────────────────────────

#: Maps (from_status, to_status) → required ApprovalGateType.
#: Transitions not in this map are rejected.
_VALID_TRANSITIONS: dict[tuple[str, str], ApprovalGateType] = {
    ("NOT_DEPLOYED", "SHADOW"):    ApprovalGateType.PROMOTION,
    # S21A: a version that has gone through the full pre-SHADOW pipeline
    # (DRAFT → TRAINING → VALIDATING) is also eligible to enter SHADOW.
    ("VALIDATING",   "SHADOW"):   ApprovalGateType.PROMOTION,
    ("SHADOW",       "CANDIDATE"): ApprovalGateType.PROMOTION,
    ("CANDIDATE",   "ACTIVE"):    ApprovalGateType.DEPLOYMENT,
    # Re-promotion paths (e.g. after rollback the model sits at RETIRED)
    ("RETIRED",      "SHADOW"):   ApprovalGateType.PROMOTION,
    ("RETIRED",      "CANDIDATE"): ApprovalGateType.PROMOTION,
    ("RETIRED",      "ACTIVE"):   ApprovalGateType.DEPLOYMENT,
}


#: S21A: target statuses that require every artifact + the manifest for the
#: model version to carry a valid signature before the transition may proceed.
_SIGNATURE_GATED_STATUSES: set[str] = {"SHADOW", "CANDIDATE", "ACTIVE"}


class PromotionBlockedError(Exception):
    """Raised when a promotion is blocked by a failed signature verification."""


#: S21A pre-SHADOW pipeline stages: DRAFT → TRAINING → VALIDATING.
#: These are automatic pipeline-progression transitions (a newly trained
#: version moves through them without requiring a human ApprovalGate — the
#: gated approval decision only starts at the SHADOW boundary and beyond).
#: Not in ``_VALID_TRANSITIONS`` because they carry no ``ApprovalGateType``.
_PRE_SHADOW_ORDER: list[str] = ["NOT_DEPLOYED", "DRAFT", "TRAINING", "VALIDATING"]
_PRE_SHADOW_TRANSITIONS: set[tuple[str, str]] = {
    (_PRE_SHADOW_ORDER[i], _PRE_SHADOW_ORDER[i + 1]) for i in range(len(_PRE_SHADOW_ORDER) - 1)
}


def _gate_type_for(from_status: str, to_status: str) -> ApprovalGateType:
    key = (from_status, to_status)
    if key not in _VALID_TRANSITIONS:
        raise ValueError(
            f"Invalid stage transition: {from_status!r} → {to_status!r}. "
            f"Valid transitions: {list(_VALID_TRANSITIONS)}"
        )
    return _VALID_TRANSITIONS[key]


def _is_valid_pre_shadow_transition(from_status: str, to_status: str) -> bool:
    return (from_status, to_status) in _PRE_SHADOW_TRANSITIONS


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class DeploymentEvent:
    """Immutable audit-log record for a deployment lifecycle event.

    Attributes:
        event_id: Unique identifier.
        model_version: QDE / model version string.
        event_type: One of PROMOTED, RETIRED, ROLLED_BACK, SIGNALIUM_NOTIFIED,
            ROLLBACK_REQUESTED, PROMOTION_REQUESTED.
        from_status: Deployment status before the event.
        to_status: Deployment status after the event (empty for notifications).
        actor: Who executed the action.
        approval_id: The ``ApprovalRequest`` request_id that authorised this
            event (empty for informational events like SIGNALIUM_NOTIFIED).
        created_at: Unix timestamp.
        payload: Event-specific detail dict (notification payload, error info, etc.).
    """

    event_id: str = field(default_factory=lambda: f"devt-{uuid.uuid4().hex[:12]}")
    model_version: str = ""
    event_type: str = ""
    from_status: str = ""
    to_status: str = ""
    actor: str = ""
    approval_id: str = ""
    created_at: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "model_version": self.model_version,
            "event_type": self.event_type,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "actor": self.actor,
            "approval_id": self.approval_id,
            "created_at": round(self.created_at, 3),
            "payload": self.payload,
        }


@dataclass
class DeploymentRecord:
    """Current deployment state for one model version.

    Stored in the ``model_deployments`` table.
    """

    model_version: str
    status: str = "NOT_DEPLOYED"
    previous_status: str = ""
    weights_url: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)
    last_approval_id: str = ""
    last_notification_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "status": self.status,
            "previous_status": self.previous_status,
            "weights_url": self.weights_url,
            "manifest": self.manifest,
            "metadata": self.metadata,
            "updated_at": round(self.updated_at, 3),
            "last_approval_id": self.last_approval_id,
            "last_notification_id": self.last_notification_id,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DeploymentRecord:
        r = DeploymentRecord(model_version=d["model_version"])
        r.status = d.get("status", "NOT_DEPLOYED")
        r.previous_status = d.get("previous_status", "")
        r.weights_url = d.get("weights_url", "")
        r.manifest = d.get("manifest", {})
        r.metadata = d.get("metadata", {})
        r.updated_at = d.get("updated_at", 0.0)
        r.last_approval_id = d.get("last_approval_id", "")
        r.last_notification_id = d.get("last_notification_id", "")
        return r


# ── DeploymentManager ──────────────────────────────────────────────────────────


class DeploymentManager:
    """Governs deployment stage transitions and Signalium notifications.

    Args:
        db: ``ResearchLabDatabase`` — state is persisted here.
        approval_gate: ``ApprovalGate`` — used to create and consume approval
            requests.  When ``None`` (tests), approval steps are skipped.
        signalium_notifier: Optional callable ``(payload: dict) → dict`` that
            sends the deployment notification to Signalium and returns the
            response.  When ``None``, the notification is skipped and a warning
            is logged.
    """

    def __init__(
        self,
        db: ResearchLabDatabase,
        approval_gate: Any | None = None,
        signalium_notifier: Callable[[dict], dict] | None = None,
        artifact_registry: Any | None = None,
        signer: ArtifactSigner | None = None,
    ) -> None:
        self._db = db
        self._gate = approval_gate
        self._notifier = signalium_notifier
        # S21A: when an ArtifactRegistry is supplied, every promotion into
        # SHADOW/CANDIDATE/ACTIVE verifies each artifact's signature before
        # the transition is allowed to proceed. Without a registry, signature
        # enforcement is skipped (e.g. deployments not yet integrated with
        # the artifact pipeline) — this mirrors `signalium_notifier=None`.
        self._registry = artifact_registry
        self._signer = signer or ArtifactSigner()

    # ── Promotion flow ─────────────────────────────────────────────────────────

    def request_promotion(
        self,
        model_version: str,
        target_status: str,
        requested_by: str,
        weights_url: str = "",
        manifest: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an ``ApprovalRequest`` for a stage transition.

        Args:
            model_version: QDE / model version string (e.g. ``"v2.1"``).
            target_status: Target ``DeploymentStatus`` string (e.g. ``"ACTIVE"``).
            requested_by: Username or service name making the request.
            weights_url: URL where the model weights can be downloaded by Signalium.
            manifest: Deployment manifest dict (architecture, feature pipeline, etc.).
            metadata: Arbitrary metadata to attach to the deployment record.
            context: Extra context attached to the approval request for reviewers.

        Returns:
            Dict with ``approval_request``, ``model_version``,
            ``from_status``, ``target_status``.

        Raises:
            ValueError: If the transition is invalid.
        """
        current = self._db.get_model_deployment(model_version)
        from_status = current["status"] if current else "NOT_DEPLOYED"
        gate_type = _gate_type_for(from_status, target_status)

        # Persist / update the deployment record with the new artifact refs
        rec = DeploymentRecord.from_dict(current) if current else DeploymentRecord(model_version)
        if weights_url:
            rec.weights_url = weights_url
        if manifest is not None:
            rec.manifest = manifest
        if metadata is not None:
            rec.metadata = metadata
        rec.updated_at = time.time()
        self._db.upsert_model_deployment(rec.to_dict())

        # Log PROMOTION_REQUESTED event
        evt = DeploymentEvent(
            model_version=model_version,
            event_type="PROMOTION_REQUESTED",
            from_status=from_status,
            to_status=target_status,
            actor=requested_by,
            payload={
                "gate_type": gate_type.value,
                "weights_url": weights_url,
                "manifest": manifest or {},
                "metadata": metadata or {},
                "context": context or {},
            },
        )
        self._db.insert_deployment_event(evt.to_dict())

        # Create ApprovalRequest if gate is configured
        approval_request: dict[str, Any] = {}
        if self._gate is not None:
            resource_id = f"{model_version}:{target_status}"
            req = self._gate.create_request(
                gate_type=gate_type,
                resource_id=resource_id,
                resource_type="model_version",
                requested_by=requested_by,
                action_summary=(
                    f"Promote {model_version!r} from {from_status!r} to {target_status!r}"
                ),
                requestor_context={
                    "model_version": model_version,
                    "from_status": from_status,
                    "target_status": target_status,
                    "weights_url": weights_url,
                    **(context or {}),
                },
            )
            approval_request = req.to_dict() if hasattr(req, "to_dict") else vars(req)

        return {
            "model_version": model_version,
            "from_status": from_status,
            "target_status": target_status,
            "gate_type": gate_type.value,
            "event_id": evt.event_id,
            "approval_request": approval_request,
        }

    def apply_promotion(
        self,
        request_id: str,
        approved_by: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """Approve an existing ApprovalRequest and execute the stage transition.

        Atomically:
        1. Approves the gate request (PENDING → APPROVED).
        2. Consumes it (APPROVED → CONSUMED).
        3. Transitions the model version's deployment status.
        4. If the target is ACTIVE: retires the previous ACTIVE model and
           sends a deployment notification to Signalium.
        5. Logs a permanent ``DeploymentEvent``.

        Args:
            request_id: The ``ApprovalRequest.request_id`` to approve.
            approved_by: Who approved the request.
            notes: Optional notes for the audit trail.

        Returns:
            Dict with ``model_version``, ``from_status``, ``to_status``,
            ``signalium_notified``, ``event_id``.
        """
        if self._gate is None:
            raise ValueError("No approval gate configured — cannot apply promotion.")

        # Retrieve request metadata before mutation
        gate_req = self._gate.get_request(request_id)
        if gate_req is None:
            raise ValueError(f"Approval request {request_id!r} not found.")

        resource_id = gate_req.resource_id
        gate_type = ApprovalGateType(gate_req.gate_type)

        # Derive model_version and target_status from resource_id
        parts = resource_id.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Unexpected resource_id format {resource_id!r} — expected 'version:status'"
            )
        model_version, target_status = parts[0], parts[1]

        # S21A: verify artifact/manifest signatures BEFORE touching the
        # approval gate — a blocked promotion must not burn the approval
        # (which would force a whole new approval cycle just to retry after
        # fixing/re-exporting the artifacts).
        if target_status in _SIGNATURE_GATED_STATUSES:
            self._enforce_signature_verification(
                model_version=model_version,
                target_status=target_status,
                actor=approved_by,
                approval_id=request_id,
            )

        # Step 1: Approve (PENDING → APPROVED)
        self._gate.approve(request_id, approved_by=approved_by, reason=notes)
        # Step 2: Require-approval atomically consumes (APPROVED → CONSUMED)
        consumed = self._gate.require_approval(gate_type, resource_id, consumed_by=approved_by)

        # Step 3: Load current record and execute transition
        current = self._db.get_model_deployment(model_version)
        from_status = current["status"] if current else "NOT_DEPLOYED"
        rec = DeploymentRecord.from_dict(current) if current else DeploymentRecord(model_version)
        rec.previous_status = from_status
        rec.status = target_status
        rec.last_approval_id = request_id
        rec.updated_at = time.time()

        # Step 4a: Single-ACTIVE enforcement — retire the previous ACTIVE
        retired_count = 0
        if target_status == "ACTIVE":
            retired_count = self._db.retire_active_deployments(model_version)
            if retired_count:
                logger.info(
                    f"DeploymentManager: retired {retired_count} previous ACTIVE "
                    f"model(s) before activating {model_version!r}"
                )

        self._db.upsert_model_deployment(rec.to_dict())

        # Step 4b: Signalium notification for ACTIVE promotions
        signalium_notified = False
        notification_event_id = ""
        if target_status == "ACTIVE":
            notif_payload = {
                "model_version": model_version,
                "weights_url": rec.weights_url,
                "manifest": rec.manifest,
                "metadata": rec.metadata,
                "deployed_at": rec.updated_at,
                "actor": approved_by,
                "approval_id": request_id,
            }
            notif_evt = self._send_signalium_notification(
                model_version=model_version,
                actor=approved_by,
                approval_id=request_id,
                payload=notif_payload,
            )
            signalium_notified = notif_evt.payload.get("success", False)
            notification_event_id = notif_evt.event_id
            rec.last_notification_id = notification_event_id
            self._db.upsert_model_deployment(rec.to_dict())

        # Step 5: Log PROMOTED event
        evt = DeploymentEvent(
            model_version=model_version,
            event_type="PROMOTED",
            from_status=from_status,
            to_status=target_status,
            actor=approved_by,
            approval_id=request_id,
            payload={
                "notes": notes,
                "retired_previous_count": retired_count,
                "signalium_notified": signalium_notified,
                "notification_event_id": notification_event_id,
            },
        )
        self._db.insert_deployment_event(evt.to_dict())
        logger.info(
            f"DeploymentManager: {model_version!r} promoted "
            f"{from_status!r} → {target_status!r} (approved_by={approved_by!r})"
        )

        return {
            "model_version": model_version,
            "from_status": from_status,
            "to_status": target_status,
            "approval_id": request_id,
            "signalium_notified": signalium_notified,
            "retired_previous_count": retired_count,
            "event_id": evt.event_id,
        }

    # ── Pre-SHADOW pipeline progression (S21A) ────────────────────────────────

    def advance_stage(
        self,
        model_version: str,
        target_status: str,
        actor: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """Advance a version through the pre-SHADOW pipeline stages.

        Unlike ``request_promotion``/``apply_promotion``, this covers the
        automatic DRAFT → TRAINING → VALIDATING progression a freshly trained
        version goes through before it is ever eligible for SHADOW — no human
        ``ApprovalGate`` is required for these internal pipeline stages.

        Args:
            model_version: QDE / model version string.
            target_status: One of DRAFT, TRAINING, VALIDATING.
            actor: Username or service advancing the stage.
            notes: Optional notes for the audit trail.

        Returns:
            Dict with ``model_version``, ``from_status``, ``to_status``, ``event_id``.

        Raises:
            ValueError: If the transition is not a legal pre-SHADOW step.
        """
        current = self._db.get_model_deployment(model_version)
        from_status = current["status"] if current else "NOT_DEPLOYED"

        if not _is_valid_pre_shadow_transition(from_status, target_status):
            raise ValueError(
                f"Invalid stage transition: {from_status!r} → {target_status!r}. "
                f"Valid pre-SHADOW transitions: {sorted(_PRE_SHADOW_TRANSITIONS)}"
            )

        rec = DeploymentRecord.from_dict(current) if current else DeploymentRecord(model_version)
        rec.previous_status = from_status
        rec.status = target_status
        rec.updated_at = time.time()
        self._db.upsert_model_deployment(rec.to_dict())

        evt = DeploymentEvent(
            model_version=model_version,
            event_type="STAGE_ADVANCED",
            from_status=from_status,
            to_status=target_status,
            actor=actor,
            payload={"notes": notes},
        )
        self._db.insert_deployment_event(evt.to_dict())
        logger.info(
            f"DeploymentManager: {model_version!r} advanced "
            f"{from_status!r} → {target_status!r} (actor={actor!r})"
        )

        return {
            "model_version": model_version,
            "from_status": from_status,
            "to_status": target_status,
            "event_id": evt.event_id,
        }

    # ── Rollback flow ──────────────────────────────────────────────────────────

    def request_rollback(
        self,
        model_version: str,
        reason: str,
        requested_by: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a ``ROLLBACK`` ``ApprovalRequest`` for the given model version.

        The model_version here is the version that should be rolled back TO
        (i.e. the most recent RETIRED version to restore).  If model_version
        is empty or ``"latest_retired"``, the manager resolves the most recent
        RETIRED version automatically.

        Returns:
            Dict with ``model_version`` (resolved), ``approval_request``.
        """
        # Resolve the target model_version if not explicit
        if not model_version or model_version == "latest_retired":
            retired = self._db.list_model_deployments(status="RETIRED", limit=1)
            if not retired:
                raise ValueError("No RETIRED model version found to roll back to.")
            model_version = retired[0]["model_version"]

        target_rec = self._db.get_model_deployment(model_version)
        if target_rec is None:
            raise ValueError(
                f"Model version {model_version!r} does not exist in the deployment registry. "
                "Only RETIRED versions that were previously deployed can be rolled back to."
            )
        if target_rec.get("status") not in ("RETIRED", "ROLLED_BACK"):
            raise ValueError(
                f"Model version {model_version!r} is in status "
                f"{target_rec['status']!r} — only RETIRED versions can be rolled back to."
            )

        # Log ROLLBACK_REQUESTED event
        evt = DeploymentEvent(
            model_version=model_version,
            event_type="ROLLBACK_REQUESTED",
            from_status=target_rec["status"] if target_rec else "RETIRED",
            to_status="ACTIVE",
            actor=requested_by,
            payload={"reason": reason, "context": context or {}},
        )
        self._db.insert_deployment_event(evt.to_dict())

        approval_request: dict[str, Any] = {}
        if self._gate is not None:
            req = self._gate.create_request(
                gate_type=ApprovalGateType.ROLLBACK,
                resource_id=model_version,
                resource_type="model_version",
                requested_by=requested_by,
                action_summary=(
                    f"Rollback: restore {model_version!r} to ACTIVE "
                    f"(reason: {reason!r})"
                ),
                requestor_context={
                    "model_version": model_version,
                    "reason": reason,
                    **(context or {}),
                },
            )
            approval_request = req.to_dict() if hasattr(req, "to_dict") else vars(req)

        return {
            "model_version": model_version,
            "event_id": evt.event_id,
            "approval_request": approval_request,
        }

    def apply_rollback(
        self,
        request_id: str,
        approved_by: str,
    ) -> dict[str, Any]:
        """Approve a ROLLBACK request and restore the target version to ACTIVE.

        Atomically:
        1. Approves and consumes the ROLLBACK gate.
        2. Retires the current ACTIVE model (if any).
        3. Restores the target RETIRED version to ACTIVE.
        4. Sends a Signalium deployment notification with the restored version's
           artifact references.
        5. Logs permanent ``DeploymentEvent`` records for the rollback.

        Returns:
            Dict with ``model_version``, ``signalium_notified``, ``event_id``.
        """
        if self._gate is None:
            raise ValueError("No approval gate configured — cannot apply rollback.")

        gate_req = self._gate.get_request(request_id)
        if gate_req is None:
            raise ValueError(f"Approval request {request_id!r} not found.")

        model_version = gate_req.resource_id

        # S21A: verify signatures for the restored version BEFORE consuming
        # the ROLLBACK approval — restoring to ACTIVE is a promotion in all
        # but name and must be held to the same signing bar.
        self._enforce_signature_verification(
            model_version=model_version,
            target_status="ACTIVE",
            actor=approved_by,
            approval_id=request_id,
        )

        # Approve (PENDING → APPROVED) then atomically consume (APPROVED → CONSUMED)
        self._gate.approve(request_id, approved_by=approved_by, reason="Rollback approved")
        self._gate.require_approval(
            ApprovalGateType.ROLLBACK, model_version, consumed_by=approved_by
        )

        # Re-validate target is eligible BEFORE mutating any active state
        target = self._db.get_model_deployment(model_version)
        if target is None:
            raise ValueError(
                f"Model version {model_version!r} does not exist in the deployment registry. "
                "Rollback aborted — no active state was changed."
            )
        if target.get("status") not in ("RETIRED", "ROLLED_BACK"):
            raise ValueError(
                f"Model version {model_version!r} is in status {target['status']!r} — "
                "only RETIRED versions can be restored. Rollback aborted."
            )

        # Retire current ACTIVE (only after eligibility confirmed)
        retired_current = self._db.retire_active_deployments(model_version)

        # Restore target to ACTIVE
        from_status = target["status"]
        rec = DeploymentRecord.from_dict(target)
        rec.previous_status = from_status
        rec.status = "ACTIVE"
        rec.last_approval_id = request_id
        rec.updated_at = time.time()
        self._db.upsert_model_deployment(rec.to_dict())

        # Signalium notification with the restored version's artifact refs
        notif_payload = {
            "model_version": model_version,
            "weights_url": rec.weights_url,
            "manifest": rec.manifest,
            "metadata": rec.metadata,
            "deployed_at": rec.updated_at,
            "actor": approved_by,
            "approval_id": request_id,
            "is_rollback": True,
        }
        notif_evt = self._send_signalium_notification(
            model_version=model_version,
            actor=approved_by,
            approval_id=request_id,
            payload=notif_payload,
        )
        signalium_notified = notif_evt.payload.get("success", False)

        # Log ROLLED_BACK event
        evt = DeploymentEvent(
            model_version=model_version,
            event_type="ROLLED_BACK",
            from_status=from_status,
            to_status="ACTIVE",
            actor=approved_by,
            approval_id=request_id,
            payload={
                "retired_current_count": retired_current,
                "signalium_notified": signalium_notified,
                "notification_event_id": notif_evt.event_id,
            },
        )
        self._db.insert_deployment_event(evt.to_dict())
        logger.info(
            f"DeploymentManager: rollback — {model_version!r} restored to ACTIVE "
            f"(approved_by={approved_by!r}, signalium_notified={signalium_notified})"
        )

        return {
            "model_version": model_version,
            "from_status": from_status,
            "to_status": "ACTIVE",
            "approval_id": request_id,
            "signalium_notified": signalium_notified,
            "retired_current_count": retired_current,
            "event_id": evt.event_id,
        }

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_current(self) -> dict[str, Any] | None:
        """Return the current ACTIVE model deployment record, or None."""
        rows = self._db.list_model_deployments(status="ACTIVE", limit=1)
        return rows[0] if rows else None

    def get_history(
        self,
        model_version: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the full deployment event log, optionally filtered."""
        return self._db.list_deployment_events(
            model_version=model_version,
            event_type=event_type,
            limit=limit,
        )

    def get_deployment_state(self, model_version: str) -> dict[str, Any] | None:
        """Return the current deployment state for a specific model version."""
        return self._db.get_model_deployment(model_version)

    # ── Signature verification (S21A) ───────────────────────────────────────

    def _enforce_signature_verification(
        self,
        model_version: str,
        target_status: str,
        actor: str,
        approval_id: str,
    ) -> None:
        """Verify every artifact/manifest for *model_version*, raising and
        auditing a ``SIGNATURE_VERIFICATION_FAILED`` event if any is missing
        or invalid.

        No-op when no ``ArtifactRegistry`` was supplied to this manager.
        """
        if self._registry is None:
            return

        results = self.verify_artifact_signatures(model_version)
        failures = [r for r in results if not r.verified]

        if not results:
            failures = [
                SignatureResult(
                    name="<no artifacts registered>",
                    verified=False,
                    reason=f"No registered artifacts found for model_version {model_version!r}",
                )
            ]

        if failures:
            evt = DeploymentEvent(
                model_version=model_version,
                event_type="SIGNATURE_VERIFICATION_FAILED",
                from_status="",
                to_status=target_status,
                actor=actor,
                approval_id=approval_id,
                payload={"failures": [f.to_dict() for f in failures]},
            )
            self._db.insert_deployment_event(evt.to_dict())
            reasons = "; ".join(f"{f.name}: {f.reason}" for f in failures)
            logger.error(
                f"DeploymentManager: promotion of {model_version!r} to "
                f"{target_status!r} BLOCKED — signature verification failed: {reasons}"
            )
            raise PromotionBlockedError(
                f"Promotion of {model_version!r} to {target_status!r} blocked — "
                f"signature verification failed for: {reasons}"
            )

    def verify_artifact_signatures(self, model_version: str) -> list[SignatureResult]:
        """Verify the integrity hash + signature of every registered artifact
        for *model_version*.

        Returns:
            One ``SignatureResult`` per registered artifact (empty list if
            none are registered, or if no ``ArtifactRegistry`` is configured).
        """
        if self._registry is None:
            return []

        results: list[SignatureResult] = []
        artifacts = self._registry.list_artifacts(model_version=model_version, limit=10_000)
        for art in artifacts:
            results.append(self._verify_one_artifact(art))
        return results

    def _verify_one_artifact(self, art: dict[str, Any]) -> SignatureResult:
        name = art.get("name", art.get("artifact_id", "<unknown>"))
        path_str = art.get("path", "")
        if not path_str:
            return SignatureResult(name=name, verified=False, reason="no stored path on record")

        from pathlib import Path

        path = Path(path_str)
        if not path.exists():
            return SignatureResult(name=name, verified=False, reason=f"artifact file missing: {path}")

        data = path.read_bytes()

        # Integrity: the file's live bytes must still match the hash recorded
        # at registration time.
        import hashlib

        actual_hash = hashlib.sha256(data).hexdigest()
        expected_hash = art.get("artifact_hash", "")
        if actual_hash != expected_hash:
            return SignatureResult(
                name=name, verified=False, reason="artifact hash mismatch — file has been modified"
            )

        sig_path = Path(f"{path_str}.sig")
        if not sig_path.exists():
            return SignatureResult(name=name, verified=False, reason="missing signature file")

        signature_hex = sig_path.read_text(encoding="utf-8").strip()
        if not self._signer.can_verify:
            return SignatureResult(
                name=name,
                verified=False,
                reason="no verification public key configured (ARTIFACT_SIGNING_PUBLIC_KEY)",
            )
        try:
            ok = self._signer.verify(data, signature_hex)
        except SigningKeyError as exc:
            return SignatureResult(name=name, verified=False, reason=str(exc))
        if not ok:
            return SignatureResult(name=name, verified=False, reason="invalid signature")
        return SignatureResult(name=name, verified=True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _send_signalium_notification(
        self,
        model_version: str,
        actor: str,
        approval_id: str,
        payload: dict[str, Any],
    ) -> DeploymentEvent:
        """Send a deployment notification to Signalium and log the result."""
        success = False
        error_msg = ""
        response: dict[str, Any] = {}

        if self._notifier is not None:
            try:
                response = self._notifier(payload) or {}
                success = True
                logger.info(
                    f"DeploymentManager: Signalium notification sent for "
                    f"{model_version!r} — response={response}"
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = repr(exc)
                logger.warning(
                    f"DeploymentManager: Signalium notification failed for "
                    f"{model_version!r}: {exc!r}"
                )
        else:
            logger.warning(
                f"DeploymentManager: no Signalium notifier configured — "
                f"skipping notification for {model_version!r}"
            )

        evt = DeploymentEvent(
            model_version=model_version,
            event_type="SIGNALIUM_NOTIFIED",
            from_status="",
            to_status="",
            actor=actor,
            approval_id=approval_id,
            payload={
                "notification_payload": payload,
                "success": success,
                "response": response,
                "error": error_msg,
            },
        )
        self._db.insert_deployment_event(evt.to_dict())
        return evt
