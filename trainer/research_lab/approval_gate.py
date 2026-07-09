"""S20 — Human Approval Gate enforcement layer.

ApprovalGate is the sole interface for:
  1. Creating approval requests (research lab or downstream systems call create_request)
  2. Admin resolution (approve / reject)
  3. Gate enforcement (require_approval — called before executing any gated action)
  4. Expiry management (expire_stale is called on each list/check to auto-expire PENDING)

Atomic-consume guarantee
------------------------
`require_approval()` calls `consume()` on the matching ApprovalRequest and
writes the CONSUMED status in a single DB transaction using an optimistic-lock
UPDATE WHERE status='APPROVED'. If the row was already consumed (e.g. by a
concurrent caller), the DB rowcount returns 0 and ApprovalRequiredError is
raised — the action is blocked.

Usage example (in a training pipeline)::

    gate = ApprovalGate(db)
    try:
        req = gate.require_approval(
            ApprovalGateType.TRAINING_START,
            resource_id="exp-abc123",
        )
        # Only reaches here if approval was found and atomically consumed
        _start_training("exp-abc123")
    except ApprovalRequiredError as e:
        return {"error": e.detail}
"""

from __future__ import annotations

import time
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.approval_models import (
    DEFAULT_TTL_SECONDS,
    ApprovalAuditEntry,
    ApprovalAlreadyResolved,
    ApprovalGateType,
    ApprovalRequest,
    ApprovalRequiredError,
    ApprovalStatus,
)
from trainer.research_lab.rl_database import ResearchLabDatabase


def _from_dict(d: dict[str, Any]) -> ApprovalRequest:
    """Reconstruct an ApprovalRequest from a persisted dict."""
    return ApprovalRequest(
        request_id=d["request_id"],
        gate_type=d.get("gate_type", ""),
        resource_id=d.get("resource_id", ""),
        resource_type=d.get("resource_type", ""),
        requested_by=d.get("requested_by", ""),
        action_summary=d.get("action_summary", ""),
        requestor_context=d.get("requestor_context", {}),
        status=d.get("status", ApprovalStatus.PENDING.value),
        created_at=d.get("created_at", 0.0),
        expires_at=d.get("expires_at", 0.0),
        updated_at=d.get("updated_at", 0.0),
        resolved_by=d.get("resolved_by", ""),
        resolved_at=d.get("resolved_at"),
        resolution_reason=d.get("resolution_reason", ""),
        consumed_at=d.get("consumed_at"),
        consumed_by=d.get("consumed_by", ""),
        audit_log=d.get("audit_log", []),
    )


class ApprovalGate:
    """Human approval gate for Research Lab consequential actions.

    Thread-safe through the underlying SQLite WAL-mode ResearchLabDatabase.
    Atomic consume uses a single UPDATE WHERE status='APPROVED' + rowcount check.
    """

    def __init__(
        self, db: ResearchLabDatabase, default_ttl: float = DEFAULT_TTL_SECONDS
    ) -> None:
        self._db = db
        self._default_ttl = default_ttl

    # ── Request creation ──────────────────────────────────────────────────────

    def create_request(
        self,
        gate_type: ApprovalGateType,
        resource_id: str,
        resource_type: str,
        requested_by: str,
        action_summary: str,
        requestor_context: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> ApprovalRequest:
        """Create a new approval request in PENDING status.

        Idempotent at the gate_type+resource_id level: if an open (PENDING)
        request already exists for the same gate/resource, return it instead
        of creating a duplicate.
        """
        # Check for an existing open request
        existing = self._db.list_approval_requests(
            gate_type=gate_type.value,
            resource_id=resource_id,
            status=ApprovalStatus.PENDING.value,
        )
        # Expire any stale ones first, then filter to live ones
        live = []
        for row in existing:
            req = _from_dict(row)
            if req.is_expired():
                req._expire()
                self._db.upsert_approval_request(req.to_dict())
            else:
                live.append(req)

        if live:
            logger.debug(
                f"ApprovalGate: returning existing PENDING request "
                f"{live[0].request_id} for gate={gate_type.value} resource={resource_id!r}"
            )
            return live[0]

        req = ApprovalRequest.new(
            gate_type=gate_type,
            resource_id=resource_id,
            resource_type=resource_type,
            requested_by=requested_by,
            action_summary=action_summary,
            requestor_context=requestor_context,
            ttl_seconds=ttl_seconds or self._default_ttl,
        )
        self._db.upsert_approval_request(req.to_dict())

        # Re-read from the DB: if a concurrent caller raced to create the same
        # (gate_type, resource_id) PENDING request, INSERT OR IGNORE silently
        # dropped our row and persisted theirs instead.  Always return the
        # canonical persisted row so callers never hold a phantom request_id.
        persisted = self._db.list_approval_requests(
            gate_type=gate_type.value,
            resource_id=resource_id,
            status=ApprovalStatus.PENDING.value,
        )
        # Expire stale stragglers before selecting the canonical live row
        canonical = None
        for row in persisted:
            obj = _from_dict(row)
            if obj.is_expired():
                obj._expire()
                self._db.upsert_approval_request(obj.to_dict())
            else:
                canonical = obj
                break

        if canonical is None:
            # Should not happen under normal operation; log and return the
            # in-memory object so callers still get a sensible response.
            logger.warning(
                f"ApprovalGate: no PENDING row found after insert for "
                f"gate={gate_type.value} resource={resource_id!r} — "
                "returning in-memory object (check DB constraints)"
            )
            return req

        logger.info(
            f"ApprovalGate: created {canonical.request_id} gate={gate_type.value} "
            f"resource={resource_id!r} requested_by={requested_by!r}"
        )
        return canonical

    # ── Admin resolution ──────────────────────────────────────────────────────

    def approve(self, request_id: str, approved_by: str, reason: str = "") -> ApprovalRequest:
        """Admin approves a PENDING request → APPROVED."""
        req = self._load_or_raise(request_id)
        req.approve(approved_by=approved_by, reason=reason)
        self._db.upsert_approval_request(req.to_dict())
        logger.info(
            f"ApprovalGate: approved {request_id} by {approved_by!r} "
            f"gate={req.gate_type} resource={req.resource_id!r}"
        )
        return req

    def reject(self, request_id: str, rejected_by: str, reason: str) -> ApprovalRequest:
        """Admin rejects a PENDING request → REJECTED."""
        req = self._load_or_raise(request_id)
        req.reject(rejected_by=rejected_by, reason=reason)
        self._db.upsert_approval_request(req.to_dict())
        logger.info(
            f"ApprovalGate: rejected {request_id} by {rejected_by!r} — {reason!r}"
        )
        return req

    def expire_request(self, request_id: str) -> ApprovalRequest:
        """Manually expire a PENDING request (admin / cleanup)."""
        req = self._load_or_raise(request_id)
        if req.status != ApprovalStatus.PENDING.value:
            raise ApprovalAlreadyResolved(request_id, req.status)
        req._expire()
        self._db.upsert_approval_request(req.to_dict())
        logger.info(f"ApprovalGate: manually expired {request_id}")
        return req

    # ── Gate enforcement ──────────────────────────────────────────────────────

    def require_approval(
        self,
        gate_type: ApprovalGateType,
        resource_id: str,
        consumed_by: str = "system",
    ) -> ApprovalRequest:
        """Check for a valid unconsumed approval and atomically consume it.

        This is the core gate enforcement call. Downstream systems call this
        immediately before executing any gated action.

        Raises:
            ApprovalRequiredError: If no valid (APPROVED, not CONSUMED or EXPIRED)
                approval exists for this gate_type + resource_id combination.

        Returns:
            The consumed ApprovalRequest (status=CONSUMED) on success.
        """
        # Expire stale PENDING requests first (keeps the DB tidy)
        self._expire_stale(gate_type=gate_type, resource_id=resource_id)

        # Atomically consume an APPROVED request in the DB
        # This UPDATE only succeeds if exactly one row has status=APPROVED
        consumed_at = time.time()
        rows_updated = self._db.atomic_consume_approval(
            gate_type=gate_type.value,
            resource_id=resource_id,
            consumed_by=consumed_by,
            consumed_at=consumed_at,
        )

        if rows_updated == 0:
            # Check why: no request at all, or wrong status?
            all_reqs = self._db.list_approval_requests(
                gate_type=gate_type.value, resource_id=resource_id
            )
            if not all_reqs:
                reason = "No approval request exists"
            else:
                statuses = [r.get("status") for r in all_reqs]
                reason = f"Existing requests are in status {statuses} — none are APPROVED"
            raise ApprovalRequiredError(
                gate_type=gate_type.value,
                resource_id=resource_id,
                reason=reason,
            )

        # Fetch the now-CONSUMED record to return it
        consumed_reqs = self._db.list_approval_requests(
            gate_type=gate_type.value,
            resource_id=resource_id,
            status=ApprovalStatus.CONSUMED.value,
        )
        # Most recently consumed is the one we just consumed
        consumed_reqs.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
        result = _from_dict(consumed_reqs[0])

        logger.info(
            f"ApprovalGate: consumed {result.request_id} gate={gate_type.value} "
            f"resource={resource_id!r} consumed_by={consumed_by!r}"
        )
        return result

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Fetch a specific approval request by ID."""
        row = self._db.get_approval_request(request_id)
        return _from_dict(row) if row else None

    def list_requests(
        self,
        gate_type: str | None = None,
        resource_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List approval requests with optional filters.

        Before returning, auto-expires any PENDING requests past their TTL.
        """
        if status is None or status == ApprovalStatus.PENDING.value:
            self._expire_stale()
        return self._db.list_approval_requests(
            gate_type=gate_type, resource_id=resource_id, status=status, limit=limit
        )

    def audit_log(
        self, resource_id: str | None = None, gate_type: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Return all approval decisions as a permanent audit trail."""
        rows = self._db.list_approval_requests(
            gate_type=gate_type, resource_id=resource_id, limit=limit
        )
        # Flatten individual audit_log entries for easy consumption
        entries = []
        for row in rows:
            for entry in row.get("audit_log", []):
                entries.append({
                    "request_id": row["request_id"],
                    "gate_type": row["gate_type"],
                    "resource_id": row["resource_id"],
                    **entry,
                })
        entries.sort(key=lambda e: e.get("timestamp", 0))
        return entries

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_or_raise(self, request_id: str) -> ApprovalRequest:
        row = self._db.get_approval_request(request_id)
        if row is None:
            raise ValueError(f"Approval request {request_id!r} not found")
        return _from_dict(row)

    def _expire_stale(
        self, gate_type: ApprovalGateType | None = None, resource_id: str | None = None
    ) -> int:
        """Expire all PENDING requests past their TTL. Returns count expired."""
        pending = self._db.list_approval_requests(
            gate_type=gate_type.value if gate_type else None,
            resource_id=resource_id,
            status=ApprovalStatus.PENDING.value,
            limit=1000,
        )
        expired = 0
        for row in pending:
            req = _from_dict(row)
            if req.is_expired():
                req._expire()
                self._db.upsert_approval_request(req.to_dict())
                expired += 1
        return expired
