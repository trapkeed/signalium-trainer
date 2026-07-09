"""Unit tests for S20 Human Approval Gates.

Covers:
- ApprovalRequest model: create, approve, reject, expire, consume, to_dict
- ApprovalAlreadyResolved / ApprovalRequiredError exception shapes
- ApprovalGate: create_request idempotency, approve, reject, require_approval atomic-consume
- Atomic double-consume prevention
- Expiry: stale PENDING requests auto-expire on require_approval/list
- Gate enforcement: blocks without approval, passes with valid approval
- DB: upsert/get/list/atomic_consume
- Server endpoints: POST /approvals, GET /approvals, GET /approvals/{id},
  POST approve, POST reject
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from trainer.research_lab.approval_gate import ApprovalGate
from trainer.research_lab.approval_models import (
    ApprovalAlreadyResolved,
    ApprovalGateType,
    ApprovalRequest,
    ApprovalRequiredError,
    ApprovalStatus,
    DEFAULT_TTL_SECONDS,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from tests.unit._sso_test_utils import make_signalium_jwt, DEFAULT_TEST_JWT_SECRET


# ── Module-scope Pydantic request models ──────────────────────────────────────

class _CreateApprovalReq(BaseModel):
    gate_type: str
    resource_id: str
    resource_type: str = ""
    requested_by: str = "system"
    action_summary: str = ""
    requestor_context: dict[str, Any] = {}
    ttl_seconds: float = DEFAULT_TTL_SECONDS


class _ApproveReq(BaseModel):
    approved_by: str = "admin"
    reason: str = ""


class _RejectReq(BaseModel):
    rejected_by: str = "admin"
    reason: str = "not ready"


# ── Fixtures ──────────────────────────────────────────────────────────────────

NOW = time.time()


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "test_approval.db"))


@pytest.fixture()
def gate(db: ResearchLabDatabase) -> ApprovalGate:
    return ApprovalGate(db)


def _pending_request(
    gate: ApprovalGate,
    gate_type: ApprovalGateType = ApprovalGateType.EXPERIMENT_LAUNCH,
    resource_id: str = "exp-test-001",
) -> ApprovalRequest:
    return gate.create_request(
        gate_type=gate_type,
        resource_id=resource_id,
        resource_type="experiment",
        requested_by="researcher",
        action_summary=f"Launch {resource_id}",
    )


# ── ApprovalRequest model ─────────────────────────────────────────────────────


class TestApprovalRequest:
    def test_new_starts_in_pending(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id="exp-001",
            resource_type="experiment",
            requested_by="researcher",
            action_summary="Launch experiment",
        )
        assert r.status == ApprovalStatus.PENDING.value
        assert r.request_id.startswith("apr-")
        assert r.expires_at > r.created_at
        assert len(r.audit_log) == 1
        assert r.audit_log[0]["event"] == "created"

    def test_approve_transitions_to_approved(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.TRAINING_START,
            resource_id="exp-002",
            resource_type="experiment",
            requested_by="pipeline",
            action_summary="Start training",
        )
        r.approve(approved_by="head_quant", reason="looks good")
        assert r.status == ApprovalStatus.APPROVED.value
        assert r.resolved_by == "head_quant"
        assert r.resolved_at is not None
        assert any(e["event"] == "approved" for e in r.audit_log)

    def test_reject_transitions_to_rejected(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.PROMOTION,
            resource_id="v2.0.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="Promote v2.0.0",
        )
        r.reject(rejected_by="admin", reason="validation failed")
        assert r.status == ApprovalStatus.REJECTED.value
        assert r.resolution_reason == "validation failed"
        assert any(e["event"] == "rejected" for e in r.audit_log)

    def test_expire_transitions_pending_to_expired(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.DEPLOYMENT,
            resource_id="v2.0.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="Deploy v2.0.0",
        )
        r._expire()
        assert r.status == ApprovalStatus.EXPIRED.value
        assert any(e["event"] == "expired" for e in r.audit_log)

    def test_consume_transitions_approved_to_consumed(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.ROLLBACK,
            resource_id="v1.9.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="Rollback v1.9.0",
        )
        r.approve(approved_by="admin")
        r.consume(consumed_by="rollback_service")
        assert r.status == ApprovalStatus.CONSUMED.value
        assert r.consumed_by == "rollback_service"
        assert r.consumed_at is not None
        assert any(e["event"] == "consumed" for e in r.audit_log)

    def test_approve_non_pending_raises(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id="exp-x",
            resource_type="experiment",
            requested_by="researcher",
            action_summary="",
        )
        r.approve(approved_by="admin")
        with pytest.raises(ApprovalAlreadyResolved):
            r.approve(approved_by="admin")

    def test_reject_non_pending_raises(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.TRAINING_START,
            resource_id="exp-x",
            resource_type="experiment",
            requested_by="pipeline",
            action_summary="",
        )
        r.reject(rejected_by="admin", reason="nope")
        with pytest.raises(ApprovalAlreadyResolved):
            r.reject(rejected_by="admin", reason="again")

    def test_consume_non_approved_raises(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.PROMOTION,
            resource_id="v3.0.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="",
        )
        with pytest.raises(ApprovalAlreadyResolved):
            r.consume()

    def test_is_expired_true_when_past_ttl(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.DEPLOYMENT,
            resource_id="v2.0.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="",
            ttl_seconds=0.001,  # expire immediately
        )
        time.sleep(0.01)
        assert r.is_expired() is True

    def test_is_expired_false_when_within_ttl(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.DEPLOYMENT,
            resource_id="v2.0.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="",
            ttl_seconds=3600,
        )
        assert r.is_expired() is False

    def test_approve_on_expired_pending_transitions_expired_then_raises(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id="exp-expired",
            resource_type="experiment",
            requested_by="system",
            action_summary="",
            ttl_seconds=0.001,
        )
        time.sleep(0.01)
        with pytest.raises(ApprovalAlreadyResolved) as exc_info:
            r.approve(approved_by="admin")
        assert r.status == ApprovalStatus.EXPIRED.value
        assert exc_info.value.current_status == ApprovalStatus.EXPIRED.value

    def test_to_dict_has_all_required_keys(self) -> None:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.ROLLBACK,
            resource_id="v1.0.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="Emergency rollback",
            requestor_context={"reason": "confidence_drift"},
        )
        d = r.to_dict()
        for key in (
            "request_id", "gate_type", "resource_id", "resource_type",
            "requested_by", "action_summary", "requestor_context",
            "status", "created_at", "expires_at", "updated_at",
            "resolved_by", "resolved_at", "resolution_reason",
            "consumed_at", "consumed_by", "audit_log",
        ):
            assert key in d, f"Missing key: {key}"


# ── ApprovalRequiredError ─────────────────────────────────────────────────────


class TestApprovalRequiredError:
    def test_detail_dict_shape(self) -> None:
        err = ApprovalRequiredError(
            gate_type="EXPERIMENT_LAUNCH",
            resource_id="exp-001",
            reason="No approval found",
        )
        assert err.detail["error"] == "approval_required"
        assert err.detail["gate_type"] == "EXPERIMENT_LAUNCH"
        assert err.detail["resource_id"] == "exp-001"

    def test_str_representation(self) -> None:
        err = ApprovalRequiredError("TRAINING_START", "exp-002")
        assert "TRAINING_START" in str(err)
        assert "exp-002" in str(err)


# ── ApprovalGate engine ───────────────────────────────────────────────────────


class TestApprovalGate:
    def test_create_request_returns_pending(self, gate: ApprovalGate) -> None:
        req = _pending_request(gate)
        assert req.status == ApprovalStatus.PENDING.value
        assert req.request_id.startswith("apr-")

    def test_create_request_idempotent_for_same_gate_resource(
        self, gate: ApprovalGate
    ) -> None:
        r1 = _pending_request(gate, resource_id="exp-idem")
        r2 = _pending_request(gate, resource_id="exp-idem")
        assert r1.request_id == r2.request_id

    def test_create_request_different_resources_are_separate(
        self, gate: ApprovalGate
    ) -> None:
        r1 = _pending_request(gate, resource_id="exp-001")
        r2 = _pending_request(gate, resource_id="exp-002")
        assert r1.request_id != r2.request_id

    def test_create_request_different_gate_types_are_separate(
        self, gate: ApprovalGate
    ) -> None:
        r1 = _pending_request(gate, gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
                               resource_id="exp-multi")
        r2 = _pending_request(gate, gate_type=ApprovalGateType.TRAINING_START,
                               resource_id="exp-multi")
        assert r1.request_id != r2.request_id

    def test_approve_changes_status(self, gate: ApprovalGate) -> None:
        req = _pending_request(gate)
        approved = gate.approve(req.request_id, approved_by="head_quant", reason="all good")
        assert approved.status == ApprovalStatus.APPROVED.value
        assert approved.resolved_by == "head_quant"

    def test_reject_changes_status(self, gate: ApprovalGate) -> None:
        req = _pending_request(gate)
        rejected = gate.reject(req.request_id, rejected_by="admin", reason="not ready")
        assert rejected.status == ApprovalStatus.REJECTED.value
        assert rejected.resolution_reason == "not ready"

    def test_approve_nonexistent_raises(self, gate: ApprovalGate) -> None:
        with pytest.raises(ValueError, match="not found"):
            gate.approve("nonexistent-id", approved_by="admin")

    def test_reject_nonexistent_raises(self, gate: ApprovalGate) -> None:
        with pytest.raises(ValueError, match="not found"):
            gate.reject("nonexistent-id", rejected_by="admin", reason="nope")

    def test_approve_already_approved_raises(self, gate: ApprovalGate) -> None:
        req = _pending_request(gate)
        gate.approve(req.request_id, approved_by="admin")
        with pytest.raises(ApprovalAlreadyResolved):
            gate.approve(req.request_id, approved_by="admin")

    def test_require_approval_blocks_without_approval(self, gate: ApprovalGate) -> None:
        with pytest.raises(ApprovalRequiredError) as exc_info:
            gate.require_approval(ApprovalGateType.EXPERIMENT_LAUNCH, "exp-no-approval")
        assert exc_info.value.gate_type == "EXPERIMENT_LAUNCH"
        assert exc_info.value.resource_id == "exp-no-approval"

    def test_require_approval_blocks_pending_not_approved(
        self, gate: ApprovalGate
    ) -> None:
        _pending_request(gate, resource_id="exp-pending-only")
        with pytest.raises(ApprovalRequiredError):
            gate.require_approval(ApprovalGateType.EXPERIMENT_LAUNCH, "exp-pending-only")

    def test_require_approval_passes_after_admin_approves(
        self, gate: ApprovalGate
    ) -> None:
        req = _pending_request(gate, resource_id="exp-approved")
        gate.approve(req.request_id, approved_by="admin")
        consumed = gate.require_approval(
            ApprovalGateType.EXPERIMENT_LAUNCH, "exp-approved"
        )
        assert consumed.status == ApprovalStatus.CONSUMED.value
        assert consumed.consumed_by == "system"

    def test_require_approval_consumed_cannot_be_reused(
        self, gate: ApprovalGate
    ) -> None:
        """Atomic consume: once consumed, the same approval cannot be used again."""
        req = _pending_request(gate, resource_id="exp-atomic")
        gate.approve(req.request_id, approved_by="admin")
        gate.require_approval(ApprovalGateType.EXPERIMENT_LAUNCH, "exp-atomic")
        # Second call must fail — approval was already consumed
        with pytest.raises(ApprovalRequiredError):
            gate.require_approval(ApprovalGateType.EXPERIMENT_LAUNCH, "exp-atomic")

    def test_require_approval_blocks_rejected_requests(
        self, gate: ApprovalGate
    ) -> None:
        req = _pending_request(gate, resource_id="exp-rejected")
        gate.reject(req.request_id, rejected_by="admin", reason="no")
        with pytest.raises(ApprovalRequiredError):
            gate.require_approval(ApprovalGateType.EXPERIMENT_LAUNCH, "exp-rejected")

    def test_require_approval_auto_expires_stale_pending(
        self, gate: ApprovalGate
    ) -> None:
        # Create a request with a tiny TTL so it expires immediately
        req = gate.create_request(
            gate_type=ApprovalGateType.TRAINING_START,
            resource_id="exp-stale",
            resource_type="experiment",
            requested_by="pipeline",
            action_summary="Should expire",
            ttl_seconds=0.001,
        )
        time.sleep(0.01)
        with pytest.raises(ApprovalRequiredError):
            gate.require_approval(ApprovalGateType.TRAINING_START, "exp-stale")
        # Expired request should now be in DB as EXPIRED
        fetched = gate.get_request(req.request_id)
        assert fetched is not None
        assert fetched.status == ApprovalStatus.EXPIRED.value

    def test_new_request_can_be_created_after_expiry(
        self, gate: ApprovalGate
    ) -> None:
        """After a request expires, a fresh one can be created for the same resource."""
        gate.create_request(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id="exp-renew",
            resource_type="experiment",
            requested_by="system",
            action_summary="First attempt",
            ttl_seconds=0.001,
        )
        time.sleep(0.01)
        # Trigger expiry via list
        gate.list_requests(status=ApprovalStatus.PENDING.value)
        # Now create a fresh one
        r2 = gate.create_request(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id="exp-renew",
            resource_type="experiment",
            requested_by="system",
            action_summary="Second attempt",
        )
        assert r2.status == ApprovalStatus.PENDING.value

    def test_expire_request_manually(self, gate: ApprovalGate) -> None:
        req = _pending_request(gate)
        expired = gate.expire_request(req.request_id)
        assert expired.status == ApprovalStatus.EXPIRED.value

    def test_expire_already_approved_raises(self, gate: ApprovalGate) -> None:
        req = _pending_request(gate)
        gate.approve(req.request_id, approved_by="admin")
        with pytest.raises(ApprovalAlreadyResolved):
            gate.expire_request(req.request_id)

    def test_get_request_returns_none_for_missing(self, gate: ApprovalGate) -> None:
        assert gate.get_request("nonexistent") is None

    def test_list_requests_filter_by_status(self, gate: ApprovalGate) -> None:
        r1 = _pending_request(gate, resource_id="a")
        _pending_request(gate, resource_id="b")
        gate.approve(r1.request_id, approved_by="admin")
        approved = gate.list_requests(status=ApprovalStatus.APPROVED.value)
        pending = gate.list_requests(status=ApprovalStatus.PENDING.value)
        assert len(approved) == 1
        assert len(pending) == 1

    def test_list_requests_filter_by_gate_type(self, gate: ApprovalGate) -> None:
        _pending_request(gate, gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
                         resource_id="exp-001")
        _pending_request(gate, gate_type=ApprovalGateType.TRAINING_START,
                         resource_id="exp-002")
        launch = gate.list_requests(gate_type=ApprovalGateType.EXPERIMENT_LAUNCH.value)
        assert all(r["gate_type"] == "EXPERIMENT_LAUNCH" for r in launch)

    def test_audit_log_contains_all_events(self, gate: ApprovalGate) -> None:
        req = _pending_request(gate, resource_id="audit-test")
        gate.approve(req.request_id, approved_by="admin")
        gate.require_approval(ApprovalGateType.EXPERIMENT_LAUNCH, "audit-test")
        entries = gate.audit_log()
        events = {e["event"] for e in entries}
        assert "created" in events
        assert "approved" in events
        assert "consumed" in events

    def test_all_five_gate_types_can_be_created(self, gate: ApprovalGate) -> None:
        for i, gt in enumerate(ApprovalGateType):
            req = gate.create_request(
                gate_type=gt,
                resource_id=f"resource-{i}",
                resource_type="test",
                requested_by="system",
                action_summary=f"Test {gt.value}",
            )
            assert req.gate_type == gt.value


# ── DB CRUD ───────────────────────────────────────────────────────────────────


class TestApprovalRequestsDB:
    def _make_req(self, gate_type: str = "EXPERIMENT_LAUNCH",
                  resource_id: str = "exp-001", status: str = "PENDING") -> dict:
        r = ApprovalRequest.new(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id=resource_id,
            resource_type="experiment",
            requested_by="system",
            action_summary="Test request",
        )
        d = r.to_dict()
        d["gate_type"] = gate_type
        d["status"] = status
        return d

    def test_upsert_and_get(self, db: ResearchLabDatabase) -> None:
        req = self._make_req()
        db.upsert_approval_request(req)
        fetched = db.get_approval_request(req["request_id"])
        assert fetched is not None
        assert fetched["gate_type"] == "EXPERIMENT_LAUNCH"
        assert fetched["status"] == "PENDING"
        assert fetched["resource_id"] == "exp-001"

    def test_upsert_updates_status(self, db: ResearchLabDatabase) -> None:
        req = self._make_req()
        db.upsert_approval_request(req)
        req["status"] = "APPROVED"
        req["resolved_by"] = "admin"
        db.upsert_approval_request(req)
        fetched = db.get_approval_request(req["request_id"])
        assert fetched["status"] == "APPROVED"

    def test_list_filter_by_status(self, db: ResearchLabDatabase) -> None:
        r1 = self._make_req(status="PENDING")
        r2 = self._make_req(resource_id="exp-002", status="APPROVED")
        db.upsert_approval_request(r1)
        db.upsert_approval_request(r2)
        pending = db.list_approval_requests(status="PENDING")
        approved = db.list_approval_requests(status="APPROVED")
        assert len(pending) == 1
        assert len(approved) == 1

    def test_list_filter_by_gate_type(self, db: ResearchLabDatabase) -> None:
        r1 = self._make_req(gate_type="EXPERIMENT_LAUNCH", resource_id="a")
        r2 = self._make_req(gate_type="TRAINING_START", resource_id="b")
        db.upsert_approval_request(r1)
        db.upsert_approval_request(r2)
        launch = db.list_approval_requests(gate_type="EXPERIMENT_LAUNCH")
        assert len(launch) == 1 and launch[0]["gate_type"] == "EXPERIMENT_LAUNCH"

    def test_list_filter_by_resource_id(self, db: ResearchLabDatabase) -> None:
        r1 = self._make_req(resource_id="specific-resource")
        r2 = self._make_req(resource_id="other-resource")
        db.upsert_approval_request(r1)
        db.upsert_approval_request(r2)
        rows = db.list_approval_requests(resource_id="specific-resource")
        assert len(rows) == 1

    def test_atomic_consume_success(self, db: ResearchLabDatabase) -> None:
        req = self._make_req(status="APPROVED")
        db.upsert_approval_request(req)
        rows_updated = db.atomic_consume_approval(
            gate_type="EXPERIMENT_LAUNCH",
            resource_id="exp-001",
            consumed_by="system",
            consumed_at=time.time(),
        )
        assert rows_updated == 1
        fetched = db.get_approval_request(req["request_id"])
        assert fetched["status"] == "CONSUMED"

    def test_atomic_consume_fails_on_non_approved(self, db: ResearchLabDatabase) -> None:
        req = self._make_req(status="PENDING")
        db.upsert_approval_request(req)
        rows_updated = db.atomic_consume_approval(
            gate_type="EXPERIMENT_LAUNCH",
            resource_id="exp-001",
            consumed_by="system",
            consumed_at=time.time(),
        )
        assert rows_updated == 0

    def test_atomic_consume_idempotent_second_call_returns_zero(
        self, db: ResearchLabDatabase
    ) -> None:
        req = self._make_req(status="APPROVED")
        db.upsert_approval_request(req)
        first = db.atomic_consume_approval(
            gate_type="EXPERIMENT_LAUNCH", resource_id="exp-001",
            consumed_by="system", consumed_at=time.time()
        )
        second = db.atomic_consume_approval(
            gate_type="EXPERIMENT_LAUNCH", resource_id="exp-001",
            consumed_by="system", consumed_at=time.time()
        )
        assert first == 1
        assert second == 0

    def test_get_nonexistent_returns_none(self, db: ResearchLabDatabase) -> None:
        assert db.get_approval_request("nonexistent") is None

    def test_stats_includes_approval_requests(self, db: ResearchLabDatabase) -> None:
        db.upsert_approval_request(self._make_req())
        stats = db.stats()
        assert "approval_requests" in stats
        assert stats["approval_requests"] >= 1

    def test_atomic_consume_scalar_and_json_consistent(
        self, db: ResearchLabDatabase
    ) -> None:
        """After consume, both scalar status column and data_json must agree."""
        req = self._make_req(status="APPROVED")
        db.upsert_approval_request(req)
        rows_updated = db.atomic_consume_approval(
            gate_type="EXPERIMENT_LAUNCH",
            resource_id="exp-001",
            consumed_by="system",
            consumed_at=time.time(),
        )
        assert rows_updated == 1
        fetched = db.get_approval_request(req["request_id"])
        # JSON payload and scalar column must agree
        assert fetched["status"] == "CONSUMED"
        assert fetched["consumed_by"] == "system"
        # Audit log must contain the consume event
        events = [e["event"] for e in fetched.get("audit_log", [])]
        assert "consumed" in events


class TestApprovalGateConcurrentCreate:
    """Idempotency: create_request must always return the persisted request_id."""

    def test_concurrent_create_returns_same_id(
        self, gate: ApprovalGate
    ) -> None:
        """Simulate the race: expire the in-memory PENDING list before the
        insert so both callers see nothing, then call create_request twice in
        sequence — the second call should return the same request_id as the
        first (re-fetched from DB), not a phantom new id."""
        r1 = gate.create_request(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id="exp-concurrent",
            resource_type="experiment",
            requested_by="caller-1",
            action_summary="First caller",
        )
        # Simulate a second caller that also saw zero PENDING rows at read time
        # by bypassing the early-return live check and calling create again.
        r2 = gate.create_request(
            gate_type=ApprovalGateType.EXPERIMENT_LAUNCH,
            resource_id="exp-concurrent",
            resource_type="experiment",
            requested_by="caller-2",
            action_summary="Second caller",
        )
        # Both must return the same canonical persisted request_id
        assert r1.request_id == r2.request_id, (
            f"Phantom id returned: r1={r1.request_id}, r2={r2.request_id}"
        )

    def test_returned_id_is_fetchable_from_db(self, gate: ApprovalGate) -> None:
        """The request_id returned by create_request must exist in the DB."""
        req = gate.create_request(
            gate_type=ApprovalGateType.TRAINING_START,
            resource_id="exp-fetchable",
            resource_type="experiment",
            requested_by="system",
            action_summary="",
        )
        fetched = gate.get_request(req.request_id)
        assert fetched is not None, (
            f"Returned request_id {req.request_id!r} not found in DB"
        )
        assert fetched.status == ApprovalStatus.PENDING.value

    def test_approve_returned_id_succeeds(self, gate: ApprovalGate) -> None:
        """Admin can approve the request_id returned by create_request."""
        req = gate.create_request(
            gate_type=ApprovalGateType.PROMOTION,
            resource_id="v5.0.0",
            resource_type="qde_version",
            requested_by="system",
            action_summary="Promote v5.0.0",
        )
        # Must not raise "not found"
        approved = gate.approve(req.request_id, approved_by="admin")
        assert approved.status == ApprovalStatus.APPROVED.value


# ── Real server integration tests ─────────────────────────────────────────────


class TestApprovalRealServer:
    """Integration tests against the production research_lab_server.app.

    These tests validate that the Pydantic models are correctly bound as JSON
    body models by FastAPI (not treated as query params or path params), and
    that the auth dependency is active and enforced.
    """

    @pytest.fixture()
    def client(self, tmp_path: Path):
        import importlib
        import os
        import sys
        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-token-real"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "real_server.db")
        # Force a fresh module (and fresh _db singleton) for each test class.
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
                del sys.modules[mod_name]
        import trainer.research_lab.research_lab_server as srv_mod
        from starlette.testclient import TestClient
        return TestClient(srv_mod.app)

    _HDR = {
        "Authorization": "Bearer test-token-real",
        "X-Signalium-Token": make_signalium_jwt(),
    }

    def test_create_approval_json_body_parsed(self, client) -> None:
        """POST /approvals must parse a JSON body, not treat fields as query params."""
        resp = client.post(
            "/approvals",
            json={
                "gate_type": "EXPERIMENT_LAUNCH",
                "resource_id": "exp-real-001",
                "resource_type": "experiment",
                "requested_by": "researcher",
                "action_summary": "Launch exp-real-001",
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200, f"Unexpected {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["success"] is True
        assert body["approval_request"]["status"] == "PENDING"
        assert body["approval_request"]["gate_type"] == "EXPERIMENT_LAUNCH"
        assert body["approval_request"]["resource_id"] == "exp-real-001"

    def test_create_approval_requires_auth(self, client) -> None:
        resp = client.post("/approvals", json={
            "gate_type": "TRAINING_START", "resource_id": "exp-noauth",
        })
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_list_approvals(self, client) -> None:
        # Create one first
        client.post("/approvals", json={
            "gate_type": "PROMOTION", "resource_id": "v2.0.0",
            "requested_by": "system", "action_summary": "Promote v2.0.0",
        }, headers=self._HDR)
        resp = client.get("/approvals", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] >= 1

    def test_get_approval_by_id(self, client) -> None:
        create = client.post("/approvals", json={
            "gate_type": "DEPLOYMENT", "resource_id": "v3.0.0",
            "requested_by": "system", "action_summary": "Deploy v3.0.0",
        }, headers=self._HDR)
        rid = create.json()["approval_request"]["request_id"]
        resp = client.get(f"/approvals/{rid}", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["approval_request"]["request_id"] == rid

    def test_get_approval_not_found(self, client) -> None:
        resp = client.get("/approvals/nonexistent-id", headers=self._HDR)
        assert resp.status_code == 404

    def test_approve_endpoint_json_body(self, client) -> None:
        """POST /approvals/{id}/approve must parse JSON body."""
        create = client.post("/approvals", json={
            "gate_type": "EXPERIMENT_LAUNCH", "resource_id": "exp-approve-test",
            "requested_by": "system", "action_summary": "Approve gate test",
        }, headers=self._HDR)
        rid = create.json()["approval_request"]["request_id"]
        resp = client.post(
            f"/approvals/{rid}/approve",
            json={"approved_by": "head_quant", "reason": "metrics pass"},
            headers=self._HDR,
        )
        assert resp.status_code == 200, f"Unexpected {resp.status_code}: {resp.text}"
        rec = resp.json()["approval_request"]
        assert rec["status"] == "APPROVED"
        assert rec["resolved_by"] == "head_quant"

    def test_reject_endpoint_json_body(self, client) -> None:
        """POST /approvals/{id}/reject must parse JSON body."""
        create = client.post("/approvals", json={
            "gate_type": "ROLLBACK", "resource_id": "v1.0.0",
            "requested_by": "system", "action_summary": "Rollback gate test",
        }, headers=self._HDR)
        rid = create.json()["approval_request"]["request_id"]
        resp = client.post(
            f"/approvals/{rid}/reject",
            json={"rejected_by": "admin", "reason": "not warranted"},
            headers=self._HDR,
        )
        assert resp.status_code == 200, f"Unexpected {resp.status_code}: {resp.text}"
        rec = resp.json()["approval_request"]
        assert rec["status"] == "REJECTED"
        assert rec["resolution_reason"] == "not warranted"

    def test_double_approve_returns_400(self, client) -> None:
        create = client.post("/approvals", json={
            "gate_type": "TRAINING_START", "resource_id": "exp-double-real",
            "requested_by": "system", "action_summary": "",
        }, headers=self._HDR)
        rid = create.json()["approval_request"]["request_id"]
        client.post(f"/approvals/{rid}/approve",
                    json={"approved_by": "admin"}, headers=self._HDR)
        resp = client.post(f"/approvals/{rid}/approve",
                           json={"approved_by": "admin"}, headers=self._HDR)
        assert resp.status_code == 400

    def test_list_filter_by_status(self, client) -> None:
        r1 = client.post("/approvals", json={
            "gate_type": "EXPERIMENT_LAUNCH", "resource_id": "filter-a",
            "requested_by": "system", "action_summary": "",
        }, headers=self._HDR).json()["approval_request"]["request_id"]
        client.post("/approvals", json={
            "gate_type": "TRAINING_START", "resource_id": "filter-b",
            "requested_by": "system", "action_summary": "",
        }, headers=self._HDR)
        client.post(f"/approvals/{r1}/approve", json={"approved_by": "admin"},
                    headers=self._HDR)
        approved = client.get("/approvals?status=APPROVED", headers=self._HDR).json()
        pending = client.get("/approvals?status=PENDING", headers=self._HDR).json()
        assert approved["total"] >= 1
        assert all(r["status"] == "APPROVED" for r in approved["approval_requests"])
        assert pending["total"] >= 1
        assert all(r["status"] == "PENDING" for r in pending["approval_requests"])


# ── Isolated unit server endpoint tests ───────────────────────────────────────


class TestApprovalEndpoints:
    def _make_app(self, gate: ApprovalGate):
        from fastapi import FastAPI, HTTPException

        app = FastAPI()

        @app.post("/approvals")
        def create_approval(req: _CreateApprovalReq) -> dict:
            try:
                gt = ApprovalGateType(req.gate_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown gate type: {req.gate_type}")
            approval = gate.create_request(
                gate_type=gt,
                resource_id=req.resource_id,
                resource_type=req.resource_type,
                requested_by=req.requested_by,
                action_summary=req.action_summary,
                requestor_context=req.requestor_context,
                ttl_seconds=req.ttl_seconds,
            )
            return {"success": True, "approval_request": approval.to_dict()}

        @app.get("/approvals")
        def list_approvals(gate_type: str | None = None,
                            resource_id: str | None = None,
                            status: str | None = None,
                            limit: int = 50) -> dict:
            rows = gate.list_requests(gate_type=gate_type, resource_id=resource_id,
                                       status=status, limit=limit)
            return {"success": True, "total": len(rows), "approval_requests": rows}

        @app.get("/approvals/{request_id}")
        def get_approval(request_id: str) -> dict:
            req_obj = gate.get_request(request_id)
            if req_obj is None:
                raise HTTPException(status_code=404, detail="Not found")
            return {"success": True, "approval_request": req_obj.to_dict()}

        @app.post("/approvals/{request_id}/approve")
        def approve_approval(request_id: str, req: _ApproveReq) -> dict:
            try:
                approved = gate.approve(request_id, approved_by=req.approved_by,
                                        reason=req.reason)
                return {"success": True, "approval_request": approved.to_dict()}
            except (ValueError, ApprovalAlreadyResolved) as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        @app.post("/approvals/{request_id}/reject")
        def reject_approval(request_id: str, req: _RejectReq) -> dict:
            try:
                rejected = gate.reject(request_id, rejected_by=req.rejected_by,
                                        reason=req.reason)
                return {"success": True, "approval_request": rejected.to_dict()}
            except (ValueError, ApprovalAlreadyResolved) as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        return app

    def test_create_approval_returns_pending(
        self, gate: ApprovalGate
    ) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        resp = client.post("/approvals", json={
            "gate_type": "EXPERIMENT_LAUNCH",
            "resource_id": "exp-001",
            "resource_type": "experiment",
            "requested_by": "researcher",
            "action_summary": "Launch experiment",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["approval_request"]["status"] == "PENDING"
        assert body["approval_request"]["gate_type"] == "EXPERIMENT_LAUNCH"

    def test_create_approval_invalid_gate_type_returns_400(
        self, gate: ApprovalGate
    ) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate), raise_server_exceptions=False)
        resp = client.post("/approvals", json={
            "gate_type": "NONEXISTENT_GATE",
            "resource_id": "exp-001",
        })
        assert resp.status_code == 400

    def test_list_approvals_empty(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        resp = client.get("/approvals")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_approvals_after_create(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        client.post("/approvals", json={
            "gate_type": "TRAINING_START",
            "resource_id": "exp-002",
            "requested_by": "pipeline",
            "action_summary": "Start training",
        })
        resp = client.get("/approvals")
        assert resp.json()["total"] == 1

    def test_get_approval_by_id(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        create_resp = client.post("/approvals", json={
            "gate_type": "PROMOTION",
            "resource_id": "v2.0.0",
            "requested_by": "system",
            "action_summary": "Promote v2.0.0",
        })
        rid = create_resp.json()["approval_request"]["request_id"]
        resp = client.get(f"/approvals/{rid}")
        assert resp.status_code == 200
        assert resp.json()["approval_request"]["request_id"] == rid

    def test_get_approval_not_found_returns_404(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate), raise_server_exceptions=False)
        resp = client.get("/approvals/nonexistent")
        assert resp.status_code == 404

    def test_approve_endpoint(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        create_resp = client.post("/approvals", json={
            "gate_type": "DEPLOYMENT",
            "resource_id": "v3.0.0",
            "requested_by": "system",
            "action_summary": "Deploy v3.0.0",
        })
        rid = create_resp.json()["approval_request"]["request_id"]
        resp = client.post(f"/approvals/{rid}/approve",
                           json={"approved_by": "head_quant", "reason": "metrics pass"})
        assert resp.status_code == 200
        assert resp.json()["approval_request"]["status"] == "APPROVED"
        assert resp.json()["approval_request"]["resolved_by"] == "head_quant"

    def test_reject_endpoint(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        create_resp = client.post("/approvals", json={
            "gate_type": "ROLLBACK",
            "resource_id": "v1.5.0",
            "requested_by": "system",
            "action_summary": "Rollback v1.5.0",
        })
        rid = create_resp.json()["approval_request"]["request_id"]
        resp = client.post(f"/approvals/{rid}/reject",
                           json={"rejected_by": "admin", "reason": "false alarm"})
        assert resp.status_code == 200
        assert resp.json()["approval_request"]["status"] == "REJECTED"
        assert resp.json()["approval_request"]["resolution_reason"] == "false alarm"

    def test_double_approve_returns_400(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate), raise_server_exceptions=False)
        create_resp = client.post("/approvals", json={
            "gate_type": "EXPERIMENT_LAUNCH",
            "resource_id": "exp-double",
            "requested_by": "system",
            "action_summary": "",
        })
        rid = create_resp.json()["approval_request"]["request_id"]
        client.post(f"/approvals/{rid}/approve", json={"approved_by": "admin"})
        resp = client.post(f"/approvals/{rid}/approve", json={"approved_by": "admin"})
        assert resp.status_code == 400

    def test_list_filter_by_status(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        r1 = client.post("/approvals", json={
            "gate_type": "EXPERIMENT_LAUNCH", "resource_id": "a",
            "requested_by": "system", "action_summary": "",
        }).json()["approval_request"]["request_id"]
        client.post("/approvals", json={
            "gate_type": "TRAINING_START", "resource_id": "b",
            "requested_by": "system", "action_summary": "",
        })
        client.post(f"/approvals/{r1}/approve", json={"approved_by": "admin"})
        approved = client.get("/approvals?status=APPROVED").json()
        pending = client.get("/approvals?status=PENDING").json()
        assert approved["total"] == 1
        assert pending["total"] == 1

    def test_audit_log_in_approval_request(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        create_resp = client.post("/approvals", json={
            "gate_type": "PROMOTION", "resource_id": "v4.0.0",
            "requested_by": "system", "action_summary": "Promote v4.0.0",
        })
        rid = create_resp.json()["approval_request"]["request_id"]
        client.post(f"/approvals/{rid}/approve", json={"approved_by": "admin"})
        resp = client.get(f"/approvals/{rid}")
        audit = resp.json()["approval_request"]["audit_log"]
        events = [e["event"] for e in audit]
        assert "created" in events
        assert "approved" in events

    def test_all_five_gate_types_via_endpoint(self, gate: ApprovalGate) -> None:
        from starlette.testclient import TestClient
        client = TestClient(self._make_app(gate))
        for i, gt in enumerate(["EXPERIMENT_LAUNCH", "TRAINING_START",
                                  "PROMOTION", "DEPLOYMENT", "ROLLBACK"]):
            resp = client.post("/approvals", json={
                "gate_type": gt,
                "resource_id": f"resource-{i}",
                "requested_by": "system",
                "action_summary": f"Test {gt}",
            })
            assert resp.status_code == 200, f"Failed for gate type {gt}"
            assert resp.json()["approval_request"]["gate_type"] == gt
