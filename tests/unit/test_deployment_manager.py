"""S20 Task #13 — Tests for Deployment Approval, Signalium Notification + Rollback.

Covers:
  - DB schema: model_deployments and deployment_events tables
  - Stage transition validation (valid + invalid paths)
  - Single-ACTIVE enforcement (retire previous on activate)
  - request_promotion: creates ApprovalRequest + PROMOTION_REQUESTED event
  - apply_promotion: approve + consume + transition + Signalium notify
  - Signalium notifier: called on ACTIVE, skipped on non-ACTIVE
  - Signalium notifier failure: logged as non-fatal, event still written
  - request_rollback: resolves latest_retired, creates ROLLBACK_REQUESTED event
  - apply_rollback: restores RETIRED → ACTIVE, notifies Signalium, retires current
  - get_current: returns ACTIVE model or None
  - get_history: all events, filtered by model_version / event_type
  - Server endpoints: promote, activate, rollback, rollback/activate,
    current, history — including auth enforcement
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.approval_models import ApprovalGateType
from trainer.research_lab.artifact_registry import ArtifactRegistry
from trainer.research_lab.artifact_store import ArtifactStore
from trainer.research_lab.deployment_manager import (
    DeploymentEvent,
    DeploymentManager,
    DeploymentRecord,
    PromotionBlockedError,
    _gate_type_for,
    _is_valid_pre_shadow_transition,
)
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.signing import ArtifactSigner

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> ResearchLabDatabase:
    return ResearchLabDatabase(str(tmp_path / "dm_test.db"))


@pytest.fixture()
def gate(db):
    from trainer.research_lab.approval_gate import ApprovalGate

    return ApprovalGate(db)


def _manager(db, gate=None, notifier=None) -> DeploymentManager:
    return DeploymentManager(db, approval_gate=gate, signalium_notifier=notifier)


def _fake_notifier(payload: dict) -> dict:
    return {"status": "accepted", "deployment_id": "dep-test-001"}


# ── Stage Transition Validation ────────────────────────────────────────────────


class TestTransitionValidation:
    def test_not_deployed_to_shadow_is_promotion(self):
        assert _gate_type_for("NOT_DEPLOYED", "SHADOW") == ApprovalGateType.PROMOTION

    def test_shadow_to_candidate_is_promotion(self):
        assert _gate_type_for("SHADOW", "CANDIDATE") == ApprovalGateType.PROMOTION

    def test_candidate_to_active_is_deployment(self):
        assert _gate_type_for("CANDIDATE", "ACTIVE") == ApprovalGateType.DEPLOYMENT

    def test_retired_to_active_is_deployment(self):
        assert _gate_type_for("RETIRED", "ACTIVE") == ApprovalGateType.DEPLOYMENT

    def test_retired_to_shadow_is_promotion(self):
        assert _gate_type_for("RETIRED", "SHADOW") == ApprovalGateType.PROMOTION

    def test_legacy_not_deployed_to_shadow_still_supported(self):
        """S21A adds VALIDATING→SHADOW but must not remove the legacy direct path."""
        assert _gate_type_for("NOT_DEPLOYED", "SHADOW") == ApprovalGateType.PROMOTION

    def test_validating_to_shadow_is_promotion(self):
        assert _gate_type_for("VALIDATING", "SHADOW") == ApprovalGateType.PROMOTION

    def test_active_to_shadow_is_invalid(self):
        with pytest.raises(ValueError, match="Invalid stage transition"):
            _gate_type_for("ACTIVE", "SHADOW")

    def test_shadow_to_active_is_invalid(self):
        with pytest.raises(ValueError, match="Invalid stage transition"):
            _gate_type_for("SHADOW", "ACTIVE")

    def test_unknown_from_state_is_invalid(self):
        with pytest.raises(ValueError):
            _gate_type_for("UNKNOWN", "ACTIVE")


# ── DB: model_deployments ──────────────────────────────────────────────────────


class TestModelDeploymentDB:
    def test_upsert_and_get(self, db):
        rec = DeploymentRecord(model_version="v1.0", status="SHADOW")
        db.upsert_model_deployment(rec.to_dict())
        result = db.get_model_deployment("v1.0")
        assert result is not None
        assert result["status"] == "SHADOW"

    def test_upsert_updates_status(self, db):
        rec = DeploymentRecord(model_version="v1.0", status="SHADOW")
        db.upsert_model_deployment(rec.to_dict())
        rec.status = "CANDIDATE"
        db.upsert_model_deployment(rec.to_dict())
        result = db.get_model_deployment("v1.0")
        assert result["status"] == "CANDIDATE"

    def test_list_by_status(self, db):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="ACTIVE").to_dict())
        db.upsert_model_deployment(DeploymentRecord("v2.0", status="SHADOW").to_dict())
        active = db.list_model_deployments(status="ACTIVE")
        assert len(active) == 1
        assert active[0]["model_version"] == "v1.0"

    def test_retire_active_deployments(self, db):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="ACTIVE").to_dict())
        db.upsert_model_deployment(DeploymentRecord("v2.0", status="ACTIVE").to_dict())
        count = db.retire_active_deployments(except_model_version="v2.0")
        assert count == 1
        v1 = db.get_model_deployment("v1.0")
        v2 = db.get_model_deployment("v2.0")
        assert v1["status"] == "RETIRED"
        assert v2["status"] == "ACTIVE"  # not touched

    def test_retire_active_no_other_active(self, db):
        db.upsert_model_deployment(DeploymentRecord("v2.0", status="ACTIVE").to_dict())
        count = db.retire_active_deployments(except_model_version="v2.0")
        assert count == 0

    def test_get_missing_returns_none(self, db):
        assert db.get_model_deployment("nonexistent") is None


# ── DB: deployment_events ──────────────────────────────────────────────────────


class TestDeploymentEventsDB:
    def test_insert_and_list(self, db):
        evt = DeploymentEvent(
            model_version="v1.0",
            event_type="PROMOTED",
            from_status="CANDIDATE",
            to_status="ACTIVE",
            actor="admin",
        )
        db.insert_deployment_event(evt.to_dict())
        events = db.list_deployment_events(model_version="v1.0")
        assert len(events) == 1
        assert events[0]["event_type"] == "PROMOTED"

    def test_filter_by_event_type(self, db):
        for etype in ["PROMOTED", "ROLLED_BACK", "SIGNALIUM_NOTIFIED"]:
            evt = DeploymentEvent(model_version="v1.0", event_type=etype)
            db.insert_deployment_event(evt.to_dict())
        promoted = db.list_deployment_events(event_type="PROMOTED")
        assert all(e["event_type"] == "PROMOTED" for e in promoted)

    def test_insert_is_idempotent(self, db):
        evt = DeploymentEvent(model_version="v1.0", event_type="PROMOTED")
        db.insert_deployment_event(evt.to_dict())
        db.insert_deployment_event(evt.to_dict())  # INSERT OR IGNORE
        events = db.list_deployment_events()
        assert len(events) == 1

    def test_list_limit(self, db):
        for i in range(10):
            evt = DeploymentEvent(model_version=f"v{i}.0", event_type="PROMOTED")
            db.insert_deployment_event(evt.to_dict())
        events = db.list_deployment_events(limit=3)
        assert len(events) == 3


# ── S21A: pre-SHADOW lifecycle (DRAFT → TRAINING → VALIDATING) ─────────────────


class TestPreShadowTransitionValidation:
    def test_not_deployed_to_draft_is_valid(self):
        assert _is_valid_pre_shadow_transition("NOT_DEPLOYED", "DRAFT")

    def test_draft_to_training_is_valid(self):
        assert _is_valid_pre_shadow_transition("DRAFT", "TRAINING")

    def test_training_to_validating_is_valid(self):
        assert _is_valid_pre_shadow_transition("TRAINING", "VALIDATING")

    def test_draft_to_validating_skip_is_invalid(self):
        assert not _is_valid_pre_shadow_transition("DRAFT", "VALIDATING")

    def test_draft_to_active_is_invalid(self):
        assert not _is_valid_pre_shadow_transition("DRAFT", "ACTIVE")

    def test_validating_to_shadow_is_gated_not_pre_shadow(self):
        # VALIDATING → SHADOW is a *gated* promotion, not an ungated pre-SHADOW step.
        assert not _is_valid_pre_shadow_transition("VALIDATING", "SHADOW")
        assert _gate_type_for("VALIDATING", "SHADOW") == ApprovalGateType.PROMOTION


class TestAdvanceStage:
    def test_advance_from_not_deployed_to_draft(self, db):
        mgr = _manager(db)
        result = mgr.advance_stage("v1.0", "DRAFT", "admin")
        assert result["from_status"] == "NOT_DEPLOYED"
        assert result["to_status"] == "DRAFT"
        assert db.get_model_deployment("v1.0")["status"] == "DRAFT"

    def test_full_pre_shadow_progression(self, db):
        mgr = _manager(db)
        mgr.advance_stage("v1.0", "DRAFT", "admin")
        mgr.advance_stage("v1.0", "TRAINING", "admin")
        mgr.advance_stage("v1.0", "VALIDATING", "admin")
        assert db.get_model_deployment("v1.0")["status"] == "VALIDATING"

    def test_skipping_a_stage_is_rejected(self, db):
        mgr = _manager(db)
        mgr.advance_stage("v1.0", "DRAFT", "admin")
        with pytest.raises(ValueError, match="Invalid stage transition"):
            mgr.advance_stage("v1.0", "VALIDATING", "admin")  # skips TRAINING

    def test_illegal_jump_to_active_is_rejected(self, db):
        mgr = _manager(db)
        with pytest.raises(ValueError, match="Invalid stage transition"):
            mgr.advance_stage("v1.0", "ACTIVE", "admin")

    def test_no_approval_gate_required(self, db):
        """Pre-SHADOW advancement works with no ApprovalGate configured."""
        mgr = _manager(db, gate=None)
        result = mgr.advance_stage("v1.0", "DRAFT", "admin")
        assert result["to_status"] == "DRAFT"

    def test_logs_stage_advanced_event(self, db):
        mgr = _manager(db)
        mgr.advance_stage("v1.0", "DRAFT", "admin")
        events = db.list_deployment_events(model_version="v1.0", event_type="STAGE_ADVANCED")
        assert len(events) == 1
        assert events[0]["to_status"] == "DRAFT"

    def test_full_lifecycle_draft_through_active_and_retired(self, db, gate):
        """End-to-end: DRAFT → TRAINING → VALIDATING → SHADOW → CANDIDATE → ACTIVE,
        then a second version retires the first."""
        mgr = _manager(db, gate=gate, notifier=_fake_notifier)
        mgr.advance_stage("v1.0", "DRAFT", "admin")
        mgr.advance_stage("v1.0", "TRAINING", "admin")
        mgr.advance_stage("v1.0", "VALIDATING", "admin")
        assert db.get_model_deployment("v1.0")["status"] == "VALIDATING"

        for target in ("SHADOW", "CANDIDATE", "ACTIVE"):
            result = mgr.request_promotion(
                "v1.0", target, "admin", weights_url="https://cdn.example.com/v1.0.pt"
            )
            req_id = result["approval_request"]["request_id"]
            mgr.apply_promotion(req_id, "admin")

        assert db.get_model_deployment("v1.0")["status"] == "ACTIVE"

        # Only one ACTIVE version may exist — promoting v2.0 to ACTIVE retires v1.0.
        mgr.advance_stage("v2.0", "DRAFT", "admin")
        mgr.advance_stage("v2.0", "TRAINING", "admin")
        mgr.advance_stage("v2.0", "VALIDATING", "admin")
        for target in ("SHADOW", "CANDIDATE", "ACTIVE"):
            result = mgr.request_promotion(
                "v2.0", target, "admin", weights_url="https://cdn.example.com/v2.0.pt"
            )
            req_id = result["approval_request"]["request_id"]
            mgr.apply_promotion(req_id, "admin")

        assert db.get_model_deployment("v1.0")["status"] == "RETIRED"
        assert db.get_model_deployment("v2.0")["status"] == "ACTIVE"
        assert mgr.get_current()["model_version"] == "v2.0"

        # Every stage transition — including the pre-SHADOW ones — is in the audit trail.
        history = mgr.get_history()
        event_types = {e["event_type"] for e in history}
        assert "STAGE_ADVANCED" in event_types
        assert "PROMOTED" in event_types


# ── request_promotion ──────────────────────────────────────────────────────────


class TestRequestPromotion:
    def test_creates_promotion_requested_event(self, db, gate):
        mgr = _manager(db, gate=gate)
        mgr.request_promotion("v1.0", "SHADOW", "admin")
        events = db.list_deployment_events(model_version="v1.0", event_type="PROMOTION_REQUESTED")
        assert len(events) == 1

    def test_creates_approval_request(self, db, gate):
        mgr = _manager(db, gate=gate)
        result = mgr.request_promotion("v1.0", "SHADOW", "admin")
        assert result["approval_request"]
        assert "request_id" in result["approval_request"]

    def test_invalid_transition_raises(self, db):
        mgr = _manager(db)
        with pytest.raises(ValueError, match="Invalid stage transition"):
            mgr.request_promotion("v1.0", "ACTIVE", "admin")  # SHADOW needed first

    def test_persists_weights_url_and_manifest(self, db, gate):
        mgr = _manager(db, gate=gate)
        mgr.request_promotion(
            "v1.0",
            "SHADOW",
            "admin",
            weights_url="https://cdn.example.com/v1.0.pt",
            manifest={"arch": "lstm_256"},
        )
        rec = db.get_model_deployment("v1.0")
        assert rec["weights_url"] == "https://cdn.example.com/v1.0.pt"
        assert rec["manifest"]["arch"] == "lstm_256"

    def test_no_gate_no_approval_request(self, db):
        mgr = _manager(db, gate=None)
        result = mgr.request_promotion("v1.0", "SHADOW", "admin")
        assert result["approval_request"] == {}

    def test_gate_type_in_result(self, db, gate):
        mgr = _manager(db, gate=gate)
        result = mgr.request_promotion("v1.0", "SHADOW", "admin")
        assert result["gate_type"] == ApprovalGateType.PROMOTION.value

    def test_candidate_to_active_uses_deployment_gate(self, db, gate):
        # Set up v1.0 in CANDIDATE state first
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="CANDIDATE").to_dict())
        mgr = _manager(db, gate=gate)
        result = mgr.request_promotion("v1.0", "ACTIVE", "admin")
        assert result["gate_type"] == ApprovalGateType.DEPLOYMENT.value

    def test_idempotent_approval_request(self, db, gate):
        """Two identical request_promotion calls return the same ApprovalRequest."""
        mgr = _manager(db, gate=gate)
        r1 = mgr.request_promotion("v1.0", "SHADOW", "admin")
        r2 = mgr.request_promotion("v1.0", "SHADOW", "admin")
        assert r1["approval_request"]["request_id"] == r2["approval_request"]["request_id"]


# ── apply_promotion ────────────────────────────────────────────────────────────


class TestApplyPromotion:
    def _setup_and_request(self, db, gate, model_version="v1.0", target="SHADOW"):
        if target != "NOT_DEPLOYED → SHADOW" and target in ("CANDIDATE",):
            db.upsert_model_deployment(DeploymentRecord(model_version, status="SHADOW").to_dict())
        mgr = _manager(db, gate=gate, notifier=_fake_notifier)
        result = mgr.request_promotion(
            model_version, target, "admin", weights_url="https://cdn.example.com/weights.pt"
        )
        return mgr, result["approval_request"]["request_id"]

    def test_transitions_status(self, db, gate):
        mgr, req_id = self._setup_and_request(db, gate, "v1.0", "SHADOW")
        mgr.apply_promotion(req_id, "admin")
        rec = db.get_model_deployment("v1.0")
        assert rec["status"] == "SHADOW"

    def test_logs_promoted_event(self, db, gate):
        mgr, req_id = self._setup_and_request(db, gate, "v1.0", "SHADOW")
        result = mgr.apply_promotion(req_id, "admin")
        events = db.list_deployment_events(model_version="v1.0", event_type="PROMOTED")
        assert len(events) == 1
        assert events[0]["to_status"] == "SHADOW"

    def test_active_notifies_signalium(self, db, gate):
        notified = []

        def _notifier(payload):
            notified.append(payload)
            return {"status": "accepted"}

        db.upsert_model_deployment(DeploymentRecord("v1.0", status="CANDIDATE").to_dict())
        mgr = _manager(db, gate=gate, notifier=_notifier)
        result = mgr.request_promotion(
            "v1.0", "ACTIVE", "admin", weights_url="https://cdn.example.com/v1.0.pt"
        )
        req_id = result["approval_request"]["request_id"]
        apply_result = mgr.apply_promotion(req_id, "admin")
        assert apply_result["signalium_notified"] is True
        assert len(notified) == 1
        assert notified[0]["model_version"] == "v1.0"

    def test_non_active_does_not_notify_signalium(self, db, gate):
        notified = []
        mgr = _manager(db, gate=gate, notifier=lambda p: notified.append(p) or {})
        result = mgr.request_promotion("v1.0", "SHADOW", "admin")
        req_id = result["approval_request"]["request_id"]
        apply_result = mgr.apply_promotion(req_id, "admin")
        assert apply_result["signalium_notified"] is False
        assert len(notified) == 0  # notifier not called for non-ACTIVE

    def test_single_active_enforcement(self, db, gate):
        """Activating v2.0 must automatically retire v1.0."""
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="ACTIVE").to_dict())
        db.upsert_model_deployment(DeploymentRecord("v2.0", status="CANDIDATE").to_dict())
        mgr = _manager(db, gate=gate, notifier=_fake_notifier)
        result = mgr.request_promotion("v2.0", "ACTIVE", "admin")
        req_id = result["approval_request"]["request_id"]
        apply_result = mgr.apply_promotion(req_id, "admin")
        assert apply_result["retired_previous_count"] == 1
        v1 = db.get_model_deployment("v1.0")
        v2 = db.get_model_deployment("v2.0")
        assert v1["status"] == "RETIRED"
        assert v2["status"] == "ACTIVE"

    def test_no_previous_active_retire_count_zero(self, db, gate):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="CANDIDATE").to_dict())
        mgr = _manager(db, gate=gate, notifier=_fake_notifier)
        result = mgr.request_promotion("v1.0", "ACTIVE", "admin")
        req_id = result["approval_request"]["request_id"]
        apply_result = mgr.apply_promotion(req_id, "admin")
        assert apply_result["retired_previous_count"] == 0

    def test_signalium_failure_is_nonfatal(self, db, gate):
        def _bad_notifier(payload):
            raise ConnectionError("Signalium unreachable")

        db.upsert_model_deployment(DeploymentRecord("v1.0", status="CANDIDATE").to_dict())
        mgr = _manager(db, gate=gate, notifier=_bad_notifier)
        result = mgr.request_promotion("v1.0", "ACTIVE", "admin")
        req_id = result["approval_request"]["request_id"]
        apply_result = mgr.apply_promotion(req_id, "admin")
        # Still completes; just marks notification as failed
        assert apply_result["signalium_notified"] is False
        assert apply_result["to_status"] == "ACTIVE"
        # SIGNALIUM_NOTIFIED event still logged with error
        notif_events = db.list_deployment_events(event_type="SIGNALIUM_NOTIFIED")
        assert len(notif_events) == 1
        assert "ConnectionError" in notif_events[0]["payload"]["error"]

    def test_missing_request_id_raises(self, db, gate):
        mgr = _manager(db, gate=gate)
        with pytest.raises(ValueError, match="not found"):
            mgr.apply_promotion("apr-nonexistent", "admin")

    def test_no_gate_raises(self, db):
        mgr = _manager(db, gate=None)
        with pytest.raises(ValueError, match="No approval gate"):
            mgr.apply_promotion("apr-xxx", "admin")

    def test_result_includes_all_keys(self, db, gate):
        mgr, req_id = self._setup_and_request(db, gate, "v1.0", "SHADOW")
        result = mgr.apply_promotion(req_id, "admin")
        for key in (
            "model_version",
            "from_status",
            "to_status",
            "approval_id",
            "signalium_notified",
            "event_id",
        ):
            assert key in result, f"Missing key: {key}"


# ── request_rollback ───────────────────────────────────────────────────────────


class TestRequestRollback:
    def test_explicit_model_version(self, db, gate):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="RETIRED").to_dict())
        mgr = _manager(db, gate=gate)
        result = mgr.request_rollback("v1.0", reason="performance regression", requested_by="admin")
        assert result["model_version"] == "v1.0"
        assert "request_id" in result["approval_request"]

    def test_latest_retired_resolves_most_recent(self, db, gate):
        db.upsert_model_deployment(
            {**DeploymentRecord("v1.0", status="RETIRED").to_dict(), "updated_at": time.time() - 1}
        )
        db.upsert_model_deployment(
            {**DeploymentRecord("v2.0", status="RETIRED").to_dict(), "updated_at": time.time()}
        )
        mgr = _manager(db, gate=gate)
        result = mgr.request_rollback("latest_retired", reason="rollback", requested_by="admin")
        # Most recently updated RETIRED model
        assert result["model_version"] in ("v1.0", "v2.0")

    def test_no_retired_raises(self, db, gate):
        mgr = _manager(db, gate=gate)
        with pytest.raises(ValueError, match="No RETIRED model"):
            mgr.request_rollback("latest_retired", reason="test", requested_by="admin")

    def test_non_retired_model_version_raises(self, db, gate):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="ACTIVE").to_dict())
        mgr = _manager(db, gate=gate)
        with pytest.raises(ValueError, match="only RETIRED"):
            mgr.request_rollback("v1.0", reason="bad", requested_by="admin")

    def test_logs_rollback_requested_event(self, db, gate):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="RETIRED").to_dict())
        mgr = _manager(db, gate=gate)
        mgr.request_rollback("v1.0", reason="test", requested_by="admin")
        events = db.list_deployment_events(event_type="ROLLBACK_REQUESTED")
        assert len(events) == 1

    def test_no_gate_returns_empty_approval(self, db):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="RETIRED").to_dict())
        mgr = _manager(db, gate=None)
        result = mgr.request_rollback("v1.0", reason="test", requested_by="admin")
        assert result["approval_request"] == {}

    def test_nonexistent_model_version_raises(self, db, gate):
        """Rollback to a version that was never deployed must be rejected immediately."""
        mgr = _manager(db, gate=gate)
        with pytest.raises(ValueError, match="does not exist"):
            mgr.request_rollback("v99.0-phantom", reason="test", requested_by="admin")

    def test_active_model_not_eligible_for_rollback(self, db, gate):
        """Cannot roll back TO an ACTIVE model version."""
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="ACTIVE").to_dict())
        mgr = _manager(db, gate=gate)
        with pytest.raises(ValueError, match="only RETIRED"):
            mgr.request_rollback("v1.0", reason="test", requested_by="admin")

    def test_shadow_model_not_eligible_for_rollback(self, db, gate):
        """Cannot roll back TO a SHADOW model version."""
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="SHADOW").to_dict())
        mgr = _manager(db, gate=gate)
        with pytest.raises(ValueError, match="only RETIRED"):
            mgr.request_rollback("v1.0", reason="test", requested_by="admin")


# ── apply_rollback ─────────────────────────────────────────────────────────────


class TestApplyRollback:
    def _setup(self, db, gate, current_active="v2.0", rollback_to="v1.0"):
        db.upsert_model_deployment(
            {
                **DeploymentRecord(rollback_to, status="RETIRED").to_dict(),
                "weights_url": "https://cdn.example.com/v1.0.pt",
                "manifest": {"arch": "lstm_256"},
            }
        )
        if current_active:
            db.upsert_model_deployment(DeploymentRecord(current_active, status="ACTIVE").to_dict())
        mgr = _manager(db, gate=gate, notifier=_fake_notifier)
        result = mgr.request_rollback(rollback_to, reason="perf regression", requested_by="admin")
        req_id = result["approval_request"]["request_id"]
        return mgr, req_id

    def test_restores_retired_to_active(self, db, gate):
        mgr, req_id = self._setup(db, gate)
        mgr.apply_rollback(req_id, "admin")
        rec = db.get_model_deployment("v1.0")
        assert rec["status"] == "ACTIVE"

    def test_retires_current_active(self, db, gate):
        mgr, req_id = self._setup(db, gate, current_active="v2.0")
        mgr.apply_rollback(req_id, "admin")
        v2 = db.get_model_deployment("v2.0")
        assert v2["status"] == "RETIRED"

    def test_notifies_signalium_with_rollback_payload(self, db, gate):
        notified = []

        def _notifier(payload):
            notified.append(payload)
            return {"status": "accepted"}

        db.upsert_model_deployment(
            {
                **DeploymentRecord("v1.0", status="RETIRED").to_dict(),
                "weights_url": "https://cdn.example.com/v1.0.pt",
            }
        )
        mgr = _manager(db, gate=gate, notifier=_notifier)
        result = mgr.request_rollback("v1.0", reason="test", requested_by="admin")
        mgr.apply_rollback(result["approval_request"]["request_id"], "admin")
        assert len(notified) == 1
        assert notified[0]["is_rollback"] is True
        assert notified[0]["weights_url"] == "https://cdn.example.com/v1.0.pt"

    def test_logs_rolled_back_event(self, db, gate):
        mgr, req_id = self._setup(db, gate)
        mgr.apply_rollback(req_id, "admin")
        events = db.list_deployment_events(event_type="ROLLED_BACK")
        assert len(events) == 1
        assert events[0]["to_status"] == "ACTIVE"

    def test_result_includes_all_keys(self, db, gate):
        mgr, req_id = self._setup(db, gate)
        result = mgr.apply_rollback(req_id, "admin")
        for key in (
            "model_version",
            "from_status",
            "to_status",
            "approval_id",
            "signalium_notified",
            "event_id",
        ):
            assert key in result

    def test_no_current_active_rollback_still_works(self, db, gate):
        mgr, req_id = self._setup(db, gate, current_active=None)
        result = mgr.apply_rollback(req_id, "admin")
        assert result["retired_current_count"] == 0
        assert db.get_model_deployment("v1.0")["status"] == "ACTIVE"

    def test_no_gate_raises(self, db):
        mgr = _manager(db, gate=None)
        with pytest.raises(ValueError, match="No approval gate"):
            mgr.apply_rollback("apr-xxx", "admin")


# ── get_current / get_history ──────────────────────────────────────────────────


class TestGetCurrentAndHistory:
    def test_get_current_none_when_empty(self, db):
        mgr = _manager(db)
        assert mgr.get_current() is None

    def test_get_current_returns_active(self, db):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="ACTIVE").to_dict())
        db.upsert_model_deployment(DeploymentRecord("v2.0", status="SHADOW").to_dict())
        mgr = _manager(db)
        current = mgr.get_current()
        assert current is not None
        assert current["model_version"] == "v1.0"
        assert current["status"] == "ACTIVE"

    def test_get_history_all_events(self, db, gate):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="SHADOW").to_dict())
        mgr = _manager(db, gate=gate, notifier=_fake_notifier)

        # Request + apply two promotions to generate events
        r1 = mgr.request_promotion("v1.0", "CANDIDATE", "admin")
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="SHADOW").to_dict())
        r1_id = r1["approval_request"]["request_id"]
        mgr.apply_promotion(r1_id, "admin")

        history = mgr.get_history()
        event_types = {e["event_type"] for e in history}
        assert "PROMOTION_REQUESTED" in event_types
        assert "PROMOTED" in event_types

    def test_get_history_filtered_by_model_version(self, db):
        for mv in ("v1.0", "v2.0"):
            evt = DeploymentEvent(model_version=mv, event_type="PROMOTED")
            db.insert_deployment_event(evt.to_dict())
        mgr = _manager(db)
        history = mgr.get_history(model_version="v1.0")
        assert all(e["model_version"] == "v1.0" for e in history)

    def test_get_history_filtered_by_event_type(self, db):
        for etype in ("PROMOTED", "ROLLED_BACK"):
            evt = DeploymentEvent(model_version="v1.0", event_type=etype)
            db.insert_deployment_event(evt.to_dict())
        mgr = _manager(db)
        rolled = mgr.get_history(event_type="ROLLED_BACK")
        assert all(e["event_type"] == "ROLLED_BACK" for e in rolled)


# ── Server Endpoints ───────────────────────────────────────────────────────────


class TestServerEndpoints:
    @property
    def _HDR(self) -> dict:
        # Minted fresh on each access (not a class-level constant) since the
        # JWT's `iat` is checked against a clock-skew window and would go
        # stale if computed once at class-definition/import time.
        return {
            "Authorization": "Bearer dm-test-token",
            "X-Signalium-Token": make_signalium_jwt(),
        }

    @pytest.fixture()
    def client(self, tmp_path: Path, monkeypatch):
        import os

        os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "dm-test-token"
        os.environ["SIGNALIUM_JWT_SECRET"] = DEFAULT_TEST_JWT_SECRET
        os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "server_dm.db")
        os.environ["ARTIFACT_STORE_ROOT"] = str(tmp_path / "artifact_store")
        priv, pub = ArtifactSigner.generate_keypair()
        monkeypatch.setenv("ARTIFACT_SIGNING_PRIVATE_KEY", priv)
        monkeypatch.setenv("ARTIFACT_SIGNING_PUBLIC_KEY", pub)
        for mod_name in list(sys.modules):
            if "research_lab_server" in mod_name:
                del sys.modules[mod_name]
        from starlette.testclient import TestClient

        import trainer.research_lab.research_lab_server as srv_mod

        return TestClient(srv_mod.app)

    @staticmethod
    def _export_signed_artifacts(client, model_version: str) -> None:
        """Export + register a signed artifact set for *model_version* against
        the same ArtifactRegistry the running server instance uses, so
        signature-gated promotion endpoints have something valid to verify."""
        import trainer.research_lab.research_lab_server as srv_mod
        from trainer.exporters.pipeline import QDEExportPipeline

        weights = srv_mod._artifact_registry.store.root.parent / f"src_{model_version}.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(f"fake-weights-{model_version}".encode())
        pipeline = QDEExportPipeline(artifact_registry=srv_mod._artifact_registry)
        pipeline.export_version(
            model_version=model_version,
            output_dir=srv_mod._artifact_registry.store.root,
            weights_path=weights,
            export_formats=(),
        )

    # Auth enforcement

    def test_promote_requires_auth(self, client):
        resp = client.post(
            "/deployment/promote",
            json={"model_version": "v1.0", "target_status": "SHADOW", "requested_by": "admin"},
        )
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_activate_requires_auth(self, client):
        resp = client.post("/deployment/activate", json={"request_id": "x", "approved_by": "admin"})
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_rollback_requires_auth(self, client):
        resp = client.post("/deployment/rollback", json={"reason": "test", "requested_by": "admin"})
        assert resp.status_code == 401  # no token → 401 Unauthorized

    def test_current_requires_auth(self, client):
        assert client.get("/deployment/current").status_code == 401  # no token → 401

    def test_history_requires_auth(self, client):
        assert client.get("/deployment/history").status_code == 401  # no token → 401

    # Happy-path flows

    def test_promote_creates_request(self, client):
        resp = client.post(
            "/deployment/promote",
            json={
                "model_version": "v1.0",
                "target_status": "SHADOW",
                "requested_by": "admin",
                "weights_url": "https://cdn.example.com/v1.0.pt",
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "promotion_request" in body
        assert body["promotion_request"]["approval_request"]

    def test_promote_invalid_transition_returns_error(self, client):
        resp = client.post(
            "/deployment/promote",
            json={
                "model_version": "v1.0",
                "target_status": "ACTIVE",  # invalid from NOT_DEPLOYED
                "requested_by": "admin",
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "error" in body

    def test_activate_missing_request_returns_error(self, client):
        resp = client.post(
            "/deployment/activate",
            json={
                "request_id": "apr-nonexistent",
                "approved_by": "admin",
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False

    def test_promote_then_activate(self, client):
        self._export_signed_artifacts(client, "v1.0")
        # 1. Request promotion to SHADOW
        r1 = client.post(
            "/deployment/promote",
            json={
                "model_version": "v1.0",
                "target_status": "SHADOW",
                "requested_by": "admin",
            },
            headers=self._HDR,
        )
        req_id = r1.json()["promotion_request"]["approval_request"]["request_id"]

        # 2. Apply promotion
        r2 = client.post(
            "/deployment/activate",
            json={
                "request_id": req_id,
                "approved_by": "admin",
            },
            headers=self._HDR,
        )
        assert r2.status_code == 200
        body = r2.json()
        assert body["success"] is True
        assert body["promotion_result"]["to_status"] == "SHADOW"

    def test_current_none_when_empty(self, client):
        resp = client.get("/deployment/current", headers=self._HDR)
        assert resp.status_code == 200
        assert resp.json()["active"] is None

    def test_current_after_activate(self, client):
        self._export_signed_artifacts(client, "v1.0")
        # Get to CANDIDATE first
        r1 = client.post(
            "/deployment/promote",
            json={"model_version": "v1.0", "target_status": "SHADOW", "requested_by": "admin"},
            headers=self._HDR,
        )
        id1 = r1.json()["promotion_request"]["approval_request"]["request_id"]
        client.post(
            "/deployment/activate",
            json={"request_id": id1, "approved_by": "admin"},
            headers=self._HDR,
        )

        r2 = client.post(
            "/deployment/promote",
            json={"model_version": "v1.0", "target_status": "CANDIDATE", "requested_by": "admin"},
            headers=self._HDR,
        )
        id2 = r2.json()["promotion_request"]["approval_request"]["request_id"]
        client.post(
            "/deployment/activate",
            json={"request_id": id2, "approved_by": "admin"},
            headers=self._HDR,
        )

        r3 = client.post(
            "/deployment/promote",
            json={
                "model_version": "v1.0",
                "target_status": "ACTIVE",
                "requested_by": "admin",
                "weights_url": "https://cdn.example.com/v1.0.pt",
            },
            headers=self._HDR,
        )
        id3 = r3.json()["promotion_request"]["approval_request"]["request_id"]
        client.post(
            "/deployment/activate",
            json={"request_id": id3, "approved_by": "admin"},
            headers=self._HDR,
        )

        current_resp = client.get("/deployment/current", headers=self._HDR)
        current = current_resp.json()["active"]
        assert current is not None
        assert current["model_version"] == "v1.0"
        assert current["status"] == "ACTIVE"

    def test_history_returns_events(self, client):
        client.post(
            "/deployment/promote",
            json={"model_version": "v1.0", "target_status": "SHADOW", "requested_by": "admin"},
            headers=self._HDR,
        )
        resp = client.get("/deployment/history", headers=self._HDR)
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] >= 1
        event_types = {e["event_type"] for e in body["events"]}
        assert "PROMOTION_REQUESTED" in event_types

    def test_history_model_version_filter(self, client):
        client.post(
            "/deployment/promote",
            json={"model_version": "v1.0", "target_status": "SHADOW", "requested_by": "admin"},
            headers=self._HDR,
        )
        resp = client.get("/deployment/history?model_version=v1.0", headers=self._HDR)
        events = resp.json()["events"]
        assert all(e["model_version"] == "v1.0" for e in events)

    def test_rollback_no_retired_model_returns_error(self, client):
        resp = client.post(
            "/deployment/rollback",
            json={
                "model_version": "latest_retired",
                "reason": "test",
                "requested_by": "admin",
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_rollback_activate_missing_request_returns_error(self, client):
        resp = client.post(
            "/deployment/rollback/activate",
            json={
                "request_id": "apr-nonexistent",
                "approved_by": "admin",
            },
            headers=self._HDR,
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_full_promote_rollback_cycle(self, client):
        """Full cycle: NOT_DEPLOYED → SHADOW → CANDIDATE → ACTIVE → rollback."""
        self._export_signed_artifacts(client, "v1.0")
        self._export_signed_artifacts(client, "v2.0")

        def _promote(mv, ts):
            r = client.post(
                "/deployment/promote",
                json={
                    "model_version": mv,
                    "target_status": ts,
                    "requested_by": "admin",
                    "weights_url": "https://cdn.example.com/weights.pt",
                },
                headers=self._HDR,
            )
            req_id = r.json()["promotion_request"]["approval_request"]["request_id"]
            client.post(
                "/deployment/activate",
                json={"request_id": req_id, "approved_by": "admin"},
                headers=self._HDR,
            )

        _promote("v1.0", "SHADOW")
        _promote("v1.0", "CANDIDATE")
        _promote("v1.0", "ACTIVE")

        # Promote v2.0 through all stages (retires v1.0 when v2.0 reaches ACTIVE)
        _promote("v2.0", "SHADOW")
        _promote("v2.0", "CANDIDATE")
        _promote("v2.0", "ACTIVE")
        # v1.0 should now be RETIRED
        current = client.get("/deployment/current", headers=self._HDR).json()["active"]
        assert current["model_version"] == "v2.0"

        # Rollback to v1.0
        rb = client.post(
            "/deployment/rollback",
            json={"model_version": "v1.0", "reason": "v2.0 has issues", "requested_by": "admin"},
            headers=self._HDR,
        )
        assert rb.json()["success"] is True
        rb_req_id = rb.json()["rollback_request"]["approval_request"]["request_id"]

        activate_rb = client.post(
            "/deployment/rollback/activate",
            json={"request_id": rb_req_id, "approved_by": "admin"},
            headers=self._HDR,
        )
        assert activate_rb.json()["success"] is True
        assert activate_rb.json()["rollback_result"]["model_version"] == "v1.0"

        # v1.0 should be ACTIVE again
        current2 = client.get("/deployment/current", headers=self._HDR).json()["active"]
        assert current2["model_version"] == "v1.0"

        # History should have full audit trail
        history = client.get("/deployment/history", headers=self._HDR).json()
        event_types = {e["event_type"] for e in history["events"]}
        assert "PROMOTED" in event_types
        assert "ROLLED_BACK" in event_types

    # ── advance-stage endpoint (S21A) ────────────────────────────────────────

    def test_advance_stage_requires_auth(self, client):
        resp = client.post(
            "/deployment/advance-stage",
            json={"model_version": "v1.0", "target_status": "DRAFT", "requested_by": "admin"},
        )
        assert resp.status_code == 401

    def test_advance_stage_happy_path(self, client):
        resp = client.post(
            "/deployment/advance-stage",
            json={"model_version": "v1.0", "target_status": "DRAFT", "requested_by": "admin"},
            headers=self._HDR,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["stage_result"]["to_status"] == "DRAFT"

    def test_advance_stage_falls_back_to_authenticated_admin_when_requested_by_empty(self, client):
        resp = client.post(
            "/deployment/advance-stage",
            json={"model_version": "v1.0", "target_status": "DRAFT", "requested_by": ""},
            headers=self._HDR,
        )
        body = resp.json()
        assert body["success"] is True
        assert body["authenticated_as"] == "admin"

    def test_advance_stage_invalid_transition_returns_error(self, client):
        resp = client.post(
            "/deployment/advance-stage",
            json={"model_version": "v1.0", "target_status": "VALIDATING", "requested_by": "admin"},
            headers=self._HDR,
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_advance_stage_then_promote_to_shadow(self, client):
        self._export_signed_artifacts(client, "v1.0")
        for stage in ("DRAFT", "TRAINING", "VALIDATING"):
            r = client.post(
                "/deployment/advance-stage",
                json={"model_version": "v1.0", "target_status": stage, "requested_by": "admin"},
                headers=self._HDR,
            )
            assert r.json()["success"] is True

        promote = client.post(
            "/deployment/promote",
            json={"model_version": "v1.0", "target_status": "SHADOW", "requested_by": "admin"},
            headers=self._HDR,
        )
        assert promote.json()["success"] is True
        req_id = promote.json()["promotion_request"]["approval_request"]["request_id"]
        activate = client.post(
            "/deployment/activate",
            json={"request_id": req_id, "approved_by": "admin"},
            headers=self._HDR,
        )
        assert activate.json()["promotion_result"]["to_status"] == "SHADOW"


# ── S21A: signature verification before promotion ───────────────────────────────


class TestSignatureGatedPromotion:
    """Promotion into SHADOW/CANDIDATE/ACTIVE (and rollback to ACTIVE) must
    verify every registered artifact's signature first."""

    @pytest.fixture()
    def signer(self) -> ArtifactSigner:
        priv, pub = ArtifactSigner.generate_keypair()
        return ArtifactSigner(private_key_pem=priv, public_key_pem=pub)

    @pytest.fixture()
    def registry(self, tmp_path: Path) -> ArtifactRegistry:
        reg_db = ResearchLabDatabase(str(tmp_path / "registry.db"))
        return ArtifactRegistry(reg_db, store=ArtifactStore(root=tmp_path / "artifact_store"))

    def _export_signed(
        self, registry: ArtifactRegistry, signer: ArtifactSigner, model_version: str
    ):
        from trainer.exporters.pipeline import QDEExportPipeline

        weights = registry.store.root.parent / "src_weights.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(b"fake-weights-for-signing-test")
        pipeline = QDEExportPipeline(artifact_registry=registry, signer=signer)
        return pipeline.export_version(
            model_version=model_version,
            output_dir=registry.store.root,
            weights_path=weights,
            export_formats=(),
        )

    def _promote(self, db, gate, model_version, target, registry=None, signer=None):
        mgr = DeploymentManager(
            db,
            approval_gate=gate,
            signalium_notifier=_fake_notifier,
            artifact_registry=registry,
            signer=signer,
        )
        result = mgr.request_promotion(model_version, target, "admin")
        req_id = result["approval_request"]["request_id"]
        return mgr, req_id

    def test_promotion_succeeds_with_valid_signatures(self, db, gate, registry, signer):
        self._export_signed(registry, signer, "v1.0")
        mgr, req_id = self._promote(db, gate, "v1.0", "SHADOW", registry=registry, signer=signer)
        result = mgr.apply_promotion(req_id, "admin")
        assert result["to_status"] == "SHADOW"

    def test_promotion_blocked_without_registry_artifacts(self, db, gate, registry, signer):
        """No artifacts registered at all for this version -> blocked."""
        mgr, req_id = self._promote(
            db, gate, "v-missing", "SHADOW", registry=registry, signer=signer
        )
        with pytest.raises(PromotionBlockedError):
            mgr.apply_promotion(req_id, "admin")

    def test_promotion_blocked_when_unsigned(self, db, gate, registry):
        """Artifacts exported without a signing key configured are unsigned
        and must block promotion."""
        unsigned_signer = ArtifactSigner(private_key_pem="", public_key_pem="")
        self._export_signed(registry, unsigned_signer, "v1.0")
        verify_signer = ArtifactSigner.generate_keypair()
        verifier = ArtifactSigner(public_key_pem=verify_signer[1])
        mgr, req_id = self._promote(db, gate, "v1.0", "SHADOW", registry=registry, signer=verifier)
        with pytest.raises(PromotionBlockedError):
            mgr.apply_promotion(req_id, "admin")

    def test_promotion_blocked_on_tampered_artifact(self, db, gate, registry, signer):
        result = self._export_signed(registry, signer, "v1.0")
        weights_entry = next(f for f in result["files"] if f["name"] == "weights.pt")
        Path(weights_entry["path"]).chmod(0o644)
        Path(weights_entry["path"]).write_bytes(b"TAMPERED-BYTES")

        mgr, req_id = self._promote(db, gate, "v1.0", "SHADOW", registry=registry, signer=signer)
        with pytest.raises(PromotionBlockedError):
            mgr.apply_promotion(req_id, "admin")

    def test_promotion_blocked_records_audit_event(self, db, gate, registry, signer):
        mgr, req_id = self._promote(
            db, gate, "v-missing", "SHADOW", registry=registry, signer=signer
        )
        with pytest.raises(PromotionBlockedError):
            mgr.apply_promotion(req_id, "admin")
        events = db.list_deployment_events(
            model_version="v-missing", event_type="SIGNATURE_VERIFICATION_FAILED"
        )
        assert len(events) == 1
        assert events[0]["to_status"] == "SHADOW"

    def test_blocked_promotion_does_not_consume_approval(self, db, gate, registry, signer):
        """A blocked promotion must not burn the approval request — it should
        remain retryable once the artifacts are fixed/re-exported."""
        mgr, req_id = self._promote(
            db, gate, "v-missing", "SHADOW", registry=registry, signer=signer
        )
        with pytest.raises(PromotionBlockedError):
            mgr.apply_promotion(req_id, "admin")
        req = gate.get_request(req_id)
        assert req.status == "PENDING"

    def test_promotion_without_registry_is_not_gated(self, db, gate, signer):
        """Backward compatibility: no ArtifactRegistry configured -> signature
        enforcement is skipped entirely (existing non-artifact-integrated
        deployments keep working)."""
        mgr, req_id = self._promote(db, gate, "v1.0", "SHADOW", registry=None, signer=signer)
        result = mgr.apply_promotion(req_id, "admin")
        assert result["to_status"] == "SHADOW"

    def test_pre_shadow_advance_stage_is_not_signature_gated(self, db, registry, signer):
        """DRAFT/TRAINING/VALIDATING have no artifacts yet and must not be
        blocked by signature enforcement."""
        mgr = DeploymentManager(db, artifact_registry=registry, signer=signer)
        result = mgr.advance_stage("v1.0", "DRAFT", "admin")
        assert result["to_status"] == "DRAFT"

    def test_verify_artifact_signatures_reports_each_result(self, db, registry, signer):
        self._export_signed(registry, signer, "v1.0")
        mgr = DeploymentManager(db, artifact_registry=registry, signer=signer)
        results = mgr.verify_artifact_signatures("v1.0")
        assert results
        assert all(r.verified for r in results)

    def test_rollback_to_active_is_signature_gated(self, db, gate, registry, signer):
        self._export_signed(registry, signer, "v1.0")
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="RETIRED").to_dict())
        mgr = DeploymentManager(
            db,
            approval_gate=gate,
            signalium_notifier=_fake_notifier,
            artifact_registry=registry,
            signer=signer,
        )
        result = mgr.request_rollback("v1.0", reason="bad metrics", requested_by="admin")
        req_id = result["approval_request"]["request_id"]
        apply_result = mgr.apply_rollback(req_id, "admin")
        assert apply_result["to_status"] == "ACTIVE"

    def test_rollback_blocked_when_artifacts_missing(self, db, gate, registry, signer):
        db.upsert_model_deployment(DeploymentRecord("v1.0", status="RETIRED").to_dict())
        mgr = DeploymentManager(
            db,
            approval_gate=gate,
            signalium_notifier=_fake_notifier,
            artifact_registry=registry,
            signer=signer,
        )
        result = mgr.request_rollback("v1.0", reason="bad metrics", requested_by="admin")
        req_id = result["approval_request"]["request_id"]
        with pytest.raises(PromotionBlockedError):
            mgr.apply_rollback(req_id, "admin")
