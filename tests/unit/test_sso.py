"""S24D — Single Sign-On Service Provider tests.

Covers:
  - JWT validation edge cases (bad signature, expired, stale iat, unknown
    role, missing claims, wrong algorithm, malformed token, misconfigured
    server).
  - Session establishment / reuse via SessionManager.
  - Audit log content (success + rejection rows) via AuditLogger.
  - Capability enforcement per role, including SuperAdmin/Admin bypass and
    Auditor/Founder being blocked from every mutating capability.
  - End-to-end enforcement against the real server app for a representative
    mutating endpoint per role.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import pytest

from tests.unit._sso_test_utils import DEFAULT_TEST_JWT_SECRET, make_signalium_jwt
from trainer.research_lab.rl_database import ResearchLabDatabase
from trainer.research_lab.session_manager import SessionManager
from trainer.research_lab.audit_log import AuditLogger
from trainer.research_lab.sso import (
    Capability,
    MalformedToken,
    MissingClaim,
    MissingToken,
    Role,
    ServerMisconfigured,
    SignaliumIdentity,
    StaleRequestTimestamp,
    TokenExpired,
    UnknownRole,
    ROLE_CAPABILITIES,
    has_capability,
    validate_signalium_jwt,
)


# ── JWT validation ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _jwt_env(monkeypatch):
    monkeypatch.setenv("SIGNALIUM_JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("SIGNALIUM_JWT_SECRET", DEFAULT_TEST_JWT_SECRET)
    monkeypatch.setenv("SIGNALIUM_JWT_MAX_SKEW_S", "300")
    monkeypatch.delenv("SIGNALIUM_ISSUER", raising=False)


class TestJWTValidation:
    def test_valid_token_round_trips(self):
        token = make_signalium_jwt(role="QuantResearcher", sub="alice")
        identity = validate_signalium_jwt(token)
        assert identity.admin_id == "alice"
        assert identity.role == Role.QUANT_RESEARCHER

    def test_missing_token(self):
        with pytest.raises(MissingToken):
            validate_signalium_jwt("")

    def test_malformed_token_structure(self):
        with pytest.raises(MalformedToken):
            validate_signalium_jwt("not-a-jwt")

    def test_bad_signature_rejected(self):
        token = make_signalium_jwt(secret="wrong-secret")
        with pytest.raises(Exception) as exc_info:
            validate_signalium_jwt(token)
        assert "signature" in str(exc_info.value).lower()

    def test_expired_token_rejected(self):
        now = time.time()
        token = make_signalium_jwt(now=now - 1000, exp_offset_s=1.0)
        with pytest.raises(TokenExpired):
            validate_signalium_jwt(token, now=now)

    def test_stale_request_timestamp_rejected(self):
        now = time.time()
        # iat far in the past, but exp still valid — must fail the skew check.
        token = make_signalium_jwt(now=now - 10_000, exp_offset_s=20_000)
        with pytest.raises(StaleRequestTimestamp):
            validate_signalium_jwt(token, now=now)

    def test_futuristic_request_timestamp_rejected(self):
        now = time.time()
        token = make_signalium_jwt(now=now + 10_000, exp_offset_s=20_000)
        with pytest.raises(StaleRequestTimestamp):
            validate_signalium_jwt(token, now=now)

    def test_unknown_role_rejected(self):
        token = make_signalium_jwt(role="NotARealRole")
        with pytest.raises(UnknownRole):
            validate_signalium_jwt(token)

    def test_missing_claim_rejected(self):
        # Hand-build a token missing the 'role' claim.
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
        ).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "x", "iat": time.time(), "exp": time.time() + 60}).encode()
        ).rstrip(b"=").decode()
        import hashlib
        import hmac

        signing_input = f"{header}.{payload}".encode()
        sig = hmac.new(DEFAULT_TEST_JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        token = f"{header}.{payload}.{sig_b64}"
        with pytest.raises(MissingClaim):
            validate_signalium_jwt(token)

    def test_server_misconfigured_without_secret(self, monkeypatch):
        monkeypatch.delenv("SIGNALIUM_JWT_SECRET", raising=False)
        token = make_signalium_jwt()
        with pytest.raises(ServerMisconfigured):
            validate_signalium_jwt(token)

    def test_algorithm_mismatch_rejected(self):
        token = make_signalium_jwt()
        # Server configured for a different algorithm than the token claims.
        os.environ["SIGNALIUM_JWT_ALGORITHM"] = "RS256"
        os.environ["SIGNALIUM_JWT_PUBLIC_KEY"] = "irrelevant"
        try:
            with pytest.raises(Exception):
                validate_signalium_jwt(token)
        finally:
            os.environ["SIGNALIUM_JWT_ALGORITHM"] = "HS256"
            os.environ.pop("SIGNALIUM_JWT_PUBLIC_KEY", None)

    def test_jti_preserved_on_identity(self):
        token = make_signalium_jwt(jti="tok-123")
        identity = validate_signalium_jwt(token)
        assert identity.jti == "tok-123"


# ── Capability model ─────────────────────────────────────────────────────────


class TestCapabilityModel:
    def test_super_admin_and_admin_bypass_all(self):
        for role in (Role.SUPER_ADMIN, Role.ADMIN):
            for cap in Capability:
                assert has_capability(role, cap)

    def test_auditor_has_no_mutation_capabilities(self):
        mutating_caps = [c for c in Capability if c != Capability.UNRESTRICTED]
        for cap in mutating_caps:
            assert not has_capability(Role.AUDITOR, cap)

    def test_founder_has_no_mutation_capabilities(self):
        mutating_caps = [c for c in Capability if c != Capability.UNRESTRICTED]
        for cap in mutating_caps:
            assert not has_capability(Role.FOUNDER, cap)

    def test_quant_researcher_grants(self):
        for cap in (
            Capability.EXPERIMENT_CREATION,
            Capability.DATASET_MANAGEMENT,
            Capability.FEATURE_ENGINEERING,
            Capability.SHADOW_TESTING,
            Capability.RESEARCH_WORKFLOWS,
        ):
            assert has_capability(Role.QUANT_RESEARCHER, cap)
        assert not has_capability(Role.QUANT_RESEARCHER, Capability.APPROVE_DEPLOYMENTS)

    def test_ml_researcher_grants(self):
        for cap in (
            Capability.TRAINING_RUNS,
            Capability.VALIDATION,
            Capability.ARCHITECTURE_RESEARCH,
            Capability.RETRAINING_WORKFLOWS,
        ):
            assert has_capability(Role.ML_RESEARCHER, cap)
        assert not has_capability(Role.ML_RESEARCHER, Capability.APPROVE_PROMOTIONS)

    def test_deployment_approver_grants(self):
        for cap in (
            Capability.APPROVE_PROMOTIONS,
            Capability.APPROVE_DEPLOYMENTS,
            Capability.APPROVE_ROLLBACKS,
        ):
            assert has_capability(Role.DEPLOYMENT_APPROVER, cap)
        assert not has_capability(Role.DEPLOYMENT_APPROVER, Capability.TRAINING_RUNS)

    def test_every_role_present_in_table(self):
        for role in Role:
            assert role in ROLE_CAPABILITIES


# ── Session establishment ────────────────────────────────────────────────────


class TestSessionManager:
    @pytest.fixture()
    def db(self, tmp_path: Path) -> ResearchLabDatabase:
        return ResearchLabDatabase(str(tmp_path / "sso_sessions.db"))

    def _identity(self, jti: str | None = "jti-1", expires_at: float | None = None) -> SignaliumIdentity:
        now = time.time()
        return SignaliumIdentity(
            admin_id="alice",
            role=Role.QUANT_RESEARCHER,
            originating_service="signalium",
            issued_at=now,
            expires_at=expires_at if expires_at is not None else now + 3600,
            jti=jti,
        )

    def test_establish_creates_session(self, db):
        mgr = SessionManager(db)
        record = mgr.establish(self._identity())
        assert record["admin_id"] == "alice"
        assert record["role"] == "QuantResearcher"

    def test_repeated_token_reuses_session(self, db):
        mgr = SessionManager(db)
        first = mgr.establish(self._identity(jti="jti-dup"))
        second = mgr.establish(self._identity(jti="jti-dup"))
        assert first["session_id"] == second["session_id"]
        assert len(db.list_research_sessions(admin_id="alice")) == 1

    def test_expired_session_is_replaced(self, db):
        mgr = SessionManager(db)
        expired = mgr.establish(self._identity(jti="jti-exp", expires_at=time.time() - 10))
        renewed = mgr.establish(self._identity(jti="jti-exp", expires_at=time.time() + 3600))
        # Same session_id (keyed by jti) but record content refreshed with a
        # future expiry.
        assert renewed["session_id"] == expired["session_id"]
        assert renewed["expires_at"] > time.time()

    def test_missing_jti_creates_distinct_sessions(self, db):
        mgr = SessionManager(db)
        first = mgr.establish(self._identity(jti=None))
        second = mgr.establish(self._identity(jti=None))
        assert first["session_id"] != second["session_id"]


# ── Audit logging ────────────────────────────────────────────────────────────


class TestAuditLogger:
    @pytest.fixture()
    def db(self, tmp_path: Path) -> ResearchLabDatabase:
        return ResearchLabDatabase(str(tmp_path / "sso_audit.db"))

    def _identity(self) -> SignaliumIdentity:
        now = time.time()
        return SignaliumIdentity(
            admin_id="bob",
            role=Role.ML_RESEARCHER,
            originating_service="signalium",
            issued_at=now,
            expires_at=now + 3600,
        )

    def test_log_success_contains_required_fields(self, db):
        audit = AuditLogger(db)
        record = audit.log_success(
            self._identity(),
            action="POST /research/experiments",
            model_version="v1.0",
            datasets=["ds-1"],
            experiments=["exp-1"],
        )
        assert record["admin_id"] == "bob"
        assert record["role"] == "MLResearcher"
        assert "timestamp" in record
        assert record["originating_service"] == "signalium"
        assert record["action"] == "POST /research/experiments"
        assert record["model_version"] == "v1.0"
        assert record["datasets"] == ["ds-1"]
        assert record["experiments"] == ["exp-1"]
        assert record["outcome"] == "SUCCESS"

        rows = db.list_research_audit_log(admin_id="bob")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "SUCCESS"

    def test_log_rejected_records_reason(self, db):
        audit = AuditLogger(db)
        audit.log_rejected(reason="Token expired", action="GET /research/health")
        rows = db.list_research_audit_log(outcome="REJECTED")
        assert len(rows) == 1
        assert rows[0]["extra"]["reason"] == "Token expired"

    def test_log_is_append_only(self, db):
        audit = AuditLogger(db)
        audit.log_success(self._identity(), action="a")
        audit.log_success(self._identity(), action="b")
        rows = db.list_research_audit_log(admin_id="bob")
        assert len(rows) == 2


# ── End-to-end enforcement against the real server ───────────────────────────


@pytest.fixture()
def sso_server_client(tmp_path: Path, monkeypatch):
    from starlette.testclient import TestClient

    monkeypatch.setenv("RESEARCH_LAB_ADMIN_TOKEN", "sso-e2e-token")
    monkeypatch.setenv("RESEARCH_LAB_DB_PATH", str(tmp_path / "sso_e2e.db"))
    monkeypatch.setenv("SIGNALIUM_JWT_SECRET", DEFAULT_TEST_JWT_SECRET)
    monkeypatch.setenv("SIGNALIUM_JWT_ALGORITHM", "HS256")

    for mod_name in list(sys.modules):
        if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
            del sys.modules[mod_name]
    import trainer.research_lab.research_lab_server as srv_mod

    return TestClient(srv_mod.app)


def _hdr(role: str) -> dict:
    return {
        "Authorization": "Bearer sso-e2e-token",
        "X-Signalium-Token": make_signalium_jwt(role=role),
    }


class TestServerCapabilityEnforcement:
    @staticmethod
    def _experiment_payload(name: str) -> dict:
        return {
            "name": name,
            "qde_version": "v1.0",
            "training_dataset": "ds-1",
            "researcher": "tester",
        }

    def test_missing_signalium_token_rejected(self, sso_server_client):
        resp = sso_server_client.post(
            "/research/experiments",
            headers={"Authorization": "Bearer sso-e2e-token"},
            json=self._experiment_payload("exp"),
        )
        assert resp.status_code == 401

    def test_quant_researcher_can_create_experiment(self, sso_server_client):
        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("QuantResearcher"),
            json=self._experiment_payload("exp-qr"),
        )
        assert resp.status_code == 200

    def test_auditor_cannot_create_experiment(self, sso_server_client):
        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("Auditor"),
            json=self._experiment_payload("exp-aud"),
        )
        assert resp.status_code == 403

    def test_founder_cannot_create_experiment(self, sso_server_client):
        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("Founder"),
            json=self._experiment_payload("exp-founder"),
        )
        assert resp.status_code == 403

    def test_auditor_can_read(self, sso_server_client):
        resp = sso_server_client.get("/research/experiments", headers=_hdr("Auditor"))
        assert resp.status_code == 200

    def test_ml_researcher_cannot_approve_promotion_gate(self, sso_server_client):
        create = sso_server_client.post(
            "/approvals",
            headers=_hdr("QuantResearcher"),
            json={
                "gate_type": "PROMOTION",
                "resource_id": "v-sso-1.0",
                "resource_type": "model_version",
                "requested_by": "qr",
                "action_summary": "promote",
            },
        )
        assert create.status_code == 200
        request_id = create.json()["approval_request"]["request_id"]

        resp = sso_server_client.post(
            f"/approvals/{request_id}/approve",
            headers=_hdr("MLResearcher"),
            json={"approved_by": "", "reason": "looks good"},
        )
        assert resp.status_code == 403

    def test_deployment_approver_can_approve_promotion_gate(self, sso_server_client):
        create = sso_server_client.post(
            "/approvals",
            headers=_hdr("QuantResearcher"),
            json={
                "gate_type": "PROMOTION",
                "resource_id": "v-sso-2.0",
                "resource_type": "model_version",
                "requested_by": "qr",
                "action_summary": "promote",
            },
        )
        assert create.status_code == 200
        request_id = create.json()["approval_request"]["request_id"]

        resp = sso_server_client.post(
            f"/approvals/{request_id}/approve",
            headers=_hdr("DeploymentApprover"),
            json={"approved_by": "", "reason": "looks good"},
        )
        assert resp.status_code == 200

    def test_super_admin_bypasses_all_capability_checks(self, sso_server_client):
        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("SuperAdmin"),
            json=self._experiment_payload("exp-sa"),
        )
        assert resp.status_code == 200

    def test_expired_token_rejected_end_to_end(self, sso_server_client):
        now = time.time()
        stale_token = make_signalium_jwt(role="QuantResearcher", now=now - 1000, exp_offset_s=1.0)
        resp = sso_server_client.post(
            "/research/experiments",
            headers={
                "Authorization": "Bearer sso-e2e-token",
                "X-Signalium-Token": stale_token,
            },
            json={"name": "exp-expired", "hypothesis": "h", "created_by": "x"},
        )
        assert resp.status_code == 401

    def test_successful_request_produces_audit_row(self, sso_server_client):
        import trainer.research_lab.research_lab_server as srv_mod

        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("QuantResearcher"),
            json=self._experiment_payload("exp-audit"),
        )
        assert resp.status_code == 200
        rows = srv_mod._db.list_research_audit_log(outcome="SUCCESS")
        assert any("research/experiments" in r["action"] for r in rows)

    def test_rejected_request_produces_audit_row(self, sso_server_client):
        import trainer.research_lab.research_lab_server as srv_mod

        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("Auditor"),
            json=self._experiment_payload("exp-rej"),
        )
        assert resp.status_code == 403
        # A capability-denied request still authenticates successfully (the
        # identity is valid) but must NOT be recorded as a SUCCESS audit
        # row — the middleware inspects the final status code and logs it
        # as REJECTED instead.
        rows = srv_mod._db.list_research_audit_log(outcome="REJECTED")
        assert any("research/experiments" in r["action"] and r["role"] == "Auditor" for r in rows)
        success_rows = srv_mod._db.list_research_audit_log(outcome="SUCCESS")
        assert not any(r["role"] == "Auditor" for r in success_rows)

    def test_audit_persistence_failure_does_not_mask_401(self, sso_server_client, monkeypatch):
        import trainer.research_lab.research_lab_server as srv_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated audit DB outage")

        monkeypatch.setattr(srv_mod._audit, "log_rejected", _boom)
        resp = sso_server_client.post(
            "/research/experiments",
            headers={"Authorization": "Bearer sso-e2e-token"},
            json=self._experiment_payload("exp-auditfail"),
        )
        # Auth failure (missing Signalium token) must still surface as 401,
        # not a 500 from the broken audit logger.
        assert resp.status_code == 401

    def test_audit_persistence_failure_does_not_mask_success(self, sso_server_client, monkeypatch):
        import trainer.research_lab.research_lab_server as srv_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated audit DB outage")

        monkeypatch.setattr(srv_mod._audit, "log_success", _boom)
        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("QuantResearcher"),
            json=self._experiment_payload("exp-successauditfail"),
        )
        assert resp.status_code == 200

    def test_non_sso_error_during_validation_yields_401_not_500(self, sso_server_client, monkeypatch):
        import trainer.research_lab.research_lab_server as srv_mod

        def _boom(*args, **kwargs):
            raise ValueError("simulated validator internals crash")

        monkeypatch.setattr(srv_mod, "validate_signalium_jwt", _boom)
        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("QuantResearcher"),
            json=self._experiment_payload("exp-validatorcrash"),
        )
        assert resp.status_code == 401

    def test_session_establish_failure_does_not_prevent_success(self, sso_server_client, monkeypatch):
        import trainer.research_lab.research_lab_server as srv_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated session DB outage")

        monkeypatch.setattr(srv_mod._session_mgr, "establish", _boom)
        resp = sso_server_client.post(
            "/research/experiments",
            headers=_hdr("QuantResearcher"),
            json=self._experiment_payload("exp-sessionfail"),
        )
        assert resp.status_code == 200

    def test_unhandled_route_exception_still_returns_500_and_is_audited(self, sso_server_client, monkeypatch):
        import trainer.research_lab.research_lab_server as srv_mod
        from starlette.testclient import TestClient

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated handler crash")

        monkeypatch.setattr(srv_mod._api, "create_experiment", _boom)
        # raise_server_exceptions=False so the TestClient behaves like a
        # real deployment: an unhandled exception becomes a 500 response
        # instead of propagating into the test process.
        non_raising_client = TestClient(srv_mod.app, raise_server_exceptions=False)
        resp = non_raising_client.post(
            "/research/experiments",
            headers=_hdr("QuantResearcher"),
            json=self._experiment_payload("exp-crash"),
        )
        assert resp.status_code == 500
        rows = srv_mod._db.list_research_audit_log(outcome="REJECTED")
        assert any("Unhandled exception" in r["extra"].get("reason", "") for r in rows)
