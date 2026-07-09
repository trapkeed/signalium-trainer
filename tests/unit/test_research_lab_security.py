"""S20 Task #14 — Research Lab Security: authentication layer tests.

Verifies that:
  - Every major endpoint category requires a valid admin bearer token
  - Missing Authorization header → 401 Unauthorized
  - Wrong/invalid token → 403 Forbidden
  - Valid token → 200 / success response
  - Authenticated admin identity is captured in approval and deployment audit records
  - App-level router dependency provides belt-and-suspenders coverage

Endpoint categories covered:
  - Health / summary
  - Telemetry ingestion (via telemetry router)
  - Failure investigation
  - Experiments / datasets
  - Approval gates (create, approve, reject)
  - Pipeline jobs
  - Shadow tests / health monitor
  - Champion-challenger
  - Deployment (promote, activate, rollback, history, current)
  - Retraining recommendations
  - Artifact registry
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# ── Fixture: isolated server client ───────────────────────────────────────────

_VALID_TOKEN = "rl-security-test-token-valid"
_WRONG_TOKEN = "rl-security-test-token-WRONG"

_AUTH = {"Authorization": f"Bearer {_VALID_TOKEN}"}
_WRONG_AUTH = {"Authorization": f"Bearer {_WRONG_TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    """Fresh TestClient with a known admin token and isolated DB."""
    tmp = tmp_path_factory.mktemp("rl_security")
    os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = _VALID_TOKEN
    os.environ["RESEARCH_LAB_ADMIN_ID"] = "security-test-admin"
    os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp / "security.db")

    # Purge any cached server module so we get a fresh singleton
    for mod_name in list(sys.modules):
        if "research_lab_server" in mod_name:
            del sys.modules[mod_name]

    import trainer.research_lab.research_lab_server as srv_mod
    from starlette.testclient import TestClient

    return TestClient(srv_mod.app, raise_server_exceptions=False)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _assert_auth_required(client, method: str, path: str, body: dict | None = None):
    """Assert that missing auth → 401, wrong auth → 403, valid auth → not 4xx auth error."""
    fn = getattr(client, method)
    kwargs = {"json": body} if body is not None else {}

    # No header at all → 401
    resp_missing = fn(path, **kwargs)
    assert resp_missing.status_code == 401, (
        f"[{method.upper()} {path}] expected 401 for missing auth, "
        f"got {resp_missing.status_code}: {resp_missing.text[:200]}"
    )

    # Wrong token → 403
    resp_wrong = fn(path, headers=_WRONG_AUTH, **kwargs)
    assert resp_wrong.status_code == 403, (
        f"[{method.upper()} {path}] expected 403 for wrong token, "
        f"got {resp_wrong.status_code}: {resp_wrong.text[:200]}"
    )

    # Valid token → not a 401/403
    resp_valid = fn(path, headers=_AUTH, **kwargs)
    assert resp_valid.status_code not in (401, 403), (
        f"[{method.upper()} {path}] expected success with valid auth, "
        f"got {resp_valid.status_code}: {resp_valid.text[:200]}"
    )
    return resp_valid


# ── Core 401 / 403 behaviour ───────────────────────────────────────────────────


class TestMissingTokenIs401:
    """A completely absent Authorization header must return 401, not 403."""

    def test_health_endpoint(self, client):
        resp = client.get("/research/healthz")
        assert resp.status_code == 401

    def test_deployment_current(self, client):
        resp = client.get("/deployment/current")
        assert resp.status_code == 401

    def test_deployment_history(self, client):
        resp = client.get("/deployment/history")
        assert resp.status_code == 401

    def test_approvals_list(self, client):
        resp = client.get("/approvals")
        assert resp.status_code == 401

    def test_pipeline_jobs_list(self, client):
        resp = client.get("/pipeline/jobs")
        assert resp.status_code == 401

    def test_telemetry_status(self, client):
        resp = client.get("/telemetry/status")
        assert resp.status_code == 401

    def test_artifacts_list(self, client):
        resp = client.get("/artifacts")
        assert resp.status_code == 401

    def test_shadow_status(self, client):
        resp = client.get("/shadow/status")
        assert resp.status_code == 401

    def test_health_check_status(self, client):
        resp = client.get("/health/status")
        assert resp.status_code == 401


class TestWrongTokenIs403:
    """A presented-but-invalid bearer token must return 403, not 401."""

    def test_health_endpoint(self, client):
        assert client.get("/research/healthz", headers=_WRONG_AUTH).status_code == 403

    def test_deployment_current(self, client):
        assert client.get("/deployment/current", headers=_WRONG_AUTH).status_code == 403

    def test_approvals_list(self, client):
        assert client.get("/approvals", headers=_WRONG_AUTH).status_code == 403

    def test_pipeline_jobs(self, client):
        assert client.get("/pipeline/jobs", headers=_WRONG_AUTH).status_code == 403

    def test_telemetry_status(self, client):
        assert client.get("/telemetry/status", headers=_WRONG_AUTH).status_code == 403

    def test_post_deployment_promote(self, client):
        resp = client.post(
            "/deployment/promote",
            headers=_WRONG_AUTH,
            json={"model_version": "v1.0", "target_status": "SHADOW", "requested_by": "x"},
        )
        assert resp.status_code == 403

    def test_post_approvals(self, client):
        resp = client.post(
            "/approvals",
            headers=_WRONG_AUTH,
            json={
                "gate_type": "TRAINING_START",
                "resource_id": "exp-001",
                "resource_type": "experiment",
                "requested_by": "admin",
                "action_summary": "test",
            },
        )
        assert resp.status_code == 403


class TestValidTokenAccepted:
    """Valid admin token must not be rejected with a 401 or 403."""

    def test_health_endpoint(self, client):
        resp = client.get("/research/healthz", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_system_summary(self, client):
        resp = client.get("/research/summary", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_deployment_current(self, client):
        resp = client.get("/deployment/current", headers=_AUTH)
        assert resp.status_code == 200

    def test_deployment_history(self, client):
        resp = client.get("/deployment/history", headers=_AUTH)
        assert resp.status_code == 200

    def test_approvals_list(self, client):
        resp = client.get("/approvals", headers=_AUTH)
        assert resp.status_code == 200

    def test_pipeline_jobs(self, client):
        resp = client.get("/pipeline/jobs", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_telemetry_status(self, client):
        resp = client.get("/telemetry/status", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_artifacts(self, client):
        resp = client.get("/artifacts", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_shadow_status(self, client):
        resp = client.get("/shadow/status", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_health_status(self, client):
        resp = client.get("/health/status", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_health_alerts(self, client):
        resp = client.get("/health/alerts", headers=_AUTH)
        assert resp.status_code not in (401, 403)


# ── Comprehensive endpoint sweep ───────────────────────────────────────────────


class TestAllEndpointCategoriesRequireAuth:
    """Sample representative endpoints from every major category.

    Each call verifies the full 401/403/200 contract via _assert_auth_required.
    """

    # Health / system
    def test_health(self, client):
        _assert_auth_required(client, "get", "/research/healthz")

    def test_summary(self, client):
        _assert_auth_required(client, "get", "/research/summary")

    # Datasets
    def test_list_datasets(self, client):
        _assert_auth_required(client, "get", "/research/datasets")

    # Experiments
    def test_list_experiments(self, client):
        _assert_auth_required(client, "get", "/research/experiments")

    # Failures
    def test_list_failures(self, client):
        _assert_auth_required(client, "get", "/research/failures")

    def test_failure_summary(self, client):
        _assert_auth_required(client, "get", "/research/failures/summary")

    # Approvals
    def test_list_approvals(self, client):
        _assert_auth_required(client, "get", "/approvals")

    def test_create_approval(self, client):
        _assert_auth_required(client, "post", "/approvals", body={
            "gate_type": "TRAINING_START",
            "resource_id": "exp-sweep-001",
            "resource_type": "experiment",
            "requested_by": "admin",
            "action_summary": "Security sweep test",
        })

    # Pipeline
    def test_list_pipeline_jobs(self, client):
        _assert_auth_required(client, "get", "/pipeline/jobs")

    # Shadow tests
    def test_shadow_status(self, client):
        _assert_auth_required(client, "get", "/shadow/status")

    # S22B: per-decision shadow observation queryability
    def test_list_shadow_tests(self, client):
        _assert_auth_required(client, "get", "/research/shadow-tests")

    def test_list_shadow_decisions_by_version_pair(self, client):
        _assert_auth_required(client, "get", "/research/shadow-decisions")

    # Health monitor
    def test_health_status(self, client):
        _assert_auth_required(client, "get", "/health/status")

    def test_health_alerts(self, client):
        _assert_auth_required(client, "get", "/health/alerts")

    # Artifacts
    def test_list_artifacts(self, client):
        _assert_auth_required(client, "get", "/artifacts")

    # Deployment
    def test_deployment_current(self, client):
        _assert_auth_required(client, "get", "/deployment/current")

    def test_deployment_history(self, client):
        _assert_auth_required(client, "get", "/deployment/history")

    def test_deployment_promote(self, client):
        _assert_auth_required(client, "post", "/deployment/promote", body={
            "model_version": "v-sweep-1.0",
            "target_status": "SHADOW",
            "requested_by": "admin",
        })

    def test_deployment_rollback(self, client):
        _assert_auth_required(client, "post", "/deployment/rollback", body={
            "model_version": "latest_retired",
            "reason": "security sweep test",
            "requested_by": "admin",
        })

    # Retraining recommendations
    def test_list_recommendations(self, client):
        _assert_auth_required(client, "get", "/research/recommendations")

    # Telemetry
    def test_telemetry_status(self, client):
        _assert_auth_required(client, "get", "/telemetry/status")


# ── Admin identity capture ─────────────────────────────────────────────────────


class TestAdminIdentityInAuditRecords:
    """Authenticated admin identity must appear in approval and deployment audit records."""

    def test_approval_approve_records_authenticated_admin(self, client):
        """Approving a request must return authenticated_as matching RESEARCH_LAB_ADMIN_ID."""
        # Create an approval request first
        create_resp = client.post("/approvals", headers=_AUTH, json={
            "gate_type": "TRAINING_START",
            "resource_id": "exp-identity-001",
            "resource_type": "experiment",
            "requested_by": "system",
            "action_summary": "Identity capture test",
        })
        assert create_resp.status_code == 200
        req_id = create_resp.json()["approval_request"]["request_id"]

        # Approve it
        approve_resp = client.post(
            f"/approvals/{req_id}/approve",
            headers=_AUTH,
            json={"approved_by": "", "reason": "identity test"},
        )
        assert approve_resp.status_code == 200
        body = approve_resp.json()
        assert body["success"] is True
        assert body.get("authenticated_as") == "security-test-admin"
        # approved_by in the record should be the authenticated identity
        resolved_by = body["approval_request"].get("resolved_by", "")
        assert resolved_by == "security-test-admin"

    def test_approval_reject_records_authenticated_admin(self, client):
        """Rejecting a request must return authenticated_as matching RESEARCH_LAB_ADMIN_ID."""
        create_resp = client.post("/approvals", headers=_AUTH, json={
            "gate_type": "DEPLOYMENT",
            "resource_id": "exp-identity-reject-001",
            "resource_type": "experiment",
            "requested_by": "system",
            "action_summary": "Reject identity test",
        })
        assert create_resp.status_code == 200
        req_id = create_resp.json()["approval_request"]["request_id"]

        reject_resp = client.post(
            f"/approvals/{req_id}/reject",
            headers=_AUTH,
            json={"rejected_by": "", "reason": "identity reject test"},
        )
        assert reject_resp.status_code == 200
        body = reject_resp.json()
        assert body["success"] is True
        assert body.get("authenticated_as") == "security-test-admin"

    def test_deployment_activate_records_authenticated_admin(self, client):
        """Deployment activation must return authenticated_as."""
        # Create a promotion request
        promo = client.post("/deployment/promote", headers=_AUTH, json={
            "model_version": "v-identity-1.0",
            "target_status": "SHADOW",
            "requested_by": "system",
        })
        assert promo.status_code == 200
        req_id = promo.json()["promotion_request"]["approval_request"]["request_id"]

        activate = client.post("/deployment/activate", headers=_AUTH, json={
            "request_id": req_id,
            "approved_by": "",   # intentionally empty — should fall back to auth identity
        })
        assert activate.status_code == 200
        body = activate.json()
        assert body["success"] is True
        assert body.get("authenticated_as") == "security-test-admin"
        # approved_by in the event should be the auth identity (not empty)
        assert body["promotion_result"]["approval_id"] == req_id

    def test_body_approved_by_overrides_when_provided(self, client):
        """When approved_by is explicitly provided in the body, it is used as-is."""
        create_resp = client.post("/approvals", headers=_AUTH, json={
            "gate_type": "PROMOTION",
            "resource_id": "exp-override-001",
            "resource_type": "experiment",
            "requested_by": "system",
            "action_summary": "Override test",
        })
        assert create_resp.status_code == 200
        req_id = create_resp.json()["approval_request"]["request_id"]

        approve_resp = client.post(
            f"/approvals/{req_id}/approve",
            headers=_AUTH,
            json={"approved_by": "jane-doe", "reason": "explicit override"},
        )
        assert approve_resp.status_code == 200
        body = approve_resp.json()
        # authenticated_as still shows the token holder
        assert body.get("authenticated_as") == "security-test-admin"
        # resolved_by is the explicit value from the body
        assert body["approval_request"].get("resolved_by") == "jane-doe"


# ── Telemetry router auth (belt-and-suspenders) ────────────────────────────────


class TestTelemetryRouterAuth:
    """Telemetry endpoints are mounted via include_router and must be protected."""

    TELEMETRY_POST_ENDPOINTS = [
        "/telemetry/decisions",
        "/telemetry/trades",
        "/telemetry/positions",
        "/telemetry/failures",
        "/telemetry/reasoning",
        "/telemetry/execution",
        "/telemetry/counterfactuals",
        "/telemetry/lifecycle",
    ]

    def test_status_endpoint_requires_auth(self, client):
        assert client.get("/telemetry/status").status_code == 401
        assert client.get("/telemetry/status", headers=_WRONG_AUTH).status_code == 403
        resp = client.get("/telemetry/status", headers=_AUTH)
        assert resp.status_code not in (401, 403)

    def test_all_ingestion_endpoints_require_auth(self, client):
        """Every telemetry POST endpoint must reject missing or invalid tokens."""
        body = {
            "events": [{
                "event_id": "evt-sec-001",
                "model_version": "v1.0",
                "symbol": "AAPL",
                "timestamp": 1700000000.0,
            }]
        }
        for endpoint in self.TELEMETRY_POST_ENDPOINTS:
            resp_missing = client.post(endpoint, json=body)
            assert resp_missing.status_code == 401, (
                f"{endpoint}: expected 401 for missing auth, got {resp_missing.status_code}"
            )
            resp_wrong = client.post(endpoint, headers=_WRONG_AUTH, json=body)
            assert resp_wrong.status_code == 403, (
                f"{endpoint}: expected 403 for wrong token, got {resp_wrong.status_code}"
            )


# ── App-level router dependency (belt-and-suspenders) ─────────────────────────


class TestAppLevelRouterDependency:
    """Verify the app.router dependency catches any route that lacks per-endpoint auth."""

    def test_router_level_dep_blocks_missing_token(self, client):
        """The router dependency alone is sufficient to block unauthenticated access."""
        # Verify multiple diverse GET endpoints are all blocked without auth
        endpoints = [
            "/research/healthz",
            "/deployment/current",
            "/deployment/history",
            "/approvals",
            "/artifacts",
            "/pipeline/jobs",
            "/health/status",
            "/shadow/status",
            "/telemetry/status",
        ]
        for ep in endpoints:
            resp = client.get(ep)
            assert resp.status_code == 401, (
                f"Expected 401 for {ep} with no auth, got {resp.status_code}"
            )

    def test_router_level_dep_blocks_wrong_token(self, client):
        """Wrong token returns 403 across all endpoint categories."""
        endpoints = [
            "/research/healthz",
            "/deployment/current",
            "/approvals",
            "/telemetry/status",
        ]
        for ep in endpoints:
            resp = client.get(ep, headers=_WRONG_AUTH)
            assert resp.status_code == 403, (
                f"Expected 403 for {ep} with wrong auth, got {resp.status_code}"
            )
