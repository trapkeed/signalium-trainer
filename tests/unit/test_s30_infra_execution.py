"""Tests for trainer.infra: RunPod execution credentials + backend abstraction."""

from __future__ import annotations

import time

import pytest

from trainer.infra.execution_backend import (
    LocalWorkerBackend,
    RunPodWorkerBackend,
    select_backend,
)
from trainer.infra.execution_credentials import (
    CredentialExpired,
    InvalidCredentialSignature,
    MissingCredentialField,
    RunPodCredentials,
)

SECRET = "shared-secret"


def _valid_payload(expires_in: float = 3600.0) -> dict:
    return {
        "pod_id": "pod-123",
        "endpoint": "https://pod-123.runpod.io",
        "worker_token": "tok-abc",
        "expiration_timestamp": time.time() + expires_in,
        "storage_credentials": {"key": "val"},
        "artifact_destination": "s3://bucket/path",
    }


def _signed_credentials(expires_in: float = 3600.0) -> RunPodCredentials:
    payload = _valid_payload(expires_in)
    sig = RunPodCredentials.sign(payload, SECRET)
    return RunPodCredentials.from_dict({**payload, "signature": sig})


class TestRunPodCredentials:
    def test_from_dict_missing_field_raises(self):
        with pytest.raises(MissingCredentialField):
            RunPodCredentials.from_dict({"pod_id": "p1"})

    def test_valid_signed_credential_passes_assert_usable(self):
        creds = _signed_credentials()
        creds.assert_usable(SECRET)  # should not raise

    def test_expired_credential_raises(self):
        creds = _signed_credentials(expires_in=-10.0)
        with pytest.raises(CredentialExpired):
            creds.assert_usable(SECRET)

    def test_unsigned_credential_raises(self):
        payload = _valid_payload()
        creds = RunPodCredentials.from_dict(payload)  # no signature
        with pytest.raises(InvalidCredentialSignature):
            creds.assert_usable(SECRET)

    def test_wrong_secret_raises(self):
        creds = _signed_credentials()
        with pytest.raises(InvalidCredentialSignature):
            creds.assert_usable("wrong-secret")

    def test_tampered_field_invalidates_signature(self):
        payload = _valid_payload()
        sig = RunPodCredentials.sign(payload, SECRET)
        tampered = RunPodCredentials.from_dict({**payload, "pod_id": "pod-999", "signature": sig})
        with pytest.raises(InvalidCredentialSignature):
            tampered.assert_usable(SECRET)


class TestLocalWorkerBackend:
    def test_name(self):
        assert LocalWorkerBackend().name() == "local_worker"

    def test_health_check_always_healthy(self):
        assert LocalWorkerBackend().health_check()["healthy"] is True

    def test_submit_success(self):
        backend = LocalWorkerBackend()
        result = backend.submit(lambda: {"ok": True}, job_name="test-job")
        assert result.status == "SUCCEEDED"
        assert result.output == {"ok": True}
        assert result.backend == "local_worker"

    def test_submit_failure_is_captured_not_raised(self):
        backend = LocalWorkerBackend()

        def boom():
            raise ValueError("nope")

        result = backend.submit(boom, job_name="failing-job")
        assert result.status == "FAILED"
        assert "nope" in result.error


class TestRunPodWorkerBackend:
    def test_health_check_rejects_expired_credentials(self):
        creds = _signed_credentials(expires_in=-5.0)
        backend = RunPodWorkerBackend(creds, SECRET)
        health = backend.health_check()
        assert health["healthy"] is False

    def test_health_check_accepts_valid_credentials(self):
        creds = _signed_credentials()
        backend = RunPodWorkerBackend(creds, SECRET)
        health = backend.health_check()
        assert health["healthy"] is True
        assert health["pod_id"] == "pod-123"

    def test_submit_rejects_expired_credentials_without_dispatch(self):
        creds = _signed_credentials(expires_in=-5.0)
        calls = []
        backend = RunPodWorkerBackend(creds, SECRET, dispatch_fn=lambda c, n: calls.append(n))
        result = backend.submit(lambda: {}, job_name="job-1")
        assert result.status == "FAILED"
        assert calls == []  # dispatch never invoked

    def test_submit_with_no_dispatch_fn_fails_cleanly(self):
        creds = _signed_credentials()
        backend = RunPodWorkerBackend(creds, SECRET)
        result = backend.submit(lambda: {}, job_name="job-2")
        assert result.status == "FAILED"
        assert "dispatch_fn" in result.error

    def test_submit_success_with_dispatch_fn(self):
        creds = _signed_credentials()
        backend = RunPodWorkerBackend(
            creds, SECRET, dispatch_fn=lambda c, n: {"job": n, "pod": c.pod_id}
        )
        result = backend.submit(lambda: {}, job_name="job-3")
        assert result.status == "SUCCEEDED"
        assert result.output == {"job": "job-3", "pod": "pod-123"}


class TestSelectBackend:
    def test_select_local_worker(self):
        backend = select_backend("local_worker")
        assert backend.name() == "local_worker"

    def test_select_runpod_worker_requires_credentials(self):
        with pytest.raises(ValueError):
            select_backend("runpod_worker")

    def test_select_runpod_worker_with_credentials(self):
        creds = _signed_credentials()
        backend = select_backend("runpod_worker", runpod_credentials=creds, runpod_shared_secret=SECRET)
        assert backend.name() == "runpod_worker"

    def test_select_unknown_backend_raises(self):
        with pytest.raises(ValueError):
            select_backend("magic_worker")
