"""Tests for trainer.infra.runpod_jobs — job lifecycle management."""

from __future__ import annotations

import time

import pytest

from trainer.infra.execution_credentials import RunPodCredentials
from trainer.infra.runpod_jobs import (
    GpuOption,
    GpuUnavailableError,
    JobNotFoundError,
    JobStatus,
    RunPodJobManager,
    RunPodNotConfigured,
    select_gpu,
)

SHARED_SECRET = "test-shared-secret"


def _credentials(expired: bool = False) -> RunPodCredentials:
    payload = {
        "pod_id": "pod-1",
        "endpoint": "https://example.runpod.io",
        "worker_token": "tok-1",
        "expiration_timestamp": time.time() - 10 if expired else time.time() + 3600,
        "artifact_destination": "s3://bucket/artifacts",
    }
    sig = RunPodCredentials.sign(payload, SHARED_SECRET)
    return RunPodCredentials.from_dict({**payload, "signature": sig})


class FakeRunPodClient:
    def __init__(self) -> None:
        self.terminated: list[str] = []

    def list_gpu_options(self):
        return [
            GpuOption("NVIDIA RTX A4000", 3, 0.3),
            GpuOption("NVIDIA A100 80GB PCIe", 1, 1.5),
        ]

    def create_pod(self, *, gpu_type, credentials, job_name):
        return f"pod-for-{job_name}"

    def poll_status(self, pod_id):
        return JobStatus.RUNNING

    def fetch_artifacts(self, pod_id, destination):
        return [f"{destination}/model.pt"]

    def sync_checkpoint(self, pod_id, checkpoint_uri):
        return f"synced://{checkpoint_uri}"

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


class TestUnconfigured:
    def test_create_job_raises_not_configured(self):
        manager = RunPodJobManager()
        with pytest.raises(RunPodNotConfigured):
            manager.create_job(_credentials(), job_name="train-1")

    def test_is_configured_false_by_default(self):
        assert RunPodJobManager().is_configured() is False


class TestGpuSelection:
    def test_selects_cheapest_eligible(self):
        options = [GpuOption("NVIDIA A100 80GB PCIe", 1, 1.5), GpuOption("NVIDIA RTX A4000", 3, 0.3)]
        assert select_gpu(options).gpu_type == "NVIDIA RTX A4000"

    def test_respects_min_count(self):
        options = [GpuOption("NVIDIA RTX A4000", 1, 0.3), GpuOption("NVIDIA A100 80GB PCIe", 5, 1.5)]
        assert select_gpu(options, min_count=2).gpu_type == "NVIDIA A100 80GB PCIe"

    def test_raises_when_none_eligible(self):
        options = [GpuOption("NVIDIA RTX A4000", 0, 0.3)]
        with pytest.raises(GpuUnavailableError):
            select_gpu(options, min_count=1)

    def test_respects_max_price(self):
        options = [GpuOption("NVIDIA A100 80GB PCIe", 5, 1.5), GpuOption("NVIDIA RTX A4000", 5, 0.3)]
        assert select_gpu(options, max_price_per_hr=0.5).gpu_type == "NVIDIA RTX A4000"


class TestJobLifecycle:
    def test_create_monitor_fetch_sync_cleanup(self):
        client = FakeRunPodClient()
        manager = RunPodJobManager(client=client, shared_secret=SHARED_SECRET)
        job = manager.create_job(_credentials(), job_name="train-1")
        assert job.status == JobStatus.PENDING
        assert job.gpu_type == "NVIDIA RTX A4000"

        refreshed = manager.refresh_status(job.job_id)
        assert refreshed.status == JobStatus.RUNNING

        uris = manager.fetch_artifacts(job.job_id, "s3://bucket/out")
        assert uris == ["s3://bucket/out/model.pt"]

        synced = manager.sync_checkpoint(job.job_id, "checkpoint.pt")
        assert synced == "synced://checkpoint.pt"

        manager.cleanup(job.job_id)
        assert manager.get_job(job.job_id).status == JobStatus.TERMINATED
        assert client.terminated == [job.pod_id]

    def test_create_job_rejects_expired_credentials(self):
        client = FakeRunPodClient()
        manager = RunPodJobManager(client=client, shared_secret=SHARED_SECRET)
        with pytest.raises(Exception):
            manager.create_job(_credentials(expired=True), job_name="train-1")

    def test_get_unknown_job_raises(self):
        manager = RunPodJobManager(client=FakeRunPodClient(), shared_secret=SHARED_SECRET)
        with pytest.raises(JobNotFoundError):
            manager.get_job("nonexistent")

    def test_list_jobs_filters_by_status(self):
        client = FakeRunPodClient()
        manager = RunPodJobManager(client=client, shared_secret=SHARED_SECRET)
        job = manager.create_job(_credentials(), job_name="train-1")
        assert manager.list_jobs(JobStatus.PENDING) == [job]
        assert manager.list_jobs(JobStatus.RUNNING) == []

    def test_status_summary_reflects_configured_and_counts(self):
        client = FakeRunPodClient()
        manager = RunPodJobManager(client=client, shared_secret=SHARED_SECRET)
        manager.create_job(_credentials(), job_name="train-1")
        summary = manager.status_summary()
        assert summary["configured"] is True
        assert summary["total_jobs"] == 1
        assert summary["by_status"]["PENDING"] == 1
