"""RunPod Integration — job lifecycle management.

Trainer never holds a RunPod API key of its own (see
``execution_credentials.py`` for why). This module adds the missing
higher-level job-lifecycle surface named in the spec on top of the existing
``RunPodWorkerBackend``/``RunPodCredentials`` primitives: job creation with
GPU selection, status monitoring, artifact retrieval, checkpoint sync, and
cleanup.

Every network call (create/poll/download/sync/terminate) is injected via a
``RunPodClient`` protocol so this module has no hard dependency on the
RunPod SDK or any particular HTTP client, and is fully testable offline —
consistent with the ``dispatch_fn`` injection pattern in
``execution_backend.py``. With no client configured, every method raises
``RunPodNotConfigured`` rather than silently no-op'ing, per the S31 stub
convention (see ``market_data/provider_interface.py``).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.infra.execution_credentials import RunPodCredentials


class RunPodError(Exception):
    """Base class for RunPod job-management failures."""


class RunPodNotConfigured(RunPodError):
    """No RunPodClient wired up yet — permanent-until-configured, distinct
    from a transient job/network failure."""


class JobNotFoundError(RunPodError):
    pass


class GpuUnavailableError(RunPodError):
    pass


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TERMINATED = "TERMINATED"


#: GPU types Trainer knows how to request, in preference order for
#: "cheapest that fits" auto-selection. Values are RunPod's own type
#: identifiers; kept as data (not an enum) since RunPod's catalog changes
#: independently of this codebase.
GPU_PREFERENCE_ORDER: list[str] = [
    "NVIDIA RTX A4000",
    "NVIDIA RTX A5000",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA H100 PCIe",
]


@dataclass
class GpuOption:
    gpu_type: str
    available_count: int
    price_per_hr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "available_count": self.available_count,
            "price_per_hr": self.price_per_hr,
        }


def select_gpu(
    options: list[GpuOption], *, min_count: int = 1, max_price_per_hr: float | None = None
) -> GpuOption:
    """Pick the cheapest available GPU satisfying the constraints, breaking
    ties by ``GPU_PREFERENCE_ORDER`` so selection is deterministic across
    runs with identical pricing."""
    eligible = [
        o
        for o in options
        if o.available_count >= min_count
        and (max_price_per_hr is None or o.price_per_hr <= max_price_per_hr)
    ]
    if not eligible:
        raise GpuUnavailableError(
            f"No GPU option satisfies min_count={min_count} max_price_per_hr={max_price_per_hr!r} "
            f"(offered: {[o.gpu_type for o in options]})"
        )

    def sort_key(o: GpuOption) -> tuple[float, int]:
        try:
            pref_rank = GPU_PREFERENCE_ORDER.index(o.gpu_type)
        except ValueError:
            pref_rank = len(GPU_PREFERENCE_ORDER)
        return (o.price_per_hr, pref_rank)

    return min(eligible, key=sort_key)


@dataclass
class RunPodJob:
    job_id: str
    pod_id: str
    gpu_type: str
    status: JobStatus
    created_at: float
    updated_at: float
    checkpoint_uri: str | None = None
    artifact_uris: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "pod_id": self.pod_id,
            "gpu_type": self.gpu_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "checkpoint_uri": self.checkpoint_uri,
            "artifact_uris": self.artifact_uris,
            "error": self.error,
        }


class RunPodClient(Protocol):
    """Injected network layer. Implementations wrap the real RunPod API (or
    a test double); this module never talks HTTP directly."""

    def list_gpu_options(self) -> list[GpuOption]: ...

    def create_pod(
        self, *, gpu_type: str, credentials: RunPodCredentials, job_name: str
    ) -> str: ...

    def poll_status(self, pod_id: str) -> JobStatus: ...

    def fetch_artifacts(self, pod_id: str, destination: str) -> list[str]: ...

    def sync_checkpoint(self, pod_id: str, checkpoint_uri: str) -> str: ...

    def terminate_pod(self, pod_id: str) -> None: ...


class RunPodJobManager:
    """Tracks the job lifecycle in-process: creation -> monitoring ->
    artifact retrieval / checkpoint sync -> cleanup. Mirrors the
    fail-closed-until-configured convention used throughout S30/S31: every
    method raises ``RunPodNotConfigured`` until a real ``RunPodClient`` is
    supplied.
    """

    def __init__(self, client: RunPodClient | None = None, shared_secret: str = "") -> None:
        self._client = client
        self._shared_secret = shared_secret
        self._jobs: dict[str, RunPodJob] = {}

    def is_configured(self) -> bool:
        return self._client is not None

    def _require_client(self) -> RunPodClient:
        if self._client is None:
            raise RunPodNotConfigured("RunPodJobManager has no RunPodClient configured yet")
        return self._client

    # ── Job creation with GPU selection ─────────────────────────────────────

    def create_job(
        self,
        credentials: RunPodCredentials,
        *,
        job_name: str = "",
        min_gpu_count: int = 1,
        max_price_per_hr: float | None = None,
    ) -> RunPodJob:
        client = self._require_client()
        credentials.assert_usable(self._shared_secret)

        options = client.list_gpu_options()
        chosen = select_gpu(options, min_count=min_gpu_count, max_price_per_hr=max_price_per_hr)
        pod_id = client.create_pod(
            gpu_type=chosen.gpu_type, credentials=credentials, job_name=job_name
        )

        job_id = f"runpod-job-{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = RunPodJob(
            job_id=job_id,
            pod_id=pod_id,
            gpu_type=chosen.gpu_type,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        self._jobs[job_id] = job
        logger.info(f"RunPodJobManager: created job {job_id} on pod {pod_id} ({chosen.gpu_type})")
        return job

    # ── Monitoring ───────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> RunPodJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Unknown RunPod job {job_id!r}")
        return job

    def refresh_status(self, job_id: str) -> RunPodJob:
        client = self._require_client()
        job = self.get_job(job_id)
        job.status = client.poll_status(job.pod_id)
        job.updated_at = time.time()
        return job

    def list_jobs(self, status: JobStatus | None = None) -> list[RunPodJob]:
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    # ── Artifact retrieval / checkpoint sync ────────────────────────────────

    def fetch_artifacts(self, job_id: str, destination: str) -> list[str]:
        client = self._require_client()
        job = self.get_job(job_id)
        uris = client.fetch_artifacts(job.pod_id, destination)
        job.artifact_uris = uris
        job.updated_at = time.time()
        return uris

    def sync_checkpoint(self, job_id: str, checkpoint_uri: str) -> str:
        client = self._require_client()
        job = self.get_job(job_id)
        synced_uri = client.sync_checkpoint(job.pod_id, checkpoint_uri)
        job.checkpoint_uri = synced_uri
        job.updated_at = time.time()
        return synced_uri

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup(self, job_id: str) -> None:
        client = self._require_client()
        job = self.get_job(job_id)
        try:
            client.terminate_pod(job.pod_id)
        except (
            Exception
        ) as exc:  # noqa: BLE001 — cleanup must not raise past this point; record and move on
            job.error = f"cleanup failed: {exc!r}"
            logger.error(f"RunPodJobManager: cleanup failed for job {job_id}: {exc!r}")
        job.status = JobStatus.TERMINATED
        job.updated_at = time.time()

    def status_summary(self) -> dict[str, Any]:
        return {
            "configured": self.is_configured(),
            "total_jobs": len(self._jobs),
            "by_status": {
                status.value: len([j for j in self._jobs.values() if j.status == status])
                for status in JobStatus
            },
        }
