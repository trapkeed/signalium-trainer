"""S30-spec — Execution-Backend Abstraction (local_worker / runpod_worker).

The training engine must remain execution-backend agnostic. This module
defines the minimal ``ExecutionBackend`` protocol both backends implement,
plus the two concrete backends named in the spec. Neither backend knows
anything about RunPod API keys — ``RunPodWorkerBackend`` only ever receives
a pre-issued, pre-signed ``RunPodCredentials`` bundle (see
``execution_credentials.py``) and rejects it outright if expired or
unsigned.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.infra.execution_credentials import RunPodCredentials


@dataclass
class JobResult:
    job_id: str
    backend: str
    status: str  # "SUCCEEDED" | "FAILED"
    started_at: float
    finished_at: float
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "backend": self.backend,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output": self.output,
            "error": self.error,
        }


class ExecutionBackend(Protocol):
    """Every execution backend implements this surface. The training engine
    calls only these three methods and never branches on backend type."""

    def name(self) -> str: ...

    def health_check(self) -> dict[str, Any]: ...

    def submit(self, job_fn: Callable[[], dict[str, Any]], *, job_name: str = "") -> JobResult: ...


class LocalWorkerBackend:
    """Runs the job function synchronously in-process. Used for
    development, tests, and any environment with no external compute
    attached."""

    def name(self) -> str:
        return "local_worker"

    def health_check(self) -> dict[str, Any]:
        return {
            "backend": self.name(),
            "healthy": True,
            "detail": "in-process execution always available",
        }

    def submit(self, job_fn: Callable[[], dict[str, Any]], *, job_name: str = "") -> JobResult:
        job_id = f"local-{uuid.uuid4().hex[:12]}"
        started_at = time.time()
        try:
            output = job_fn()
            return JobResult(
                job_id=job_id,
                backend=self.name(),
                status="SUCCEEDED",
                started_at=started_at,
                finished_at=time.time(),
                output=output or {},
            )
        except Exception as exc:  # noqa: BLE001 — job failures are data, not process crashes
            logger.error(f"LocalWorkerBackend: job {job_name!r} failed: {exc!r}")
            return JobResult(
                job_id=job_id,
                backend=self.name(),
                status="FAILED",
                started_at=started_at,
                finished_at=time.time(),
                error=str(exc),
            )


class RunPodWorkerBackend:
    """Dispatches the job to a remote RunPod worker using a pre-issued
    credential bundle. Trainer never authenticates to RunPod directly with
    its own key — it is handed a scoped, expiring credential per job by
    Signalium and must validate it before every dispatch.

    ``dispatch_fn`` is the actual network call (kept injectable so this
    class has no hard dependency on any particular HTTP client, and so
    tests can run fully offline). It receives ``(credentials, job_name)``
    and must return a result dict; this class only handles the
    credential-validation gate and result bookkeeping around that call.
    """

    def __init__(
        self,
        credentials: RunPodCredentials,
        shared_secret: str,
        dispatch_fn: Callable[[RunPodCredentials, str], dict[str, Any]] | None = None,
    ) -> None:
        self._credentials = credentials
        self._shared_secret = shared_secret
        self._dispatch_fn = dispatch_fn

    def name(self) -> str:
        return "runpod_worker"

    def health_check(self) -> dict[str, Any]:
        try:
            self._credentials.assert_usable(self._shared_secret)
            return {"backend": self.name(), "healthy": True, "pod_id": self._credentials.pod_id}
        except Exception as exc:  # noqa: BLE001
            return {"backend": self.name(), "healthy": False, "detail": str(exc)}

    def submit(self, job_fn: Callable[[], dict[str, Any]], *, job_name: str = "") -> JobResult:
        job_id = f"runpod-{uuid.uuid4().hex[:12]}"
        started_at = time.time()
        try:
            self._credentials.assert_usable(self._shared_secret)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"RunPodWorkerBackend: credential rejected for job {job_name!r}: {exc!r}")
            return JobResult(
                job_id=job_id,
                backend=self.name(),
                status="FAILED",
                started_at=started_at,
                finished_at=time.time(),
                error=str(exc),
            )
        if self._dispatch_fn is None:
            return JobResult(
                job_id=job_id,
                backend=self.name(),
                status="FAILED",
                started_at=started_at,
                finished_at=time.time(),
                error="no dispatch_fn configured — RunPodWorkerBackend cannot reach the pod without one",
            )
        try:
            output = self._dispatch_fn(self._credentials, job_name)
            return JobResult(
                job_id=job_id,
                backend=self.name(),
                status="SUCCEEDED",
                started_at=started_at,
                finished_at=time.time(),
                output=output or {},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"RunPodWorkerBackend: job {job_name!r} failed: {exc!r}")
            return JobResult(
                job_id=job_id,
                backend=self.name(),
                status="FAILED",
                started_at=started_at,
                finished_at=time.time(),
                error=str(exc),
            )


def select_backend(
    backend_name: str,
    *,
    runpod_credentials: RunPodCredentials | None = None,
    runpod_shared_secret: str = "",
    runpod_dispatch_fn: Callable[[RunPodCredentials, str], dict[str, Any]] | None = None,
) -> ExecutionBackend:
    """Factory so callers configure a backend by name (e.g. from config)
    without importing backend classes directly, keeping the training
    engine's dependency on this module to a single call site."""
    if backend_name == "local_worker":
        return LocalWorkerBackend()
    if backend_name == "runpod_worker":
        if runpod_credentials is None:
            raise ValueError("runpod_worker backend requires runpod_credentials")
        return RunPodWorkerBackend(runpod_credentials, runpod_shared_secret, runpod_dispatch_fn)
    raise ValueError(
        f"Unknown execution backend {backend_name!r} — expected 'local_worker' or 'runpod_worker'"
    )
