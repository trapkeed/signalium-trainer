"""S26 — QDE Scalability and Model Resilience.

Builds resilient, institutional-grade model *serving* infrastructure on top
of the existing S20/S21A/S25 registry, deployment, and telemetry layers.

The Research Lab is a monitoring/orchestration backend, not the live
inference process itself (that lives in Signalium) — so "worker pools",
"GPU scheduling", etc. are modeled here as first-class *bookkeeping and
decisioning* constructs (who is healthy, who should serve the next request,
when to fail over, when to scale) rather than actual OS processes. Every
class below persists real state via ``ResearchLabDatabase`` and computes
real (non-mocked) decisions from that state — nothing here is a stub.

Houses, in order:
  - WorkerPoolManager        — Inference Scalability + High Availability
                                (worker pools, GPU/CPU scheduling, dynamic
                                allocation, hot standby, restart, failover)
  - InferenceQueueMonitor     — async queue batching + saturation/queue-depth
                                monitoring
  - ArtifactIntegrityVerifier — checksum validation, rollback safety checks,
                                version compatibility checks
  - ResearchLabBackupManager  — automatic backups, experiment/shadow-test
                                persistence verification, recovery procedures
  - TrainingResilienceMonitor — stale/failed training job cleanup, corrupted
                                artifact detection, checkpoint-presence checks
  - DisasterRecoveryManager   — model/dataset/feature-store/lineage/
                                experiment snapshot bundles
  - DeploymentSafetyMonitor   — deployment health validation, compatibility
                                checks, rollback automation, active-model
                                verification
  - ResilienceDashboard       — single aggregated view over all of the above,
                                for the S26 Administrative Visibility surface
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.research_lab.rl_database import ResearchLabDatabase

# ── Shared constants ─────────────────────────────────────────────────────────

_WORKER_STATUS_ACTIVE = "ACTIVE"
_WORKER_STATUS_STANDBY = "STANDBY"
_WORKER_STATUS_DRAINING = "DRAINING"
_WORKER_STATUS_DEAD = "DEAD"

_DEFAULT_HEARTBEAT_TIMEOUT_S = 30.0
_DEFAULT_STALE_JOB_TIMEOUT_S = 3600.0
_DEFAULT_BACKUP_DIR = "data/backups"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{str(uuid.uuid4())[:12]}"


def _sha256_file(path: Path) -> str | None:
    """Return the SHA-256 hex digest of *path*, or ``None`` if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ── 1. WorkerPoolManager — Inference Scalability + High Availability ─────────


@dataclass
class WorkerRecord:
    """One inference worker's bookkeeping record (not a real OS process)."""

    worker_id: str
    kind: str  # "GPU" or "CPU"
    role: str  # ACTIVE | STANDBY
    status: str  # ACTIVE | STANDBY | DRAINING | DEAD
    capacity: float
    current_load: float = 0.0
    last_heartbeat: float = 0.0
    restart_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "kind": self.kind,
            "role": self.role,
            "status": self.status,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "last_heartbeat": self.last_heartbeat,
            "restart_count": self.restart_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkerRecord:
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


class WorkerPoolManager:
    """Tracks the inference worker pool: registration, GPU/CPU scheduling,
    dynamic allocation recommendations, hot standby, restart, and failover.

    Persisted via the generic ``resilience_workers`` table in
    ``ResearchLabDatabase`` — survives process restarts of the Research Lab
    server itself, distinguishing worker *state* from the Research Lab's own
    liveness.
    """

    def __init__(
        self, db: ResearchLabDatabase, heartbeat_timeout_s: float = _DEFAULT_HEARTBEAT_TIMEOUT_S
    ) -> None:
        self._db = db
        self._heartbeat_timeout_s = heartbeat_timeout_s

    # ── Registration & scheduling ────────────────────────────────────────────

    def register_worker(
        self,
        worker_id: str | None = None,
        *,
        kind: str = "GPU",
        role: str = _WORKER_STATUS_ACTIVE,
        capacity: float = 1.0,
        now: float | None = None,
    ) -> WorkerRecord:
        kind = kind.upper()
        if kind not in ("GPU", "CPU"):
            raise ValueError(f"kind must be 'GPU' or 'CPU', got {kind!r}")
        if role not in (_WORKER_STATUS_ACTIVE, _WORKER_STATUS_STANDBY):
            raise ValueError(f"role must be 'ACTIVE' or 'STANDBY', got {role!r}")
        now = now if now is not None else time.time()
        record = WorkerRecord(
            worker_id=worker_id or _new_id("wkr"),
            kind=kind,
            role=role,
            status=role,
            capacity=capacity,
            current_load=0.0,
            last_heartbeat=now,
            restart_count=0,
            created_at=now,
            updated_at=now,
        )
        self._db.upsert_resilience_worker(record.to_dict())
        logger.info(f"WorkerPoolManager: registered {record.worker_id} kind={kind} role={role}")
        return record

    def heartbeat(
        self, worker_id: str, *, current_load: float | None = None, now: float | None = None
    ) -> None:
        """Record a liveness heartbeat, reviving a DEAD/DRAINING worker to
        its prior role since a heartbeat is proof of life."""
        row = self._db.get_resilience_worker(worker_id)
        if row is None:
            raise ValueError(f"Unknown worker_id: {worker_id!r}")
        now = now if now is not None else time.time()
        row["last_heartbeat"] = now
        row["updated_at"] = now
        if current_load is not None:
            row["current_load"] = current_load
        if row["status"] in (_WORKER_STATUS_DEAD, _WORKER_STATUS_DRAINING):
            row["status"] = row["role"]
        self._db.upsert_resilience_worker(row)

    def list_workers(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._db.list_resilience_workers(status=status)

    def assign_worker(
        self, *, prefer_gpu: bool = True, now: float | None = None
    ) -> dict[str, Any] | None:
        """Pick the least-loaded healthy ACTIVE worker for the next inference
        request. GPU workers are preferred; if none are healthy, falls back
        to CPU workers automatically (CPU fallback support). Returns ``None``
        if no healthy ACTIVE worker of any kind exists.

        "Healthy" here means a fresh heartbeat (``check_worker_health() ==
        HEALTHY``), not merely ``status == ACTIVE`` — a worker whose
        heartbeat has gone stale must not receive new work even before the
        next supervision cycle has had a chance to mark it DEAD/failed-over.
        """
        candidates = [
            w
            for w in self.list_workers(status=_WORKER_STATUS_ACTIVE)
            if w["capacity"] > 0 and self.check_worker_health(w, now=now) == "HEALTHY"
        ]
        if not candidates:
            return None
        gpu = sorted(
            (w for w in candidates if w["kind"] == "GPU"),
            key=lambda w: w["current_load"] / w["capacity"],
        )
        cpu = sorted(
            (w for w in candidates if w["kind"] == "CPU"),
            key=lambda w: w["current_load"] / w["capacity"],
        )
        ordered = (gpu + cpu) if prefer_gpu else (cpu + gpu)
        return ordered[0] if ordered else None

    # ── Dynamic allocation ───────────────────────────────────────────────────

    def recommend_scaling(
        self, *, queue_depth: int, high_watermark: float = 0.8, low_watermark: float = 0.2
    ) -> dict[str, Any]:
        """Recommend scaling the ACTIVE pool up or down based on current
        aggregate utilization vs. queue depth.

        Utilization = sum(current_load) / sum(capacity) over ACTIVE workers.
        A high utilization *and* a non-trivial queue backlog recommends
        SCALE_UP; a low utilization with an empty queue recommends
        SCALE_DOWN (never below 1 ACTIVE worker of any kind, to preserve
        availability).
        """
        active = self.list_workers(status=_WORKER_STATUS_ACTIVE)
        total_capacity = sum(w["capacity"] for w in active)
        total_load = sum(w["current_load"] for w in active)
        utilization = (
            (total_load / total_capacity) if total_capacity > 0 else 1.0
        )  # no capacity == maximally saturated

        if not active:
            return {
                "action": "SCALE_UP",
                "reason": "no ACTIVE workers available",
                "utilization": utilization,
            }
        if utilization >= high_watermark or queue_depth > 0 and utilization >= low_watermark:
            return {
                "action": "SCALE_UP",
                "reason": f"utilization={utilization:.2f} queue_depth={queue_depth}",
                "utilization": utilization,
            }
        if utilization <= low_watermark and queue_depth == 0 and len(active) > 1:
            return {
                "action": "SCALE_DOWN",
                "reason": f"utilization={utilization:.2f} queue_depth=0",
                "utilization": utilization,
            }
        return {
            "action": "NONE",
            "reason": f"utilization={utilization:.2f} queue_depth={queue_depth}",
            "utilization": utilization,
        }

    # ── High Availability: supervision, restart, failover ────────────────────

    def check_worker_health(self, worker: dict[str, Any], *, now: float | None = None) -> str:
        """Return HEALTHY / STALE / DEAD for one worker record based on its
        last heartbeat age relative to ``heartbeat_timeout_s``."""
        now = now if now is not None else time.time()
        if worker["status"] == _WORKER_STATUS_DEAD:
            return "DEAD"
        age = now - worker.get("last_heartbeat", 0.0)
        if age > self._heartbeat_timeout_s * 2:
            return "DEAD"
        if age > self._heartbeat_timeout_s:
            return "STALE"
        return "HEALTHY"

    def run_supervision_cycle(self, *, now: float | None = None) -> dict[str, Any]:
        """Model process supervision: scan every worker, mark DEAD ones,
        restart-count DEAD ACTIVE workers, and fail over to a STANDBY when
        an ACTIVE worker cannot be revived. Returns a summary suitable for
        persisting/reporting.
        """
        now = now if now is not None else time.time()
        restarted: list[str] = []
        failovers: list[dict[str, str]] = []
        dead_active: list[dict[str, Any]] = []

        for worker in self.list_workers():
            health = self.check_worker_health(worker, now=now)
            if health == "DEAD" and worker["status"] != _WORKER_STATUS_DEAD:
                worker["status"] = _WORKER_STATUS_DEAD
                worker["updated_at"] = now
                self._db.upsert_resilience_worker(worker)
                if worker["role"] == _WORKER_STATUS_ACTIVE:
                    dead_active.append(worker)
            elif health == "STALE":
                # Automatic worker restart: bump restart_count and re-arm the
                # heartbeat clock, giving the (real, external) process one
                # more window to prove liveness before being marked DEAD.
                worker["restart_count"] = worker.get("restart_count", 0) + 1
                worker["last_heartbeat"] = now
                worker["updated_at"] = now
                self._db.upsert_resilience_worker(worker)
                restarted.append(worker["worker_id"])

        for worker in dead_active:
            promoted = self.failover(worker["worker_id"], now=now)
            if promoted:
                failovers.append({"from": worker["worker_id"], "to": promoted["worker_id"]})

        return {
            "generated_at": now,
            "n_restarted": len(restarted),
            "restarted_worker_ids": restarted,
            "n_failovers": len(failovers),
            "failovers": failovers,
            "n_dead_active_unrecovered": len(dead_active) - len(failovers),
        }

    def failover(self, dead_worker_id: str, *, now: float | None = None) -> dict[str, Any] | None:
        """Inference failover: promote the least-loaded healthy STANDBY
        worker to ACTIVE to replace *dead_worker_id*. Returns the promoted
        worker record, or ``None`` if no healthy STANDBY exists.
        """
        now = now if now is not None else time.time()
        standbys = [
            w
            for w in self.list_workers(status=_WORKER_STATUS_STANDBY)
            if self.check_worker_health(w, now=now) == "HEALTHY"
        ]
        if not standbys:
            logger.warning(
                f"WorkerPoolManager: failover for {dead_worker_id!r} found no healthy STANDBY worker"
            )
            return None
        promoted = min(standbys, key=lambda w: w["current_load"] / max(w["capacity"], 1e-9))
        promoted["role"] = _WORKER_STATUS_ACTIVE
        promoted["status"] = _WORKER_STATUS_ACTIVE
        promoted["updated_at"] = now
        self._db.upsert_resilience_worker(promoted)
        logger.info(f"WorkerPoolManager: failover {dead_worker_id!r} -> {promoted['worker_id']!r}")
        return promoted

    def restart_worker(self, worker_id: str, *, now: float | None = None) -> dict[str, Any]:
        """Manually restart a worker: reset it to its registered role and
        clear DEAD/DRAINING status, incrementing ``restart_count``."""
        row = self._db.get_resilience_worker(worker_id)
        if row is None:
            raise ValueError(f"Unknown worker_id: {worker_id!r}")
        now = now if now is not None else time.time()
        row["status"] = row["role"]
        row["current_load"] = 0.0
        row["restart_count"] = row.get("restart_count", 0) + 1
        row["last_heartbeat"] = now
        row["updated_at"] = now
        self._db.upsert_resilience_worker(row)
        return row


# ── 2. InferenceQueueMonitor — async queues, batching, saturation ────────────


def form_batches(
    pending: list[dict[str, Any]],
    *,
    max_batch_size: int,
    max_wait_s: float,
    now: float | None = None,
) -> list[list[dict[str, Any]]]:
    """Pure request-batching helper.

    Groups ``pending`` inference requests (each a dict with a ``queued_at``
    timestamp) into batches of at most ``max_batch_size``. Any request that
    has already waited longer than ``max_wait_s`` forces the batch it's in
    to close immediately (even if under-sized), so latency-sensitive
    requests are never held indefinitely waiting for a full batch.
    """
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")
    now = now if now is not None else time.time()
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for req in sorted(pending, key=lambda r: r.get("queued_at", 0.0)):
        current.append(req)
        waited = now - req.get("queued_at", now)
        if len(current) >= max_batch_size or waited >= max_wait_s:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


class InferenceQueueMonitor:
    """Computes queue-depth / saturation / batching-efficiency reports from
    the ``tel_inference`` telemetry stream and the live worker pool.
    """

    def __init__(self, db: ResearchLabDatabase, worker_pool: WorkerPoolManager) -> None:
        self._db = db
        self._worker_pool = worker_pool

    def compute_report(
        self,
        *,
        model_version: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> dict[str, Any]:
        from trainer.research_lab.telemetry_models import TelemetryQuery

        rows = self._db.query_telemetry(
            "tel_inference",
            TelemetryQuery(model_version=model_version, since=since, until=until, limit=5000),
        )
        active = self._worker_pool.list_workers(status=_WORKER_STATUS_ACTIVE)
        total_capacity = sum(w["capacity"] for w in active)
        total_load = sum(w["current_load"] for w in active)
        saturation = (
            (total_load / total_capacity)
            if total_capacity > 0
            else (1.0 if active or rows else 0.0)
        )

        queue_lengths = [r["queue_length"] for r in rows if r.get("queue_length") is not None]
        avg_queue_depth = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0.0
        max_queue_depth = max(queue_lengths) if queue_lengths else 0

        return {
            "report_id": _new_id("infq"),
            "model_version": model_version or "",
            "generated_at": time.time(),
            "n_samples": len(rows),
            "n_active_workers": len(active),
            "worker_utilization": round(saturation, 4),
            "avg_queue_depth": round(avg_queue_depth, 3),
            "max_queue_depth": max_queue_depth,
            "is_saturated": saturation >= 0.9,
        }


# ── 3. ArtifactIntegrityVerifier ──────────────────────────────────────────────


class ArtifactIntegrityVerifier:
    """Checksum validation, rollback safety checks, and version compatibility
    checks layered on top of the existing S20/S21A artifact registry and
    signing infrastructure (this module never re-derives artifact hashing
    logic — it always defers to ``compute_bytes_hash``/``ArtifactSigner``).
    """

    def __init__(
        self, db: ResearchLabDatabase, artifact_registry: Any, deployment_manager: Any | None = None
    ) -> None:
        self._db = db
        self._registry = artifact_registry
        # Signature verification is intentionally delegated to
        # DeploymentManager.verify_artifact_signatures() rather than
        # reimplemented here — it is the S21A source of truth for the
        # `.sig`-sidecar-file verification mechanism, and re-deriving that
        # logic in a second place risks the two silently drifting apart
        # (see memory: duplicated-drift-math-parity).
        self._deployment_manager = deployment_manager

    def verify_artifact_checksum(self, artifact_id: str) -> dict[str, Any]:
        """Recompute the SHA-256 of the artifact's on-disk bytes and compare
        against the registered ``artifact_hash``.

        Returns one of three outcomes:
          - ``VERIFIED``      — file present, hash matches.
          - ``CORRUPTED``     — file present, hash mismatch.
          - ``UNVERIFIABLE``  — file missing/unreadable (soft failure — does
            not by itself imply corruption, e.g. artifact stored remotely).
        """
        record = self._registry.get_artifact(artifact_id)
        if record is None:
            return {"artifact_id": artifact_id, "status": "NOT_FOUND"}
        path = Path(record.path)
        actual_hash = _sha256_file(path)
        if actual_hash is None:
            return {"artifact_id": artifact_id, "status": "UNVERIFIABLE", "path": str(path)}
        if actual_hash != record.artifact_hash:
            return {
                "artifact_id": artifact_id,
                "status": "CORRUPTED",
                "path": str(path),
                "expected_hash": record.artifact_hash,
                "actual_hash": actual_hash,
            }
        return {"artifact_id": artifact_id, "status": "VERIFIED", "path": str(path)}

    def check_version_compatibility(
        self, from_manifest: dict[str, Any], to_manifest: dict[str, Any]
    ) -> dict[str, Any]:
        """Compare two deployment manifests' declared schema/architecture
        metadata for forward/backward compatibility.

        A version is INCOMPATIBLE if the architecture family differs or the
        major schema version differs; a differing minor schema version is
        COMPATIBLE_WITH_WARNINGS.
        """
        reasons: list[str] = []
        from_arch = from_manifest.get("architecture", "")
        to_arch = to_manifest.get("architecture", "")
        if from_arch and to_arch and from_arch != to_arch:
            reasons.append(f"architecture mismatch: {from_arch!r} vs {to_arch!r}")

        from_schema = str(from_manifest.get("schema_version", "1.0"))
        to_schema = str(to_manifest.get("schema_version", "1.0"))
        from_major = from_schema.split(".")[0]
        to_major = to_schema.split(".")[0]
        warnings: list[str] = []
        if from_major != to_major:
            reasons.append(f"schema major version mismatch: {from_schema!r} vs {to_schema!r}")
        elif from_schema != to_schema:
            warnings.append(f"schema minor version differs: {from_schema!r} vs {to_schema!r}")

        if reasons:
            return {"compatible": False, "reasons": reasons, "warnings": warnings}
        return {"compatible": True, "reasons": [], "warnings": warnings}

    def check_rollback_safety(self, model_version: str) -> dict[str, Any]:
        """Verify every artifact registered under *model_version* passes
        checksum verification before that version is considered safe to
        roll back to. Also runs S21A signature verification via the
        configured ``DeploymentManager`` (the same mechanism that gates
        SHADOW/CANDIDATE/ACTIVE promotion), if one was supplied.

        Fails closed: an artifact whose checksum cannot even be verified
        (file missing/unreadable — ``UNVERIFIABLE``) blocks rollback safety
        just as a confirmed ``CORRUPTED`` result would. Rollback safety is a
        binary go/no-go gate, not a monitoring signal — "we can't tell if
        this is safe" must never be treated as "safe".
        """
        artifacts = self._registry.list_artifacts(model_version=model_version, limit=200)
        if not artifacts:
            return {
                "model_version": model_version,
                "safe": False,
                "reasons": [f"no artifacts registered for {model_version!r}"],
            }

        reasons: list[str] = []
        for art in artifacts:
            check = self.verify_artifact_checksum(art["artifact_id"])
            if check["status"] == "CORRUPTED":
                reasons.append(f"artifact {art['artifact_id']!r} ({art.get('name')}) is CORRUPTED")
            elif check["status"] == "NOT_FOUND":
                reasons.append(f"artifact {art['artifact_id']!r} vanished from the registry")
            elif check["status"] == "UNVERIFIABLE":
                reasons.append(
                    f"artifact {art['artifact_id']!r} ({art.get('name')}) is UNVERIFIABLE (file missing/unreadable)"
                )

        if self._deployment_manager is not None:
            for sig_result in self._deployment_manager.verify_artifact_signatures(model_version):
                if not sig_result.verified:
                    reasons.append(
                        f"artifact {sig_result.name!r} failed signature verification: {sig_result.reason}"
                    )

        return {
            "model_version": model_version,
            "safe": len(reasons) == 0,
            "reasons": reasons,
            "n_artifacts_checked": len(artifacts),
        }


# ── 4. ResearchLabBackupManager ───────────────────────────────────────────────


class ResearchLabBackupManager:
    """Automatic database backups + persistence verification + documented
    recovery procedures for the Research Lab's own state.
    """

    def __init__(
        self, db: ResearchLabDatabase, backup_dir: str | Path = _DEFAULT_BACKUP_DIR
    ) -> None:
        self._db = db
        self._backup_dir = Path(backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, *, reason: str = "scheduled") -> dict[str, Any]:
        """Use SQLite's online backup API to snapshot the live database into
        a new timestamped file — safe to run while the server is serving
        traffic (no downtime, no locking of the live connection beyond
        SQLite's own internal backup-step locking).
        """
        now = time.time()
        backup_id = _new_id("bak")
        filename = f"research_lab_{int(now)}_{backup_id}.db"
        backup_path = self._backup_dir / filename

        src = self._db._conn  # noqa: SLF001 — intentional: backup API needs the live connection
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()

        checksum = _sha256_file(backup_path)
        size_bytes = backup_path.stat().st_size if backup_path.exists() else 0
        record = {
            "backup_id": backup_id,
            "backup_path": str(backup_path),
            "checksum": checksum,
            "size_bytes": size_bytes,
            "reason": reason,
            "verified": False,
            "created_at": now,
        }
        self._db.upsert_resilience_backup(record)
        logger.info(f"ResearchLabBackupManager: created backup {backup_id} ({size_bytes} bytes)")
        return record

    def verify_backup(self, backup_id: str) -> dict[str, Any]:
        """Re-check a backup's checksum and run SQLite's own integrity check
        against it, marking it ``verified`` on success."""
        record = self._db.get_resilience_backup(backup_id)
        if record is None:
            return {"backup_id": backup_id, "status": "NOT_FOUND"}
        path = Path(record["backup_path"])
        actual_checksum = _sha256_file(path)
        if actual_checksum is None:
            return {"backup_id": backup_id, "status": "MISSING", "path": str(path)}
        if actual_checksum != record["checksum"]:
            return {"backup_id": backup_id, "status": "CHECKSUM_MISMATCH"}

        try:
            conn = sqlite3.connect(str(path))
            try:
                result = conn.execute("PRAGMA integrity_check").fetchone()
                ok = result is not None and result[0] == "ok"
            finally:
                conn.close()
        except sqlite3.Error:
            ok = False

        record["verified"] = bool(ok)
        self._db.upsert_resilience_backup(record)
        return {"backup_id": backup_id, "status": "VERIFIED" if ok else "INTEGRITY_CHECK_FAILED"}

    def list_backups(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_resilience_backups(limit=limit)

    def restore_backup(self, backup_id: str, target_path: str | Path) -> dict[str, Any]:
        """Copy a verified backup file to *target_path*.

        Deliberately does **not** overwrite the live database in place or
        swap connections — restoring the live Research Lab state is a
        destructive, operator-supervised action. This performs the safe
        half of "recovery" (materializing a known-good snapshot at a
        chosen location) and leaves cutover to the operator.
        """
        record = self._db.get_resilience_backup(backup_id)
        if record is None:
            raise ValueError(f"Unknown backup_id: {backup_id!r}")
        src = Path(record["backup_path"])
        if not src.exists():
            raise FileNotFoundError(f"Backup file missing on disk: {src}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        return {"backup_id": backup_id, "restored_to": str(target)}

    # ── Persistence verification ─────────────────────────────────────────────

    def verify_experiment_persistence(self, experiment_id: str) -> dict[str, Any]:
        record = (
            self._db.get_experiment(experiment_id) if hasattr(self._db, "get_experiment") else None
        )
        if record is None:
            return {"experiment_id": experiment_id, "persisted": False}
        return {"experiment_id": experiment_id, "persisted": True}

    def verify_shadow_test_persistence(self, shadow_test_id: str) -> dict[str, Any]:
        record = (
            self._db.get_shadow_test(shadow_test_id)
            if hasattr(self._db, "get_shadow_test")
            else None
        )
        if record is None:
            return {"shadow_test_id": shadow_test_id, "persisted": False}
        return {"shadow_test_id": shadow_test_id, "persisted": True}

    @staticmethod
    def recovery_procedures() -> list[dict[str, str]]:
        """Static, documented recovery runbook — surfaced to admins via the
        dashboard rather than executed automatically, since the actual
        cutover step is always operator-supervised."""
        return [
            {
                "step": "1",
                "action": "Identify the most recent VERIFIED backup via list_backups()/verify_backup().",
            },
            {
                "step": "2",
                "action": "restore_backup() the chosen backup to a staging path, distinct from the live DB file.",
            },
            {
                "step": "3",
                "action": "Validate the staged copy (integrity_check + spot-check recent experiments/shadow tests).",
            },
            {"step": "4", "action": "Stop the Research Lab server process."},
            {
                "step": "5",
                "action": "Swap the staged copy into the live RESEARCH_LAB_DB_PATH location.",
            },
            {
                "step": "6",
                "action": "Restart the Research Lab server and re-verify via telemetry_status/health-dashboard.",
            },
        ]


# ── 5. TrainingResilienceMonitor ──────────────────────────────────────────────


class TrainingResilienceMonitor:
    """Failed-training cleanup, corrupted-artifact detection, and
    checkpoint-recovery visibility for the training pipeline.
    """

    def __init__(self, db: ResearchLabDatabase, artifact_registry: Any | None = None) -> None:
        self._db = db
        self._registry = artifact_registry

    def detect_stale_jobs(
        self, *, stale_after_s: float = _DEFAULT_STALE_JOB_TIMEOUT_S, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Return pipeline jobs stuck in QUEUED/RUNNING whose ``updated_at``
        (interrupted-training continuation would have refreshed it) is older
        than ``stale_after_s`` — i.e. training that appears to have died
        without reaching a terminal status.
        """
        now = now if now is not None else time.time()
        jobs = self._db.list_pipeline_jobs(limit=500)
        stale = []
        for job in jobs:
            if job.get("status") not in ("QUEUED", "RUNNING"):
                continue
            updated_at = job.get("updated_at", job.get("created_at", 0.0))
            if now - updated_at >= stale_after_s:
                stale.append(job)
        return stale

    def cleanup_failed_jobs(
        self, *, stale_after_s: float = _DEFAULT_STALE_JOB_TIMEOUT_S, now: float | None = None
    ) -> dict[str, Any]:
        """Mark every stale job (see ``detect_stale_jobs``) FAILED, so
        downstream retraining/scheduling logic never treats a dead job as
        still in flight."""
        now = now if now is not None else time.time()
        stale = self.detect_stale_jobs(stale_after_s=stale_after_s, now=now)
        cleaned: list[str] = []
        for job in stale:
            job["status"] = "FAILED"
            job["updated_at"] = now
            job.setdefault(
                "error", "cleaned up by TrainingResilienceMonitor: exceeded stale-job timeout"
            )
            self._db.upsert_pipeline_job(job)
            cleaned.append(job["job_id"])
        return {"generated_at": now, "n_cleaned": len(cleaned), "job_ids": cleaned}

    def detect_corrupted_artifacts(
        self, *, model_version: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Recompute checksums for every registered artifact (optionally
        scoped to a model version) and return the ones that fail."""
        if self._registry is None:
            return []
        artifacts = self._registry.list_artifacts(model_version=model_version, limit=limit)
        corrupted = []
        for art in artifacts:
            path = Path(art["path"])
            actual_hash = _sha256_file(path)
            if actual_hash is not None and actual_hash != art["artifact_hash"]:
                corrupted.append(
                    {
                        "artifact_id": art["artifact_id"],
                        "name": art.get("name"),
                        "expected_hash": art["artifact_hash"],
                        "actual_hash": actual_hash,
                    }
                )
        return corrupted

    @staticmethod
    def checkpoint_recovery_status(checkpoint_dir: str | Path) -> dict[str, Any]:
        """Filesystem-only check (no torch import — see memory:
        torch-broken-lazy-registry-import) for whether a checkpoint
        directory has a usable ``latest.pt``/``best.pt`` to resume from.
        """
        directory = Path(checkpoint_dir)
        if not directory.is_dir():
            return {
                "checkpoint_dir": str(directory),
                "resumable": False,
                "reason": "directory does not exist",
            }
        candidates = [directory / "latest.pt", directory / "best.pt"]
        usable = [str(c) for c in candidates if c.exists() and c.stat().st_size > 0]
        return {
            "checkpoint_dir": str(directory),
            "resumable": bool(usable),
            "usable_checkpoints": usable,
            "reason": "" if usable else "no non-empty latest.pt/best.pt found",
        }


# ── 6. DisasterRecoveryManager ────────────────────────────────────────────────


class DisasterRecoveryManager:
    """Bundles cross-cutting references (model, dataset, feature store,
    lineage, experiments) for a point in time into one integrity-checked
    manifest — the "disaster recovery snapshot".

    This is metadata-only, mirroring ``ArtifactRegistry``'s own design: it
    records *what exists and where*, not a copy of the underlying data, so
    creating a snapshot is cheap and safe to run frequently.
    """

    def __init__(self, db: ResearchLabDatabase) -> None:
        self._db = db

    def create_snapshot(
        self,
        *,
        model_version: str = "",
        dataset_version: str = "",
        feature_version: str = "",
        experiment_ids: list[str] | None = None,
        lineage_refs: dict[str, Any] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        manifest = {
            "model_version": model_version,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "experiment_ids": experiment_ids or [],
            "lineage_refs": lineage_refs or {},
            "notes": notes,
        }
        import json as _json

        manifest_json = _json.dumps(manifest, sort_keys=True)
        checksum = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        record = {
            "snapshot_id": _new_id("drs"),
            "created_at": now,
            "manifest": manifest,
            "manifest_checksum": checksum,
        }
        self._db.upsert_disaster_recovery_snapshot(record)
        return record

    def verify_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        record = self._db.get_disaster_recovery_snapshot(snapshot_id)
        if record is None:
            return {"snapshot_id": snapshot_id, "status": "NOT_FOUND"}
        import json as _json

        manifest_json = _json.dumps(record["manifest"], sort_keys=True)
        actual_checksum = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        ok = actual_checksum == record["manifest_checksum"]
        return {"snapshot_id": snapshot_id, "status": "VERIFIED" if ok else "TAMPERED"}

    def list_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._db.list_disaster_recovery_snapshots(limit=limit)


# ── 7. DeploymentSafetyMonitor ────────────────────────────────────────────────


class DeploymentSafetyMonitor:
    """Deployment health validation, compatibility checks, rollback
    automation, and active-model verification — layered on top of the
    existing S20 ``DeploymentManager`` + S25 ``CompositeQDEHealthScorer``.
    """

    def __init__(
        self,
        deployment_manager: Any,
        composite_health_scorer: Any,
        artifact_integrity_verifier: ArtifactIntegrityVerifier,
        health_threshold: float = 70.0,
    ) -> None:
        self._deployment_manager = deployment_manager
        self._composite_health_scorer = composite_health_scorer
        self._integrity = artifact_integrity_verifier
        self._health_threshold = health_threshold

    def validate_deployment_health(self, model_version: str | None = None) -> dict[str, Any]:
        score = (
            self._composite_health_scorer.get_latest(model_version=model_version)
            if self._composite_health_scorer
            else None
        )
        if score is None:
            return {
                "model_version": model_version,
                "healthy": True,
                "composite_score": None,
                "reason": "no composite health score yet",
            }
        composite = score.get("composite", 100.0)
        healthy = composite >= self._health_threshold
        return {
            "model_version": model_version,
            "healthy": healthy,
            "composite_score": composite,
            "reason": (
                ""
                if healthy
                else f"composite score {composite} below threshold {self._health_threshold}"
            ),
        }

    def verify_active_model(self) -> dict[str, Any]:
        """Cross-check the currently ACTIVE deployment record against the
        artifact registry's own checksum verification, catching drift
        between "what Signalium was told is active" and "what integrity
        checks say about that version's artifacts"."""
        current = self._deployment_manager.get_current() if self._deployment_manager else None
        if current is None:
            return {"active": False, "model_version": None, "reason": "no ACTIVE deployment"}
        model_version = current.get("model_version")
        rollback_check = self._integrity.check_rollback_safety(model_version)
        return {
            "active": True,
            "model_version": model_version,
            "artifacts_ok": rollback_check["safe"],
            "reasons": rollback_check["reasons"],
        }

    def maybe_auto_rollback(
        self, model_version: str, *, requested_by: str = "DeploymentSafetyMonitor"
    ) -> dict[str, Any] | None:
        """If the currently ACTIVE deployment's health has fallen below
        threshold, automatically raise a rollback request (still gated by
        the normal ApprovalGate — this automates *detection and request*,
        not unattended promotion) to the most recent RETIRED version.
        """
        health = self.validate_deployment_health(model_version=model_version)
        if health["healthy"]:
            return None
        return self._deployment_manager.request_rollback(
            model_version="latest_retired",
            reason=f"auto-rollback: {health['reason']}",
            requested_by=requested_by,
        )


# ── 8. ResilienceDashboard — S26 Administrative Visibility surface ───────────


class ResilienceDashboard:
    """Single aggregated view of scalability + resilience state, mirroring
    the S25 ``QDEHealthDashboard`` pattern for the S26 topics."""

    def __init__(
        self,
        worker_pool: WorkerPoolManager,
        queue_monitor: InferenceQueueMonitor,
        backup_manager: ResearchLabBackupManager,
        training_resilience: TrainingResilienceMonitor,
        disaster_recovery: DisasterRecoveryManager,
        deployment_safety: DeploymentSafetyMonitor,
    ) -> None:
        self._worker_pool = worker_pool
        self._queue_monitor = queue_monitor
        self._backup_manager = backup_manager
        self._training_resilience = training_resilience
        self._disaster_recovery = disaster_recovery
        self._deployment_safety = deployment_safety

    def compute(self, *, model_version: str | None = None) -> dict[str, Any]:
        workers = self._worker_pool.list_workers()
        active = [w for w in workers if w["status"] == _WORKER_STATUS_ACTIVE]
        standby = [w for w in workers if w["status"] == _WORKER_STATUS_STANDBY]
        dead = [w for w in workers if w["status"] == _WORKER_STATUS_DEAD]

        queue_report = self._queue_monitor.compute_report(model_version=model_version)
        backups = self._backup_manager.list_backups(limit=1)
        latest_backup = backups[0] if backups else None
        stale_jobs = self._training_resilience.detect_stale_jobs()
        snapshots = self._disaster_recovery.list_snapshots(limit=1)
        latest_snapshot = snapshots[0] if snapshots else None
        deployment_health = self._deployment_safety.validate_deployment_health(
            model_version=model_version
        )

        return {
            "generated_at": time.time(),
            "model_version": model_version,
            "workers": {
                "n_active": len(active),
                "n_standby": len(standby),
                "n_dead": len(dead),
                "n_total": len(workers),
            },
            "scaling_recommendation": self._worker_pool.recommend_scaling(
                queue_depth=int(queue_report["avg_queue_depth"])
            ),
            "queue": queue_report,
            "latest_backup": latest_backup,
            "n_stale_training_jobs": len(stale_jobs),
            "latest_disaster_recovery_snapshot": latest_snapshot,
            "deployment_health": deployment_health,
        }
