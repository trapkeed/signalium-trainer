"""Phase 16 — AutoML Resource Manager.

Tracks and optimizes compute resource usage during AutoML searches.

Manages:
  - GPU utilization tracking
  - CPU utilization tracking
  - Memory usage (RAM + VRAM)
  - Parallel worker slots
  - Checkpoint frequency
  - Disk usage estimates
  - Resume-after-interruption support

Usage::

    rm = ResourceManager(max_workers=4, gpu_memory_gb=40.0)
    token = rm.acquire_slot()
    rm.record_usage(trial_id="t-01", cpu_pct=45.0, ram_mb=8192)
    rm.release_slot(token)
    report = rm.usage_report()
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Resource snapshot
# ---------------------------------------------------------------------------


@dataclass
class ResourceSnapshot:
    """A point-in-time resource measurement for a trial.

    Attributes:
        trial_id: Trial being measured.
        timestamp: UTC ISO-8601 timestamp.
        cpu_percent: CPU utilization 0-100 per core (averaged).
        ram_mb: RAM used in megabytes.
        gpu_percent: GPU utilization 0-100 (if available).
        vram_mb: GPU memory used in megabytes.
        disk_mb: Disk usage in megabytes.
        elapsed_seconds: Elapsed trial time.
    """

    trial_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    cpu_percent: float = 0.0
    ram_mb: float = 0.0
    gpu_percent: float = 0.0
    vram_mb: float = 0.0
    disk_mb: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "ram_mb": self.ram_mb,
            "gpu_percent": self.gpu_percent,
            "vram_mb": self.vram_mb,
            "disk_mb": self.disk_mb,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# Worker slot
# ---------------------------------------------------------------------------


@dataclass
class WorkerSlot:
    """A reserved parallel worker slot."""

    token: str = field(default_factory=lambda: str(uuid.uuid4()))
    trial_id: str = ""
    acquired_at: float = field(default_factory=time.monotonic)
    released_at: float | None = None

    @property
    def is_active(self) -> bool:
        return self.released_at is None

    @property
    def elapsed_seconds(self) -> float:
        end = self.released_at or time.monotonic()
        return end - self.acquired_at


# ---------------------------------------------------------------------------
# Resource manager
# ---------------------------------------------------------------------------


class ResourceManager:
    """Manages compute resources for AutoML parallel execution.

    Args:
        max_workers: Maximum number of concurrent trials.
        gpu_memory_gb: Total GPU memory in GB (0 = CPU only).
        max_ram_gb: Maximum RAM in GB before throttling.
        max_disk_gb: Maximum disk usage in GB.
        checkpoint_dir: Directory for resume checkpoints.
    """

    def __init__(
        self,
        max_workers: int = 1,
        gpu_memory_gb: float = 0.0,
        max_ram_gb: float = 32.0,
        max_disk_gb: float = 100.0,
        checkpoint_dir: str = "experiments/automl/checkpoints",
    ) -> None:
        self.max_workers = max_workers
        self.gpu_memory_gb = gpu_memory_gb
        self.max_ram_gb = max_ram_gb
        self.max_disk_gb = max_disk_gb
        self.checkpoint_dir = checkpoint_dir

        self._lock = threading.Lock()
        self._slots: dict[str, WorkerSlot] = {}
        self._snapshots: dict[str, list[ResourceSnapshot]] = {}
        self._total_gpu_allocated_mb: float = 0.0

    @property
    def active_workers(self) -> int:
        with self._lock:
            return sum(1 for s in self._slots.values() if s.is_active)

    @property
    def available_slots(self) -> int:
        return max(0, self.max_workers - self.active_workers)

    def can_acquire(self) -> bool:
        """Return True if a worker slot is available."""
        return self.available_slots > 0

    def acquire_slot(self, trial_id: str = "") -> str | None:
        """Acquire a worker slot. Returns token or None if no slot available."""
        with self._lock:
            if self.active_workers >= self.max_workers:
                return None
            slot = WorkerSlot(trial_id=trial_id)
            self._slots[slot.token] = slot
            return slot.token

    def release_slot(self, token: str) -> bool:
        """Release a worker slot. Returns True if slot was found."""
        with self._lock:
            if token in self._slots:
                self._slots[token].released_at = time.monotonic()
                return True
            return False

    def record_usage(
        self,
        trial_id: str,
        cpu_percent: float = 0.0,
        ram_mb: float = 0.0,
        gpu_percent: float = 0.0,
        vram_mb: float = 0.0,
        disk_mb: float = 0.0,
        elapsed_seconds: float = 0.0,
    ) -> ResourceSnapshot:
        """Record a resource usage snapshot for a trial."""
        snap = ResourceSnapshot(
            trial_id=trial_id,
            cpu_percent=cpu_percent,
            ram_mb=ram_mb,
            gpu_percent=gpu_percent,
            vram_mb=vram_mb,
            disk_mb=disk_mb,
            elapsed_seconds=elapsed_seconds,
        )
        with self._lock:
            if trial_id not in self._snapshots:
                self._snapshots[trial_id] = []
            self._snapshots[trial_id].append(snap)
        return snap

    def get_trial_usage(self, trial_id: str) -> dict[str, Any]:
        """Return aggregated resource usage for a trial."""
        snaps = self._snapshots.get(trial_id, [])
        if not snaps:
            return {"trial_id": trial_id, "n_snapshots": 0}
        return {
            "trial_id": trial_id,
            "n_snapshots": len(snaps),
            "peak_cpu_percent": max(s.cpu_percent for s in snaps),
            "avg_cpu_percent": sum(s.cpu_percent for s in snaps) / len(snaps),
            "peak_ram_mb": max(s.ram_mb for s in snaps),
            "avg_ram_mb": sum(s.ram_mb for s in snaps) / len(snaps),
            "peak_gpu_percent": max(s.gpu_percent for s in snaps),
            "avg_gpu_percent": sum(s.gpu_percent for s in snaps) / len(snaps),
            "peak_vram_mb": max(s.vram_mb for s in snaps),
            "total_elapsed_seconds": snaps[-1].elapsed_seconds if snaps else 0.0,
        }

    def usage_report(self) -> dict[str, Any]:
        """Return a full resource usage report across all trials."""
        all_snaps = [s for snaps in self._snapshots.values() for s in snaps]
        active = self.active_workers
        report = {
            "max_workers": self.max_workers,
            "active_workers": active,
            "available_slots": self.available_slots,
            "gpu_memory_gb": self.gpu_memory_gb,
            "total_trials_tracked": len(self._snapshots),
            "total_snapshots": len(all_snaps),
        }
        if all_snaps:
            report.update(
                {
                    "peak_cpu_percent": max(s.cpu_percent for s in all_snaps),
                    "avg_cpu_percent": sum(s.cpu_percent for s in all_snaps) / len(all_snaps),
                    "peak_ram_mb": max(s.ram_mb for s in all_snaps),
                    "avg_ram_mb": sum(s.ram_mb for s in all_snaps) / len(all_snaps),
                    "peak_gpu_percent": max(s.gpu_percent for s in all_snaps),
                    "peak_vram_mb": max(s.vram_mb for s in all_snaps),
                }
            )
        return report

    def should_throttle(self) -> bool:
        """Return True if resource pressure is high and execution should pause."""
        report = self.usage_report()
        ram_gb = report.get("peak_ram_mb", 0) / 1024
        if ram_gb > self.max_ram_gb * 0.9:
            return True
        vram_gb = report.get("peak_vram_mb", 0) / 1024
        if self.gpu_memory_gb > 0 and vram_gb > self.gpu_memory_gb * 0.95:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "gpu_memory_gb": self.gpu_memory_gb,
            "max_ram_gb": self.max_ram_gb,
            "max_disk_gb": self.max_disk_gb,
            "checkpoint_dir": self.checkpoint_dir,
        }
