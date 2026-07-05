"""Phase 19.1 — Experiment Reproducibility Capture.

Automatically captures the full execution environment so that any
historical experiment can be reproduced with identical inputs and
identical configuration.

Captured fields:

  System:      OS name/version, CPU model, RAM GB, GPU model, CUDA version
  Runtime:     Python version, PyTorch version, random seed
  Repository:  Git commit hash, Git branch, repository version
  Docker:      Image tag (when REPLIT_IMAGE or IMAGE_TAG env var is set)
  Environment: Non-sensitive env vars (filtered allow-list)

Usage::

    snapshot = EnvironmentSnapshot.capture(random_seed=42)
    d = snapshot.to_dict()
    restored = EnvironmentSnapshot.from_dict(d)
    report = snapshot.to_report()
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Allowed environment-variable prefixes (non-sensitive)
# ---------------------------------------------------------------------------

_ALLOWED_ENV_PREFIXES: List[str] = [
    "LANG", "LC_", "TZ", "PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "SIGNALIUM_", "TRAINER_", "TORCH_", "NCCL_",
    "HOSTNAME", "SHELL", "TERM",
]

_SENSITIVE_SUBSTRINGS: List[str] = [
    "SECRET", "TOKEN", "PASSWORD", "PASSWD", "KEY", "CREDENTIAL",
    "API_KEY", "AUTH", "PRIVATE", "CERT",
]


def _safe_env_vars() -> Dict[str, str]:
    """Return non-sensitive environment variables matching the allow-list."""
    result: Dict[str, str] = {}
    for key, val in os.environ.items():
        key_upper = key.upper()
        if any(s in key_upper for s in _SENSITIVE_SUBSTRINGS):
            continue
        if any(key_upper.startswith(p) for p in _ALLOWED_ENV_PREFIXES):
            result[key] = val[:256]  # truncate very long values
    return result


def _git_info() -> Dict[str, str]:
    """Try to read git commit hash and branch; return empty strings on failure."""
    info: Dict[str, str] = {"commit_hash": "", "branch": "", "repo_version": ""}
    try:
        info["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        pass
    try:
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        pass
    try:
        info["repo_version"] = subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        pass
    return info


def _pytorch_info() -> Dict[str, str]:
    """Try to read PyTorch and CUDA versions; return empty strings on failure."""
    info: Dict[str, str] = {"pytorch_version": "", "cuda_version": ""}
    try:
        import torch  # type: ignore
        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or ""  # type: ignore[attr-defined]
    except Exception:
        pass
    return info


def _gpu_info() -> str:
    """Try to read GPU model via nvidia-smi; return empty string on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return out.split("\n")[0].strip()
    except Exception:
        return ""


def _cpu_info() -> str:
    """Read CPU model from /proc/cpuinfo or platform; return best-effort string."""
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float:
    """Read total RAM in GB from /proc/meminfo; return 0.0 on failure."""
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1_048_576, 2)
    except Exception:
        pass
    return 0.0


def _docker_image() -> str:
    """Read Docker image tag from common env vars; return empty string if not found."""
    for var in ("IMAGE_TAG", "DOCKER_IMAGE", "REPLIT_IMAGE", "CONTAINER_IMAGE"):
        val = os.environ.get(var, "")
        if val:
            return val
    return ""


# ---------------------------------------------------------------------------
# EnvironmentSnapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentSnapshot:
    """Complete environment snapshot for full experiment reproducibility."""

    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Repository
    git_commit_hash: str = ""
    git_branch: str = ""
    repo_version: str = ""

    # Runtime
    python_version: str = ""
    pytorch_version: str = ""
    cuda_version: str = ""
    random_seed: Optional[int] = None

    # Hardware
    os_name: str = ""
    os_version: str = ""
    cpu_model: str = ""
    ram_gb: float = 0.0
    gpu_model: str = ""

    # Docker
    docker_image: str = ""

    # Environment variables (non-sensitive)
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Training metrics (filled after training)
    training_duration_seconds: float = 0.0
    training_cost_estimate_usd: float = 0.0

    @classmethod
    def capture(cls, random_seed: Optional[int] = None) -> "EnvironmentSnapshot":
        """Capture the current execution environment."""
        git = _git_info()
        torch_info = _pytorch_info()
        return cls(
            git_commit_hash=git["commit_hash"],
            git_branch=git["branch"],
            repo_version=git["repo_version"],
            python_version=sys.version,
            pytorch_version=torch_info["pytorch_version"],
            cuda_version=torch_info["cuda_version"],
            random_seed=random_seed,
            os_name=platform.system(),
            os_version=platform.version(),
            cpu_model=_cpu_info(),
            ram_gb=_ram_gb(),
            gpu_model=_gpu_info(),
            docker_image=_docker_image(),
            env_vars=_safe_env_vars(),
        )

    def with_training_metrics(
        self,
        duration_seconds: float,
        cost_estimate_usd: float = 0.0,
    ) -> "EnvironmentSnapshot":
        """Return a copy with training duration and cost filled in."""
        from dataclasses import replace
        return replace(
            self,
            training_duration_seconds=duration_seconds,
            training_cost_estimate_usd=cost_estimate_usd,
        )

    def reproducibility_score(self) -> float:
        """Return a 0–1 score indicating how reproducible this snapshot is.

        Scoring:
          - git commit hash present: +0.25
          - python version present: +0.15
          - pytorch version present: +0.15
          - random seed set: +0.20
          - OS captured: +0.10
          - hardware captured (cpu/gpu/ram): +0.10
          - env vars captured: +0.05
        """
        score = 0.0
        if self.git_commit_hash:
            score += 0.25
        if self.python_version:
            score += 0.15
        if self.pytorch_version:
            score += 0.15
        if self.random_seed is not None:
            score += 0.20
        if self.os_name:
            score += 0.10
        if self.cpu_model:
            score += 0.05
        if self.gpu_model or self.ram_gb > 0:
            score += 0.05
        if self.env_vars:
            score += 0.05
        return min(1.0, score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "captured_at": self.captured_at,
            "git_commit_hash": self.git_commit_hash,
            "git_branch": self.git_branch,
            "repo_version": self.repo_version,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "random_seed": self.random_seed,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "cpu_model": self.cpu_model,
            "ram_gb": self.ram_gb,
            "gpu_model": self.gpu_model,
            "docker_image": self.docker_image,
            "env_vars": self.env_vars,
            "training_duration_seconds": self.training_duration_seconds,
            "training_cost_estimate_usd": self.training_cost_estimate_usd,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvironmentSnapshot":
        return cls(
            snapshot_id=d.get("snapshot_id", str(uuid.uuid4())),
            captured_at=d.get("captured_at", datetime.now(timezone.utc).isoformat()),
            git_commit_hash=d.get("git_commit_hash", ""),
            git_branch=d.get("git_branch", ""),
            repo_version=d.get("repo_version", ""),
            python_version=d.get("python_version", ""),
            pytorch_version=d.get("pytorch_version", ""),
            cuda_version=d.get("cuda_version", ""),
            random_seed=d.get("random_seed"),
            os_name=d.get("os_name", ""),
            os_version=d.get("os_version", ""),
            cpu_model=d.get("cpu_model", ""),
            ram_gb=d.get("ram_gb", 0.0),
            gpu_model=d.get("gpu_model", ""),
            docker_image=d.get("docker_image", ""),
            env_vars=d.get("env_vars", {}),
            training_duration_seconds=d.get("training_duration_seconds", 0.0),
            training_cost_estimate_usd=d.get("training_cost_estimate_usd", 0.0),
        )

    def to_report(self) -> str:
        """Return a human-readable reproducibility report."""
        score = self.reproducibility_score()
        lines = [
            "=== Environment Reproducibility Snapshot ===",
            f"Snapshot ID : {self.snapshot_id}",
            f"Captured At : {self.captured_at}",
            f"Repro Score : {score:.0%}",
            "",
            "--- Repository ---",
            f"Git Commit  : {self.git_commit_hash or 'N/A'}",
            f"Git Branch  : {self.git_branch or 'N/A'}",
            f"Repo Ver    : {self.repo_version or 'N/A'}",
            "",
            "--- Runtime ---",
            f"Python      : {self.python_version or 'N/A'}",
            f"PyTorch     : {self.pytorch_version or 'N/A'}",
            f"CUDA        : {self.cuda_version or 'N/A'}",
            f"Random Seed : {self.random_seed if self.random_seed is not None else 'N/A'}",
            "",
            "--- Hardware ---",
            f"OS          : {self.os_name} {self.os_version}",
            f"CPU         : {self.cpu_model or 'N/A'}",
            f"RAM         : {self.ram_gb} GB",
            f"GPU         : {self.gpu_model or 'N/A'}",
            "",
            "--- Training ---",
            f"Duration    : {self.training_duration_seconds:.1f}s",
            f"Cost Est.   : ${self.training_cost_estimate_usd:.4f}",
            "",
            "--- Docker ---",
            f"Image       : {self.docker_image or 'N/A'}",
            "",
            f"--- Environment Variables ({len(self.env_vars)}) ---",
        ]
        for k, v in sorted(self.env_vars.items())[:20]:
            lines.append(f"  {k}={v}")
        if len(self.env_vars) > 20:
            lines.append(f"  ... +{len(self.env_vars) - 20} more")
        return "\n".join(lines)
