"""InferenceContext — captures the runtime environment for traceability.

Collects:
  - PyTorch version
  - CUDA availability and version
  - Python version
  - Git commit hash
  - Device information
  - Memory usage (GPU or system RAM)

All fields fall back gracefully when information is unavailable.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import torch


def _pytorch_version() -> str:
    return torch.__version__


def _cuda_version() -> str | None:
    if torch.cuda.is_available():
        return torch.version.cuda or "unknown"
    return None


def _python_version() -> str:
    return sys.version.split()[0]


def _git_commit(short: bool = True) -> str:
    """Try to read the current git commit hash.

    Returns a 12-char short hash on success, or "unknown" on failure.
    """
    try:
        raw = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        sha = raw.decode().strip()
        return sha[:12] if short else sha
    except Exception:
        return "unknown"


def _device_name(device: torch.device) -> str:
    """Human-readable device name for a torch.device."""
    if device.type == "cuda":
        try:
            idx = device.index or 0
            return torch.cuda.get_device_name(idx)
        except Exception:
            return f"cuda:{device.index}"
    if device.type == "mps":
        return "Apple MPS"
    return "cpu"


def _memory_used_mb(device: torch.device) -> float:
    """Approximate current memory consumption in MB."""
    if device.type == "cuda":
        try:
            idx = device.index or 0
            return torch.cuda.memory_allocated(idx) / (1024**2)
        except Exception:
            return 0.0
    return 0.0


class InferenceContext:
    """Captures and holds static environment information for an inference session.

    Create once per adapter initialisation and reuse across calls to avoid
    repeated subprocess and CUDA calls.

    Args:
        device: The torch.device being used for inference.
        inference_engine_version: Caller-supplied version string.
    """

    def __init__(
        self,
        device: torch.device,
        inference_engine_version: str = "1.0.0",
    ) -> None:
        self._device = device
        self._inference_engine_version = inference_engine_version
        self._pytorch_version = _pytorch_version()
        self._cuda_version = _cuda_version()
        self._python_version = _python_version()
        self._git_commit = _git_commit(short=True)
        self._device_name = _device_name(device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def git_commit_hash(self) -> str:
        return self._git_commit

    @property
    def pytorch_version(self) -> str:
        return self._pytorch_version

    @property
    def cuda_version(self) -> str | None:
        return self._cuda_version

    @property
    def python_version(self) -> str:
        return self._python_version

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def inference_engine_version(self) -> str:
        return self._inference_engine_version

    def memory_used_mb(self) -> float:
        """Current memory usage (queried fresh each call)."""
        return _memory_used_mb(self._device)

    def to_dict(self) -> dict[str, Any]:
        """Snapshot of the current context as a plain dict."""
        return {
            "device": str(self._device),
            "device_name": self._device_name,
            "pytorch_version": self._pytorch_version,
            "cuda_version": self._cuda_version,
            "python_version": self._python_version,
            "git_commit_hash": self._git_commit,
            "inference_engine_version": self._inference_engine_version,
        }
