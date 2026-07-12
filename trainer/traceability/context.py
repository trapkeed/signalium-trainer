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

try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001 — broken .so or not installed
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _pytorch_version() -> str:
    return _torch.__version__ if _TORCH_AVAILABLE else "unavailable"


def _cuda_version() -> str | None:
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        return _torch.version.cuda or "unknown"
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


def _device_name(device: Any) -> str:
    """Human-readable device name for a torch.device (or plain string fallback)."""
    if not _TORCH_AVAILABLE:
        return str(device)
    if device.type == "cuda":
        try:
            idx = device.index or 0
            return _torch.cuda.get_device_name(idx)
        except Exception:
            return f"cuda:{device.index}"
    if device.type == "mps":
        return "Apple MPS"
    return "cpu"


def _memory_used_mb(device: Any) -> float:
    """Approximate current memory consumption in MB."""
    if not _TORCH_AVAILABLE:
        return 0.0
    if device.type == "cuda":
        try:
            idx = device.index or 0
            return _torch.cuda.memory_allocated(idx) / (1024**2)
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
        device: Any,
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
    def device(self) -> Any:
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
