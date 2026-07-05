"""Reproducibility Manager — Phase 14.9B.

Captures and verifies the complete set of signals needed to reproduce a
training run exactly:

  - Git commit + dirty state
  - Dataset version, feature version, label version
  - SHA-256 config hash (covers all hyper-parameters)
  - RNG seed
  - Python / PyTorch / NumPy versions
  - Timestamp

Usage::

    from trainer.training.reproducibility import ReproducibilityManager

    mgr = ReproducibilityManager(config)
    fingerprint = mgr.capture(
        dataset_version="v3.2.1",
        feature_version="v2.0.0",
        label_version="v1.4.0",
    )
    mgr.save(fingerprint, Path("checkpoints/run_001/fingerprint.json"))

    # Later — verify a checkpoint was produced under the same conditions:
    ok = mgr.verify(Path("checkpoints/run_001/fingerprint.json"))
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger

from trainer.utils.git import get_git_commit, is_dirty


# ---------------------------------------------------------------------------
# Run fingerprint
# ---------------------------------------------------------------------------


@dataclass
class RunFingerprint:
    """All signals needed to reproduce a training run.

    Attributes:
        git_commit: HEAD commit SHA, or None if not in a git repo.
        git_dirty: True if there are uncommitted changes.
        dataset_version: Dataset version identifier.
        feature_version: Feature engineering pipeline version.
        label_version: Label generation version.
        config_hash: SHA-256 of the serialized training config.
        seed: RNG seed.
        timestamp: ISO-8601 timestamp of run start.
        python_version: Python runtime version string.
        torch_version: PyTorch version string.
        numpy_version: NumPy version string.
        cuda_version: CUDA version string (empty when not available).
        hostname: Machine hostname for cross-run traceability.
        warnings: Any reproducibility warnings detected at capture time.
    """

    git_commit: Optional[str] = None
    git_dirty: bool = False
    dataset_version: Optional[str] = None
    feature_version: Optional[str] = None
    label_version: Optional[str] = None
    config_hash: str = ""
    seed: int = 42
    timestamp: str = ""
    python_version: str = ""
    torch_version: str = ""
    numpy_version: str = ""
    cuda_version: str = ""
    hostname: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunFingerprint:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> RunFingerprint:
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Reproducibility manager
# ---------------------------------------------------------------------------


class ReproducibilityManager:
    """Capture, save, and verify run fingerprints.

    Args:
        config: Full ``TrainerConfig`` — hashed for the fingerprint.
    """

    def __init__(self, config: Any) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(
        self,
        dataset_version: Optional[str] = None,
        feature_version: Optional[str] = None,
        label_version: Optional[str] = None,
    ) -> RunFingerprint:
        """Build a ``RunFingerprint`` for the current run.

        Args:
            dataset_version: Version string from the dataset pipeline.
            feature_version: Version of the feature engineering pipeline.
            label_version: Version of the label generation pipeline.

        Returns:
            A fully-populated ``RunFingerprint``.
        """
        warnings: list[str] = []

        commit = get_git_commit()
        dirty = is_dirty()
        if dirty:
            warnings.append(
                "Working tree has uncommitted changes — run is not fully reproducible."
            )
        if commit is None:
            warnings.append("Not in a git repository — git commit unavailable.")

        config_hash = self.hash_config(self._config)
        seed = getattr(getattr(self._config, "training", None), "seed", 0)

        cuda_ver = ""
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda or ""

        hostname = _safe_hostname()

        fp = RunFingerprint(
            git_commit=commit,
            git_dirty=dirty,
            dataset_version=dataset_version,
            feature_version=feature_version,
            label_version=label_version,
            config_hash=config_hash,
            seed=seed,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            python_version=sys.version,
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            cuda_version=cuda_ver,
            hostname=hostname,
            warnings=warnings,
        )

        if warnings:
            for w in warnings:
                logger.warning("Reproducibility warning: {}", w)

        logger.info(
            "RunFingerprint captured — git={} dirty={} config_hash={} seed={}",
            commit[:8] if commit else "N/A",
            dirty,
            config_hash[:12],
            seed,
        )
        return fp

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, fingerprint: RunFingerprint, path: Path) -> None:
        """Persist the fingerprint to disk as JSON.

        Args:
            fingerprint: The captured ``RunFingerprint``.
            path: Destination file path (parent directories created automatically).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fingerprint.to_json(), encoding="utf-8")
        logger.info("RunFingerprint saved → {}", path)

    def load(self, path: Path) -> RunFingerprint:
        """Load a saved fingerprint from disk.

        Args:
            path: Path to the JSON fingerprint file.

        Returns:
            ``RunFingerprint`` instance.
        """
        return RunFingerprint.from_json(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(
        self,
        reference_path: Path,
        check_config: bool = True,
        check_git: bool = True,
        check_versions: bool = True,
    ) -> tuple[bool, list[str]]:
        """Compare the current environment against a saved fingerprint.

        Args:
            reference_path: Path to the reference fingerprint JSON.
            check_config: Whether to verify the config hash.
            check_git: Whether to verify git commit.
            check_versions: Whether to verify Python/torch versions.

        Returns:
            Tuple of ``(passed, issues)`` where ``issues`` is a list of
            human-readable discrepancy descriptions.
        """
        ref = self.load(reference_path)
        issues: list[str] = []

        current_config_hash = self.hash_config(self._config)
        current_seed = getattr(getattr(self._config, "training", None), "seed", None)
        current_commit = get_git_commit()

        if check_config and current_config_hash != ref.config_hash:
            issues.append(
                f"Config hash mismatch: current={current_config_hash[:12]}, "
                f"reference={ref.config_hash[:12]}"
            )

        if ref.seed != current_seed:
            issues.append(
                f"Seed mismatch: current={current_seed}, reference={ref.seed}"
            )

        if check_git and current_commit != ref.git_commit:
            issues.append(
                f"Git commit mismatch: current={current_commit}, "
                f"reference={ref.git_commit}"
            )

        if check_versions:
            if torch.__version__ != ref.torch_version:
                issues.append(
                    f"PyTorch version mismatch: current={torch.__version__}, "
                    f"reference={ref.torch_version}"
                )
            if np.__version__ != ref.numpy_version:
                issues.append(
                    f"NumPy version mismatch: current={np.__version__}, "
                    f"reference={ref.numpy_version}"
                )

        passed = len(issues) == 0
        if passed:
            logger.info("Reproducibility verified — all checks passed.")
        else:
            for issue in issues:
                logger.warning("Reproducibility issue: {}", issue)

        return passed, issues

    # ------------------------------------------------------------------
    # Config hashing
    # ------------------------------------------------------------------

    @staticmethod
    def hash_config(config: Any) -> str:
        """Compute a stable SHA-256 hash of any config object.

        Serialises the config via its ``__dict__`` (recursively) and hashes
        the resulting JSON string.  Handles nested dataclasses.

        Args:
            config: Any config object (dataclass, dict, or primitive).

        Returns:
            64-character lowercase hex string (SHA-256).
        """
        serialised = json.dumps(
            _serialise(config), sort_keys=True, default=str
        )
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # RNG state snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def capture_rng_state() -> dict[str, Any]:
        """Snapshot all RNG states (Python, NumPy, PyTorch, CUDA).

        Returns:
            Dict with keys "python", "numpy", "torch", "cuda" (list of tensors).
        """
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().tolist(),
        }
        if torch.cuda.is_available():
            state["cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        return state

    @staticmethod
    def restore_rng_state(state: dict[str, Any]) -> None:
        """Restore all RNG states from a previously captured snapshot.

        Args:
            state: Dict returned by ``capture_rng_state()``.
        """
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
        if "cuda" in state and torch.cuda.is_available():
            tensors = [
                torch.tensor(s, dtype=torch.uint8) for s in state["cuda"]
            ]
            torch.cuda.set_rng_state_all(tensors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialise(obj: Any) -> Any:
    """Recursively serialise a config object into JSON-compatible primitives."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialise(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    return str(obj)


def _safe_hostname() -> str:
    try:
        return platform.node()
    except Exception:
        return "unknown"
