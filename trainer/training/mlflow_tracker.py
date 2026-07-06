"""MLflow experiment tracking integration."""

from __future__ import annotations

import dataclasses
import platform
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger

from trainer.config.schema import MLflowConfig, TrainerConfig
from trainer.utils.git import get_git_commit, is_dirty
from trainer.utils.gpu import get_gpu_info


class MLflowTracker:
    """Wraps MLflow to track experiments, metrics, and artifacts.

    Args:
        config: MLflowConfig section of TrainerConfig.
        run_id: Optional existing run ID to resume logging into.
    """

    def __init__(self, config: MLflowConfig, run_id: str | None = None) -> None:
        self._config = config
        self._run_id = run_id
        self._active_run: mlflow.ActiveRun | None = None

        if config.enabled:
            mlflow.set_tracking_uri(config.tracking_uri)
            mlflow.set_experiment(config.experiment_name)

    def start_run(self, trainer_config: TrainerConfig) -> str:
        """Start an MLflow run and log initial metadata.

        Args:
            trainer_config: Full config to log as parameters.

        Returns:
            Active run ID.
        """
        if not self._config.enabled:
            return "disabled"

        self._active_run = mlflow.start_run(
            run_id=self._run_id,
            run_name=self._config.run_name,
            tags={
                **self._config.tags,
                "environment": trainer_config.environment.value,
                "git_commit": get_git_commit() or "unknown",
                "git_dirty": str(is_dirty()),
                "python": platform.python_version(),
                "platform": platform.system(),
            },
        )
        run_id = self._active_run.info.run_id
        logger.info("MLflow run started: {}", run_id)

        # Log flat params from config
        flat_params = self._flatten(dataclasses.asdict(trainer_config))
        for key, value in flat_params.items():
            try:
                mlflow.log_param(key, str(value)[:250])
            except Exception:
                pass

        # Log GPU info
        gpu_info = get_gpu_info()
        if gpu_info.get("available"):
            for dev in gpu_info.get("devices", []):
                mlflow.set_tag(f"gpu.{dev['index']}.name", dev["name"])
                mlflow.set_tag(f"gpu.{dev['index']}.memory_gb", dev["total_memory_gb"])

        return run_id

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics.

        Args:
            metrics: Metric name → float value.
            step: Optional global step counter.
        """
        if not self._config.enabled:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        """Log a file or directory as an artifact.

        Args:
            path: Path to the file or directory.
        """
        if not self._config.enabled or not self._config.log_artifacts:
            return
        if path.is_dir():
            mlflow.log_artifacts(str(path))
        else:
            mlflow.log_artifact(str(path))

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run.

        Args:
            status: "FINISHED", "FAILED", or "KILLED".
        """
        if not self._config.enabled:
            return
        mlflow.end_run(status=status)
        logger.info("MLflow run ended with status: {}", status)

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Recursively flatten a nested dict with dot-separated keys."""
        result: dict[str, Any] = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(MLflowTracker._flatten(v, prefix=full_key))
            else:
                result[full_key] = v
        return result
