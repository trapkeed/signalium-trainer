"""BenchmarkRunner — runs multiple models under identical experimental conditions."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from trainer.framework.metrics import MetricsEngine
from trainer.research.benchmark.conditions import BenchmarkConditions
from trainer.research.experiment.record import ExperimentRecord


def _set_global_seed(seed: int) -> None:
    """Apply seed to Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ModelRunResult:
    """Results from a single model's benchmark run.

    Attributes:
        model_name: Registry name of the model.
        experiment_id: ID of the corresponding ExperimentRecord.
        metrics: Final evaluation metric dict.
        training_time_seconds: Wall-clock training duration.
        inference_latency_ms: Median inference time per sample in milliseconds.
        model_size_mb: Serialised model size on disk in megabytes.
        gpu_memory_mb: Peak GPU memory allocated during training (0 if CPU).
        history: Epoch-by-epoch metric history (list of dicts).
        error: Error message if the run failed; None otherwise.
        status: ``"completed"`` or ``"failed"``.
    """

    model_name: str
    experiment_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    inference_latency_ms: float = 0.0
    model_size_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)
    error: Optional[str] = None
    status: str = "completed"


@dataclass
class BenchmarkResult:
    """Aggregated results from a complete benchmark run.

    Attributes:
        benchmark_id: UUID of this benchmark session.
        conditions: The BenchmarkConditions applied to all runs.
        runs: Per-model ModelRunResult list.
        total_duration_seconds: Wall-clock time for the entire benchmark.
    """

    benchmark_id: str
    conditions: BenchmarkConditions
    runs: list[ModelRunResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def get_run(self, model_name: str) -> Optional[ModelRunResult]:
        """Retrieve the run result for a specific model.

        Args:
            model_name: Registry name of the model.

        Returns:
            ModelRunResult or None if the model did not run.
        """
        for run in self.runs:
            if run.model_name == model_name:
                return run
        return None

    def completed_runs(self) -> list[ModelRunResult]:
        """Return only successful run results."""
        return [r for r in self.runs if r.status == "completed"]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "benchmark_id": self.benchmark_id,
            "conditions": self.conditions.to_dict(),
            "total_duration_seconds": self.total_duration_seconds,
            "runs": [
                {
                    "model_name": r.model_name,
                    "experiment_id": r.experiment_id,
                    "status": r.status,
                    "metrics": r.metrics,
                    "training_time_seconds": r.training_time_seconds,
                    "inference_latency_ms": r.inference_latency_ms,
                    "model_size_mb": r.model_size_mb,
                    "gpu_memory_mb": r.gpu_memory_mb,
                    "error": r.error,
                }
                for r in self.runs
            ],
        }


class BenchmarkRunner:
    """Orchestrates benchmark runs across multiple model configurations.

    Guarantees for each run:
    - Same random seed (Python + NumPy + PyTorch)
    - Same train/val split indices
    - Same evaluation metrics
    - Same device and batch size

    Example::

        conditions = BenchmarkConditions(random_seed=42, max_epochs=10)

        runner = BenchmarkRunner(conditions=conditions)

        runner.register_model(
            name="quant_v1",
            model_factory=lambda: ModelFactory.create("quant_v1", trainer_config),
        )

        result = runner.run(train_data, val_data)

    Args:
        conditions: BenchmarkConditions applied to all model runs.
        experiment_tracker: Optional ExperimentTracker for persisting records.
    """

    def __init__(
        self,
        conditions: BenchmarkConditions,
        experiment_tracker: Optional[Any] = None,
    ) -> None:
        conditions.validate()
        self._conditions = conditions
        self._tracker = experiment_tracker
        self._model_entries: list[dict[str, Any]] = []

    def register_model(
        self,
        name: str,
        model_factory: Callable[[], nn.Module],
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> "BenchmarkRunner":
        """Register a model for benchmarking.

        Args:
            name: Human-readable model name (need not match registry name).
            model_factory: Zero-argument callable that returns a fresh model instance.
            hyperparameters: Optional dict to attach to the ExperimentRecord.

        Returns:
            Self, for method chaining.
        """
        self._model_entries.append(
            {
                "name": name,
                "factory": model_factory,
                "hyperparameters": hyperparameters or {},
            }
        )
        return self

    def run(
        self,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
        criterion: Optional[nn.Module] = None,
        learning_rate: float = 1e-3,
        benchmark_id: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run all registered models under identical conditions.

        Each model gets a fresh random seed reset before training begins.

        Args:
            train_data: Tuple of (x_train, y_train) tensors.
            val_data: Tuple of (x_val, y_val) tensors.
            criterion: Loss function (default: CrossEntropyLoss).
            learning_rate: Learning rate for all models.
            benchmark_id: Optional UUID override.

        Returns:
            BenchmarkResult with per-model metrics.
        """
        import uuid as _uuid

        bid = benchmark_id or str(_uuid.uuid4())
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        c = self._conditions
        benchmark_start = time.perf_counter()
        runs: list[ModelRunResult] = []

        for entry in self._model_entries:
            run_result = self._run_single(
                entry=entry,
                train_data=train_data,
                val_data=val_data,
                criterion=criterion,
                learning_rate=learning_rate,
            )
            runs.append(run_result)

        total = time.perf_counter() - benchmark_start
        return BenchmarkResult(
            benchmark_id=bid,
            conditions=c,
            runs=runs,
            total_duration_seconds=round(total, 3),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_single(
        self,
        entry: dict[str, Any],
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
        learning_rate: float,
    ) -> ModelRunResult:
        """Train and evaluate one model under the benchmark conditions."""
        c = self._conditions
        name = entry["name"]

        _set_global_seed(c.random_seed)

        device = torch.device(c.device)
        x_train, y_train = train_data[0].to(device), train_data[1].to(device)
        x_val, y_val = val_data[0].to(device), val_data[1].to(device)

        num_classes = int(y_train.max().item()) + 1
        engine = MetricsEngine(num_classes=num_classes)

        # Build fresh model
        try:
            model = entry["factory"]()
            model = model.to(device)
        except Exception as exc:
            return ModelRunResult(
                model_name=name, status="failed", error=f"Model build failed: {exc}"
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        history: list[dict[str, float]] = []

        # Peak GPU memory tracking
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_start = time.perf_counter()
        try:
            dataset_size = x_train.shape[0]
            bs = min(c.batch_size, dataset_size)

            for epoch in range(c.max_epochs):
                model.train()
                idx = torch.randperm(dataset_size, device=device)
                epoch_loss = 0.0
                n_batches = 0

                for start in range(0, dataset_size, bs):
                    batch_idx = idx[start : start + bs]
                    xb = x_train[batch_idx]
                    yb = y_train[batch_idx]
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                # Validation
                model.eval()
                engine.reset()
                with torch.no_grad():
                    val_logits = model(x_val)
                    engine.update(val_logits, y_val)

                epoch_metrics = engine.compute()
                epoch_metrics["train_loss"] = epoch_loss / max(n_batches, 1)
                history.append(epoch_metrics)

        except Exception as exc:
            return ModelRunResult(
                model_name=name,
                status="failed",
                error=f"Training failed at epoch: {exc}",
                history=history,
            )

        training_time = time.perf_counter() - train_start

        # Final validation metrics
        model.eval()
        engine.reset()
        with torch.no_grad():
            val_logits = model(x_val)
            engine.update(val_logits, y_val)
        final_metrics = engine.compute()

        # Inference latency (median over 50 forward passes on 1 sample)
        latency_ms = self._measure_latency(model, x_val[:1], device)

        # Model size
        size_mb = self._model_size_mb(model)

        # GPU peak memory
        gpu_mem_mb = 0.0
        if torch.cuda.is_available() and device.type == "cuda":
            gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

        # Build ExperimentRecord
        record = ExperimentRecord(
            name=f"benchmark-{name}",
            model_name=name,
            hyperparameters=entry.get("hyperparameters", {}),
            metrics={
                **final_metrics,
                "inference_latency_ms": latency_ms,
                "model_size_mb": size_mb,
                "gpu_memory_mb": gpu_mem_mb,
            },
        )
        record.complete(
            {**final_metrics, "inference_latency_ms": latency_ms, "model_size_mb": size_mb}
        )
        if self._tracker is not None:
            self._tracker.save(record)

        return ModelRunResult(
            model_name=name,
            experiment_id=record.experiment_id,
            metrics={
                **final_metrics,
                "inference_latency_ms": latency_ms,
                "model_size_mb": size_mb,
                "gpu_memory_mb": gpu_mem_mb,
            },
            training_time_seconds=round(training_time, 3),
            inference_latency_ms=latency_ms,
            model_size_mb=size_mb,
            gpu_memory_mb=gpu_mem_mb,
            history=history,
            status="completed",
        )

    @staticmethod
    def _measure_latency(model: nn.Module, x: torch.Tensor, device: torch.device) -> float:
        """Measure median inference latency in ms over 50 passes."""
        latencies = []
        with torch.no_grad():
            for _ in range(50):
                start = time.perf_counter()
                _ = model(x)
                latencies.append((time.perf_counter() - start) * 1000)
        latencies.sort()
        return round(latencies[len(latencies) // 2], 4)

    @staticmethod
    def _model_size_mb(model: nn.Module) -> float:
        """Compute model parameter size in MB."""
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        return round(total_bytes / 1e6, 4)
