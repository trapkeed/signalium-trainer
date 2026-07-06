"""Training & Inference Profiler — Phase 14.9B.

Benchmarks:
  - Training speed   (batches/sec, samples/sec, time-per-epoch estimate)
  - Inference speed  (ms/sample, samples/sec, p50/p95/p99 latency)
  - GPU memory usage (peak MB, average MB, reserved MB)
  - CPU memory usage (RSS MB)
  - Model size       (total params, trainable params, disk size estimate MB)

Usage::

    profiler = TrainingProfiler(model, config, device)

    speed = profiler.profile_training_speed(train_loader, n_batches=50)
    infer = profiler.profile_inference_speed(n_runs=200)
    mem   = profiler.profile_gpu_memory(train_loader, n_batches=20)
    size  = profiler.profile_model_size()

    report = profiler.full_benchmark(train_loader)
    print(report.summary())
"""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingSpeedResult:
    batches_per_sec: float
    samples_per_sec: float
    ms_per_batch: float
    estimated_epoch_minutes: float
    n_batches_profiled: int
    batch_size: int


@dataclass
class InferenceSpeedResult:
    ms_per_sample: float
    samples_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    n_runs: int
    batch_size: int
    device: str


@dataclass
class GPUMemoryResult:
    peak_mb: float
    allocated_mb: float
    reserved_mb: float
    available: bool


@dataclass
class CPUMemoryResult:
    rss_mb: float


@dataclass
class ModelSizeResult:
    total_params: int
    trainable_params: int
    non_trainable_params: int
    estimated_mb: float
    layers: int


@dataclass
class BenchmarkReport:
    training_speed: Optional[TrainingSpeedResult] = None
    inference_speed: Optional[InferenceSpeedResult] = None
    gpu_memory: Optional[GPUMemoryResult] = None
    cpu_memory: Optional[CPUMemoryResult] = None
    model_size: Optional[ModelSizeResult] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "BENCHMARK REPORT — Phase 14.9B", "=" * 60]

        if self.model_size:
            s = self.model_size
            lines += [
                f"Model Size",
                f"  Total params:      {s.total_params:,}",
                f"  Trainable params:  {s.trainable_params:,}",
                f"  Estimated size:    {s.estimated_mb:.2f} MB",
            ]

        if self.training_speed:
            s = self.training_speed
            lines += [
                "",
                f"Training Speed",
                f"  Batches/sec:       {s.batches_per_sec:.2f}",
                f"  Samples/sec:       {s.samples_per_sec:.2f}",
                f"  ms/batch:          {s.ms_per_batch:.2f}",
                f"  Epoch estimate:    {s.estimated_epoch_minutes:.1f} min "
                f"(per {s.n_batches_profiled} sampled batches)",
            ]

        if self.inference_speed:
            s = self.inference_speed
            lines += [
                "",
                f"Inference Speed ({s.device})",
                f"  ms/sample:         {s.ms_per_sample:.3f}",
                f"  Samples/sec:       {s.samples_per_sec:.2f}",
                f"  Latency p50:       {s.latency_p50_ms:.3f} ms",
                f"  Latency p95:       {s.latency_p95_ms:.3f} ms",
                f"  Latency p99:       {s.latency_p99_ms:.3f} ms",
            ]

        if self.gpu_memory:
            s = self.gpu_memory
            if s.available:
                lines += [
                    "",
                    f"GPU Memory",
                    f"  Peak allocated:   {s.peak_mb:.1f} MB",
                    f"  Current allocated:{s.allocated_mb:.1f} MB",
                    f"  Reserved:         {s.reserved_mb:.1f} MB",
                ]

        if self.cpu_memory:
            lines += [
                "",
                f"CPU Memory",
                f"  RSS:               {self.cpu_memory.rss_mb:.1f} MB",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_speed": asdict(self.training_speed) if self.training_speed else None,
            "inference_speed": asdict(self.inference_speed) if self.inference_speed else None,
            "gpu_memory": asdict(self.gpu_memory) if self.gpu_memory else None,
            "cpu_memory": asdict(self.cpu_memory) if self.cpu_memory else None,
            "model_size": asdict(self.model_size) if self.model_size else None,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------


class TrainingProfiler:
    """Benchmarks training and inference performance.

    Args:
        model: The model to profile.
        device: The torch device.
        config: Optional TrainerConfig for context.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Any = None,
    ) -> None:
        self._model = model
        self._device = device
        self._config = config

    # ------------------------------------------------------------------
    # Training speed
    # ------------------------------------------------------------------

    def profile_training_speed(
        self,
        loader: Any,
        n_batches: int = 50,
    ) -> TrainingSpeedResult:
        """Measure batches/sec and samples/sec during a forward+backward pass.

        Args:
            loader: Training DataLoader.
            n_batches: Number of batches to profile.

        Returns:
            ``TrainingSpeedResult``.
        """
        self._model.train()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        batch_size = 0
        times: list[float] = []
        _warmup = 5  # discard first N batches for cache warmup

        for i, batch in enumerate(loader):
            if i >= n_batches + _warmup:
                break

            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x = batch[0]
                y = batch[1] if not isinstance(batch[1], dict) else None
            else:
                continue

            x = x.to(self._device, non_blocking=True)
            if y is not None and not isinstance(y, dict):
                y = y.to(self._device, non_blocking=True)
            batch_size = x.shape[0]

            _sync()
            t0 = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            out = self._model(x)
            if y is not None and isinstance(out, torch.Tensor):
                try:
                    loss = criterion(out, y.long() if y.dtype == torch.float32 else y)
                    loss.backward()
                except Exception:
                    out.sum().backward()
            else:
                if isinstance(out, torch.Tensor):
                    out.sum().backward()
            optimizer.step()

            _sync()
            elapsed = time.perf_counter() - t0

            if i >= _warmup:
                times.append(elapsed)

        if not times:
            logger.warning("Training speed profile: no batches completed.")
            return TrainingSpeedResult(0, 0, 0, 0, 0, batch_size)

        avg_ms_per_batch = (sum(times) / len(times)) * 1000
        batches_per_sec = 1000 / avg_ms_per_batch if avg_ms_per_batch > 0 else 0
        samples_per_sec = batches_per_sec * batch_size
        epoch_steps_estimate = max(len(loader), n_batches)
        epoch_minutes = (avg_ms_per_batch / 1000) * epoch_steps_estimate / 60

        logger.info(
            "Training speed: {:.1f} batches/sec, {:.0f} samples/sec, " "{:.1f} min/epoch estimate",
            batches_per_sec,
            samples_per_sec,
            epoch_minutes,
        )
        return TrainingSpeedResult(
            batches_per_sec=batches_per_sec,
            samples_per_sec=samples_per_sec,
            ms_per_batch=avg_ms_per_batch,
            estimated_epoch_minutes=epoch_minutes,
            n_batches_profiled=len(times),
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Inference speed
    # ------------------------------------------------------------------

    def profile_inference_speed(
        self,
        input_shape: tuple[int, ...],
        n_runs: int = 200,
        batch_size: int = 1,
    ) -> InferenceSpeedResult:
        """Measure inference latency distribution.

        Args:
            input_shape: Feature tensor shape excluding batch dim,
                         e.g. ``(seq_len, input_dim)``.
            n_runs: Number of inference passes to measure.
            batch_size: Batch size for benchmarking.

        Returns:
            ``InferenceSpeedResult``.
        """
        self._model.eval()
        x = torch.zeros(batch_size, *input_shape, device=self._device)

        latencies_ms: list[float] = []
        _warmup = min(20, n_runs // 5)

        with torch.no_grad():
            for i in range(n_runs + _warmup):
                _sync()
                t0 = time.perf_counter()
                _ = self._model(x)
                _sync()
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if i >= _warmup:
                    latencies_ms.append(elapsed_ms)

        import statistics

        latencies_ms.sort()
        n = len(latencies_ms)
        p50 = latencies_ms[int(n * 0.50)]
        p95 = latencies_ms[int(n * 0.95)]
        p99 = latencies_ms[min(int(n * 0.99), n - 1)]
        avg_ms = statistics.mean(latencies_ms)
        ms_per_sample = avg_ms / batch_size
        samples_per_sec = (batch_size * 1000) / avg_ms if avg_ms > 0 else 0

        device_str = str(self._device)
        logger.info(
            "Inference: {:.3f} ms/sample | p50={:.3f} p95={:.3f} p99={:.3f} | "
            "{:.0f} samples/sec",
            ms_per_sample,
            p50,
            p95,
            p99,
            samples_per_sec,
        )
        return InferenceSpeedResult(
            ms_per_sample=ms_per_sample,
            samples_per_sec=samples_per_sec,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            n_runs=n_runs,
            batch_size=batch_size,
            device=device_str,
        )

    # ------------------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------------------

    def profile_gpu_memory(
        self,
        loader: Any,
        n_batches: int = 20,
    ) -> GPUMemoryResult:
        """Measure GPU memory allocation during forward passes.

        Args:
            loader: DataLoader to pull batches from.
            n_batches: Number of forward passes to profile.

        Returns:
            ``GPUMemoryResult``.
        """
        if not torch.cuda.is_available() or self._device.type != "cuda":
            logger.info("GPU memory profiling skipped — CUDA not available.")
            return GPUMemoryResult(0.0, 0.0, 0.0, available=False)

        torch.cuda.reset_peak_memory_stats(self._device)
        torch.cuda.empty_cache()
        gc.collect()

        self._model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_batches:
                    break
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    x = batch[0].to(self._device, non_blocking=True)
                    _ = self._model(x)

        peak = torch.cuda.max_memory_allocated(self._device) / 1e6
        alloc = torch.cuda.memory_allocated(self._device) / 1e6
        reserved = torch.cuda.memory_reserved(self._device) / 1e6

        logger.info(
            "GPU memory: peak={:.1f} MB, current={:.1f} MB, reserved={:.1f} MB",
            peak,
            alloc,
            reserved,
        )
        return GPUMemoryResult(
            peak_mb=peak, allocated_mb=alloc, reserved_mb=reserved, available=True
        )

    # ------------------------------------------------------------------
    # Model size
    # ------------------------------------------------------------------

    def profile_model_size(self) -> ModelSizeResult:
        """Compute model parameter counts and estimated disk size.

        Returns:
            ``ModelSizeResult``.
        """
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        non_trainable = total - trainable
        # Estimate: float32 = 4 bytes per param
        estimated_mb = (total * 4) / 1e6
        layers = sum(1 for _ in self._model.modules())

        logger.info(
            "Model size: {:,} total params ({:,} trainable) | ~{:.2f} MB",
            total,
            trainable,
            estimated_mb,
        )
        return ModelSizeResult(
            total_params=total,
            trainable_params=trainable,
            non_trainable_params=non_trainable,
            estimated_mb=estimated_mb,
            layers=layers,
        )

    # ------------------------------------------------------------------
    # CPU memory
    # ------------------------------------------------------------------

    def profile_cpu_memory(self) -> CPUMemoryResult:
        """Measure current process RSS memory.

        Returns:
            ``CPUMemoryResult`` with RSS in MB.
        """
        try:
            import psutil, os

            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / 1e6
        except ImportError:
            import resource

            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3

        return CPUMemoryResult(rss_mb=rss_mb)

    # ------------------------------------------------------------------
    # Full benchmark
    # ------------------------------------------------------------------

    def full_benchmark(
        self,
        loader: Any,
        input_shape: Optional[tuple[int, ...]] = None,
        n_training_batches: int = 50,
        n_inference_runs: int = 200,
    ) -> BenchmarkReport:
        """Run all profilers and return a consolidated ``BenchmarkReport``.

        Args:
            loader: DataLoader used for training-speed and GPU-memory profiling.
            input_shape: Feature shape for inference profiling. If None, skips
                         inference profiling.
            n_training_batches: Batches for training speed profile.
            n_inference_runs: Runs for inference speed profile.

        Returns:
            ``BenchmarkReport`` with all populated fields.
        """
        report = BenchmarkReport(
            metadata={
                "device": str(self._device),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
        )

        report.model_size = self.profile_model_size()
        report.training_speed = self.profile_training_speed(loader, n_batches=n_training_batches)
        report.gpu_memory = self.profile_gpu_memory(loader, n_batches=20)
        report.cpu_memory = self.profile_cpu_memory()

        if input_shape is not None:
            report.inference_speed = self.profile_inference_speed(
                input_shape=input_shape,
                n_runs=n_inference_runs,
            )

        logger.info("Full benchmark complete.\n{}", report.summary())
        return report


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _sync() -> None:
    """Synchronize CUDA if available for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
