"""Phase 25 — Performance Benchmark Engine (9 metrics)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.cert.cert_types import CheckStatus, BenchmarkMetric, clamp
from trainer.cert.cert_models import CertificationInput, BenchmarkResult


# (threshold, unit, higher_is_better)
_METRIC_CONFIG: Dict[str, tuple] = {
    BenchmarkMetric.INFERENCE_LATENCY.value: (50.0,   "ms",   False),   # < 50ms
    BenchmarkMetric.TRAINING_TIME.value:     (3600.0, "s",    False),   # < 1 hour
    BenchmarkMetric.GPU_USAGE.value:         (95.0,   "%",    False),   # < 95%
    BenchmarkMetric.CPU_USAGE.value:         (90.0,   "%",    False),   # < 90%
    BenchmarkMetric.MEMORY_USAGE.value:      (16384.0,"MB",   False),   # < 16GB
    BenchmarkMetric.DISK_USAGE.value:        (10240.0,"MB",   False),   # < 10GB
    BenchmarkMetric.MODEL_SIZE.value:        (2048.0, "MB",   False),   # < 2GB
    BenchmarkMetric.API_THROUGHPUT.value:    (100.0,  "rps",  True),    # > 100 rps
    BenchmarkMetric.RESPONSE_TIME.value:     (200.0,  "ms",   False),   # < 200ms
}

# Warning threshold as fraction of limit
_WARN_FRACTION = 0.80


def _eval_status(
    value: float, threshold: float, higher_is_better: bool
) -> CheckStatus:
    if higher_is_better:
        if value >= threshold:
            return CheckStatus.PASS
        if value >= threshold * _WARN_FRACTION:
            return CheckStatus.WARNING
        return CheckStatus.FAIL
    else:
        if value <= threshold * _WARN_FRACTION:
            return CheckStatus.PASS
        if value <= threshold:
            return CheckStatus.WARNING
        return CheckStatus.FAIL


class BenchmarkEngine:
    """
    Evaluates 9 performance benchmark metrics.

    Values are taken from inp.benchmark_values when provided;
    otherwise a default (simulated) estimate is used.
    """

    # Default simulated values (good-enough baseline)
    _DEFAULTS: Dict[str, float] = {
        BenchmarkMetric.INFERENCE_LATENCY.value: 25.0,
        BenchmarkMetric.TRAINING_TIME.value:     1800.0,
        BenchmarkMetric.GPU_USAGE.value:         75.0,
        BenchmarkMetric.CPU_USAGE.value:         60.0,
        BenchmarkMetric.MEMORY_USAGE.value:      4096.0,
        BenchmarkMetric.DISK_USAGE.value:        512.0,
        BenchmarkMetric.MODEL_SIZE.value:        256.0,
        BenchmarkMetric.API_THROUGHPUT.value:    200.0,
        BenchmarkMetric.RESPONSE_TIME.value:     80.0,
    }

    def run_all(self, inp: CertificationInput) -> List[BenchmarkResult]:
        results = []
        for metric in BenchmarkMetric:
            threshold, unit, higher = _METRIC_CONFIG[metric.value]
            value = inp.benchmark_values.get(metric.value, self._DEFAULTS[metric.value])
            status = _eval_status(value, threshold, higher)
            results.append(BenchmarkResult(
                metric=metric,
                value=value,
                unit=unit,
                threshold=threshold,
                status=status,
            ))
        passed = sum(1 for r in results if r.status == CheckStatus.PASS)
        logger.debug(f"[BE] Benchmark: {passed}/{len(results)} metrics passing")
        return results

    def overall_score(self, results: List[BenchmarkResult]) -> float:
        """Convert benchmark results to a 0-100 score."""
        if not results:
            return 50.0
        weights = {
            CheckStatus.PASS:           100.0,
            CheckStatus.WARNING:         60.0,
            CheckStatus.FAIL:            10.0,
            CheckStatus.NOT_APPLICABLE:  50.0,
        }
        total = sum(weights.get(r.status, 50.0) for r in results)
        return clamp(total / len(results))
