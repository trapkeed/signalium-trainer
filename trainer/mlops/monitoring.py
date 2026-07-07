"""Phase 18 — Production Monitoring.

Continuously monitors production model health metrics:

  - Prediction latency (ms)
  - Inference throughput (predictions/second)
  - CPU usage (%)
  - GPU usage (%)
  - Memory usage (MB)
  - API response times (ms)
  - Prediction confidence (rolling)
  - Prediction distribution (buy/sell/no_trade ratios)
  - Trade frequency (signals per hour)
  - Signal quality (hit rate proxy)

Usage::

    monitor = ProductionMonitor(window=500)
    monitor.record_inference(
        model_id="m1",
        latency_ms=12.5,
        confidence=0.78,
        predicted_class="buy",
        cpu_pct=45.0,
        memory_mb=2048.0,
    )
    snapshot = monitor.get_snapshot("m1")
    print(snapshot.avg_latency_ms, snapshot.throughput_rps)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Inference record
# ---------------------------------------------------------------------------


@dataclass
class InferenceRecord:
    """One inference event for production monitoring."""

    model_id: str
    latency_ms: float = 0.0
    api_response_ms: float = 0.0
    throughput_hint: float = 1.0  # predictions submitted in this batch
    cpu_pct: float = 0.0
    gpu_pct: float = 0.0
    memory_mb: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    predicted_class: str = "no_trade"
    buy_prob: float = 0.33
    sell_prob: float = 0.33
    no_trade_prob: float = 0.34
    instrument: str = ""
    timeframe: str = ""
    regime: str = ""
    is_signal: bool = False  # True if predicted != no_trade
    unix_ts: float = field(default_factory=time.time)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------------
# Monitoring snapshot
# ---------------------------------------------------------------------------


@dataclass
class MonitoringSnapshot:
    """Aggregated monitoring metrics for one model over the rolling window.

    Attributes:
        model_id: Model these metrics belong to.
        window: Rolling window size.
        n_records: Records in window.
        avg_latency_ms: Mean inference latency.
        p95_latency_ms: 95th percentile latency.
        p99_latency_ms: 99th percentile latency.
        max_latency_ms: Maximum latency in window.
        avg_api_response_ms: Mean API response time.
        throughput_rps: Estimated predictions per second.
        avg_cpu_pct: Mean CPU usage.
        avg_gpu_pct: Mean GPU usage.
        avg_memory_mb: Mean memory usage.
        peak_memory_mb: Peak memory in window.
        avg_confidence: Mean prediction confidence.
        avg_uncertainty: Mean prediction uncertainty.
        confidence_trend: "stable" | "declining" | "improving" | "insufficient".
        distribution: Dict of predicted class → fraction.
        signal_rate: Fraction of predictions that are actionable (not no_trade).
        buy_rate: Fraction predicted buy.
        sell_rate: Fraction predicted sell.
        no_trade_rate: Fraction predicted no_trade.
        latency_alert: Whether latency exceeds threshold.
        memory_alert: Whether memory exceeds threshold.
    """

    model_id: str
    window: int = 500
    n_records: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    avg_api_response_ms: float = 0.0
    throughput_rps: float = 0.0
    avg_cpu_pct: float = 0.0
    avg_gpu_pct: float = 0.0
    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_confidence: float = 0.5
    avg_uncertainty: float = 0.5
    confidence_trend: str = "insufficient"
    distribution: dict[str, float] = field(default_factory=dict)
    signal_rate: float = 0.0
    buy_rate: float = 0.0
    sell_rate: float = 0.0
    no_trade_rate: float = 0.0
    latency_alert: bool = False
    memory_alert: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "window": self.window,
            "n_records": self.n_records,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "avg_api_response_ms": self.avg_api_response_ms,
            "throughput_rps": self.throughput_rps,
            "avg_cpu_pct": self.avg_cpu_pct,
            "avg_gpu_pct": self.avg_gpu_pct,
            "avg_memory_mb": self.avg_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_confidence": self.avg_confidence,
            "avg_uncertainty": self.avg_uncertainty,
            "confidence_trend": self.confidence_trend,
            "distribution": self.distribution,
            "signal_rate": self.signal_rate,
            "buy_rate": self.buy_rate,
            "sell_rate": self.sell_rate,
            "no_trade_rate": self.no_trade_rate,
            "latency_alert": self.latency_alert,
            "memory_alert": self.memory_alert,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Production monitor
# ---------------------------------------------------------------------------


class ProductionMonitor:
    """Rolling-window production monitor for all active models.

    Args:
        window: Number of inference records to keep per model.
        latency_alert_ms: Latency threshold for alerts.
        memory_alert_mb: Memory threshold for alerts.
        confidence_alert_threshold: Avg confidence below this → alert.
    """

    def __init__(
        self,
        window: int = 500,
        latency_alert_ms: float = 100.0,
        memory_alert_mb: float = 8192.0,
        confidence_alert_threshold: float = 0.40,
    ) -> None:
        self.window = window
        self.latency_alert_ms = latency_alert_ms
        self.memory_alert_mb = memory_alert_mb
        self.confidence_alert_threshold = confidence_alert_threshold
        self._buffers: dict[str, deque[InferenceRecord]] = {}
        self._all_records: dict[str, list[InferenceRecord]] = {}

    def record_inference(
        self,
        model_id: str,
        latency_ms: float = 0.0,
        api_response_ms: float = 0.0,
        confidence: float = 0.5,
        uncertainty: float = 0.5,
        predicted_class: str = "no_trade",
        buy_prob: float = 0.33,
        sell_prob: float = 0.33,
        no_trade_prob: float = 0.34,
        cpu_pct: float = 0.0,
        gpu_pct: float = 0.0,
        memory_mb: float = 0.0,
        instrument: str = "",
        timeframe: str = "",
        regime: str = "",
    ) -> InferenceRecord:
        """Record one inference event."""
        is_signal = predicted_class in ("buy", "sell")
        rec = InferenceRecord(
            model_id=model_id,
            latency_ms=latency_ms,
            api_response_ms=api_response_ms or latency_ms,
            cpu_pct=cpu_pct,
            gpu_pct=gpu_pct,
            memory_mb=memory_mb,
            confidence=confidence,
            uncertainty=uncertainty,
            predicted_class=predicted_class,
            buy_prob=buy_prob,
            sell_prob=sell_prob,
            no_trade_prob=no_trade_prob,
            instrument=instrument,
            timeframe=timeframe,
            regime=regime,
            is_signal=is_signal,
        )
        if model_id not in self._buffers:
            self._buffers[model_id] = deque(maxlen=self.window)
            self._all_records[model_id] = []
        self._buffers[model_id].append(rec)
        self._all_records[model_id].append(rec)
        return rec

    def get_snapshot(self, model_id: str) -> MonitoringSnapshot:
        """Compute monitoring snapshot from the rolling window."""
        buf = list(self._buffers.get(model_id, []))
        n = len(buf)
        if n == 0:
            return MonitoringSnapshot(model_id=model_id, window=self.window)

        latencies = sorted(r.latency_ms for r in buf)
        avg_lat = sum(latencies) / n
        p95 = latencies[int(0.95 * n)] if n >= 20 else max(latencies)
        p99 = latencies[int(0.99 * n)] if n >= 100 else max(latencies)

        avg_api = sum(r.api_response_ms for r in buf) / n
        avg_cpu = sum(r.cpu_pct for r in buf) / n
        avg_gpu = sum(r.gpu_pct for r in buf) / n
        avg_mem = sum(r.memory_mb for r in buf) / n
        peak_mem = max(r.memory_mb for r in buf)
        avg_conf = sum(r.confidence for r in buf) / n
        avg_unc = sum(r.uncertainty for r in buf) / n

        n_buy = sum(1 for r in buf if r.predicted_class == "buy")
        n_sell = sum(1 for r in buf if r.predicted_class == "sell")
        n_nt = sum(1 for r in buf if r.predicted_class == "no_trade")
        signal_rate = (n_buy + n_sell) / n

        if n >= 20:
            half = n // 2
            first_conf = sum(r.confidence for r in buf[:half]) / half
            second_conf = sum(r.confidence for r in buf[half:]) / (n - half)
            if second_conf > first_conf + 0.03:
                trend = "improving"
            elif second_conf < first_conf - 0.03:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient"

        if n >= 2:
            time_span = buf[-1].unix_ts - buf[0].unix_ts
            throughput = n / time_span if time_span > 0 else float(n)
        else:
            throughput = 0.0

        return MonitoringSnapshot(
            model_id=model_id,
            window=self.window,
            n_records=n,
            avg_latency_ms=round(avg_lat, 3),
            p95_latency_ms=round(p95, 3),
            p99_latency_ms=round(p99, 3),
            max_latency_ms=round(max(latencies), 3),
            avg_api_response_ms=round(avg_api, 3),
            throughput_rps=round(throughput, 4),
            avg_cpu_pct=round(avg_cpu, 2),
            avg_gpu_pct=round(avg_gpu, 2),
            avg_memory_mb=round(avg_mem, 2),
            peak_memory_mb=round(peak_mem, 2),
            avg_confidence=round(avg_conf, 4),
            avg_uncertainty=round(avg_unc, 4),
            confidence_trend=trend,
            distribution={"buy": n_buy / n, "sell": n_sell / n, "no_trade": n_nt / n},
            signal_rate=round(signal_rate, 4),
            buy_rate=round(n_buy / n, 4),
            sell_rate=round(n_sell / n, 4),
            no_trade_rate=round(n_nt / n, 4),
            latency_alert=avg_lat > self.latency_alert_ms,
            memory_alert=peak_mem > self.memory_alert_mb,
        )

    def all_snapshots(self) -> dict[str, MonitoringSnapshot]:
        return {mid: self.get_snapshot(mid) for mid in self._buffers}

    def n_inferences(self, model_id: str) -> int:
        return len(self._all_records.get(model_id, []))

    def total_inferences(self) -> int:
        return sum(len(v) for v in self._all_records.values())

    def latency_trend(self, model_id: str, recent: int = 50) -> dict[str, Any]:
        """Return latency trend data for a model."""
        buf = list(self._buffers.get(model_id, []))[-recent:]
        return {
            "chart_type": "line",
            "model_id": model_id,
            "data": [
                {"idx": i, "latency_ms": round(r.latency_ms, 2), "timestamp": r.timestamp}
                for i, r in enumerate(buf)
            ],
        }

    def confidence_trend_chart(
        self, model_id: str, recent: int = 100, window: int = 20
    ) -> dict[str, Any]:
        """Return rolling confidence chart data."""
        buf = list(self._buffers.get(model_id, []))[-recent:]
        data = []
        for i in range(len(buf)):
            w = buf[max(0, i - window) : i + 1]
            avg = sum(r.confidence for r in w) / len(w)
            data.append({"idx": i, "rolling_confidence": round(avg, 4)})
        return {
            "chart_type": "line",
            "model_id": model_id,
            "data": data,
        }
