"""S30-spec — Provider Manager: priority ordering, health-based failover,
and hot switching between market data providers.

Priority ordering per the spec:
  - LIVE / inference context:    [OANDA, Tiingo]
  - TRAINING / historical context: [Tiingo, OANDA]

Failover triggers named in the spec, all handled here rather than pushed
onto individual providers:
  - request timeout
  - stale data (caller-supplied staleness check via ``max_staleness_s``)
  - missing symbol (provider raises ``SymbolNotSupportedError``)
  - missing timeframe (provider raises ``TimeframeNotSupportedError``)
  - excessive latency (measured around every call, threshold-gated)

On any trigger, the manager marks the current provider degraded for a
cooldown window and hot-switches to the next candidate in priority order
for that context — no restart or reconnection cycle required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.market_data.provider_interface import (
    HealthStatus,
    MarketDataProvider,
    ProviderError,
    ProviderTimeout,
    StaleDataError,
)


class DataContext(str, Enum):
    LIVE = "live"
    TRAINING = "training"


@dataclass
class _ProviderState:
    provider: MarketDataProvider
    degraded_until: float = 0.0
    consecutive_failures: int = 0
    last_failure_reason: str = ""

    def is_degraded(self, now: float) -> bool:
        return now < self.degraded_until


@dataclass
class FailoverEvent:
    context: DataContext
    from_provider: str
    to_provider: str
    reason: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.value,
            "from_provider": self.from_provider,
            "to_provider": self.to_provider,
            "reason": self.reason,
            "at": self.at,
        }


class AllProvidersUnavailable(ProviderError):
    pass


class ProviderManager:
    def __init__(
        self,
        live_priority: list[MarketDataProvider],
        training_priority: list[MarketDataProvider],
        *,
        latency_threshold_ms: float = 2000.0,
        degrade_cooldown_s: float = 60.0,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._states: dict[DataContext, list[_ProviderState]] = {
            DataContext.LIVE: [_ProviderState(p) for p in live_priority],
            DataContext.TRAINING: [_ProviderState(p) for p in training_priority],
        }
        self._latency_threshold_ms = latency_threshold_ms
        self._degrade_cooldown_s = degrade_cooldown_s
        self._max_consecutive_failures = max_consecutive_failures
        self._failover_log: list[FailoverEvent] = []

    def _ordered_states(self, context: DataContext, now: float) -> list[_ProviderState]:
        states = self._states[context]
        healthy = [s for s in states if not s.is_degraded(now)]
        degraded = [s for s in states if s.is_degraded(now)]
        return healthy + degraded  # prefer non-degraded, but still try degraded as last resort

    def active_provider(self, context: DataContext) -> MarketDataProvider:
        now = time.time()
        candidates = self._ordered_states(context, now)
        if not candidates:
            raise AllProvidersUnavailable(f"No providers configured for context {context.value!r}")
        return candidates[0].provider

    def health_snapshot(self, context: DataContext) -> list[dict[str, Any]]:
        now = time.time()
        out = []
        for state in self._states[context]:
            try:
                health = state.provider.health_check()
            except Exception as exc:  # noqa: BLE001
                health = HealthStatus(state.provider.name(), healthy=False, detail=str(exc))
            out.append(
                {
                    **health.to_dict(),
                    "degraded": state.is_degraded(now),
                    "consecutive_failures": state.consecutive_failures,
                }
            )
        return out

    def _mark_degraded(
        self, context: DataContext, state: _ProviderState, reason: str, now: float
    ) -> None:
        state.consecutive_failures += 1
        state.last_failure_reason = reason
        if state.consecutive_failures >= self._max_consecutive_failures:
            state.degraded_until = now + self._degrade_cooldown_s

    def _mark_success(self, state: _ProviderState) -> None:
        state.consecutive_failures = 0
        state.degraded_until = 0.0

    def call(
        self,
        context: DataContext,
        method_name: str,
        *args: Any,
        max_staleness_s: float | None = None,
        staleness_extractor: Callable[[Any], float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke ``method_name`` on the highest-priority healthy provider
        for ``context``, failing over to the next candidate on any trigger.

        ``staleness_extractor``, if given, receives the raw result and
        returns its data timestamp; if ``now - timestamp > max_staleness_s``
        the result is treated as a failure trigger (stale data) and the
        manager fails over rather than returning it.
        """
        now = time.time()
        candidates = self._ordered_states(context, now)
        if not candidates:
            raise AllProvidersUnavailable(f"No providers configured for context {context.value!r}")

        last_exc: Exception | None = None
        for state in candidates:
            provider = state.provider
            method = getattr(provider, method_name, None)
            if method is None:
                last_exc = AttributeError(f"{provider.name()} has no method {method_name!r}")
                continue
            started = time.time()
            try:
                result = method(*args, **kwargs)
                latency_ms = (time.time() - started) * 1000.0

                if latency_ms > self._latency_threshold_ms:
                    self._mark_degraded(
                        context, state, f"excessive latency ({latency_ms:.0f}ms)", now
                    )
                    last_exc = ProviderTimeout(f"{provider.name()} exceeded latency threshold")
                    continue

                if max_staleness_s is not None and staleness_extractor is not None:
                    data_ts = staleness_extractor(result)
                    if (time.time() - data_ts) > max_staleness_s:
                        self._mark_degraded(context, state, "stale data", now)
                        last_exc = StaleDataError(f"{provider.name()} returned stale data")
                        continue

                self._mark_success(state)
                return result

            except Exception as exc:  # noqa: BLE001 — any provider failure is a failover trigger
                self._mark_degraded(context, state, str(exc), now)
                last_exc = exc
                self._failover_log.append(
                    FailoverEvent(
                        context=context,
                        from_provider=provider.name(),
                        to_provider="(next candidate)",
                        reason=str(exc),
                    )
                )
                logger.warning(
                    f"ProviderManager: {provider.name()} failed for {method_name!r}: {exc!r}, failing over"
                )
                continue

        raise AllProvidersUnavailable(
            f"All providers exhausted for context {context.value!r} calling {method_name!r}: {last_exc!r}"
        ) from last_exc

    def recent_failovers(self, limit: int = 50) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._failover_log[-limit:]]
