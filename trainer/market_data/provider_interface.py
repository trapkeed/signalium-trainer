"""S30-spec — Pluggable Market Data Provider Interface.

Every concrete provider (OANDA, Tiingo, and any future provider) implements
this same surface, so ``ProviderManager`` can hot-switch and fail over
between them without any provider-specific branching elsewhere in the
codebase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderError(Exception):
    """Base class for provider-level failures (used for failover triggers)."""


class ProviderNotConfigured(ProviderError):
    """Raised when a provider has no credentials/API key configured yet.

    Distinct from a transient failure — this is a permanent-until-configured
    state, and ``ProviderManager`` treats it the same as any other failure
    for failover purposes but logs it differently so operators know to
    supply credentials rather than investigate an outage.
    """


class ProviderTimeout(ProviderError):
    pass


class StaleDataError(ProviderError):
    pass


class SymbolNotSupportedError(ProviderError):
    pass


class TimeframeNotSupportedError(ProviderError):
    pass


@dataclass
class HealthStatus:
    provider: str
    healthy: bool
    latency_ms: float | None = None
    detail: str = ""
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "healthy": self.healthy,
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "checked_at": self.checked_at,
        }


class MarketDataProvider(Protocol):
    """Interface every market data provider must implement.

    Concrete providers (see ``providers/oanda_provider.py``,
    ``providers/tiingo_provider.py``) are pluggable stubs until API
    credentials are supplied — calling any data method before then raises
    ``ProviderNotConfigured`` rather than silently returning fake data.
    """

    def name(self) -> str: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def health_check(self) -> HealthStatus: ...

    def get_price(self, symbol: str) -> dict[str, Any]: ...

    def get_candles(
        self, symbol: str, timeframe: str, count: int = 100
    ) -> list[dict[str, Any]]: ...

    def get_ticks(self, symbol: str, count: int = 100) -> list[dict[str, Any]]: ...

    def get_spread(self, symbol: str) -> float: ...

    def get_orderbook(self, symbol: str) -> dict[str, Any]: ...

    def get_calendar(self, start: float, end: float) -> list[dict[str, Any]]: ...

    def get_news(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]: ...
