"""Shared base for pluggable-but-not-yet-configured providers.

Both OANDA and Tiingo stubs subclass this: they implement ``connect()`` to
check for an API key (via env var, injected explicitly, or a Replit
connector — whichever the concrete subclass wires up) and raise
``ProviderNotConfigured`` from every data method until credentials are
present. This means the provider architecture (interface, manager,
failover, symbol normalization) is fully exercisable and testable today,
without needing real OANDA/Tiingo keys.
"""

from __future__ import annotations

import time
from typing import Any

from trainer.market_data.provider_interface import HealthStatus, ProviderNotConfigured


class BaseStubProvider:
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._connected = False

    def _require_configured(self) -> None:
        if not self._api_key:
            raise ProviderNotConfigured(
                f"{self.name()} has no API key configured — set one before calling data methods"
            )

    def connect(self) -> None:
        self._require_configured()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def health_check(self) -> HealthStatus:
        started = time.time()
        if not self._api_key:
            return HealthStatus(self.name(), healthy=False, detail="no API key configured")
        return HealthStatus(
            self.name(), healthy=self._connected,
            latency_ms=(time.time() - started) * 1000.0,
            detail="connected" if self._connected else "configured but not connected",
        )

    def get_price(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        raise NotImplementedError

    def get_candles(self, symbol: str, timeframe: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        raise NotImplementedError

    def get_ticks(self, symbol: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        raise NotImplementedError

    def get_spread(self, symbol: str) -> float:
        self._require_configured()
        raise NotImplementedError

    def get_orderbook(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        raise NotImplementedError

    def get_calendar(self, start: float, end: float) -> list[dict[str, Any]]:
        self._require_configured()
        raise NotImplementedError

    def get_news(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        self._require_configured()
        raise NotImplementedError
