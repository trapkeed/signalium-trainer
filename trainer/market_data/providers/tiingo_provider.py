"""S30-spec — Tiingo Provider (pluggable stub).

Primary TRAINING (historical) provider per S30 priority ordering. Reads its
API key from ``TIINGO_API_KEY`` if present; otherwise every data method
raises ``ProviderNotConfigured``. Mirrors ``OandaProvider``'s structure —
same seam (``_request``) left for the real HTTP client.
"""

from __future__ import annotations

import os
from typing import Any

from trainer.market_data.providers.base_stub_provider import BaseStubProvider


class TiingoProvider(BaseStubProvider):
    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            api_key=api_key if api_key is not None else os.environ.get("TIINGO_API_KEY")
        )

    def name(self) -> str:
        return "tiingo"

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError(
            "Tiingo HTTP client not yet wired — add credentials and implement _request"
        )

    def get_price(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        return self._request(f"/iex/{symbol}")

    def get_candles(self, symbol: str, timeframe: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request(
            f"/tiingo/fx/{symbol}/prices", {"resampleFreq": timeframe, "count": count}
        )

    def get_ticks(self, symbol: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        # Tiingo is a historical/aggregated provider — tick-level data is
        # not part of its API. This deliberately raises rather than
        # returning an empty list, so ProviderManager's failover treats
        # "this provider structurally cannot serve ticks" the same as any
        # other unavailability rather than a silent empty success.
        raise NotImplementedError("Tiingo does not provide tick-level data")

    def get_spread(self, symbol: str) -> float:
        self._require_configured()
        raise NotImplementedError("Tiingo does not provide bid/ask spread data")

    def get_orderbook(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        raise NotImplementedError("Tiingo does not provide order book data")

    def get_calendar(self, start: float, end: float) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request("/calendar", {"start": start, "end": end})

    def get_news(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request("/news", {"tickers": symbol, "limit": limit})
