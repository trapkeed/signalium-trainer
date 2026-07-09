"""S30-spec — OANDA Provider (pluggable stub).

Primary LIVE-inference provider per S30 priority ordering. Reads its API
key from ``OANDA_API_KEY`` (and account id from ``OANDA_ACCOUNT_ID``) if
present; otherwise every data method raises ``ProviderNotConfigured``.
Real HTTP wiring is intentionally deferred — this class defines exactly
the seams a future real implementation fills in (``_request``), so
swapping the stub for a live client requires no changes to
``ProviderManager`` or anything upstream.
"""

from __future__ import annotations

import os
from typing import Any

from trainer.market_data.providers.base_stub_provider import BaseStubProvider


class OandaProvider(BaseStubProvider):
    def __init__(self, api_key: str | None = None, account_id: str | None = None) -> None:
        super().__init__(api_key=api_key if api_key is not None else os.environ.get("OANDA_API_KEY"))
        self._account_id = account_id if account_id is not None else os.environ.get("OANDA_ACCOUNT_ID")

    def name(self) -> str:
        return "oanda"

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Seam for the real HTTP client. Not implemented until OANDA
        credentials are supplied and this stub is graduated to a live
        client — deliberately raising here (rather than a silent mock)
        keeps the "implement only missing functionality" boundary honest:
        the architecture is complete, the network call is not."""
        raise NotImplementedError("OANDA HTTP client not yet wired — add credentials and implement _request")

    def get_price(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        return self._request(f"/pricing", {"instruments": symbol})

    def get_candles(self, symbol: str, timeframe: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request(f"/instruments/{symbol}/candles", {"granularity": timeframe, "count": count})

    def get_ticks(self, symbol: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request(f"/instruments/{symbol}/ticks", {"count": count})

    def get_spread(self, symbol: str) -> float:
        self._require_configured()
        price = self._request("/pricing", {"instruments": symbol})
        return float(price.get("spread", 0.0)) if isinstance(price, dict) else 0.0

    def get_orderbook(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        return self._request(f"/instruments/{symbol}/orderBook")

    def get_calendar(self, start: float, end: float) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request("/calendar", {"start": start, "end": end})

    def get_news(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request("/news", {"symbol": symbol, "limit": limit})
