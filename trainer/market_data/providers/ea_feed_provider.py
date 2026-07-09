"""S33-spec — EA Feed Provider (pluggable stub).

Primary AUTONOMOUS-mode inference provider. Unlike OANDA/Tiingo, this
provider represents a locally-pushed feed from a broker-side Expert Advisor
(MT4/MT5) that streams live ticks/candles to the trainer over a webhook or
local socket rather than an outbound REST client — autonomous trading needs
the exact prices/spreads the EA sees on its own broker connection, not a
third-party aggregator's view.

Reads its endpoint/shared-secret from ``EA_FEED_ENDPOINT`` /
``EA_FEED_SHARED_SECRET`` if present; otherwise every data method raises
``ProviderNotConfigured``, mirroring ``OandaProvider``/``TiingoProvider``.
"""

from __future__ import annotations

import os
from typing import Any

from trainer.market_data.providers.base_stub_provider import BaseStubProvider


class EAFeedProvider(BaseStubProvider):
    def __init__(self, api_key: str | None = None, endpoint: str | None = None) -> None:
        super().__init__(api_key=api_key if api_key is not None else os.environ.get("EA_FEED_SHARED_SECRET"))
        self._endpoint = endpoint if endpoint is not None else os.environ.get("EA_FEED_ENDPOINT")

    def name(self) -> str:
        return "ea_feed"

    def _require_configured(self) -> None:
        super()._require_configured()
        if not self._endpoint:
            from trainer.market_data.provider_interface import ProviderNotConfigured

            raise ProviderNotConfigured(f"{self.name()} has no endpoint configured — set EA_FEED_ENDPOINT")

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Seam for the real EA feed client (webhook consumer / local socket
        reader). Not implemented until an EA feed endpoint is wired up —
        raising here keeps the architecture complete without faking data."""
        raise NotImplementedError("EA feed client not yet wired — add EA_FEED_ENDPOINT and implement _request")

    def get_price(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        return self._request(f"/price/{symbol}")

    def get_candles(self, symbol: str, timeframe: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request(f"/candles/{symbol}", {"timeframe": timeframe, "count": count})

    def get_ticks(self, symbol: str, count: int = 100) -> list[dict[str, Any]]:
        self._require_configured()
        return self._request(f"/ticks/{symbol}", {"count": count})

    def get_spread(self, symbol: str) -> float:
        self._require_configured()
        price = self._request(f"/price/{symbol}")
        return float(price.get("spread", 0.0)) if isinstance(price, dict) else 0.0

    def get_orderbook(self, symbol: str) -> dict[str, Any]:
        self._require_configured()
        # An MT4/5 EA feed does not expose order-book depth — structurally
        # unavailable, same convention as Tiingo's tick/spread/orderbook gaps.
        raise NotImplementedError("EA feed does not provide order book data")

    def get_calendar(self, start: float, end: float) -> list[dict[str, Any]]:
        self._require_configured()
        raise NotImplementedError("EA feed does not provide economic calendar data")

    def get_news(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        self._require_configured()
        raise NotImplementedError("EA feed does not provide news data")
