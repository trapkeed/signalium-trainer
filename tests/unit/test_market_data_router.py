"""Tests for trainer.market_data.market_data_router."""

from __future__ import annotations

from typing import Any

import pytest

from trainer.market_data.market_data_router import MarketDataRouter, SymbolNotCanonicalError
from trainer.market_data.provider_priority import ProviderName, ProviderPriority, TradingMode
from trainer.market_data.symbol_normalizer import SymbolNormalizer


class FakeProvider:
    def __init__(self, name: str, price: float = 1.2345, fail: bool = False) -> None:
        self._name = name
        self._price = price
        self._fail = fail

    def name(self) -> str:
        return self._name

    def get_price(self, symbol: str) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError(f"{self._name} unavailable")
        return {"symbol": symbol, "price": self._price, "provider": self._name}

    def health_check(self):
        from trainer.market_data.provider_interface import HealthStatus

        return HealthStatus(self._name, healthy=not self._fail)


@pytest.fixture
def providers():
    return {
        "oanda": FakeProvider("oanda", price=1.1),
        "tiingo": FakeProvider("tiingo", price=1.2),
        "ea_feed": FakeProvider("ea_feed", price=1.3),
    }


class TestPriorityResolution:
    def test_manual_priority(self, providers):
        router = MarketDataRouter(providers)
        assert router.priority_for(TradingMode.MANUAL) == ["OANDA", "TIINGO"]

    def test_autonomous_priority(self, providers):
        router = MarketDataRouter(providers)
        assert router.priority_for(TradingMode.AUTONOMOUS) == ["EA_FEED", "OANDA", "TIINGO"]

    def test_training_priority(self, providers):
        router = MarketDataRouter(providers)
        assert router.priority_for(context="training") == ["OANDA", "TIINGO"]

    def test_missing_mode_for_inference_raises(self, providers):
        router = MarketDataRouter(providers)
        with pytest.raises(ValueError):
            router.priority_for(context="inference")


class TestActiveProvider:
    def test_manual_uses_oanda_first(self, providers):
        router = MarketDataRouter(providers)
        assert router.active_provider_name(TradingMode.MANUAL) == "oanda"

    def test_autonomous_uses_ea_feed_first(self, providers):
        router = MarketDataRouter(providers)
        assert router.active_provider_name(TradingMode.AUTONOMOUS) == "ea_feed"

    def test_missing_ea_feed_degrades_gracefully(self):
        router = MarketDataRouter({"oanda": FakeProvider("oanda"), "tiingo": FakeProvider("tiingo")})
        assert router.active_provider_name(TradingMode.AUTONOMOUS) == "oanda"


class TestRouting:
    def test_inference_auto_normalizes_symbol(self, providers):
        router = MarketDataRouter(providers)
        result = router.route("EURUSDm", "get_price", mode=TradingMode.MANUAL)
        assert result["symbol"] == "EURUSD"
        assert result["provider"] == "oanda"

    def test_inference_fails_over_on_provider_error(self):
        router = MarketDataRouter({"oanda": FakeProvider("oanda", fail=True), "tiingo": FakeProvider("tiingo")})
        result = router.route("EURUSD", "get_price", mode=TradingMode.MANUAL)
        assert result["provider"] == "tiingo"

    def test_training_requires_canonical_symbol(self, providers):
        router = MarketDataRouter(providers)
        result = router.route("EURUSD", "get_price", context="training")
        assert result["symbol"] == "EURUSD"

    def test_training_rejects_non_canonical_symbol(self, providers):
        router = MarketDataRouter(providers)
        with pytest.raises(SymbolNotCanonicalError):
            router.route("EURUSDm", "get_price", context="training")

    def test_disabled_symbol_raises(self, providers):
        normalizer = SymbolNormalizer()
        normalizer.disable_symbol("EURUSD")
        router = MarketDataRouter(providers, normalizer=normalizer)
        with pytest.raises(ValueError):
            router.route("EURUSD", "get_price", mode=TradingMode.MANUAL)


class TestDiagnostics:
    def test_diagnostics_contains_priority_and_contexts(self, providers):
        router = MarketDataRouter(providers)
        diag = router.diagnostics()
        assert diag["priority"]["inference_priority_autonomous"] == ["EA_FEED", "OANDA", "TIINGO"]
        assert set(diag["contexts"].keys()) == {"training", "manual", "autonomous"}
        assert diag["contexts"]["manual"]["active_provider"] == "oanda"

    def test_diagnostics_lists_configured_providers(self, providers):
        router = MarketDataRouter(providers)
        assert router.diagnostics()["configured_providers"] == ["ea_feed", "oanda", "tiingo"]
