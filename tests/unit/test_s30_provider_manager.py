"""Tests for trainer.market_data: provider_interface stubs + provider_manager failover."""

from __future__ import annotations

import time

import pytest

from trainer.market_data.provider_interface import ProviderNotConfigured
from trainer.market_data.provider_manager import (
    AllProvidersUnavailable,
    DataContext,
    ProviderManager,
)
from trainer.market_data.providers.oanda_provider import OandaProvider
from trainer.market_data.providers.tiingo_provider import TiingoProvider


class TestStubProvidersNotConfigured:
    def test_oanda_without_key_raises_on_data_call(self):
        provider = OandaProvider(api_key=None)
        with pytest.raises(ProviderNotConfigured):
            provider.get_price("EURUSD")

    def test_tiingo_without_key_raises_on_data_call(self):
        provider = TiingoProvider(api_key=None)
        with pytest.raises(ProviderNotConfigured):
            provider.get_candles("EURUSD", "1day")

    def test_oanda_health_check_reports_unconfigured(self):
        provider = OandaProvider(api_key=None)
        health = provider.health_check()
        assert health.healthy is False

    def test_oanda_with_key_connect_succeeds(self):
        provider = OandaProvider(api_key="fake-key")
        provider.connect()
        health = provider.health_check()
        assert health.healthy is True

    def test_tiingo_ticks_not_supported_even_with_key(self):
        provider = TiingoProvider(api_key="fake-key")
        with pytest.raises(NotImplementedError):
            provider.get_ticks("EURUSD")

    def test_oanda_data_call_with_key_hits_unimplemented_request_seam(self):
        provider = OandaProvider(api_key="fake-key")
        with pytest.raises(NotImplementedError):
            provider.get_price("EURUSD")


class _FakeProvider:
    """Minimal MarketDataProvider stand-in for failover tests."""

    def __init__(self, provider_name: str, behavior: str = "ok", latency_s: float = 0.0):
        self._name = provider_name
        self._behavior = behavior  # "ok" | "error" | "slow"
        self._latency_s = latency_s
        self.calls = 0

    def name(self) -> str:
        return self._name

    def get_price(self, symbol: str):
        self.calls += 1
        if self._latency_s:
            time.sleep(self._latency_s)
        if self._behavior == "error":
            raise RuntimeError(f"{self._name} boom")
        return {"symbol": symbol, "provider": self._name, "ts": time.time()}


class TestProviderManagerFailover:
    def test_uses_first_healthy_provider(self):
        p1 = _FakeProvider("p1")
        p2 = _FakeProvider("p2")
        manager = ProviderManager([p1, p2], [p2, p1])
        result = manager.call(DataContext.LIVE, "get_price", "EURUSD")
        assert result["provider"] == "p1"
        assert p2.calls == 0

    def test_fails_over_to_second_provider_on_error(self):
        p1 = _FakeProvider("p1", behavior="error")
        p2 = _FakeProvider("p2", behavior="ok")
        manager = ProviderManager([p1, p2], [p2, p1], max_consecutive_failures=1)
        result = manager.call(DataContext.LIVE, "get_price", "EURUSD")
        assert result["provider"] == "p2"

    def test_all_providers_unavailable_raises(self):
        p1 = _FakeProvider("p1", behavior="error")
        p2 = _FakeProvider("p2", behavior="error")
        manager = ProviderManager([p1, p2], [p2, p1], max_consecutive_failures=1)
        with pytest.raises(AllProvidersUnavailable):
            manager.call(DataContext.LIVE, "get_price", "EURUSD")

    def test_excessive_latency_triggers_failover(self):
        p1 = _FakeProvider("p1", latency_s=0.05)
        p2 = _FakeProvider("p2")
        manager = ProviderManager(
            [p1, p2], [p2, p1], latency_threshold_ms=10.0, max_consecutive_failures=1
        )
        result = manager.call(DataContext.LIVE, "get_price", "EURUSD")
        assert result["provider"] == "p2"

    def test_degraded_provider_marked_after_max_failures(self):
        p1 = _FakeProvider("p1", behavior="error")
        p2 = _FakeProvider("p2")
        manager = ProviderManager(
            [p1, p2], [p2, p1], max_consecutive_failures=1, degrade_cooldown_s=60.0
        )
        manager.call(DataContext.LIVE, "get_price", "EURUSD")
        active = manager.active_provider(DataContext.LIVE)
        assert active.name() == "p2"  # p1 now degraded, p2 preferred

    def test_health_snapshot_reports_all_providers(self):
        p1 = _FakeProvider("p1")
        p2 = _FakeProvider("p2")
        manager = ProviderManager([p1, p2], [p2, p1])
        snapshot = manager.health_snapshot(DataContext.LIVE)
        assert len(snapshot) == 2

    def test_live_and_training_priority_are_independent(self):
        oanda = _FakeProvider("oanda")
        tiingo = _FakeProvider("tiingo")
        manager = ProviderManager(live_priority=[oanda, tiingo], training_priority=[tiingo, oanda])
        assert manager.active_provider(DataContext.LIVE).name() == "oanda"
        assert manager.active_provider(DataContext.TRAINING).name() == "tiingo"

    def test_stale_data_triggers_failover(self):
        p1 = _FakeProvider("p1")
        p2 = _FakeProvider("p2")
        manager = ProviderManager([p1, p2], [p2, p1], max_consecutive_failures=1)
        result = manager.call(
            DataContext.LIVE,
            "get_price",
            "EURUSD",
            max_staleness_s=10.0,
            staleness_extractor=lambda r: (
                time.time() - 100 if r["provider"] == "p1" else time.time()
            ),
        )
        assert result["provider"] == "p2"

    def test_no_providers_configured_raises(self):
        manager = ProviderManager([], [])
        with pytest.raises(AllProvidersUnavailable):
            manager.active_provider(DataContext.LIVE)

    def test_recent_failovers_logged(self):
        p1 = _FakeProvider("p1", behavior="error")
        p2 = _FakeProvider("p2")
        manager = ProviderManager([p1, p2], [p2, p1], max_consecutive_failures=1)
        manager.call(DataContext.LIVE, "get_price", "EURUSD")
        events = manager.recent_failovers()
        assert len(events) >= 1
        assert events[0]["from_provider"] == "p1"
