"""Tests for trainer.market_data.providers.ea_feed_provider."""

from __future__ import annotations

import pytest

from trainer.market_data.provider_interface import ProviderNotConfigured
from trainer.market_data.providers.ea_feed_provider import EAFeedProvider


class TestUnconfigured:
    def test_name_is_ea_feed(self):
        assert EAFeedProvider().name() == "ea_feed"

    def test_raises_not_configured_without_endpoint(self):
        provider = EAFeedProvider(api_key="secret")
        with pytest.raises(ProviderNotConfigured):
            provider.get_price("EURUSD")

    def test_raises_not_configured_without_secret(self):
        provider = EAFeedProvider(endpoint="http://localhost:9999")
        with pytest.raises(ProviderNotConfigured):
            provider.get_price("EURUSD")


class TestConfiguredButUnwired:
    def test_get_price_raises_not_implemented(self):
        provider = EAFeedProvider(api_key="secret", endpoint="http://localhost:9999")
        with pytest.raises(NotImplementedError):
            provider.get_price("EURUSD")

    def test_get_orderbook_structurally_unavailable(self):
        provider = EAFeedProvider(api_key="secret", endpoint="http://localhost:9999")
        with pytest.raises(NotImplementedError):
            provider.get_orderbook("EURUSD")
