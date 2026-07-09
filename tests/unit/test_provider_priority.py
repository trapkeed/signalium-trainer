"""Tests for trainer.market_data.provider_priority."""

from __future__ import annotations

from trainer.market_data.provider_priority import ProviderName, ProviderPriority, TradingMode


class TestDefaults:
    def test_manual_priority_is_oanda_then_tiingo(self):
        p = ProviderPriority()
        assert p.for_mode(TradingMode.MANUAL) == [ProviderName.OANDA, ProviderName.TIINGO]

    def test_autonomous_priority_is_ea_feed_then_oanda_then_tiingo(self):
        p = ProviderPriority()
        assert p.for_mode(TradingMode.AUTONOMOUS) == [
            ProviderName.EA_FEED,
            ProviderName.OANDA,
            ProviderName.TIINGO,
        ]

    def test_training_providers_default(self):
        p = ProviderPriority()
        assert p.training_providers == [ProviderName.OANDA, ProviderName.TIINGO]

    def test_to_dict(self):
        p = ProviderPriority()
        d = p.to_dict()
        assert d["inference_priority_autonomous"] == ["EA_FEED", "OANDA", "TIINGO"]
        assert d["inference_priority_manual"] == ["OANDA", "TIINGO"]
        assert d["training_providers"] == ["OANDA", "TIINGO"]

    def test_for_mode_returns_copy(self):
        p = ProviderPriority()
        lst = p.for_mode(TradingMode.MANUAL)
        lst.append(ProviderName.EA_FEED)
        assert p.inference_priority_manual == [ProviderName.OANDA, ProviderName.TIINGO]


class TestCustomPriority:
    def test_custom_priority_overrides_defaults(self):
        p = ProviderPriority(inference_priority_manual=[ProviderName.TIINGO])
        assert p.for_mode(TradingMode.MANUAL) == [ProviderName.TIINGO]
