"""Tests for SymbolNormalizer's optional SymbolStore persistence + logging."""

from __future__ import annotations

from trainer.market_data.symbol_normalizer import SymbolNormalizer
from trainer.market_data.symbol_store import SymbolStore


class TestPersistedAliases:
    def test_alias_persists_across_normalizer_instances(self):
        store = SymbolStore(":memory:")
        norm1 = SymbolNormalizer(store=store)
        norm1.add_symbol_alias("eurusd_i", "EURUSD")

        norm2 = SymbolNormalizer(store=store)
        result = norm2.normalize("EURUSD_I")
        assert result.canonical == "EURUSD"
        assert result.matched_via == "admin_alias"

    def test_remove_alias_persists(self):
        store = SymbolStore(":memory:")
        norm1 = SymbolNormalizer(store=store)
        norm1.add_symbol_alias("eurusd_i", "EURUSD")
        norm1.remove_alias("eurusd_i")

        norm2 = SymbolNormalizer(store=store)
        assert norm2.list_symbol_aliases() == []

    def test_disable_enable_persists(self):
        store = SymbolStore(":memory:")
        norm1 = SymbolNormalizer(store=store)
        norm1.disable_symbol("EURUSD")

        norm2 = SymbolNormalizer(store=store)
        assert norm2.is_enabled("EURUSD") is False

        norm2.enable_symbol("EURUSD")
        norm3 = SymbolNormalizer(store=store)
        assert norm3.is_enabled("EURUSD") is True

    def test_add_symbol_alias_is_add_alias(self):
        norm = SymbolNormalizer()
        assert norm.add_symbol_alias.__func__ is norm.add_alias.__func__


class TestNormalizationEventLogging:
    def test_normalize_logs_event_to_store(self):
        store = SymbolStore(":memory:")
        norm = SymbolNormalizer(store=store)
        norm.normalize("EURUSD")
        events = store.recent_events()
        assert len(events) == 1
        assert events[0]["raw"] == "EURUSD"
        assert events[0]["canonical"] == "EURUSD"

    def test_unresolved_normalize_logs_null_canonical(self):
        store = SymbolStore(":memory:")
        norm = SymbolNormalizer(store=store)
        norm.normalize("TOTALLY_UNKNOWN_SYMBOL_XYZ")
        events = store.recent_events()
        assert events[0]["canonical"] is None

    def test_normalize_without_store_does_not_error(self):
        norm = SymbolNormalizer()
        result = norm.normalize("EURUSD")
        assert result.canonical == "EURUSD"
