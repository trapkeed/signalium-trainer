"""Tests for trainer.market_data.symbol_store."""

from __future__ import annotations

import pytest

from trainer.market_data.symbol_store import SymbolStore


@pytest.fixture
def store() -> SymbolStore:
    return SymbolStore(":memory:")


class TestAliasPersistence:
    def test_save_and_list_alias(self, store):
        store.save_alias("eurusd_i", "EURUSD")
        aliases = store.list_aliases()
        assert len(aliases) == 1
        assert aliases[0].raw == "EURUSD_I"
        assert aliases[0].canonical == "EURUSD"

    def test_save_alias_upsert(self, store):
        store.save_alias("gold_x", "XAUUSD")
        store.save_alias("gold_x", "GOLD2")
        aliases = store.list_aliases()
        assert len(aliases) == 1
        assert aliases[0].canonical == "GOLD2"

    def test_remove_alias(self, store):
        store.save_alias("eurusd_i", "EURUSD")
        store.remove_alias("eurusd_i")
        assert store.list_aliases() == []

    def test_list_aliases_filtered_by_canonical(self, store):
        store.save_alias("a1", "EURUSD")
        store.save_alias("a2", "GBPUSD")
        assert [a.raw for a in store.list_aliases(canonical="eurusd")] == ["A1"]


class TestSymbolState:
    def test_set_enabled_persists(self, store):
        store.set_enabled("EURUSD", False)
        assert store.get_state() == {"EURUSD": False}
        store.set_enabled("EURUSD", True)
        assert store.get_state() == {"EURUSD": True}


class TestNormalizationEvents:
    def test_log_and_recent_events(self, store):
        store.log_event("EURUSDm", "EURUSD", "structural", True, broker=None)
        store.log_event("UNKNOWNSYM", None, "unresolved", False, broker="oanda")
        events = store.recent_events()
        assert len(events) == 2

    def test_unresolved_only_filter(self, store):
        store.log_event("EURUSDm", "EURUSD", "structural", True)
        store.log_event("UNKNOWNSYM", None, "unresolved", False)
        events = store.recent_events(unresolved_only=True)
        assert len(events) == 1
        assert events[0]["raw"] == "UNKNOWNSYM"

    def test_event_stats(self, store):
        store.log_event("EURUSDm", "EURUSD", "structural", True)
        store.log_event("UNKNOWNSYM", None, "unresolved", False)
        stats = store.event_stats()
        assert stats["total_events"] == 2
        assert stats["unresolved_events"] == 1
        assert stats["by_matched_via"] == {"structural": 1, "unresolved": 1}
