"""Tests for trainer.market_data.symbol_normalizer."""

from __future__ import annotations

import pytest

from trainer.market_data.symbol_normalizer import SymbolNormalizer, UnknownSymbolError


@pytest.fixture
def norm() -> SymbolNormalizer:
    return SymbolNormalizer()


class TestExactAndBuiltin:
    def test_exact_canonical_match(self, norm):
        result = norm.normalize("EURUSD")
        assert result.canonical == "EURUSD"
        assert result.matched_via == "exact"
        assert result.enabled is True

    def test_builtin_alias_gold(self, norm):
        result = norm.normalize("GOLD")
        assert result.canonical == "XAUUSD"
        assert result.matched_via == "builtin_alias"

    def test_builtin_alias_case_insensitive(self, norm):
        result = norm.normalize("gold")
        assert result.canonical == "XAUUSD"

    def test_builtin_alias_slash_pair(self, norm):
        result = norm.normalize("EUR/USD")
        assert result.canonical == "EURUSD"


class TestStructuralStripping:
    def test_dot_suffix_stripped(self, norm):
        result = norm.normalize("EURUSD.r")
        assert result.canonical == "EURUSD"
        assert result.matched_via == "structural"

    def test_trailing_letter_stripped(self, norm):
        result = norm.normalize("EURUSDm")
        assert result.canonical == "EURUSD"

    def test_separator_stripped(self, norm):
        result = norm.normalize("EUR_USD")
        assert result.canonical == "EURUSD"

    def test_pro_suffix_stripped(self, norm):
        result = norm.normalize("EURUSD.pro")
        assert result.canonical == "EURUSD"


class TestAdminAliases:
    def test_add_alias_resolves(self, norm):
        norm.add_alias("MY-CUSTOM-EU", "EURUSD")
        result = norm.normalize("MY-CUSTOM-EU")
        assert result.canonical == "EURUSD"
        assert result.matched_via == "admin_alias"

    def test_add_alias_unknown_canonical_raises(self, norm):
        with pytest.raises(UnknownSymbolError):
            norm.add_alias("XYZ", "NOT_A_REAL_SYMBOL")

    def test_remove_alias(self, norm):
        norm.add_alias("TEMP", "EURUSD")
        norm.remove_alias("TEMP")
        result = norm.normalize("TEMP")
        assert result.canonical is None


class TestRuntimeLearning:
    def test_learn_from_observation(self, norm):
        norm.learn_from_observation("WEIRD-SPELLING", "GBPUSD")
        result = norm.normalize("WEIRD-SPELLING")
        assert result.canonical == "GBPUSD"
        assert result.matched_via == "learned"

    def test_learn_unknown_canonical_raises(self, norm):
        with pytest.raises(UnknownSymbolError):
            norm.learn_from_observation("X", "FAKE")


class TestSymbolManagement:
    def test_disable_symbol_marks_unresolved_enabled_false(self, norm):
        norm.disable_symbol("EURUSD")
        result = norm.normalize("EURUSD")
        assert result.canonical == "EURUSD"
        assert result.enabled is False

    def test_resolve_raises_on_disabled(self, norm):
        norm.disable_symbol("EURUSD")
        with pytest.raises(UnknownSymbolError):
            norm.resolve("EURUSD")

    def test_resolve_returns_canonical_when_enabled(self, norm):
        assert norm.resolve("gold") == "XAUUSD"

    def test_add_new_symbol(self, norm):
        norm.add_symbol("BTCUSD", asset_class="crypto", enabled=False)
        symbols = {s["symbol"]: s for s in norm.list_symbols()}
        assert "BTCUSD" in symbols
        assert symbols["BTCUSD"]["enabled"] is False

    def test_unresolved_symbol(self, norm):
        result = norm.normalize("TOTALLY_UNKNOWN_THING_9999")
        assert result.canonical is None
        assert result.matched_via == "unresolved"

    def test_empty_string(self, norm):
        result = norm.normalize("")
        assert result.canonical is None
