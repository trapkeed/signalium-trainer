"""Unit tests for trainer.symbols — the canonical symbol translation layer.

Covers:
  - SymbolTranslationLayer core operations (translate, is_canonical,
    assert_canonical, batch_translate, batch_assert_canonical)
  - Process-level singleton (get_translator, configure_translator,
    reset_translator)
  - Module-level convenience wrappers
  - SymbolNormalizer additions (is_canonical_symbol, canonical_symbols)
  - Boundary guards:
      - TradeOutcomePath.__post_init__ (labels)
      - DecisionRecord.create         (memory / decisions)
      - validate_command              (protocol)
      - register_backtest             (research)
  - No import of qde_inference or execution from trainer.symbols
"""

from __future__ import annotations

import time

import pytest

from trainer.market_data.symbol_normalizer import SymbolNormalizer
from trainer.symbols import (
    SymbolTranslationError,
    SymbolTranslationLayer,
    assert_canonical,
    batch_assert_canonical,
    batch_translate,
    configure_translator,
    get_translator,
    is_canonical,
    reset_translator,
    translate,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the process-level singleton before and after every test."""
    reset_translator()
    yield
    reset_translator()


def _fresh_layer() -> SymbolTranslationLayer:
    """Return a SymbolTranslationLayer backed by the default SymbolNormalizer."""
    return SymbolTranslationLayer()


# ─────────────────────────────────────────────────────────────────────────────
# 01–05  SymbolNormalizer new methods
# ─────────────────────────────────────────────────────────────────────────────


def test_01_is_canonical_symbol_true_for_known():
    n = SymbolNormalizer()
    assert n.is_canonical_symbol("EURUSD") is True
    assert n.is_canonical_symbol("XAUUSD") is True
    assert n.is_canonical_symbol("US30") is True


def test_02_is_canonical_symbol_case_insensitive():
    n = SymbolNormalizer()
    assert n.is_canonical_symbol("eurusd") is True
    assert n.is_canonical_symbol("Eurusd") is True


def test_03_is_canonical_symbol_false_for_broker_variant():
    n = SymbolNormalizer()
    assert n.is_canonical_symbol("EURUSD.r") is False
    assert n.is_canonical_symbol("EURUSDm") is False
    assert n.is_canonical_symbol("GOLD") is False  # builtin alias, not canonical
    assert n.is_canonical_symbol("EUR/USD") is False  # builtin alias, not canonical
    assert n.is_canonical_symbol("UNKNOWN_XYZ") is False


def test_04_canonical_symbols_returns_sorted_list():
    n = SymbolNormalizer()
    symbols = n.canonical_symbols()
    assert isinstance(symbols, list)
    assert symbols == sorted(symbols)
    assert "EURUSD" in symbols
    assert "XAUUSD" in symbols


def test_05_canonical_symbols_includes_dynamically_added():
    n = SymbolNormalizer()
    n.add_symbol("BTCUSD", "crypto")
    assert "BTCUSD" in n.canonical_symbols()


# ─────────────────────────────────────────────────────────────────────────────
# 06–15  SymbolTranslationLayer.translate
# ─────────────────────────────────────────────────────────────────────────────


def test_06_translate_canonical_passthrough():
    layer = _fresh_layer()
    assert layer.translate("EURUSD") == "EURUSD"
    assert layer.translate("XAUUSD") == "XAUUSD"


def test_07_translate_broker_suffix():
    layer = _fresh_layer()
    assert layer.translate("EURUSD.r") == "EURUSD"
    assert layer.translate("EURUSDm") == "EURUSD"
    assert layer.translate("EURUSD.pro") == "EURUSD"


def test_08_translate_builtin_alias():
    layer = _fresh_layer()
    assert layer.translate("GOLD") == "XAUUSD"
    assert layer.translate("EUR/USD") == "EURUSD"


def test_09_translate_case_insensitive():
    layer = _fresh_layer()
    assert layer.translate("eurusd") == "EURUSD"
    assert layer.translate("gold") == "XAUUSD"


def test_10_translate_with_broker():
    layer = _fresh_layer()
    # oanda profile: US30_USD -> US30
    assert layer.translate("US30_USD", broker="oanda") == "US30"


def test_11_translate_unknown_raises_translation_error():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError):
        layer.translate("XXXUNKNOWN999")


def test_12_translate_unknown_error_message_contains_raw():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError, match="XXXUNKNOWN999"):
        layer.translate("XXXUNKNOWN999")


def test_13_translate_unknown_with_broker_in_error():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError, match="oanda"):
        layer.translate("XXXUNKNOWN999", broker="oanda")


def test_14_translate_empty_string_raises():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError):
        layer.translate("")


def test_15_translate_returns_str():
    layer = _fresh_layer()
    result = layer.translate("EURUSD.r")
    assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# 16–20  SymbolTranslationLayer.is_canonical
# ─────────────────────────────────────────────────────────────────────────────


def test_16_is_canonical_true():
    layer = _fresh_layer()
    assert layer.is_canonical("EURUSD") is True
    assert layer.is_canonical("XAUUSD") is True


def test_17_is_canonical_false_for_broker_variant():
    layer = _fresh_layer()
    assert layer.is_canonical("EURUSD.r") is False
    assert layer.is_canonical("GOLD") is False
    assert layer.is_canonical("EUR/USD") is False


def test_18_is_canonical_case_insensitive():
    layer = _fresh_layer()
    assert layer.is_canonical("eurusd") is True


def test_19_is_canonical_false_for_unknown():
    layer = _fresh_layer()
    assert layer.is_canonical("UNKNOWNXYZ") is False


def test_20_is_canonical_false_for_empty():
    layer = _fresh_layer()
    assert layer.is_canonical("") is False


# ─────────────────────────────────────────────────────────────────────────────
# 21–28  SymbolTranslationLayer.assert_canonical
# ─────────────────────────────────────────────────────────────────────────────


def test_21_assert_canonical_passes_for_canonical():
    layer = _fresh_layer()
    layer.assert_canonical("EURUSD")  # must not raise
    layer.assert_canonical("XAUUSD")
    layer.assert_canonical("US30")


def test_22_assert_canonical_raises_for_broker_variant():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError):
        layer.assert_canonical("EURUSD.r")


def test_23_assert_canonical_raises_for_alias():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError):
        layer.assert_canonical("GOLD")


def test_24_assert_canonical_raises_for_unknown():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError):
        layer.assert_canonical("UNKNOWNXYZ")


def test_25_assert_canonical_error_contains_symbol():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError, match="EURUSD.r"):
        layer.assert_canonical("EURUSD.r")


def test_26_assert_canonical_error_is_value_error_subclass():
    layer = _fresh_layer()
    with pytest.raises(ValueError):  # SymbolTranslationError subclasses ValueError
        layer.assert_canonical("GOLD")


def test_27_assert_canonical_case_insensitive_passes():
    layer = _fresh_layer()
    layer.assert_canonical("eurusd")  # must not raise — "EURUSD" is canonical


def test_28_known_canonicals_non_empty():
    layer = _fresh_layer()
    result = layer.known_canonicals()
    assert len(result) >= 10
    assert "EURUSD" in result


# ─────────────────────────────────────────────────────────────────────────────
# 29–34  batch_translate / batch_assert_canonical
# ─────────────────────────────────────────────────────────────────────────────


def test_29_batch_translate_all_canonical():
    layer = _fresh_layer()
    result = layer.batch_translate(["EURUSD", "GBPUSD", "XAUUSD"])
    assert result == {"EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "XAUUSD": "XAUUSD"}


def test_30_batch_translate_broker_variants():
    layer = _fresh_layer()
    result = layer.batch_translate(["EURUSD.r", "GOLD"])
    assert result["EURUSD.r"] == "EURUSD"
    assert result["GOLD"] == "XAUUSD"


def test_31_batch_translate_raises_on_unknown():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError):
        layer.batch_translate(["EURUSD", "XXXUNKNOWN"])


def test_32_batch_assert_canonical_passes():
    layer = _fresh_layer()
    layer.batch_assert_canonical(["EURUSD", "GBPUSD", "XAUUSD"])  # must not raise


def test_33_batch_assert_canonical_raises_for_any_broker():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError):
        layer.batch_assert_canonical(["EURUSD", "EURUSD.r"])


def test_34_batch_assert_canonical_error_contains_bad_symbols():
    layer = _fresh_layer()
    with pytest.raises(SymbolTranslationError, match="EURUSD.r"):
        layer.batch_assert_canonical(["EURUSD", "EURUSD.r"])


# ─────────────────────────────────────────────────────────────────────────────
# 35–40  register_broker_alias
# ─────────────────────────────────────────────────────────────────────────────


def test_35_register_broker_alias_makes_translatable():
    layer = _fresh_layer()
    layer.register_broker_alias("MY_EUR_USD", "EURUSD")
    assert layer.translate("MY_EUR_USD") == "EURUSD"


def test_36_register_broker_alias_does_not_make_canonical():
    layer = _fresh_layer()
    layer.register_broker_alias("MY_EUR_USD", "EURUSD")
    assert layer.is_canonical("MY_EUR_USD") is False


def test_37_register_broker_alias_invalid_canonical_raises():
    layer = _fresh_layer()
    with pytest.raises(Exception):
        layer.register_broker_alias("FOO", "NOTACANONICAL")


# ─────────────────────────────────────────────────────────────────────────────
# 38–45  Singleton management
# ─────────────────────────────────────────────────────────────────────────────


def test_38_get_translator_returns_layer():
    t = get_translator()
    assert isinstance(t, SymbolTranslationLayer)


def test_39_get_translator_returns_same_object():
    a = get_translator()
    b = get_translator()
    assert a is b


def test_40_configure_translator_with_normalizer():
    n = SymbolNormalizer({"BTCUSD": "crypto"})
    configure_translator(normalizer=n)
    t = get_translator()
    assert t.is_canonical("BTCUSD") is True
    assert t.is_canonical("EURUSD") is False  # not in the custom normalizer


def test_41_configure_translator_with_layer():
    custom = SymbolTranslationLayer(SymbolNormalizer({"ETHUSD": "crypto"}))
    configure_translator(layer=custom)
    assert get_translator() is custom


def test_42_configure_translator_both_raises():
    n = SymbolNormalizer()
    layer = SymbolTranslationLayer(n)
    with pytest.raises(ValueError):
        configure_translator(normalizer=n, layer=layer)


def test_43_reset_translator_clears_singleton():
    _ = get_translator()  # build it
    reset_translator()
    assert get_translator() is not _  # rebuilt fresh after reset


def test_44_module_level_translate():
    assert translate("EURUSD.r") == "EURUSD"


def test_45_module_level_is_canonical():
    assert is_canonical("EURUSD") is True
    assert is_canonical("GOLD") is False


def test_46_module_level_assert_canonical_passes():
    assert_canonical("EURUSD")  # must not raise


def test_47_module_level_assert_canonical_raises():
    with pytest.raises(SymbolTranslationError):
        assert_canonical("EURUSD.r")


def test_48_module_level_batch_translate():
    result = batch_translate(["EURUSD.r", "GOLD"])
    assert result["EURUSD.r"] == "EURUSD"
    assert result["GOLD"] == "XAUUSD"


def test_49_module_level_batch_assert_canonical():
    batch_assert_canonical(["EURUSD", "GBPUSD"])  # must not raise
    with pytest.raises(SymbolTranslationError):
        batch_assert_canonical(["EURUSD", "GOLD"])


# ─────────────────────────────────────────────────────────────────────────────
# 50–58  Boundary: TradeOutcomePath (labels / training_signals)
# ─────────────────────────────────────────────────────────────────────────────


def test_50_outcome_path_accepts_canonical():
    from trainer.training_signals.outcome_path import TradeOutcomePath

    path = TradeOutcomePath(instrument="EURUSD", direction=1, entry_price=1.10)
    assert path.instrument == "EURUSD"


def test_51_outcome_path_rejects_broker_variant():
    from trainer.training_signals.outcome_path import TradeOutcomePath

    with pytest.raises(SymbolTranslationError):
        TradeOutcomePath(instrument="EURUSD.r", direction=1, entry_price=1.10)


def test_52_outcome_path_rejects_builtin_alias():
    from trainer.training_signals.outcome_path import TradeOutcomePath

    with pytest.raises(SymbolTranslationError):
        TradeOutcomePath(instrument="GOLD", direction=1, entry_price=1900.0)


def test_53_outcome_path_rejects_unknown():
    from trainer.training_signals.outcome_path import TradeOutcomePath

    with pytest.raises(SymbolTranslationError):
        TradeOutcomePath(instrument="UNKNOWNXYZ", direction=1, entry_price=1.0)


def test_54_outcome_path_accepts_all_defaults():
    from trainer.training_signals.outcome_path import TradeOutcomePath

    # All symbols in DEFAULT_SYMBOLS must pass
    for sym in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NAS100"]:
        path = TradeOutcomePath(instrument=sym, direction=1, entry_price=1.0)
        assert path.instrument == sym


# ─────────────────────────────────────────────────────────────────────────────
# 55–62  Boundary: DecisionRecord.create (memory / decisions)
# ─────────────────────────────────────────────────────────────────────────────


def _make_snapshot_and_reasoning():
    from trainer.memory.decision_models import DecisionReasoning, DecisionSnapshot

    snap = DecisionSnapshot()
    reason = DecisionReasoning()
    return snap, reason


def test_55_decision_record_accepts_canonical():
    from trainer.memory.decision_models import DecisionRecord, DecisionType

    snap, reason = _make_snapshot_and_reasoning()
    rec = DecisionRecord.create(DecisionType.BUY, "EURUSD", snap, reason)
    assert rec.instrument == "EURUSD"


def test_56_decision_record_rejects_broker_variant():
    from trainer.memory.decision_models import DecisionRecord, DecisionType

    snap, reason = _make_snapshot_and_reasoning()
    with pytest.raises(SymbolTranslationError):
        DecisionRecord.create(DecisionType.BUY, "EURUSD.r", snap, reason)


def test_57_decision_record_rejects_builtin_alias():
    from trainer.memory.decision_models import DecisionRecord, DecisionType

    snap, reason = _make_snapshot_and_reasoning()
    with pytest.raises(SymbolTranslationError):
        DecisionRecord.create(DecisionType.BUY, "GOLD", snap, reason)


def test_58_decision_record_rejects_unknown():
    from trainer.memory.decision_models import DecisionRecord, DecisionType

    snap, reason = _make_snapshot_and_reasoning()
    with pytest.raises(SymbolTranslationError):
        DecisionRecord.create(DecisionType.NO_TRADE, "UNKNOWNXYZ", snap, reason)


def test_59_decision_record_accepts_multiple_canonicals():
    from trainer.memory.decision_models import DecisionRecord, DecisionType

    snap, reason = _make_snapshot_and_reasoning()
    for sym in ["GBPUSD", "USDJPY", "XAUUSD"]:
        rec = DecisionRecord.create(DecisionType.NO_TRADE, sym, snap, reason)
        assert rec.instrument == sym


# ─────────────────────────────────────────────────────────────────────────────
# 63–68  Boundary: protocol validate_command (canonical symbol check)
# ─────────────────────────────────────────────────────────────────────────────


def _make_open_cmd(symbol: str) -> "object":
    from trainer.protocol import open_position_command

    return open_position_command(
        position_id="",
        symbol=symbol,
        direction="BUY",
        size_lots=0.1,
        stop_loss=1.08,
        take_profit=1.14,
        entry_price=1.10,
        qde_version="v1.0",
        ttl_seconds=60,
    )


def test_63_validate_command_passes_for_canonical():
    from trainer.protocol import validate_command

    cmd = _make_open_cmd("EURUSD")
    result = validate_command(cmd)
    # should be valid — no canonical-symbol errors
    symbol_errors = [
        e for e in result.errors if "canonical" in e.lower() or "not a canonical" in e.lower()
    ]
    assert symbol_errors == [], symbol_errors


def test_64_validate_command_fails_for_broker_variant():
    from trainer.protocol import validate_command

    cmd = _make_open_cmd("EURUSD.r")
    result = validate_command(cmd)
    assert not result.valid
    assert any("canonical" in e.lower() or "symbol" in e.lower() for e in result.errors)


def test_65_validate_command_fails_for_builtin_alias():
    from trainer.protocol import validate_command

    cmd = _make_open_cmd("GOLD")
    result = validate_command(cmd)
    assert not result.valid
    assert any("canonical" in e.lower() or "symbol" in e.lower() for e in result.errors)


def test_66_validate_command_fails_for_unknown_symbol():
    from trainer.protocol import validate_command

    cmd = _make_open_cmd("UNKNOWNXYZ")
    result = validate_command(cmd)
    assert not result.valid


# ─────────────────────────────────────────────────────────────────────────────
# 69–75  Boundary: research lab register_backtest
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def _rl_api():
    from trainer.research_lab.research_lab_api import ResearchLabAPI
    from trainer.research_lab.rl_database import ResearchLabDatabase

    db = ResearchLabDatabase(":memory:")
    return ResearchLabAPI(db=db)


def _create_completed_experiment(api):
    resp = api.create_experiment(
        name="test-exp",
        qde_version="v1",
        training_dataset="ds-001",
        researcher="test",
    )
    exp_id = resp.data["experiment_id"]
    # advance to RUNNING

    db_rec = api._db.get_experiment(exp_id)
    db_rec["status"] = "RUNNING"
    db_rec["training_started_at"] = time.time()
    api._db.upsert_experiment(db_rec)
    return exp_id


def test_69_register_backtest_accepts_canonical(_rl_api):
    exp_id = _create_completed_experiment(_rl_api)
    db_rec = _rl_api._db.get_experiment(exp_id)
    db_rec["status"] = "COMPLETED"
    db_rec["training_completed_at"] = time.time()
    _rl_api._db.upsert_experiment(db_rec)
    resp = _rl_api.register_backtest(
        experiment_id=exp_id,
        start_date="2024-01-01",
        end_date="2024-06-01",
        instruments=["EURUSD", "GBPUSD"],
        metrics={"sharpe": 1.2},
    )
    # backtest should be registered successfully (instruments are canonical)
    assert resp.success or resp.data.get("backtest_id")


def test_70_register_backtest_rejects_broker_variant(_rl_api):
    exp_id = _create_completed_experiment(_rl_api)
    with pytest.raises(SymbolTranslationError):
        _rl_api.register_backtest(
            experiment_id=exp_id,
            start_date="2024-01-01",
            end_date="2024-06-01",
            instruments=["EURUSD.r", "GBPUSD"],
            metrics={"sharpe": 1.2},
        )


def test_71_register_backtest_rejects_builtin_alias(_rl_api):
    exp_id = _create_completed_experiment(_rl_api)
    with pytest.raises(SymbolTranslationError):
        _rl_api.register_backtest(
            experiment_id=exp_id,
            start_date="2024-01-01",
            end_date="2024-06-01",
            instruments=["GOLD"],
            metrics={"sharpe": 0.9},
        )


def test_72_register_backtest_rejects_mixed_good_bad(_rl_api):
    """Even one non-canonical symbol in a mixed list must be rejected."""
    exp_id = _create_completed_experiment(_rl_api)
    with pytest.raises(SymbolTranslationError):
        _rl_api.register_backtest(
            experiment_id=exp_id,
            start_date="2024-01-01",
            end_date="2024-06-01",
            instruments=["EURUSD", "GBPUSD", "EURUSD.r"],
            metrics={"sharpe": 1.0},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 73–75  No forbidden imports in trainer.symbols
# ─────────────────────────────────────────────────────────────────────────────


def test_73_symbols_package_does_not_import_qde_inference():
    """trainer.symbols must never import qde_inference."""
    import importlib
    import sys

    mod = sys.modules.get("trainer.symbols.translation")
    if mod is None:
        mod = importlib.import_module("trainer.symbols.translation")
    import inspect

    source = inspect.getsource(mod)
    assert "qde_inference" not in source


def test_74_symbols_package_does_not_import_execution():
    """trainer.symbols must never import trainer.execution."""
    import importlib
    import sys

    mod = sys.modules.get("trainer.symbols.translation")
    if mod is None:
        mod = importlib.import_module("trainer.symbols.translation")
    import inspect

    source = inspect.getsource(mod)
    assert "trainer.execution" not in source


def test_75_symbol_translation_error_is_value_error():
    assert issubclass(SymbolTranslationError, ValueError)
