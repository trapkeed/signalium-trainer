"""S32 — Setup Memory Engine tests."""

from __future__ import annotations

import tempfile

import pytest

from trainer.setup_memory.engine import (
    InvalidTransition,
    SetupMemoryConfig,
    SetupMemoryEngine,
)
from trainer.setup_memory.models import Direction, SetupState
from trainer.setup_memory.store import SetupMemoryStore


@pytest.fixture()
def engine() -> SetupMemoryEngine:
    db_path = tempfile.mktemp(suffix=".db")
    return SetupMemoryEngine(
        store=SetupMemoryStore(db_path),
        config=SetupMemoryConfig(cooldown_s=0.2, entry_window_timeout_s=0.2, setup_max_age_s=0.5),
    )


def test_new_setup_starts_idle(engine: SetupMemoryEngine) -> None:
    record = engine.get_or_create("EURUSD", "1h", "EUR_USD")
    assert record.state == SetupState.IDLE
    assert record.broker_symbol == "EUR_USD"


def test_full_forward_lifecycle(engine: SetupMemoryEngine) -> None:
    seq = [
        SetupState.WATCHING,
        SetupState.BUILDING_SETUP,
        SetupState.ARMED,
        SetupState.ENTRY_WINDOW,
        SetupState.ACTIVE_POSITION,
        SetupState.COOLDOWN,
        SetupState.IDLE,
    ]
    for expected in seq:
        record = engine.advance("EURUSD", "1h", broker_symbol="EUR_USD")
        assert record.state == expected


def test_illegal_jump_rejected(engine: SetupMemoryEngine) -> None:
    with pytest.raises(InvalidTransition):
        engine.advance_to("EURUSD", "1h", SetupState.ARMED, "illegal jump")


def test_strengthen_increases_confidence_without_changing_state(
    engine: SetupMemoryEngine,
) -> None:
    engine.advance("EURUSD", "1h")
    record = engine.strengthen("EURUSD", "1h", 0.4)
    assert record.state == SetupState.WATCHING
    assert record.state_confidence == pytest.approx(0.4)


def test_weaken_demotes_when_confidence_collapses(engine: SetupMemoryEngine) -> None:
    engine.advance("EURUSD", "1h")  # WATCHING
    engine.advance("EURUSD", "1h")  # BUILDING_SETUP
    engine.strengthen("EURUSD", "1h", 0.5)
    record = engine.weaken("EURUSD", "1h", 0.6)
    assert record.state == SetupState.WATCHING


def test_invalidate_moves_to_cooldown(engine: SetupMemoryEngine) -> None:
    engine.advance("EURUSD", "1h")
    record = engine.invalidate("EURUSD", "1h", "structure broke down")
    assert record.state == SetupState.COOLDOWN
    assert record.invalidation_reason == "structure broke down"


def test_invalidate_active_position_raises(engine: SetupMemoryEngine) -> None:
    for _ in range(5):
        engine.advance("EURUSD", "1h")
    record = engine.get("EURUSD", "1h")
    assert record is not None and record.state == SetupState.ACTIVE_POSITION
    with pytest.raises(InvalidTransition):
        engine.invalidate("EURUSD", "1h", "should not work")


def test_entry_window_expiry_forces_cooldown(engine: SetupMemoryEngine) -> None:
    import time

    for _ in range(4):
        engine.advance("EURUSD", "1h")
    assert engine.get("EURUSD", "1h").state == SetupState.ENTRY_WINDOW  # type: ignore[union-attr]
    time.sleep(0.3)
    record = engine.check_expiry("EURUSD", "1h")
    assert record.state == SetupState.COOLDOWN


def test_cooldown_elapses_to_idle(engine: SetupMemoryEngine) -> None:
    import time

    engine.advance("EURUSD", "1h")
    engine.invalidate("EURUSD", "1h", "x")
    time.sleep(0.3)
    record = engine.check_expiry("EURUSD", "1h")
    assert record.state == SetupState.IDLE


def test_stale_setup_expires(engine: SetupMemoryEngine) -> None:
    import time

    engine.advance("EURUSD", "1h")
    time.sleep(0.6)
    record = engine.check_expiry("EURUSD", "1h")
    assert record.state == SetupState.COOLDOWN
    assert "stale" in record.invalidation_reason


def test_persists_across_engine_restarts() -> None:
    db_path = tempfile.mktemp(suffix=".db")
    engine1 = SetupMemoryEngine(store=SetupMemoryStore(db_path))
    engine1.advance("GBPUSD", "4h", broker_symbol="GBP_USD")
    engine1.advance("GBPUSD", "4h")
    engine1.strengthen("GBPUSD", "4h", 0.3)

    engine2 = SetupMemoryEngine(store=SetupMemoryStore(db_path))
    record = engine2.get("GBPUSD", "4h")
    assert record is not None
    assert record.state == SetupState.BUILDING_SETUP
    assert record.state_confidence == pytest.approx(0.3)


def test_expected_fields_are_stored(engine: SetupMemoryEngine) -> None:
    record = engine.advance(
        "EURUSD",
        "1h",
        expected_direction=Direction.LONG,
        expected_move_size=42.0,
        expected_holding_period_s=7200.0,
    )
    assert record.expected_direction == Direction.LONG
    assert record.expected_move_size == 42.0
    assert record.expected_holding_period_s == 7200.0


def test_feature_vector_shape_and_neutral_default(engine: SetupMemoryEngine) -> None:
    fv = engine.feature_vector("UNKNOWNSYM", "1h")
    assert fv["state"] == "IDLE"
    assert len(fv["state_onehot"]) == 7
    assert sum(fv["state_onehot"]) == 1.0

    engine.advance("EURUSD", "1h")
    fv2 = engine.feature_vector("EURUSD", "1h")
    assert fv2["state"] == "WATCHING"
    assert fv2["state_onehot"][SetupMemoryEngine.STATE_ORDER.index(SetupState.WATCHING)] == 1.0


def test_history_records_transitions(engine: SetupMemoryEngine) -> None:
    engine.advance("EURUSD", "1h")
    engine.advance("EURUSD", "1h")
    hist = engine.history("EURUSD", "1h")
    assert len(hist) == 2
    assert hist[0]["to_state"] == "BUILDING_SETUP"
    assert hist[1]["to_state"] == "WATCHING"


def test_check_all_expiry_covers_every_record(engine: SetupMemoryEngine) -> None:
    engine.advance("EURUSD", "1h")
    engine.advance("GBPUSD", "4h")
    results = engine.check_all_expiry()
    keys = {(r.symbol, r.timeframe) for r in results}
    assert ("EURUSD", "1h") in keys
    assert ("GBPUSD", "4h") in keys
