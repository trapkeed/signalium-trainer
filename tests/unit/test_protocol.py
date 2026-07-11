"""Tests for QDE Command Protocol (Phase C8).

Coverage
--------
CommandAction enum                          tests 1–5
Payload dataclasses to_dict / from_dict     tests 6–19
ProtocolCommand construction / helpers      tests 20–26
CommandValidationResult                     tests 27–28
validate_command — envelope                 tests 29–38
validate_command — payload per action       tests 39–48
generate_command factory                    tests 49–54
Convenience constructors                    tests 55–61
serialize_command / deserialize_command     tests 62–65
to_json / from_json                         tests 66–68
No QDEDecision import in protocol package   test  69
Package-level re-exports                    test  70
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid

import pytest

# ── module under test ──────────────────────────────────────────────────────
from trainer.protocol import (
    CommandAction,
    CommandValidationResult,
    ProtocolCommand,
    add_position_command,
    cancel_order_command,
    close_position_command,
    deserialize_command,
    from_json,
    generate_command,
    move_sl_command,
    move_tp_command,
    open_position_command,
    partial_close_command,
    serialize_command,
    to_json,
    validate_command,
)
from trainer.protocol.command_types import (
    AddPositionPayload,
    CancelOrderPayload,
    ClosePositionPayload,
    MoveSLPayload,
    MoveTPPayload,
    OpenPositionPayload,
    PartialClosePayload,
    PAYLOAD_CLASS,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────

T0 = 1_700_000_000.0  # fixed "now" for deterministic timestamps


def _future(offset: float = 60.0) -> float:
    return T0 + offset


def valid_open_payload() -> dict:
    return OpenPositionPayload(
        direction="BUY", size_lots=0.1, stop_loss=1.0800, take_profit=1.1000
    ).to_dict()


def valid_cmd(
    action: CommandAction = CommandAction.OPEN_POSITION,
    position_id: str = "pos-001",
    symbol: str = "EURUSD",
    payload: dict | None = None,
    qde_version: str = "qde-v1.0",
    ttl: float = 60.0,
) -> ProtocolCommand:
    return ProtocolCommand(
        command_id=str(uuid.uuid4()),
        qde_version=qde_version,
        position_id=position_id,
        symbol=symbol,
        action=action,
        payload=payload if payload is not None else valid_open_payload(),
        expires_at=_future(ttl),
        created_at=T0,
    )


# ─────────────────────────────────────────────────────────────────────────
# 1–5  CommandAction enum
# ─────────────────────────────────────────────────────────────────────────


def test_01_all_seven_actions_exist():
    expected = {
        "OPEN_POSITION", "CLOSE_POSITION", "PARTIAL_CLOSE",
        "MOVE_SL", "MOVE_TP", "ADD_POSITION", "CANCEL_ORDER",
    }
    assert {a.value for a in CommandAction} == expected


def test_02_requires_existing_position():
    assert CommandAction.CLOSE_POSITION.requires_existing_position() is True
    assert CommandAction.PARTIAL_CLOSE.requires_existing_position()  is True
    assert CommandAction.MOVE_SL.requires_existing_position()        is True
    assert CommandAction.MOVE_TP.requires_existing_position()        is True
    assert CommandAction.ADD_POSITION.requires_existing_position()   is True
    assert CommandAction.OPEN_POSITION.requires_existing_position()  is False
    assert CommandAction.CANCEL_ORDER.requires_existing_position()   is False


def test_03_is_entry():
    assert CommandAction.OPEN_POSITION.is_entry() is True
    assert CommandAction.ADD_POSITION.is_entry()  is True
    assert CommandAction.CLOSE_POSITION.is_entry() is False
    assert CommandAction.MOVE_SL.is_entry()        is False


def test_04_is_exit():
    assert CommandAction.CLOSE_POSITION.is_exit() is True
    assert CommandAction.PARTIAL_CLOSE.is_exit()  is True
    assert CommandAction.OPEN_POSITION.is_exit()  is False
    assert CommandAction.MOVE_SL.is_exit()        is False


def test_05_str_enum_value():
    # .value is always the bare string regardless of Python version
    assert CommandAction.OPEN_POSITION.value == "OPEN_POSITION"
    # Because CommandAction inherits from str, direct equality with the
    # raw string still works (the member IS a str).
    assert CommandAction.OPEN_POSITION == "OPEN_POSITION"
    assert CommandAction("MOVE_SL") == CommandAction.MOVE_SL


# ─────────────────────────────────────────────────────────────────────────
# 6–19  Payload dataclasses
# ─────────────────────────────────────────────────────────────────────────


def test_06_open_payload_to_dict_keys():
    p = OpenPositionPayload(direction="BUY", size_lots=0.1, stop_loss=1.08, take_profit=1.10)
    d = p.to_dict()
    for key in ("direction", "size_lots", "order_type", "entry_price", "stop_loss", "take_profit", "comment"):
        assert key in d, f"missing key: {key}"


def test_07_open_payload_round_trip():
    p = OpenPositionPayload(direction="SELL", size_lots=0.5, stop_loss=1.1200, take_profit=1.0950,
                             order_type="LIMIT", entry_price=1.1100, comment="breakout")
    assert OpenPositionPayload.from_dict(p.to_dict()).direction == "SELL"
    assert abs(OpenPositionPayload.from_dict(p.to_dict()).size_lots - 0.5) < 1e-4


def test_08_close_payload_round_trip():
    p = ClosePositionPayload(reason="THESIS_INVALIDATED", urgency=0.9)
    p2 = ClosePositionPayload.from_dict(p.to_dict())
    assert p2.reason == "THESIS_INVALIDATED"
    assert abs(p2.urgency - 0.9) < 1e-4


def test_09_partial_close_payload_round_trip():
    p = PartialClosePayload(close_fraction=0.5, reason="REDUCE")
    p2 = PartialClosePayload.from_dict(p.to_dict())
    assert abs(p2.close_fraction - 0.5) < 1e-4
    assert p2.reason == "REDUCE"


def test_10_move_sl_payload_round_trip():
    p = MoveSLPayload(new_stop_loss=1.0850, reason="BREAKEVEN")
    p2 = MoveSLPayload.from_dict(p.to_dict())
    assert abs(p2.new_stop_loss - 1.0850) < 1e-4
    assert p2.reason == "BREAKEVEN"


def test_11_move_tp_payload_round_trip():
    p = MoveTPPayload(new_take_profit=1.1050, reason="TARGET_ADVANCE")
    p2 = MoveTPPayload.from_dict(p.to_dict())
    assert abs(p2.new_take_profit - 1.1050) < 1e-4


def test_12_add_payload_round_trip():
    p = AddPositionPayload(size_lots=0.2, order_type="MARKET", new_stop_loss=1.0820)
    p2 = AddPositionPayload.from_dict(p.to_dict())
    assert abs(p2.size_lots - 0.2) < 1e-4
    assert abs(p2.new_stop_loss - 1.0820) < 1e-4


def test_13_cancel_payload_round_trip():
    p = CancelOrderPayload(order_id="ord-999", reason="SIGNAL_EXPIRED")
    p2 = CancelOrderPayload.from_dict(p.to_dict())
    assert p2.order_id == "ord-999"


def test_14_payload_class_dispatch_table_complete():
    """PAYLOAD_CLASS must have an entry for every CommandAction."""
    for action in CommandAction:
        assert action in PAYLOAD_CLASS, f"PAYLOAD_CLASS missing entry for {action}"


def test_15_open_payload_floats_rounded_4dp():
    p = OpenPositionPayload(size_lots=0.123456789, stop_loss=1.080012345, take_profit=1.100056789)
    d = p.to_dict()
    # size_lots rounded to 4dp
    assert len(str(d["size_lots"]).split(".")[-1]) <= 4 or d["size_lots"] == round(0.123456789, 4)


def test_16_close_payload_urgency_defaults_to_1():
    p = ClosePositionPayload()
    assert p.urgency == 1.0


def test_17_add_payload_entry_price_defaults_to_zero():
    p = AddPositionPayload(size_lots=0.1)
    assert p.entry_price == 0.0


def test_18_partial_close_payload_default_fraction():
    p = PartialClosePayload()
    assert p.close_fraction == 0.5


def test_19_move_sl_default_reason():
    p = MoveSLPayload()
    assert p.reason == "TRAIL"


# ─────────────────────────────────────────────────────────────────────────
# 20–26  ProtocolCommand
# ─────────────────────────────────────────────────────────────────────────


def test_20_command_has_all_required_fields():
    cmd = valid_cmd()
    for attr in ("command_id", "qde_version", "position_id", "symbol", "action",
                 "payload", "expires_at", "created_at"):
        assert hasattr(cmd, attr), f"missing attribute: {attr}"


def test_21_is_expired_false_while_live():
    cmd = valid_cmd(ttl=60.0)
    assert cmd.is_expired(now=T0) is False


def test_22_is_expired_true_after_ttl():
    cmd = valid_cmd(ttl=10.0)
    assert cmd.is_expired(now=T0 + 11.0) is True


def test_23_ttl_remaining():
    cmd = valid_cmd(ttl=60.0)
    assert abs(cmd.ttl(now=T0) - 60.0) < 0.5


def test_24_ttl_zero_when_expired():
    cmd = valid_cmd(ttl=5.0)
    assert cmd.ttl(now=T0 + 100.0) == 0.0


def test_25_to_dict_contains_required_keys():
    cmd = valid_cmd()
    d = cmd.to_dict()
    for key in ("command_id", "qde_version", "position_id", "symbol", "action",
                "payload", "expires_at", "created_at", "source_decision_id"):
        assert key in d


def test_26_from_dict_round_trip():
    cmd = valid_cmd()
    cmd2 = ProtocolCommand.from_dict(cmd.to_dict())
    assert cmd2.command_id == cmd.command_id
    assert cmd2.symbol == cmd.symbol
    assert cmd2.action == cmd.action
    assert cmd2.position_id == cmd.position_id
    assert abs(cmd2.expires_at - cmd.expires_at) < 0.01


# ─────────────────────────────────────────────────────────────────────────
# 27–28  CommandValidationResult
# ─────────────────────────────────────────────────────────────────────────


def test_27_valid_result_is_truthy():
    r = CommandValidationResult(valid=True)
    assert bool(r) is True


def test_28_invalid_result_is_falsy():
    r = CommandValidationResult(valid=False, errors=["missing field"])
    assert bool(r) is False


# ─────────────────────────────────────────────────────────────────────────
# 29–38  validate_command — envelope checks
# ─────────────────────────────────────────────────────────────────────────


def test_29_valid_open_position_passes():
    cmd = valid_cmd(action=CommandAction.OPEN_POSITION)
    result = validate_command(cmd, now=T0)
    assert result.valid, result.errors


def test_30_missing_command_id_fails():
    cmd = valid_cmd()
    cmd.command_id = ""
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("command_id" in e for e in result.errors)


def test_31_missing_qde_version_fails():
    cmd = valid_cmd(qde_version="")
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("qde_version" in e for e in result.errors)


def test_32_missing_symbol_fails():
    cmd = valid_cmd(symbol="")
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("symbol" in e for e in result.errors)


def test_33_expired_command_fails():
    cmd = valid_cmd(ttl=-10.0)   # already expired
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("expired" in e.lower() for e in result.errors)


def test_34_close_without_position_id_fails():
    cmd = valid_cmd(action=CommandAction.CLOSE_POSITION, position_id="",
                    payload=ClosePositionPayload().to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("position_id" in e for e in result.errors)


def test_35_partial_close_without_position_id_fails():
    cmd = valid_cmd(action=CommandAction.PARTIAL_CLOSE, position_id="",
                    payload=PartialClosePayload().to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid


def test_36_move_sl_without_position_id_fails():
    cmd = valid_cmd(action=CommandAction.MOVE_SL, position_id="",
                    payload=MoveSLPayload(new_stop_loss=1.08).to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid


def test_37_cancel_order_without_position_id_passes():
    """CANCEL_ORDER does not require position_id in the envelope."""
    cmd = valid_cmd(action=CommandAction.CANCEL_ORDER, position_id="",
                    payload=CancelOrderPayload(order_id="ord-1").to_dict())
    result = validate_command(cmd, now=T0)
    assert result.valid, result.errors


def test_38_open_position_without_position_id_passes():
    """OPEN_POSITION does not require a pre-existing position_id."""
    cmd = valid_cmd(action=CommandAction.OPEN_POSITION, position_id="")
    result = validate_command(cmd, now=T0)
    # Only payload errors (stop_loss provided) so should pass envelope check
    # Rebuild with valid payload explicitly
    cmd2 = valid_cmd(action=CommandAction.OPEN_POSITION, position_id="")
    r = validate_command(cmd2, now=T0)
    assert r.valid, r.errors


# ─────────────────────────────────────────────────────────────────────────
# 39–48  validate_command — payload checks per action
# ─────────────────────────────────────────────────────────────────────────


def test_39_open_position_zero_size_fails():
    cmd = valid_cmd()
    cmd.payload = OpenPositionPayload(direction="BUY", size_lots=0.0,
                                       stop_loss=1.08, take_profit=1.10).to_dict()
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("size_lots" in e for e in result.errors)


def test_40_open_position_bad_direction_fails():
    cmd = valid_cmd()
    cmd.payload = OpenPositionPayload(direction="LONG", size_lots=0.1,
                                       stop_loss=1.08).to_dict()
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("direction" in e for e in result.errors)


def test_41_open_position_zero_stop_loss_fails():
    cmd = valid_cmd()
    cmd.payload = OpenPositionPayload(direction="BUY", size_lots=0.1, stop_loss=0.0).to_dict()
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("stop_loss" in e for e in result.errors)


def test_42_open_position_limit_requires_entry_price():
    cmd = valid_cmd()
    cmd.payload = OpenPositionPayload(direction="BUY", size_lots=0.1,
                                       stop_loss=1.08, order_type="LIMIT",
                                       entry_price=0.0).to_dict()
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("entry_price" in e for e in result.errors)


def test_43_close_position_invalid_urgency_fails():
    cmd = valid_cmd(action=CommandAction.CLOSE_POSITION,
                    payload=ClosePositionPayload(urgency=1.5).to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("urgency" in e for e in result.errors)


def test_44_partial_close_invalid_fraction_fails():
    cmd = valid_cmd(action=CommandAction.PARTIAL_CLOSE,
                    payload=PartialClosePayload(close_fraction=0.0).to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("close_fraction" in e for e in result.errors)


def test_45_partial_close_fraction_above_one_fails():
    cmd = valid_cmd(action=CommandAction.PARTIAL_CLOSE,
                    payload=PartialClosePayload(close_fraction=1.5).to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid


def test_46_move_sl_zero_price_fails():
    cmd = valid_cmd(action=CommandAction.MOVE_SL,
                    payload=MoveSLPayload(new_stop_loss=0.0).to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("new_stop_loss" in e for e in result.errors)


def test_47_move_tp_zero_price_fails():
    cmd = valid_cmd(action=CommandAction.MOVE_TP,
                    payload=MoveTPPayload(new_take_profit=0.0).to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("new_take_profit" in e for e in result.errors)


def test_48_add_position_zero_size_fails():
    cmd = valid_cmd(action=CommandAction.ADD_POSITION,
                    payload=AddPositionPayload(size_lots=0.0).to_dict())
    result = validate_command(cmd, now=T0)
    assert not result.valid
    assert any("size_lots" in e for e in result.errors)


# ─────────────────────────────────────────────────────────────────────────
# 49–54  generate_command factory
# ─────────────────────────────────────────────────────────────────────────


def test_49_generate_command_returns_protocol_command():
    cmd = generate_command(
        action=CommandAction.OPEN_POSITION,
        position_id="pos-1",
        symbol="GBPUSD",
        payload=valid_open_payload(),
        qde_version="qde-v2",
        now=T0,
    )
    assert isinstance(cmd, ProtocolCommand)


def test_50_generate_command_sets_unique_command_id():
    cmd1 = generate_command(CommandAction.OPEN_POSITION, "p1", "EURUSD",
                             valid_open_payload(), now=T0)
    cmd2 = generate_command(CommandAction.OPEN_POSITION, "p2", "EURUSD",
                             valid_open_payload(), now=T0)
    assert cmd1.command_id != cmd2.command_id


def test_51_generate_command_expires_at_correct():
    cmd = generate_command(CommandAction.OPEN_POSITION, "p1", "EURUSD",
                            valid_open_payload(), ttl_seconds=45.0, now=T0)
    assert abs(cmd.expires_at - (T0 + 45.0)) < 0.01


def test_52_generate_command_accepts_string_action():
    cmd = generate_command("CLOSE_POSITION", "p1", "EURUSD",
                            ClosePositionPayload(reason="MANUAL").to_dict(),
                            qde_version="v1", now=T0)
    assert cmd.action == CommandAction.CLOSE_POSITION


def test_53_generate_command_source_decision_id():
    cmd = generate_command(CommandAction.OPEN_POSITION, "p1", "EURUSD",
                            valid_open_payload(), source_decision_id="dec-abc", now=T0)
    assert cmd.source_decision_id == "dec-abc"


def test_54_generate_command_copies_payload():
    payload = valid_open_payload()
    cmd = generate_command(CommandAction.OPEN_POSITION, "p1", "EURUSD", payload, now=T0)
    payload["size_lots"] = 99.9   # mutate original
    assert cmd.payload["size_lots"] != 99.9  # command payload is a copy


# ─────────────────────────────────────────────────────────────────────────
# 55–61  Convenience constructors
# ─────────────────────────────────────────────────────────────────────────


def test_55_open_position_command():
    cmd = open_position_command("p1", "EURUSD", "BUY", 0.1, 1.0800,
                                  take_profit=1.1000, qde_version="v1", now=T0)
    assert cmd.action == CommandAction.OPEN_POSITION
    assert cmd.payload["direction"] == "BUY"
    assert abs(cmd.payload["size_lots"] - 0.1) < 1e-4


def test_56_close_position_command():
    cmd = close_position_command("p1", "EURUSD", reason="THESIS_INVALIDATED",
                                   qde_version="v1", now=T0)
    assert cmd.action == CommandAction.CLOSE_POSITION
    assert cmd.payload["reason"] == "THESIS_INVALIDATED"


def test_57_partial_close_command():
    cmd = partial_close_command("p1", "EURUSD", close_fraction=0.5,
                                  qde_version="v1", now=T0)
    assert cmd.action == CommandAction.PARTIAL_CLOSE
    assert abs(cmd.payload["close_fraction"] - 0.5) < 1e-4


def test_58_move_sl_command():
    cmd = move_sl_command("p1", "EURUSD", new_stop_loss=1.0820,
                           reason="BREAKEVEN", qde_version="v1", now=T0)
    assert cmd.action == CommandAction.MOVE_SL
    assert abs(cmd.payload["new_stop_loss"] - 1.0820) < 1e-4


def test_59_move_tp_command():
    cmd = move_tp_command("p1", "EURUSD", new_take_profit=1.1050,
                           qde_version="v1", now=T0)
    assert cmd.action == CommandAction.MOVE_TP
    assert abs(cmd.payload["new_take_profit"] - 1.1050) < 1e-4


def test_60_add_position_command():
    cmd = add_position_command("p1", "EURUSD", size_lots=0.05,
                                 new_stop_loss=1.0810, qde_version="v1", now=T0)
    assert cmd.action == CommandAction.ADD_POSITION
    assert abs(cmd.payload["size_lots"] - 0.05) < 1e-4


def test_61_cancel_order_command():
    cmd = cancel_order_command("p1", "EURUSD", order_id="ord-42",
                                 reason="SIGNAL_EXPIRED", qde_version="v1", now=T0)
    assert cmd.action == CommandAction.CANCEL_ORDER
    assert cmd.payload["order_id"] == "ord-42"


# ─────────────────────────────────────────────────────────────────────────
# 62–65  serialize_command / deserialize_command
# ─────────────────────────────────────────────────────────────────────────


def test_62_serialize_returns_dict():
    cmd = valid_cmd()
    d = serialize_command(cmd)
    assert isinstance(d, dict)


def test_63_serialized_dict_is_json_safe():
    cmd = valid_cmd()
    d = serialize_command(cmd)
    # Must not raise
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_64_deserialize_round_trip():
    cmd = valid_cmd()
    cmd2 = deserialize_command(serialize_command(cmd))
    assert cmd2.command_id == cmd.command_id
    assert cmd2.action     == cmd.action
    assert cmd2.symbol     == cmd.symbol
    assert abs(cmd2.expires_at - cmd.expires_at) < 0.01


def test_65_serialize_action_is_string():
    cmd = valid_cmd()
    d = serialize_command(cmd)
    assert isinstance(d["action"], str)
    assert d["action"] == cmd.action.value


# ─────────────────────────────────────────────────────────────────────────
# 66–68  to_json / from_json
# ─────────────────────────────────────────────────────────────────────────


def test_66_to_json_returns_string():
    cmd = valid_cmd()
    s = to_json(cmd)
    assert isinstance(s, str)


def test_67_to_json_is_valid_json():
    cmd = valid_cmd()
    s = to_json(cmd)
    parsed = json.loads(s)   # must not raise
    assert isinstance(parsed, dict)


def test_68_from_json_round_trip():
    cmd = valid_cmd()
    cmd2 = from_json(to_json(cmd))
    assert cmd2.command_id == cmd.command_id
    assert cmd2.action     == cmd.action
    assert cmd2.symbol     == cmd.symbol


# ─────────────────────────────────────────────────────────────────────────
# 69  No QDEDecision imports in the protocol package
# ─────────────────────────────────────────────────────────────────────────


def test_69_no_qde_decision_import_in_protocol():
    """trainer.protocol must not import qde_inference or QDEDecision."""
    import importlib
    import pkgutil
    import trainer.protocol

    bad_modules: list[str] = []
    pkg_path = trainer.protocol.__path__

    for _, modname, _ in pkgutil.walk_packages(pkg_path, prefix="trainer.protocol."):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        src_file = getattr(mod, "__file__", "") or ""
        if not src_file.endswith(".py"):
            continue
        with open(src_file) as fh:
            src = fh.read()
        # Detect only actual import statements, not docstring mentions
        for line in src.splitlines():
            stripped = line.strip()
            if re.match(r"^(from|import)\s", stripped):
                if "qde_inference" in stripped or "QDEDecision" in stripped:
                    bad_modules.append(f"{modname}: {stripped}")

    assert not bad_modules, (
        "protocol package must not import qde_inference or QDEDecision:\n"
        + "\n".join(bad_modules)
    )


# ─────────────────────────────────────────────────────────────────────────
# 70  Package-level re-exports
# ─────────────────────────────────────────────────────────────────────────


def test_70_package_re_exports_three_primary_callables():
    import trainer.protocol as proto
    assert callable(proto.generate_command)
    assert callable(proto.validate_command)
    assert callable(proto.serialize_command)
