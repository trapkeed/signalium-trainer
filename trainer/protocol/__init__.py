"""QDE Command Protocol — public package interface (Phase C8).

The three primary callables exposed at package level:

    ``generate_command(action, position_id, symbol, payload, ...)``
        Build a ``ProtocolCommand`` with a fresh ``command_id`` and
        computed ``expires_at``.

    ``validate_command(cmd, now)``
        Check envelope invariants and payload constraints; returns a
        ``CommandValidationResult`` (truthy when valid).

    ``serialize_command(cmd)``
        Return a JSON-serialisable ``dict``; pair with
        ``deserialize_command()`` for full round-trip.

Seven typed convenience constructors are also re-exported so callers never
need to hand-craft payload dicts:

    open_position_command / close_position_command / partial_close_command
    move_sl_command / move_tp_command / add_position_command
    cancel_order_command

Core types
----------
    ``CommandAction``           — the seven command verbs
    ``ProtocolCommand``         — universal command envelope
    ``CommandValidationResult`` — validation outcome
    Payload classes:
        ``OpenPositionPayload``, ``ClosePositionPayload``,
        ``PartialClosePayload``, ``MoveSLPayload``, ``MoveTPPayload``,
        ``AddPositionPayload``, ``CancelOrderPayload``
"""

from trainer.protocol.command_builder import (
    DEFAULT_TTL_SECONDS,
    add_position_command,
    cancel_order_command,
    close_position_command,
    generate_command,
    move_sl_command,
    move_tp_command,
    open_position_command,
    partial_close_command,
)
from trainer.protocol.command_codec import (
    deserialize_command,
    from_json,
    serialize_command,
    to_json,
)
from trainer.protocol.command_types import (
    AddPositionPayload,
    CancelOrderPayload,
    ClosePositionPayload,
    CommandAction,
    CommandValidationResult,
    MoveSLPayload,
    MoveTPPayload,
    OpenPositionPayload,
    PAYLOAD_CLASS,
    PartialClosePayload,
    ProtocolCommand,
)
from trainer.protocol.command_validator import validate_command

__all__ = [
    # ── primary callables ──────────────────────────────────────────────
    "generate_command",
    "validate_command",
    "serialize_command",
    # ── codec ──────────────────────────────────────────────────────────
    "deserialize_command",
    "to_json",
    "from_json",
    # ── convenience constructors ───────────────────────────────────────
    "open_position_command",
    "close_position_command",
    "partial_close_command",
    "move_sl_command",
    "move_tp_command",
    "add_position_command",
    "cancel_order_command",
    # ── types ──────────────────────────────────────────────────────────
    "CommandAction",
    "ProtocolCommand",
    "CommandValidationResult",
    "OpenPositionPayload",
    "ClosePositionPayload",
    "PartialClosePayload",
    "MoveSLPayload",
    "MoveTPPayload",
    "AddPositionPayload",
    "CancelOrderPayload",
    "PAYLOAD_CLASS",
    "DEFAULT_TTL_SECONDS",
]
