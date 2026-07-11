"""QDE Command Protocol — command generation (Phase C8).

``generate_command()``
    The primary factory: builds a ``ProtocolCommand`` from explicit
    components, assigns a fresh ``command_id``, and computes ``expires_at``
    from the caller-supplied TTL.

Convenience constructors
------------------------
One typed helper per action so callers never have to hand-craft payload
dicts and can instead pass strongly-typed arguments:

    open_position_command(...)
    close_position_command(...)
    partial_close_command(...)
    move_sl_command(...)
    move_tp_command(...)
    add_position_command(...)
    cancel_order_command(...)

Design constraints
------------------
* Pure functions — no I/O, no state, no mutation of arguments.
* No import from ``qde_inference`` or ``execution/`` packages.
"""

from __future__ import annotations

import time
import uuid

from trainer.protocol.command_types import (
    AddPositionPayload,
    CancelOrderPayload,
    ClosePositionPayload,
    CommandAction,
    MoveSLPayload,
    MoveTPPayload,
    OpenPositionPayload,
    PartialClosePayload,
    ProtocolCommand,
)

# Default lifetime: 30 s — long enough for network round-trips yet short
# enough that a stale command doesn't linger as market conditions change.
DEFAULT_TTL_SECONDS: float = 30.0


# ─────────────────────────────────────────────────────────────────────────
# Primary factory
# ─────────────────────────────────────────────────────────────────────────


def generate_command(
    action: CommandAction | str,
    position_id: str,
    symbol: str,
    payload: dict,
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Generate a ``ProtocolCommand`` ready for dispatch.

    Parameters
    ----------
    action:
        The ``CommandAction`` (or its string value) to issue.
    position_id:
        Target position / order identifier.  For ``OPEN_POSITION`` this is
        the pre-assigned identifier the EA will use for the new position.
    symbol:
        Canonical instrument symbol (e.g. ``"EURUSD"``).
    payload:
        Action-specific parameters.  Use the typed payload helpers
        (``OpenPositionPayload``, etc.) with ``.to_dict()`` to produce
        the correct structure.
    qde_version:
        Model version string for traceability (e.g. ``"qde-v3.1.0"``).
    ttl_seconds:
        Lifetime of the command in seconds (default 30).  After this the
        command is considered stale and the EA should discard it.
    source_decision_id:
        Optional link back to the ``QDEDecision`` that generated the command.
    now:
        Override for the current time (default ``time.time()``).  Useful in
        tests to get deterministic ``created_at`` and ``expires_at``.
    """
    now = now if now is not None else time.time()

    if isinstance(action, str):
        action = CommandAction(action)

    return ProtocolCommand(
        command_id=str(uuid.uuid4()),
        qde_version=qde_version,
        position_id=position_id,
        symbol=symbol,
        action=action,
        payload=dict(payload),
        expires_at=now + max(0.0, ttl_seconds),
        created_at=now,
        source_decision_id=source_decision_id,
    )


# ─────────────────────────────────────────────────────────────────────────
# Typed convenience constructors
# ─────────────────────────────────────────────────────────────────────────


def open_position_command(
    position_id: str,
    symbol: str,
    direction: str,
    size_lots: float,
    stop_loss: float,
    take_profit: float = 0.0,
    order_type: str = "MARKET",
    entry_price: float = 0.0,
    comment: str = "",
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Build an OPEN_POSITION command."""
    payload = OpenPositionPayload(
        direction=direction,
        size_lots=size_lots,
        order_type=order_type,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        comment=comment,
    ).to_dict()
    return generate_command(
        action=CommandAction.OPEN_POSITION,
        position_id=position_id,
        symbol=symbol,
        payload=payload,
        qde_version=qde_version,
        ttl_seconds=ttl_seconds,
        source_decision_id=source_decision_id,
        now=now,
    )


def close_position_command(
    position_id: str,
    symbol: str,
    reason: str = "MANUAL",
    urgency: float = 1.0,
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Build a CLOSE_POSITION command."""
    payload = ClosePositionPayload(reason=reason, urgency=urgency).to_dict()
    return generate_command(
        action=CommandAction.CLOSE_POSITION,
        position_id=position_id,
        symbol=symbol,
        payload=payload,
        qde_version=qde_version,
        ttl_seconds=ttl_seconds,
        source_decision_id=source_decision_id,
        now=now,
    )


def partial_close_command(
    position_id: str,
    symbol: str,
    close_fraction: float,
    reason: str = "REDUCE",
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Build a PARTIAL_CLOSE command."""
    payload = PartialClosePayload(close_fraction=close_fraction, reason=reason).to_dict()
    return generate_command(
        action=CommandAction.PARTIAL_CLOSE,
        position_id=position_id,
        symbol=symbol,
        payload=payload,
        qde_version=qde_version,
        ttl_seconds=ttl_seconds,
        source_decision_id=source_decision_id,
        now=now,
    )


def move_sl_command(
    position_id: str,
    symbol: str,
    new_stop_loss: float,
    reason: str = "TRAIL",
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Build a MOVE_SL command."""
    payload = MoveSLPayload(new_stop_loss=new_stop_loss, reason=reason).to_dict()
    return generate_command(
        action=CommandAction.MOVE_SL,
        position_id=position_id,
        symbol=symbol,
        payload=payload,
        qde_version=qde_version,
        ttl_seconds=ttl_seconds,
        source_decision_id=source_decision_id,
        now=now,
    )


def move_tp_command(
    position_id: str,
    symbol: str,
    new_take_profit: float,
    reason: str = "TARGET_ADVANCE",
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Build a MOVE_TP command."""
    payload = MoveTPPayload(new_take_profit=new_take_profit, reason=reason).to_dict()
    return generate_command(
        action=CommandAction.MOVE_TP,
        position_id=position_id,
        symbol=symbol,
        payload=payload,
        qde_version=qde_version,
        ttl_seconds=ttl_seconds,
        source_decision_id=source_decision_id,
        now=now,
    )


def add_position_command(
    position_id: str,
    symbol: str,
    size_lots: float,
    order_type: str = "MARKET",
    entry_price: float = 0.0,
    new_stop_loss: float = 0.0,
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Build an ADD_POSITION (scale-in) command."""
    payload = AddPositionPayload(
        size_lots=size_lots,
        order_type=order_type,
        entry_price=entry_price,
        new_stop_loss=new_stop_loss,
    ).to_dict()
    return generate_command(
        action=CommandAction.ADD_POSITION,
        position_id=position_id,
        symbol=symbol,
        payload=payload,
        qde_version=qde_version,
        ttl_seconds=ttl_seconds,
        source_decision_id=source_decision_id,
        now=now,
    )


def cancel_order_command(
    position_id: str,
    symbol: str,
    order_id: str = "",
    reason: str = "SIGNAL_EXPIRED",
    qde_version: str = "",
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    source_decision_id: str = "",
    now: float | None = None,
) -> ProtocolCommand:
    """Build a CANCEL_ORDER command."""
    payload = CancelOrderPayload(order_id=order_id, reason=reason).to_dict()
    return generate_command(
        action=CommandAction.CANCEL_ORDER,
        position_id=position_id,
        symbol=symbol,
        payload=payload,
        qde_version=qde_version,
        ttl_seconds=ttl_seconds,
        source_decision_id=source_decision_id,
        now=now,
    )


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "generate_command",
    "open_position_command",
    "close_position_command",
    "partial_close_command",
    "move_sl_command",
    "move_tp_command",
    "add_position_command",
    "cancel_order_command",
]
