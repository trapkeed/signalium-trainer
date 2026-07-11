"""QDE Command Protocol — canonical command types (Phase C8).

The QDE is the brain; the EA (Expert Advisor / execution body) is the hands.
This module defines the *normalised command envelope* that flows from the QDE
to any connected execution body over any transport layer.

Every command carries a universal envelope (``ProtocolCommand``) plus an
action-specific ``payload`` dict.  Typed payload helpers (``OpenPositionPayload``,
etc.) are provided for construction and validation; they are always stored as
plain dicts inside the envelope so the envelope itself remains serialisation-
agnostic.

Seven command actions
---------------------
    OPEN_POSITION   — instruct EA to enter a new position.
    CLOSE_POSITION  — instruct EA to close an existing position fully.
    PARTIAL_CLOSE   — instruct EA to close a fraction of an existing position.
    MOVE_SL         — instruct EA to move the stop-loss on an existing position.
    MOVE_TP         — instruct EA to move the take-profit on an existing position.
    ADD_POSITION    — instruct EA to scale into an existing position.
    CANCEL_ORDER    — instruct EA to cancel a pending (not yet filled) order.

Envelope required fields (present on every command regardless of action)
------------------------------------------------------------------------
    command_id          — unique UUID for this command (dedup / idempotency).
    qde_version         — model version that issued the command (traceability).
    position_id         — target position / order identifier.
    symbol              — canonical instrument symbol (e.g. ``"EURUSD"``).
    action              — one of the seven ``CommandAction`` values.
    payload             — action-specific parameters as a plain dict.
    expires_at          — Unix timestamp after which the command is stale.

Design constraints
------------------
* No import from ``qde_inference`` or ``execution/`` package.
* ``payload`` is always a plain ``dict`` — no live typed objects cross the
  envelope boundary so serialisation remains trivial.
* All dataclasses implement ``to_dict()`` / ``from_dict()`` with floats
  rounded to 4 dp.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

# ─────────────────────────────────────────────────────────────────────────
# CommandAction
# ─────────────────────────────────────────────────────────────────────────


class CommandAction(str, Enum):
    """The seven normalised command verbs the QDE issues to the EA."""

    OPEN_POSITION = "OPEN_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    MOVE_SL = "MOVE_SL"
    MOVE_TP = "MOVE_TP"
    ADD_POSITION = "ADD_POSITION"
    CANCEL_ORDER = "CANCEL_ORDER"

    def requires_existing_position(self) -> bool:
        """True for actions that target an already-open position."""
        return self in (
            CommandAction.CLOSE_POSITION,
            CommandAction.PARTIAL_CLOSE,
            CommandAction.MOVE_SL,
            CommandAction.MOVE_TP,
            CommandAction.ADD_POSITION,
        )

    def is_entry(self) -> bool:
        """True if the command requests new market exposure."""
        return self in (CommandAction.OPEN_POSITION, CommandAction.ADD_POSITION)

    def is_exit(self) -> bool:
        """True if the command reduces or removes market exposure."""
        return self in (CommandAction.CLOSE_POSITION, CommandAction.PARTIAL_CLOSE)


# ─────────────────────────────────────────────────────────────────────────
# Payload dataclasses — typed helpers for constructing payload dicts
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class OpenPositionPayload:
    """Payload for OPEN_POSITION commands."""

    direction: str = "BUY"  # "BUY" | "SELL"
    size_lots: float = 0.0
    order_type: str = "MARKET"  # "MARKET" | "LIMIT" | "STOP"
    entry_price: float = 0.0  # 0.0 → current market price
    stop_loss: float = 0.0
    take_profit: float = 0.0
    comment: str = ""

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "size_lots": round(self.size_lots, 4),
            "order_type": self.order_type,
            "entry_price": round(self.entry_price, 5),
            "stop_loss": round(self.stop_loss, 5),
            "take_profit": round(self.take_profit, 5),
            "comment": self.comment,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OpenPositionPayload":
        return cls(
            direction=d.get("direction", "BUY"),
            size_lots=float(d.get("size_lots", 0.0)),
            order_type=d.get("order_type", "MARKET"),
            entry_price=float(d.get("entry_price", 0.0)),
            stop_loss=float(d.get("stop_loss", 0.0)),
            take_profit=float(d.get("take_profit", 0.0)),
            comment=d.get("comment", ""),
        )


@dataclass
class ClosePositionPayload:
    """Payload for CLOSE_POSITION commands."""

    reason: str = (
        "MANUAL"  # e.g. "THESIS_INVALIDATED" | "TAKE_PROFIT" | "STOP_LOSS" | "MANUAL" | "TIMEOUT"
    )
    urgency: float = 1.0  # 0-1; 1 = execute immediately at market

    def to_dict(self) -> dict:
        return {"reason": self.reason, "urgency": round(self.urgency, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> "ClosePositionPayload":
        return cls(reason=d.get("reason", "MANUAL"), urgency=float(d.get("urgency", 1.0)))


@dataclass
class PartialClosePayload:
    """Payload for PARTIAL_CLOSE commands."""

    close_fraction: float = 0.5  # 0 < fraction ≤ 1.0
    reason: str = "REDUCE"

    def to_dict(self) -> dict:
        return {"close_fraction": round(self.close_fraction, 4), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "PartialClosePayload":
        return cls(
            close_fraction=float(d.get("close_fraction", 0.5)),
            reason=d.get("reason", "REDUCE"),
        )


@dataclass
class MoveSLPayload:
    """Payload for MOVE_SL commands."""

    new_stop_loss: float = 0.0
    reason: str = "TRAIL"  # "BREAKEVEN" | "TRAIL" | "CONVICTION_DROP"

    def to_dict(self) -> dict:
        return {"new_stop_loss": round(self.new_stop_loss, 5), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "MoveSLPayload":
        return cls(
            new_stop_loss=float(d.get("new_stop_loss", 0.0)), reason=d.get("reason", "TRAIL")
        )


@dataclass
class MoveTPPayload:
    """Payload for MOVE_TP commands."""

    new_take_profit: float = 0.0
    reason: str = "TARGET_ADVANCE"

    def to_dict(self) -> dict:
        return {"new_take_profit": round(self.new_take_profit, 5), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "MoveTPPayload":
        return cls(
            new_take_profit=float(d.get("new_take_profit", 0.0)),
            reason=d.get("reason", "TARGET_ADVANCE"),
        )


@dataclass
class AddPositionPayload:
    """Payload for ADD_POSITION (scale-in) commands."""

    size_lots: float = 0.0
    order_type: str = "MARKET"
    entry_price: float = 0.0  # 0.0 → current market price
    new_stop_loss: float = 0.0  # combined stop for the total (enlarged) position; 0.0 = unchanged

    def to_dict(self) -> dict:
        return {
            "size_lots": round(self.size_lots, 4),
            "order_type": self.order_type,
            "entry_price": round(self.entry_price, 5),
            "new_stop_loss": round(self.new_stop_loss, 5),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AddPositionPayload":
        return cls(
            size_lots=float(d.get("size_lots", 0.0)),
            order_type=d.get("order_type", "MARKET"),
            entry_price=float(d.get("entry_price", 0.0)),
            new_stop_loss=float(d.get("new_stop_loss", 0.0)),
        )


@dataclass
class CancelOrderPayload:
    """Payload for CANCEL_ORDER commands."""

    order_id: str = ""
    reason: str = "SIGNAL_EXPIRED"

    def to_dict(self) -> dict:
        return {"order_id": self.order_id, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "CancelOrderPayload":
        return cls(order_id=d.get("order_id", ""), reason=d.get("reason", "SIGNAL_EXPIRED"))


# Dispatch table: action → typed payload class (used by validator / codec)
PAYLOAD_CLASS: dict[CommandAction, type] = {
    CommandAction.OPEN_POSITION: OpenPositionPayload,
    CommandAction.CLOSE_POSITION: ClosePositionPayload,
    CommandAction.PARTIAL_CLOSE: PartialClosePayload,
    CommandAction.MOVE_SL: MoveSLPayload,
    CommandAction.MOVE_TP: MoveTPPayload,
    CommandAction.ADD_POSITION: AddPositionPayload,
    CommandAction.CANCEL_ORDER: CancelOrderPayload,
}


# ─────────────────────────────────────────────────────────────────────────
# ProtocolCommand — universal command envelope
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ProtocolCommand:
    """The universal envelope for every QDE → EA command.

    The envelope is action-agnostic; action-specific data lives in
    ``payload`` (always a plain dict, never a live typed object).

    ``command_id`` uniquely identifies this command and is used by the EA
    for deduplication and idempotent execution.  ``source_decision_id``
    links the command back to the ``QDEDecision`` (or brain decision) that
    originated it for full traceability.
    """

    # Required fields on every command
    command_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    qde_version: str = ""
    position_id: str = ""
    symbol: str = ""
    action: CommandAction = CommandAction.OPEN_POSITION
    payload: dict = field(default_factory=dict)
    expires_at: float = field(default_factory=lambda: time.time() + 30.0)
    created_at: float = field(default_factory=time.time)

    # Optional traceability link back to the originating decision
    source_decision_id: str = ""

    # ── helpers ──────────────────────────────────────────────────────────────

    def is_expired(self, now: float | None = None) -> bool:
        """True if the command's TTL has elapsed."""
        now = now if now is not None else time.time()
        return now >= self.expires_at

    def ttl(self, now: float | None = None) -> float:
        """Remaining lifetime in seconds (0.0 if already expired)."""
        now = now if now is not None else time.time()
        return max(0.0, self.expires_at - now)

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "command_id": self.command_id,
            "qde_version": self.qde_version,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "payload": dict(self.payload),
            "expires_at": round(self.expires_at, 3),
            "created_at": round(self.created_at, 3),
            "source_decision_id": self.source_decision_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProtocolCommand":
        return cls(
            command_id=d.get("command_id", str(uuid.uuid4())),
            qde_version=d.get("qde_version", ""),
            position_id=d.get("position_id", ""),
            symbol=d.get("symbol", ""),
            action=CommandAction(d["action"]),
            payload=dict(d.get("payload", {})),
            expires_at=float(d.get("expires_at", time.time() + 30.0)),
            created_at=float(d.get("created_at", time.time())),
            source_decision_id=d.get("source_decision_id", ""),
        )


# ─────────────────────────────────────────────────────────────────────────
# CommandValidationResult
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class CommandValidationResult:
    """Outcome of ``validate_command()``.

    Truthy when the command is valid (``bool(result) is True``).
    ``errors`` contains blocking failures; ``warnings`` are non-blocking.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


__all__ = [
    "CommandAction",
    "OpenPositionPayload",
    "ClosePositionPayload",
    "PartialClosePayload",
    "MoveSLPayload",
    "MoveTPPayload",
    "AddPositionPayload",
    "CancelOrderPayload",
    "PAYLOAD_CLASS",
    "ProtocolCommand",
    "CommandValidationResult",
]
