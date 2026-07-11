"""QDE Command Protocol — command validation (Phase C8).

``validate_command(cmd, now)``
    Checks envelope invariants and action-specific payload constraints.
    Returns a ``CommandValidationResult`` (truthy when valid).

Validation layers
-----------------
1. Envelope completeness — required string fields non-empty, action recognisable.
2. Temporal validity     — command has not already expired.
3. Position-ID presence  — actions targeting an existing position must carry one.
4. Payload schema        — required payload keys present with correct types.
5. Payload value ranges  — numeric fields within legal bounds.

Design constraints
------------------
* Pure function — no I/O, no state, no mutation.
* No import from ``qde_inference`` or ``execution/`` packages.
"""

from __future__ import annotations

import time

from trainer.protocol.command_types import (
    CommandAction,
    CommandValidationResult,
    ProtocolCommand,
)

# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

_VALID_DIRECTIONS = frozenset({"BUY", "SELL"})
_VALID_ORDER_TYPES = frozenset({"MARKET", "LIMIT", "STOP"})

# Actions for which a non-empty position_id is mandatory
_NEEDS_POSITION_ID = frozenset(
    {
        CommandAction.CLOSE_POSITION,
        CommandAction.PARTIAL_CLOSE,
        CommandAction.MOVE_SL,
        CommandAction.MOVE_TP,
        CommandAction.ADD_POSITION,
    }
)


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────


def validate_command(
    cmd: ProtocolCommand,
    now: float | None = None,
) -> CommandValidationResult:
    """Validate *cmd* and return a ``CommandValidationResult``.

    ``now`` defaults to ``time.time()`` and is accepted as a parameter so
    unit tests can inject a fixed timestamp without monkeypatching.
    """
    now = now if now is not None else time.time()
    errors:   list[str] = []
    warnings: list[str] = []

    _check_envelope(cmd, now, errors, warnings)
    if not errors:
        _check_payload(cmd, errors, warnings)

    return CommandValidationResult(valid=not errors, errors=errors, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────
# Layer 1–3: envelope checks
# ─────────────────────────────────────────────────────────────────────────


def _check_envelope(
    cmd: ProtocolCommand,
    now: float,
    errors: list[str],
    warnings: list[str],
) -> None:
    # ── required string fields ───────────────────────────────────────────
    if not cmd.command_id or not cmd.command_id.strip():
        errors.append("command_id must be a non-empty string.")
    if not cmd.qde_version or not cmd.qde_version.strip():
        errors.append("qde_version must be a non-empty string.")
    if not cmd.symbol or not cmd.symbol.strip():
        errors.append("symbol must be a non-empty string.")

    # ── action recognisable ──────────────────────────────────────────────
    if not isinstance(cmd.action, CommandAction):
        try:
            CommandAction(cmd.action)
        except ValueError:
            errors.append(f"Unknown action '{cmd.action}'.")

    # ── temporal validity ────────────────────────────────────────────────
    if cmd.expires_at <= now:
        errors.append(
            f"Command has already expired "
            f"(expires_at={cmd.expires_at:.3f}, now={now:.3f})."
        )
    elif cmd.expires_at - now < 1.0:
        warnings.append("Command TTL < 1 s; may expire before the EA processes it.")

    # ── position_id presence ─────────────────────────────────────────────
    if cmd.action in _NEEDS_POSITION_ID and not cmd.position_id:
        errors.append(
            f"position_id is required for {cmd.action.value} commands."
        )

    # ── created_at sanity ────────────────────────────────────────────────
    if cmd.created_at > now + 5.0:
        warnings.append("created_at is more than 5 s in the future — clock skew?")


# ─────────────────────────────────────────────────────────────────────────
# Layer 4–5: payload schema + value-range checks
# ─────────────────────────────────────────────────────────────────────────


def _check_payload(
    cmd: ProtocolCommand,
    errors: list[str],
    warnings: list[str],
) -> None:
    p = cmd.payload
    action = cmd.action

    if action == CommandAction.OPEN_POSITION:
        _check_open(p, errors, warnings)
    elif action == CommandAction.CLOSE_POSITION:
        _check_close(p, errors, warnings)
    elif action == CommandAction.PARTIAL_CLOSE:
        _check_partial_close(p, errors, warnings)
    elif action == CommandAction.MOVE_SL:
        _check_move_sl(p, errors, warnings)
    elif action == CommandAction.MOVE_TP:
        _check_move_tp(p, errors, warnings)
    elif action == CommandAction.ADD_POSITION:
        _check_add(p, errors, warnings)
    elif action == CommandAction.CANCEL_ORDER:
        _check_cancel(p, errors, warnings)


def _check_open(p: dict, errors: list[str], warnings: list[str]) -> None:
    direction = p.get("direction", "")
    if direction not in _VALID_DIRECTIONS:
        errors.append(
            f"OPEN_POSITION: direction must be one of {sorted(_VALID_DIRECTIONS)}, got '{direction}'."
        )
    size = float(p.get("size_lots", 0.0))
    if size <= 0.0:
        errors.append(f"OPEN_POSITION: size_lots must be > 0, got {size}.")
    order_type = p.get("order_type", "MARKET")
    if order_type not in _VALID_ORDER_TYPES:
        errors.append(
            f"OPEN_POSITION: order_type must be one of {sorted(_VALID_ORDER_TYPES)}, got '{order_type}'."
        )
    sl = float(p.get("stop_loss", 0.0))
    if sl <= 0.0:
        errors.append(f"OPEN_POSITION: stop_loss must be > 0, got {sl}.")
    tp = float(p.get("take_profit", 0.0))
    if tp <= 0.0:
        warnings.append("OPEN_POSITION: take_profit is 0 — consider setting a take-profit level.")
    # Limit/Stop orders need a price
    if order_type in ("LIMIT", "STOP") and float(p.get("entry_price", 0.0)) <= 0.0:
        errors.append(
            f"OPEN_POSITION: entry_price must be > 0 for {order_type} orders."
        )


def _check_close(p: dict, errors: list[str], warnings: list[str]) -> None:
    urgency = float(p.get("urgency", 1.0))
    if not (0.0 <= urgency <= 1.0):
        errors.append(f"CLOSE_POSITION: urgency must be in [0, 1], got {urgency}.")
    if not p.get("reason"):
        warnings.append("CLOSE_POSITION: reason not specified.")


def _check_partial_close(p: dict, errors: list[str], warnings: list[str]) -> None:
    frac = float(p.get("close_fraction", 0.0))
    if not (0.0 < frac <= 1.0):
        errors.append(
            f"PARTIAL_CLOSE: close_fraction must be in (0, 1], got {frac}."
        )
    if not p.get("reason"):
        warnings.append("PARTIAL_CLOSE: reason not specified.")


def _check_move_sl(p: dict, errors: list[str], warnings: list[str]) -> None:
    sl = float(p.get("new_stop_loss", 0.0))
    if sl <= 0.0:
        errors.append(f"MOVE_SL: new_stop_loss must be > 0, got {sl}.")


def _check_move_tp(p: dict, errors: list[str], warnings: list[str]) -> None:
    tp = float(p.get("new_take_profit", 0.0))
    if tp <= 0.0:
        errors.append(f"MOVE_TP: new_take_profit must be > 0, got {tp}.")


def _check_add(p: dict, errors: list[str], warnings: list[str]) -> None:
    size = float(p.get("size_lots", 0.0))
    if size <= 0.0:
        errors.append(f"ADD_POSITION: size_lots must be > 0, got {size}.")
    order_type = p.get("order_type", "MARKET")
    if order_type not in _VALID_ORDER_TYPES:
        errors.append(
            f"ADD_POSITION: order_type must be one of {sorted(_VALID_ORDER_TYPES)}, got '{order_type}'."
        )
    if order_type in ("LIMIT", "STOP") and float(p.get("entry_price", 0.0)) <= 0.0:
        errors.append(
            f"ADD_POSITION: entry_price must be > 0 for {order_type} orders."
        )


def _check_cancel(p: dict, errors: list[str], warnings: list[str]) -> None:
    if not p.get("order_id"):
        warnings.append("CANCEL_ORDER: order_id not specified in payload.")


__all__ = ["validate_command"]
