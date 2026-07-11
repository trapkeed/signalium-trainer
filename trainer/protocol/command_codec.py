"""QDE Command Protocol — serialisation / deserialisation (Phase C8).

``serialize_command(cmd)``
    Returns a JSON-serialisable ``dict`` representation of a
    ``ProtocolCommand``.  The dict contains only JSON-native types (str,
    int, float, bool, list, dict) so it can be passed directly to
    ``json.dumps()`` without a custom encoder.

``deserialize_command(data)``
    Reconstructs a ``ProtocolCommand`` from the dict produced by
    ``serialize_command()``.

``to_json(cmd)``
    Serialises to a compact UTF-8 JSON string.

``from_json(json_str)``
    Deserialises from a JSON string.

Design constraints
------------------
* Pure functions — no I/O, no state, no mutation.
* No import from ``qde_inference`` or ``execution/`` packages.
* The payload is stored opaquely (as-is dict); the codec does not
  re-validate it — that is the responsibility of ``validate_command()``.
"""

from __future__ import annotations

import json

from trainer.protocol.command_types import ProtocolCommand

# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def serialize_command(cmd: ProtocolCommand) -> dict:
    """Return a JSON-serialisable dict for *cmd*.

    All float values are rounded to 5 decimal places; the ``action`` enum
    is serialised as its string value.  The resulting dict can be passed
    directly to ``json.dumps()`` with no custom encoder.
    """
    return {
        "command_id": cmd.command_id,
        "qde_version": cmd.qde_version,
        "position_id": cmd.position_id,
        "symbol": cmd.symbol,
        "action": cmd.action.value,
        "payload": _serialise_payload(cmd.payload),
        "expires_at": round(cmd.expires_at, 3),
        "created_at": round(cmd.created_at, 3),
        "source_decision_id": cmd.source_decision_id,
    }


def deserialize_command(data: dict) -> ProtocolCommand:
    """Reconstruct a ``ProtocolCommand`` from a serialised dict.

    Raises ``KeyError`` if ``action`` is missing; raises ``ValueError`` if
    the ``action`` value is not a recognised ``CommandAction``.
    """
    return ProtocolCommand.from_dict(data)


def to_json(cmd: ProtocolCommand, indent: int | None = None) -> str:
    """Serialise *cmd* to a UTF-8 JSON string.

    ``indent`` controls pretty-printing (``None`` → compact).
    """
    return json.dumps(serialize_command(cmd), ensure_ascii=False, indent=indent)


def from_json(json_str: str) -> ProtocolCommand:
    """Deserialise a ``ProtocolCommand`` from a JSON string.

    Raises ``json.JSONDecodeError`` on malformed input; raises ``ValueError``
    on an unknown action.
    """
    return deserialize_command(json.loads(json_str))


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────


def _serialise_payload(payload: dict) -> dict:
    """Deep-copy a payload dict, rounding floats to 5 dp.

    Handles nested dicts and lists; non-numeric values are copied as-is.
    This ensures the payload is always JSON-safe even when callers pass
    Python floats that happen to be ``NaN`` / ``Inf`` (those get coerced to
    ``0.0`` to prevent ``json.dumps()`` failures).
    """
    out: dict = {}
    for k, v in payload.items():
        out[k] = _coerce(v)
    return out


def _coerce(value: object) -> object:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):  # NaN or Inf
            return 0.0
        return round(value, 5)
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(item) for item in value]
    return value


__all__ = [
    "serialize_command",
    "deserialize_command",
    "to_json",
    "from_json",
]
