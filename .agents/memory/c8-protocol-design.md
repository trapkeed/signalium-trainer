---
name: C8 QDE Command Protocol
description: Design decisions for trainer/protocol — the normalised command envelope from QDE (brain) to EA (execution body).
---

## Rule
`trainer/protocol/` is the sole boundary between QDE cognitive output and execution actions.
Nothing in `trainer/protocol/` may import `qde_inference` or `QDEDecision`.

## Structure
- `command_types.py`    — `CommandAction` (7 verbs), typed payload classes, `ProtocolCommand`, `CommandValidationResult`
- `command_builder.py`  — `generate_command()` primary factory + 7 typed convenience constructors
- `command_validator.py`— `validate_command()` pure function; 5 validation layers (envelope, temporal, position-id, payload schema, value ranges)
- `command_codec.py`    — `serialize_command()`, `deserialize_command()`, `to_json()`, `from_json()`
- `__init__.py`         — re-exports all three primary callables + all types

## Key decisions

**`payload` is always a plain dict** — typed payload helpers (`OpenPositionPayload` etc.) are for construction/validation only; they are never stored live in the envelope.  This keeps serialisation trivial.

**`command_id` is a fresh UUID from `generate_command()`** — never re-used; EA uses it for idempotent execution.

**`expires_at` = now + ttl_seconds (default 30 s)** — validated at `validate_command()` time; expired commands are blocked with an error, not a warning.

**`position_id` is required for CLOSE/PARTIAL_CLOSE/MOVE_SL/MOVE_TP/ADD** — not required for OPEN_POSITION or CANCEL_ORDER (enforced by `_NEEDS_POSITION_ID` frozenset in validator).

**MOVE_SL/MOVE_TP require price > 0** — zero means "unchanged" in some broker APIs but is rejected here to prevent ambiguous no-ops.

**`str(CommandAction.X)` changed in Python 3.11+** — use `.value` or direct `==` comparison (the class inherits from `str`), not `str()`.

## Why
- Clean separation: QDE expresses intent in cognitive vocabulary; protocol translates to EA-compatible action verbs without coupling the two packages.
- `payload: dict` avoids polymorphic type unions across the transport boundary.
- Expiry enforcement prevents stale commands from reaching the EA after market conditions change.
