"""Phase 21.5 — Lifecycle States, Transition Map, and State Metadata."""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Set


class LifecycleState(str, Enum):
    # ── Forward path ─────────────────────────────────────────────────────────
    IDEA = "IDEA"
    WATCHLIST = "WATCHLIST"
    PREPARATION = "PREPARATION"
    ENTRY_READY = "ENTRY_READY"
    PENDING_ENTRY = "PENDING_ENTRY"
    ACTIVE = "ACTIVE"
    POSITION_BUILDING = "POSITION_BUILDING"
    PROTECTED = "PROTECTED"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    RUNNER = "RUNNER"
    TRAILING = "TRAILING"
    EXIT_READY = "EXIT_READY"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    # ── Alternative outcomes ─────────────────────────────────────────────────
    INVALIDATED = "INVALIDATED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    STOPPED = "STOPPED"
    FORCED_EXIT = "FORCED_EXIT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


# ── Terminal states (no further transitions) ──────────────────────────────────
TERMINAL_STATES: FrozenSet[LifecycleState] = frozenset(
    {
        LifecycleState.CLOSED,
        LifecycleState.INVALIDATED,
        LifecycleState.CANCELLED,
        LifecycleState.EXPIRED,
    }
)

# States that resolve to CLOSED
CLOSING_STATES: FrozenSet[LifecycleState] = frozenset(
    {
        LifecycleState.STOPPED,
        LifecycleState.FORCED_EXIT,
        LifecycleState.EMERGENCY_EXIT,
    }
)

# States where a position is open
POSITION_OPEN_STATES: FrozenSet[LifecycleState] = frozenset(
    {
        LifecycleState.ACTIVE,
        LifecycleState.POSITION_BUILDING,
        LifecycleState.PROTECTED,
        LifecycleState.PARTIAL_EXIT,
        LifecycleState.RUNNER,
        LifecycleState.TRAILING,
        LifecycleState.EXIT_READY,
        LifecycleState.CLOSING,
        LifecycleState.STOPPED,
        LifecycleState.FORCED_EXIT,
        LifecycleState.EMERGENCY_EXIT,
    }
)

# Pre-entry states
PRE_ENTRY_STATES: FrozenSet[LifecycleState] = frozenset(
    {
        LifecycleState.IDEA,
        LifecycleState.WATCHLIST,
        LifecycleState.PREPARATION,
        LifecycleState.ENTRY_READY,
        LifecycleState.PENDING_ENTRY,
    }
)

# ── Valid transitions map ─────────────────────────────────────────────────────
VALID_TRANSITIONS: Dict[LifecycleState, FrozenSet[LifecycleState]] = {
    LifecycleState.IDEA: frozenset(
        {
            LifecycleState.WATCHLIST,
            LifecycleState.INVALIDATED,
            LifecycleState.CANCELLED,
        }
    ),
    LifecycleState.WATCHLIST: frozenset(
        {
            LifecycleState.PREPARATION,
            LifecycleState.INVALIDATED,
            LifecycleState.CANCELLED,
            LifecycleState.EXPIRED,
        }
    ),
    LifecycleState.PREPARATION: frozenset(
        {
            LifecycleState.ENTRY_READY,
            LifecycleState.WATCHLIST,
            LifecycleState.INVALIDATED,
            LifecycleState.CANCELLED,
            LifecycleState.EXPIRED,
        }
    ),
    LifecycleState.ENTRY_READY: frozenset(
        {
            LifecycleState.PENDING_ENTRY,
            LifecycleState.PREPARATION,
            LifecycleState.INVALIDATED,
            LifecycleState.CANCELLED,
        }
    ),
    LifecycleState.PENDING_ENTRY: frozenset(
        {
            LifecycleState.ACTIVE,
            LifecycleState.ENTRY_READY,
            LifecycleState.CANCELLED,
            LifecycleState.EXPIRED,
        }
    ),
    LifecycleState.ACTIVE: frozenset(
        {
            LifecycleState.POSITION_BUILDING,
            LifecycleState.PROTECTED,
            LifecycleState.PARTIAL_EXIT,
            LifecycleState.EXIT_READY,
            LifecycleState.CLOSING,
            LifecycleState.STOPPED,
            LifecycleState.FORCED_EXIT,
            LifecycleState.EMERGENCY_EXIT,
        }
    ),
    LifecycleState.POSITION_BUILDING: frozenset(
        {
            LifecycleState.ACTIVE,
            LifecycleState.PROTECTED,
            LifecycleState.STOPPED,
            LifecycleState.EMERGENCY_EXIT,
        }
    ),
    LifecycleState.PROTECTED: frozenset(
        {
            LifecycleState.PARTIAL_EXIT,
            LifecycleState.RUNNER,
            LifecycleState.EXIT_READY,
            LifecycleState.STOPPED,
        }
    ),
    LifecycleState.PARTIAL_EXIT: frozenset(
        {
            LifecycleState.RUNNER,
            LifecycleState.TRAILING,
            LifecycleState.CLOSING,
        }
    ),
    LifecycleState.RUNNER: frozenset(
        {
            LifecycleState.TRAILING,
            LifecycleState.EXIT_READY,
            LifecycleState.CLOSING,
        }
    ),
    LifecycleState.TRAILING: frozenset(
        {
            LifecycleState.EXIT_READY,
            LifecycleState.CLOSING,
            LifecycleState.STOPPED,
        }
    ),
    LifecycleState.EXIT_READY: frozenset(
        {
            LifecycleState.CLOSING,
        }
    ),
    LifecycleState.CLOSING: frozenset(
        {
            LifecycleState.CLOSED,
        }
    ),
    LifecycleState.STOPPED: frozenset(
        {
            LifecycleState.CLOSED,
        }
    ),
    LifecycleState.FORCED_EXIT: frozenset(
        {
            LifecycleState.CLOSED,
        }
    ),
    LifecycleState.EMERGENCY_EXIT: frozenset(
        {
            LifecycleState.CLOSED,
        }
    ),
    # Terminal
    LifecycleState.CLOSED: frozenset(),
    LifecycleState.INVALIDATED: frozenset(),
    LifecycleState.CANCELLED: frozenset(),
    LifecycleState.EXPIRED: frozenset(),
}

# ── State ordering (for progress metrics) ────────────────────────────────────
STATE_ORDER: Dict[LifecycleState, int] = {
    LifecycleState.IDEA: 0,
    LifecycleState.WATCHLIST: 1,
    LifecycleState.PREPARATION: 2,
    LifecycleState.ENTRY_READY: 3,
    LifecycleState.PENDING_ENTRY: 4,
    LifecycleState.ACTIVE: 5,
    LifecycleState.POSITION_BUILDING: 6,
    LifecycleState.PROTECTED: 7,
    LifecycleState.PARTIAL_EXIT: 8,
    LifecycleState.RUNNER: 9,
    LifecycleState.TRAILING: 10,
    LifecycleState.EXIT_READY: 11,
    LifecycleState.CLOSING: 12,
    LifecycleState.CLOSED: 13,
    # Alternatives get high order (resolution states)
    LifecycleState.INVALIDATED: 14,
    LifecycleState.CANCELLED: 15,
    LifecycleState.EXPIRED: 16,
    LifecycleState.STOPPED: 17,
    LifecycleState.FORCED_EXIT: 18,
    LifecycleState.EMERGENCY_EXIT: 19,
}

# ── Expected duration hints per state (hours) ─────────────────────────────────
STATE_EXPECTED_DURATION_HOURS: Dict[LifecycleState, float] = {
    LifecycleState.IDEA: 24.0,
    LifecycleState.WATCHLIST: 48.0,
    LifecycleState.PREPARATION: 12.0,
    LifecycleState.ENTRY_READY: 4.0,
    LifecycleState.PENDING_ENTRY: 2.0,
    LifecycleState.ACTIVE: 24.0,
    LifecycleState.POSITION_BUILDING: 8.0,
    LifecycleState.PROTECTED: 48.0,
    LifecycleState.PARTIAL_EXIT: 24.0,
    LifecycleState.RUNNER: 72.0,
    LifecycleState.TRAILING: 48.0,
    LifecycleState.EXIT_READY: 2.0,
    LifecycleState.CLOSING: 1.0,
    LifecycleState.CLOSED: 0.0,
    LifecycleState.INVALIDATED: 0.0,
    LifecycleState.CANCELLED: 0.0,
    LifecycleState.EXPIRED: 0.0,
    LifecycleState.STOPPED: 0.0,
    LifecycleState.FORCED_EXIT: 0.0,
    LifecycleState.EMERGENCY_EXIT: 0.0,
}


def is_terminal(state: LifecycleState) -> bool:
    return state in TERMINAL_STATES or not VALID_TRANSITIONS.get(state)


def is_position_open(state: LifecycleState) -> bool:
    return state in POSITION_OPEN_STATES


def valid_next_states(state: LifecycleState):
    return VALID_TRANSITIONS.get(state, frozenset())


def can_transition(from_state: LifecycleState, to_state: LifecycleState) -> bool:
    return to_state in valid_next_states(from_state)
