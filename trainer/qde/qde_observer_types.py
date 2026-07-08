"""Phase S7 — Market Observer types and enums.

QDE moves from "reason only when asked" to an always-running market
observer: for every symbol it tracks, an internal state is maintained even
when no trade is open. The state is *context/memory*, not a rule engine —
it summarises "where in the trade lifecycle is this symbol's opportunity
right now" so the decision models (`qde_inference.py`,
`qde_position_manager.py`) get a richer, temporally-aware input instead of
a single stateless snapshot.
"""

from __future__ import annotations

from enum import Enum


class ObserverState(str, Enum):
    """Per-symbol lifecycle state maintained by `QDEMarketObserver`."""

    OBSERVING = "OBSERVING"  # baseline monitoring, no setup forming
    PREPARING = "PREPARING"  # early signs of a tradeable setup forming
    ARMED = "ARMED"  # setup quality high enough to warrant close attention
    ENTRY_WINDOW = "ENTRY_WINDOW"  # entry trigger imminent / active
    POSITION_ACTIVE = "POSITION_ACTIVE"  # a position is open for this symbol
    MANAGING = "MANAGING"  # position open and actively being managed
    EXITING = "EXITING"  # exit in progress (reduce/close signalled)
    COOLDOWN = "COOLDOWN"  # post-exit or post-invalidated-setup quiet period


# States in which no position is open ("pre-trade" side of the lifecycle).
PRE_TRADE_STATES = frozenset(
    {
        ObserverState.OBSERVING,
        ObserverState.PREPARING,
        ObserverState.ARMED,
        ObserverState.ENTRY_WINDOW,
    }
)

# States in which a position is open ("in-trade" side of the lifecycle).
IN_TRADE_STATES = frozenset(
    {
        ObserverState.POSITION_ACTIVE,
        ObserverState.MANAGING,
        ObserverState.EXITING,
    }
)


class ObserverEvent(str, Enum):
    """Reasons recorded alongside each state transition (for telemetry)."""

    SETUP_FORMING = "SETUP_FORMING"
    SETUP_STRENGTHENED = "SETUP_STRENGTHENED"
    SETUP_WEAKENED = "SETUP_WEAKENED"
    SETUP_INVALIDATED = "SETUP_INVALIDATED"
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"
    ENTRY_WINDOW_EXPIRED = "ENTRY_WINDOW_EXPIRED"
    POSITION_OPENED = "POSITION_OPENED"
    MANAGEMENT_REVIEW = "MANAGEMENT_REVIEW"
    EXIT_SIGNALLED = "EXIT_SIGNALLED"
    POSITION_CLOSED = "POSITION_CLOSED"
    COOLDOWN_ELAPSED = "COOLDOWN_ELAPSED"
    OVERTRADE_GUARD = "OVERTRADE_GUARD"  # blocked re-arming too soon
    REVENGE_TRADE_GUARD = "REVENGE_TRADE_GUARD"  # blocked re-entry after a loss
    NO_CHANGE = "NO_CHANGE"


# Valid transition map: state -> set of states reachable directly from it.
# Enforced by `QDEMarketObserver` so a malformed strategy object cannot jump
# the lifecycle (e.g. straight from OBSERVING to MANAGING).
VALID_TRANSITIONS: dict[ObserverState, frozenset[ObserverState]] = {
    ObserverState.OBSERVING: frozenset({ObserverState.OBSERVING, ObserverState.PREPARING}),
    ObserverState.PREPARING: frozenset(
        {ObserverState.PREPARING, ObserverState.ARMED, ObserverState.OBSERVING}
    ),
    ObserverState.ARMED: frozenset(
        {ObserverState.ARMED, ObserverState.ENTRY_WINDOW, ObserverState.OBSERVING}
    ),
    ObserverState.ENTRY_WINDOW: frozenset(
        {
            ObserverState.ENTRY_WINDOW,
            ObserverState.POSITION_ACTIVE,
            ObserverState.COOLDOWN,
            ObserverState.OBSERVING,
        }
    ),
    ObserverState.POSITION_ACTIVE: frozenset(
        {ObserverState.POSITION_ACTIVE, ObserverState.MANAGING}
    ),
    ObserverState.MANAGING: frozenset({ObserverState.MANAGING, ObserverState.EXITING}),
    ObserverState.EXITING: frozenset({ObserverState.EXITING, ObserverState.COOLDOWN}),
    ObserverState.COOLDOWN: frozenset({ObserverState.COOLDOWN, ObserverState.OBSERVING}),
}


def is_valid_transition(from_state: ObserverState, to_state: ObserverState) -> bool:
    return to_state in VALID_TRANSITIONS.get(from_state, frozenset())
