# Phase S7 — QDE as an Always-Running Market Observer

## Goal

Before Phase S7, QDE only reasoned when explicitly invoked: once per entry
decision (`qde_inference.py`) or once per 30s review of an *already open*
position (`qde_position_manager.py`). There was no notion of QDE
continuously watching a symbol between trades.

Phase S7 adds that continuous layer: `QDEMarketObserver` is ticked on a
per-symbol basis (including symbols with no open position) and maintains a
lifecycle **state** for each one — context/memory for the existing decision
models to consult, not a new rule engine that places trades itself. No
runtime branch anywhere derives an entry/exit order directly from
`ObserverState`; it only ever changes a label plus advisory telemetry.

## State model

```
OBSERVING → PREPARING → ARMED → ENTRY_WINDOW → POSITION_ACTIVE → MANAGING → EXITING → COOLDOWN → OBSERVING
```

| State | Meaning |
|---|---|
| `OBSERVING` | Baseline monitoring, no setup forming. |
| `PREPARING` | Early signs of a tradeable setup. |
| `ARMED` | Setup quality high enough to warrant close attention. |
| `ENTRY_WINDOW` | Entry trigger imminent/active (time-boxed, `ENTRY_WINDOW_SECONDS`). |
| `POSITION_ACTIVE` | A position just opened for this symbol. |
| `MANAGING` | Position open and under ongoing review. |
| `EXITING` | Exit in progress. |
| `COOLDOWN` | Post-exit or post-invalidated-setup quiet period. |

Transitions are validated against an explicit `VALID_TRANSITIONS` table
(`qde_observer_types.py`) — an invalid jump (e.g. straight from `OBSERVING`
to `MANAGING`) is rejected and the state holds, rather than silently
applied. A position-open event always wins immediately over any pre-trade
state, since a fill can land even if the tracked state lagged by one tick.

## Anti-pattern guards (Phase S7 goals)

| Goal | Mechanism |
|---|---|
| Minimize overtrading | `MIN_REARM_GAP_SECONDS` dwell requirement in `OBSERVING` before re-arming; trips record `OVERTRADE_GUARD` / `guard_tripped="OVERTRADE"`. |
| Minimize low-quality entries | `ENTRY_WINDOW_THRESHOLD` readiness gate before `ARMED → ENTRY_WINDOW`; readiness blends microstructure/volatility/liquidity/session quality. |
| Minimize revenge trading | Cooldown scales with `consecutive_losses` (`BASE_COOLDOWN_SECONDS` × `LOSS_COOLDOWN_MULTIPLIER` after `REVENGE_TRADE_LOSS_STREAK`); attempts to re-arm during the extended window trip `REVENGE_TRADE_GUARD`. |
| Minimize late entries | `ENTRY_WINDOW_SECONDS` time-boxes the entry opportunity; expiry emits `ENTRY_WINDOW_EXPIRED` and returns to `OBSERVING`. |
| Maximize opportunity capture | `ObserverMonitor`/`ObserverDashboard` track the `ARMED → POSITION_ACTIVE` conversion rate (`opportunity_capture_rate`) so under-capture is visible. |

## Modules

| Module | Purpose |
|---|---|
| `qde_observer_types.py` | `ObserverState`, `ObserverEvent` enums; transition table. |
| `qde_observer_models.py` | `SymbolObservationInput` (per-tick input, composes existing `MarketContext`/`PositionMetrics`/`PortfolioMetrics`), `ObserverMemory` (persisted per-symbol context), `ObservationResult` (per-tick output). |
| `qde_observer.py` | `QDEMarketObserver.observe()` — pure per-tick state-machine evaluation; no owned loop, caller drives cadence (same convention as `qde_position_manager.py`). |
| `qde_observer_database.py` | SQLite (WAL) persistence: `observer_state` (current snapshot per symbol, for restart rehydration) + `observer_transitions` (append-only history). |
| `qde_observer_monitoring.py` | `ObserverMonitor` — rolling-window telemetry: state distribution, guard-trip rates, opportunity-capture rate. |
| `qde_observer_api.py` | `QDEObserverAPI` — wraps engine + db + monitor + dashboard behind an `ApiResponse` envelope; rehydrates state from the DB on startup. |
| `qde_observer_dashboard.py` | `ObserverDashboard` — 4 chart-ready views: state timeline, dwell histogram, transition funnel, guard trips. |

## API contract

Exposed via `qde_server.py` (FastAPI, port 8001) and proxied by
`apps/trainer-api/src/routes/qde.ts` (internal-service routes, not
lifecycle-authority-gated — same tier as position management):

| Method | Path | Purpose |
|---|---|---|
| POST | `/qde/observer/{instrument}/observe` | Feed one tick, advance state. |
| GET | `/qde/observer/{instrument}/state` | Current memory snapshot. |
| GET | `/qde/observer/states?state=` | All tracked symbols, optional state filter. |
| GET | `/qde/observer/{instrument}/history?limit=` | Transition history. |
| GET | `/qde/observer/summary` | Aggregate counts by state. |
| GET | `/qde/observer/{instrument}/telemetry` | Rolling-window telemetry for one symbol. |
| GET | `/qde/observer/telemetry` | Telemetry for all symbols. |
| GET | `/qde/observer/dashboard/state-timeline` | Bucketed state distribution over time. |
| GET | `/qde/observer/dashboard/dwell-histogram` | Current dwell time per state. |
| GET | `/qde/observer/dashboard/transition-funnel` | OBSERVING → POSITION_ACTIVE conversion. |
| GET | `/qde/observer/dashboard/guard-trips` | Overtrade/revenge-trade guard trip counts over time. |

## Verification performed

- `py_compile` clean across all new/changed `trainer/qde/*.py` modules.
- `tsc --noEmit` on `apps/trainer-api` — no new errors from `qde.ts` changes
  (a pre-existing, unrelated failure in `health.ts` from an unbuilt
  `packages/api-zod/dist` persists regardless of this change).
- In-process functional test driving one symbol through the full cycle
  `OBSERVING → PREPARING → ARMED → ENTRY_WINDOW → POSITION_ACTIVE →
  MANAGING → EXITING → COOLDOWN → OBSERVING`, confirming correct
  transitions, event labelling, and cooldown timing after a losing close.
- `QDEObserverAPI` round-trip test: observe → persist → simulate a process
  restart with a fresh `QDEObserverAPI` instance → confirm state rehydrates
  from SQLite rather than resetting to `OBSERVING`.
- FastAPI app import test confirming all new `/qde/observer/*` routes
  register without conflicting with existing routes.
