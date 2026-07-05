# Developer Guide — Trade Lifecycle Intelligence Engine

## Architecture

The Phase 21.5 engine is a 14-module pure Python package located at
`trainer/lifecycle/`. All modules use stdlib + loguru only. No torch,
no numpy, no pandas.

## Module Map

```
trainer/lifecycle/
├── lifecycle_states.py      # LifecycleState enum, transition map, helpers
├── lifecycle_models.py      # Data classes: MarketContext, StateRecord, Event, etc.
├── state_machine.py         # Deterministic FSM with 20 transition rules
├── transition_engine.py     # Evaluates and ranks valid transitions
├── position_monitor.py      # Position management recommendations
├── lifecycle_confidence.py  # Per-state confidence, duration, reward projections
├── event_engine.py          # Detects and emits lifecycle events
├── ai_reasoning.py          # Institutional natural-language explanations
├── trade_lifecycle.py       # TradeLifecycle — the living trade object
├── lifecycle_analytics.py   # Analytics across a portfolio of trades
├── lifecycle_optimizer.py   # Optimisation recommendations from historical paths
├── lifecycle_database.py    # SQLite persistence (6 tables)
├── lifecycle_dashboard.py   # 6 dashboard views
└── lifecycle_api.py         # 16 public API endpoints
```

## Dependency Order

```
lifecycle_states
    └── lifecycle_models
            ├── state_machine
            │     └── transition_engine
            ├── position_monitor
            ├── lifecycle_confidence
            ├── event_engine
            └── ai_reasoning
                      └── trade_lifecycle
                                ├── lifecycle_analytics
                                ├── lifecycle_optimizer
                                ├── lifecycle_database
                                ├── lifecycle_dashboard
                                └── lifecycle_api
```

## Importing

```python
# Recommended: import from the package
from trainer.lifecycle import (
    TradeLifecycle,
    LifecycleAPI,
    MarketContext,
    PositionMetrics,
    LifecycleState,
)
```

## Key Design Decisions

### 1. Recommendation-Only

The engine NEVER places trades or manages orders. Every output is a
recommendation that a downstream execution layer may choose to act on.

### 2. Persistent SQLite Connection

`LifecycleDatabase` with `db_path=":memory:"` stores `self._persistent_conn`
and reuses it in the `_conn()` context manager. A new `sqlite3.connect(":memory:")`
would create a fresh empty database on every call.

### 3. Living Trade Object

`TradeLifecycle` is the single source of truth for a trade. It holds:
- Complete state history (every state ever entered)
- All lifecycle events
- Latest market context and position metrics
- All recommendation methods

### 4. Spec-from-file-location Imports in Tests

Tests use `importlib.util.spec_from_file_location` to avoid triggering
the torch import chain from the main `trainer` package. `unittest discover`
will show 0 tests for lifecycle files — this is expected. Run explicitly:

```bash
python3 -m unittest tests/unit/test_lifecycle.py -v
```

### 5. Event Severity Levels

| Severity | When used |
|---|---|
| INFO | Normal state changes, positive developments |
| WARN | Deteriorating conditions, elevated risks |
| CRITICAL | Emergency exits, trend invalidation, extreme news/vol |

## Adding a New Lifecycle State

1. Add to `LifecycleState` enum in `lifecycle_states.py`
2. Add to relevant sets (`TERMINAL_STATES`, `POSITION_OPEN_STATES`, etc.)
3. Add to `VALID_TRANSITIONS` map
4. Add to `STATE_ORDER`, `STATE_EXPECTED_DURATION_HOURS`
5. Add `TransitionRule` entries in `state_machine.py`
6. Add `_BASE_TRANSITION_PROB`, `_BASE_FAILURE_PROB`, `_BASE_REWARD` entries
   in `lifecycle_confidence.py`
7. Add headline in `ai_reasoning.py`
8. Add narrative in `lifecycle_dashboard.py`
9. Update tests

## Running Tests

```bash
cd signalium-trainer
python3 -m unittest tests/unit/test_lifecycle.py -v
```

Expected: **130+ tests, all passing**.

## Configuration

All thresholds are in `conf/lifecycle/lifecycle.yaml`. The engine reads
configuration through its constructor arguments — the YAML serves as the
reference specification for all tunable parameters.

## Error Handling

- Invalid transitions return `False` from `transition_to()` — never raise
- API endpoints return `ApiResponse(success=False, error=...)` on failures
- All methods are safe to call with `None` context / metrics
- Loguru `logger.debug()` / `logger.info()` / `logger.warning()` throughout
