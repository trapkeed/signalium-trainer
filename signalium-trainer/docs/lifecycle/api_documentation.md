# API Documentation — Trade Lifecycle Intelligence Engine

## Overview

`LifecycleAPI` provides 16 endpoints for the Trade Lifecycle Intelligence
Engine. All endpoints return an `ApiResponse` with `success`, `endpoint`,
`data`, and optional `error` / `warnings` fields.

## Initialisation

```python
from trainer.lifecycle import LifecycleAPI, LifecycleDatabase

# Default in-memory database
api = LifecycleAPI()

# Persistent file database
api = LifecycleAPI(db=LifecycleDatabase("/path/to/lifecycle.db"))
```

---

## Endpoints

### 1. `create_lifecycle`

Create a new trade lifecycle starting at IDEA state.

```python
r = api.create_lifecycle(
    instrument = "EURUSD",
    direction  = "BUY",           # BUY | SELL
    confidence = 0.65,
    context    = MarketContext(...),  # optional
    tags       = ["swing", "H4"],    # optional
)
# r.data: full TradeLifecycle.to_dict()
```

---

### 2. `get_lifecycle`

Retrieve a lifecycle by trade_id.

```python
r = api.get_lifecycle(trade_id)
# r.data: lifecycle dict or DB row
```

---

### 3. `transition`

Execute a validated state transition.

```python
r = api.transition(
    trade_id   = tid,
    new_state  = LifecycleState.WATCHLIST,
    reason     = "Setup detected on H4",
    confidence = 0.68,                      # optional
    context    = MarketContext(...),         # optional
    metrics    = PositionMetrics(...),       # optional
)
# r.success = True if transition was permitted and executed
```

---

### 4. `auto_transition`

Let the engine recommend and execute the best transition automatically.

```python
r = api.auto_transition(trade_id, context=ctx)
# r.data: {"action": "TRANSITIONED" | "STAY", "new_state": ..., "reason": ...}
```

---

### 5. `recommend_transition`

Get transition recommendations without acting (read-only).

```python
r = api.recommend_transition(trade_id, context=ctx)
# r.data: TransitionRecommendation.to_dict()
```

---

### 6. `position_management`

Get position management recommendation for current state.

```python
r = api.position_management(trade_id, context=ctx, metrics=metrics)
# r.data: PositionManagementPlan.to_dict()
# Primary actions: HOLD | SCALE_IN | SCALE_OUT | REDUCE_EXPOSURE |
#                  MOVE_STOP | MOVE_TARGET | EXIT_IMMEDIATELY | WAIT
```

---

### 7. `lifecycle_confidence`

Get confidence assessment for current state.

```python
r = api.lifecycle_confidence(trade_id, context=ctx)
# r.data: LifecycleConfidence.to_dict()
# Fields: confidence, expected_duration_h, expected_reward,
#         expected_risk, transition_prob, failure_prob, notes
```

---

### 8. `detect_events`

Detect and log lifecycle events from market condition changes.

```python
r = api.detect_events(trade_id, context=ctx, metrics=metrics, prev_context=prev)
# r.data: {"new_events": N, "events": [...]}
# Events are automatically persisted to the database
```

---

### 9. `reasoning`

Generate institutional AI reasoning for the current state.

```python
r = api.reasoning(trade_id, context=ctx, metrics=metrics, prev_conf=0.65)
# r.data: ReasoningReport.to_dict()
# Fields: headline, rationale, risk_factors, action_items,
#         outlook, confidence_narrative
```

---

### 10. `state_history`

Retrieve the full state history for a trade.

```python
r = api.state_history(trade_id)
# r.data: {"history": [...StateRecord dicts...], "count": N}
```

---

### 11. `lifecycle_analytics`

Compute aggregate analytics across all active lifecycles.

```python
r = api.lifecycle_analytics(trade_ids=["T1", "T2"])  # optional filter
# r.data: LifecycleAnalyticsReport.to_dict()
# Includes: failure_rate, cancellation_rate, expiration_rate,
#           state_analytics, transition_matrix, top_paths
```

---

### 12. `optimisation`

Generate optimisation recommendations from lifecycle paths.

```python
r = api.optimisation()
# r.data: OptimisationReport.to_dict()
# Analyses: breakeven timing, trailing stop, scaling, expiry, emergency exits
```

---

### 13. `timeline_dashboard`

Lifecycle timeline view for a single trade.

```python
r = api.timeline_dashboard(trade_id)
# r.data["view"] = "lifecycle_timeline"
# r.data["timeline"] = [...step dicts...]
```

---

### 14. `confidence_dashboard`

Confidence timeline for a single trade.

```python
r = api.confidence_dashboard(trade_id)
# r.data["view"] = "confidence_timeline"
# r.data["trend"] = "IMPROVING" | "DECLINING" | "STABLE"
```

---

### 15. `transition_dashboard`

State transition matrix across all active trades.

```python
r = api.transition_dashboard()
# r.data["view"] = "transition_matrix"
# r.data["prob_matrix"] = {from_state: {to_state: probability}}
```

---

### 16. `statistics_dashboard`

Aggregate statistics panel.

```python
r = api.statistics_dashboard()
# r.data["view"] = "lifecycle_statistics"
# r.data["state_visit_counts"] = {state: count}
# r.data["final_state_dist"] = {state: count}
```

---

## Utility Endpoints

```python
# Database row counts
r = api.db_stats()

# All active trades in memory registry
r = api.active_trades()
```

## ApiResponse Fields

| Field | Type | Description |
|---|---|---|
| `success` | bool | Whether the operation succeeded |
| `endpoint` | str | Name of the endpoint called |
| `data` | dict | Response payload |
| `error` | str | Error message (if success=False) |
| `warnings` | list[str] | Non-fatal warnings |
