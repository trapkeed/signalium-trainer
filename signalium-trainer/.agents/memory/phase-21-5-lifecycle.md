---
name: Phase 21.5 lifecycle package
description: Design rules and gotchas for the trainer/lifecycle/ engine.
---

## The Package

`trainer/lifecycle/` — 14 modules, stdlib + loguru only, zero model changes.
Pure recommendation layer — does NOT execute trades.

## Module Load Order (for spec_from_file_location tests)

```
lifecycle_states → lifecycle_models → state_machine → transition_engine
→ position_monitor → lifecycle_confidence → event_engine → ai_reasoning
→ trade_lifecycle → lifecycle_analytics → lifecycle_optimizer
→ lifecycle_database → lifecycle_dashboard → lifecycle_api
```

## 20 Lifecycle States

Forward path (14): IDEA → WATCHLIST → PREPARATION → ENTRY_READY → PENDING_ENTRY
→ ACTIVE → POSITION_BUILDING → PROTECTED → PARTIAL_EXIT → RUNNER → TRAILING
→ EXIT_READY → CLOSING → CLOSED

Alternative outcomes (6): INVALIDATED, CANCELLED, EXPIRED, STOPPED, FORCED_EXIT, EMERGENCY_EXIT

Terminal states (no further transitions): CLOSED, INVALIDATED, CANCELLED, EXPIRED

CLOSING_STATES (resolve to CLOSED): STOPPED, FORCED_EXIT, EMERGENCY_EXIT

## Critical API Test Gotcha

`TradeLifecycle.detect_events()` uses:
- `prev_conf=` (NOT `prev_confidence=`)
- `prev_context=`

Wrong kwarg name causes TypeError — discovered in test run.

## StateMachine Key Validation Rules

| Rule string | Passes when |
|---|---|
| `news_risk < 0.40` | context.news_risk < 0.40 |
| `spread_ok` | spread < atr × 0.30 |
| `trend_valid` | context.trend_valid == True |
| `session_ok` | session != "CLOSED" |
| `momentum_aligned` | abs(momentum) > 0.20 |

## TransitionEngine Priority

1. EMERGENCY_EXIT if news_risk > 0.80 OR volatility > 0.035
2. INVALIDATED if confidence < 0.40 OR (trend broken AND pre-entry)
3. Best allowed positive transition (max expected_benefit × confidence)
4. STAY (no conditions met)

## SQLite :memory: Pattern

Same as Phase 20/21: store `self._persistent_conn` at init; reuse in `_conn()` context manager.
New `sqlite3.connect(":memory:")` always creates a fresh empty DB.

## Database Tables (6)

trade_lifecycle, lifecycle_events, transition_rules, state_history,
lifecycle_analytics, optimisation_results

## Test Counts

- Phase 20 portfolio: 129 tests
- Phase 21 execution: 158 tests
- Phase 21.5 lifecycle: 180 tests
- Total: 467 tests (all green)

## AI Reasoning — confidence_narrative wording

- delta ≥ 0.08 → "Confidence increased from X to Y"
- delta ≤ -0.08 → "Confidence declined from X to Y"
- stable → "Confidence stable at X"

Tests check for "increased" / "declined" (lowercase, no "significantly").
