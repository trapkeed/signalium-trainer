---
name: C13 Patience Engine design
description: NO_ACTION reward, patience scoring, adaptive quality threshold, skipped-opportunity lifecycle. Package at trainer/qde/brain/patience/.
---

## Package location

`trainer/qde/brain/patience/` (new package)

- `patience_types.py` — all enums, dataclasses, constants
- `patience_scoring.py` — `patience_score()`, `minimum_quality_threshold()`, reward math
- `patience_engine.py` — `evaluate_wait_decision()`, `PatienceEngine`
- `__init__.py` — re-exports everything

## Three public callables (C13 interface)

```python
from trainer.qde.brain.patience import evaluate_wait_decision, patience_score, minimum_quality_threshold
```

**`evaluate_wait_decision(cognitive_decision, opp_score, quality_threshold=None, account_features=None, capital_unavailable=False, now=None) → WaitDecision`**
Checks 7 conditions in priority order; collects all applicable WaitReasons:
1. COGNITIVE_PASSIVE (action=WAIT/IGNORE)
2. UNCERTAINTY_HIGH (uncertainty_score ≥ 0.55)
3. OVERTRADING (is_overtrading or trade_frequency_ratio ≥ 1.5)
4. CONFIRMATION_PENDING (action=MONITOR)
5. IMPULSIVE_GUARD (entry_urgency > 0.70 and conviction_score < 0.40)
6. LOW_QUALITY (opportunity_score < quality_threshold)
7. CAPITAL_UNAVAILABLE (explicit flag)

**`patience_score(state) → PatienceMetrics`**
composite = 0.50×discipline + 0.30×(1−impulsive_rate) + 0.20×(1−miss_rate)

**`minimum_quality_threshold(state) → float`**
Adaptive version of _TIER_3_MIN=0.20. Raises on impulsive entries or poor skip accuracy; lowers on high miss_rate or drought. Bounded [0.10, 0.55].

## NO_ACTION reward design

**Prospective** (at skip time): base=0.20 + per-reason bonuses. Reduced by -0.15 if opp_score ≥ 0.65. Clamped [0, 1].

**Realized** (at resolution):
- CORRECT_SKIP: +0.60 + 0.40×tanh(pnl/0.5) → [0.60, 1.00]
- MISSED_OPPORTUNITY: -0.40 - 0.40×tanh(pnl/0.5) → [-0.40, -0.80]
- NEUTRAL: 0.0

## SkippedOpportunity lifecycle

1. `evaluate_wait_decision()` returns `WaitDecision` with `skip_record` pre-built
2. `PatienceEngine.record_skip(decision)` persists to bounded history (max 500)
3. `PatienceEngine.resolve_skipped_opportunity(skip_id, outcome, pnl_estimate)` sets realized reward, updates counters, calls `_update_threshold()`
4. `PatienceEngine.age_pending_skips(window_seconds)` auto-resolves as NEUTRAL

## Adaptive threshold adjustment rules

Applied to `state.current_threshold` on each resolve:
- +THRESHOLD_STEP_UP (0.025): impulsive_entry_rate > 0.30
- +THRESHOLD_STEP_UP (0.025): discipline_score < 0.40 (wrong skips dominate)
- -THRESHOLD_STEP_DOWN (0.015): miss_rate > 0.50 (too conservative)
- -THRESHOLD_STEP_DOWN (0.015): drought (>90% of recent_wait_flags are True)

**Important**: when discipline < 0.40 AND miss_rate > 0.50, both adjustments apply simultaneously. Net = +0.010 (rise wins). Tests covering "lowers when miss_rate high" must ensure discipline ≥ 0.40.

## Integration points (existing infrastructure)

- Takes `CognitiveDecision` (decision/decision_types.py) — reads .action, .uncertainty_score, .conviction_score, .entry_urgency
- Takes `OpportunityScore` (opportunity/opportunity_score.py from C11) — reads .opportunity_score
- Optionally takes `AccountFeatures` (qde_account_models.py) — reads .is_overtrading, .trade_frequency_ratio
- Does NOT import QDEDecision or any execution-layer type
- DEFAULT_QUALITY_THRESHOLD=0.20 mirrors _TIER_3_MIN from capital_priority_engine
- UNCERTAINTY_NO_TRADE_THRESHOLD=0.55 mirrors prefers_no_trade() in uncertainty_engine

## PatienceEngine

One instance per symbol (mirrors BrainState per-symbol pattern). Key methods: `evaluate_wait_decision()`, `record_skip()`, `record_trade(was_impulsive)`, `record_overtrading_event()`, `resolve_skipped_opportunity()`, `age_pending_skips()`, `patience_score()`, `minimum_quality_threshold()`, `summary()`.
