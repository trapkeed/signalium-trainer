---
name: Phase 21.6 memory package
description: Design rules and gotchas for the trainer/memory/ Institutional Decision Memory engine.
---

## The Package

`trainer/memory/` — 15 modules (+ __init__), stdlib + loguru only, zero model changes.
Permanent institutional record of every Signalium decision. NOT a retraining system.

## Module Load Order (for spec_from_file_location tests)

```
decision_types → decision_models → decision_memory → outcome_tracker →
quality_engine → counterfactual_engine → pattern_discovery → self_evaluation →
knowledge_base → decision_lineage → decision_benchmarking → memory_search →
memory_database → memory_dashboard → memory_api
```

## Decision Types (11)

BUY, SELL, NO_TRADE, WAIT, CANCEL, SCALE_IN, SCALE_OUT, MOVE_STOP, MOVE_TARGET, PARTIAL_CLOSE, EXIT

## Outcome Types (10)

CORRECT_DIRECTION, INCORRECT_DIRECTION, HIT_TP, HIT_SL, CANCELLED, EXPIRED,
MISSED_OPPORTUNITY, PARTIAL_SUCCESS, NO_TRADE, UNEXPECTED_MARKET_EVENT

## Quality Factor Weights (10 factors, must sum to 1.0)

FORECAST_ACCURACY(0.20), EXECUTION_QUALITY(0.15), RISK_MANAGEMENT(0.15),
LIFECYCLE_MANAGEMENT(0.10), PORTFOLIO_IMPACT(0.10), CALIBRATION(0.10),
CONFIDENCE_RELIABILITY(0.08), EXPECTED_VALUE(0.07), CONSISTENCY(0.03), DECISION_ROBUSTNESS(0.02)

**assert abs(sum(QUALITY_WEIGHTS.values()) - 1.0) < 1e-9** — asserted at module load.

## Grade Thresholds

EXCELLENT≥0.85, GOOD≥0.70, ACCEPTABLE≥0.55, POOR≥0.40, REJECT<0.40

## Critical API Gotchas

- `CounterfactualEngine.analyse()` returns `[]` if `record.outcome is None` — must record outcome first.
- `PatternDiscovery.discover()` returns early with "Insufficient data" if < 3 completed decisions.
- `QualityEngine.score()` can run without outcome — uses forecast probabilities as proxy.
- `MemoryDatabase` with `:memory:` — same persistent connection pattern as all prior phases.

## Database Tables (9)

decision_memory, decision_snapshots, decision_reasoning, decision_outcomes,
counterfactual_analysis, decision_scores, decision_patterns, knowledge_base, decision_lineage

## Counterfactual Scenarios (6)

WAIT, NO_TRADE, LARGER_TP (1.5×), TIGHTER_SL (0.7×), DELAYED_ENTRY, EARLY_TRAIL

## Knowledge Base Entry Types (9)

DECISION, TRADE, EXPERIMENT, MODEL, FEATURE, REGIME, RESEARCH, LIFECYCLE_EVENT, PORTFOLIO_STATE

## Lineage Depth (8 levels)

Decision → Forecast → Model → Dataset → FeatureSet → TrainingRun → Experiment → GitCommit

## Dashboard Views (8)

decision_timeline, decision_tree, quality_distribution, failure_analysis,
counterfactual_explorer, knowledge_graph, learning_timeline, decision_heatmaps

## API Endpoints (16)

record_decision, get_decision, record_outcome, score_decision, score_all,
analyse_counterfactuals, summarise_outcomes, discover_patterns, self_evaluate,
search_decisions, knowledge_search, index_decision_in_knowledge_base,
build_lineage, benchmark, get_dashboard, system_stats

## Test Count

- Phase 21.6 memory: 157 tests
- Total across all phases: 624 tests (all green)
