# Decision Memory Guide — Phase 21.6

## Overview

The Decision Memory is Signalium's permanent institutional record of every recommendation it has ever produced. Every BUY, SELL, NO_TRADE, WAIT, and trade management decision is captured as a `DecisionRecord` with a globally unique UUID.

## Decision Types

| Type | Category | Description |
|---|---|---|
| BUY | Entry | Open long position |
| SELL | Entry | Open short position |
| NO_TRADE | Passive | Deliberately abstain |
| WAIT | Passive | Delay until conditions improve |
| CANCEL | Exit | Cancel pending order |
| SCALE_IN | Management | Add to existing position |
| SCALE_OUT | Management | Reduce existing position |
| MOVE_STOP | Management | Adjust stop loss level |
| MOVE_TARGET | Management | Adjust take profit level |
| PARTIAL_CLOSE | Management | Close part of position |
| EXIT | Exit | Close entire position |

## Decision Record Structure

```
DecisionRecord
├── decision_id       # UUID (globally unique)
├── decision_type     # DecisionType enum
├── instrument        # e.g. "EURUSD"
├── timestamp         # Unix epoch (float)
├── snapshot          # DecisionSnapshot — full context at decision time
├── reasoning         # DecisionReasoning — structured reasoning
├── outcome           # DecisionOutcome (set later)
├── score             # DecisionScore (set by QualityEngine)
├── counterfactuals   # List[CounterfactualResult]
├── tags              # List[str]
├── lineage_id        # Links to DecisionLineage
└── trade_id          # Links to TradeLifecycle
```

## Decision Snapshot

The snapshot captures the complete market and model state at decision time:

| Field | Description |
|---|---|
| price, atr, spread | Market microstructure |
| volatility, session | Market conditions |
| liquidity_estimate | 0–1 scale |
| market_regime | TRENDING_UP, RANGING, BREAKOUT, etc. |
| volatility_regime | LOW, NORMAL, HIGH, EXTREME |
| forecast_direction | BUY or SELL |
| forecast_confidence | 0–1 scale |
| expected_move | In price units |
| expected_holding_h | Expected trade duration (hours) |
| tp_probability | P(hitting take profit) |
| sl_probability | P(hitting stop loss) |
| model_version | Producing model identifier |
| champion_model | Current champion label |
| research_score | 0–1 research conviction |
| ensemble_output | Raw ensemble signal |
| portfolio_exposure | Fraction of account |
| correlation_risk | 0–1 portfolio correlation risk |
| risk_metrics | Dict: VaR, max drawdown, etc. |
| feature_snapshot | Dict: active features at decision time |
| expert_weights | Dict: expert contribution weights |

## Decision Reasoning

Reasoning is always structured — never free-form chain-of-thought.

```python
DecisionReasoning(
    primary_reasons        = ["Uptrend confirmed", "RSI bullish"],
    supporting_factors     = ["MACD positive", "Session peak"],
    contradicting_factors  = ["High spread"],
    risk_factors           = ["News risk", "Low liquidity"],
    uncertainty_sources    = ["Forecast uncertainty ±0.002"],
    alternatives_considered = ["WAIT", "NO_TRADE", "SCALE_IN"],
    final_rationale        = "Strong trend alignment with acceptable risk.",
)
```

## Recording a Decision

```python
from trainer.memory import MemoryAPI, DecisionType, DecisionSnapshot, DecisionReasoning

api = MemoryAPI(db_path="decision_memory.db")

record = api.record_decision(
    decision_type = DecisionType.BUY,
    instrument    = "EURUSD",
    snapshot      = DecisionSnapshot(
        price               = 1.10500,
        atr                 = 0.00120,
        spread              = 0.00002,
        session             = "LONDON",
        market_regime       = "TRENDING_UP",
        forecast_confidence = 0.75,
        tp_probability      = 0.68,
        sl_probability      = 0.25,
        expected_move       = 0.00180,
    ),
    reasoning = DecisionReasoning(
        primary_reasons        = ["Trend confirmed on H4", "Strong momentum"],
        supporting_factors     = ["Session liquidity high"],
        contradicting_factors  = ["Minor resistance at 1.1060"],
        risk_factors           = ["CPI release in 4 hours"],
        uncertainty_sources    = ["Model uncertainty ±0.003"],
        alternatives_considered = ["WAIT", "NO_TRADE"],
        final_rationale        = "Risk/reward favourable at 2.7× RR.",
    ),
    tags    = ["pullback", "trend_continuation"],
    trade_id = "trade-abc-123",
)
print(record.decision_id)   # UUID
```

## Recording an Outcome

```python
api.record_outcome(
    decision_id   = record.decision_id,
    outcome_type  = OutcomeType.HIT_TP,
    actual_pnl    = 0.00180,
    hit_tp        = True,
    direction_correct = True,
    holding_time_h = 22.5,
)
```

## Outcome Types

| Outcome | Category | Description |
|---|---|---|
| CORRECT_DIRECTION | Positive | Forecast direction was correct |
| HIT_TP | Positive | Take profit was reached |
| PARTIAL_SUCCESS | Positive | Partial take profit |
| INCORRECT_DIRECTION | Negative | Forecast direction was wrong |
| HIT_SL | Negative | Stop loss was triggered |
| UNEXPECTED_MARKET_EVENT | Negative | Black swan or news event |
| CANCELLED | Neutral | Order cancelled before fill |
| EXPIRED | Neutral | Order timed out |
| MISSED_OPPORTUNITY | Neutral | NO_TRADE when market moved favourably |
| NO_TRADE | Neutral | No position taken |

## Database Tables

The memory database maintains 9 tables:

1. `decision_memory` — master record
2. `decision_snapshots` — full context snapshot
3. `decision_reasoning` — structured reasoning
4. `decision_outcomes` — recorded outcomes
5. `counterfactual_analysis` — alternative scenarios
6. `decision_scores` — quality scores
7. `decision_patterns` — discovered patterns
8. `knowledge_base` — institutional knowledge entries
9. `decision_lineage` — traceability graph

## Filtering and Retrieval

```python
# Filter by instrument
records = api.memory.filter(instrument="EURUSD")

# Filter by type with outcome
records = api.memory.filter(
    decision_type  = DecisionType.BUY,
    has_outcome    = True,
    min_confidence = 0.70,
)

# Decisions without outcomes (awaiting outcome recording)
pending = api.memory.decisions_without_outcomes()
```

## Statistics

```python
stats = api.system_stats()
# {
#   "memory": {"total_decisions": 150, "with_outcome": 120, ...},
#   "database": {"decisions": 150, "outcomes": 120, "scores": 90, ...},
#   "knowledge": {"total_entries": 200, ...},
#   "lineage": {"total_lineages": 150, "avg_depth": 7.8},
# }
```
