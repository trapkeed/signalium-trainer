# QDE Intelligence Architecture

## Overview

The Quant Decision Engine (QDE) is a **single trading intelligence**. It is the sole reasoning system in the Signalium Trainer that may produce trade decisions. Nothing else in the codebase produces decisions.

The QDE owns 13 trading responsibilities and uses 11 internal cognitive capabilities to fulfil them. The capabilities are internal faculties — they do not exist independently, they do not decide, and they do not communicate with anything outside the QDE.

---

## The Single Intelligence Principle

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QDE Intelligence                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Internal Capabilities                     │   │
│  │                                                             │   │
│  │  Market Understanding    Opportunity Discovery              │   │
│  │  Expected Value Reasoning   Risk Reasoning                  │   │
│  │  Exposure Reasoning         Trade Management                │   │
│  │  Position Management        Capital Preservation            │   │
│  │  Market Memory              Reflection                      │   │
│  │  Learning                                                   │   │
│  │                                                             │   │
│  │  [ All advisory — none produce decisions ]                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                    QDEIntelligence.reason()                         │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
                         QDEDecision
                   (The only output that leaves)
```

---

## Data Flow

### Input

```
MarketSnapshot
  ├── instrument, timestamp, bid, ask, spread_pips
  ├── recent_bars (OHLCV history)
  ├── atr_14, atr_50, rsi_14, adx_14, bb_width, ema_fast, ema_slow
  ├── session, liquidity, volume_ratio
  ├── account (AccountState)
  ├── open_trades (List[OpenTrade])
  ├── all_open_trades (portfolio-wide)
  └── features (model-computed signals dict)
```

### Capability Outputs (Internal Only)

| Capability | Output Type | Advisory Content |
|---|---|---|
| Market Understanding | `MarketContext` | Regime, trend, volatility, structural levels |
| Opportunity Discovery | `OpportunityReport` | Scored setup candidates |
| Expected Value Reasoning | `EVScore` | EV, win probability, R/R ratio |
| Risk Reasoning | `RiskAssessment` | Stop placement, pip distance, environment |
| Exposure Reasoning | `ExposureReport` | Gross exposure, headroom, correlation risk |
| Trade Management | `TradeManagementSignals` | Flagged trade IDs needing action |
| Position Management | `PositionSizeRecommendation` | Lot size, risk amount, Kelly fraction |
| Capital Preservation | `PreservationStatus` | Circuit breaker status, limits |
| Market Memory | `MemoryContext` | Pattern matches, historical win rates |
| Reflection | `ReflectionInsight` | Post-trade accuracy, lessons |
| Learning | `LearningUpdate` | Updated trust scores, regime priors |

### Output

```
QDEDecision
  ├── decision_type: ENTER | EXIT | SCALE | PARTIAL_CLOSE | BREAK_EVEN | HOLD | PRESERVE
  ├── entry: EntrySignal (direction, price, stop, TP, lot size, EV score)
  ├── exits: List[ExitSignal] (trade_id, price, reason, urgency)
  ├── scaling: List[ScalingSignal] (trade_id, additional lots, new stop)
  ├── partial_closes: List[PartialCloseSignal] (trade_id, fraction)
  ├── break_evens: List[BreakEvenSignal] (trade_id, new stop)
  ├── regime, regime_transition, trend_direction, trend_strength
  ├── reversal_detected, continuation_detected
  ├── circuit_breaker, preservation_active
  ├── capability_contributions (capability → weighted confidence)
  ├── reasoning_summary (human-readable trace)
  └── overall_confidence
```

---

## Reasoning Cycle — Six Stages

```
Stage 1: CAPITAL PRESERVATION (safety gate)
  CapitalPreservationCapability → PreservationStatus
  If circuit breaker TRIGGERED → return QDEDecision.preserve() immediately
  No further processing.

Stage 2: MARKET UNDERSTANDING
  MarketUnderstandingCapability → MarketContext
  MarketMemoryCapability.recall() → MemoryContext (regime history)
  [Fulfils: Market Understanding, Regime Understanding]

Stage 3: TRADE MANAGEMENT (open position evaluation)
  TradeManagementCapability → TradeManagementSignals
  resolve_exits() → List[ExitSignal]
  resolve_break_evens() → List[BreakEvenSignal]
  resolve_partial_closes() → List[PartialCloseSignal]
  [Fulfils: Exit, Break Even, Partial Close, Trade Duration,
            Reversal Detection, Continuation Detection]

Stage 4: OPPORTUNITY DISCOVERY & EV (entry path, if entries allowed)
  OpportunityDiscoveryCapability → OpportunityReport
  MarketMemoryCapability.recall() → MemoryContext (pattern match)
  RiskReasoningCapability → RiskAssessment
  ExpectedValueReasoningCapability → EVScore
  → build EntrySignal if EV threshold met
  [Fulfils: Entry, position sizing inputs]

Stage 5: EXPOSURE VALIDATION
  ExposureReasoningCapability → ExposureReport
  If exposure exhausted → veto entry
  [Fulfils: Exposure]

Stage 6: DECISION SYNTHESIS
  All capability outputs synthesised into one QDEDecision.
  classify_decision_type() → DecisionType
  compute_contributions() → capability confidence scores
  compute_overall_confidence() → 0-1 score
  PositionManagementCapability → lot size incorporated into EntrySignal
  [Fulfils: Position Size, Scaling]
  → QDEDecision (the only output)
```

---

## 13 QDE Responsibilities → Capability Owners

| Responsibility | Primary Capability | Stage |
|---|---|---|
| Entry | Opportunity Discovery + EV Reasoning | 4 |
| Exit | Trade Management | 3 |
| Position Size | Position Management | 4–6 |
| Scaling | Trade Management + Position Management | 3, 5 |
| Partial Close | Trade Management | 3 |
| Break Even | Trade Management | 3 |
| Exposure | Exposure Reasoning | 5 |
| Capital Preservation | Capital Preservation | 1 |
| Trade Duration | Trade Management | 3 |
| Reversal Detection | Trade Management | 3 |
| Continuation Detection | Trade Management | 3 |
| Regime Understanding | Market Understanding | 2 |
| Market Understanding | Market Understanding | 2 |

---

## 11 Capabilities → QDE Responsibility Contribution

| Capability | Contributes To |
|---|---|
| Market Understanding | Regime Understanding, Market Understanding, context for all |
| Opportunity Discovery | Entry |
| Expected Value Reasoning | Entry (EV gate) |
| Risk Reasoning | Entry (stop/TP), Position Size |
| Exposure Reasoning | Exposure, Entry (veto) |
| Trade Management | Exit, Break Even, Partial Close, Scaling, Duration, Reversal, Continuation |
| Position Management | Position Size, Scaling |
| Capital Preservation | Capital Preservation (circuit breaker), Entry (gate) |
| Market Memory | Adjusts EV priors, Regime Understanding enrichment |
| Reflection | Post-trade — feeds Learning |
| Learning | Updates capability weights for next cycle |

---

## Invariant Enforcement

The architecture enforces the single-intelligence principle through code structure:

1. `QDEDecision` is defined in `cap_models.py` — it is **not imported** by any capability module. Capability modules only import their own output types.

2. The only module that imports `QDEDecision` and constructs it is `qde_intelligence.py`.

3. Capability outputs have no methods named `decide`, `trade`, `enter`, `exit`, `buy`, or `sell`. They contain scored data only.

4. The `reason()` method of `QDEIntelligence` is the only code path that produces a `QDEDecision`. It is called externally; all internal stages only produce capability output types.

5. `reflect_and_learn()` is a post-trade method — it processes outcomes and updates priors but does not produce a `QDEDecision`.
