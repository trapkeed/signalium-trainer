# QDE Capability Contract

## Purpose

This document defines the contract every capability module in `trainer/qde/capabilities/` must satisfy. The contract is not enforced by a type system alone — it is enforced by architecture, code review, and this document.

**Violating the capability contract is a breaking architectural change.**

---

## The Contract

Every capability module must satisfy all of the following:

### 1. Advisory outputs only

A capability returns an advisory output object. The output describes the current state of the market, a scored candidate, or a recommended parameter. It does NOT:

- Produce a `QDEDecision`
- Contain fields named `action`, `trade`, `buy`, `sell`, `enter`, `exit` (unless read-only descriptors of past state)
- Instruct a position size, stop loss placement, or lot size as a binding instruction — only as a recommendation

### 2. No imports of QDEDecision or QDEIntelligence

Capability modules must not import from `qde_intelligence.py`. They must not import `QDEDecision` from `cap_models.py`.

Permitted imports from `cap_models.py`:
- `MarketSnapshot`, `OpenTrade`, `AccountState`, `OHLCV`
- `CapabilityOutput` and any specific advisory output type (e.g., `MarketContext`, `EVScore`)
- Enumerations: `Direction`, `RegimeType`, `TrendStrength`, etc.
- Signal types: `SetupCandidate`, `MemoryPattern`, `TradeOutcome`, etc.

Prohibited:
- `QDEDecision`
- `EntrySignal` (this is part of QDEDecision, only QDEIntelligence constructs it)
- `ExitSignal`, `ScalingSignal`, `PartialCloseSignal`, `BreakEvenSignal`
- Any import from `qde_intelligence.py`

### 3. Pure data transformation

A capability receives a `MarketSnapshot` (and optionally other capability outputs) and returns a typed output. It must not:

- Write to external state (files, databases, APIs) during a live reasoning cycle
- Emit trade orders to any exchange or broker
- Call other capabilities directly (capabilities are wired by `QDEIntelligence`, not by each other)
- Modify the `MarketSnapshot` object

### 4. Confidence scoring

Every capability output extends `CapabilityOutput`, which requires a `confidence` field (0.0–1.0). This field represents how confident the capability is in its own output. It is used by `QDEIntelligence` to weight contributions to the final decision.

### 5. Warnings propagation

If a capability detects an abnormal condition (e.g., wide spread, illiquid market, missing data), it populates the `warnings: list[str]` field on `CapabilityOutput`. It does NOT refuse to return an output — it always returns, with reduced confidence and populated warnings.

### 6. Determinism (given the same inputs)

For a given `MarketSnapshot`, a capability must return the same output (no side-effectful randomness). This enables reproducibility for backtesting and validation.

---

## Capability Output Types Reference

| Capability | Output Type | Key Fields |
|---|---|---|
| Market Understanding | `MarketContext` | regime, trend_direction, volatility_state, structural_levels |
| Opportunity Discovery | `OpportunityReport` | candidates, best_candidate, opportunity_quality, market_is_tradeable |
| Expected Value Reasoning | `EVScore` | ev, win_probability, risk_reward_ratio, ev_threshold_met |
| Risk Reasoning | `RiskAssessment` | stop_loss_price, stop_distance_pips, risk_environment |
| Exposure Reasoning | `ExposureReport` | gross_exposure_pct, adding_exposure_is_safe, headroom |
| Trade Management | `TradeManagementSignals` | trades_needing_exit, reversal_detected, continuation_detected |
| Position Management | `PositionSizeRecommendation` | lot_size, risk_pct, kelly_fraction |
| Capital Preservation | `PreservationStatus` | circuit_breaker, new_entries_allowed, forced_exit_required |
| Market Memory | `MemoryContext` | matched_patterns, historical_win_rate_this_setup, anomaly_detected |
| Reflection | `ReflectionInsight` | reasoning_was_correct, capability_accuracy, key_lesson |
| Learning | `LearningUpdate` | updated_weights, capability_trust_scores, convergence_delta |

---

## What a Capability May NOT Do

The following are prohibited at the capability level:

| Prohibited Action | Reason |
|---|---|
| Construct `QDEDecision` | Only `QDEIntelligence` may do this |
| Construct `EntrySignal`, `ExitSignal`, `ScalingSignal`, `PartialCloseSignal`, `BreakEvenSignal` | These are decision components — only the QDE constructs them |
| Call `reason()` on another intelligence | There is only one intelligence |
| Modify `snapshot.open_trades` | Snapshots are read-only during a cycle |
| Communicate with a broker or exchange | The trainer is headless — all outputs are internal |
| Return `None` | Always return a typed capability output, even if empty/low-confidence |
| Raise unhandled exceptions | Catch internally and return with `confidence=0.0` and a warning |

---

## Adding a New Capability

To add a new capability:

1. Create `trainer/qde/capabilities/<name>.py`
2. Define an output dataclass extending `CapabilityOutput`
3. Define a capability class with a single primary method (e.g., `analyse()`, `evaluate()`, `compute()`)
4. Import the capability in `trainer/qde/capabilities/__init__.py`
5. Instantiate it in `QDEIntelligence.__init__()`
6. Wire its output into the appropriate reasoning stage in `QDEIntelligence.reason()`
7. Add its capability name and default weight to `QDEConfig.capability_weights`
8. Update `docs/qde/intelligence_architecture.md` with the new responsibility mapping

**The new capability must not produce a `QDEDecision`.** If the task requires producing a decision, it belongs in `qde_intelligence.py`, not in a capability module.

---

## Capability Call Graph

```
QDEIntelligence.reason(MarketSnapshot)
  │
  ├── CapitalPreservationCapability.evaluate(snapshot) → PreservationStatus
  │
  ├── MarketUnderstandingCapability.analyse(snapshot) → MarketContext
  ├── MarketMemoryCapability.recall(snapshot, None, context) → MemoryContext
  │
  ├── TradeManagementCapability.evaluate(snapshot, context) → TradeManagementSignals
  │
  ├── OpportunityDiscoveryCapability.discover(snapshot, context) → OpportunityReport
  ├── MarketMemoryCapability.recall(snapshot, candidate, context) → MemoryContext
  ├── RiskReasoningCapability.assess(snapshot, candidate, context) → RiskAssessment
  ├── ExpectedValueReasoningCapability.score(snapshot, candidate, risk, memory, context) → EVScore
  ├── ExposureReasoningCapability.analyse(snapshot, equity) → ExposureReport
  ├── PositionManagementCapability.compute(snapshot, risk, ev, context, equity) → PositionSizeRecommendation
  │
  └── [synthesis] → QDEDecision
```

Capabilities do not call each other. All wiring is inside `QDEIntelligence.reason()`.
Post-trade cycle:
```
QDEIntelligence.reflect_and_learn(TradeOutcome)
  │
  ├── ReflectionCapability.reflect(outcome) → ReflectionInsight
  ├── LearningCapability.learn(insight) → LearningUpdate
  └── MarketMemoryCapability.store(HistoricalOutcome)
```
