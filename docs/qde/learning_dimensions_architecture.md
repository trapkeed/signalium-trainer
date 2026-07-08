# Architecture: Legacy Engines → Learning Dimensions

**Date:** 2026-07-08
**Status:** Proposed target architecture (companion to `hardcoded_behavior_audit.md`)

## Purpose

This document defines the conversion of six legacy, rule-based "engines" into **evaluation dimensions** owned by the trainer. The trainer's job changes from *implementing* trading behavior to *evaluating* it: the QDE learns the behavior, and each dimension below becomes a scored axis the trainer uses to judge QDE outputs — never a rulebook the QDE is forced through.

This is a naming, ownership, and responsibility shift, not just a rename. Under the current architecture (see audit), each legacy engine independently makes a rule-based decision (e.g. `expected_drawdown = 0.6 * stop_dist + 0.4 * tail_risk`) and that decision is used directly in production. Under the target architecture, the QDE makes the decision, and the corresponding evaluation dimension only scores how good that decision was against labeled outcomes and benchmarks.

---

## 1. Conversion Map

| # | Legacy Engine | Current Files | New Learning Dimension | What Changes |
|---|---|---|---|---|
| 1 | **Risk Engine** | `trainer/execution/risk_engine.py`, `trainer/icari/portfolio_risk_engine.py` | **Risk Reasoning Evaluation** | Fixed VaR/CVaR/tail-risk formulas become an evaluation of the QDE's own risk estimate vs. realized risk |
| 2 | **Portfolio Engine** | `trainer/portfolio/exposure_engine.py`, `trainer/portfolio/position_allocator.py` | **Exposure Management Evaluation** | Hardcoded concentration limits (e.g. `ASSET_CLASS_LIMIT = 0.70`) become benchmark constraints the QDE's allocation is scored against, not gates it is forced to obey |
| 3 | **Trade Lifecycle Engine** | `trainer/lifecycle/trade_lifecycle.py`, `trainer/lifecycle/state_machine.py` | **Trade Management Evaluation** | The deterministic FSM's transition rules become labeled trajectories; the QDE proposes state transitions, evaluated against optimal/realized management paths |
| 4 | **Expected Value Engine** | `trainer/execution/risk_engine.py`, `trainer/icari/position_sizing_engine.py` | **Expected Value Evaluation** | The formulaic EV calc (`p_success * tp2_dist − p_fail * stop_dist`) becomes a calibration target: does the QDE's implied EV match realized P&L distribution |
| 5 | **Conviction Engine** | `trainer/lifecycle/lifecycle_confidence.py`, `trainer/research/confidence.py` | **Decision Confidence Evaluation** | Rule-adjusted confidence scores become a calibration evaluation (reliability diagrams, Brier score) of the QDE's own confidence output |
| 6 | **Context Engine** | `trainer/mie/market_regime_engine.py` | **Market Understanding Evaluation** | The 16-regime rule hierarchy (ADX/RSI/volatility thresholds) becomes labeled regime ground truth used to evaluate whether the QDE's learned market representation aligns with known regime outcomes |

---

## 2. Architectural Principle

```
BEFORE (rule-based engines drive production decisions)

  Market Data ──▶ Context Engine ──▶ Conviction Engine ──▶ Risk Engine ──▶ Portfolio Engine ──▶ Trade Lifecycle Engine ──▶ Executed Trade
                  (rules)             (rules)                (rules)         (rules)               (rules)


AFTER (QDE decides; dimensions only evaluate)

  Market Data ──▶ QDE (learned) ──▶ Decision (regime read, confidence, risk, exposure, EV, trade plan) ──▶ Executed Trade
                                          │
                                          ▼
                     ┌────────────────────────────────────────────────────────────┐
                     │                 Trainer Evaluation Layer                     │
                     │                                                              │
                     │  Market Understanding Evaluation   (was: Context Engine)     │
                     │  Decision Confidence Evaluation    (was: Conviction Engine)  │
                     │  Risk Reasoning Evaluation         (was: Risk Engine)        │
                     │  Exposure Management Evaluation    (was: Portfolio Engine)   │
                     │  Expected Value Evaluation         (was: EV Engine)          │
                     │  Trade Management Evaluation       (was: Trade Lifecycle)    │
                     │                                                              │
                     │  → scores, calibration metrics, benchmark comparisons        │
                     │  → feeds training objectives + promotion evaluation          │
                     └────────────────────────────────────────────────────────────┘
```

The key inversion: today the arrows into "Executed Trade" pass **through** each rule engine sequentially, so each engine's hardcoded logic is load-bearing. In the target state, the QDE alone produces the decision; every former engine sits **beside** the decision path as a read-only evaluator, scoring the decision after the fact (or in shadow/backtest mode before it goes live).

---

## 3. Per-Dimension Specification

### 3.1 Risk Reasoning Evaluation *(was: Risk Engine)*
- **Legacy behavior removed:** fixed weighting formulas for expected drawdown, VaR/CVaR multipliers, tail-risk blending.
- **What it evaluates:** whether the QDE's implied risk (position size, stop distance, drawdown exposure) is consistent with realized volatility and realized drawdown outcomes.
- **Metrics:** VaR backtesting (Kupiec/Christoffersen tests), realized-vs-predicted drawdown error, tail-risk calibration.
- **Benchmark dataset:** historical trade outcomes bucketed by volatility regime, with realized max adverse excursion as ground truth.

### 3.2 Exposure Management Evaluation *(was: Portfolio Engine)*
- **Legacy behavior removed:** hardcoded per-asset-class/correlation exposure caps used as hard gates.
- **What it evaluates:** whether the QDE's portfolio-level allocation stays within healthy diversification and concentration bounds *because it learned to*, not because it was capped.
- **Metrics:** concentration ratio drift vs. benchmark, correlation-adjusted exposure vs. realized co-movement, diversification ratio over time.
- **Benchmark dataset:** historical multi-asset exposure snapshots labeled with subsequent portfolio drawdown/volatility outcomes.

### 3.3 Trade Management Evaluation *(was: Trade Lifecycle Engine)*
- **Legacy behavior removed:** deterministic FSM with fixed confidence gates per transition (e.g. 0.40 / 0.55 / 0.65).
- **What it evaluates:** whether the QDE's chosen sequence of trade states (idea → watchlist → entry → scale → exit) matches or outperforms optimal-in-hindsight management of the same trade.
- **Metrics:** trajectory divergence from optimal management path, timing quality of entries/exits/scale actions, missed-opportunity cost.
- **Benchmark dataset:** labeled "gold path" trade lifecycles reconstructed from historical price action (optimal entry/exit/scale points known after the fact).

### 3.4 Expected Value Evaluation *(was: Expected Value Engine)*
- **Legacy behavior removed:** closed-form EV formula from fixed hit-probabilities and fixed R:R targets.
- **What it evaluates:** whether the QDE's implied EV for a trade is well-calibrated against realized P&L, i.e. trades the model rates high-EV should actually earn more on average.
- **Metrics:** EV calibration curve, realized-vs-predicted EV correlation, profit-factor consistency across EV buckets.
- **Benchmark dataset:** historical trades bucketed by model-implied EV vs. realized outcome.

### 3.5 Decision Confidence Evaluation *(was: Conviction Engine)*
- **Legacy behavior removed:** rule-adjusted confidence scores (base empirical probability + manual adjustments like "increase fail_prob if news_risk > 0.50").
- **What it evaluates:** whether the QDE's own stated confidence is trustworthy — a 70%-confidence decision should win roughly 70% of the time.
- **Metrics:** Brier score, reliability diagrams/calibration curves, overconfidence/underconfidence bias by regime.
- **Benchmark dataset:** historical decisions with model confidence at time of decision vs. actual outcome.

### 3.6 Market Understanding Evaluation *(was: Context Engine)*
- **Legacy behavior removed:** rule hierarchy classifying market into 16 fixed regimes via ADX/RSI/volatility thresholds.
- **What it evaluates:** whether the QDE's learned internal representation of "what kind of market this is" aligns with known regime outcomes and produces regime-appropriate behavior, without needing an explicit rule-based classifier in production.
- **Metrics:** regime-conditional performance breakdown, agreement with human/rule-labeled regimes (used only as a validation label, not as production logic), representation probing (does the learned embedding separate known regimes).
- **Benchmark dataset:** historical windows labeled with regime type (trending/ranging/volatile/etc.), used purely as evaluation ground truth.

---

## 4. Where This Lives in the Trainer

| Layer | Role |
|---|---|
| **Training labels** | Ground-truth regime tags, optimal trade-management trajectories, realized EV/risk outcomes — derived from historical data, not from the legacy engines' rule output |
| **Training objectives** | Loss functions that push the QDE toward calibrated confidence, well-timed trade management, and risk-aware sizing |
| **Evaluation metrics** | The six dimensions above, each with defined metrics (calibration, trajectory divergence, VaR backtests, etc.) |
| **Benchmark datasets** | Fixed historical datasets per dimension, versioned, used consistently across QDE versions for fair comparison |
| **Research experiments** | Ablations/comparisons (e.g. "does the QDE's confidence calibrate better with feature X") that inform future label/objective design |
| **QDE promotion (`trainer/qde/`)** | Consumes the six evaluation dimensions as inputs to the promotion decision (replacing today's hardcoded floors in `qde_models.py` — see the companion audit) |

## 5. Migration Path

1. **Instrument, don't remove first:** keep each legacy engine running in shadow mode, but stop treating its output as ground truth — start logging QDE decisions alongside it.
2. **Build the benchmark datasets** for each dimension from historical data (regime labels, optimal trajectories, realized EV/risk), independent of the legacy engines' internal formulas.
3. **Wire evaluation metrics** for each dimension into the QDE pipeline (`trainer/qde/qde_pipeline.py`) as scored checks, replacing the current calls into `RiskEngine`, `ExposureEngine`, `TradeLifecycle`/`StateMachine`, `LifecycleConfidenceEngine`, and `MarketRegimeEngine` for production decisioning.
4. **Flip promotion gating** in `trainer/qde/qde_models.py` from fixed floors to dimension-score thresholds validated against the benchmark datasets.
5. **Decommission the rule engines from the production decision path** once the QDE's scores on all six dimensions meet or exceed the legacy engines' historical performance on the same benchmarks.

---

## 6. Non-Goals

- This document does not change any production code paths yet — it defines the target architecture only.
- The legacy engines are not deleted; their formulas become the initial baseline the QDE must beat on each evaluation dimension before cutover.
