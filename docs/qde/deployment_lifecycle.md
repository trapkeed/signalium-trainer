# QDE Deployment Lifecycle

## Overview

Deployment of a new QDE version is a graduated, multi-stage process designed to minimise risk to live capital. A version must successfully complete the full **Training Pipeline** before it is eligible for deployment consideration.

---

## Training Pipeline

The pipeline is a sequential, gate-controlled process. Each stage must pass before the next begins.

```
Historical Data
      │
      │  Load and validate training dataset.
      │  Check for data freshness, completeness, and label quality.
      ▼
Training
      │
      │  Train model on approved historical data.
      │  Record training metrics (loss curves, convergence).
      ▼
Validation
      │
      │  Walk-forward cross-validation across multiple market regimes.
      │  Evaluate accuracy, precision, recall, calibration (ECE).
      ▼
Backtesting
      │
      │  Full out-of-sample backtest on held-out data.
      │  Compute Sharpe, Sortino, drawdown, profit factor, win rate.
      ▼
Shadow Testing
      │
      │  Deploy in shadow mode alongside the incumbent ACTIVE version.
      │  Log all decisions without executing them in live markets.
      │  Compare decision distribution against incumbent.
      ▼
Deployment Candidate
      │
      │  Pipeline complete. Version is now eligible for promotion.
      ▼
   (Promotion Evaluation)
```

### Pipeline Stage Gate Rules

- A stage may be **advanced** (PASSED), **failed** (FAILED), or **skipped** (SKIPPED, e.g. to reuse cached artifacts).
- Failure at any stage moves the version to `FAILED` and halts the pipeline.
- A `FAILED` version may be re-registered and restarted from scratch.
- All stage metrics are recorded on the version and used in promotion evaluation.

---

## A/B Deployment Strategies

Once a version is a Deployment Candidate, three deployment strategies are available before full promotion.

### 1. Shadow Deployment

The challenger runs in parallel alongside the champion. Its decisions are **logged but never executed**. This is the safest validation strategy.

```
Live Market Signal
       │
       ├──► Control (ACTIVE)   ──► Execute decision
       │
       └──► Challenger (SHADOW) ──► Log decision only
```

**Use when:** You need confidence in the decision distribution before any live exposure.

### 2. Canary Deployment

A configurable fraction of live decisions (`traffic_fraction`, default 5%) is routed to the challenger. The remainder goes to the champion.

```
Live Market Signal
       │
       ├──[95%]──► Control (ACTIVE)     ──► Execute
       │
       └──[ 5%]──► Challenger (CANARY)  ──► Execute
```

**Use when:** Shadow testing passed and you want limited live exposure with real P&L feedback.

**Constraint:** `traffic_fraction` must be in the range (0%, 50%].

### 3. Side-by-Side Evaluation

Both versions receive identical inputs simultaneously. Neither version's output affects live execution — this is a purely analytical comparison on live data.

```
Live Market Signal
       │
       ├──► Control     ──► Record metrics (not executed)
       │
       └──► Challenger  ──► Record metrics (not executed)
```

**Use when:** You want a rigorous statistical comparison under identical market conditions.

---

## Promotion to ACTIVE

After A/B testing, a version may be promoted to ACTIVE by passing the five-dimension promotion evaluation. See [Promotion Procedures](promotion_procedures.md) for full details.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/qde/versions/:id/pipeline/start` | `POST` | Start the training pipeline |
| `/api/qde/pipeline/:runId/advance` | `POST` | Advance to next stage |
| `/api/qde/pipeline/:runId/fail` | `POST` | Fail the current stage |
| `/api/qde/pipeline/:runId/skip` | `POST` | Skip the current stage |
| `/api/qde/pipeline/:runId` | `GET` | Get pipeline run status |
| `/api/qde/pipeline` | `GET` | List all pipeline runs |
| `/api/qde/ab/shadow` | `POST` | Create shadow test |
| `/api/qde/ab/canary` | `POST` | Create canary test |
| `/api/qde/ab/side-by-side` | `POST` | Create side-by-side test |
| `/api/qde/ab/:testId/results` | `POST` | Record test metrics |
| `/api/qde/ab/:testId/conclude` | `POST` | Conclude test with winner |
| `/api/qde/ab/:testId/compare` | `GET` | Statistical comparison |
| `/api/qde/ab` | `GET` | List all A/B tests |
