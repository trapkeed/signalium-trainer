# Research Lab — Research Flow

## Overview

The Research Lab is an administrator-only internal subsystem of the Signalium Trainer. Its sole purpose is the continuous improvement of the Quant Decision Engine (QDE). It has no frontend — all capabilities are exposed through internal APIs only.

No retail user may access the Research Lab. Only the Signalium admin frontend may consume these APIs, and every request must carry a valid admin token.

---

## Access Control

All Research Lab endpoints are prefixed `/api/admin/research/*` and require:

```
Authorization: Bearer <RESEARCH_LAB_ADMIN_TOKEN>
```

or

```
X-Admin-Token: <RESEARCH_LAB_ADMIN_TOKEN>
```

Requests without a valid token receive `401 Unauthorized` or `403 Forbidden`. If `RESEARCH_LAB_ADMIN_TOKEN` is not configured in the environment, all requests receive `503 Service Unavailable` (fail-closed, never fail-open).

---

## Research Lab Responsibilities

| Capability | Description |
|---|---|
| Dataset Creation | Register and version training datasets |
| Feature Discovery | Track, rank, and promote discovered features |
| Experiment Tracking | Full lifecycle record for every training run |
| Model Benchmarking | Compare multiple QDE versions on standard metrics |
| Backtesting | Register historical simulation results per experiment |
| A/B Evaluation | Shadow / Canary / Side-by-side comparison (via QDE AB engine) |
| Parameter Search | Hyperparameter search across Grid, Random, Bayesian, Evolutionary strategies |
| Retraining Validation | Gate-controlled validation before any new QDE is deployed |
| Deployment Approval | Signed off approval by an administrator after gate evaluation |
| Failure Analysis | Record, investigate, and resolve QDE failure events |
| Market Regime Analysis | Per-regime performance breakdown across market conditions |

---

## End-to-End Research Cycle

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATASET                                                 │
│     Register a versioned dataset with instruments,         │
│     date range, row count, and label quality score.        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  2. FEATURE DISCOVERY                                       │
│     Record candidate features with importance scores.      │
│     Accept high-value features; reject low-value ones.     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  3. PARAMETER SEARCH                                        │
│     Run a hyperparameter search (Grid/Random/Bayesian).    │
│     Record trial results. Complete the search.             │
│     Best parameters feed into the next experiment.         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  4. EXPERIMENT                                              │
│     Create an experiment with QDE version, dataset,        │
│     hyperparameters, and researcher attribution.           │
│     Start → Complete/Fail with full metrics.               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  5. BACKTESTING                                             │
│     Register backtest results against the experiment.      │
│     Metrics flow back into the experiment record.          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  6. BENCHMARKING & COMPARISON                               │
│     Compare this version against QDE v11, v12, v13, ...    │
│     on all 8 standard metrics. Identify the winner.        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  7. REGIME ANALYSIS                                         │
│     Evaluate per-regime performance: trending, ranging,    │
│     high-volatility, crisis, recovery regimes.             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  8. DEPLOYMENT GATES                                        │
│     Run the three mandatory gates (Decision Quality,       │
│     Risk Adjusted Return, Drawdown Stability).             │
│     Admin approves or rejects deployment.                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  9. QDE LIFECYCLE                                           │
│     Register the approved version in the QDE registry.     │
│     Run the QDE pipeline. Promote to ACTIVE.               │
└─────────────────────────────────────────────────────────────┘
```

---

## API Base

Research Lab server: `http://localhost:8002`
Express proxy prefix: `/api/admin/research`

Start the server:
```bash
RESEARCH_LAB_ADMIN_TOKEN=<secret> uvicorn trainer.research_lab.research_lab_server:app --port 8002
```
