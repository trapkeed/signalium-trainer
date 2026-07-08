# Research Lab — Validation Flow

## Overview

Validation confirms that a new QDE version is genuinely better than the incumbent before it is approved for deployment. The Research Lab enforces a multi-layer validation stack: experiment-level backtesting, walk-forward cross-validation, regime analysis, and the three mandatory deployment gates.

---

## Validation Stack

```
Layer 1: Backtesting
   Historical simulation on held-out data.
   Produces: Sharpe, Sortino, drawdown, profit factor, win rate, EV.

Layer 2: Walk-Forward Validation
   Multi-fold time-series validation across market regimes.
   Produces: Consistency score, fold-level metrics variance.

Layer 3: Regime Analysis
   Per-regime performance breakdown.
   Produces: Best/worst regime, overall regime stability score.

Layer 4: Deployment Gate Evaluation
   Three mandatory gates (Decision Quality, RAR, Drawdown).
   Produces: Pass/fail per gate, overall APPROVED or REJECTED.

Layer 5: Retraining Validation
   Explicit comparison of challenger vs. incumbent experiment metrics.
   Confirms that retraining delivered genuine improvement.
```

---

## Backtesting

Register a backtest result against an experiment. Key metrics are automatically merged back into the experiment record.

```
POST /api/admin/research/backtests
Authorization: Bearer <token>

{
  "experiment_id": "exp-9b8f45ad",
  "dataset_id": "ds-a4f2c1d0",
  "metrics": {
    "sharpe_ratio": 1.74,
    "max_drawdown": 0.087,
    "profit_factor": 2.1,
    "win_rate": 0.593,
    "expected_value": 0.0031,
    "decision_quality": 0.681
  },
  "notes": "Out-of-sample backtest 2024-01-01 to 2026-06-30"
}
```

---

## Regime Analysis

Validate performance across all market regimes before approving deployment.

```
POST /api/admin/research/regime
Authorization: Bearer <token>

{
  "qde_version": "QDE v13",
  "regime_results": {
    "TRENDING_BULL":    { "sharpe_ratio": 2.1, "max_drawdown": 0.06, "win_rate": 0.64 },
    "TRENDING_BEAR":    { "sharpe_ratio": 1.4, "max_drawdown": 0.11, "win_rate": 0.54 },
    "RANGING":          { "sharpe_ratio": 0.9, "max_drawdown": 0.09, "win_rate": 0.49 },
    "HIGH_VOLATILITY":  { "sharpe_ratio": 0.6, "max_drawdown": 0.15, "win_rate": 0.44 },
    "CRISIS":           { "sharpe_ratio": 0.3, "max_drawdown": 0.22, "win_rate": 0.38 },
    "RECOVERY":         { "sharpe_ratio": 1.8, "max_drawdown": 0.08, "win_rate": 0.59 }
  },
  "notes": "Q3 2026 regime validation"
}
```

The response includes `best_regime`, `worst_regime`, and an `overall_regime_stability` score (0–1). A stability score below 0.5 indicates the model behaves very differently across regimes and may need regime-specific tuning.

---

## Retraining Validation

Validate that a retrained version genuinely improves on the current incumbent:

```
POST /api/admin/research/experiments/:experimentId/validate-retraining
Authorization: Bearer <token>

{
  "baseline_experiment_id": "exp-b3e1a2c9"
}
```

This runs all three deployment gates comparing the challenger experiment against the baseline. An `APPROVED` result is required before the deployment approval step.

---

## Walk-Forward Consistency Check

The consistency dimension (tracked in experiments) is evaluated through `profit_factor` and `win_rate` stability across folds. When recording experiment metrics, include fold-level statistics:

```
{
  "win_rate": 0.593,
  "win_rate_std": 0.034,       // Low std = consistent across folds
  "profit_factor": 2.1,
  "profit_factor_std": 0.18,
  "sharpe_ratio": 1.74,
  "n_validation_folds": 12
}
```

---

## Validation Checklist

Before approving any deployment, verify:

| Checkpoint | How to Verify |
|---|---|
| Backtest registered | `GET /admin/research/backtests?experiment_id=...` |
| Regime analysis complete | `GET /admin/research/regime?qde_version=...` |
| Retraining validation passed | `POST /admin/research/experiments/:id/validate-retraining` |
| Model comparison reviewed | `GET /admin/research/comparisons` |
| Deployment gates passed | `POST /admin/research/experiments/:id/gates` → `passed: true` |
| No open CRITICAL failures | `GET /admin/research/failures/summary` → `open_critical: []` |

All six checkpoints must be satisfied before an administrator signs off on deployment.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/admin/research/backtests` | `POST` | Register a backtest result |
| `/api/admin/research/backtests` | `GET` | List backtests (filter by experiment) |
| `/api/admin/research/regime` | `POST` | Run regime analysis |
| `/api/admin/research/regime` | `GET` | List regime reports |
| `/api/admin/research/regime/:id` | `GET` | Get specific regime report |
| `/api/admin/research/experiments/:id/gates` | `POST` | Evaluate deployment gates |
| `/api/admin/research/experiments/:id/validate-retraining` | `POST` | Retraining comparison |
| `/api/admin/research/comparisons` | `POST` | Compare multiple QDE versions |
| `/api/admin/research/failures/summary` | `GET` | Check open critical failures |
