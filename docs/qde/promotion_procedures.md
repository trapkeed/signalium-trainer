# QDE Promotion Procedures

## Overview

A QDE version may only be promoted to ACTIVE status if it demonstrably improves across **all five promotion dimensions** relative to the incumbent champion. This rule is enforced programmatically — no version can bypass the evaluation gate.

---

## The Five Promotion Dimensions

### 1. Risk Adjusted Return

**Metric:** `max(sharpe_ratio, sortino_ratio)`

The challenger must deliver a higher risk-adjusted return than the incumbent. Both Sharpe and Sortino ratios are considered; the better of the two is used.

| Check | Rule |
|---|---|
| Relative | `challenger_RAR ≥ incumbent_RAR + min_risk_adjusted_return_delta` |
| Absolute floor | `challenger_RAR ≥ min_sharpe` (default 0.5) |

### 2. Drawdown

**Metric:** `max_drawdown` (lower is better)

The challenger must reduce maximum drawdown compared to the incumbent, subject to an absolute ceiling.

| Check | Rule |
|---|---|
| Relative (if `drawdown_must_improve=True`) | `incumbent_drawdown − challenger_drawdown ≥ 0` |
| Absolute ceiling | `challenger_drawdown ≤ max_drawdown_abs` (default 30%) |

### 3. Consistency

**Metric:** Composite of `win_rate` and `profit_factor`

Consistency measures reliable performance across different market conditions and walk-forward folds.

| Check | Rule |
|---|---|
| Relative | `challenger_consistency_score ≥ incumbent_score + min_consistency_delta` |
| Absolute: win rate | `win_rate ≥ min_win_rate` (default 45%) |
| Absolute: profit factor | `profit_factor ≥ min_profit_factor` (default 1.1) |

### 4. Capital Efficiency

**Metric:** `capital_efficiency` (return per unit of margin used)

A version must use capital more efficiently than its predecessor.

| Check | Rule |
|---|---|
| Relative | `challenger_CE ≥ incumbent_CE + min_capital_efficiency_delta` |
| No incumbent | `challenger_CE ≥ 0.0` |

### 5. Decision Quality

**Metric:** `decision_quality` (entry/exit prediction precision)

The quality of individual trade decisions must improve or match the incumbent.

| Check | Rule |
|---|---|
| Relative | `challenger_DQ ≥ incumbent_DQ + min_decision_quality_delta` |
| No incumbent | `challenger_DQ > 0.0` |

---

## Default Promotion Criteria

```python
QDEPromotionCriteria(
    min_risk_adjusted_return_delta = 0.0,   # Must not regress
    max_drawdown_allowed            = 0.25,  # 25% max drawdown threshold
    drawdown_must_improve           = True,  # Drawdown must decrease
    min_consistency_delta           = 0.0,   # Must not regress
    min_capital_efficiency_delta    = 0.0,   # Must not regress
    min_decision_quality_delta      = 0.0,   # Must not regress

    # Absolute floors (always enforced, regardless of incumbent)
    min_sharpe          = 0.5,
    min_profit_factor   = 1.1,
    max_drawdown_abs    = 0.30,   # Hard ceiling: 30%
    min_win_rate        = 0.45,
)
```

---

## Promotion Workflow

### Step 1 — Pipeline Completion

The version must be in `DEPLOYMENT_CANDIDATE` stage. Initiate the pipeline and advance through all stages:

```
POST /api/qde/versions/:id/pipeline/start
POST /api/qde/pipeline/:runId/advance  (×6 for all stages)
```

### Step 2 — A/B Testing (Recommended)

Run at minimum a shadow test before evaluating promotion:

```
POST /api/qde/ab/shadow
{ "challenger_version_id": "qde-v3-a4f2c1d0" }
```

Record results and conclude the test before proceeding:

```
POST /api/qde/ab/:testId/results
POST /api/qde/ab/:testId/conclude
```

### Step 3 — Promotion Evaluation

Evaluate the challenger against the incumbent across all five dimensions:

```
POST /api/qde/versions/:id/promotion/evaluate
{ "notes": "Post-shadow evaluation — v3 candidate" }
```

The response contains a `QDEPromotionDecision` with per-dimension results:

```json
{
  "is_eligible": true,
  "passed_dimensions": ["Risk Adjusted Return", "Drawdown", "Consistency", "Capital Efficiency", "Decision Quality"],
  "failed_dimensions": [],
  "dimension_results": [...]
}
```

### Step 4A — Manual Approval (Default)

By default, human sign-off is required even when all dimensions pass:

```
POST /api/qde/versions/:id/promotion/approve
{ "approver": "quant-lead", "notes": "Reviewed shadow metrics — approved" }
```

### Step 4B — Automatic Promotion (CI Pipeline)

Set `QDE_AUTO_PROMOTE=true` to enable automatic promotion when all dimensions pass:

```
POST /api/qde/versions/:id/promotion/auto
```

---

## What Happens on Promotion

1. The challenger version's stage is updated to `ACTIVE`.
2. The previous `ACTIVE` version is automatically moved to `RETIRED`.
3. The `deployed_at` timestamp is recorded on the new active version.
4. The promotion decision is persisted in the promotion history.

---

## Promotion History

Every evaluation — successful or failed — is recorded:

```
GET /api/qde/promotion/history
GET /api/qde/promotion/criteria
```

---

## Handling a Rejected Promotion

If a promotion evaluation fails:

1. **Review the failed dimensions** — each `DimensionResult` shows the challenger value, incumbent value, delta, and threshold.
2. **Do not skip training** — address the root cause in model architecture, features, or training data.
3. **Re-register and re-run** — register a new version, run the full pipeline, and re-evaluate.
4. **Consider loosening criteria only with justification** — changes to promotion thresholds require documented rationale.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/qde/versions/:id/promotion/evaluate` | `POST` | Evaluate promotion eligibility |
| `/api/qde/versions/:id/promotion/approve` | `POST` | Manually approve promotion |
| `/api/qde/versions/:id/promotion/auto` | `POST` | Evaluate and auto-promote |
| `/api/qde/promotion/history` | `GET` | All past promotion decisions |
| `/api/qde/promotion/criteria` | `GET` | Current promotion criteria |
