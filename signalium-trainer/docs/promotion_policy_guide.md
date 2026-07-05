# Promotion Policy Guide

## Phase 15 — Signalium Quant

---

## Overview

The Promotion Policy is the gatekeeper between a validated model and production deployment. Every model that passes validation is evaluated against the configured thresholds. Only models passing **all required rules** are eligible for promotion.

**No model may enter production without passing the Promotion Policy.**

---

## Promotion Flow

```
Model Validated
      ↓
PromotionPolicy.evaluate()
      ↓
┌─────────────────────────────────────────────────────────┐
│  Check all rules:                                        │
│  - min_accuracy, min_f1, min_precision, min_recall       │
│  - min_sharpe, min_sortino                               │
│  - max_drawdown                                          │
│  - min_profit_factor                                     │
│  - max_calibration_error (ECE)                          │
│  - min_trade_count                                       │
│  - min_win_rate                                         │
│  - min_validation_folds                                  │
│  - custom rules                                          │
└─────────────────────────────────────────────────────────┘
      ↓                          ↓
  All pass                  Any fail
      ↓                          ↓
is_eligible=True           is_eligible=False
      ↓                     NOT promotable
require_manual_approval?
      ↓               ↓
    False           True
      ↓               ↓
Auto-promoted     Human must call
is_promoted=True  policy.approve()
```

---

## Configuring the Policy

### Example: Conservative Institutional Policy

```python
from trainer.validation import PromotionPolicy

policy = PromotionPolicy(
    # Statistical requirements
    min_accuracy=0.55,
    min_precision=0.53,
    min_recall=0.53,
    min_f1=0.54,

    # Calibration requirements
    max_calibration_error=0.08,    # ECE ≤ 8%
    max_ece=0.08,                  # alias

    # Trading requirements
    min_profit_factor=1.50,
    min_sharpe=1.00,
    min_sortino=1.20,
    max_drawdown=0.20,             # 20% max drawdown
    min_win_rate=0.50,

    # Operational requirements
    min_trade_count=100,           # minimum trades evaluated
    min_confidence_quality=0.60,   # 60%+ high-confidence predictions
    min_validation_folds=5,        # at least 5 walk-forward folds

    # Always require human approval
    require_manual_approval=True,
)
```

### Example: Permissive Development Policy

```python
policy = PromotionPolicy(
    min_accuracy=0.45,
    max_drawdown=0.40,
    require_manual_approval=False,  # auto-promote
)
```

---

## Evaluating a Model

```python
metrics = {
    "accuracy": 0.612,
    "f1_macro": 0.594,
    "precision_macro": 0.601,
    "recall_macro": 0.587,
    "sharpe_ratio": 1.45,
    "sortino_ratio": 1.82,
    "max_drawdown": 0.12,
    "profit_factor": 1.72,
    "win_rate": 0.58,
    "ece": 0.04,
}

decision = policy.evaluate(
    model_id="quant_v1",
    metrics=metrics,
    n_trades=245,
    n_folds=12,
    notes="Candidate for Q3 deployment",
)

print(decision.summary())
```

Output:
```
=== Promotion Decision: quant_v1 ===
  Status        : ELIGIBLE
  Promotion     : NOT PROMOTED
  Decided At    : 2026-07-05T10:00:00+00:00

  Passed Rules  : 10/10
    [PASS] Minimum Accuracy = 0.6120
    [PASS] Minimum F1 = 0.5940
    [PASS] Minimum Sharpe = 1.4500
    [PASS] Maximum Drawdown = 0.1200
    [PASS] Maximum ECE = 0.0400
    ...

  Notes         : Candidate for Q3 deployment
```

---

## Manual Approval

When `require_manual_approval=True`, a senior researcher must call `.approve()`:

```python
if decision.is_eligible:
    approved_decision = policy.approve(
        decision,
        approver="Dr. Jane Smith",
    )
    # is_promoted is now True
    print(f"Approved: {approved_decision.is_promoted}")
```

### Approval Audit Trail

All approvals are recorded:
```python
decision.notes  # "Candidate for Q3 deployment | Approved by: Dr. Jane Smith"
```

---

## Custom Rules

Add arbitrary rules beyond the built-in thresholds:

```python
from trainer.validation import PromotionPolicy, PromotionRule

policy = PromotionPolicy(min_accuracy=0.55)

# Add a custom Ulcer Index rule
policy.add_rule(PromotionRule(
    metric="ulcer_index",
    threshold=0.15,
    is_max=True,              # lower is better
    label="Maximum Ulcer Index",
    critical=True,            # failure blocks promotion
))

# Add a non-critical advisory rule (warning only)
policy.add_rule(PromotionRule(
    metric="calmar_ratio",
    threshold=1.0,
    is_max=False,
    label="Advisory: Calmar Ratio",
    critical=False,           # failure generates warning, not block
))
```

---

## Promotion to Champion Status

Once a model is approved, update the leaderboard:

```python
# Via API
resp = engine.api.approve_promotion(run_id, approver="Dr. Smith")

# Via leaderboard directly
lb.update(model_id, {
    "promotion_status": "champion",
    "is_champion": True,
})
```

Only one model per model family may hold Champion status. Approving a new Champion automatically retires the previous one.

---

## Policy Introspection

```python
# List all configured rules
for rule in policy.describe():
    print(f"{rule['label']}: {'>=' if not rule['is_max'] else '<='} {rule['threshold']}")

# Count rules
print(f"Total rules: {policy.n_rules}")

# Check if manual approval required
print(f"Manual approval: {policy.requires_manual_approval}")
```

---

## Promotion Status Reference

| Status | Meaning | Can Be Promoted? |
|--------|---------|-----------------|
| experimental | Freshly validated | No |
| candidate | Passed minimum thresholds | Requires champion comparison |
| challenger | Compared favourably to champion | Requires manual approval |
| champion | Current production model | N/A |
| retired | Decommissioned | No |
