# Research Guide

## Phase 15 — Signalium Quant

---

## Overview

This guide explains how quantitative researchers use the Institutional Validation Platform in their day-to-day model evaluation workflow.

---

## Standard Research Workflow

### Step 1: Train a Model

Training is handled separately from validation. Phase 15 is evaluation-only.

```python
# Assuming model is already trained
model = torch.load("experiments/quant_v1_checkpoint.pt")
model.eval()
```

### Step 2: Configure the Policy

Set institutional thresholds before running validation:

```python
from trainer.validation import PromotionPolicy

policy = PromotionPolicy(
    min_accuracy=0.55,
    min_sharpe=1.0,
    max_drawdown=0.20,
    min_profit_factor=1.5,
    max_ece=0.08,
    min_trade_count=100,
    require_manual_approval=True,
)
```

### Step 3: Run Validation

```python
from trainer.validation import InstitutionalValidationEngine, EngineConfig

engine = InstitutionalValidationEngine(
    db_path="experiments/validation.db",
    leaderboard_path="experiments/leaderboard.json",
    policy=policy,
    export_dir="reports/2026-Q3/",
    config=EngineConfig(
        strategy="expanding",
        min_train_size=2000,
        val_size=500,
        step_size=100,
        n_mc_simulations=1000,
    ),
)

result = engine.validate(
    model=model,
    x=x_data,               # (N, seq_len, n_features)
    y=labels,               # (N,)
    model_id="quant_v1",
    model_version="2.0.0",
    extra_metadata={
        "training_date": "2026-07-01",
        "dataset_source": "historical_v4",
    },
)

print(result.summary())
```

### Step 4: Review Results

```python
# Statistical performance
print(f"Accuracy: {result.statistical.accuracy:.4f}")
print(f"F1: {result.statistical.f1_macro:.4f}")
print(f"ECE: {result.statistical.ece:.4f}")

# Trading performance
print(f"Sharpe: {result.trading.sharpe_ratio:.3f}")
print(f"Profit Factor: {result.trading.profit_factor:.3f}")
print(f"Max Drawdown: {result.trading.max_drawdown:.3f}")

# Promotion eligibility
print(f"Eligible: {result.promotion_decision.is_eligible}")
print(f"Failed rules: {result.promotion_decision.failed_rules}")
```

### Step 5: Approve Promotion (if eligible)

```python
if result.promotion_decision.is_eligible:
    approved = engine.api.approve_promotion(
        run_id=result.run_id,
        approver="Dr. Jane Smith",
    )
    print(f"Approved: {approved['data']['approved']}")
```

### Step 6: Update Leaderboard

```python
# View current leaderboard
print(engine.leaderboard.format_table())

# Compare with current champion
champion = engine.leaderboard.champion()
if champion:
    comparison = engine.leaderboard.compare(champion.model_id, "quant_v1")
    print(comparison)
```

---

## Multi-Asset Validation

Run validation across all 14 canonical instruments:

```python
from trainer.validation import MultiAssetValidator, ExpandingWindowValidator

strategy = ExpandingWindowValidator(min_train_size=500, val_size=100, step_size=50)

validator = MultiAssetValidator(
    instruments=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"],
    timeframes=["1H", "4H", "1D"],
    strategy=strategy,
)

dataset_map = {
    ("EURUSD", "1H"): (x_eurusd_1h, y_eurusd_1h),
    ("GBPUSD", "1H"): (x_gbpusd_1h, y_gbpusd_1h),
    # ...
}

multi_result = validator.validate(model, dataset_map)
print(f"Worst instrument: {multi_result.worst_instrument}")
print(f"Worst timeframe: {multi_result.worst_timeframe}")
```

---

## Regime Analysis

Understand where the model struggles by regime:

```python
from trainer.validation import MarketRegimeClassifier, RegimeDistribution

clf = MarketRegimeClassifier()
regimes = clf.classify(ohlcv_tensor, timestamps=timestamps)
dist = RegimeDistribution.from_labels(regimes)
print(dist.summary())

# Compute accuracy per regime
regime_performance = {}
for regime in set(r.value for r in regimes):
    mask = [r.value == regime for r in regimes]
    regime_probs = probs[mask]
    regime_labels = y[mask]
    if len(regime_labels) > 10:
        preds = regime_probs.argmax(dim=-1)
        acc = float((preds == regime_labels).float().mean())
        regime_performance[regime] = {"accuracy": acc, "n_samples": int(mask.count(True))}
```

---

## Drift Monitoring

Monitor for distribution shift between validation windows:

```python
from trainer.validation import DriftAnalyzer

analyzer = DriftAnalyzer(psi_threshold=0.20)

# Compare training vs validation distributions
report = analyzer.detect(
    x_ref=x_train,
    x_current=x_val,
    y_ref=y_train,
    y_current=y_val,
    probs_ref=probs_train,
    probs_current=probs_val,
    acc_ref=baseline_accuracy,
    acc_current=current_accuracy,
    feature_names=feature_names,
)

if report.drift_severity in ("moderate", "severe"):
    print("WARNING: Significant drift detected — consider retraining")
    print(report.summary())
```

---

## Failure Analysis

Identify specific weaknesses for targeted improvement:

```python
from trainer.validation import FailureAnalyzer

analyzer = FailureAnalyzer(top_k=5)
failure_report = analyzer.analyze(
    instrument_results=multi_asset_result,
    regime_performance=regime_performance,
    head_results=head_results,
    session_performance=session_perf,
)

print(failure_report.summary())
for rec in failure_report.recommendations:
    print(f"→ {rec}")
```

---

## Database Queries

Query historical validation runs:

```python
from trainer.validation import ValidationDatabase

db = ValidationDatabase("experiments/validation.db")

# All runs for a model
runs = db.list_runs(model_id="quant_v1", limit=50)

# Best run ever
best = db.get_best_run("quant_v1")

# Most recent run
latest = db.get_latest_run("quant_v1")

# Filter by status
completed = db.list_runs(status="completed")

# Count total runs
print(f"Total validation runs: {db.count_runs()}")

db.close()
```

---

## Exporting Reports

```python
from trainer.validation import ReportExporter, ExportConfig

exporter = ReportExporter(
    output_dir=Path("reports/quant_v1_v2.0/"),
    config=ExportConfig(
        formats=["json", "csv", "html"],
        title="Quant V1 — Release Candidate 2.0",
        include_charts=True,
    ),
)

paths = exporter.export_all(run_record, leaderboard)
for p in paths:
    print(f"Exported: {p}")
```
