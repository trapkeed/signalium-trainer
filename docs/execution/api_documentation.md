# API Documentation — Execution Intelligence

## Overview

`ExecutionApi` provides 16 pure-Python endpoints. All methods accept and return plain dicts. No HTTP layer, no broker connection.

## Initialisation

```python
from trainer.execution import ExecutionApi, BrokerConstraints, RiskLimits

api = ExecutionApi(
    db_path       = "data/execution.db",   # or ":memory:"
    sizing_method = "FIXED_RISK",
)
```

## Response Format

All responses follow a consistent envelope:

```json
{
  "status": "ok",          // or "error"
  "data": { ... },
  "timestamp": 1720134567.123,
  "meta": { ... }          // optional
}
```

---

## Endpoints

### 1. `plan(forecast_dict)` → full execution plan

Generates a complete 11-component execution plan for one forecast.

```python
resp = api.plan({
    "instrument": "EURUSD", "direction": "BUY",
    "confidence": 0.75, "expected_return": 0.08, "expected_risk": 0.10,
    "current_price": 1.0850, "atr": 0.0012, "spread": 0.0001,
    "session": "LONDON", "volatility": 0.009, "momentum": 0.3,
    "news_risk": 0.1, "account_size": 10000.0,
})
# resp["data"] contains executive_summary, entry_zones, stop_loss,
# take_profit, sizing, management, timing, risk_profile,
# order_strategy, trade_score, simulation
```

### 2. `plan_batch(forecast_dicts)` → list of plans (sorted by score desc)

### 3. `entry_zones(forecast_dict)` → EntryZones dict

```json
{
  "primary_entry": 1.0847, "secondary_entry": 1.0843,
  "aggressive_entry": 1.0849, "conservative_entry": 1.0842,
  "late_entry": 1.0840, "invalidation": 1.0832
}
```

### 4. `entry_quality(forecast_dict)` → EntryQualityResult dict

```json
{
  "grade": "GOOD", "score": 0.7234, "tradeable": true,
  "reasons": [...], "risk_factors": [...],
  "sub_scores": { "spread": 0.85, "session": 0.85, ... }
}
```

### 5. `stop_loss(forecast_dict)` → StopLossRecommendation dict

```json
{
  "initial_stop": 1.0832, "adaptive_stop": 1.0836,
  "atr_stop": 1.0826, "structure_stop": 1.0820,
  "volatility_stop": 1.0817, "emergency_stop": 1.0808,
  "stop_confidence": 0.81, "expected_hit_probability": 0.28
}
```

### 6. `take_profit(forecast_dict)` → TakeProfitRecommendation dict

```json
{
  "tp1": 1.0862, "tp2": 1.0869, "tp3": 1.0876,
  "runner_target": 1.0946,
  "probabilities": { "tp1": 0.675, "tp2": 0.472, "runner": 0.231 },
  "best_expectancy_level": "TP2"
}
```

### 7. `sizing(forecast_dict, method="FIXED_RISK")` → SizingResult dict

```json
{
  "lots": 0.08, "units": 8000, "risk_amount": 12.48,
  "risk_pct": 0.00124, "kelly_fraction": 0.35,
  "margin_required": 28.84, "commission_cost": 0.56
}
```

### 8. `management(forecast_dict)` → ManagementPlan dict

### 9. `timing(forecast_dict)` → TimingAssessment dict

```json
{
  "recommendation": "EXECUTE_NOW", "overall_score": 0.82,
  "session": "LONDON", "expected_slippage": 0.000027,
  "flags": { "spread_ok": true, "session_ok": true, ... }
}
```

### 10. `risk(forecast_dict)` → RiskProfile dict

```json
{
  "prob_success": 0.472, "prob_failure": 0.238,
  "risk_reward_ratio": 1.5, "expected_value": 0.000234,
  "positive_ev": true, "tail_risk": 0.000850
}
```

### 11. `order(forecast_dict)` → OrderRecommendation dict

```json
{
  "primary_type": "LIMIT", "alternative_type": "STOP_LIMIT",
  "limit_price": 1.0847, "expiry_hours": 24.0,
  "fill_confidence": 0.65
}
```

### 12. `score(forecast_dict)` → TradeScore dict

```json
{
  "grade": "GOOD", "overall_score": 0.71,
  "sub_scores": { "forecast_quality": 0.72, "entry_quality": 0.69, ... }
}
```

### 13. `simulation(forecast_dict)` → SimulationResult dict

```json
{
  "expected_fill": 1.0847, "expected_slippage": 0.000027,
  "fill_quality": "GOOD",
  "expected_pnl_pct": 0.0042, "pnl_std_dev": 0.0081
}
```

### 14. `dashboard(forecast_dicts)` → 8 dashboard views

Returns all of: execution_dashboard, risk_dashboard, trade_timeline, probability_charts, stop_distribution, target_distribution, execution_scorecards, fill_quality_analysis.

### 15. `stats()` → database statistics

```json
{
  "execution_plans": 47, "executable_plans": 38,
  "trade_scores": 47, "average_score": 0.6821
}
```

### 16. `history(instrument=None, limit=10)` → stored plan history
