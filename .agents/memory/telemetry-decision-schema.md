---
name: Telemetry DecisionEvent schema
description: Live telemetry decisions use direction string, not probability fields — health monitors need fallback logic.
---

# Telemetry DecisionEvent Schema

The `TelemetryWarehouse.ingest_decision()` / `DecisionEvent.new()` API uses:
- `direction: str` (e.g. "buy", "sell", "no_trade") — **not** `buy_prob`/`sell_prob`/`no_trade_prob`
- `confidence: float`
- `regime: str`
- `symbol: str`, `model_version: str`, `timestamp: float`

**Why:** The live telemetry schema was designed for decisions from the QDE engine which emits a single direction, not probabilities. Probability fields only exist in `ModelHealthMonitor`'s `HealthRecord` (the MLOps-side in-memory monitor).

**How to apply:** Any code that reads from `warehouse.query_decisions()` and wants a class distribution must handle both schemas:
1. If `buy_prob` is present in any record → use explicit probability fields
2. Otherwise → count `direction` field occurrences

Pattern (used in `health_monitor._class_distribution`):
```python
if any("buy_prob" in d for d in decisions):
    return {"buy": mean(buy_prob), ...}
# fallback:
return {"buy": count("buy")/n, "sell": count("sell")/n, "no_trade": rest/n}
```
