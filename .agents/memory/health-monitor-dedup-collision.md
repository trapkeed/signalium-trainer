---
name: Health monitor recommendation dedup collision
description: Multiple DriftDimensions mapping to one coarse EvaluationDimension can silently suppress distinct CRITICAL recommendations if dedup only keys on the coarse dimension.
---

When several fine-grained health/drift dimensions are intentionally mapped to the same coarse `EvaluationDimension` (e.g. for recommendation categorization without extending an enum), retraining-recommendation dedup must not key on that coarse dimension alone — otherwise an open recommendation from one fine-grained dimension silently suppresses a later, distinct CRITICAL alert from another that happens to share the same coarse category.

**Why:** Discovered when adding ENTROPY/UNCERTAINTY/CALIBRATION dimensions to `trainer/research_lab/health_monitor.py`'s `_check_*` methods — all three were mapped to `EvaluationDimension.CONFIDENCE_DRIFT` for simplicity (no enum change), and the original dedup logic in `_maybe_create_recommendation` matched only on that mapped value, so a CRITICAL entropy alert would prevent a CALIBRATION recommendation (with its distinct `CALIBRATION_ONLY` scope) from ever being created.

**How to apply:** When adding a new fine-grained signal that reuses an existing coarse categorization value, also dedupe on the fine-grained source identifier (stash it in the recommendation's `evidence` dict, e.g. `_health_dimension`) so dedup requires a match on *both* the coarse category and the fine-grained source before treating two alerts as duplicates.
