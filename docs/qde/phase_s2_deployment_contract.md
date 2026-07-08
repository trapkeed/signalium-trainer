# Phase S2 Deployment Contract — Versioning & Rollback for the 10-Head Schema

**Date:** 2026-07-08
**Modules:** `trainer/qde/qde_models.py` (`QDEVersion`), `trainer/qde/qde_rollback.py` (`QDERollbackManager`)
**Related:** [`phase_s2_schema_contract.md`](phase_s2_schema_contract.md), [`rollback_procedures.md`](rollback_procedures.md), [`phase_s1_migration_report.md`](phase_s1_migration_report.md) (documents the gap this closes).

## What changed

`QDEVersion` gained two fields, closing the gap flagged in Phase S1:

```python
output_schema_version: str = ""       # e.g. "s2_v1_10head"
feature_contract_version: str = ""    # reserved for the input-feature contract
```

Both default to `""` so every pre-existing `QDEVersion` (created before this phase) remains valid — an empty string means "schema not recorded," not "schema mismatch."

`QDERollbackManager.rollback_to()` gained an opt-in `require_schema_match: bool = False` parameter. When `True`, it refuses the rollback with a `ValueError` if the target version's `output_schema_version` is non-empty, the current active version's is non-empty, and they differ. When either side has no recorded schema, the check is skipped (treated as compatible) so legacy versions stay rollback-eligible.

## Why opt-in rather than always-on

Defaulting to `False` preserves the existing rollback contract ("any RETIRED/DEPLOYMENT_CANDIDATE/SHADOW_TESTING/ACTIVE version is a valid target") documented in `rollback_procedures.md` for every existing caller. Operators/automation that specifically care about schema safety (e.g. an automated rollback triggered by a monitoring alert, where an operator isn't present to sanity-check the target) should pass `require_schema_match=True` explicitly. A future phase can flip the default once all live versions reliably populate `output_schema_version`.

## Registering a version with its schema

```python
QDEVersion.new(
    version_number=3,
    label="QDE v3 — 10-head heuristic baseline",
    output_schema_version=OUTPUT_SCHEMA_VERSION,  # from trainer.qde.qde_outputs
)
```

Any pipeline step that registers a new `QDEVersion` (training, promotion) should pass the schema version of the model artifact it is registering, so `output_schema_version` is populated going forward.

## Forward compatibility with Phase S3

Phase S3 turns QDE into a live position manager reasoning every 30 seconds over open-position telemetry. This does not require a new output *schema* — the 10 heads already carry what S3 needs:

- `reduce_signal` / `scale_signal` → map directly to `REDUCE` / `ADD`.
- `management_confidence` → the `management_confidence` field S3's response contract calls for.
- `risk_signal` → the `risk_signal` field S3's response contract calls for.
- `exit_horizon`, `attention_focus` → inputs to deciding `CLOSE` vs. `TRAIL` vs. `MOVE_TO_BREAKEVEN` vs. `LOCK_PROFIT`.

What S3 *will* need to add is a new **API/inference contract** (a `manage_position()`-style call distinct from `decide()`, invoked on a timer rather than per-signal) and a **lifecycle/telemetry contract** (what open-position telemetry looks like as an input). Those are new contracts layered on top of this schema, not a redefinition of it — `output_schema_version` should stay `s2_v1_10head` through S3 unless the heads themselves change.

## Test evidence

`tests/unit/test_qde_rollback.py` (5 new tests: default fields, explicit fields, `to_dict()` inclusion, strict-check block on mismatch, strict-check pass on match, strict-check pass when either side has no recorded schema).
