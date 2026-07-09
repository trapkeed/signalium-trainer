---
name: S30 readiness aggregators are pure rollups
description: Design pattern for production-verification/startup-check/autonomous-readiness modules added in S30
---

`trainer/research_lab/production_verification.py`, `startup_self_check.py`, and
`autonomous_readiness.py` are aggregation-only modules — they never re-derive
model/artifact/health/dataset/lineage verification logic. Each dimension
delegates to an already-existing checker (e.g. `ArtifactIntegrityVerifier`,
`check_feature_health`, `DatasetValidator`, `QDERegistry`, composite health
scorer, `LearningLoopScheduler.status()`).

**Why:** every prior audited system already had per-dimension verifiers; the
only genuinely missing piece was a single rolled-up report. Re-deriving any
check risks the "duplicated-drift-math-parity" failure mode already seen
elsewhere in this codebase.

**How to apply:** every dependency on these aggregators is optional
(`None`-able constructor arg). An unwired dependency reports `SKIPPED`, not a
crash or a `FAIL` — this lets partially-wired test/dev environments still
produce a usable report. Only a dimension that *was* checked and failed
produces `FAIL`/`NOT_READY`. `ProductionVerificationReport`/
`AutonomousReadinessReport`/`StartupSelfCheckReport` overall status is the
worst status among the individually-run dimensions.
