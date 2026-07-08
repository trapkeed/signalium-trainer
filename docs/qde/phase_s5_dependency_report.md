# Phase S5 Dependency Report — Multi-Model Ensemble Architecture

**Date:** 2026-07-08
**Related:** [`phase_s5_archive_report.md`](phase_s5_archive_report.md), [`phase_s5_migration_report.md`](phase_s5_migration_report.md), [`decision_engines_context_providers.md`](decision_engines_context_providers.md), [`architecture_audit_v2.md`](../architecture_audit_v2.md) (prior audit that first flagged `ensemble/` for removal)

## Purpose

Verify, as of this audit, exactly what still depends on the multi-model ensemble architecture, and confirm the REMOVE/KEEP boundary requested for Phase S5 is actually satisfied in the current codebase.

## Important finding: removal predates this session

`trainer/ensemble/` and its 15 files (`engine.py`, `expert.py`, `routing.py`, `strategies.py`, `meta_model.py`, `weighting.py`, `calibration.py`, `failure_analysis.py`, `online_scoring.py`, `leaderboard.py`, `database.py`, `api.py`, `export.py`, `visualization.py`, `__init__.py`) and `tests/unit/test_ensemble_platform.py` **do not exist in this repository at any point in its visible git history.** The commit `d49cddf — "Remove multi-brain architecture and related ensemble systems"` is the graft point (the oldest visible commit, i.e. the seed state of the repo) — the removal happened before any Phase S1–S4 work in this session, and before this Phase S5 request was made.

This changes the shape of the S5 task from "remove the ensemble system" to "audit that the removal already performed is complete and compliant, close the gaps it left (archiving, dataset export, dependency/archive reporting), and report honestly on the validation requirement that could not be applied retroactively." See [`phase_s5_migration_report.md`](phase_s5_migration_report.md) for how that gap is handled.

## REMOVE list — verified absent

| Class | Status | Verification |
|---|---|---|
| `ExpertRouter` | Absent | No `class ExpertRouter` anywhere in `trainer/` or `apps/` |
| `ExpertRegistry` | Absent | No `class ExpertRegistry` anywhere in `trainer/` or `apps/` |
| `VotingEngine` | Absent | No `class VotingEngine`; the 8 ensemble-era voting strategies (`SimpleAveragingStrategy`, `WeightedAveragingStrategy`, `PerformanceWeightedStrategy`, `HardVotingStrategy`, `SoftVotingStrategy`, `MixtureOfExpertsStrategy`, `StackingStrategy`, `BlendingStrategy`) are absent |
| `MetaModel` | Absent | No `class EnsembleMetaModel` / `MetaFeatureBuilder` |
| `WeightingEngine` | Absent | No `class DynamicWeightEngine` or equivalent per-expert weighting engine |
| `Leaderboards` (ensemble) | Absent | No `class EnsembleLeaderboard`. Note: `trainer/research/leaderboard/leaderboard.py`, `trainer/research/leaderboard_irse.py`, and `trainer/validation/leaderboard.py` do exist, but these are research/validation-scoped model leaderboards (Phase 15/19.2), not the ensemble's production expert-ranking leaderboard — see KEEP list below |
| `EnsembleDatabase` | Absent | No `class EnsembleDatabase`; no `experiments/ensemble/ensemble.db` file exists anywhere in the repository |
| `CANARY` A/B test mode | Absent | `trainer/qde/qde_ab_testing.py` docstring explicitly states "CANARY mode (routing live traffic to a challenger) has been removed"; no `QDEABTestType.CANARY` enum member, no `create_canary()`/`create_canary_test()` methods, no `/qde/ab/canary` endpoint |

**Zero external consumers found.** `grep` across `trainer/`, `apps/`, and every test file found no import of `trainer.ensemble`, no reference to `ExpertRouter`/`ExpertRegistry`/`VotingEngine`/`MetaModel`/`WeightingEngine`/`EnsembleDatabase`, and no `EnsembleLeaderboard`. The ensemble module, when it existed, was self-contained — consistent with the finding already recorded in `docs/ensemble_removal_migration.md`.

## KEEP list — verified present and intact

| Requirement | Verified location | Role |
|---|---|---|
| Shadow inference | `trainer/qde/qde_shadow.py` — `ShadowDispatcher`, `DualDispatchResult`, `evaluate_and_maybe_promote()` | Fans live input out to champion + challenger; challenger's decision is logged, never executed |
| Champion model | `trainer/mlops/champion.py`; `trainer/qde/qde_registry.py` / `trainer/registry/unified_registry.py` (ACTIVE stage) | Tracks the one production-executing model version |
| Challenger model | `trainer/qde/qde_ab_testing.py` (`QDEABTestingEngine`, SHADOW / SIDE_BY_SIDE test types); `trainer/qde/qde_shadow.py` | A candidate version observed against live/replay data; never executes trades |
| Research comparisons | `trainer/research_lab/rl_models.py` (`ModelComparisonResult`, `ShadowDecisionRecord`, `FailureEvent`); `trainer/research/leaderboard/`, `trainer/research/leaderboard_irse.py`, `trainer/validation/leaderboard.py` | Multiple model *versions* may be compared for research/promotion purposes; none co-execute in production |

No KEEP module imports anything from the (already-removed) `trainer.ensemble` package.

## Residual debris found and cleaned up in this audit

| Item | Finding | Action taken |
|---|---|---|
| `config/ensemble/ensemble.yaml` | Dead configuration file for the AEIP (Adaptive Ensemble Intelligence Platform, Phase 17) ensemble strategy — `grep` confirms no code in `trainer/` or `apps/` reads this file or `config/ensemble/` | Moved to `docs/archive/ensemble/ensemble.yaml` (see [archive report](phase_s5_archive_report.md)); empty `config/ensemble/` directory removed |
| `experiments/ensemble/` (referenced by the dead config as `db_path`/`export_dir`) | Directory does not exist anywhere in the repository or its history | No action — nothing to archive; documented in the archive report |
| `docs/adaptive_ensemble_guide.md`, `docs/ensemble_architecture_report.md`, `docs/ensemble_removal_migration.md` | Historical documentation of the ensemble system and its removal, left in place from before this session | Retained as-is (historical record); this Phase S5 work supersedes them for current-state purposes but does not delete them |

## Conclusion

The REMOVE/KEEP boundary requested for Phase S5 is satisfied in the current codebase. No further code deletion is required — the ensemble system's code was already fully removed prior to this session. The remaining Phase S5 obligations (archiving the four historical-data categories into Research Lab datasets, and reporting on the pre-removal validation requirement) are addressed in the archive and migration reports.
