# Phase S5 Archive Report — Ensemble Historical Data

**Date:** 2026-07-08
**Related:** [`phase_s5_dependency_report.md`](phase_s5_dependency_report.md), [`phase_s5_migration_report.md`](phase_s5_migration_report.md)

## Scope

Per the S5 requirement, four categories of historical data tracked by the (already-removed) ensemble system were to be archived and exported into Research Lab datasets rather than silently discarded:

1. Expert performance
2. Voting history
3. Leaderboard history
4. Calibration history

## Finding: no persisted instance data exists to export

`trainer/ensemble/database.py` (`EnsembleDatabase`) was the SQLite persistence layer for these four categories, configured (per the archived `config/ensemble/ensemble.yaml`, see below) to write to `experiments/ensemble/ensemble.db`. A full-repository search at S5 audit time found:

- No `experiments/` directory anywhere in the repository.
- No `*.db` / `*.sqlite*` file matching `ensemble` anywhere in the repository or its git history.
- No CSV/JSON export artifacts under any `export_dir` referenced by the archived config.

**Conclusion:** the ensemble system's code was removed at the very start of this repository's visible history (see the dependency report), and no run of that system ever persisted real expert-performance, voting, leaderboard, or calibration rows into this environment. There is no actual historical data to migrate — only the *schema and intent* of what would have been tracked, which is documented below and preserved for reference.

This is reported plainly rather than assumed silently: if real ensemble history exists outside this repository (e.g. in a different environment or a prior deployment's data volume), it was not reachable from this workspace and could not be included in this export.

## What was archived

### 1. Dead configuration file
`config/ensemble/ensemble.yaml` (the Phase 17 Adaptive Ensemble Intelligence Platform config — strategy selection, weight factors, rolling windows, calibration settings) was confirmed unread by any current code path and moved to `docs/archive/ensemble/ensemble.yaml` for historical reference rather than left live in `config/`.

### 2. Research Lab placeholder datasets

Four `ResearchDataset` records were registered via `ResearchLabAPI.create_dataset()` (persisted in `data/research_lab.db`, the standard Research Lab store — see `trainer/research_lab/rl_database.py`) so the four required categories have a durable, discoverable home even though there is no historical row data to attach to them yet:

| Dataset name | Dataset ID | Tags | What it documents |
|---|---|---|---|
| `ensemble_expert_performance_archive` | `ds-b07837f8-d75` | `ensemble-archive`, `expert-performance`, `phase-s5`, `no-persisted-data-found` | Schema previously produced by `trainer/ensemble/online_scoring.py` (`OnlineScorer`, `PredictionRecord`) |
| `ensemble_voting_history_archive` | `ds-e5361b65-9d0` | `ensemble-archive`, `voting-history`, `phase-s5`, `no-persisted-data-found` | The 8 voting/combination strategies in `trainer/ensemble/strategies.py`, persisted via `EnsembleDatabase` |
| `ensemble_leaderboard_history_archive` | `ds-14d457bd-300` | `ensemble-archive`, `leaderboard-history`, `phase-s5`, `no-persisted-data-found` | `trainer/ensemble/leaderboard.py` (`EnsembleLeaderboard`) expert-ranking history; superseded by the research/validation leaderboards on the KEEP list |
| `ensemble_calibration_history_archive` | `ds-1788749b-866` | `ensemble-archive`, `calibration-history`, `phase-s5`, `no-persisted-data-found` | `trainer/ensemble/calibration.py` (`ExpertCalibrator`, `CalibrationReport`) per-expert calibration history; superseded by `trainer/calibration/` for the single active model |

Each dataset's `notes` field records the source module, why no rows could be imported, and the doc where the original schema is described (this report), so a future agent or researcher who finds these datasets in Research Lab understands immediately why they are empty and what they represent. `row_count=0` and `feature_count=0` are set explicitly rather than omitted, so downstream tooling that lists datasets can distinguish "known-empty archive placeholder" from "not yet computed."

Verification:
```
$ python -c "from trainer.research_lab.rl_database import ResearchLabDatabase; \
  db = ResearchLabDatabase(db_path='data/research_lab.db'); \
  print([d['name'] for d in db.list_datasets()])"
['ensemble_calibration_history_archive', 'ensemble_leaderboard_history_archive',
 'ensemble_voting_history_archive', 'ensemble_expert_performance_archive']
```

## What was NOT archived (already existed / out of scope)

- `docs/ensemble_architecture_report.md`, `docs/adaptive_ensemble_guide.md`, `docs/ensemble_removal_migration.md` — these were already present in the repository as historical documentation of the ensemble system's design and its removal. They were left in place rather than moved, since they already serve an archival purpose.
- The ensemble source code itself (`trainer/ensemble/*.py`) — not recoverable from this repository; it does not exist in any commit reachable from `HEAD`. Its structure (files, classes, one-line description of each) is preserved in `docs/architecture_audit_v2.md` (lines documenting the pre-removal audit) and in `docs/ensemble_removal_migration.md`'s file-by-file table, both of which predate this report and remain the authoritative code-shape record.

## Retention policy

These four placeholder datasets should be kept indefinitely as a record that the ensemble-era data categories were considered and accounted for, not overlooked. If a future research effort resurrects any of this functionality in a Research-Lab-only (never production) context, new datasets should reference these by name/lineage rather than reusing the same `dataset_id`s, since these are explicitly tagged as empty archival placeholders.
