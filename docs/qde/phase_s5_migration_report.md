# Phase S5 Migration Report — Multi-Model Architecture Removal

**Date:** 2026-07-08
**Related:** [`phase_s5_dependency_report.md`](phase_s5_dependency_report.md), [`phase_s5_archive_report.md`](phase_s5_archive_report.md), [`../ensemble_removal_migration.md`](../ensemble_removal_migration.md) (pre-existing removal record, present since before this session), [`../architecture_audit_v2.md`](../architecture_audit_v2.md)

## Objective (as requested)

Remove the multi-model ensemble architecture (`ExpertRouter`, `ExpertRegistry`, `VotingEngine`, `MetaModel`, `WeightingEngine`, Leaderboards, `EnsembleDatabase`), archive its historical data (expert performance, voting history, leaderboard history, calibration history) into Research Lab datasets, keep shadow inference / champion / challenger / research comparisons intact, and — critically — require that a single QDE model demonstrably outperform the ensemble system on direction accuracy, drawdown, MFE, MAE, risk-adjusted return, management quality, and latency **before** the ensemble is allowed to be removed.

## Compliance gap found: the validation gate could not be enforced retroactively

This is the central finding of this report and is stated plainly rather than glossed over:

**`trainer/ensemble/` does not exist anywhere in this repository's git history.** The oldest commit reachable from `HEAD` (`d49cddf`, the graft/root of this shallow repository) is itself titled *"Remove multi-brain architecture and related ensemble systems"* — meaning the ensemble code was already gone before the very first commit visible to this workspace, and therefore before any Phase S1–S4 work in this session and before this Phase S5 request was made.

Consequently:
- **The required "single QDE model must outperform ensemble" validation could not be executed by this task.** Running that comparison requires the ensemble system's actual inference code (to generate its predictions on a shared evaluation set alongside the QDE model), and that code does not exist in this environment to run. Re-implementing the entire ensemble system from scratch solely to satisfy a before/after gate — then deleting it again — would (a) contradict the "One Model, One Brain" direction the rest of this project has been moving toward since S1, and (b) still not answer the real question (whether the *actual, historically-run* ensemble outperformed), since no historical predictions or backtest artifacts from that system were found either (see the archive report — no `experiments/ensemble/` data of any kind exists).
- **No historical validation evidence was found as a substitute.** A search of `docs/` for content matching the seven required metrics (direction accuracy, drawdown, MFE, MAE, risk-adjusted return, management quality, latency) in the context of an ensemble-vs-QDE comparison returned nothing. `docs/architecture_audit_v2.md` (the audit that originally recommended removing `ensemble/`) argued the removal on architectural grounds (multiple simultaneous decision-makers, "One Decision Maker" violation) — not on a head-to-head performance comparison.
- **This is being reported rather than backfilled with fabricated numbers.** Inventing comparison metrics for a system that cannot be run and was never measured would misrepresent the state of the project. The honest state is: the removal already happened, for architectural-compliance reasons, without the performance-gate evidence this phase's spec requires.

### What this means going forward

The removal is a fait accompli in the current codebase and cannot be un-done by this task without reconstructing ~4,800 lines of deleted ensemble code — which is not a productive use of this phase given the project's stated direction. The actionable response is:
1. Treat this report as the record that the gate was **not** satisfied at the time of removal, for future audits.
2. Close the parts of Phase S5 that remain genuinely actionable now: confirming the REMOVE/KEEP boundary (dependency report), archiving what historical data exists or can be accounted for (archive report — see finding that no real historical rows exist), and documenting this gap clearly.
3. If a live champion-vs-challenger comparison equivalent to the required gate is wanted going forward, it is achievable today using the retained SHADOW/SIDE_BY_SIDE infrastructure (`trainer/qde/qde_ab_testing.py`, `trainer/qde/qde_shadow.py`) — e.g. running the single QDE model against a would-be multi-model baseline is no longer possible since that baseline doesn't exist, but any future architecture change of this kind should run its comparison *before* deletion, using `ModelComparisonResult` (`trainer/research_lab/rl_models.py`), which already reports exactly this class of metrics for QDE-version-vs-version comparisons.

## What this phase actually delivered

| Requirement | Status | Detail |
|---|---|---|
| Remove `ExpertRouter`, `ExpertRegistry`, `VotingEngine`, `MetaModel`, `WeightingEngine`, Leaderboards, `EnsembleDatabase` | **Already done** (pre-session) | Verified absent; see [dependency report](phase_s5_dependency_report.md) |
| Archive `trainer/ensemble/` | **Not recoverable** — code does not exist in reachable git history | Its structure is preserved in `docs/ensemble_removal_migration.md` and `docs/architecture_audit_v2.md`, both pre-existing |
| Archive expert performance / voting history / leaderboard history / calibration history → Research Lab datasets | **Done, as placeholders** | No real historical data existed in the repository to migrate (see [archive report](phase_s5_archive_report.md)); 4 `ResearchDataset` records registered in `data/research_lab.db` documenting the schema and the empty-data finding |
| Keep shadow inference, champion model, challenger model, research comparisons | **Verified intact** | See [dependency report](phase_s5_dependency_report.md) KEEP table |
| Validation: single QDE model must outperform ensemble on 7 named metrics before removal | **Not satisfied — reported as a gap, not backfilled** | Ensemble code unavailable to run the comparison; no historical evidence found; documented above |
| Migration report | **This document** | |
| Dependency report | [`phase_s5_dependency_report.md`](phase_s5_dependency_report.md) | |
| Archive report | [`phase_s5_archive_report.md`](phase_s5_archive_report.md) | |

## Residual cleanup performed in this phase

- `config/ensemble/ensemble.yaml` (dead config, unread by any code) moved to `docs/archive/ensemble/ensemble.yaml`; empty `config/ensemble/` directory removed.
- 4 placeholder `ResearchDataset` entries created in the Research Lab database (`data/research_lab.db`) for the four requested archive categories.

No source code in `trainer/` or `apps/` was modified in this phase — there was no ensemble code left to modify or remove, and the KEEP systems (shadow inference, champion/challenger, research comparisons) required no changes since they already satisfy Phase S5's requirements.
