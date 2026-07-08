---
name: Ensemble already removed pre-history
description: trainer/ensemble/ was deleted before this repo's visible git history begins — relevant for any future "remove/audit the ensemble" task
---

`trainer/ensemble/` (ExpertRouter, ExpertRegistry, VotingEngine/8 voting
strategies, EnsembleMetaModel, DynamicWeightEngine, EnsembleLeaderboard,
EnsembleDatabase, etc.) and `tests/unit/test_ensemble_platform.py` do not
exist in any commit reachable from `HEAD`. The graft/root commit of this
repo's history is itself titled "Remove multi-brain architecture and related
ensemble systems" — i.e. the removal predates every phase of work done in
this session (S1 onward) and predates this repo's visible history entirely.

**Why this matters:** a later "remove the ensemble" or "migrate off
multi-model architecture" request will find nothing left to remove. Also, no
historical ensemble runtime data (expert performance, voting history,
leaderboard history, calibration history) was ever persisted in this
environment — no `experiments/ensemble/` directory or `*.db` file exists —
so there is no real data to export into Research Lab datasets, only the
schema/intent, which should be documented rather than fabricated.

**How to apply:**
- Before doing removal work, run `git log --oneline --all | tail -5` to find
  the graft point, and check whether the target code already doesn't exist.
- If a task requires "before removal" validation (e.g. single-model must
  outperform the system being removed) and the system is already gone,
  do not fabricate comparison numbers — report the gap honestly in the
  migration report and explain why it can't be produced retroactively.
- Pre-existing docs `docs/ensemble_removal_migration.md` and
  `docs/architecture_audit_v2.md` contain the original file-by-file
  removal rationale/structure and remain the authoritative record of what
  the ensemble code looked like, since the code itself isn't recoverable
  from this repo.
- Phase S5 (2026-07-08) archived the one remaining live artifact
  (`config/ensemble/ensemble.yaml`, dead/unread config) to
  `docs/archive/ensemble/` and registered 4 placeholder `ResearchDataset`
  rows in `data/research_lab.db` tagged `ensemble-archive` +
  `no-persisted-data-found` for the 4 requested categories.
