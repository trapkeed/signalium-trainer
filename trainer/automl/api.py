"""Phase 16 — AutoML Platform API.

Internal REST-like API layer for the Institutional AutoML Platform.
Returns structured dict responses — the HTTP layer is left to the CLI
or a FastAPI wrapper.

API endpoints:
  - AutoMLAPI.create_job()             → create a new search job
  - AutoMLAPI.get_job()                → get job status and summary
  - AutoMLAPI.list_jobs()              → list all jobs
  - AutoMLAPI.pause_job()              → pause a running job
  - AutoMLAPI.resume_job()             → resume a paused job
  - AutoMLAPI.get_trial()              → get a single trial record
  - AutoMLAPI.list_trials()            → list trials for a job
  - AutoMLAPI.get_optimization_history() → score history for a job
  - AutoMLAPI.get_leaderboard()        → ranked experiment leaderboard
  - AutoMLAPI.get_pareto_front()       → Pareto-optimal trials
  - AutoMLAPI.get_promotion_candidates() → trials eligible for promotion
  - AutoMLAPI.approve_promotion()      → manually approve a promotion
  - AutoMLAPI.get_dashboard()          → full visualization dashboard
  - AutoMLAPI.get_stats()              → platform statistics

Usage::

    api = AutoMLAPI(database, ranker, scorer)
    response = api.get_leaderboard(job_id="...", limit=20)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from trainer.automl.database import AutoMLDatabase
from trainer.automl.objectives import ObjectiveScorer, build_research_scorer
from trainer.automl.ranking import ExperimentRanker
from trainer.automl.visualization import AutoMLVisualizationBuilder

# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _ok(data: Any, message: str = "") -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": message,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _error(message: str, code: str = "ERROR") -> Dict[str, Any]:
    return {
        "status": "error",
        "code": code,
        "message": message,
        "data": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# AutoML API
# ---------------------------------------------------------------------------


class AutoMLAPI:
    """Internal API for the Institutional AutoML Platform.

    All methods return ``{"status": "ok"|"error", "data": ..., ...}``.

    Args:
        database: AutoMLDatabase instance.
        ranker: ExperimentRanker for scoring and ranking.
        scorer: ObjectiveScorer for Pareto front computation.
        viz: AutoMLVisualizationBuilder for chart data.
    """

    def __init__(
        self,
        database: AutoMLDatabase,
        ranker: Optional[ExperimentRanker] = None,
        scorer: Optional[ObjectiveScorer] = None,
        viz: Optional[AutoMLVisualizationBuilder] = None,
    ) -> None:
        self.db = database
        self.ranker = ranker or ExperimentRanker()
        self.scorer = scorer or build_research_scorer()
        self.viz = viz or AutoMLVisualizationBuilder()

    # -----------------------------------------------------------------------
    # Search jobs
    # -----------------------------------------------------------------------

    def create_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and persist a new search job from a config dict."""
        try:
            from trainer.automl.experiment import SearchJob

            job = SearchJob(
                name=job_config.get("name", "search_job"),
                strategy=job_config.get("strategy", "random"),
                n_trials=job_config.get("n_trials", 20),
                objective=job_config.get("objective", "composite_research_score"),
                objectives=job_config.get("objectives", []),
                search_space_name=job_config.get("search_space_name", "signalium_standard"),
                budget_type=job_config.get("budget_type", "trials"),
                budget_value=float(job_config.get("budget_value", 20)),
                pruner_type=job_config.get("pruner_type", "median"),
                parallel_workers=job_config.get("parallel_workers", 1),
                max_retries=job_config.get("max_retries", 2),
                metadata=job_config.get("metadata", {}),
            )
            self.db.save_job(job)
            logger.info(f"Created search job {job.job_id} ({job.name})")
            return _ok(job.to_dict(), f"Job {job.name} created")
        except Exception as e:
            logger.error(f"create_job failed: {e}")
            return _error(str(e), "CREATE_JOB_ERROR")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get full job status and summary."""
        try:
            job = self.db.get_job(job_id)
            if not job:
                return _error(f"Job {job_id!r} not found", "NOT_FOUND")
            records = self.db.get_records_for_job(job_id)
            job["n_records"] = len(records)
            job["n_completed"] = sum(1 for r in records if r.get("status") == "completed")
            job["n_pruned"] = sum(1 for r in records if r.get("pruned"))
            return _ok(job)
        except Exception as e:
            logger.error(f"get_job failed: {e}")
            return _error(str(e), "GET_JOB_ERROR")

    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """List all search jobs."""
        try:
            jobs = self.db.list_jobs(status=status, limit=limit)
            return _ok({"jobs": jobs, "count": len(jobs)})
        except Exception as e:
            logger.error(f"list_jobs failed: {e}")
            return _error(str(e), "LIST_JOBS_ERROR")

    def pause_job(self, job_id: str) -> Dict[str, Any]:
        """Pause a running job."""
        try:
            self.db.update_job_status(job_id, "paused")
            return _ok({"job_id": job_id, "status": "paused"}, "Job paused")
        except Exception as e:
            return _error(str(e), "PAUSE_JOB_ERROR")

    def resume_job(self, job_id: str) -> Dict[str, Any]:
        """Resume a paused job."""
        try:
            self.db.update_job_status(job_id, "running")
            return _ok({"job_id": job_id, "status": "running"}, "Job resumed")
        except Exception as e:
            return _error(str(e), "RESUME_JOB_ERROR")

    # -----------------------------------------------------------------------
    # Trials
    # -----------------------------------------------------------------------

    def get_trial(self, record_id: str) -> Dict[str, Any]:
        """Get a single experiment record."""
        try:
            record = self.db.get_record(record_id)
            if not record:
                return _error(f"Record {record_id!r} not found", "NOT_FOUND")
            return _ok(record)
        except Exception as e:
            return _error(str(e), "GET_TRIAL_ERROR")

    def list_trials(
        self,
        job_id: Optional[str] = None,
        status: Optional[str] = None,
        backbone: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List experiment records with optional filters."""
        try:
            records = self.db.list_records(
                job_id=job_id, status=status, backbone=backbone, limit=limit, offset=offset
            )
            return _ok({"trials": records, "count": len(records)})
        except Exception as e:
            return _error(str(e), "LIST_TRIALS_ERROR")

    # -----------------------------------------------------------------------
    # Optimization history
    # -----------------------------------------------------------------------

    def get_optimization_history(self, job_id: str) -> Dict[str, Any]:
        """Return score history for a job."""
        try:
            history = self.db.score_history(job_id)
            chart = self.viz.optimization_history(
                self.db.get_records_for_job(job_id, status="completed")
            )
            return _ok({"history": history, "chart": chart})
        except Exception as e:
            return _error(str(e), "OPT_HISTORY_ERROR")

    # -----------------------------------------------------------------------
    # Leaderboard
    # -----------------------------------------------------------------------

    def get_leaderboard(
        self, job_id: Optional[str] = None, limit: int = 20, sort_by: str = "primary_score"
    ) -> Dict[str, Any]:
        """Return ranked experiment leaderboard."""
        try:
            if job_id:
                records = self.db.get_records_for_job(job_id, status="completed")
            else:
                records = self.db.list_records(status="completed", limit=500)
            ranking = self.ranker.rank_all(records)
            top = [e.to_dict() for e in ranking[:limit]]
            return _ok(
                {
                    "leaderboard": top,
                    "total": len(ranking),
                    "job_id": job_id,
                }
            )
        except Exception as e:
            return _error(str(e), "LEADERBOARD_ERROR")

    # -----------------------------------------------------------------------
    # Pareto front
    # -----------------------------------------------------------------------

    def get_pareto_front(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Return Pareto-optimal trials across all objectives."""
        try:
            if job_id:
                records = self.db.get_records_for_job(job_id, status="completed")
            else:
                records = self.db.list_records(status="completed", limit=500)
            if not records:
                return _ok({"pareto": [], "total": 0})
            metrics_list = [r.get("validation_results") or r.get("config") or {} for r in records]
            pareto_idxs = self.scorer.pareto_front(metrics_list)
            pareto = [records[i] for i in pareto_idxs]
            return _ok(
                {
                    "pareto": pareto,
                    "pareto_indices": pareto_idxs,
                    "total_evaluated": len(records),
                    "n_pareto": len(pareto),
                }
            )
        except Exception as e:
            return _error(str(e), "PARETO_ERROR")

    # -----------------------------------------------------------------------
    # Promotion
    # -----------------------------------------------------------------------

    def get_promotion_candidates(
        self, job_id: Optional[str] = None, limit: int = 20
    ) -> Dict[str, Any]:
        """Return trials that passed promotion criteria."""
        try:
            if job_id:
                records = self.db.get_records_for_job(job_id, status="completed")
                candidates = [r for r in records if r.get("promotion_eligible")]
            else:
                all_records = self.db.list_records(status="completed", limit=500)
                candidates = [r for r in all_records if r.get("promotion_eligible")]
            candidates.sort(key=lambda r: r.get("primary_score") or 0.0, reverse=True)
            history = self.db.get_promotion_history(job_id=job_id, limit=50)
            return _ok(
                {
                    "candidates": candidates[:limit],
                    "total_candidates": len(candidates),
                    "promotion_history": history,
                }
            )
        except Exception as e:
            return _error(str(e), "PROMOTION_CANDIDATES_ERROR")

    def approve_promotion(
        self, trial_id: str, job_id: str, model_version: str = "", notes: str = ""
    ) -> Dict[str, Any]:
        """Manually approve a trial for promotion to champion."""
        try:
            self.db.save_promotion_decision(
                trial_id=trial_id,
                job_id=job_id,
                model_version=model_version,
                is_eligible=True,
                auto_approved=False,
                manually_approved=True,
                champion_status=True,
                rollback_eligible=True,
                rules=[],
                notes=notes or "Manually approved via AutoML API.",
            )
            logger.info(f"Manual promotion approved: trial {trial_id}")
            return _ok(
                {
                    "trial_id": trial_id,
                    "champion_status": True,
                    "manually_approved": True,
                },
                "Promotion approved",
            )
        except Exception as e:
            return _error(str(e), "APPROVE_PROMOTION_ERROR")

    # -----------------------------------------------------------------------
    # Dashboard
    # -----------------------------------------------------------------------

    def get_dashboard(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Return full visualization dashboard data."""
        try:
            if job_id:
                records = self.db.get_records_for_job(job_id)
            else:
                records = self.db.list_records(limit=500)
            pareto_response = self.get_pareto_front(job_id)
            pareto_idxs = None
            if pareto_response["status"] == "ok":
                pareto_idxs = pareto_response["data"].get("pareto_indices")
            dashboard = self.viz.full_dashboard(records, pareto_indices=pareto_idxs)
            backbone_summary = self.db.backbone_summary(job_id)
            return _ok(
                {
                    "charts": dashboard,
                    "backbone_summary": backbone_summary,
                    "stats": self.db.stats(),
                }
            )
        except Exception as e:
            return _error(str(e), "DASHBOARD_ERROR")

    # -----------------------------------------------------------------------
    # Platform stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return overall AutoML platform statistics."""
        try:
            stats = self.db.stats()
            jobs = self.db.list_jobs(limit=10)
            stats["recent_jobs"] = jobs
            return _ok(stats)
        except Exception as e:
            return _error(str(e), "STATS_ERROR")

    # -----------------------------------------------------------------------
    # Champion/Challenger workflow
    # -----------------------------------------------------------------------

    def champion_challenger_compare(
        self, champion_trial_id: str, challenger_trial_id: str
    ) -> Dict[str, Any]:
        """Compare champion and challenger side-by-side."""
        try:
            champion = self.db.get_record(champion_trial_id)
            challenger = self.db.get_record(challenger_trial_id)
            if not champion:
                return _error(f"Champion {champion_trial_id!r} not found", "NOT_FOUND")
            if not challenger:
                return _error(f"Challenger {challenger_trial_id!r} not found", "NOT_FOUND")

            champ_metrics = champion.get("validation_results") or {}
            chall_metrics = challenger.get("validation_results") or {}

            champ_score = champion.get("primary_score") or 0.0
            chall_score = challenger.get("primary_score") or 0.0
            winner = "champion" if champ_score >= chall_score else "challenger"
            score_delta = chall_score - champ_score

            return _ok(
                {
                    "champion": {
                        "trial_id": champion_trial_id,
                        "score": champ_score,
                        "backbone": champion.get("backbone"),
                        "metrics": champ_metrics,
                    },
                    "challenger": {
                        "trial_id": challenger_trial_id,
                        "score": chall_score,
                        "backbone": challenger.get("backbone"),
                        "metrics": chall_metrics,
                    },
                    "winner": winner,
                    "score_delta": round(score_delta, 6),
                    "challenger_improves": score_delta > 0,
                }
            )
        except Exception as e:
            return _error(str(e), "CHAMPION_CHALLENGER_ERROR")
