"""Phase 16 — Institutional AutoML Engine.

Top-level orchestrator for the Institutional AutoML and Research
Optimization Platform.

Runs the complete AutoML search pipeline from a single call:

  1.  Load configuration (YAML or dict)
  2.  Build search space
  3.  Select search strategy
  4.  Initialize pruner
  5.  Initialize resource manager
  6.  Queue and execute trials
      a. Suggest hyperparameters
      b. Evaluate backbone configuration
      c. Run Phase 15 Institutional Validation (evaluation only)
      d. Record intermediate scores for pruning
      e. Prune poor performers
      f. Record resource usage
      g. Persist experiment record to DB
  7.  Rank all completed experiments
  8.  Integrate with Phase 15 leaderboard
  9.  Evaluate promotion eligibility
  10. Generate visualization dashboard
  11. Generate final AutoML report
  12. Persist final job state

No model is trained or modified directly. Training callbacks are injected
by the caller. Evaluation is always performed by the Phase 15 platform.

Usage::

    engine = AutoMLEngine.from_yaml("conf/automl/automl.yaml")
    result = engine.run(
        train_fn=my_train_fn,   # (config, backbone) → model + metrics dict
        model_id="quant_v1",
    )
    print(result.summary())
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from trainer.automl.search_space import SearchSpace, build_standard_space, BaseParameter
from trainer.automl.objectives import (
    ObjectiveScorer,
    build_research_scorer,
    compute_composite_research_score,
)
from trainer.automl.pruning import BasePruner, NoPruner, build_pruner
from trainer.automl.strategies import BaseStrategy, build_strategy, Suggestion
from trainer.automl.experiment import Trial, SearchJob, ExperimentRecord, TrialStatus
from trainer.automl.resource_manager import ResourceManager
from trainer.automl.backbone_search import (
    BackboneCatalog,
    BackboneSpec,
    BackboneEvalResult,
    BackboneSearcher,
)
from trainer.automl.ranking import ExperimentRanker, RankingResult
from trainer.automl.database import AutoMLDatabase
from trainer.automl.visualization import AutoMLVisualizationBuilder
from trainer.automl.api import AutoMLAPI

# ---------------------------------------------------------------------------
# Engine result
# ---------------------------------------------------------------------------


@dataclass
class AutoMLResult:
    """Complete result of an AutoML search run.

    Attributes:
        job: The completed SearchJob.
        champion_trial: Best trial found.
        champion_record: Experiment record for the champion.
        ranking: Full multi-dimensional ranking.
        dashboard: Visualization data dict.
        promotion_eligible_trials: Trials that passed promotion.
        report: Final AutoML platform report dict.
        total_duration_seconds: Total wall-clock time.
    """

    job: SearchJob
    champion_trial: Optional[Trial] = None
    champion_record: Optional[Dict[str, Any]] = None
    ranking: Optional[RankingResult] = None
    dashboard: Optional[Dict[str, Any]] = None
    promotion_eligible_trials: List[str] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    total_duration_seconds: float = 0.0

    def summary(self) -> str:
        j = self.job
        champ = self.champion_trial
        lines = [
            f"AutoML Search: {j.name} [{j.strategy}]",
            f"  Status: {j.status}",
            f"  Trials: {j.n_completed} completed / {j.n_pruned} pruned / {j.n_failed} failed",
            f"  Best score: {j.best_score:.4f}" if j.best_score else "  Best score: N/A",
            f"  Champion: {champ.trial_id if champ else 'N/A'}",
            f"  Backbone: {champ.backbone if champ else 'N/A'}",
            f"  Promotion eligible: {len(self.promotion_eligible_trials)}",
            f"  Duration: {self.total_duration_seconds:.1f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job": self.job.summary(),
            "champion_trial_id": self.champion_trial.trial_id if self.champion_trial else None,
            "champion_backbone": self.champion_trial.backbone if self.champion_trial else None,
            "best_score": self.job.best_score,
            "promotion_eligible_trials": self.promotion_eligible_trials,
            "total_duration_seconds": self.total_duration_seconds,
            "report": self.report,
        }


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------


@dataclass
class AutoMLConfig:
    """Configuration for the AutoML engine.

    All fields are settable from YAML — no code changes needed to adjust
    the search space, strategy, objectives, pruner, or budgets.
    """

    name: str = "automl_search"
    strategy: str = "random"
    n_trials: int = 20
    seed: Optional[int] = None

    objective: str = "composite_research_score"
    objectives: List[str] = field(default_factory=list)

    backbones: List[str] = field(default_factory=list)
    search_space_name: str = "signalium_standard"
    custom_search_space: Optional[Dict[str, Any]] = None

    pruner: str = "median"
    pruner_config: Dict[str, Any] = field(default_factory=dict)

    budget_type: str = "trials"
    budget_seconds: Optional[float] = None
    target_score: Optional[float] = None

    max_workers: int = 1
    gpu_memory_gb: float = 0.0
    max_ram_gb: float = 32.0
    max_retries: int = 2
    checkpoint_frequency: int = 5

    db_path: str = "experiments/automl/automl.db"
    export_dir: str = "experiments/automl/reports"

    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    git_commit: str = ""
    hardware: str = "CPU"

    promotion_thresholds: Dict[str, Any] = field(default_factory=dict)
    auto_promote: bool = False
    require_validation: bool = True

    strategy_config: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AutoMLConfig":
        cfg = AutoMLConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    @staticmethod
    def from_yaml(path: str) -> "AutoMLConfig":
        try:
            import yaml

            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return AutoMLConfig.from_dict(data.get("automl", data))
        except ImportError:
            raise ImportError("PyYAML is required for YAML config loading.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "n_trials": self.n_trials,
            "seed": self.seed,
            "objective": self.objective,
            "objectives": self.objectives,
            "backbones": self.backbones,
            "search_space_name": self.search_space_name,
            "pruner": self.pruner,
            "budget_type": self.budget_type,
            "max_workers": self.max_workers,
            "gpu_memory_gb": self.gpu_memory_gb,
            "max_retries": self.max_retries,
            "db_path": self.db_path,
            "export_dir": self.export_dir,
            "auto_promote": self.auto_promote,
            "require_validation": self.require_validation,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AutoMLEngine:
    """Institutional AutoML and Research Optimization Engine.

    Orchestrates the full AutoML pipeline: search, validation, ranking,
    leaderboard integration, and promotion.

    Args:
        config: AutoMLConfig (or dict/YAML path).
        search_space: Custom SearchSpace (overrides config).
        pruner: Custom BasePruner (overrides config).
        resource_manager: Custom ResourceManager (overrides config).
        scorer: Custom ObjectiveScorer (overrides config).
        database: Custom AutoMLDatabase (overrides config).
    """

    def __init__(
        self,
        config: Optional[AutoMLConfig] = None,
        search_space: Optional[SearchSpace] = None,
        pruner: Optional[BasePruner] = None,
        resource_manager: Optional[ResourceManager] = None,
        scorer: Optional[ObjectiveScorer] = None,
        database: Optional[AutoMLDatabase] = None,
    ) -> None:
        self.config = config or AutoMLConfig()
        self.search_space = search_space or self._build_space()
        self.pruner = pruner or self._build_pruner()
        self.resource_manager = resource_manager or self._build_resource_manager()
        self.scorer = scorer or build_research_scorer()
        self.db = database or AutoMLDatabase(self.config.db_path)
        self.backbone_catalog = BackboneCatalog()
        self.ranker = ExperimentRanker(self.scorer)
        self.viz = AutoMLVisualizationBuilder()
        self.api = AutoMLAPI(self.db, self.ranker, self.scorer, self.viz)

    @staticmethod
    def from_yaml(yaml_path: str) -> "AutoMLEngine":
        """Instantiate engine from a YAML configuration file."""
        config = AutoMLConfig.from_yaml(yaml_path)
        return AutoMLEngine(config=config)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AutoMLEngine":
        """Instantiate engine from a config dict."""
        config = AutoMLConfig.from_dict(d)
        return AutoMLEngine(config=config)

    # -----------------------------------------------------------------------
    # Internal builders
    # -----------------------------------------------------------------------

    def _build_space(self) -> SearchSpace:
        cfg = self.config
        if cfg.custom_search_space:
            return SearchSpace.from_dict(cfg.custom_search_space)
        return build_standard_space()

    def _build_pruner(self) -> BasePruner:
        cfg = self.config
        pruner_cfg = {"type": cfg.pruner, **cfg.pruner_config}
        return build_pruner(pruner_cfg)

    def _build_resource_manager(self) -> ResourceManager:
        cfg = self.config
        return ResourceManager(
            max_workers=cfg.max_workers,
            gpu_memory_gb=cfg.gpu_memory_gb,
            max_ram_gb=cfg.max_ram_gb,
        )

    def _build_strategy(self) -> BaseStrategy:
        cfg = self.config
        strategy_cfg = {
            "type": cfg.strategy,
            "n_trials": cfg.n_trials,
            "seed": cfg.seed,
            **cfg.strategy_config,
        }
        return build_strategy(strategy_cfg, self.search_space)

    def _build_job(self) -> SearchJob:
        cfg = self.config
        return SearchJob(
            name=cfg.name,
            strategy=cfg.strategy,
            n_trials=cfg.n_trials,
            objective=cfg.objective,
            objectives=cfg.objectives or [cfg.objective],
            search_space_name=cfg.search_space_name,
            budget_type=cfg.budget_type,
            budget_value=float(cfg.budget_seconds or cfg.n_trials),
            pruner_type=cfg.pruner,
            parallel_workers=cfg.max_workers,
            max_retries=cfg.max_retries,
        )

    # -----------------------------------------------------------------------
    # Main run
    # -----------------------------------------------------------------------

    def run(
        self,
        train_fn: Optional[Callable[[Dict[str, Any], BackboneSpec], Dict[str, Any]]] = None,
        model_id: str = "automl_model",
        validation_engine: Optional[Any] = None,
    ) -> AutoMLResult:
        """Execute the full AutoML search pipeline.

        Args:
            train_fn: Callable(config, backbone_spec) → metrics dict.
                When None, a mock evaluator is used (for testing/dry runs).
            model_id: Model identifier string.
            validation_engine: Optional Phase 15 InstitutionalValidationEngine.
                When provided, each trial's output is passed through Phase 15
                validation. When None, metrics from train_fn are used directly.

        Returns:
            AutoMLResult with champion, ranking, and report.
        """
        wall_start = time.monotonic()
        cfg = self.config

        logger.info(
            f"AutoML Engine starting: {cfg.name} [{cfg.strategy}] " f"n_trials={cfg.n_trials}"
        )

        job = self._build_job()
        job.start()
        self.db.save_job(job)

        strategy = self._build_strategy()
        trial_history: Dict[str, List[tuple]] = {}
        records: List[ExperimentRecord] = []

        budget_start = time.monotonic()

        for trial_num in range(cfg.n_trials):
            if strategy.is_exhausted():
                logger.info(f"Strategy exhausted after {trial_num} trials.")
                break

            if not self._within_budget(budget_start, trial_num, job):
                logger.info(f"Budget reached after {trial_num} trials.")
                break

            trial = Trial(
                job_id=job.job_id,
                random_seed=cfg.seed,
                dataset_version=cfg.dataset_version,
                feature_version=cfg.feature_version,
                label_version=cfg.label_version,
                git_commit=cfg.git_commit,
                hardware=cfg.hardware,
            )

            try:
                suggestion = strategy.suggest(trial.trial_id)
                trial.config = suggestion.config
                backbone_name = trial.config.get("backbone", "lstm")
                if cfg.backbones and backbone_name not in cfg.backbones:
                    backbone_name = cfg.backbones[0]
                trial.backbone = backbone_name
                backbone_spec = self.backbone_catalog.from_config(
                    {
                        "name": backbone_name,
                        **trial.config,
                    }
                )

                trial.start()
                logger.info(
                    f"Trial {trial_num + 1}/{cfg.n_trials} "
                    f"[{trial.trial_id[:8]}] backbone={backbone_name}"
                )

                train_start = time.monotonic()
                metrics = self._evaluate_trial(
                    trial, backbone_spec, train_fn, validation_engine, model_id
                )
                train_duration = time.monotonic() - train_start

                if metrics is None:
                    raise RuntimeError("Evaluation returned None")

                composite = compute_composite_research_score(metrics)
                metrics["composite_research_score"] = composite
                primary_score = metrics.get(cfg.objective, composite)
                if primary_score is None:
                    primary_score = composite

                trial.add_intermediate(trial_num, float(primary_score))
                trial_history[trial.trial_id] = trial.intermediate_scores

                pruning_decision = self.pruner.should_prune(
                    trial_id=trial.trial_id,
                    step=trial_num,
                    score=float(primary_score),
                    history={k: v for k, v in trial_history.items() if k != trial.trial_id},
                )

                if pruning_decision.prune:
                    trial.prune(trial_num, pruning_decision.reason)
                    self.db.save_pruning_decision(
                        trial_id=trial.trial_id,
                        job_id=job.job_id,
                        step=trial_num,
                        score=float(primary_score),
                        threshold=pruning_decision.threshold,
                        prune=True,
                        reason=pruning_decision.reason,
                    )
                else:
                    trial.complete(
                        score=float(primary_score),
                        metrics=metrics,
                        training_duration=train_duration,
                    )
                    strategy.report(trial.trial_id, float(primary_score))
                    trial.promotion_eligible = self._check_promotion(
                        metrics, cfg.promotion_thresholds
                    )
                    job.add_trial(trial)

                snap = self.resource_manager.record_usage(
                    trial_id=trial.trial_id,
                    elapsed_seconds=train_duration,
                )
                trial.resource_usage = snap.to_dict()

            except StopIteration:
                logger.info("Search strategy exhausted.")
                break
            except Exception as e:
                logger.error(f"Trial {trial.trial_id[:8]} failed: {e}")
                trial.fail(str(e), retry=trial.retry_count < cfg.max_retries)
                if trial.status == TrialStatus.RETRYING:
                    trial.retry_count += 1
                job.add_trial(trial)

            record = ExperimentRecord.from_trial(
                trial,
                job,
                validation_results=trial.metrics,
            )
            record.composite_research_score = compute_composite_research_score(trial.metrics)
            record.per_objective_scores = self.scorer.per_objective_scores(trial.metrics)
            self.db.save_record(record)
            records.append(record)

            if trial_num > 0 and trial_num % cfg.checkpoint_frequency == 0:
                self.db.save_job(job)
                logger.debug(f"Checkpoint saved at trial {trial_num}")

        job.complete()
        self.db.save_job(job)

        ranking_result = self._compute_ranking(records)
        self._apply_ranks(records, ranking_result)

        promotion_eligible = [r.trial_id for r in records if r.promotion_eligible]

        if cfg.auto_promote and promotion_eligible:
            self._auto_promote(job, promotion_eligible)

        all_record_dicts = [r.to_dict() for r in records]
        dashboard = self.viz.full_dashboard(all_record_dicts)

        report = self._build_report(job, ranking_result, records, dashboard)

        wall_duration = time.monotonic() - wall_start
        logger.info(
            f"AutoML search complete: best={job.best_score}, "
            f"champion={job.champion_trial_id}, "
            f"duration={wall_duration:.1f}s"
        )

        return AutoMLResult(
            job=job,
            champion_trial=job.champion_trial,
            champion_record=self.db.best_record(job.job_id),
            ranking=ranking_result,
            dashboard=dashboard,
            promotion_eligible_trials=promotion_eligible,
            report=report,
            total_duration_seconds=wall_duration,
        )

    # -----------------------------------------------------------------------
    # Trial evaluation
    # -----------------------------------------------------------------------

    def _evaluate_trial(
        self,
        trial: Trial,
        backbone_spec: BackboneSpec,
        train_fn: Optional[Callable],
        validation_engine: Optional[Any],
        model_id: str,
    ) -> Dict[str, Any]:
        """Run training + optional Phase 15 validation for one trial."""
        if train_fn is not None:
            metrics = train_fn(trial.config, backbone_spec)
        else:
            metrics = self._mock_evaluate(trial.config, backbone_spec)

        if not isinstance(metrics, dict):
            metrics = {"validation_score": float(metrics)}

        if validation_engine is not None:
            try:
                val_result = validation_engine.evaluate_metrics(metrics)
                if isinstance(val_result, dict):
                    metrics.update(val_result)
            except Exception as e:
                logger.warning(
                    f"Phase 15 validation failed for trial " f"{trial.trial_id[:8]}: {e}"
                )

        return metrics

    def _mock_evaluate(self, config: Dict[str, Any], backbone: BackboneSpec) -> Dict[str, Any]:
        """Deterministic mock evaluator for testing/dry runs.

        Returns plausible but random-seeded metrics based on config hash.
        """
        import hashlib, math

        seed = int(hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:8], 16)
        rng_val = (seed % 1000) / 1000.0

        lr = float(config.get("learning_rate", 1e-3))
        dropout = float(config.get("dropout", 0.1))
        hidden = int(config.get("hidden_size", 128))

        base = 0.5 + 0.3 * rng_val
        lr_factor = 1.0 - abs(math.log10(lr) + 3) * 0.05
        size_factor = min(hidden / 512, 1.0) * 0.1
        dropout_factor = 1.0 - dropout * 0.3

        validation_score = min(0.95, max(0.30, base * lr_factor * dropout_factor + size_factor))
        sharpe = (validation_score - 0.5) * 4.0
        pf = 1.0 + max(0.0, sharpe * 0.5)
        drawdown = max(0.01, 0.30 - validation_score * 0.2)

        return {
            "validation_score": round(validation_score, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sharpe * 1.2, 4),
            "profit_factor": round(pf, 4),
            "max_drawdown": round(drawdown, 4),
            "win_rate": round(0.4 + validation_score * 0.2, 4),
            "accuracy": round(validation_score, 4),
            "f1_score": round(validation_score * 0.95, 4),
            "expected_calibration_error": round(max(0.01, 0.2 - validation_score * 0.15), 4),
            "n_trades": int(rng_val * 500 + 50),
            "backbone": backbone.name,
        }

    # -----------------------------------------------------------------------
    # Budget check
    # -----------------------------------------------------------------------

    def _within_budget(self, budget_start: float, trial_num: int, job: SearchJob) -> bool:
        cfg = self.config
        if cfg.budget_type == "time" and cfg.budget_seconds:
            return (time.monotonic() - budget_start) < cfg.budget_seconds
        if cfg.budget_type == "metric" and cfg.target_score:
            return job.best_score is None or job.best_score < cfg.target_score
        return True

    # -----------------------------------------------------------------------
    # Promotion eligibility
    # -----------------------------------------------------------------------

    def _check_promotion(self, metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> bool:
        """Check if metrics meet all configured promotion thresholds."""
        if not thresholds:
            defaults = {
                "min_validation_score": 0.55,
                "min_sharpe_ratio": 0.5,
                "max_drawdown": 0.35,
                "min_profit_factor": 1.0,
            }
            thresholds = defaults

        for key, threshold in thresholds.items():
            if key.startswith("min_"):
                metric_name = key[4:]
                val = metrics.get(metric_name)
                if val is not None and float(val) < float(threshold):
                    return False
            elif key.startswith("max_"):
                metric_name = key[4:]
                val = metrics.get(metric_name)
                if val is not None and float(val) > float(threshold):
                    return False
        return True

    # -----------------------------------------------------------------------
    # Auto promotion
    # -----------------------------------------------------------------------

    def _auto_promote(self, job: SearchJob, eligible_ids: List[str]) -> None:
        """Automatically promote the best eligible trial."""
        best_eligible = None
        best_score = -1e9
        for trial in job.trials:
            if trial.trial_id in eligible_ids:
                if trial.score is not None and trial.score > best_score:
                    best_score = trial.score
                    best_eligible = trial
        if best_eligible:
            self.db.save_promotion_decision(
                trial_id=best_eligible.trial_id,
                job_id=job.job_id,
                model_version=best_eligible.model_version or "automl",
                is_eligible=True,
                auto_approved=True,
                manually_approved=False,
                champion_status=True,
                rollback_eligible=True,
                rules=[],
                notes=f"Auto-promoted by AutoML engine. Score: {best_score:.4f}",
            )
            logger.info(
                f"Auto-promoted trial {best_eligible.trial_id[:8]} " f"(score={best_score:.4f})"
            )

    # -----------------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------------

    def _compute_ranking(self, records: List[ExperimentRecord]) -> RankingResult:
        record_dicts = [r.to_dict() for r in records]
        return self.ranker.full_ranking(record_dicts)

    def _apply_ranks(self, records: List[ExperimentRecord], ranking: RankingResult) -> None:
        rank_map = {e.trial_id: e.rank for e in ranking.overall}
        for r in records:
            rank = rank_map.get(r.trial_id)
            if rank is not None:
                r.overall_rank = rank
                self.db.update_record_rank(r.record_id, rank)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    def _build_report(
        self,
        job: SearchJob,
        ranking: RankingResult,
        records: List[ExperimentRecord],
        dashboard: Dict[str, Any],
    ) -> Dict[str, Any]:
        completed = [r for r in records if r.status == TrialStatus.COMPLETED]
        backbone_summary = self.db.backbone_summary(job.job_id)
        champion = ranking.champion

        return {
            "report_type": "automl_platform",
            "job_id": job.job_id,
            "job_name": job.name,
            "strategy": job.strategy,
            "status": job.status,
            "n_trials_total": len(records),
            "n_completed": len(completed),
            "n_pruned": job.n_pruned,
            "n_failed": job.n_failed,
            "best_score": job.best_score,
            "champion_trial_id": job.champion_trial_id,
            "champion_backbone": (job.champion_trial.backbone if job.champion_trial else None),
            "champion_metrics": (job.champion_trial.metrics if job.champion_trial else {}),
            "top_5_ranking": [e.to_dict() for e in ranking.overall[:5]],
            "backbone_summary": backbone_summary,
            "resource_usage": self.resource_manager.usage_report(),
            "promotion_decisions": self.db.get_promotion_history(job_id=job.job_id, limit=20),
            "db_stats": self.db.stats(),
            "search_convergence": dashboard.get("search_convergence", {}),
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
