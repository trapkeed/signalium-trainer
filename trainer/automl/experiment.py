"""Phase 16 — AutoML Experiment and Trial Data Structures.

Defines the core data models for the AutoML platform:

  - Trial          : a single hyperparameter evaluation run
  - SearchJob      : a complete search campaign (one or many trials)
  - ExperimentRecord : a rich record of one trial's execution and results

Tracks everything required by the Phase 16 spec:
  Experiment ID, Parent Experiment, Configuration, Hyperparameters,
  Dataset Version, Feature Version, Label Version, Backbone, Random Seed,
  Training Duration, Validation Results, Promotion Eligibility, Git Commit,
  Model Version, Hardware Used.

Usage::

    job = SearchJob(name="sharpe_sweep", strategy="bayesian", n_trials=30)
    trial = Trial(job_id=job.job_id, config={"learning_rate": 1e-3})
    job.add_trial(trial)
    trial.complete(score=0.82, metrics={"sharpe_ratio": 1.4})
    record = trial.to_record()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Trial status enum
# ---------------------------------------------------------------------------


class TrialStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"
    RETRYING = "retrying"


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """A single hyperparameter evaluation run.

    Attributes:
        trial_id: Unique UUID for this trial.
        job_id: Parent search job UUID.
        experiment_id: Optional external experiment UUID (e.g. MLflow).
        parent_trial_id: UUID of parent trial (population-based training).
        config: Hyperparameter configuration dict.
        backbone: Selected backbone name.
        random_seed: Random seed used.
        dataset_version: Dataset version tag.
        feature_version: Feature version tag.
        label_version: Label version tag.
        git_commit: Git commit hash at trial start.
        model_version: Generated model version string.
        hardware: Hardware descriptor (e.g. "A100-40GB", "CPU").
        status: TrialStatus string.
        score: Primary objective score (higher = better).
        metrics: Full metrics dict from validation.
        pruning_step: Step at which trial was pruned (None = not pruned).
        pruning_reason: Reason for pruning.
        start_time: UTC ISO-8601 start timestamp.
        end_time: UTC ISO-8601 end timestamp.
        training_duration_seconds: Elapsed training time.
        validation_duration_seconds: Elapsed validation time.
        error_message: Error message if trial failed.
        retry_count: Number of retries attempted.
        resource_usage: Dict with CPU/GPU/memory snapshots.
        intermediate_scores: List of (step, score) for pruning.
        promotion_eligible: Whether trial passed promotion criteria.
        leaderboard_rank: Rank assigned after leaderboard update.
        notes: Free-text notes.
    """

    trial_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""
    experiment_id: str = ""
    parent_trial_id: str | None = None

    config: dict[str, Any] = field(default_factory=dict)
    backbone: str = ""
    random_seed: int | None = None
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    git_commit: str = ""
    model_version: str = ""
    hardware: str = ""

    status: str = TrialStatus.PENDING
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    pruning_step: int | None = None
    pruning_reason: str = ""

    start_time: str = ""
    end_time: str = ""
    training_duration_seconds: float = 0.0
    validation_duration_seconds: float = 0.0
    error_message: str = ""
    retry_count: int = 0

    resource_usage: dict[str, Any] = field(default_factory=dict)
    intermediate_scores: list[tuple] = field(default_factory=list)

    promotion_eligible: bool = False
    leaderboard_rank: int | None = None
    notes: str = ""

    def start(self) -> None:
        """Mark trial as running."""
        self.status = TrialStatus.RUNNING
        self.start_time = datetime.now(UTC).isoformat()

    def complete(
        self,
        score: float,
        metrics: dict[str, Any] | None = None,
        training_duration: float = 0.0,
        validation_duration: float = 0.0,
    ) -> None:
        """Mark trial as completed with its results."""
        self.status = TrialStatus.COMPLETED
        self.score = score
        self.metrics = metrics or {}
        self.training_duration_seconds = training_duration
        self.validation_duration_seconds = validation_duration
        self.end_time = datetime.now(UTC).isoformat()

    def prune(self, step: int, reason: str = "") -> None:
        """Mark trial as pruned."""
        self.status = TrialStatus.PRUNED
        self.pruning_step = step
        self.pruning_reason = reason
        self.end_time = datetime.now(UTC).isoformat()
        logger.info(f"Trial {self.trial_id} pruned at step {step}: {reason}")

    def fail(self, error: str, retry: bool = False) -> None:
        """Mark trial as failed."""
        self.status = TrialStatus.RETRYING if retry else TrialStatus.FAILED
        self.error_message = error
        self.end_time = datetime.now(UTC).isoformat()
        logger.warning(f"Trial {self.trial_id} {'retrying' if retry else 'failed'}: {error}")

    def add_intermediate(self, step: int, score: float) -> None:
        """Record an intermediate score for pruning decisions."""
        self.intermediate_scores.append((step, score))

    @property
    def duration_seconds(self) -> float:
        return self.training_duration_seconds + self.validation_duration_seconds

    @property
    def is_terminal(self) -> bool:
        return self.status in (TrialStatus.COMPLETED, TrialStatus.PRUNED, TrialStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "job_id": self.job_id,
            "experiment_id": self.experiment_id,
            "parent_trial_id": self.parent_trial_id,
            "config": self.config,
            "backbone": self.backbone,
            "random_seed": self.random_seed,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "git_commit": self.git_commit,
            "model_version": self.model_version,
            "hardware": self.hardware,
            "status": self.status,
            "score": self.score,
            "metrics": self.metrics,
            "pruning_step": self.pruning_step,
            "pruning_reason": self.pruning_reason,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "training_duration_seconds": self.training_duration_seconds,
            "validation_duration_seconds": self.validation_duration_seconds,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "resource_usage": self.resource_usage,
            "intermediate_scores": list(self.intermediate_scores),
            "promotion_eligible": self.promotion_eligible,
            "leaderboard_rank": self.leaderboard_rank,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Trial:
        t = Trial(**{k: v for k, v in d.items() if k in Trial.__dataclass_fields__})
        t.intermediate_scores = [tuple(x) for x in d.get("intermediate_scores", [])]
        return t


# ---------------------------------------------------------------------------
# Search job
# ---------------------------------------------------------------------------


@dataclass
class SearchJob:
    """A complete hyperparameter search campaign.

    Attributes:
        job_id: Unique UUID for this search job.
        name: Human-readable name.
        strategy: Search strategy name ("random", "bayesian", etc.).
        n_trials: Target number of trials.
        objective: Primary objective name.
        objectives: List of all objective names.
        search_space_name: Name of the search space used.
        status: "pending" | "running" | "completed" | "failed" | "paused".
        budget_type: "trials" | "time" | "metric" | "budget" | "scheduled".
        budget_value: Budget limit (n trials, seconds, target score, etc.).
        pruner_type: Pruner strategy name.
        parallel_workers: Number of concurrent workers.
        checkpoint_frequency: Save checkpoint every N trials.
        max_retries: Max retries per failing trial.
        created_at: UTC ISO-8601 creation timestamp.
        started_at: UTC ISO-8601 start timestamp.
        completed_at: UTC ISO-8601 completion timestamp.
        trials: List of Trial objects.
        champion_trial_id: Best trial ID so far.
        best_score: Best score achieved.
        metadata: Arbitrary extra metadata.
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "search_job"
    strategy: str = "random"
    n_trials: int = 20
    objective: str = "composite_research_score"
    objectives: list[str] = field(default_factory=list)
    search_space_name: str = "signalium_standard"
    status: str = "pending"
    budget_type: str = "trials"
    budget_value: float = 20.0
    pruner_type: str = "median"
    parallel_workers: int = 1
    checkpoint_frequency: int = 5
    max_retries: int = 2
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str = ""
    completed_at: str = ""
    trials: list[Trial] = field(default_factory=list)
    champion_trial_id: str | None = None
    best_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_trial(self, trial: Trial) -> None:
        trial.job_id = self.job_id
        self.trials.append(trial)
        if trial.score is not None and (self.best_score is None or trial.score > self.best_score):
            self.best_score = trial.score
            self.champion_trial_id = trial.trial_id

    def start(self) -> None:
        self.status = "running"
        self.started_at = datetime.now(UTC).isoformat()

    def complete(self) -> None:
        self.status = "completed"
        self.completed_at = datetime.now(UTC).isoformat()

    def fail(self) -> None:
        self.status = "failed"
        self.completed_at = datetime.now(UTC).isoformat()

    def pause(self) -> None:
        self.status = "paused"

    @property
    def n_completed(self) -> int:
        return sum(1 for t in self.trials if t.status == TrialStatus.COMPLETED)

    @property
    def n_pruned(self) -> int:
        return sum(1 for t in self.trials if t.status == TrialStatus.PRUNED)

    @property
    def n_failed(self) -> int:
        return sum(1 for t in self.trials if t.status == TrialStatus.FAILED)

    @property
    def completed_trials(self) -> list[Trial]:
        return [t for t in self.trials if t.status == TrialStatus.COMPLETED]

    @property
    def champion_trial(self) -> Trial | None:
        if self.champion_trial_id is None:
            return None
        for t in self.trials:
            if t.trial_id == self.champion_trial_id:
                return t
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "strategy": self.strategy,
            "status": self.status,
            "n_trials": self.n_trials,
            "n_completed": self.n_completed,
            "n_pruned": self.n_pruned,
            "n_failed": self.n_failed,
            "best_score": self.best_score,
            "champion_trial_id": self.champion_trial_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "objective": self.objective,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.summary()
        d.update(
            {
                "objectives": self.objectives,
                "search_space_name": self.search_space_name,
                "budget_type": self.budget_type,
                "budget_value": self.budget_value,
                "pruner_type": self.pruner_type,
                "parallel_workers": self.parallel_workers,
                "checkpoint_frequency": self.checkpoint_frequency,
                "max_retries": self.max_retries,
                "trials": [t.to_dict() for t in self.trials],
                "metadata": self.metadata,
            }
        )
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SearchJob:
        trials_data = d.pop("trials", [])
        job = SearchJob(
            **{k: v for k, v in d.items() if k in SearchJob.__dataclass_fields__ and k != "trials"}
        )
        job.trials = [Trial.from_dict(td) for td in trials_data]
        return job


# ---------------------------------------------------------------------------
# Experiment record
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    """Full record of one AutoML trial for database persistence.

    Combines trial data with derived analysis fields used for
    ranking, leaderboard, and reporting.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trial_id: str = ""
    job_id: str = ""
    job_name: str = ""
    strategy: str = ""

    config: dict[str, Any] = field(default_factory=dict)
    backbone: str = ""
    random_seed: int | None = None
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    git_commit: str = ""
    model_version: str = ""
    hardware: str = ""

    status: str = TrialStatus.PENDING
    primary_score: float | None = None
    per_objective_scores: dict[str, float] = field(default_factory=dict)
    composite_research_score: float = 0.0

    validation_results: dict[str, Any] = field(default_factory=dict)
    promotion_eligible: bool = False
    promotion_decision: dict[str, Any] = field(default_factory=dict)

    overall_rank: int | None = None
    per_instrument_ranks: dict[str, int] = field(default_factory=dict)
    per_timeframe_ranks: dict[str, int] = field(default_factory=dict)
    per_regime_ranks: dict[str, int] = field(default_factory=dict)

    training_duration_seconds: float = 0.0
    validation_duration_seconds: float = 0.0
    resource_usage: dict[str, Any] = field(default_factory=dict)
    intermediate_scores: list[tuple] = field(default_factory=list)

    pruned: bool = False
    pruning_step: int | None = None
    pruning_reason: str = ""
    error_message: str = ""
    retry_count: int = 0

    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""

    @staticmethod
    def from_trial(
        trial: Trial, job: SearchJob, validation_results: dict[str, Any] | None = None
    ) -> ExperimentRecord:
        return ExperimentRecord(
            trial_id=trial.trial_id,
            job_id=trial.job_id or job.job_id,
            job_name=job.name,
            strategy=job.strategy,
            config=trial.config,
            backbone=trial.backbone,
            random_seed=trial.random_seed,
            dataset_version=trial.dataset_version,
            feature_version=trial.feature_version,
            label_version=trial.label_version,
            git_commit=trial.git_commit,
            model_version=trial.model_version,
            hardware=trial.hardware,
            status=trial.status,
            primary_score=trial.score,
            validation_results=validation_results or trial.metrics,
            promotion_eligible=trial.promotion_eligible,
            training_duration_seconds=trial.training_duration_seconds,
            validation_duration_seconds=trial.validation_duration_seconds,
            resource_usage=trial.resource_usage,
            intermediate_scores=list(trial.intermediate_scores),
            pruned=trial.status == TrialStatus.PRUNED,
            pruning_step=trial.pruning_step,
            pruning_reason=trial.pruning_reason,
            error_message=trial.error_message,
            retry_count=trial.retry_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "trial_id": self.trial_id,
            "job_id": self.job_id,
            "job_name": self.job_name,
            "strategy": self.strategy,
            "config": self.config,
            "backbone": self.backbone,
            "random_seed": self.random_seed,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "git_commit": self.git_commit,
            "model_version": self.model_version,
            "hardware": self.hardware,
            "status": self.status,
            "primary_score": self.primary_score,
            "per_objective_scores": self.per_objective_scores,
            "composite_research_score": self.composite_research_score,
            "validation_results": self.validation_results,
            "promotion_eligible": self.promotion_eligible,
            "promotion_decision": self.promotion_decision,
            "overall_rank": self.overall_rank,
            "per_instrument_ranks": self.per_instrument_ranks,
            "per_timeframe_ranks": self.per_timeframe_ranks,
            "per_regime_ranks": self.per_regime_ranks,
            "training_duration_seconds": self.training_duration_seconds,
            "validation_duration_seconds": self.validation_duration_seconds,
            "resource_usage": self.resource_usage,
            "intermediate_scores": list(self.intermediate_scores),
            "pruned": self.pruned,
            "pruning_step": self.pruning_step,
            "pruning_reason": self.pruning_reason,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "notes": self.notes,
        }
