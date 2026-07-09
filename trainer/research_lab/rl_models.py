"""Phase 28.5 — Research Lab data models.

All models in this file are internal to the trainer's Research Lab.
No model is exposed to retail users. All access is via admin APIs only.

Responsibilities covered:
 - Experiment tracking     (ExperimentRecord)
 - Backtesting             (embedded in ExperimentRecord / backtest dicts)
 - Validation              (DeploymentGateResult)
 - Dataset generation      (ResearchDataset, DatasetGenerationJob)
 - Failure analysis        (FailureEvent)
 - Version comparisons     (ModelComparisonResult)
 - Promotion approval      (DeploymentStatus, approve_deployment flow)
 - Rollback approval       (RollbackRequest, RollbackStatus)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ── Enumerations ──────────────────────────────────────────────────────────────


class ExperimentStatus(str, Enum):
    PROPOSED = "PROPOSED"   # auto-generated, awaiting admin approval
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class DeploymentStatus(str, Enum):
    NOT_DEPLOYED = "NOT_DEPLOYED"
    DRAFT = "DRAFT"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    SHADOW = "SHADOW"
    CANDIDATE = "CANDIDATE"
    CANARY = "CANARY"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    ROLLED_BACK = "ROLLED_BACK"


class FailureCategory(str, Enum):
    LARGE_LOSS = "LARGE_LOSS"
    UNEXPECTED_DRAWDOWN = "UNEXPECTED_DRAWDOWN"
    REGIME_FAILURE = "REGIME_FAILURE"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    MARKET_CONDITION_FAILURE = "MARKET_CONDITION_FAILURE"


class FailureSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RegimeType(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CRISIS = "CRISIS"
    RECOVERY = "RECOVERY"


class SearchStrategy(str, Enum):
    GRID = "GRID"
    RANDOM = "RANDOM"
    BAYESIAN = "BAYESIAN"
    EVOLUTIONARY = "EVOLUTIONARY"


class ShadowTestStatus(str, Enum):
    RUNNING = "RUNNING"
    CONCLUDED = "CONCLUDED"
    FAILED = "FAILED"


class ShadowTestRecommendation(str, Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    EXTEND = "EXTEND"  # Collect more data before deciding
    INCONCLUSIVE = "INCONCLUSIVE"


class RollbackStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"


class LifecycleAction(str, Enum):
    """The set of QDE version lifecycle actions that Research Lab, and only
    Research Lab, is authorized to approve. Any service that mutates a QDE
    version's stage along one of these actions must first obtain a matching
    APPROVED, unconsumed LifecycleAuthorization from Research Lab."""

    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    ARCHIVE = "ARCHIVE"
    RETIRE = "RETIRE"


class LifecycleAuthorizationStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CONSUMED = "CONSUMED"


class DatasetGenerationStatus(str, Enum):
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    FAILED = "FAILED"


# ── Experiment ────────────────────────────────────────────────────────────────


@dataclass
class ExperimentRecord:
    """Full experiment tracking record with all required fields."""

    experiment_id: str
    name: str
    qde_version: str
    training_dataset: str
    status: ExperimentStatus
    created_at: float
    updated_at: float

    # Timing
    training_started_at: float | None = None
    training_completed_at: float | None = None
    training_duration_s: float | None = None

    # Validation metrics
    validation_metrics: dict[str, float] = field(default_factory=dict)

    # Drawdown metrics
    max_drawdown: float | None = None
    avg_drawdown: float | None = None
    drawdown_duration_s: float | None = None
    recovery_factor: float | None = None

    # Expected value metrics
    expected_value: float | None = None
    expected_value_per_trade: float | None = None

    # Profit metrics
    profit_factor: float | None = None
    win_rate: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    decision_quality: float | None = None
    capital_efficiency: float | None = None

    # Deployment
    deployment_status: DeploymentStatus = DeploymentStatus.NOT_DEPLOYED
    deployment_notes: str = ""

    # Tags / metadata
    tags: list[str] = field(default_factory=list)
    researcher: str = ""
    notes: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @classmethod
    def new(
        cls,
        name: str,
        qde_version: str,
        training_dataset: str,
        researcher: str = "",
        tags: list[str] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        notes: str = "",
    ) -> ExperimentRecord:
        now = time.time()
        return cls(
            experiment_id=f"exp-{str(uuid.uuid4())[:12]}",
            name=name,
            qde_version=qde_version,
            training_dataset=training_dataset,
            status=ExperimentStatus.PENDING,
            created_at=now,
            updated_at=now,
            researcher=researcher,
            tags=tags or [],
            hyperparameters=hyperparameters or {},
            notes=notes,
        )

    def start_training(self) -> None:
        self.status = ExperimentStatus.RUNNING
        self.training_started_at = time.time()
        self.updated_at = self.training_started_at

    def complete(self, metrics: dict[str, float], notes: str = "") -> None:
        now = time.time()
        self.status = ExperimentStatus.COMPLETED
        self.training_completed_at = now
        self.updated_at = now
        if self.training_started_at:
            self.training_duration_s = now - self.training_started_at
        self.validation_metrics.update(metrics)
        for attr in (
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "avg_drawdown",
            "profit_factor",
            "win_rate",
            "expected_value",
            "expected_value_per_trade",
            "decision_quality",
            "capital_efficiency",
            "recovery_factor",
        ):
            if attr in metrics:
                setattr(self, attr, metrics[attr])
        if notes:
            self.notes = notes

    def fail(self, error: str) -> None:
        self.status = ExperimentStatus.FAILED
        self.error = error
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "qde_version": self.qde_version,
            "training_dataset": self.training_dataset,
            "status": self.status.value,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "training_started_at": (
                round(self.training_started_at, 3) if self.training_started_at else None
            ),
            "training_completed_at": (
                round(self.training_completed_at, 3) if self.training_completed_at else None
            ),
            "training_duration_s": (
                round(self.training_duration_s, 1) if self.training_duration_s else None
            ),
            "validation_metrics": {k: round(v, 6) for k, v in self.validation_metrics.items()},
            "max_drawdown": round(self.max_drawdown, 6) if self.max_drawdown is not None else None,
            "avg_drawdown": round(self.avg_drawdown, 6) if self.avg_drawdown is not None else None,
            "recovery_factor": (
                round(self.recovery_factor, 4) if self.recovery_factor is not None else None
            ),
            "expected_value": (
                round(self.expected_value, 6) if self.expected_value is not None else None
            ),
            "expected_value_per_trade": (
                round(self.expected_value_per_trade, 6)
                if self.expected_value_per_trade is not None
                else None
            ),
            "profit_factor": (
                round(self.profit_factor, 4) if self.profit_factor is not None else None
            ),
            "win_rate": round(self.win_rate, 4) if self.win_rate is not None else None,
            "sharpe_ratio": round(self.sharpe_ratio, 4) if self.sharpe_ratio is not None else None,
            "sortino_ratio": (
                round(self.sortino_ratio, 4) if self.sortino_ratio is not None else None
            ),
            "decision_quality": (
                round(self.decision_quality, 4) if self.decision_quality is not None else None
            ),
            "capital_efficiency": (
                round(self.capital_efficiency, 4) if self.capital_efficiency is not None else None
            ),
            "deployment_status": self.deployment_status.value,
            "deployment_notes": self.deployment_notes,
            "tags": self.tags,
            "researcher": self.researcher,
            "notes": self.notes,
            "hyperparameters": self.hyperparameters,
            "error": self.error,
        }


# ── Dataset ───────────────────────────────────────────────────────────────────


@dataclass
class ResearchDataset:
    """A named, versioned dataset registered for use in experiments."""

    dataset_id: str
    name: str
    version: str
    source: str
    instruments: list[str]
    start_date: str
    end_date: str
    created_at: float
    row_count: int = 0
    feature_count: int = 0
    label_quality_score: float | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    checksum: str = ""

    @classmethod
    def new(
        cls,
        name: str,
        version: str,
        source: str,
        instruments: list[str],
        start_date: str,
        end_date: str,
        row_count: int = 0,
        feature_count: int = 0,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ResearchDataset:
        return cls(
            dataset_id=f"ds-{str(uuid.uuid4())[:12]}",
            name=name,
            version=version,
            source=source,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            created_at=time.time(),
            row_count=row_count,
            feature_count=feature_count,
            tags=tags or [],
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "instruments": self.instruments,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "created_at": round(self.created_at, 3),
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "label_quality_score": (
                round(self.label_quality_score, 4) if self.label_quality_score is not None else None
            ),
            "tags": self.tags,
            "notes": self.notes,
            "checksum": self.checksum,
        }


# ── Feature ───────────────────────────────────────────────────────────────────


@dataclass
class DiscoveredFeature:
    """A feature candidate discovered during research."""

    feature_id: str
    name: str
    category: str
    importance_score: float
    discovered_at: float
    experiments: list[str] = field(default_factory=list)
    status: str = "CANDIDATE"  # CANDIDATE | ACCEPTED | REJECTED
    notes: str = ""
    # S20 auto-discovery fields (optional for backward compatibility)
    formula_description: str = ""
    evidence_strength: float = 0.0      # [0, 1]
    suggested_qde_role: str = ""        # e.g. "confidence_calibration_input"
    sample_count: int = 0

    @classmethod
    def new(
        cls,
        name: str,
        category: str,
        importance_score: float,
        notes: str = "",
        formula_description: str = "",
        evidence_strength: float = 0.0,
        suggested_qde_role: str = "",
        sample_count: int = 0,
    ) -> DiscoveredFeature:
        return cls(
            feature_id=f"feat-{str(uuid.uuid4())[:10]}",
            name=name,
            category=category,
            importance_score=importance_score,
            discovered_at=time.time(),
            notes=notes,
            formula_description=formula_description,
            evidence_strength=evidence_strength,
            suggested_qde_role=suggested_qde_role,
            sample_count=sample_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "category": self.category,
            "importance_score": round(self.importance_score, 6),
            "discovered_at": round(self.discovered_at, 3),
            "experiments": self.experiments,
            "status": self.status,
            "notes": self.notes,
            "formula_description": self.formula_description,
            "evidence_strength": round(self.evidence_strength, 4),
            "suggested_qde_role": self.suggested_qde_role,
            "sample_count": self.sample_count,
        }


# ── Model Comparison ──────────────────────────────────────────────────────────

COMPARISON_METRICS = [
    "sharpe_ratio",
    "profit_factor",
    "max_drawdown",
    "recovery_factor",
    "win_rate",
    "decision_quality",
    "expected_value",
    "capital_efficiency",
]


@dataclass
class ModelComparisonResult:
    """Side-by-side comparison of multiple QDE versions."""

    comparison_id: str
    version_ids: list[str]
    metrics: dict[str, dict[str, float]]
    rankings: dict[str, list[str]]
    overall_winner: str | None
    created_at: float
    notes: str = ""

    @classmethod
    def build(
        cls,
        version_ids: list[str],
        metrics_per_version: dict[str, dict[str, float]],
        notes: str = "",
    ) -> ModelComparisonResult:
        rankings: dict[str, list[str]] = {}
        for metric in COMPARISON_METRICS:
            lower_is_better = metric == "max_drawdown"
            ranked = sorted(
                [v for v in version_ids if metric in metrics_per_version.get(v, {})],
                key=lambda v: metrics_per_version[v].get(metric, 0.0),
                reverse=not lower_is_better,
            )
            rankings[metric] = ranked

        win_count: dict[str, int] = {v: 0 for v in version_ids}
        for ranked_list in rankings.values():
            if ranked_list:
                win_count[ranked_list[0]] = win_count.get(ranked_list[0], 0) + 1
        overall_winner = max(win_count, key=lambda v: win_count[v]) if win_count else None

        return cls(
            comparison_id=f"cmp-{str(uuid.uuid4())[:10]}",
            version_ids=version_ids,
            metrics=metrics_per_version,
            rankings=rankings,
            overall_winner=overall_winner,
            created_at=time.time(),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "version_ids": self.version_ids,
            "metrics": {
                v: {k: round(val, 6) for k, val in m.items()} for v, m in self.metrics.items()
            },
            "rankings": self.rankings,
            "overall_winner": self.overall_winner,
            "created_at": round(self.created_at, 3),
            "notes": self.notes,
        }


# ── Failure Analysis ──────────────────────────────────────────────────────────


@dataclass
class FailureEvent:
    """A recorded failure event for investigation."""

    failure_id: str
    category: FailureCategory
    severity: FailureSeverity
    qde_version: str
    detected_at: float
    description: str
    instrument: str = ""
    regime: str = ""
    loss_amount: float | None = None
    drawdown_magnitude: float | None = None
    root_cause: str = ""
    resolution: str = ""
    status: str = "OPEN"  # OPEN | INVESTIGATING | RESOLVED | WONT_FIX
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        category: FailureCategory,
        severity: FailureSeverity,
        qde_version: str,
        description: str,
        instrument: str = "",
        regime: str = "",
        loss_amount: float | None = None,
        drawdown_magnitude: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FailureEvent:
        return cls(
            failure_id=f"fail-{str(uuid.uuid4())[:10]}",
            category=category,
            severity=severity,
            qde_version=qde_version,
            detected_at=time.time(),
            description=description,
            instrument=instrument,
            regime=regime,
            loss_amount=loss_amount,
            drawdown_magnitude=drawdown_magnitude,
            tags=tags or [],
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "qde_version": self.qde_version,
            "detected_at": round(self.detected_at, 3),
            "detected_at_iso": datetime.fromtimestamp(self.detected_at, tz=UTC).isoformat(),
            "description": self.description,
            "instrument": self.instrument,
            "regime": self.regime,
            "loss_amount": round(self.loss_amount, 6) if self.loss_amount is not None else None,
            "drawdown_magnitude": (
                round(self.drawdown_magnitude, 6) if self.drawdown_magnitude is not None else None
            ),
            "root_cause": self.root_cause,
            "resolution": self.resolution,
            "status": self.status,
            "tags": self.tags,
            "metadata": self.metadata,
        }


# ── Parameter Search ──────────────────────────────────────────────────────────


@dataclass
class ParameterSearchRun:
    """A hyperparameter search run."""

    search_id: str
    name: str
    strategy: SearchStrategy
    search_space: dict[str, Any]
    objective: str
    maximize: bool
    status: str  # RUNNING | COMPLETED | FAILED
    created_at: float
    completed_at: float | None = None
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    trials: list[dict[str, Any]] = field(default_factory=list)
    n_trials: int = 0
    qde_version: str = ""
    notes: str = ""

    @classmethod
    def new(
        cls,
        name: str,
        strategy: SearchStrategy,
        search_space: dict[str, Any],
        objective: str,
        maximize: bool = True,
        n_trials: int = 20,
        qde_version: str = "",
        notes: str = "",
    ) -> ParameterSearchRun:
        return cls(
            search_id=f"search-{str(uuid.uuid4())[:10]}",
            name=name,
            strategy=strategy,
            search_space=search_space,
            objective=objective,
            maximize=maximize,
            status="RUNNING",
            created_at=time.time(),
            n_trials=n_trials,
            qde_version=qde_version,
            notes=notes,
        )

    def record_trial(
        self, params: dict[str, Any], score: float, metadata: dict[str, Any] | None = None
    ) -> None:
        trial = {
            "trial_id": len(self.trials) + 1,
            "params": params,
            "score": round(score, 6),
            "metadata": metadata or {},
        }
        self.trials.append(trial)
        if (
            self.best_score is None
            or (self.maximize and score > self.best_score)
            or (not self.maximize and score < self.best_score)
        ):
            self.best_score = score
            self.best_params = params

    def complete(self) -> None:
        self.status = "COMPLETED"
        self.completed_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_id": self.search_id,
            "name": self.name,
            "strategy": self.strategy.value,
            "search_space": self.search_space,
            "objective": self.objective,
            "maximize": self.maximize,
            "status": self.status,
            "created_at": round(self.created_at, 3),
            "completed_at": round(self.completed_at, 3) if self.completed_at else None,
            "n_trials_target": self.n_trials,
            "n_trials_completed": len(self.trials),
            "best_params": self.best_params,
            "best_score": round(self.best_score, 6) if self.best_score is not None else None,
            "trials": self.trials,
            "qde_version": self.qde_version,
            "notes": self.notes,
        }


# ── Regime Analysis ───────────────────────────────────────────────────────────


@dataclass
class RegimeAnalysisReport:
    """Performance breakdown of a QDE version across market regimes."""

    report_id: str
    qde_version: str
    created_at: float
    regime_results: dict[str, dict[str, float]]
    worst_regime: str | None
    best_regime: str | None
    overall_regime_stability: float
    recommendations: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def build(
        cls,
        qde_version: str,
        regime_results: dict[str, dict[str, float]],
        notes: str = "",
    ) -> RegimeAnalysisReport:
        sharpes = {r: m.get("sharpe_ratio", 0.0) for r, m in regime_results.items()}
        best = max(sharpes, key=lambda r: sharpes[r]) if sharpes else None
        worst = min(sharpes, key=lambda r: sharpes[r]) if sharpes else None
        values = list(sharpes.values())
        stability = (
            1.0 - (max(values) - min(values)) / (max(values) + 1e-9) if len(values) > 1 else 1.0
        )
        return cls(
            report_id=f"regime-{str(uuid.uuid4())[:10]}",
            qde_version=qde_version,
            created_at=time.time(),
            regime_results=regime_results,
            best_regime=best,
            worst_regime=worst,
            overall_regime_stability=max(0.0, min(1.0, stability)),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "qde_version": self.qde_version,
            "created_at": round(self.created_at, 3),
            "regime_results": {
                r: {k: round(v, 6) for k, v in m.items()} for r, m in self.regime_results.items()
            },
            "worst_regime": self.worst_regime,
            "best_regime": self.best_regime,
            "overall_regime_stability": round(self.overall_regime_stability, 4),
            "recommendations": self.recommendations,
            "notes": self.notes,
        }


# ── Deployment Gate ───────────────────────────────────────────────────────────


@dataclass
class GateCheckResult:
    gate: str
    passed: bool
    value: float
    baseline: float | None
    delta: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "value": round(self.value, 6),
            "baseline": round(self.baseline, 6) if self.baseline is not None else None,
            "delta": round(self.delta, 6),
            "notes": self.notes,
        }


@dataclass
class DeploymentGateResult:
    """Result of running deployment gates against an experiment."""

    gate_id: str
    experiment_id: str
    qde_version: str
    passed: bool
    gate_results: list[GateCheckResult]
    failed_gates: list[str]
    evaluated_at: str
    recommendation: str = ""

    @classmethod
    def evaluate(
        cls,
        experiment: ExperimentRecord,
        baseline_metrics: dict[str, float] | None = None,
        max_drawdown_tolerance: float = 0.02,
    ) -> DeploymentGateResult:
        gate_results: list[GateCheckResult] = []
        failed: list[str] = []
        m = experiment.validation_metrics

        # Gate 1: Sharpe ratio >= 1.0
        sharpe = m.get("sharpe_ratio", experiment.sharpe_ratio or 0.0)
        gate_results.append(
            GateCheckResult(
                gate="sharpe_ratio_min",
                passed=sharpe >= 1.0,
                value=sharpe,
                baseline=baseline_metrics.get("sharpe_ratio") if baseline_metrics else None,
                delta=sharpe
                - (baseline_metrics.get("sharpe_ratio", 0.0) if baseline_metrics else 0.0),
                notes="Sharpe ratio must be >= 1.0",
            )
        )
        if not gate_results[-1].passed:
            failed.append("sharpe_ratio_min")

        # Gate 2: Max drawdown <= threshold
        mdd = abs(m.get("max_drawdown", experiment.max_drawdown or 1.0))
        gate_results.append(
            GateCheckResult(
                gate="max_drawdown_limit",
                passed=mdd <= max_drawdown_tolerance + 0.05,
                value=mdd,
                baseline=baseline_metrics.get("max_drawdown") if baseline_metrics else None,
                delta=mdd
                - (baseline_metrics.get("max_drawdown", 0.0) if baseline_metrics else 0.0),
                notes=f"Max drawdown must be <= {max_drawdown_tolerance + 0.05:.0%}",
            )
        )
        if not gate_results[-1].passed:
            failed.append("max_drawdown_limit")

        # Gate 3: Profit factor >= 1.2
        pf = m.get("profit_factor", experiment.profit_factor or 0.0)
        gate_results.append(
            GateCheckResult(
                gate="profit_factor_min",
                passed=pf >= 1.2,
                value=pf,
                baseline=baseline_metrics.get("profit_factor") if baseline_metrics else None,
                delta=pf
                - (baseline_metrics.get("profit_factor", 0.0) if baseline_metrics else 0.0),
                notes="Profit factor must be >= 1.2",
            )
        )
        if not gate_results[-1].passed:
            failed.append("profit_factor_min")

        # Gate 4: Win rate >= 40%
        wr = m.get("win_rate", experiment.win_rate or 0.0)
        gate_results.append(
            GateCheckResult(
                gate="win_rate_min",
                passed=wr >= 0.40,
                value=wr,
                baseline=baseline_metrics.get("win_rate") if baseline_metrics else None,
                delta=wr - (baseline_metrics.get("win_rate", 0.0) if baseline_metrics else 0.0),
                notes="Win rate must be >= 40%",
            )
        )
        if not gate_results[-1].passed:
            failed.append("win_rate_min")

        # Gate 5: No regression vs. baseline (if provided)
        if baseline_metrics and "sharpe_ratio" in baseline_metrics:
            base_sharpe = baseline_metrics["sharpe_ratio"]
            regression = sharpe < base_sharpe * 0.90
            gate_results.append(
                GateCheckResult(
                    gate="no_sharpe_regression",
                    passed=not regression,
                    value=sharpe,
                    baseline=base_sharpe,
                    delta=sharpe - base_sharpe,
                    notes="New version must not regress Sharpe by more than 10%",
                )
            )
            if regression:
                failed.append("no_sharpe_regression")

        passed = len(failed) == 0
        rec = "APPROVED" if passed else f"REJECTED: {', '.join(failed)}"

        return cls(
            gate_id=f"gate-{str(uuid.uuid4())[:10]}",
            experiment_id=experiment.experiment_id,
            qde_version=experiment.qde_version,
            passed=passed,
            gate_results=gate_results,
            failed_gates=failed,
            evaluated_at=datetime.now(UTC).isoformat(),
            recommendation=rec,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "experiment_id": self.experiment_id,
            "qde_version": self.qde_version,
            "passed": self.passed,
            "gate_results": [g.to_dict() for g in self.gate_results],
            "failed_gates": self.failed_gates,
            "evaluated_at": self.evaluated_at,
            "recommendation": self.recommendation,
        }


# ── Shadow Testing ────────────────────────────────────────────────────────────


def compute_promotion_score(
    decisions_agree: bool,
    confidence_diff: float,
    pnl_diff: float,
    latency_diff_ms: float,
    risk_diff: float,
) -> float:
    """Composite [0, 1] score for a single champion/challenger comparison.

    Blends five signals into one number so live shadow-traffic results can
    drive automated promote/reject/extend decisions the same way the
    five-dimension backtest engine (`QDEPromotionEngine`) does for offline
    metrics. This score is intentionally cheap and deterministic — it never
    replaces the backtest engine, only supplies an additional, independent
    signal derived from real-time traffic.

    - Agreement (35%): the challenger deciding the same way as the champion
      is the single strongest signal that it is safe to trust.
    - Expected-PnL (25%): positive means the challenger's decision would have
      been more profitable on the same input (see `pnl_diff` semantics on
      `ShadowDecisionRecord`).
    - Confidence (15%): rewards a more confident challenger, centred at 0.5
      so equal confidence is neutral.
    - Latency (15%): a slower challenger is penalised on a 100ms scale — pure
      inference latency should not regress meaningfully.
    - Risk (10%): rewards similar or lower risk allocation; a large swing in
      either direction (even "more aggressive") is treated as unproven.
    """
    agreement_component = 1.0 if decisions_agree else 0.0

    if pnl_diff > 0:
        pnl_component = 1.0
    elif pnl_diff == 0:
        pnl_component = 0.5
    else:
        pnl_component = 0.0

    confidence_component = _clamp01(0.5 + confidence_diff)

    latency_component = 1.0 if latency_diff_ms <= 0 else max(0.0, 1.0 - latency_diff_ms / 100.0)

    risk_component = 1.0 if abs(risk_diff) <= 0.01 else max(0.0, 1.0 - abs(risk_diff) / 0.02)

    return round(
        0.35 * agreement_component
        + 0.25 * pnl_component
        + 0.15 * confidence_component
        + 0.15 * latency_component
        + 0.10 * risk_component,
        4,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class ShadowDecisionRecord:
    """A single decision comparison between champion and challenger in shadow mode.

    The challenger QDE receives the same market snapshot as the champion
    (dual dispatch — see `trainer.qde.qde_shadow.ShadowDispatcher`). The
    challenger's decision is NOT executed — only observed and recorded.

    Widened comparison fields (`confidence_diff`, `pnl_diff`,
    `latency_diff_ms`, `risk_diff`, `promotion_score`) capture more than
    simple agreement so the automated promotion pathway has enough signal to
    act on live traffic, not just backtests.
    """

    record_id: str
    test_id: str
    instrument: str
    champion_decision_type: str
    challenger_decision_type: str
    champion_confidence: float
    challenger_confidence: float
    decisions_agree: bool
    regime: str
    recorded_at: float
    market_context: dict[str, Any] = field(default_factory=dict)

    # Widened comparison fields (Task #5 — inference traffic splitting)
    confidence_diff: float = 0.0
    pnl_diff: float = 0.0
    latency_diff_ms: float = 0.0
    risk_diff: float = 0.0
    promotion_score: float = 0.0

    @classmethod
    def new(
        cls,
        test_id: str,
        instrument: str,
        champion_decision_type: str,
        challenger_decision_type: str,
        champion_confidence: float,
        challenger_confidence: float,
        regime: str = "",
        market_context: dict[str, Any] | None = None,
        pnl_diff: float = 0.0,
        latency_diff_ms: float = 0.0,
        risk_diff: float = 0.0,
        promotion_score: float | None = None,
    ) -> ShadowDecisionRecord:
        decisions_agree = champion_decision_type == challenger_decision_type
        confidence_diff = challenger_confidence - champion_confidence
        if promotion_score is None:
            promotion_score = compute_promotion_score(
                decisions_agree=decisions_agree,
                confidence_diff=confidence_diff,
                pnl_diff=pnl_diff,
                latency_diff_ms=latency_diff_ms,
                risk_diff=risk_diff,
            )
        return cls(
            record_id=f"sdr-{str(uuid.uuid4())[:10]}",
            test_id=test_id,
            instrument=instrument,
            champion_decision_type=champion_decision_type,
            challenger_decision_type=challenger_decision_type,
            champion_confidence=champion_confidence,
            challenger_confidence=challenger_confidence,
            decisions_agree=decisions_agree,
            regime=regime,
            recorded_at=time.time(),
            market_context=market_context or {},
            confidence_diff=confidence_diff,
            pnl_diff=pnl_diff,
            latency_diff_ms=latency_diff_ms,
            risk_diff=risk_diff,
            promotion_score=promotion_score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "test_id": self.test_id,
            "instrument": self.instrument,
            "champion_decision_type": self.champion_decision_type,
            "challenger_decision_type": self.challenger_decision_type,
            "champion_confidence": round(self.champion_confidence, 4),
            "challenger_confidence": round(self.challenger_confidence, 4),
            "decisions_agree": self.decisions_agree,
            "regime": self.regime,
            "recorded_at": round(self.recorded_at, 3),
            "market_context": self.market_context,
            "confidence_diff": round(self.confidence_diff, 4),
            "pnl_diff": round(self.pnl_diff, 5),
            "latency_diff_ms": round(self.latency_diff_ms, 3),
            "risk_diff": round(self.risk_diff, 5),
            "promotion_score": round(self.promotion_score, 4),
        }


@dataclass
class ShadowTestRun:
    """A shadow test session comparing a challenger QDE version against the champion.

    The challenger receives the same live market snapshots as the champion.
    The challenger's decisions are never executed — they are only compared.
    This validates behaviour in real market conditions without real risk.
    """

    test_id: str
    name: str
    challenger_version: str
    champion_version: str
    instruments: list[str]
    status: ShadowTestStatus
    started_at: float
    min_decisions: int
    concluded_at: float | None = None
    notes: str = ""

    # S27 — Institutional Validation, Governance and Certification: a
    # mandatory minimum *observation duration*, distinct from `min_decisions`.
    # A challenger that happens to see a burst of decision volume in a short
    # window (e.g. a volatile hour) must still be observed across enough real
    # wall-clock time/regimes before promotion is eligible — volume alone is
    # not a substitute for time-in-market. Defaults to 0 (disabled) so
    # existing volume-only shadow tests are unaffected unless a caller opts
    # in with an explicit minimum (institutional deployments should pass one,
    # e.g. 7 days — see `create_shadow_test`/`ShadowTestRun.new`).
    min_duration_seconds: float = 0.0

    # Decision records (in-memory; DB adapter replaces this list in production)
    decisions: list[ShadowDecisionRecord] = field(default_factory=list)

    # Computed summary (populated on conclude)
    total_decisions: int = 0
    agreement_rate: float = 0.0
    disagreement_rate: float = 0.0
    avg_confidence_diff: float = 0.0
    avg_latency_diff_ms: float = 0.0
    challenger_higher_confidence_rate: float = 0.0
    per_regime_agreement: dict[str, float] = field(default_factory=dict)
    per_instrument_agreement: dict[str, float] = field(default_factory=dict)
    recommendation: ShadowTestRecommendation = ShadowTestRecommendation.INCONCLUSIVE
    conclusion_notes: str = ""

    # Live-traffic promotion signal (Task #5 — averaged from each decision's
    # `promotion_score`; feeds the automated promote/reject/extend call below
    # and, separately, `trainer.qde.qde_shadow.evaluate_and_maybe_promote`).
    avg_promotion_score: float = 0.0

    @classmethod
    def new(
        cls,
        name: str,
        challenger_version: str,
        champion_version: str,
        instruments: list[str],
        min_decisions: int = 500,
        notes: str = "",
        min_duration_seconds: float = 0.0,
    ) -> ShadowTestRun:
        return cls(
            test_id=f"shadow-{str(uuid.uuid4())[:10]}",
            name=name,
            challenger_version=challenger_version,
            champion_version=champion_version,
            instruments=instruments,
            status=ShadowTestStatus.RUNNING,
            started_at=time.time(),
            min_decisions=min_decisions,
            notes=notes,
            min_duration_seconds=min_duration_seconds,
        )

    def add_decision(self, record: ShadowDecisionRecord) -> None:
        self.decisions.append(record)

    def conclude(self, notes: str = "") -> None:
        """Compute summary statistics and produce a promotion recommendation."""
        self.status = ShadowTestStatus.CONCLUDED
        self.concluded_at = time.time()
        self.conclusion_notes = notes

        n = len(self.decisions)
        self.total_decisions = n

        if n == 0:
            self.recommendation = ShadowTestRecommendation.INCONCLUSIVE
            return

        # S27 — mandatory minimum observation duration, distinct from
        # `min_decisions`. Used below as a hard gate on PROMOTE.
        observed_duration = (self.concluded_at or time.time()) - self.started_at

        agreements = sum(1 for d in self.decisions if d.decisions_agree)
        self.agreement_rate = agreements / n
        self.disagreement_rate = round(1.0 - self.agreement_rate, 4)
        self.avg_confidence_diff = round(sum(d.confidence_diff for d in self.decisions) / n, 4)
        self.avg_latency_diff_ms = round(sum(d.latency_diff_ms for d in self.decisions) / n, 3)

        higher_conf = sum(
            1 for d in self.decisions if d.challenger_confidence > d.champion_confidence
        )
        self.challenger_higher_confidence_rate = higher_conf / n

        # Per-regime agreement
        by_regime: dict[str, list[bool]] = {}
        for d in self.decisions:
            regime = d.regime or "UNKNOWN"
            by_regime.setdefault(regime, []).append(d.decisions_agree)
        self.per_regime_agreement = {r: round(sum(v) / len(v), 4) for r, v in by_regime.items()}

        # Per-instrument agreement
        by_instrument: dict[str, list[bool]] = {}
        for d in self.decisions:
            by_instrument.setdefault(d.instrument, []).append(d.decisions_agree)
        self.per_instrument_agreement = {
            i: round(sum(v) / len(v), 4) for i, v in by_instrument.items()
        }

        # Average promotion score across all recorded comparisons — the
        # live-traffic counterpart to the backtest engine's dimension scores.
        self.avg_promotion_score = round(sum(d.promotion_score for d in self.decisions) / n, 4)

        # Recommendation logic — driven primarily by the averaged promotion
        # score, with agreement rate as a guardrail so a high score built on
        # mostly-disagreeing decisions still can't slip through as PROMOTE.
        #
        # S27 — the minimum observation duration gate (checked above) is a
        # hard block on PROMOTE specifically: a challenger that is otherwise
        # performing badly can still be REJECTed early (no reason to keep
        # observing a clear loser), but it can never be recommended for
        # PROMOTE until it has cleared `min_duration_seconds` of real
        # wall-clock observation, no matter how good its score looks.
        duration_cleared = observed_duration >= self.min_duration_seconds
        if n < self.min_decisions:
            self.recommendation = ShadowTestRecommendation.EXTEND
        elif self.avg_promotion_score >= 0.70 and self.agreement_rate >= 0.60:
            self.recommendation = (
                ShadowTestRecommendation.PROMOTE
                if duration_cleared
                else ShadowTestRecommendation.EXTEND
            )
        elif self.avg_promotion_score < 0.40 or self.agreement_rate < 0.40:
            self.recommendation = ShadowTestRecommendation.REJECT
        else:
            self.recommendation = ShadowTestRecommendation.EXTEND

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "challenger_version": self.challenger_version,
            "champion_version": self.champion_version,
            "instruments": self.instruments,
            "status": self.status.value,
            "started_at": round(self.started_at, 3),
            "concluded_at": round(self.concluded_at, 3) if self.concluded_at else None,
            "min_decisions": self.min_decisions,
            "min_duration_seconds": self.min_duration_seconds,
            "notes": self.notes,
            "total_decisions": self.total_decisions,
            "agreement_rate": round(self.agreement_rate, 4),
            "disagreement_rate": round(self.disagreement_rate, 4),
            "avg_confidence_diff": round(self.avg_confidence_diff, 4),
            "avg_latency_diff_ms": round(self.avg_latency_diff_ms, 3),
            "challenger_higher_confidence_rate": round(self.challenger_higher_confidence_rate, 4),
            "per_regime_agreement": self.per_regime_agreement,
            "per_instrument_agreement": self.per_instrument_agreement,
            "avg_promotion_score": round(self.avg_promotion_score, 4),
            "recommendation": self.recommendation.value,
            "conclusion_notes": self.conclusion_notes,
        }


# ── Rollback Approval ──────────────────────────────────────────────────────────


@dataclass
class RollbackRequest:
    """A formal request to roll the active QDE version back to a previous one.

    Rollbacks require explicit administrator approval before the QDE registry
    is updated. The approval workflow is separate from the promotion workflow
    so that each direction of version change has its own audit trail.
    """

    request_id: str
    current_version_id: str
    target_version_id: str
    reason: str
    requested_by: str
    status: RollbackStatus
    created_at: float

    resolved_at: float | None = None
    resolved_by: str = ""
    resolution_notes: str = ""

    @classmethod
    def new(
        cls,
        current_version_id: str,
        target_version_id: str,
        reason: str,
        requested_by: str,
    ) -> RollbackRequest:
        return cls(
            request_id=f"rb-{str(uuid.uuid4())[:12]}",
            current_version_id=current_version_id,
            target_version_id=target_version_id,
            reason=reason,
            requested_by=requested_by,
            status=RollbackStatus.PENDING,
            created_at=time.time(),
        )

    def approve(self, approver: str, notes: str = "") -> None:
        self.status = RollbackStatus.APPROVED
        self.resolved_at = time.time()
        self.resolved_by = approver
        self.resolution_notes = notes

    def reject(self, rejector: str, notes: str = "") -> None:
        self.status = RollbackStatus.REJECTED
        self.resolved_at = time.time()
        self.resolved_by = rejector
        self.resolution_notes = notes

    def withdraw(self, notes: str = "") -> None:
        self.status = RollbackStatus.WITHDRAWN
        self.resolved_at = time.time()
        self.resolution_notes = notes

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "current_version_id": self.current_version_id,
            "target_version_id": self.target_version_id,
            "reason": self.reason,
            "requested_by": self.requested_by,
            "status": self.status.value,
            "created_at": round(self.created_at, 3),
            "resolved_at": round(self.resolved_at, 3) if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
        }


# ── Lifecycle Authorization ─────────────────────────────────────────────────────


@dataclass
class LifecycleAuthorization:
    """A Research-Lab-issued authorization for a specific QDE version
    lifecycle action (promote / rollback / archive / retire).

    Research Lab is the SOLE authority permitted to approve these actions.
    The QDE gateway (Express, `apps/trainer-api`) must obtain a matching
    APPROVED authorization from Research Lab and mark it CONSUMED atomically
    with executing the underlying registry mutation. This closes the gap
    where the gateway could otherwise execute lifecycle mutations directly
    against `qde_server.py` without Research Lab sign-off.

    Status machine:
        PENDING → APPROVED → CONSUMED
               ↘ REJECTED
    """

    authorization_id: str
    action: LifecycleAction
    version_id: str
    requested_by: str
    status: LifecycleAuthorizationStatus
    created_at: float

    reason: str = ""
    target_version_id: str | None = None  # relevant for ROLLBACK
    resolved_at: float | None = None
    resolved_by: str = ""
    resolution_notes: str = ""
    consumed_at: float | None = None

    @classmethod
    def new(
        cls,
        action: LifecycleAction,
        version_id: str,
        requested_by: str,
        reason: str = "",
        target_version_id: str | None = None,
    ) -> LifecycleAuthorization:
        return cls(
            authorization_id=f"lca-{str(uuid.uuid4())[:12]}",
            action=action,
            version_id=version_id,
            requested_by=requested_by,
            status=LifecycleAuthorizationStatus.PENDING,
            created_at=time.time(),
            reason=reason,
            target_version_id=target_version_id,
        )

    @classmethod
    def new_approved(
        cls,
        action: LifecycleAction,
        version_id: str,
        approved_by: str,
        reason: str = "",
        target_version_id: str | None = None,
        notes: str = "",
    ) -> LifecycleAuthorization:
        """Create an authorization that is immediately in the APPROVED state.

        Used when an approval decision recorded elsewhere in Research Lab
        (e.g. `approve_deployment`, `approve_rollback`) should also grant the
        gateway permission to execute the corresponding registry mutation.
        """
        auth = cls.new(
            action=action,
            version_id=version_id,
            requested_by=approved_by,
            reason=reason,
            target_version_id=target_version_id,
        )
        auth.approve(approved_by, notes=notes)
        return auth

    def approve(self, approver: str, notes: str = "") -> None:
        self.status = LifecycleAuthorizationStatus.APPROVED
        self.resolved_at = time.time()
        self.resolved_by = approver
        self.resolution_notes = notes

    def reject(self, rejector: str, notes: str = "") -> None:
        self.status = LifecycleAuthorizationStatus.REJECTED
        self.resolved_at = time.time()
        self.resolved_by = rejector
        self.resolution_notes = notes

    def consume(self) -> None:
        if self.status != LifecycleAuthorizationStatus.APPROVED:
            raise ValueError(
                f"Authorization {self.authorization_id} is {self.status.value}, "
                "not APPROVED; cannot consume"
            )
        self.status = LifecycleAuthorizationStatus.CONSUMED
        self.consumed_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorization_id": self.authorization_id,
            "action": self.action.value,
            "version_id": self.version_id,
            "target_version_id": self.target_version_id,
            "requested_by": self.requested_by,
            "status": self.status.value,
            "created_at": round(self.created_at, 3),
            "reason": self.reason,
            "resolved_at": round(self.resolved_at, 3) if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
            "consumed_at": round(self.consumed_at, 3) if self.consumed_at else None,
        }


# ── Dataset Generation Job ─────────────────────────────────────────────────────


@dataclass
class DatasetGenerationJob:
    """Tracks the pipeline for generating a Research Lab dataset.

    A generation job represents one full execution of the data pipeline:
    raw extraction → feature engineering → label generation → quality validation.

    Status machine:
        PENDING → GENERATING → VALIDATING → READY
                                          ↘ FAILED (from any stage)
    """

    job_id: str
    dataset_id: str
    name: str
    instruments: list[str]
    start_date: str
    end_date: str
    feature_schema: dict[str, Any]
    label_schema: dict[str, Any]
    status: DatasetGenerationStatus
    created_at: float
    requested_by: str

    started_at: float | None = None
    completed_at: float | None = None
    row_count: int = 0
    feature_count: int = 0
    quality_metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""
    notes: str = ""

    @classmethod
    def new(
        cls,
        dataset_id: str,
        name: str,
        instruments: list[str],
        start_date: str,
        end_date: str,
        feature_schema: dict[str, Any] | None = None,
        label_schema: dict[str, Any] | None = None,
        requested_by: str = "",
        notes: str = "",
    ) -> DatasetGenerationJob:
        return cls(
            job_id=f"gen-{str(uuid.uuid4())[:12]}",
            dataset_id=dataset_id,
            name=name,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            feature_schema=feature_schema or {},
            label_schema=label_schema or {},
            status=DatasetGenerationStatus.PENDING,
            created_at=time.time(),
            requested_by=requested_by,
            notes=notes,
        )

    def mark_generating(self) -> None:
        self.status = DatasetGenerationStatus.GENERATING
        self.started_at = time.time()

    def mark_validating(self, row_count: int = 0, feature_count: int = 0) -> None:
        self.status = DatasetGenerationStatus.VALIDATING
        self.row_count = row_count
        self.feature_count = feature_count

    def mark_ready(self, quality_metrics: dict[str, float] | None = None) -> None:
        self.status = DatasetGenerationStatus.READY
        self.completed_at = time.time()
        if quality_metrics:
            self.quality_metrics = quality_metrics

    def mark_failed(self, error: str) -> None:
        self.status = DatasetGenerationStatus.FAILED
        self.completed_at = time.time()
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "instruments": self.instruments,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "feature_schema": self.feature_schema,
            "label_schema": self.label_schema,
            "status": self.status.value,
            "created_at": round(self.created_at, 3),
            "requested_by": self.requested_by,
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "completed_at": round(self.completed_at, 3) if self.completed_at else None,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "quality_metrics": {k: round(v, 6) for k, v in self.quality_metrics.items()},
            "error": self.error,
            "notes": self.notes,
        }

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary without the full decisions list."""
        d = self.to_dict()
        d.pop("decisions", None)
        return d


# ── Research Notes ──────────────────────────────────────────────────────────


@dataclass
class ResearchNote:
    """A standalone, addressable research note.

    Notes are first-class persisted entities — not embedded fields on other
    records. They may optionally reference another entity (e.g. an experiment
    or a failure) via linked_entity_type / linked_entity_id, but they can
    also stand alone as free-form research commentary.
    """

    note_id: str
    content: str
    created_at: float
    author: str = ""
    linked_entity_type: str = ""
    linked_entity_id: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        content: str,
        author: str = "",
        linked_entity_type: str = "",
        linked_entity_id: str = "",
        tags: list[str] | None = None,
    ) -> ResearchNote:
        return cls(
            note_id=f"note-{str(uuid.uuid4())[:12]}",
            content=content,
            created_at=time.time(),
            author=author,
            linked_entity_type=linked_entity_type,
            linked_entity_id=linked_entity_id,
            tags=tags or [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "note_id": self.note_id,
            "content": self.content,
            "created_at": round(self.created_at, 3),
            "created_at_iso": datetime.fromtimestamp(self.created_at, tz=UTC).isoformat(),
            "author": self.author,
            "linked_entity_type": self.linked_entity_type,
            "linked_entity_id": self.linked_entity_id,
            "tags": self.tags,
        }
