"""Phase 27.7 — Research Lab data models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


# ── Enumerations ──────────────────────────────────────────────────────────────


class ExperimentStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class DeploymentStatus(str, Enum):
    NOT_DEPLOYED = "NOT_DEPLOYED"
    SHADOW = "SHADOW"
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
        # Populate named fields from metrics dict
        for attr in (
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "avg_drawdown",
            "profit_factor", "win_rate", "expected_value", "expected_value_per_trade",
            "decision_quality", "capital_efficiency", "recovery_factor",
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
            "training_started_at": round(self.training_started_at, 3) if self.training_started_at else None,
            "training_completed_at": round(self.training_completed_at, 3) if self.training_completed_at else None,
            "training_duration_s": round(self.training_duration_s, 1) if self.training_duration_s else None,
            "validation_metrics": {k: round(v, 6) for k, v in self.validation_metrics.items()},
            "max_drawdown": round(self.max_drawdown, 6) if self.max_drawdown is not None else None,
            "avg_drawdown": round(self.avg_drawdown, 6) if self.avg_drawdown is not None else None,
            "recovery_factor": round(self.recovery_factor, 4) if self.recovery_factor is not None else None,
            "expected_value": round(self.expected_value, 6) if self.expected_value is not None else None,
            "expected_value_per_trade": round(self.expected_value_per_trade, 6) if self.expected_value_per_trade is not None else None,
            "profit_factor": round(self.profit_factor, 4) if self.profit_factor is not None else None,
            "win_rate": round(self.win_rate, 4) if self.win_rate is not None else None,
            "sharpe_ratio": round(self.sharpe_ratio, 4) if self.sharpe_ratio is not None else None,
            "sortino_ratio": round(self.sortino_ratio, 4) if self.sortino_ratio is not None else None,
            "decision_quality": round(self.decision_quality, 4) if self.decision_quality is not None else None,
            "capital_efficiency": round(self.capital_efficiency, 4) if self.capital_efficiency is not None else None,
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
            "label_quality_score": round(self.label_quality_score, 4) if self.label_quality_score is not None else None,
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
    experiments: list[str] = field(default_factory=list)  # experiment_ids
    status: str = "CANDIDATE"  # CANDIDATE | ACCEPTED | REJECTED
    notes: str = ""

    @classmethod
    def new(
        cls,
        name: str,
        category: str,
        importance_score: float,
        notes: str = "",
    ) -> DiscoveredFeature:
        return cls(
            feature_id=f"feat-{str(uuid.uuid4())[:10]}",
            name=name,
            category=category,
            importance_score=importance_score,
            discovered_at=time.time(),
            notes=notes,
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
    metrics: dict[str, dict[str, float]]  # version_id → metric → value
    rankings: dict[str, list[str]]  # metric → [version_id ranked best→worst]
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

        # Overall winner: most first-place finishes
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
                v: {k: round(val, 6) for k, val in m.items()}
                for v, m in self.metrics.items()
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
            "drawdown_magnitude": round(self.drawdown_magnitude, 6) if self.drawdown_magnitude is not None else None,
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
    objective: str  # metric to optimise
    maximize: bool  # True = higher is better
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

    def record_trial(self, params: dict[str, Any], score: float, metadata: dict[str, Any] | None = None) -> None:
        trial = {
            "trial_id": len(self.trials) + 1,
            "params": params,
            "score": round(score, 6),
            "metadata": metadata or {},
        }
        self.trials.append(trial)
        if self.best_score is None or (
            self.maximize and score > self.best_score
        ) or (not self.maximize and score < self.best_score):
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
    regime_results: dict[str, dict[str, float]]  # regime → metrics
    worst_regime: str | None
    best_regime: str | None
    overall_regime_stability: float  # 0–1
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
        stability = 1.0 - (max(values) - min(values)) / (max(values) + 1e-9) if len(values) > 1 else 1.0
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
                r: {k: round(v, 6) for k, v in m.items()}
                for r, m in self.regime_results.items()
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
        """Run the three mandatory deployment gates."""
        gates: list[GateCheckResult] = []
        failed: list[str] = []
        m = experiment.validation_metrics

        # Gate 1: Decision Quality must improve
        dq = experiment.decision_quality or m.get("decision_quality", 0.0)
        b_dq = baseline_metrics.get("decision_quality") if baseline_metrics else None
        dq_delta = dq - b_dq if b_dq is not None else dq
        dq_passed = dq_delta >= 0.0 and dq > 0.0
        gates.append(GateCheckResult(
            gate="Decision Quality Improvement",
            passed=dq_passed,
            value=dq,
            baseline=b_dq,
            delta=dq_delta,
            notes="decision_quality must improve over baseline",
        ))
        if not dq_passed:
            failed.append("Decision Quality Improvement")

        # Gate 2: Risk Adjusted Return must improve
        rar = max(
            experiment.sharpe_ratio or m.get("sharpe_ratio", 0.0),
            experiment.sortino_ratio or m.get("sortino_ratio", 0.0),
        )
        b_rar = (
            max(baseline_metrics.get("sharpe_ratio", 0.0), baseline_metrics.get("sortino_ratio", 0.0))
            if baseline_metrics else None
        )
        rar_delta = rar - b_rar if b_rar is not None else rar
        rar_passed = rar_delta >= 0.0 and rar > 0.0
        gates.append(GateCheckResult(
            gate="Risk Adjusted Return Improvement",
            passed=rar_passed,
            value=rar,
            baseline=b_rar,
            delta=rar_delta,
            notes="max(sharpe, sortino) must not regress",
        ))
        if not rar_passed:
            failed.append("Risk Adjusted Return Improvement")

        # Gate 3: Maximum Drawdown must not increase materially
        dd = experiment.max_drawdown or m.get("max_drawdown", 0.0)
        b_dd = baseline_metrics.get("max_drawdown") if baseline_metrics else None
        dd_delta = dd - b_dd if b_dd is not None else -dd  # negative = improvement
        dd_passed = b_dd is None or dd <= b_dd + max_drawdown_tolerance
        gates.append(GateCheckResult(
            gate="Maximum Drawdown Stability",
            passed=dd_passed,
            value=dd,
            baseline=b_dd,
            delta=dd_delta,
            notes=f"drawdown must not increase by more than {max_drawdown_tolerance:.1%}",
        ))
        if not dd_passed:
            failed.append("Maximum Drawdown Stability")

        all_passed = len(failed) == 0
        rec = (
            "APPROVED: All deployment gates passed."
            if all_passed
            else f"REJECTED: Failed gates — {', '.join(failed)}."
        )

        return cls(
            gate_id=f"gate-{str(uuid.uuid4())[:10]}",
            experiment_id=experiment.experiment_id,
            qde_version=experiment.qde_version,
            passed=all_passed,
            gate_results=gates,
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
