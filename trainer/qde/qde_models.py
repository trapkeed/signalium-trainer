"""Phase 27.6 — QDE data models.

Core data structures for the Quant Decision Engine lifecycle.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# ── Enumerations ──────────────────────────────────────────────────────────────


class QDEStage(str, Enum):
    """Ordered pipeline stages for a QDE version."""

    REGISTERED = "REGISTERED"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    BACKTESTING = "BACKTESTING"
    SHADOW_TESTING = "SHADOW_TESTING"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    ARCHIVED = "ARCHIVED"
    ROLLED_BACK = "ROLLED_BACK"
    FAILED = "FAILED"


class QDEABTestType(str, Enum):
    SHADOW = "SHADOW"
    SIDE_BY_SIDE = "SIDE_BY_SIDE"


class PipelineStage(str, Enum):
    HISTORICAL_DATA = "HISTORICAL_DATA"
    TRAINING = "TRAINING"
    VALIDATION = "VALIDATION"
    BACKTESTING = "BACKTESTING"
    SHADOW_TESTING = "SHADOW_TESTING"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"


PIPELINE_ORDER = [
    PipelineStage.HISTORICAL_DATA,
    PipelineStage.TRAINING,
    PipelineStage.VALIDATION,
    PipelineStage.BACKTESTING,
    PipelineStage.SHADOW_TESTING,
    PipelineStage.DEPLOYMENT_CANDIDATE,
]


# ── Promotion Criteria ────────────────────────────────────────────────────────


@dataclass
class QDEPromotionCriteria:
    """The five mandatory promotion dimensions.

    All thresholds are compared against the *incumbent* active version.
    A new version must improve on (or match) every dimension to be eligible.
    """

    # Risk-Adjusted Return: Sharpe / Sortino ratio delta
    min_risk_adjusted_return_delta: float = 0.0

    # Drawdown: max_drawdown must decrease (or stay equal) — lower is better
    max_drawdown_allowed: float = 0.25
    drawdown_must_improve: bool = True

    # Consistency: win-rate or profit-factor delta across walk-forward folds
    min_consistency_delta: float = 0.0

    # Capital Efficiency: return per unit of margin used
    min_capital_efficiency_delta: float = 0.0

    # Decision Quality: precision of entry/exit predictions
    min_decision_quality_delta: float = 0.0

    # Absolute floor safeguards (applied regardless of incumbent)
    min_sharpe: float = 0.5
    min_profit_factor: float = 1.1
    max_drawdown_abs: float = 0.30
    min_win_rate: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_risk_adjusted_return_delta": self.min_risk_adjusted_return_delta,
            "max_drawdown_allowed": self.max_drawdown_allowed,
            "drawdown_must_improve": self.drawdown_must_improve,
            "min_consistency_delta": self.min_consistency_delta,
            "min_capital_efficiency_delta": self.min_capital_efficiency_delta,
            "min_decision_quality_delta": self.min_decision_quality_delta,
            "min_sharpe": self.min_sharpe,
            "min_profit_factor": self.min_profit_factor,
            "max_drawdown_abs": self.max_drawdown_abs,
            "min_win_rate": self.min_win_rate,
        }


# ── QDE Version ───────────────────────────────────────────────────────────────


@dataclass
class QDEVersion:
    """A single versioned Quant Decision Engine artifact."""

    version_id: str
    version_number: int
    label: str
    stage: QDEStage
    created_at: float
    updated_at: float

    # Lineage
    parent_version_id: str | None = None
    tags: list[str] = field(default_factory=list)
    description: str = ""

    # Performance metrics (populated after backtesting / shadow)
    metrics: dict[str, float] = field(default_factory=dict)

    # Pipeline step timestamps
    stage_timestamps: dict[str, float] = field(default_factory=dict)

    # Deployment metadata
    deployed_at: float | None = None
    retired_at: float | None = None
    archived_at: float | None = None

    # Checksum / artifact reference
    artifact_uri: str = ""
    checksum: str = ""

    # I/O contract this version was trained/registered against (Phase S2:
    # `trainer.qde.qde_outputs.OUTPUT_SCHEMA_VERSION`). Rollback uses this to
    # confirm a target version's output shape matches what the live pipeline
    # expects before swapping it in — see docs/qde/phase_s2_deployment_contract.md.
    output_schema_version: str = ""
    feature_contract_version: str = ""

    @classmethod
    def new(
        cls,
        version_number: int,
        label: str = "",
        parent_version_id: str | None = None,
        tags: list[str] | None = None,
        description: str = "",
        artifact_uri: str = "",
        output_schema_version: str = "",
        feature_contract_version: str = "",
    ) -> QDEVersion:
        now = time.time()
        version_id = f"qde-v{version_number}-{str(uuid.uuid4())[:8]}"
        return cls(
            version_id=version_id,
            version_number=version_number,
            label=label or f"QDE v{version_number}",
            stage=QDEStage.REGISTERED,
            created_at=now,
            updated_at=now,
            parent_version_id=parent_version_id,
            tags=tags or [],
            description=description,
            artifact_uri=artifact_uri,
            stage_timestamps={QDEStage.REGISTERED.value: now},
            output_schema_version=output_schema_version,
            feature_contract_version=feature_contract_version,
        )

    def advance_stage(self, new_stage: QDEStage) -> None:
        self.stage = new_stage
        self.updated_at = time.time()
        self.stage_timestamps[new_stage.value] = self.updated_at

    def record_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "version_number": self.version_number,
            "label": self.label,
            "stage": self.stage.value,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
            "parent_version_id": self.parent_version_id,
            "tags": self.tags,
            "description": self.description,
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "stage_timestamps": {k: round(v, 3) for k, v in self.stage_timestamps.items()},
            "deployed_at": round(self.deployed_at, 3) if self.deployed_at else None,
            "retired_at": round(self.retired_at, 3) if self.retired_at else None,
            "archived_at": round(self.archived_at, 3) if self.archived_at else None,
            "artifact_uri": self.artifact_uri,
            "checksum": self.checksum,
            "output_schema_version": self.output_schema_version,
            "feature_contract_version": self.feature_contract_version,
        }


# ── Promotion Decision ────────────────────────────────────────────────────────


@dataclass
class DimensionResult:
    """Evaluation result for one promotion dimension."""

    dimension: str
    passed: bool
    challenger_value: float
    incumbent_value: float | None
    delta: float
    threshold: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "passed": self.passed,
            "challenger_value": round(self.challenger_value, 6),
            "incumbent_value": (
                round(self.incumbent_value, 6) if self.incumbent_value is not None else None
            ),
            "delta": round(self.delta, 6),
            "threshold": round(self.threshold, 6),
            "notes": self.notes,
        }


@dataclass
class QDEPromotionDecision:
    """Result of a QDE promotion evaluation."""

    decision_id: str
    challenger_version_id: str
    incumbent_version_id: str | None
    is_eligible: bool
    is_promoted: bool
    dimension_results: list[DimensionResult]
    failed_dimensions: list[str]
    passed_dimensions: list[str]
    decided_at: str
    approver: str = ""
    notes: str = ""

    @classmethod
    def create(
        cls,
        challenger_version_id: str,
        incumbent_version_id: str | None,
        dimension_results: list[DimensionResult],
        auto_promote: bool = False,
        notes: str = "",
    ) -> QDEPromotionDecision:
        passed = [r.dimension for r in dimension_results if r.passed]
        failed = [r.dimension for r in dimension_results if not r.passed]
        is_eligible = len(failed) == 0
        return cls(
            decision_id=str(uuid.uuid4())[:12],
            challenger_version_id=challenger_version_id,
            incumbent_version_id=incumbent_version_id,
            is_eligible=is_eligible,
            is_promoted=is_eligible and auto_promote,
            dimension_results=dimension_results,
            failed_dimensions=failed,
            passed_dimensions=passed,
            decided_at=datetime.now(UTC).isoformat(),
            notes=notes,
        )

    def approve(self, approver: str = "") -> None:
        if not self.is_eligible:
            raise ValueError(
                f"Cannot approve promotion for {self.challenger_version_id} — "
                f"failed dimensions: {self.failed_dimensions}"
            )
        self.is_promoted = True
        self.approver = approver

    def summary(self) -> str:
        status = "ELIGIBLE" if self.is_eligible else "NOT ELIGIBLE"
        promo = "PROMOTED" if self.is_promoted else "NOT PROMOTED"
        lines = [
            f"=== QDE Promotion: {self.challenger_version_id} ===",
            f"  Status    : {status}",
            f"  Promotion : {promo}",
            f"  Decided   : {self.decided_at}",
            "",
        ]
        for r in self.dimension_results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(
                f"  [{mark}] {r.dimension}: delta={r.delta:+.4f} (threshold={r.threshold:+.4f})"
            )
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "challenger_version_id": self.challenger_version_id,
            "incumbent_version_id": self.incumbent_version_id,
            "is_eligible": self.is_eligible,
            "is_promoted": self.is_promoted,
            "passed_dimensions": self.passed_dimensions,
            "failed_dimensions": self.failed_dimensions,
            "dimension_results": [r.to_dict() for r in self.dimension_results],
            "decided_at": self.decided_at,
            "approver": self.approver,
            "notes": self.notes,
        }


# ── A/B Test ──────────────────────────────────────────────────────────────────


@dataclass
class QDEABTest:
    """An active A/B test between two QDE versions (SHADOW or SIDE_BY_SIDE only).

    Neither test type routes live decisions to the challenger.
    Only one model produces live decisions at any time.
    """

    test_id: str
    test_type: QDEABTestType
    control_version_id: str
    challenger_version_id: str
    created_at: float
    ended_at: float | None = None
    status: str = "RUNNING"

    # Accumulated results per version
    control_metrics: dict[str, float] = field(default_factory=dict)
    challenger_metrics: dict[str, float] = field(default_factory=dict)

    winner: str | None = None
    conclusion: str = ""

    @classmethod
    def create(
        cls,
        test_type: QDEABTestType,
        control_version_id: str,
        challenger_version_id: str,
    ) -> QDEABTest:
        return cls(
            test_id=str(uuid.uuid4())[:12],
            test_type=test_type,
            control_version_id=control_version_id,
            challenger_version_id=challenger_version_id,
            created_at=time.time(),
        )

    def record_results(
        self,
        control_metrics: dict[str, float],
        challenger_metrics: dict[str, float],
    ) -> None:
        self.control_metrics.update(control_metrics)
        self.challenger_metrics.update(challenger_metrics)

    def conclude(self, winner: str, conclusion: str) -> None:
        self.winner = winner
        self.conclusion = conclusion
        self.status = "CONCLUDED"
        self.ended_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_type": self.test_type.value,
            "control_version_id": self.control_version_id,
            "challenger_version_id": self.challenger_version_id,
            "status": self.status,
            "created_at": round(self.created_at, 3),
            "ended_at": round(self.ended_at, 3) if self.ended_at else None,
            "control_metrics": {k: round(v, 6) for k, v in self.control_metrics.items()},
            "challenger_metrics": {k: round(v, 6) for k, v in self.challenger_metrics.items()},
            "winner": self.winner,
            "conclusion": self.conclusion,
        }


# ── Rollback Record ───────────────────────────────────────────────────────────


@dataclass
class QDERollbackRecord:
    """An immutable record of a rollback operation.

    S27 — Institutional Validation, Governance and Certification extends this
    record with the structured forensic context a governance/compliance
    review needs to reconstruct *why* a rollback happened, not just *that* it
    happened: the metrics that triggered it, and every dataset/experiment/
    symbol/version the rollback decision touched.
    """

    rollback_id: str
    from_version_id: str
    to_version_id: str
    reason: str
    initiated_by: str
    rolled_back_at: float
    notes: str = ""

    # S27 — forensic/audit context. All default to empty so existing callers
    # of `create()` (pre-S27) remain valid.
    triggering_metrics: dict[str, float] = field(default_factory=dict)
    affected_versions: list[str] = field(default_factory=list)
    affected_datasets: list[str] = field(default_factory=list)
    affected_experiments: list[str] = field(default_factory=list)
    affected_symbols: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        from_version_id: str,
        to_version_id: str,
        reason: str,
        initiated_by: str = "system",
        notes: str = "",
        triggering_metrics: dict[str, float] | None = None,
        affected_versions: list[str] | None = None,
        affected_datasets: list[str] | None = None,
        affected_experiments: list[str] | None = None,
        affected_symbols: list[str] | None = None,
    ) -> QDERollbackRecord:
        # `affected_versions` always includes at least the two versions the
        # rollback moved between, even if the caller passes nothing — a
        # rollback event that can't answer "which versions were touched"
        # would defeat the point of this field.
        versions = list(dict.fromkeys(
            [from_version_id, to_version_id] + (affected_versions or [])
        ))
        return cls(
            rollback_id=str(uuid.uuid4())[:12],
            from_version_id=from_version_id,
            to_version_id=to_version_id,
            reason=reason,
            initiated_by=initiated_by,
            rolled_back_at=time.time(),
            notes=notes,
            triggering_metrics=dict(triggering_metrics or {}),
            affected_versions=versions,
            affected_datasets=list(affected_datasets or []),
            affected_experiments=list(affected_experiments or []),
            affected_symbols=list(affected_symbols or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "from_version_id": self.from_version_id,
            "to_version_id": self.to_version_id,
            "reason": self.reason,
            "initiated_by": self.initiated_by,
            "rolled_back_at": round(self.rolled_back_at, 3),
            "rolled_back_at_iso": datetime.fromtimestamp(self.rolled_back_at, tz=UTC).isoformat(),
            "notes": self.notes,
            "triggering_metrics": {k: round(v, 6) for k, v in self.triggering_metrics.items()},
            "affected_versions": self.affected_versions,
            "affected_datasets": self.affected_datasets,
            "affected_experiments": self.affected_experiments,
            "affected_symbols": self.affected_symbols,
        }


# ── Pipeline Run ──────────────────────────────────────────────────────────────


@dataclass
class PipelineStepResult:
    stage: PipelineStage
    status: str  # PENDING | RUNNING | PASSED | FAILED | SKIPPED
    started_at: float | None = None
    completed_at: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "completed_at": round(self.completed_at, 3) if self.completed_at else None,
            "duration_s": (
                round(self.completed_at - self.started_at, 3)
                if self.started_at and self.completed_at
                else None
            ),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "logs": self.logs,
            "error": self.error,
        }


@dataclass
class PipelineRun:
    """A full pipeline execution for one QDE version."""

    run_id: str
    version_id: str
    status: str  # RUNNING | COMPLETED | FAILED | ABORTED
    steps: list[PipelineStepResult]
    started_at: float
    completed_at: float | None = None

    @classmethod
    def create(cls, version_id: str) -> PipelineRun:
        return cls(
            run_id=str(uuid.uuid4())[:12],
            version_id=version_id,
            status="RUNNING",
            steps=[PipelineStepResult(stage=s, status="PENDING") for s in PIPELINE_ORDER],
            started_at=time.time(),
        )

    def current_step(self) -> PipelineStepResult | None:
        for step in self.steps:
            if step.status in ("PENDING", "RUNNING"):
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version_id": self.version_id,
            "status": self.status,
            "started_at": round(self.started_at, 3),
            "completed_at": round(self.completed_at, 3) if self.completed_at else None,
            "steps": [s.to_dict() for s in self.steps],
        }
