"""Phase 18 — Institutional MLOps Model Registry.

Centralised registry for every model that exists in Signalium.

Every model record stores:
  - model_id, semantic version, training date, architecture, backbone
  - parameter count, dataset/feature/label versions, git commit
  - training config, validation report, promotion/deployment/retirement status
  - approval history, hardware used, training duration, parent/child models

No production model may exist outside this registry.
Zero new dependencies — pure stdlib + loguru.

Usage::

    registry = ModelRegistry()
    model_id = registry.register(ModelRecord(
        name="signalium_quant_v1",
        version="1.0.0",
        architecture="lstm",
        backbone="lstm_256_2l",
        ...
    ))
    registry.get(model_id)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Lifecycle states
# ---------------------------------------------------------------------------


class LifecycleState:
    """All valid lifecycle states for a registered model."""
    DRAFT       = "draft"
    TRAINING    = "training"
    VALIDATING  = "validating"
    CANDIDATE   = "candidate"
    APPROVED    = "approved"
    CHAMPION    = "champion"
    PRODUCTION  = "production"
    SHADOW      = "shadow"
    ARCHIVED    = "archived"
    RETIRED     = "retired"
    ROLLED_BACK = "rolled_back"

    ALL = [
        DRAFT, TRAINING, VALIDATING, CANDIDATE, APPROVED,
        CHAMPION, PRODUCTION, SHADOW, ARCHIVED, RETIRED, ROLLED_BACK,
    ]

    ACTIVE = [CHAMPION, PRODUCTION, SHADOW]
    TERMINAL = [ARCHIVED, RETIRED, ROLLED_BACK]


# ---------------------------------------------------------------------------
# Approval record
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRecord:
    """One entry in a model's approval history."""
    approval_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    approver: str = "system"
    decision: str = "approved"        # "approved" | "rejected" | "pending"
    reason: str = ""
    from_state: str = ""
    to_state: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "approver": self.approver,
            "decision": self.decision,
            "reason": self.reason,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Model record
# ---------------------------------------------------------------------------


@dataclass
class ModelRecord:
    """Complete model registry entry.

    Attributes:
        model_id: Unique UUID.
        name: Human-readable model name.
        version: Semantic version string (e.g. "1.0.0").
        training_date: ISO-8601 UTC training completion timestamp.
        architecture: Neural architecture family (lstm, transformer, etc.).
        backbone: Specific backbone variant (e.g. "lstm_256_2l").
        parameter_count: Total trainable parameters.
        dataset_version: Dataset version tag used for training.
        feature_version: Feature engineering pipeline version.
        label_version: Label generation pipeline version.
        git_commit: Full git commit hash of training codebase.
        training_config: Full training configuration dict.
        validation_report: Validation results dict from Phase 15.
        promotion_status: Whether model is approved for promotion.
        deployment_status: Current deployment state.
        retirement_status: Whether the model is retired.
        approval_history: List of ApprovalRecord objects.
        hardware_used: Hardware description (CPU/GPU, memory).
        training_duration_seconds: Wall-clock training time.
        parent_model_id: ID of model this was fine-tuned from (optional).
        child_model_ids: IDs of models derived from this model.
        lifecycle_state: Current LifecycleState value.
        notes: Free-form notes.
        created_at: Registry entry creation timestamp.
        updated_at: Last update timestamp.
        metadata: Arbitrary extra metadata.
    """

    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "model"
    version: str = "1.0.0"
    training_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    architecture: str = ""
    backbone: str = ""
    parameter_count: int = 0
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    git_commit: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    validation_report: Dict[str, Any] = field(default_factory=dict)
    promotion_status: str = "pending"      # "pending" | "approved" | "rejected"
    deployment_status: str = "not_deployed"
    retirement_status: str = "active"      # "active" | "retired"
    approval_history: List[ApprovalRecord] = field(default_factory=list)
    hardware_used: str = ""
    training_duration_seconds: float = 0.0
    parent_model_id: Optional[str] = None
    child_model_ids: List[str] = field(default_factory=list)
    lifecycle_state: str = LifecycleState.DRAFT
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.lifecycle_state in LifecycleState.ACTIVE

    @property
    def is_terminal(self) -> bool:
        return self.lifecycle_state in LifecycleState.TERMINAL

    @property
    def is_champion(self) -> bool:
        return self.lifecycle_state == LifecycleState.CHAMPION

    @property
    def full_name(self) -> str:
        return f"{self.name}@{self.version}"

    def add_approval(self, approver: str, decision: str, reason: str,
                     from_state: str, to_state: str) -> ApprovalRecord:
        record = ApprovalRecord(
            approver=approver, decision=decision, reason=reason,
            from_state=from_state, to_state=to_state,
        )
        self.approval_history.append(record)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return record

    def add_child(self, child_id: str) -> None:
        if child_id not in self.child_model_ids:
            self.child_model_ids.append(child_id)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "full_name": self.full_name,
            "training_date": self.training_date,
            "architecture": self.architecture,
            "backbone": self.backbone,
            "parameter_count": self.parameter_count,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "git_commit": self.git_commit,
            "training_config": self.training_config,
            "validation_report": self.validation_report,
            "promotion_status": self.promotion_status,
            "deployment_status": self.deployment_status,
            "retirement_status": self.retirement_status,
            "approval_history": [a.to_dict() for a in self.approval_history],
            "hardware_used": self.hardware_used,
            "training_duration_seconds": self.training_duration_seconds,
            "parent_model_id": self.parent_model_id,
            "child_model_ids": self.child_model_ids,
            "lifecycle_state": self.lifecycle_state,
            "is_active": self.is_active,
            "is_champion": self.is_champion,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelRecord":
        r = ModelRecord(
            model_id=d.get("model_id", str(uuid.uuid4())),
            name=d.get("name", "model"),
            version=d.get("version", "1.0.0"),
            training_date=d.get("training_date", datetime.now(timezone.utc).isoformat()),
            architecture=d.get("architecture", ""),
            backbone=d.get("backbone", ""),
            parameter_count=d.get("parameter_count", 0),
            dataset_version=d.get("dataset_version", ""),
            feature_version=d.get("feature_version", ""),
            label_version=d.get("label_version", ""),
            git_commit=d.get("git_commit", ""),
            training_config=d.get("training_config", {}),
            validation_report=d.get("validation_report", {}),
            promotion_status=d.get("promotion_status", "pending"),
            deployment_status=d.get("deployment_status", "not_deployed"),
            retirement_status=d.get("retirement_status", "active"),
            hardware_used=d.get("hardware_used", ""),
            training_duration_seconds=d.get("training_duration_seconds", 0.0),
            parent_model_id=d.get("parent_model_id"),
            child_model_ids=d.get("child_model_ids", []),
            lifecycle_state=d.get("lifecycle_state", LifecycleState.DRAFT),
            notes=d.get("notes", ""),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=d.get("updated_at", datetime.now(timezone.utc).isoformat()),
            metadata=d.get("metadata", {}),
        )
        r.approval_history = [
            ApprovalRecord(**a) for a in d.get("approval_history", [])
        ]
        return r


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Centralised in-memory model registry.

    All models must be registered here before entering any governance workflow.
    Backed by EnsembleDatabase (Phase 18 DB) for persistence.
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelRecord] = {}
        self._champion_id: Optional[str] = None

    def register(self, record: ModelRecord) -> str:
        """Register a model. Returns model_id."""
        self._models[record.model_id] = record
        logger.info(f"Registered model {record.model_id} ({record.full_name})")
        return record.model_id

    def get(self, model_id: str) -> Optional[ModelRecord]:
        return self._models.get(model_id)

    def get_or_raise(self, model_id: str) -> ModelRecord:
        m = self._models.get(model_id)
        if m is None:
            raise KeyError(f"Model {model_id!r} not in registry.")
        return m

    def remove(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False

    def all_models(self) -> List[ModelRecord]:
        return list(self._models.values())

    def by_state(self, state: str) -> List[ModelRecord]:
        return [m for m in self._models.values() if m.lifecycle_state == state]

    def active(self) -> List[ModelRecord]:
        return [m for m in self._models.values() if m.is_active]

    def champion(self) -> Optional[ModelRecord]:
        if self._champion_id:
            return self._models.get(self._champion_id)
        champions = [m for m in self._models.values() if m.is_champion]
        return champions[0] if champions else None

    def set_champion(self, model_id: str, approver: str = "system") -> None:
        for m in self._models.values():
            if m.is_champion and m.model_id != model_id:
                m.lifecycle_state = LifecycleState.ARCHIVED
                m.updated_at = datetime.now(timezone.utc).isoformat()
        model = self.get_or_raise(model_id)
        old_state = model.lifecycle_state
        model.lifecycle_state = LifecycleState.CHAMPION
        model.promotion_status = "approved"
        model.add_approval(approver, "approved", "promoted to champion",
                           old_state, LifecycleState.CHAMPION)
        model.updated_at = datetime.now(timezone.utc).isoformat()
        self._champion_id = model_id
        logger.info(f"Champion set: {model_id} ({model.full_name})")

    def by_name(self, name: str) -> List[ModelRecord]:
        return [m for m in self._models.values() if m.name == name]

    def latest_version(self, name: str) -> Optional[ModelRecord]:
        candidates = self.by_name(name)
        if not candidates:
            return None
        return sorted(candidates, key=lambda m: m.version, reverse=True)[0]

    @property
    def n_models(self) -> int:
        return len(self._models)

    def summary(self) -> Dict[str, Any]:
        states: Dict[str, int] = {}
        for m in self._models.values():
            states[m.lifecycle_state] = states.get(m.lifecycle_state, 0) + 1
        champion = self.champion()
        return {
            "total_models": self.n_models,
            "champion_id": self._champion_id,
            "champion_name": champion.full_name if champion else None,
            "by_state": states,
            "n_active": len(self.active()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": {mid: m.to_dict() for mid, m in self._models.items()},
            "champion_id": self._champion_id,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelRegistry":
        registry = ModelRegistry()
        registry._champion_id = d.get("champion_id")
        for mid, mdata in d.get("models", {}).items():
            registry.register(ModelRecord.from_dict(mdata))
        return registry
