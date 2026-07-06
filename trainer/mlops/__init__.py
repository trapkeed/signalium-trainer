"""Phase 18 — Institutional MLOps, Model Governance, and Continuous Monitoring Platform.

Public API for the mlops package.

No production model may exist outside this governance framework.
Zero modifications to any Phase 1–17 module.
No automatic retraining — recommendations only.
Full backward compatibility.

Quickstart::

    from trainer.mlops import MLOpsEngine, MLOpsConfig, ModelRecord

    engine = MLOpsEngine.from_config({
        "db_path": "experiments/mlops/mlops.db",
    })
    model_id = engine.register_model({
        "name": "signalium_quant",
        "version": "1.0.0",
        "architecture": "lstm",
        "backbone": "lstm_256_2l",
        "parameter_count": 2_458_624,
        "dataset_version": "v3.2",
        "feature_version": "v2.1",
        "label_version": "v1.0",
    })
    engine.transition(model_id, "validating", actor="validator")
    engine.approve(model_id, approver="head_quant")
    engine.set_champion(model_id, approver="head_quant")
    engine.record_inference(model_id, latency_ms=11.2, confidence=0.78)
    print(engine.summary())
"""

from __future__ import annotations

# Registry
from trainer.mlops.registry import (
    LifecycleState,
    ApprovalRecord,
    ModelRecord,
    ModelRegistry,
)

# Lifecycle
from trainer.mlops.lifecycle import (
    LifecycleTransition,
    LifecycleError,
    LifecycleManager,
)

# Governance
from trainer.mlops.governance import (
    DeploymentRecord,
    GovernanceConfig,
    DeploymentGovernance,
)

# Monitoring
from trainer.mlops.monitoring import (
    InferenceRecord,
    MonitoringSnapshot,
    ProductionMonitor,
)

# Health
from trainer.mlops.health import (
    HealthRecord,
    HealthReport,
    DriftThresholds,
    ModelHealthMonitor,
)

# Alerting
from trainer.mlops.alerting import (
    AlertLevel,
    AlertCategory,
    Alert,
    AlertingSystem,
)

# Champion
from trainer.mlops.champion import (
    ChampionEvent,
    ChampionChallengerManager,
)

# Audit
from trainer.mlops.audit import (
    AuditEventType,
    AuditEvent,
    AuditSystem,
)

# Retraining
from trainer.mlops.retraining import (
    RetrainingTrigger,
    RetrainingPolicy,
    RetrainingRecommendation,
    RetrainingPolicyManager,
    build_default_policies,
)

# Database
from trainer.mlops.database import MLOpsDatabase

# API
from trainer.mlops.api import MLOpsAPI

# Visualization
from trainer.mlops.visualization import MLOpsVisualizationBuilder

# Export
from trainer.mlops.export import MLOpsExporter

# Engine
from trainer.mlops.engine import MLOpsConfig, MLOpsEngine

__all__ = [
    # Registry
    "LifecycleState",
    "ApprovalRecord",
    "ModelRecord",
    "ModelRegistry",
    # Lifecycle
    "LifecycleTransition",
    "LifecycleError",
    "LifecycleManager",
    # Governance
    "DeploymentRecord",
    "GovernanceConfig",
    "DeploymentGovernance",
    # Monitoring
    "InferenceRecord",
    "MonitoringSnapshot",
    "ProductionMonitor",
    # Health
    "HealthRecord",
    "HealthReport",
    "DriftThresholds",
    "ModelHealthMonitor",
    # Alerting
    "AlertLevel",
    "AlertCategory",
    "Alert",
    "AlertingSystem",
    # Champion
    "ChampionEvent",
    "ChampionChallengerManager",
    # Audit
    "AuditEventType",
    "AuditEvent",
    "AuditSystem",
    # Retraining
    "RetrainingTrigger",
    "RetrainingPolicy",
    "RetrainingRecommendation",
    "RetrainingPolicyManager",
    "build_default_policies",
    # Database
    "MLOpsDatabase",
    # API
    "MLOpsAPI",
    # Visualization
    "MLOpsVisualizationBuilder",
    # Export
    "MLOpsExporter",
    # Engine
    "MLOpsConfig",
    "MLOpsEngine",
]
