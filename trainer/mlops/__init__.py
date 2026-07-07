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

# Alerting
from trainer.mlops.alerting import (
    Alert,
    AlertCategory,
    AlertingSystem,
    AlertLevel,
)

# API
from trainer.mlops.api import MLOpsAPI

# Audit
from trainer.mlops.audit import (
    AuditEvent,
    AuditEventType,
    AuditSystem,
)

# Champion
from trainer.mlops.champion import (
    ChampionChallengerManager,
    ChampionEvent,
)

# Database
from trainer.mlops.database import MLOpsDatabase

# Engine
from trainer.mlops.engine import MLOpsConfig, MLOpsEngine

# Export
from trainer.mlops.export import MLOpsExporter

# Governance
from trainer.mlops.governance import (
    DeploymentGovernance,
    DeploymentRecord,
    GovernanceConfig,
)

# Health
from trainer.mlops.health import (
    DriftThresholds,
    HealthRecord,
    HealthReport,
    ModelHealthMonitor,
)

# Lifecycle
from trainer.mlops.lifecycle import (
    LifecycleError,
    LifecycleManager,
    LifecycleTransition,
)

# Monitoring
from trainer.mlops.monitoring import (
    InferenceRecord,
    MonitoringSnapshot,
    ProductionMonitor,
)

# Registry
from trainer.mlops.registry import (
    ApprovalRecord,
    LifecycleState,
    ModelRecord,
    ModelRegistry,
)

# Retraining
from trainer.mlops.retraining import (
    RetrainingPolicy,
    RetrainingPolicyManager,
    RetrainingRecommendation,
    RetrainingTrigger,
    build_default_policies,
)

# Visualization
from trainer.mlops.visualization import MLOpsVisualizationBuilder

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
