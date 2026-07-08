"""QDE — Quant Decision Engine.

Phase 27.6: Lifecycle Management (registry, pipeline, promotion, rollback, A/B).
Phase 28.3: Intelligence (single decision-maker with 11 internal capabilities).
"""

from trainer.qde.qde_api import QDELifecycleAPI
from trainer.qde.qde_intelligence import QDEIntelligence, QDEConfig
from trainer.qde.qde_models import (
    QDEVersion,
    QDEStage,
    QDEPromotionCriteria,
    QDEPromotionDecision,
    QDEABTest,
    QDEABTestType,
    QDERollbackRecord,
    PipelineRun,
    PipelineStage,
)

__all__ = [
    # Intelligence (Phase 28.3)
    "QDEIntelligence",
    "QDEConfig",
    # Lifecycle (Phase 27.6)
    "QDELifecycleAPI",
    "QDEVersion",
    "QDEStage",
    "QDEPromotionCriteria",
    "QDEPromotionDecision",
    "QDEABTest",
    "QDEABTestType",
    "QDERollbackRecord",
    "PipelineRun",
    "PipelineStage",
]
