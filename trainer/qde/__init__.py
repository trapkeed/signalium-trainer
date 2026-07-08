"""QDE — Quant Decision Engine.

Phase 27.6: Lifecycle Management (registry, pipeline, promotion, rollback, A/B).
"""

from trainer.qde.qde_api import QDELifecycleAPI
from trainer.qde.qde_models import (
    PipelineRun,
    PipelineStage,
    QDEABTest,
    QDEABTestType,
    QDEPromotionCriteria,
    QDEPromotionDecision,
    QDERollbackRecord,
    QDEStage,
    QDEVersion,
)

__all__ = [
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
