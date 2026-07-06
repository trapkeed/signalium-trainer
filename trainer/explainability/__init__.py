"""trainer.explainability — Phase 14.9C Explainability & Interpretability.

Public API::

    from trainer.explainability import (
        ExplainabilityEngine,   # high-level orchestrator
        PredictionExplanation,  # result container
        IntegratedGradients,    # path-integral attribution
        GradientAttribution,    # gradient × input saliency
        SmoothGrad,             # noise-averaged gradients
        AttributionResult,      # attribution output container
        AttentionExtractor,     # transformer attention hooks
        AttentionMap,           # captured attention weights
        PermutationImportance,  # black-box feature ranking
        FeatureImportanceResult,
        StreamContributionAnalyzer,  # hybrid-encoder stream analysis
        StreamContribution,
        AttentionMapVisualizer,      # ASCII attention heatmap
        AttentionMapChartData,
        FeatureImportanceChartData,
        FusionContributionChartData,
        EncoderContributionChartData,
    )
"""

from trainer.explainability.attention import AttentionExtractor, AttentionMap
from trainer.explainability.attribution import (
    AttributionResult,
    GradientAttribution,
    IntegratedGradients,
    SmoothGrad,
)
from trainer.explainability.explanation import ExplainabilityEngine, PredictionExplanation
from trainer.explainability.permutation import FeatureImportanceResult, PermutationImportance
from trainer.explainability.stream_analysis import StreamContribution, StreamContributionAnalyzer
from trainer.explainability.visualization import (
    AttentionMapChartData,
    AttentionMapVisualizer,
    EncoderContributionChartData,
    FeatureImportanceChartData,
    FusionContributionChartData,
)

__all__ = [
    "ExplainabilityEngine",
    "PredictionExplanation",
    "IntegratedGradients",
    "GradientAttribution",
    "SmoothGrad",
    "AttributionResult",
    "AttentionExtractor",
    "AttentionMap",
    "PermutationImportance",
    "FeatureImportanceResult",
    "StreamContributionAnalyzer",
    "StreamContribution",
    "AttentionMapVisualizer",
    "AttentionMapChartData",
    "FeatureImportanceChartData",
    "FusionContributionChartData",
    "EncoderContributionChartData",
]
