"""trainer.interfaces — Phase 14.9C Future-Ready Architecture Interfaces.

This package defines ABSTRACT INTERFACES ONLY.  No concrete implementations
are provided here.  All classes are stubs prepared for future research phases.

Do NOT instantiate any class from this package directly.

Public API::

    from trainer.interfaces import (
        BaseExternalEncoder,          # root contract for all future encoders
        BaseReinforcementLearningInterface,
        BaseCrossAssetEncoder,
        BaseNewsEncoder,
        BaseMacroEncoder,
        BaseAlternativeDataEncoder,
        BaseOrderBookEncoder,
        BaseOptionsDataEncoder,
        BaseSentimentEncoder,
        BaseGraphNeuralNetworkInterface,
    )
"""

from trainer.interfaces.base import BaseExternalEncoder
from trainer.interfaces.future_encoders import (
    BaseReinforcementLearningInterface,
    BaseCrossAssetEncoder,
    BaseNewsEncoder,
    BaseMacroEncoder,
    BaseAlternativeDataEncoder,
    BaseOrderBookEncoder,
    BaseOptionsDataEncoder,
    BaseSentimentEncoder,
    BaseGraphNeuralNetworkInterface,
)

__all__ = [
    "BaseExternalEncoder",
    "BaseReinforcementLearningInterface",
    "BaseCrossAssetEncoder",
    "BaseNewsEncoder",
    "BaseMacroEncoder",
    "BaseAlternativeDataEncoder",
    "BaseOrderBookEncoder",
    "BaseOptionsDataEncoder",
    "BaseSentimentEncoder",
    "BaseGraphNeuralNetworkInterface",
]
