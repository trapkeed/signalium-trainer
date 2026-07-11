"""QDE Brain — semantic: learned market concepts.

This package accumulates and clusters market experiences into named concepts
that emerge automatically from pattern discovery — without hardcoded rules.

Example concepts that can emerge:
  * breakout after compression
  * failed liquidity sweep
  * trend continuation after pullback
  * mean reversion after expansion

``SemanticMemory`` supports:
  * ``write_memory(episode)`` — ingest an episode, update concept centroids
  * ``recall(feature_vec, ...)`` — retrieve nearest matching concepts
  * ``concepts()`` — list all discovered concepts
  * Persistence via ``to_snapshot()`` / ``load_snapshot()``
"""

from trainer.qde.brain.semantic.semantic_memory import (
    ConceptKind,
    ConceptSeed,
    MarketConcept,
    SemanticMemory,
    SemanticRecallResult,
)

__all__ = [
    "ConceptKind",
    "ConceptSeed",
    "MarketConcept",
    "SemanticMemory",
    "SemanticRecallResult",
]
