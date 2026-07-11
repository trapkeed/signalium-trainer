"""QDE Brain — episodic: complete market experience records.

This package stores, indexes, and retrieves complete historical market
experiences — one record per trading episode (an observation that led to a
decision, or a market event significant enough to record).

Each ``Episode`` contains:
  * symbol, timestamp
  * raw market features (as a float vector for similarity search)
  * portfolio state at the time
  * active hypotheses
  * chosen action and confidence
  * reasoning text
  * outcome and trade result
  * regime context and session context

``EpisodicMemory`` supports:
  * ``write_memory(episode)`` — append a new episode
  * ``recall(instrument, ...)`` — filter episodes by symbol/session/vol/regime
  * ``retrieve_similar(features, k, ...)`` — cosine-similarity + recency recall
  * Persistence via ``to_snapshot()`` / ``load_snapshot()``
"""

from trainer.qde.brain.episodic.episodic_memory import (
    Episode,
    EpisodeFeatures,
    EpisodeOutcome,
    EpisodePortfolioState,
    EpisodicIndex,
    EpisodicMemory,
    SimilarEpisode,
)

__all__ = [
    "Episode",
    "EpisodeFeatures",
    "EpisodeOutcome",
    "EpisodePortfolioState",
    "EpisodicIndex",
    "EpisodicMemory",
    "SimilarEpisode",
]
