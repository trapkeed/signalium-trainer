"""QDE Brain — behavioral memory.

Folds raw, per-event behavioral observations (a trend run ending, a
pullback resolving, a volatility-cluster tick, a liquidity sweep, a
session closing out, a news reaction, a reversal) into a symbol's
`personality_profiles.PersonalityProfile`, continuously, the same way
`long_term_memory.LongTermMemory.record_experience` continuously folds
coarse `ExperienceRecord`s into `SymbolPersonality` — but for the
market-microstructure-shaped signals that feed the personality
dimensions, which nothing in this codebase previously captured.

Deliberately separate from `long_term_memory.py`: that module keeps
handling hypothesis-milestone/regime-outcome experiences into
`SymbolPersonality`; this module handles raw behavioral events into
`PersonalityProfile`. Neither reads or writes the other's storage.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from trainer.qde.brain.personality_profiles import (
    DEFAULT_SESSION_ALPHA,
    DEFAULT_TRAIT_ALPHA,
    PersonalityProfile,
    _ema,
)

# A trend run at/above this dwell is treated as "the trend actually
# persisted" for the trend-persistence dimension, rather than noise.
MIN_TREND_RUN_SECONDS_TO_COUNT = 300.0
# A trend run at/above this dwell is treated as a strongly-persistent
# reading (observed value close to 1.0) rather than a marginal one.
STRONG_TREND_RUN_SECONDS = 3600.0 * 4


@dataclass
class BehavioralObservation:
    """One raw, real-world behavioral event for a symbol. Every field
    besides `instrument` is optional — a single observation is expected
    to carry only whatever signal the caller actually has for this tick
    (e.g. a pullback resolving vs. a news event firing are different
    moments), and `BehavioralMemory.observe` only updates the dimensions
    for which evidence is present.
    """

    observation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    instrument: str = ""

    # A trend leg just ended (or is being sampled) after running for
    # this many seconds — feeds `trend_persistence`.
    trend_run_seconds: float | None = None
    # A pullback against the prevailing move resolved, having retraced
    # this fraction (0-1+) of the preceding move — feeds `avg_pullback_depth`.
    pullback_depth_ratio: float | None = None
    # Whether this tick's elevated-volatility regime is a continuation of
    # the prior tick's (clustering) rather than an isolated spike —
    # feeds `volatility_clustering`.
    volatility_cluster_hit: bool | None = None
    # Whether a liquidity sweep / stop hunt was detected this tick —
    # feeds `liquidity_sweep_frequency`.
    liquidity_sweep_detected: bool | None = None
    # Session label this observation belongs to (e.g. "LONDON", "NY",
    # "ASIAN", "OFF_HOURS") plus how well the symbol behaved during it
    # (-1..1) — feeds `session_affinity`.
    session: str = ""
    session_outcome_quality: float | None = None
    # A scheduled/unscheduled news event fired, with this magnitude of
    # price reaction (0-1+) — feeds `news_sensitivity`.
    news_event: bool = False
    news_reaction_magnitude: float | None = None
    # A reversal (trend flipping direction / mean reversion) was detected
    # this tick — feeds `reversal_tendency`.
    reversal_detected: bool | None = None

    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "instrument": self.instrument,
            "trend_run_seconds": self.trend_run_seconds,
            "pullback_depth_ratio": self.pullback_depth_ratio,
            "volatility_cluster_hit": self.volatility_cluster_hit,
            "liquidity_sweep_detected": self.liquidity_sweep_detected,
            "session": self.session,
            "session_outcome_quality": self.session_outcome_quality,
            "news_event": self.news_event,
            "news_reaction_magnitude": self.news_reaction_magnitude,
            "reversal_detected": self.reversal_detected,
            "created_at": round(self.created_at, 3),
        }


class BehavioralMemory:
    """Stateless — operates on the `PersonalityProfile` held by
    `BrainState.behavioral_profile`, the same convention as
    `LongTermMemory`/`WorkingMemory`: storage lives on `BrainState`,
    this class only folds new observations into it in place.
    """

    def observe(
        self,
        profile: PersonalityProfile,
        observation: BehavioralObservation,
        trait_alpha: float = DEFAULT_TRAIT_ALPHA,
        session_alpha: float = DEFAULT_SESSION_ALPHA,
        now: float | None = None,
    ) -> PersonalityProfile:
        """Fold one behavioral observation into `profile` via exponential
        moving averages, so each dimension adapts gradually rather than
        reacting to any single event. Returns `profile` (mutated in
        place) for convenience.
        """
        now = now if now is not None else time.time()
        touched = False

        if observation.trend_run_seconds is not None:
            observed = min(
                1.0,
                max(0.0, observation.trend_run_seconds) / STRONG_TREND_RUN_SECONDS,
            )
            if observation.trend_run_seconds < MIN_TREND_RUN_SECONDS_TO_COUNT:
                observed *= 0.3  # a very short-lived "trend" is weak evidence either way
            profile.trend_persistence = _ema(profile.trend_persistence, observed, trait_alpha)
            touched = True

        if observation.pullback_depth_ratio is not None:
            observed = max(0.0, min(1.0, observation.pullback_depth_ratio))
            profile.avg_pullback_depth = _ema(profile.avg_pullback_depth, observed, trait_alpha)
            touched = True

        if observation.volatility_cluster_hit is not None:
            observed = 1.0 if observation.volatility_cluster_hit else 0.0
            profile.volatility_clustering = _ema(
                profile.volatility_clustering, observed, trait_alpha
            )
            touched = True

        if observation.liquidity_sweep_detected is not None:
            observed = 1.0 if observation.liquidity_sweep_detected else 0.0
            profile.liquidity_sweep_frequency = _ema(
                profile.liquidity_sweep_frequency, observed, trait_alpha
            )
            touched = True

        if observation.session and observation.session_outcome_quality is not None:
            # Store as 0-1 (outcome quality comes in as -1..1) so it is
            # directly comparable to the other 0-1 dimensions.
            observed = (max(-1.0, min(1.0, observation.session_outcome_quality)) + 1.0) / 2.0
            prior = profile.session_affinity.get(observation.session, observed)
            profile.session_affinity[observation.session] = _ema(prior, observed, session_alpha)
            touched = True

        if observation.news_event and observation.news_reaction_magnitude is not None:
            observed = max(0.0, min(1.0, observation.news_reaction_magnitude))
            profile.news_sensitivity = _ema(profile.news_sensitivity, observed, trait_alpha)
            touched = True

        if observation.reversal_detected is not None:
            observed = 1.0 if observation.reversal_detected else 0.0
            profile.reversal_tendency = _ema(profile.reversal_tendency, observed, trait_alpha)
            touched = True

        if touched:
            profile.sample_count += 1
            profile.last_updated_at = now

        return profile


__all__ = ["BehavioralObservation", "BehavioralMemory"]
