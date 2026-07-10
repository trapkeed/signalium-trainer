"""QDE Brain — symbol personality profile data model.

`PersonalityProfile` is a structured, *named-dimension* model of how one
specific symbol tends to behave (trend persistence, pullback depth,
volatility clustering, liquidity-sweep frequency, session dependence,
news sensitivity, reversal tendency). It is deliberately a distinct
model from `brain_models.SymbolPersonality`, which stays exactly as it
is and is not touched by this module:

  - `SymbolPersonality` (`brain_models.py`) is a *free-form* trait bag
    keyed by arbitrary tag strings (e.g. "hypothesis_reliability",
    "trend_follows") plus `regime_outcome_quality`, updated by
    `long_term_memory.LongTermMemory.record_experience` from coarse
    `ExperienceRecord`s (hypothesis milestones, regime outcomes). It
    answers "has this symbol's *hypothesis-tracking* been reliable".
  - `PersonalityProfile` (this module) is a *fixed, named* schema of
    concrete market-behavior dimensions, updated continuously from raw
    `behavioral_memory.BehavioralObservation`s (trend runs, pullback
    depths, volatility/liquidity microstructure events, session
    outcomes, news reactions, reversals). It answers "how does this
    symbol *actually move*", and is the piece that did not exist
    before.

Both live side-by-side on `BrainState` (`personality` vs.
`behavioral_profile`) — neither re-derives or duplicates the other.

This module also defines the pure functions that let a learned profile
*influence* downstream decisions (hypothesis confidence, risk appetite,
setup requirements, expected holding duration). They are pure
transformations of `(profile, base_value)` — never state mutation, and
never a hard override of the caller's own value, only a bounded
adjustment. `symbol_personality_engine.SymbolPersonalityEngine` is the
orchestration layer that calls these on a caller's behalf; this module
never calls back into it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.qde.brain.setup_lifecycle import SetupThresholds

DEFAULT_TRAIT_ALPHA = 0.12  # EMA smoothing for continuous dimension updates
DEFAULT_SESSION_ALPHA = 0.15

# Neutral priors: unopinionated starting point before any evidence exists,
# so a fresh symbol's profile does not silently bias behavior before it
# has actually been observed.
NEUTRAL_TREND_PERSISTENCE = 0.5
NEUTRAL_PULLBACK_DEPTH = 0.5
NEUTRAL_VOLATILITY_CLUSTERING = 0.5
NEUTRAL_LIQUIDITY_SWEEP_FREQUENCY = 0.3
NEUTRAL_NEWS_SENSITIVITY = 0.5
NEUTRAL_REVERSAL_TENDENCY = 0.4

MIN_SAMPLES_FOR_FULL_TRUST = 20  # profile influence ramps in gradually


def _ema(prior: float, observed: float, alpha: float) -> float:
    return (1 - alpha) * prior + alpha * observed


@dataclass
class PersonalityProfile:
    """The Brain's continuously-evolving, per-symbol behavioral fingerprint.

    Every dimension is a 0-1 score updated via exponential moving
    average as new `behavioral_memory.BehavioralObservation`s arrive, so
    the profile adapts gradually rather than reacting to any single
    event — the same smoothing philosophy as `SymbolPersonality`, applied
    to a different, named set of dimensions.
    """

    instrument: str = ""

    # Higher = once a trend starts on this symbol, it tends to keep running.
    trend_persistence: float = NEUTRAL_TREND_PERSISTENCE
    # Higher = pullbacks against the prevailing move tend to run deep
    # (as a fraction of the preceding move) before resuming.
    avg_pullback_depth: float = NEUTRAL_PULLBACK_DEPTH
    # Higher = elevated-volatility regimes on this symbol tend to persist
    # / cluster rather than being one-off spikes.
    volatility_clustering: float = NEUTRAL_VOLATILITY_CLUSTERING
    # Higher = this symbol exhibits liquidity sweeps / stop hunts frequently.
    liquidity_sweep_frequency: float = NEUTRAL_LIQUIDITY_SWEEP_FREQUENCY
    # Higher = this symbol reacts sharply to news/event risk.
    news_sensitivity: float = NEUTRAL_NEWS_SENSITIVITY
    # Higher = this symbol tends to reverse/mean-revert rather than trend.
    reversal_tendency: float = NEUTRAL_REVERSAL_TENDENCY
    # Session dependence: per-session (e.g. "LONDON"/"NY"/"ASIAN"/
    # "OFF_HOURS") outcome-quality score, 0-1 — how favorably this symbol
    # tends to behave during each session, learned rather than hardcoded.
    session_affinity: dict[str, float] = field(default_factory=dict)

    sample_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def trust(self) -> float:
        """How much weight this profile has earned so far, 0-1. Ramps in
        with sample count so a brand-new symbol's still-neutral profile
        cannot swing decisions before it has actually been observed.
        """
        return min(1.0, self.sample_count / MIN_SAMPLES_FOR_FULL_TRUST)

    def best_session(self) -> str | None:
        if not self.session_affinity:
            return None
        return max(self.session_affinity, key=self.session_affinity.get)

    def worst_session(self) -> str | None:
        if not self.session_affinity:
            return None
        return min(self.session_affinity, key=self.session_affinity.get)

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "trend_persistence": round(self.trend_persistence, 4),
            "avg_pullback_depth": round(self.avg_pullback_depth, 4),
            "volatility_clustering": round(self.volatility_clustering, 4),
            "liquidity_sweep_frequency": round(self.liquidity_sweep_frequency, 4),
            "news_sensitivity": round(self.news_sensitivity, 4),
            "reversal_tendency": round(self.reversal_tendency, 4),
            "session_affinity": {k: round(v, 4) for k, v in self.session_affinity.items()},
            "sample_count": self.sample_count,
            "trust": round(self.trust(), 4),
            "created_at": round(self.created_at, 3),
            "last_updated_at": round(self.last_updated_at, 3),
        }


# ── Pure influence functions ────────────────────────────────────────────
#
# Every function below blends its adjustment in proportion to
# `profile.trust()`, so an unobserved/low-sample symbol's neutral
# defaults are a no-op, and a well-observed symbol's learned behavior
# has proportionally more say. None of these ever fully overrides the
# caller's own value — they nudge it.


def confidence_adjustment(profile: PersonalityProfile, base_confidence: float) -> float:
    """Adjust a hypothesis's confidence using learned trend persistence
    vs. reversal tendency: a symbol that reliably continues once trending
    deserves a small confidence boost on trend-following theses; a
    symbol that reliably reverses deserves a small haircut, since its
    own history argues against sustained continuation.
    """
    trust = profile.trust()
    directional_skew = profile.trend_persistence - profile.reversal_tendency  # -1..1
    adjustment = directional_skew * 0.15 * trust
    return max(0.0, min(1.0, base_confidence + adjustment))


def risk_appetite_multiplier(profile: PersonalityProfile) -> float:
    """A multiplier (typically ~0.6-1.2) a risk-sizing caller can apply
    to its own base risk budget for this symbol: symbols with clustering
    volatility, frequent liquidity sweeps, or high news sensitivity earn
    a *reduced* appetite (choppier, more prone to stop-outs); symbols
    with strong, persistent trends and low sweep frequency earn a
    slightly *increased* appetite. Never returns <=0 — this informs
    sizing, it never itself vetoes a trade.
    """
    trust = profile.trust()
    caution = (
        profile.volatility_clustering * 0.5
        + profile.liquidity_sweep_frequency * 0.3
        + profile.news_sensitivity * 0.2
    )
    confidence_boost = profile.trend_persistence * 0.3
    raw = 1.0 - (caution * 0.4) + (confidence_boost * 0.2)
    neutral = 1.0
    blended = (1 - trust) * neutral + trust * raw
    return max(0.5, min(1.25, blended))


def setup_requirement_overrides(
    profile: PersonalityProfile, base: "SetupThresholds"
) -> "SetupThresholds":
    """Tune a symbol's `setup_lifecycle.SetupThresholds` from its learned
    behavior: symbols prone to liquidity sweeps / reversals need more
    sustained confirmation before QDE treats a setup as READY (avoid
    getting swept), while symbols with strong trend persistence and low
    sweep frequency can confirm a little faster (their setups are less
    often faked out). Confirmation-tick counts only ever move by whole
    ticks and never below 1, so this can never bypass the sustained-
    evidence requirement outright.
    """
    from trainer.qde.brain.setup_lifecycle import SetupThresholds

    trust = profile.trust()
    if trust < 0.05:
        return base

    choppiness = (profile.liquidity_sweep_frequency + profile.reversal_tendency) / 2.0
    tick_adjustment = round((choppiness - 0.5) * 2 * trust)  # -1, 0, or +1 typically
    min_ticks = max(1, base.min_confirmation_ticks + tick_adjustment)

    # Steadier, trend-persistent, low-sweep symbols can be trusted at a
    # slightly lower conviction bar; choppier ones need a higher one.
    steadiness = profile.trend_persistence - choppiness  # -1..1
    conviction_adjustment = -steadiness * 0.05 * trust
    ready_min_conviction = max(0.3, min(0.9, base.ready_min_conviction + conviction_adjustment))

    return SetupThresholds(
        min_attention_weight_to_arm=base.min_attention_weight_to_arm,
        arm_min_maturity_rank=base.arm_min_maturity_rank,
        confirm_min_maturity_rank=base.confirm_min_maturity_rank,
        min_confirmation_ticks=min_ticks,
        ready_min_conviction=ready_min_conviction,
        ready_min_confidence=base.ready_min_confidence,
        uncertainty_threshold=base.uncertainty_threshold,
        max_ready_dwell_seconds=base.max_ready_dwell_seconds,
        max_executing_dwell_seconds=base.max_executing_dwell_seconds,
        cooldown_seconds=base.cooldown_seconds,
    )


def expected_holding_duration_seconds(profile: PersonalityProfile, base_seconds: float) -> float:
    """Scale a base expected-holding-duration estimate by learned trend
    persistence vs. pullback depth: symbols that trend persistently
    reward holding longer; symbols with deep, sustained pullbacks (mean
    reversion prone) argue for shorter expected holds. Bounded to
    0.5x-2x the caller's own base estimate — this informs an existing
    estimate, it never invents a duration model from scratch.
    """
    trust = profile.trust()
    skew = profile.trend_persistence - profile.avg_pullback_depth  # -1..1
    multiplier = 1.0 + skew * 0.5 * trust
    multiplier = max(0.5, min(2.0, multiplier))
    return base_seconds * multiplier


__all__ = [
    "PersonalityProfile",
    "DEFAULT_TRAIT_ALPHA",
    "DEFAULT_SESSION_ALPHA",
    "MIN_SAMPLES_FOR_FULL_TRUST",
    "confidence_adjustment",
    "risk_appetite_multiplier",
    "setup_requirement_overrides",
    "expected_holding_duration_seconds",
]
