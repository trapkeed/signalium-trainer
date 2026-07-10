"""S33 — Opportunity Quality Scoring: data model.

All ten factors named in the spec are represented as inputs, each already
normalized to a 0-100 "quality contribution" so this module can stay a
pure aggregator/weighting layer instead of re-deriving logic that already
exists elsewhere (session/liquidity/market-quality engines, correlation
engine, decision benchmarking, setup memory).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class OpportunityBand(str, Enum):
    """Suggested-behavior band for a given score, per spec."""

    IGNORE = "IGNORE"  # 0-40
    OBSERVE = "OBSERVE"  # 40-60
    WATCH_CLOSELY = "WATCH_CLOSELY"  # 60-75
    CANDIDATE_TRADE = "CANDIDATE_TRADE"  # 75-85
    INSTITUTIONAL_QUALITY = "INSTITUTIONAL_QUALITY"  # 85+


def band_for_score(score: float) -> OpportunityBand:
    if score >= 85:
        return OpportunityBand.INSTITUTIONAL_QUALITY
    if score >= 75:
        return OpportunityBand.CANDIDATE_TRADE
    if score >= 60:
        return OpportunityBand.WATCH_CLOSELY
    if score >= 40:
        return OpportunityBand.OBSERVE
    return OpportunityBand.IGNORE


@dataclass
class OpportunityScoreWeights:
    """Relative weight of each of the ten factors. Must sum to 1.0 —
    validated in :meth:`normalized`. Defaults are equal-weighted; callers
    (e.g. research/backtesting) can tune these per instrument class without
    touching the aggregation logic."""

    historical_similarity: float = 0.10
    spread_quality: float = 0.10
    volatility_quality: float = 0.10
    session_quality: float = 0.10
    structure_quality: float = 0.10
    expected_move_quality: float = 0.10
    holding_period_quality: float = 0.10
    confidence_stability: float = 0.10
    cross_asset_confirmation: float = 0.10
    historical_performance: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "historical_similarity": self.historical_similarity,
            "spread_quality": self.spread_quality,
            "volatility_quality": self.volatility_quality,
            "session_quality": self.session_quality,
            "structure_quality": self.structure_quality,
            "expected_move_quality": self.expected_move_quality,
            "holding_period_quality": self.holding_period_quality,
            "confidence_stability": self.confidence_stability,
            "cross_asset_confirmation": self.cross_asset_confirmation,
            "historical_performance": self.historical_performance,
        }

    def normalized(self) -> "OpportunityScoreWeights":
        """Rescale so weights sum to exactly 1.0 (defensive against
        hand-edited configs that don't quite add up)."""
        total = sum(self.as_dict().values())
        if total <= 0:
            raise ValueError("OpportunityScoreWeights must sum to a positive value")
        if abs(total - 1.0) < 1e-9:
            return self
        scaled = {k: v / total for k, v in self.as_dict().items()}
        return OpportunityScoreWeights(**scaled)


@dataclass
class OpportunityScoreInputs:
    """Pre-computed 0-100 quality contributions for each factor, plus
    identity/context fields. Deliberately excludes any notion of
    direction — this scores *whether* an opportunity deserves capital,
    never *which way* to take it.

    Callers are expected to source each field from the system that already
    computes it (session_intelligence, liquidity_intelligence,
    market_quality_engine, correlation_engine, decision_benchmarking,
    setup_memory) rather than recomputing it here.
    """

    symbol: str
    timeframe: str

    historical_similarity: float  # 0-100: how closely this matches winning historical setups
    spread_quality: float  # 0-100: tighter current spread vs. typical/ATR -> higher
    volatility_quality: float  # 0-100: volatility regime favorability for this setup
    session_quality: float  # 0-100: current session/liquidity window quality
    structure_quality: float  # 0-100: market structure / alignment quality
    expected_move_quality: float  # 0-100: attractiveness of expected move size
    holding_period_quality: float  # 0-100: attractiveness of expected holding period
    confidence_stability: float  # 0-100: how stable setup confidence has been over time
    cross_asset_confirmation: float  # 0-100: agreement from correlated instruments
    historical_performance: float  # 0-100: track record of similar setups

    broker_symbol: str = ""
    setup_state: str = ""
    decision_id: str = ""
    lineage_id: str = ""
    raw_context: dict = field(default_factory=dict)

    def clamped(self) -> "OpportunityScoreInputs":
        """Return a copy with every 0-100 field clamped into range —
        upstream engines occasionally overshoot slightly; scoring must not
        silently propagate an out-of-range sub-score."""
        clamp = lambda v: max(0.0, min(100.0, v))  # noqa: E731
        return OpportunityScoreInputs(
            symbol=self.symbol,
            timeframe=self.timeframe,
            historical_similarity=clamp(self.historical_similarity),
            spread_quality=clamp(self.spread_quality),
            volatility_quality=clamp(self.volatility_quality),
            session_quality=clamp(self.session_quality),
            structure_quality=clamp(self.structure_quality),
            expected_move_quality=clamp(self.expected_move_quality),
            holding_period_quality=clamp(self.holding_period_quality),
            confidence_stability=clamp(self.confidence_stability),
            cross_asset_confirmation=clamp(self.cross_asset_confirmation),
            historical_performance=clamp(self.historical_performance),
            broker_symbol=self.broker_symbol,
            setup_state=self.setup_state,
            decision_id=self.decision_id,
            lineage_id=self.lineage_id,
            raw_context=self.raw_context,
        )


@dataclass
class OpportunityScoreResult:
    """A single, persisted scoring event."""

    symbol: str
    timeframe: str
    score: float
    band: OpportunityBand
    sub_scores: dict[str, float]
    weights: dict[str, float]
    broker_symbol: str = ""
    setup_state: str = ""
    decision_id: str = ""
    lineage_id: str = ""
    scored_at: float = field(default_factory=time.time)

    def worth_capital_allocation(self) -> bool:
        """Convenience predicate matching the spec's core question — true
        from WATCH_CLOSELY upward (score >= 60)."""
        return self.score >= 60.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "score": self.score,
            "band": self.band.value,
            "sub_scores": self.sub_scores,
            "weights": self.weights,
            "broker_symbol": self.broker_symbol,
            "setup_state": self.setup_state,
            "decision_id": self.decision_id,
            "lineage_id": self.lineage_id,
            "scored_at": self.scored_at,
            "worth_capital_allocation": self.worth_capital_allocation(),
        }
