"""S33 — Opportunity Quality Scoring Engine.

Assigns every observed setup an opportunity score from 0-100 that answers
"is this worth capital allocation?" — never "which direction?". The
engine is a pure weighted aggregator over ten pre-computed 0-100 quality
contributions (see ``models.OpportunityScoreInputs``); it does not itself
compute session quality, spread quality, correlation, etc. — those already
exist elsewhere in the codebase (MIE intelligence engines, correlation
engine, decision benchmarking, setup memory) and are passed in.
"""

from __future__ import annotations

from typing import Callable

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

from trainer.qde.opportunity_scoring.models import (
    OpportunityScoreInputs,
    OpportunityScoreResult,
    OpportunityScoreWeights,
    band_for_score,
)
from trainer.qde.opportunity_scoring.store import OpportunityScoreStore


class OpportunityScoringEngine:
    """Public API for Opportunity Quality Scoring.

    Args:
        store: Injectable :class:`OpportunityScoreStore` (tests / custom path).
        weights: Optional custom factor weighting (defaults to equal weight).
        on_score: Optional callback invoked with every
            :class:`OpportunityScoreResult` — the integration seam for
            research/memory systems (e.g. push into
            ``KnowledgeBase``/``DecisionLineage`` without this module
            importing them directly and risking a dependency cycle).
    """

    def __init__(
        self,
        store: OpportunityScoreStore | None = None,
        weights: OpportunityScoreWeights | None = None,
        on_score: "Callable[[OpportunityScoreResult], None] | None" = None,
    ) -> None:
        self._store = store if store is not None else OpportunityScoreStore()
        self._weights = (weights if weights is not None else OpportunityScoreWeights()).normalized()
        self._on_score = on_score

    @property
    def weights(self) -> OpportunityScoreWeights:
        return self._weights

    def score(self, inputs: OpportunityScoreInputs) -> OpportunityScoreResult:
        """Compute, persist, and return the opportunity score for one
        observation. Every call appends a new row to score history — this
        is intentionally a time series, since the same setup is expected
        to be (re)scored repeatedly as it matures through the Setup Memory
        Engine's lifecycle."""
        clamped = inputs.clamped()
        weights = self._weights.as_dict()
        sub_scores = {
            "historical_similarity": clamped.historical_similarity,
            "spread_quality": clamped.spread_quality,
            "volatility_quality": clamped.volatility_quality,
            "session_quality": clamped.session_quality,
            "structure_quality": clamped.structure_quality,
            "expected_move_quality": clamped.expected_move_quality,
            "holding_period_quality": clamped.holding_period_quality,
            "confidence_stability": clamped.confidence_stability,
            "cross_asset_confirmation": clamped.cross_asset_confirmation,
            "historical_performance": clamped.historical_performance,
        }
        total = sum(sub_scores[factor] * weight for factor, weight in weights.items())
        total = max(0.0, min(100.0, total))
        band = band_for_score(total)

        result = OpportunityScoreResult(
            symbol=clamped.symbol,
            timeframe=clamped.timeframe,
            score=total,
            band=band,
            sub_scores=sub_scores,
            weights=weights,
            broker_symbol=clamped.broker_symbol,
            setup_state=clamped.setup_state,
            decision_id=clamped.decision_id,
            lineage_id=clamped.lineage_id,
        )
        self._store.append(result)
        logger.info(
            "opportunity_score: {}::{} = {:.1f} ({})",
            result.symbol,
            result.timeframe,
            result.score,
            result.band.value,
        )
        if self._on_score is not None:
            try:
                self._on_score(result)
            except Exception as exc:  # pragma: no cover - integration hook must not break scoring
                logger.warning("opportunity_score: on_score hook failed: {}", exc)
        return result

    # ── History / research access ────────────────────────────────────────

    def history(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> list[OpportunityScoreResult]:
        return self._store.history(symbol, timeframe, limit)

    def history_for_decision(self, decision_id: str) -> list[OpportunityScoreResult]:
        return self._store.by_decision(decision_id)

    def recent(self, limit: int = 100) -> list[OpportunityScoreResult]:
        return self._store.recent(limit)
