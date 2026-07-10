"""QDE Brain — narrative synthesis.

Builds and incrementally revises the "story" the Brain tells about a
symbol from `SymbolObservationInput`/`ObservationResult` fields it already
receives — this module composes existing typed context, it does not
re-derive volatility/liquidity/regime classification itself (that stays
owned by whatever upstream engine produced those fields).
"""

from __future__ import annotations

import time

from trainer.qde.brain.brain_models import MarketNarrative
from trainer.qde.brain.brain_types import NarrativeTone, TrendMaturity


class NarrativeEngine:
    """Stateless — operates on the `MarketNarrative` held by `BrainState`,
    revising it in place so tone/summary changes read as continuity rather
    than being recomputed from a blank slate every tick.
    """

    def update(
        self,
        prior: MarketNarrative | None,
        instrument: str,
        market_regime: str,
        volatility_state: str,
        liquidity_state: str,
        session_quality: float,
        readiness_score: float,
        dwell_seconds: float,
        observer_state: str,
        now: float | None = None,
    ) -> MarketNarrative:
        now = now if now is not None else time.time()
        narrative = prior or MarketNarrative(instrument=instrument, created_at=now)

        tone = self._infer_tone(market_regime, readiness_score, observer_state)
        maturity = self._infer_trend_maturity(market_regime, dwell_seconds, observer_state)
        session_context = self._session_context(session_quality)

        changed = (
            tone != narrative.tone
            or maturity != narrative.trend_maturity
            or market_regime != narrative.market_regime
            or volatility_state != narrative.volatility_state
            or liquidity_state != narrative.liquidity_state
        )

        narrative.tone = tone
        narrative.trend_maturity = maturity
        narrative.session_context = session_context
        narrative.volatility_state = volatility_state
        narrative.liquidity_state = liquidity_state
        narrative.market_regime = market_regime
        narrative.summary = self._summarize(
            instrument,
            tone,
            maturity,
            market_regime,
            volatility_state,
            liquidity_state,
            session_context,
        )
        narrative.updated_at = now
        if changed:
            narrative.revision_count += 1
            narrative.structure_notes.append(
                f"{time.strftime('%H:%M:%S', time.gmtime(now))} -> {narrative.summary}"
            )

        return narrative

    def _infer_tone(
        self, market_regime: str, readiness_score: float, observer_state: str
    ) -> NarrativeTone:
        regime = market_regime.upper()
        if observer_state in ("ARMED", "ENTRY_WINDOW") and readiness_score >= 0.65:
            return NarrativeTone.BREAKOUT_PENDING
        if regime in ("TREND_UP",):
            return NarrativeTone.BULLISH
        if regime in ("TREND_DOWN",):
            return NarrativeTone.BEARISH
        if regime in ("RANGE", "CHOP", "CHOPPY"):
            return NarrativeTone.CHOPPY
        if observer_state == "COOLDOWN" and readiness_score < 0.2:
            return NarrativeTone.EXHAUSTION
        return NarrativeTone.NEUTRAL

    def _infer_trend_maturity(
        self, market_regime: str, dwell_seconds: float, observer_state: str
    ) -> TrendMaturity:
        regime = market_regime.upper()
        if regime not in ("TREND_UP", "TREND_DOWN"):
            return TrendMaturity.NONE
        # Purely descriptive dwell-time buckets — never a trading rule.
        if dwell_seconds < 900:
            return TrendMaturity.EARLY
        if dwell_seconds < 3600 * 4:
            return TrendMaturity.ESTABLISHED
        if dwell_seconds < 3600 * 12:
            return TrendMaturity.LATE
        return TrendMaturity.EXHAUSTED

    def _session_context(self, session_quality: float) -> str:
        if session_quality >= 0.75:
            return "prime session liquidity"
        if session_quality >= 0.45:
            return "moderate session liquidity"
        return "thin/off-hours liquidity"

    def _summarize(
        self,
        instrument: str,
        tone: NarrativeTone,
        maturity: TrendMaturity,
        regime: str,
        volatility_state: str,
        liquidity_state: str,
        session_context: str,
    ) -> str:
        maturity_clause = (
            f", trend {maturity.value.lower()}" if maturity != TrendMaturity.NONE else ""
        )
        return (
            f"{instrument} is {tone.value.lower()} in a {regime.lower()} regime"
            f"{maturity_clause}; volatility {volatility_state.lower()}, "
            f"liquidity {liquidity_state.lower()}, {session_context}."
        )
