"""Phase 24.9 — Market Quality Engine (9 sub-scores, each 0-100)."""

from __future__ import annotations

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import (
    LiquidityResult,
    MarketInput,
    MarketQualityResult,
    MomentumResult,
    NewsIntelligenceResult,
    PredictionReliabilityResult,
    SessionResult,
    TrendResult,
    VolatilityResult,
)
from trainer.mie.mie_types import clamp


class MarketQualityEngine:
    """
    Aggregates all sub-engine results into 9 quality dimensions (0-100 each)
    and one overall Market Quality score.

    Weights for overall Market Quality:
        execution_quality  : 0.15
        prediction_quality : 0.15
        liquidity_quality  : 0.15
        volatility_quality : 0.15
        momentum_quality   : 0.10
        trend_quality      : 0.10
        session_quality    : 0.10
        news_impact_inv    : 0.10  (100 - news_impact_score)
    """

    _WEIGHTS = {
        "execution_quality": 0.15,
        "prediction_quality": 0.15,
        "liquidity_quality": 0.15,
        "volatility_quality": 0.15,
        "momentum_quality": 0.10,
        "trend_quality": 0.10,
        "session_quality": 0.10,
        "news_inv": 0.10,
    }

    assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        inp: MarketInput,
        volatility: VolatilityResult | None = None,
        momentum: MomentumResult | None = None,
        trend: TrendResult | None = None,
        liquidity: LiquidityResult | None = None,
        session: SessionResult | None = None,
        news: NewsIntelligenceResult | None = None,
        reliability: PredictionReliabilityResult | None = None,
    ) -> MarketQualityResult:

        vol_q = volatility.quality if volatility else 50.0
        mom_q = momentum.quality if momentum else 50.0
        trend_q = trend.quality if trend else 50.0
        liq_q = liquidity.quality if liquidity else 50.0
        sess_q = session.quality if session else 50.0
        news_imp = news.impact_score if news else 0.0
        pred_q = reliability.reliability_score if reliability else 50.0

        exec_q = self._execution_quality(inp, liq_q, sess_q)
        news_q = clamp(100.0 - news_imp, 0.0, 100.0)

        overall = (
            exec_q * self._WEIGHTS["execution_quality"]
            + pred_q * self._WEIGHTS["prediction_quality"]
            + liq_q * self._WEIGHTS["liquidity_quality"]
            + vol_q * self._WEIGHTS["volatility_quality"]
            + mom_q * self._WEIGHTS["momentum_quality"]
            + trend_q * self._WEIGHTS["trend_quality"]
            + sess_q * self._WEIGHTS["session_quality"]
            + news_q * self._WEIGHTS["news_inv"]
        )

        logger.debug(
            f"[MQE] {inp.symbol} overall={overall:.1f} "
            f"exec={exec_q:.0f} pred={pred_q:.0f} liq={liq_q:.0f} "
            f"vol={vol_q:.0f} mom={mom_q:.0f} trend={trend_q:.0f}"
        )
        return MarketQualityResult(
            market_quality=clamp(overall, 0.0, 100.0),
            execution_quality=clamp(exec_q, 0.0, 100.0),
            prediction_quality=clamp(pred_q, 0.0, 100.0),
            liquidity_quality=clamp(liq_q, 0.0, 100.0),
            volatility_quality=clamp(vol_q, 0.0, 100.0),
            momentum_quality=clamp(mom_q, 0.0, 100.0),
            trend_quality=clamp(trend_q, 0.0, 100.0),
            session_quality=clamp(sess_q, 0.0, 100.0),
            news_impact_score=clamp(news_imp, 0.0, 100.0),
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _execution_quality(
        self,
        inp: MarketInput,
        liq_q: float,
        sess_q: float,
    ) -> float:
        """Composite execution quality from spread, liquidity, and session."""
        spread_frac = inp.spread / max(inp.atr, 1e-9)
        spread_score = clamp(100.0 - spread_frac * 200.0, 0.0, 100.0)
        return spread_score * 0.40 + liq_q * 0.35 + sess_q * 0.25
