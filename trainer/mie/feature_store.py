"""Phase 24.9 — Feature Store Export.

Exports all MIE scores as a 12-dimensional feature vector for the
Deep Learning model. Every MIE run should produce a FeatureStoreExport
that is persisted alongside the model's input features.

Feature vector (12 dimensions, all normalised 0-1):
    [0]  market_regime_enc    (encoded regime ordinal / 15)
    [1]  risk_regime_enc      (encoded regime ordinal / 11)
    [2]  tradeability         (score / 100)
    [3]  execution_quality    (score / 100)
    [4]  market_quality       (score / 100)
    [5]  prediction_reliability (score / 100)
    [6]  behavior_similarity  (score / 100)
    [7]  calibration          (score / 100)
    [8]  historical_success   (win_rate 0-1)
    [9]  trend_quality        (score / 100)
    [10] momentum_quality     (score / 100)
    [11] liquidity_quality    (score / 100)
"""

from __future__ import annotations

from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.mie.mie_models import (
    BehaviorSimilarityResult,
    CalibrationResult,
    FeatureStoreExport,
    LiquidityResult,
    MarketInput,
    MarketQualityResult,
    MarketRegimeResult,
    MomentumResult,
    PredictionReliabilityResult,
    RiskIntelligenceResult,
    TradeabilityResult,
    TrendResult,
)
from trainer.mie.mie_types import MarketRegime, RiskRegime

_REGIME_ORDINALS = {r.value: i for i, r in enumerate(MarketRegime)}
_RISK_ORDINALS = {r.value: i for i, r in enumerate(RiskRegime)}


class FeatureStoreExporter:
    """
    Collects all sub-engine outputs and produces a FeatureStoreExport
    suitable for appending to the model's feature matrix.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def export(
        self,
        inp: MarketInput,
        regime: Optional[MarketRegimeResult] = None,
        risk: Optional[RiskIntelligenceResult] = None,
        tradeability: Optional[TradeabilityResult] = None,
        quality: Optional[MarketQualityResult] = None,
        reliability: Optional[PredictionReliabilityResult] = None,
        similarity: Optional[BehaviorSimilarityResult] = None,
        calibration: Optional[CalibrationResult] = None,
        trend: Optional[TrendResult] = None,
        momentum: Optional[MomentumResult] = None,
        liquidity: Optional[LiquidityResult] = None,
    ) -> FeatureStoreExport:

        regime_enc = _REGIME_ORDINALS.get(
            regime.regime.value if regime else MarketRegime.UNKNOWN.value, 15
        )
        risk_enc = _RISK_ORDINALS.get(risk.regime.value if risk else RiskRegime.UNKNOWN.value, 11)

        feat = FeatureStoreExport(
            symbol=inp.symbol,
            market_regime_enc=float(regime_enc),
            risk_regime_enc=float(risk_enc),
            tradeability=tradeability.score if tradeability else 0.0,
            execution_quality=quality.execution_quality if quality else 50.0,
            market_quality=quality.market_quality if quality else 50.0,
            prediction_reliability=reliability.reliability_score if reliability else 50.0,
            behavior_similarity=similarity.similarity_score if similarity else 0.0,
            calibration=calibration.calibration_score if calibration else 50.0,
            historical_success=(
                similarity.historical_success if similarity else inp.historical_win_rate
            ),
            trend_quality=trend.quality if trend else 50.0,
            momentum_quality=momentum.quality if momentum else 50.0,
            liquidity_quality=liquidity.quality if liquidity else 50.0,
        )

        logger.debug(
            f"[FSE] {inp.symbol} exported 12-dim feature vector: "
            f"trade={feat.tradeability:.1f} mq={feat.market_quality:.1f} "
            f"calib={feat.calibration:.1f}"
        )
        return feat
