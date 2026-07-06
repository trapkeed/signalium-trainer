"""Phase 24 — Capital Allocation Engine.

Determines dynamic capital allocation using 15+ factors:
forecast_confidence, research_score, execution_score, expected_value,
expected_return, expected_drawdown, portfolio_exposure, correlation,
market_regime, volatility, liquidity, model_confidence,
ensemble_confidence, decision_quality, historical_performance.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.icari.icari_types import AllocationStatus, AllocationRecommendation
from trainer.icari.icari_models import AllocationRequest, AllocationResult

# ── Regime multipliers ─────────────────────────────────────────────────────────
_REGIME_MULT = {
    "TRENDING": 1.10,
    "NORMAL": 1.00,
    "RANGING": 0.85,
    "VOLATILE": 0.65,
}

# ── Allocation factor weights (must sum to 1.0) ───────────────────────────────
_WEIGHTS = {
    "forecast_confidence": 0.15,
    "research_score": 0.12,
    "execution_score": 0.10,
    "expected_value": 0.10,
    "expected_return": 0.10,
    "model_confidence": 0.10,
    "ensemble_confidence": 0.08,
    "decision_quality": 0.08,
    "liquidity": 0.07,
    "volatility_adj": 0.05,
    "correlation_adj": 0.05,
}

assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "Factor weights must sum to 1.0"

_MAX_ALLOCATION_FRAC = 0.25  # never allocate more than 25% of capital
_MIN_CONFIDENCE_THRESHOLD = 0.30


class AllocationEngine:
    """Dynamic capital allocation engine."""

    def __init__(
        self,
        max_allocation_frac: float = _MAX_ALLOCATION_FRAC,
        min_confidence: float = _MIN_CONFIDENCE_THRESHOLD,
        base_frac: float = 0.10,  # baseline allocation
    ):
        self._max_frac = max_allocation_frac
        self._min_conf = min_confidence
        self._base_frac = base_frac
        self._history: List[AllocationResult] = []

    def allocate(self, req: AllocationRequest) -> AllocationResult:
        """Compute the capital allocation for a request."""
        factors = self._compute_factors(req)
        confidence = self._composite_confidence(factors)

        # Hard denial
        if confidence < self._min_conf:
            result = AllocationResult(
                instrument=req.instrument,
                status=AllocationStatus.DENIED,
                allocation_frac=0.0,
                allocation_usd=0.0,
                confidence=confidence,
                rationale=f"Composite confidence {confidence:.2f} below threshold {self._min_conf:.2f}",
                factors=factors,
                recommendations=[AllocationRecommendation.SKIP_TRADE],
            )
            self._history.append(result)
            return result

        # Base allocation scaled by confidence and regime
        regime_mult = _REGIME_MULT.get(req.market_regime.upper(), 1.0)
        raw_frac = self._base_frac * confidence * regime_mult

        # Exposure reduction
        exposure_penalty = max(0.0, req.portfolio_exposure - 0.50) * 0.5
        raw_frac *= 1.0 - exposure_penalty

        # Expected drawdown penalty
        if req.expected_drawdown > 0.10:
            raw_frac *= 0.80
        if req.expected_drawdown > 0.20:
            raw_frac *= 0.60

        # Cap
        frac = min(raw_frac, self._max_frac)
        frac = max(0.0, frac)

        status = AllocationStatus.APPROVED
        recs: List[AllocationRecommendation] = []

        if frac < raw_frac:
            status = AllocationStatus.REDUCED
            recs.append(AllocationRecommendation.REDUCE_PORTFOLIO_RISK)

        if req.correlation > 0.70:
            recs.append(AllocationRecommendation.REDUCE_CORRELATED)

        if req.portfolio_exposure > 0.80:
            recs.append(AllocationRecommendation.INCREASE_DIVERSIFICATION)

        result = AllocationResult(
            instrument=req.instrument,
            status=status,
            allocation_frac=round(frac, 4),
            allocation_usd=round(frac * req.capital, 2),
            confidence=round(confidence, 4),
            rationale=self._rationale(frac, confidence, regime_mult, exposure_penalty),
            factors=factors,
            recommendations=recs,
        )
        self._history.append(result)
        logger.debug(f"AllocationEngine: {req.instrument} frac={frac:.4f} status={status.value}")
        return result

    def allocate_batch(self, requests: List[AllocationRequest]) -> List[AllocationResult]:
        return [self.allocate(r) for r in requests]

    def _compute_factors(self, req: AllocationRequest) -> Dict[str, float]:
        vol_adj = max(0.0, 1.0 - req.volatility * 20)  # penalise high vol
        corr_adj = max(0.0, 1.0 - req.correlation)  # penalise high correlation

        # Expected value normalised: clamp 0–1
        ev_norm = max(0.0, min(1.0, (req.expected_value + 0.01) / 0.02))

        # Expected return normalised
        er_norm = max(0.0, min(1.0, (req.expected_return + 0.02) / 0.04))

        return {
            "forecast_confidence": req.forecast_confidence,
            "research_score": req.research_score,
            "execution_score": req.execution_score,
            "expected_value": ev_norm,
            "expected_return": er_norm,
            "model_confidence": req.model_confidence,
            "ensemble_confidence": req.ensemble_confidence,
            "decision_quality": req.decision_quality,
            "liquidity": req.liquidity,
            "volatility_adj": vol_adj,
            "correlation_adj": corr_adj,
        }

    def _composite_confidence(self, factors: Dict[str, float]) -> float:
        return sum(_WEIGHTS.get(k, 0.0) * v for k, v in factors.items())

    def _rationale(self, frac: float, conf: float, regime_mult: float, exp_pen: float) -> str:
        parts = [f"composite_confidence={conf:.2f}", f"regime_mult={regime_mult:.2f}"]
        if exp_pen > 0:
            parts.append(f"exposure_penalty={exp_pen:.2f}")
        parts.append(f"allocation={frac*100:.1f}%")
        return "; ".join(parts)

    def history(self, last_n: int = 50) -> List[AllocationResult]:
        return self._history[-last_n:]

    def denied_count(self) -> int:
        return sum(1 for r in self._history if r.status == AllocationStatus.DENIED)

    def average_confidence(self) -> float:
        if not self._history:
            return 0.0
        return sum(r.confidence for r in self._history) / len(self._history)

    def factor_weights(self) -> Dict[str, float]:
        return dict(_WEIGHTS)
