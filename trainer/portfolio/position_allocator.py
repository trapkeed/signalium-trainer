"""Phase 20 — Position Allocator.

Recommends normalized portfolio allocation weights considering:
  - Forecast confidence
  - Expected return
  - Expected risk
  - Correlation
  - Drawdown contribution
  - Risk budget
  - Portfolio diversification

Output: normalized allocation weights summing to 1.0 (long side)
or bi-directionally signed weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from trainer.portfolio.portfolio_instruments import ModelForecast
from trainer.portfolio.correlation_engine import CorrelationMatrix
from trainer.portfolio.risk_budget import RiskBudgetConfig, RiskBudgetEngine
from trainer.portfolio.regime_detector import RegimeAssessment, MarketRegime


class AllocationMethod(str, Enum):
    EQUAL_WEIGHT = "equal_weight"
    CONFIDENCE_WEIGHT = "confidence_weight"
    RISK_PARITY = "risk_parity"
    RETURN_SCALED = "return_scaled"
    SHARPE_SCALED = "sharpe_scaled"
    MINIMUM_VARIANCE = "minimum_variance"
    HYBRID = "hybrid"


@dataclass
class AllocationResult:
    weights: Dict[str, float]
    signed_weights: Dict[str, float]
    method: str
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    diversification_ratio: float
    budget_feasible: bool
    budget_warnings: List[str] = field(default_factory=list)
    regime: str = "unknown"
    instrument_count: int = 0

    def normalised_long_weights(self) -> Dict[str, float]:
        longs = {k: v for k, v in self.weights.items() if v > 0}
        total = sum(longs.values())
        if total < 1e-9:
            return {}
        return {k: round(v / total, 4) for k, v in longs.items()}

    def normalised_short_weights(self) -> Dict[str, float]:
        shorts = {k: abs(v) for k, v in self.weights.items() if v < 0}
        total = sum(shorts.values())
        if total < 1e-9:
            return {}
        return {k: round(v / total, 4) for k, v in shorts.items()}

    def to_dict(self) -> dict:
        return {
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "signed_weights": {k: round(v, 4) for k, v in self.signed_weights.items()},
            "method": self.method,
            "expected_return": round(self.expected_return, 4),
            "expected_volatility": round(self.expected_volatility, 4),
            "expected_sharpe": round(self.expected_sharpe, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "budget_feasible": self.budget_feasible,
            "budget_warnings": self.budget_warnings,
            "regime": self.regime,
            "instrument_count": self.instrument_count,
            "long_weights": self.normalised_long_weights(),
            "short_weights": self.normalised_short_weights(),
        }


class PositionAllocator:
    """Allocates portfolio positions across model forecasts."""

    RISK_FREE_RATE = 0.045

    def __init__(
        self,
        method: AllocationMethod = AllocationMethod.HYBRID,
        budget_config: Optional[RiskBudgetConfig] = None,
        risk_free_rate: float = 0.045,
    ) -> None:
        self._method = method
        self._budget_engine = RiskBudgetEngine(budget_config or RiskBudgetConfig())
        self._rfr = risk_free_rate

    def allocate(
        self,
        forecasts: List[ModelForecast],
        correlation_matrix: Optional[CorrelationMatrix] = None,
        regime: Optional[RegimeAssessment] = None,
        method: Optional[AllocationMethod] = None,
    ) -> AllocationResult:
        if not forecasts:
            return self._empty_result(method or self._method)

        m = method or self._method
        active = [f for f in forecasts if f.direction != "NEUTRAL"]
        if not active:
            return self._empty_result(m)

        # Compute raw weights by method
        if m == AllocationMethod.EQUAL_WEIGHT:
            raw = self._equal_weight(active)
        elif m == AllocationMethod.CONFIDENCE_WEIGHT:
            raw = self._confidence_weight(active)
        elif m == AllocationMethod.RISK_PARITY:
            raw = self._risk_parity(active)
        elif m == AllocationMethod.RETURN_SCALED:
            raw = self._return_scaled(active)
        elif m == AllocationMethod.SHARPE_SCALED:
            raw = self._sharpe_scaled(active)
        elif m == AllocationMethod.MINIMUM_VARIANCE:
            raw = self._minimum_variance(active, correlation_matrix)
        else:
            raw = self._hybrid(active, correlation_matrix, regime)

        # Apply regime adjustments
        if regime is not None:
            raw = self._apply_regime(raw, active, regime)

        # Budget enforcement
        signed = self._to_signed(raw, active)
        scaled = self._budget_engine.scale_to_budget(signed, active)

        # Renormalize to sum(abs) == 1.0 after budget capping
        total_abs = sum(abs(v) for v in scaled.values())
        if total_abs > 1e-9:
            scaled = {k: v / total_abs for k, v in scaled.items()}

        # Portfolio metrics
        er, ev, es = self._portfolio_stats(active, scaled, correlation_matrix)
        div = self._diversification_ratio(scaled, active, correlation_matrix)

        budget_result = self._budget_engine.check(scaled, active)

        return AllocationResult(
            weights={k: abs(v) for k, v in scaled.items()},
            signed_weights=scaled,
            method=m.value,
            expected_return=er,
            expected_volatility=ev,
            expected_sharpe=es,
            diversification_ratio=div,
            budget_feasible=budget_result.is_feasible(),
            budget_warnings=[b.name for b in budget_result.warnings + budget_result.breaches],
            regime=regime.primary_regime.value if regime else "unknown",
            instrument_count=len(active),
        )

    # ── Allocation methods ─────────────────────────────────────────────────

    @staticmethod
    def _equal_weight(forecasts: List[ModelForecast]) -> Dict[str, float]:
        n = len(forecasts)
        return {f.instrument: 1.0 / n for f in forecasts}

    @staticmethod
    def _confidence_weight(forecasts: List[ModelForecast]) -> Dict[str, float]:
        total_conf = sum(f.confidence for f in forecasts)
        if total_conf < 1e-9:
            return PositionAllocator._equal_weight(forecasts)
        return {f.instrument: f.confidence / total_conf for f in forecasts}

    @staticmethod
    def _risk_parity(forecasts: List[ModelForecast]) -> Dict[str, float]:
        inv_risk = {}
        for f in forecasts:
            risk = max(f.expected_risk, 1e-4)
            inv_risk[f.instrument] = 1.0 / risk
        total = sum(inv_risk.values())
        return {k: v / total for k, v in inv_risk.items()}

    @staticmethod
    def _return_scaled(forecasts: List[ModelForecast]) -> Dict[str, float]:
        scores = {f.instrument: max(abs(f.signed_return()) * f.confidence, 1e-6) for f in forecasts}
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def _sharpe_scaled(self, forecasts: List[ModelForecast]) -> Dict[str, float]:
        scores = {}
        for f in forecasts:
            ret = f.signed_return()
            risk = max(f.expected_risk, 1e-4)
            sharpe = max((ret - self._rfr) / risk, 1e-6)
            scores[f.instrument] = sharpe * f.confidence
        total = sum(scores.values())
        if total < 1e-9:
            return self._equal_weight(forecasts)
        return {k: v / total for k, v in scores.items()}

    @staticmethod
    def _minimum_variance(
        forecasts: List[ModelForecast], matrix: Optional[CorrelationMatrix]
    ) -> Dict[str, float]:
        if matrix is None or len(forecasts) < 2:
            return PositionAllocator._risk_parity(forecasts)
        n = len(forecasts)
        syms = [f.instrument for f in forecasts]
        risks = {f.instrument: f.expected_risk for f in forecasts}

        # Greedy minimum-variance approximation: penalize high correlation
        weights: Dict[str, float] = {}
        for i, sym in enumerate(syms):
            avg_corr = sum(abs(matrix.get(sym, syms[j])) for j in range(n) if j != i) / max(
                n - 1, 1
            )
            risk_penalty = risks[sym] * (1 + avg_corr)
            weights[sym] = 1.0 / max(risk_penalty, 1e-6)

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _hybrid(
        self,
        forecasts: List[ModelForecast],
        matrix: Optional[CorrelationMatrix],
        regime: Optional[RegimeAssessment],
    ) -> Dict[str, float]:
        w_conf = self._confidence_weight(forecasts)
        w_risk = self._risk_parity(forecasts)
        w_sharpe = self._sharpe_scaled(forecasts)

        blend: Dict[str, float] = {}
        for f in forecasts:
            sym = f.instrument
            blend[sym] = (
                0.40 * w_conf.get(sym, 0.0)
                + 0.30 * w_risk.get(sym, 0.0)
                + 0.30 * w_sharpe.get(sym, 0.0)
            )
        total = sum(blend.values())
        return {k: v / max(total, 1e-9) for k, v in blend.items()}

    @staticmethod
    def _apply_regime(
        weights: Dict[str, float], forecasts: List[ModelForecast], regime: RegimeAssessment
    ) -> Dict[str, float]:
        pos_scale = regime.adjustment_factors.get("position_size", 1.0)
        result = {k: v * pos_scale for k, v in weights.items()}
        # Re-normalize
        total = sum(result.values())
        if total < 1e-9:
            return weights
        return {k: v / total for k, v in result.items()}

    @staticmethod
    def _to_signed(weights: Dict[str, float], forecasts: List[ModelForecast]) -> Dict[str, float]:
        fc_map = {f.instrument: f for f in forecasts}
        signed = {}
        for sym, w in weights.items():
            fc = fc_map.get(sym)
            if fc is None:
                signed[sym] = w
            elif fc.direction == "SELL":
                signed[sym] = -w
            else:
                signed[sym] = w
        return signed

    def _portfolio_stats(
        self,
        forecasts: List[ModelForecast],
        weights: Dict[str, float],
        matrix: Optional[CorrelationMatrix],
    ) -> tuple:
        total_w = sum(abs(w) for w in weights.values())
        if total_w < 1e-9:
            return 0.0, 0.0, 0.0
        norm_w = {k: v / total_w for k, v in weights.items()}
        fc_map = {f.instrument: f for f in forecasts}

        er = sum(norm_w.get(fc.instrument, 0.0) * fc.signed_return() for fc in forecasts)
        var = 0.0
        syms = [f.instrument for f in forecasts]
        for i, si in enumerate(syms):
            wi = abs(norm_w.get(si, 0.0)) * fc_map[si].expected_risk
            for j, sj in enumerate(syms):
                wj = abs(norm_w.get(sj, 0.0)) * fc_map[sj].expected_risk
                rho = matrix.get(si, sj) if matrix and i != j else (1.0 if i == j else 0.0)
                var += wi * wj * rho
        ev = math.sqrt(max(var, 0.0))
        es = (er - self._rfr) / max(ev, 1e-6)
        return er, ev, es

    @staticmethod
    def _diversification_ratio(
        weights: Dict[str, float],
        forecasts: List[ModelForecast],
        matrix: Optional[CorrelationMatrix],
    ) -> float:
        weights_abs = {k: abs(v) for k, v in weights.items()}
        total = sum(weights_abs.values())
        if total < 1e-9:
            return 0.0
        n = len([v for v in weights_abs.values() if v > 1e-9])
        max_w = max(weights_abs.values()) / total
        return round(1.0 - max_w, 4)

    @staticmethod
    def _empty_result(method: AllocationMethod) -> AllocationResult:
        return AllocationResult(
            weights={},
            signed_weights={},
            method=method.value,
            expected_return=0.0,
            expected_volatility=0.0,
            expected_sharpe=0.0,
            diversification_ratio=0.0,
            budget_feasible=True,
            instrument_count=0,
        )
