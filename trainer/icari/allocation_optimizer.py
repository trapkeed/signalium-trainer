"""Phase 24 — Allocation Optimizer.

Optimizes allocations for 8 objectives using grid search (pure Python):
Max Sharpe, Max Sortino, Min Drawdown, Max Diversification,
Risk Parity, Balanced Portfolio, Capital Preservation, Expected Utility.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.icari.icari_types import OptimizationObjective
from trainer.icari.icari_models import OptimizationResult


def _mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def _std(data: List[float]) -> float:
    return statistics.stdev(data) if len(data) >= 2 else 0.0


def _portfolio_returns(
    returns: Dict[str, List[float]],
    weights: Dict[str, float],
) -> List[float]:
    instruments = list(weights.keys())
    if not instruments:
        return []
    min_len = min(len(returns.get(ins, [])) for ins in instruments)
    if min_len == 0:
        return []
    return [
        sum(weights.get(ins, 0.0) * returns[ins][i] for ins in instruments if ins in returns)
        for i in range(min_len)
    ]


def _sharpe(port_r: List[float], rfr: float = 0.0) -> float:
    mu  = _mean(port_r) - rfr / 252
    std = _std(port_r)
    return (mu / std * math.sqrt(252)) if std > 0 else 0.0


def _sortino(port_r: List[float], rfr: float = 0.0) -> float:
    mu       = _mean(port_r) - rfr / 252
    downside = [r for r in port_r if r < rfr / 252]
    d_std    = _std(downside) if len(downside) >= 2 else _std(port_r)
    return (mu / d_std * math.sqrt(252)) if d_std > 0 else 0.0


def _max_drawdown(port_r: List[float]) -> float:
    peak = trough = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for r in port_r:
        cumulative += r
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _diversification_ratio(
    returns: Dict[str, List[float]],
    weights: Dict[str, float],
) -> float:
    instruments = list(weights.keys())
    indiv_vols  = {ins: _std(returns.get(ins, [0.0])) for ins in instruments}
    weighted_vol_sum = sum(weights.get(ins, 0.0) * indiv_vols[ins] for ins in instruments)
    port_r = _portfolio_returns(returns, weights)
    port_vol = _std(port_r)
    return weighted_vol_sum / max(port_vol, 1e-9)


def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _grid_weights(
    instruments: List[str],
    steps:       int   = 5,
    max_weight:  float = 1.0,
) -> List[Dict[str, float]]:
    """Generate a coarse grid of weight vectors for small N.

    `max_weight` caps the dominant share so generated candidates already
    respect the per-instrument maximum (no post-hoc projection needed for
    these candidates).
    """
    n = len(instruments)
    if n == 0:
        return []
    if n == 1:
        return [{instruments[0]: 1.0}]

    cap = min(max_weight, 1.0)

    # Uniform: equal weight
    grids = [{ins: 1.0 / n for ins in instruments}]

    # Single dominant: cap% to one, rest equal
    dom = min(cap, 0.60)
    rest = (1.0 - dom) / max(n - 1, 1)
    for ins in instruments:
        w = {k: rest for k in instruments}
        w[ins] = dom
        grids.append(w)

    # Pairs dominant: half-cap each, rest equal
    if n >= 2:
        pair_share = min(cap, 0.40)
        other = max(0.0, 1.0 - 2 * pair_share) / max(n - 2, 1)
        for i in range(n):
            for j in range(i + 1, n):
                w = {k: other for k in instruments}
                w[instruments[i]] = pair_share
                w[instruments[j]] = pair_share
                grids.append(w)

    # Alpha sweep (capped at max_weight)
    for step in range(1, steps + 1):
        alpha = min(cap, step / steps)
        for ins in instruments:
            w = {k: (1.0 - alpha) / max(n - 1, 1) for k in instruments}
            w[ins] = alpha
            grids.append(w)

    # Normalise all
    return [_normalise(g) for g in grids]


class AllocationOptimizer:
    """Grid-search portfolio optimizer — pure Python, no scipy."""

    def __init__(
        self,
        risk_free_rate:  float = 0.045,   # annual
        max_weight:      float = 0.50,    # max weight per instrument
        min_weight:      float = 0.0,
        grid_steps:      int   = 5,
    ):
        self._rfr        = risk_free_rate
        self._max_w      = max_weight
        self._min_w      = min_weight
        self._grid_steps = grid_steps
        self._results: List[OptimizationResult] = []

    def optimize(
        self,
        objective:       OptimizationObjective,
        returns:         Dict[str, List[float]],
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Find optimal weights for the given objective."""
        instruments = list(returns.keys())
        if not instruments:
            return OptimizationResult(objective=objective)

        # Generate candidate weight vectors (already capped at max_weight)
        candidates = _grid_weights(instruments, self._grid_steps, self._max_w)

        # Add initial weights as a candidate
        if initial_weights:
            candidates.append(_normalise({k: initial_weights.get(k, 1.0) for k in instruments}))

        # Score each candidate
        best_score  = None
        best_w      = _normalise({ins: 1.0 for ins in instruments})
        best_metrics: Dict[str, float] = {}
        iters = 0

        for w in candidates:
            # Iterative projection: clamp → normalise → repeat until stable
            w = _normalise(w)
            for _ in range(20):
                changed = False
                for ins in list(w.keys()):
                    clamped = max(self._min_w, min(self._max_w, w[ins]))
                    if abs(clamped - w[ins]) > 1e-9:
                        w[ins] = clamped
                        changed = True
                w = _normalise(w)
                if not changed:
                    break

            port_r = _portfolio_returns(returns, w)
            if not port_r:
                continue

            score, metrics = self._score(objective, port_r, returns, w)
            iters += 1

            if best_score is None or score > best_score:
                best_score  = score
                best_w      = w
                best_metrics = metrics

        result = OptimizationResult(
            objective  = objective,
            weights    = {k: round(v, 4) for k, v in best_w.items()},
            metrics    = {k: round(v, 4) for k, v in best_metrics.items()},
            converged  = best_score is not None,
            iterations = iters,
        )
        self._results.append(result)
        logger.debug(f"AllocationOptimizer: {objective.value} score={best_score:.4f} iters={iters}")
        return result

    def _score(
        self,
        objective: OptimizationObjective,
        port_r:    List[float],
        returns:   Dict[str, List[float]],
        weights:   Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        sh    = _sharpe(port_r, self._rfr)
        so    = _sortino(port_r, self._rfr)
        md    = _max_drawdown(port_r)
        vol   = _std(port_r)
        div   = _diversification_ratio(returns, weights)
        mu    = _mean(port_r)

        metrics = {
            "sharpe":        sh,
            "sortino":       so,
            "max_drawdown":  md,
            "volatility":    vol * math.sqrt(252),
            "annual_return": mu * 252,
            "div_ratio":     div,
        }

        score_map = {
            OptimizationObjective.MAX_SHARPE:          sh,
            OptimizationObjective.MAX_SORTINO:         so,
            OptimizationObjective.MIN_DRAWDOWN:        -md,
            OptimizationObjective.MAX_DIVERSIFICATION: div,
            OptimizationObjective.RISK_PARITY:         -self._risk_parity_penalty(returns, weights),
            OptimizationObjective.BALANCED:            0.4 * sh + 0.3 * (-md) + 0.3 * div,
            OptimizationObjective.CAPITAL_PRESERVATION: -md * 2 - vol * math.sqrt(252),
            OptimizationObjective.EXPECTED_UTILITY:    mu * 252 - 0.5 * (vol ** 2) * 252,
        }
        return score_map.get(objective, sh), metrics

    def _risk_parity_penalty(
        self,
        returns: Dict[str, List[float]],
        weights: Dict[str, float],
    ) -> float:
        """Minimize variance of risk contributions (proxy for risk parity)."""
        instruments = list(weights.keys())
        vols = [statistics.stdev(returns.get(ins, [0.0])) if len(returns.get(ins, [])) >= 2 else 0.0
                for ins in instruments]
        rc = [weights.get(ins, 0.0) * v for ins, v in zip(instruments, vols)]
        mu_rc = _mean(rc)
        return sum((r - mu_rc) ** 2 for r in rc) if rc else 0.0

    def compare_objectives(
        self,
        returns: Dict[str, List[float]],
    ) -> Dict[str, OptimizationResult]:
        """Optimize for all objectives and return comparison."""
        return {obj.value: self.optimize(obj, returns) for obj in OptimizationObjective}

    def history(self, last_n: int = 20) -> List[OptimizationResult]:
        return self._results[-last_n:]
