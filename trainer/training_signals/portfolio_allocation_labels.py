"""Phase S4 — hindsight-optimal portfolio-allocation labels.

Explicitly NOT equal-weight / risk-parity / Sharpe-scaled rules (unlike
`trainer.portfolio.position_allocator.PositionAllocator`, which computes a
live allocation using those heuristics). Per the S4 spec: "Train using
portfolio level Sharpe and drawdown." Given the realized outcome paths of
several instruments over the same window, this labeler derives the weight
vector that would have maximized realized portfolio Sharpe subject to a
drawdown penalty — a hindsight target for training QDE's allocation-relevant
heads, not a runtime optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.training_signals.outcome_path import TradeOutcomePath

# Weight applied to the drawdown penalty term relative to the Sharpe term.
_DRAWDOWN_PENALTY_WEIGHT = 1.0


@dataclass
class HindsightAllocationLabel:
    instruments: list[str]
    optimal_weights: dict[str, float]
    per_instrument_return: dict[str, float]
    per_instrument_drawdown: dict[str, float]
    portfolio_sharpe_proxy: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "instruments": list(self.instruments),
            "optimal_weights": {k: round(v, 4) for k, v in self.optimal_weights.items()},
            "per_instrument_return": {
                k: round(v, 5) for k, v in self.per_instrument_return.items()
            },
            "per_instrument_drawdown": {
                k: round(v, 5) for k, v in self.per_instrument_drawdown.items()
            },
            "portfolio_sharpe_proxy": round(self.portfolio_sharpe_proxy, 4),
            "rationale": self.rationale,
        }


class HindsightAllocationLabeler:
    """Derives an optimal allocation-weight label across N realized outcome
    paths (one per instrument, over the same historical window).

    Each instrument's realized return is penalized by its realized drawdown
    (max adverse excursion, as a fraction of entry price) to form a
    per-instrument Sharpe-like score:

        score_i = return_i / (drawdown_i + epsilon) - penalty * drawdown_i

    Scores are floored at 0 (an instrument that lost money hindsight-optimally
    gets zero allocation, not a negative one — this labels allocation, it
    does not label short-selling a losing forecast) and normalized to sum to
    1. This is a defensible, auditable proxy for "the weights that would have
    maximized portfolio-level Sharpe while penalizing drawdown," appropriate
    as a training label; it is not claimed to be a true mean-variance
    optimum, and it never runs at inference/live time.
    """

    def label(self, paths: list[TradeOutcomePath]) -> HindsightAllocationLabel:
        if not paths:
            return HindsightAllocationLabel(
                instruments=[],
                optimal_weights={},
                per_instrument_return={},
                per_instrument_drawdown={},
                portfolio_sharpe_proxy=0.0,
                rationale="No instrument outcome paths supplied.",
            )

        instruments = [p.instrument for p in paths]
        returns: dict[str, float] = {}
        drawdowns: dict[str, float] = {}
        scores: dict[str, float] = {}

        for p in paths:
            ret = p.realized_return_pct()
            dd = p.max_adverse_excursion() / p.entry_price if p.entry_price else 0.0
            returns[p.instrument] = ret
            drawdowns[p.instrument] = dd
            raw_score = ret / (dd + 1e-6) - _DRAWDOWN_PENALTY_WEIGHT * dd
            scores[p.instrument] = max(0.0, raw_score)

        total_score = sum(scores.values())
        if total_score <= 1e-9:
            # Nothing scored positively (every instrument lost money
            # hindsight-optimally) — the optimal hindsight allocation is
            # "stay flat everywhere," represented as an equal-but-zero-weight
            # label rather than an arbitrary equal-weight fallback.
            weights = {inst: 0.0 for inst in instruments}
            rationale = (
                "No instrument produced a positive risk-adjusted score over this "
                "window — optimal hindsight allocation is fully flat (all weights 0)."
            )
        else:
            weights = {inst: scores[inst] / total_score for inst in instruments}
            rationale = (
                f"Weights derived from drawdown-penalized realized-return scores "
                f"(total_score={total_score:.4f}); higher realized return per unit "
                "of realized drawdown receives proportionally more weight."
            )

        # Portfolio-level Sharpe proxy: weighted mean return / weighted mean
        # drawdown, purely descriptive telemetry accompanying the label.
        weighted_return = sum(weights[i] * returns[i] for i in instruments)
        weighted_drawdown = sum(weights[i] * drawdowns[i] for i in instruments)
        portfolio_sharpe_proxy = (
            weighted_return / weighted_drawdown if weighted_drawdown > 1e-9 else 0.0
        )
        if math.isnan(portfolio_sharpe_proxy) or math.isinf(portfolio_sharpe_proxy):
            portfolio_sharpe_proxy = 0.0

        return HindsightAllocationLabel(
            instruments=instruments,
            optimal_weights=weights,
            per_instrument_return=returns,
            per_instrument_drawdown=drawdowns,
            portfolio_sharpe_proxy=portfolio_sharpe_proxy,
            rationale=rationale,
        )
