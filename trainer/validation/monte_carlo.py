"""Phase 15 — Monte Carlo Validation.

Estimates model robustness and equity curve statistics by randomising:
  - Trade ordering (shuffling sequences)
  - Slippage (random fill costs)
  - Spread expansion
  - Execution delays

Produces:
  - Confidence bands on equity curves
  - Expected, worst-case, and best-case outcomes
  - Median equity curve

Usage::

    mc = MonteCarloValidator(n_simulations=1000, seed=42)
    result = mc.run(trade_results, initial_capital=10_000.0)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trainer.validation.trading import TradeResult

# ---------------------------------------------------------------------------
# Monte Carlo result
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run.

    Attributes:
        n_simulations: Number of simulations run.
        equity_curves: Array of shape ``(n_simulations, n_trades + 1)``.
        final_equity_p5: 5th-percentile final equity (worst band).
        final_equity_p25: 25th-percentile final equity.
        final_equity_p50: Median final equity.
        final_equity_p75: 75th-percentile final equity.
        final_equity_p95: 95th-percentile final equity.
        max_drawdown_p50: Median max drawdown across simulations.
        max_drawdown_p95: 95th-percentile max drawdown.
        sharpe_p50: Median Sharpe ratio.
        sharpe_p5: 5th-percentile Sharpe.
        win_rate_mean: Mean win rate.
        worst_case_return: Minimum final equity in all simulations.
        best_case_return: Maximum final equity in all simulations.
        median_return: Median final equity.
        ruin_probability: Fraction of simulations ending in ruin.
        initial_capital: Starting capital.
    """

    n_simulations: int = 0
    equity_curves: np.ndarray = field(default_factory=lambda: np.empty(0))
    final_equity_p5: float = 0.0
    final_equity_p25: float = 0.0
    final_equity_p50: float = 0.0
    final_equity_p75: float = 0.0
    final_equity_p95: float = 0.0
    max_drawdown_p50: float = 0.0
    max_drawdown_p95: float = 0.0
    sharpe_p50: float = 0.0
    sharpe_p5: float = 0.0
    win_rate_mean: float = 0.0
    worst_case_return: float = 0.0
    best_case_return: float = 0.0
    median_return: float = 0.0
    ruin_probability: float = 0.0
    initial_capital: float = 10_000.0

    def summary(self) -> str:
        lines = [
            f"Monte Carlo Simulation ({self.n_simulations:,} runs)",
            f"  Initial capital   : ${self.initial_capital:,.2f}",
            f"  Median outcome    : ${self.final_equity_p50:,.2f}",
            f"  P5  (worst band)  : ${self.final_equity_p5:,.2f}",
            f"  P95 (best band)   : ${self.final_equity_p95:,.2f}",
            f"  Max Drawdown P50  : {self.max_drawdown_p50:.1%}",
            f"  Max Drawdown P95  : {self.max_drawdown_p95:.1%}",
            f"  Sharpe P50        : {self.sharpe_p50:.2f}",
            f"  Ruin Probability  : {self.ruin_probability:.1%}",
        ]
        return "\n".join(lines)

    def get_percentile_curve(self, percentile: float) -> np.ndarray:
        """Return the equity curve at a given percentile."""
        if self.equity_curves.ndim < 2:
            return np.array([])
        return np.percentile(self.equity_curves, percentile, axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "initial_capital": self.initial_capital,
            "final_equity_p5": round(self.final_equity_p5, 2),
            "final_equity_p25": round(self.final_equity_p25, 2),
            "final_equity_p50": round(self.final_equity_p50, 2),
            "final_equity_p75": round(self.final_equity_p75, 2),
            "final_equity_p95": round(self.final_equity_p95, 2),
            "max_drawdown_p50": round(self.max_drawdown_p50, 4),
            "max_drawdown_p95": round(self.max_drawdown_p95, 4),
            "sharpe_p50": round(self.sharpe_p50, 4),
            "sharpe_p5": round(self.sharpe_p5, 4),
            "win_rate_mean": round(self.win_rate_mean, 4),
            "worst_case_return": round(self.worst_case_return, 2),
            "best_case_return": round(self.best_case_return, 2),
            "ruin_probability": round(self.ruin_probability, 4),
        }


# ---------------------------------------------------------------------------
# Monte Carlo Validator
# ---------------------------------------------------------------------------


class MonteCarloValidator:
    """Monte Carlo simulation over trade sequences.

    Args:
        n_simulations: Number of random Monte Carlo paths.
        seed: Random seed for reproducibility.
        slippage_std_pips: Standard deviation of Gaussian slippage noise.
        spread_expansion_prob: Probability that spread doubles on any trade.
        delay_prob: Probability of execution delay (skips entry bar).
        ruin_fraction: Capital fraction loss defining ruin.
        initial_capital: Starting capital in account currency.
        annualisation_factor: For Sharpe computation.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        seed: int = 42,
        slippage_std_pips: float = 0.5,
        spread_expansion_prob: float = 0.05,
        delay_prob: float = 0.02,
        ruin_fraction: float = 0.50,
        initial_capital: float = 10_000.0,
        annualisation_factor: float = 252.0,
    ) -> None:
        self._n = n_simulations
        self._seed = seed
        self._slip_std = slippage_std_pips
        self._spread_exp_prob = spread_expansion_prob
        self._delay_prob = delay_prob
        self._ruin_thresh = ruin_fraction
        self._capital = initial_capital
        self._ann = annualisation_factor

    def run(self, trades: list[TradeResult]) -> MonteCarloResult:
        """Run Monte Carlo simulation over a list of trade results.

        Args:
            trades: Completed ``TradeResult`` objects from ``TradeSimulator``.

        Returns:
            ``MonteCarloResult`` with equity curve statistics.
        """
        active = [t for t in trades if t.direction != 0]
        if len(active) < 2:
            return MonteCarloResult(n_simulations=self._n, initial_capital=self._capital)

        rng = np.random.default_rng(self._seed)
        n_trades = len(active)
        base_pnl = np.array([t.pnl_pips for t in active], dtype=float)

        # Convert pips to capital fraction (assume 1 pip = 0.01% of capital for normalisation)
        capital_per_pip = self._capital * 0.0001

        final_equities = np.zeros(self._n)
        max_drawdowns = np.zeros(self._n)
        sharpes = np.zeros(self._n)
        win_rates = np.zeros(self._n)
        equity_curves = np.zeros((self._n, n_trades + 1))

        for sim in range(self._n):
            # Randomise trade order
            order = rng.permutation(n_trades)
            pnl = base_pnl[order].copy()

            # Add slippage noise
            slippage = rng.normal(0, self._slip_std, n_trades)
            spread_exp = rng.random(n_trades) < self._spread_exp_prob
            delay_mask = rng.random(n_trades) < self._delay_prob

            # Apply perturbations
            pnl -= np.abs(slippage)
            pnl[spread_exp] -= rng.uniform(1.0, 3.0, spread_exp.sum())
            pnl[delay_mask] *= rng.uniform(0.5, 0.9, delay_mask.sum())

            # Build equity curve
            equity = np.zeros(n_trades + 1)
            equity[0] = self._capital
            for i in range(n_trades):
                equity[i + 1] = equity[i] + pnl[i] * capital_per_pip

            equity_curves[sim] = equity

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
            max_drawdowns[sim] = float(dd.max())

            # Sharpe
            returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1.0)
            sharpes[sim] = (
                float(returns.mean()) * np.sqrt(self._ann) / (float(returns.std()) + 1e-9)
            )

            final_equities[sim] = float(equity[-1])
            win_rates[sim] = float((pnl > 0).mean())

        # Compute ruin probability
        ruin_capital = self._capital * (1.0 - self._ruin_thresh)
        ruin_prob = float((final_equities < ruin_capital).mean())

        return MonteCarloResult(
            n_simulations=self._n,
            equity_curves=equity_curves,
            final_equity_p5=float(np.percentile(final_equities, 5)),
            final_equity_p25=float(np.percentile(final_equities, 25)),
            final_equity_p50=float(np.percentile(final_equities, 50)),
            final_equity_p75=float(np.percentile(final_equities, 75)),
            final_equity_p95=float(np.percentile(final_equities, 95)),
            max_drawdown_p50=float(np.percentile(max_drawdowns, 50)),
            max_drawdown_p95=float(np.percentile(max_drawdowns, 95)),
            sharpe_p50=float(np.percentile(sharpes, 50)),
            sharpe_p5=float(np.percentile(sharpes, 5)),
            win_rate_mean=float(win_rates.mean()),
            worst_case_return=float(final_equities.min()),
            best_case_return=float(final_equities.max()),
            median_return=float(np.percentile(final_equities, 50)),
            ruin_probability=ruin_prob,
            initial_capital=self._capital,
        )
