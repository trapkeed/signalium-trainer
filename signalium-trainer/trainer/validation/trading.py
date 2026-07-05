"""Phase 15 — Trading Metrics Engine.

Computes all 21 institutional trading performance metrics from a sequence of
simulated trade outcomes.

Metrics:
  1.  Win Rate
  2.  Profit Factor
  3.  Average Risk/Reward
  4.  Average Holding Time
  5.  Trade Frequency
  6.  Average MFE (Maximum Favourable Excursion)
  7.  Average MAE (Maximum Adverse Excursion)
  8.  Expectancy
  9.  Maximum Drawdown
  10. Average Drawdown
  11. Recovery Factor
  12. Sharpe Ratio
  13. Sortino Ratio
  14. Calmar Ratio
  15. Ulcer Index
  16. MAR Ratio (CAGR / Max Drawdown)
  17. Risk of Ruin
  18. Capital Efficiency
  19. Return Distribution (skewness + kurtosis)
  20. Tail Risk (CVaR at 5%)
  21. TP-before-SL Accuracy

Usage::

    sim = TradeSimulator(pip_size=0.0001, spread_pips=1.5)
    trades = sim.simulate(predictions, ohlcv, slippages=None)

    engine = TradingMetricsEngine()
    metrics = engine.compute(trades)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Trade result container
# ---------------------------------------------------------------------------


@dataclass
class TradeResult:
    """A single simulated trade.

    Attributes:
        direction: 1 = BUY, -1 = SELL, 0 = NO_TRADE.
        entry_price: Entry fill price.
        exit_price: Exit fill price (0.0 for open trades).
        stop_loss: Predicted stop loss level.
        take_profit: Predicted take profit level.
        holding_bars: Bars held before exit.
        pnl_pips: Realised P&L in pips.
        mfe_pips: Maximum Favourable Excursion in pips.
        mae_pips: Maximum Adverse Excursion in pips.
        tp_hit: True when take-profit was triggered before stop-loss.
        sl_hit: True when stop-loss was triggered.
        confidence: Model confidence at entry.
        timestamp: Optional entry timestamp (Unix seconds).
    """

    direction: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    holding_bars: int = 0
    pnl_pips: float = 0.0
    mfe_pips: float = 0.0
    mae_pips: float = 0.0
    tp_hit: bool = False
    sl_hit: bool = False
    confidence: float = 0.5
    timestamp: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl_pips > 0

    @property
    def risk_reward(self) -> float:
        if self.stop_loss == 0 or self.entry_price == 0:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price) if self.take_profit else 0.0
        return reward / risk if risk > 0 else 0.0


# ---------------------------------------------------------------------------
# Trading metrics container
# ---------------------------------------------------------------------------


@dataclass
class TradingMetrics:
    """All 21 institutional trading metrics."""

    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_rr: float = 0.0
    avg_holding_bars: float = 0.0
    trade_frequency: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    recovery_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0
    mar_ratio: float = 0.0
    risk_of_ruin: float = 0.0
    capital_efficiency: float = 0.0
    return_skewness: float = 0.0
    return_kurtosis: float = 0.0
    tail_risk_cvar5: float = 0.0
    tp_before_sl_accuracy: float = 0.0
    n_trades: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_rr": round(self.avg_rr, 4),
            "avg_holding_bars": round(self.avg_holding_bars, 2),
            "trade_frequency": round(self.trade_frequency, 4),
            "avg_mfe_pips": round(self.avg_mfe, 2),
            "avg_mae_pips": round(self.avg_mae, 2),
            "expectancy": round(self.expectancy, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_drawdown": round(self.avg_drawdown, 4),
            "recovery_factor": round(self.recovery_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "ulcer_index": round(self.ulcer_index, 4),
            "mar_ratio": round(self.mar_ratio, 4),
            "risk_of_ruin": round(self.risk_of_ruin, 4),
            "capital_efficiency": round(self.capital_efficiency, 4),
            "return_skewness": round(self.return_skewness, 4),
            "return_kurtosis": round(self.return_kurtosis, 4),
            "tail_risk_cvar5": round(self.tail_risk_cvar5, 4),
            "tp_before_sl_accuracy": round(self.tp_before_sl_accuracy, 4),
            "n_trades": self.n_trades,
        }


# ---------------------------------------------------------------------------
# Trading Metrics Engine
# ---------------------------------------------------------------------------


class TradingMetricsEngine:
    """Computes all 21 institutional trading metrics from trade results.

    Args:
        annualisation_factor: Bars per year for Sharpe/Sortino/Calmar.
            Common values: 252 (daily), 252*6.5 (hourly equity),
            365*24 (hourly crypto), 252*24*60 (1-minute forex).
        risk_free_rate: Annual risk-free rate for Sharpe denominator.
        ruin_threshold: Fraction of capital loss defining ruin (default 50%).

    Usage::

        engine = TradingMetricsEngine(annualisation_factor=252)
        metrics = engine.compute(trade_list)
    """

    def __init__(
        self,
        annualisation_factor: float = 252.0,
        risk_free_rate: float = 0.0,
        ruin_threshold: float = 0.50,
    ) -> None:
        self._ann = annualisation_factor
        self._rfr = risk_free_rate / self._ann  # per-bar risk-free
        self._ruin = ruin_threshold

    def compute(self, trades: list[TradeResult]) -> TradingMetrics:
        """Compute all trading metrics.

        Args:
            trades: List of completed ``TradeResult`` objects.

        Returns:
            ``TradingMetrics`` with all 21 metrics.
        """
        active_trades = [t for t in trades if t.direction != 0]
        if not active_trades:
            return TradingMetrics(n_trades=0)

        n = len(active_trades)
        pnls = np.array([t.pnl_pips for t in active_trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        # ── 1. Win Rate
        win_rate = len(wins) / n

        # ── 2. Profit Factor
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # ── 3. Average R:R
        rrs = [t.risk_reward for t in active_trades if t.risk_reward > 0]
        avg_rr = float(np.mean(rrs)) if rrs else 0.0

        # ── 4. Average Holding Time
        avg_holding = float(np.mean([t.holding_bars for t in active_trades]))

        # ── 5. Trade Frequency (trades per 100 bars — normalised by holding)
        total_bars = max(sum(t.holding_bars for t in active_trades), 1)
        trade_frequency = (n / total_bars) * 100

        # ── 6. Average MFE
        avg_mfe = float(np.mean([t.mfe_pips for t in active_trades]))

        # ── 7. Average MAE
        avg_mae = float(np.mean([t.mae_pips for t in active_trades]))

        # ── 8. Expectancy = WinRate × AvgWin + LossRate × AvgLoss
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss_val = float(losses.mean()) if len(losses) > 0 else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss_val

        # ── 9-10. Drawdown
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / np.where(peak != 0, np.abs(peak), 1.0)
        max_drawdown = float(drawdowns.max())
        avg_drawdown = float(drawdowns.mean())

        # ── 11. Recovery Factor
        total_return = float(equity[-1])
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float("inf")

        # ── 12. Sharpe Ratio
        returns = pnls / max(abs(pnls.max()), 1.0)
        excess = returns - self._rfr
        sharpe = (float(excess.mean()) * np.sqrt(self._ann)
                  / (float(excess.std()) + 1e-9))

        # ── 13. Sortino Ratio
        downside = excess[excess < 0]
        downside_std = float(downside.std()) + 1e-9
        sortino = (float(excess.mean()) * np.sqrt(self._ann) / downside_std)

        # ── 14. Calmar Ratio
        annualised_return = float(returns.mean()) * self._ann
        calmar = annualised_return / max_drawdown if max_drawdown > 0 else float("inf")

        # ── 15. Ulcer Index
        ulcer = float(np.sqrt(np.mean(drawdowns ** 2)))

        # ── 16. MAR Ratio (same as Calmar but uses actual total return)
        mar = annualised_return / max_drawdown if max_drawdown > 0 else float("inf")

        # ── 17. Risk of Ruin
        risk_of_ruin = self._estimate_risk_of_ruin(win_rate, avg_win, abs(avg_loss_val))

        # ── 18. Capital Efficiency (total return / max capital at risk)
        cap_efficiency = total_return / (gross_loss + 1e-9)

        # ── 19. Return Distribution
        skewness = float(self._skewness(pnls))
        kurtosis = float(self._kurtosis(pnls))

        # ── 20. Tail Risk (CVaR at 5%)
        cutoff = int(max(1, len(pnls) * 0.05))
        tail_risk = float(np.sort(pnls)[:cutoff].mean())

        # ── 21. TP-before-SL Accuracy
        tp_sl_trades = [t for t in active_trades if t.tp_hit or t.sl_hit]
        tp_acc = (sum(1 for t in tp_sl_trades if t.tp_hit) / len(tp_sl_trades)
                  if tp_sl_trades else 0.0)

        return TradingMetrics(
            win_rate=win_rate,
            profit_factor=min(profit_factor, 999.0),
            avg_rr=avg_rr,
            avg_holding_bars=avg_holding,
            trade_frequency=trade_frequency,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            expectancy=expectancy,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            recovery_factor=min(recovery_factor, 999.0),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=min(calmar, 999.0),
            ulcer_index=ulcer,
            mar_ratio=min(mar, 999.0),
            risk_of_ruin=risk_of_ruin,
            capital_efficiency=min(cap_efficiency, 999.0),
            return_skewness=skewness,
            return_kurtosis=kurtosis,
            tail_risk_cvar5=tail_risk,
            tp_before_sl_accuracy=tp_acc,
            n_trades=n,
        )

    @staticmethod
    def _estimate_risk_of_ruin(
        win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Closed-form Risk of Ruin estimate (Kelly-derived).

        R = ((1 - edge) / (1 + edge)) ^ (ruin_fraction / risk_per_trade)
        Simplified: probability of ruin given win rate and payoff ratio.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 1.0
        payoff = avg_win / avg_loss
        edge = win_rate - (1 - win_rate) / payoff
        if edge >= 0:
            return 0.0  # positive edge → theoretical ruin is 0
        # Ruggiero's formula approximation
        return min(1.0, float(((1 - win_rate) / win_rate) ** 10))

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        mean = x.mean()
        std = x.std() + 1e-9
        return float(((x - mean) ** 3).mean() / std ** 3)

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        mean = x.mean()
        std = x.std() + 1e-9
        return float(((x - mean) ** 4).mean() / std ** 4) - 3.0  # excess kurtosis


# ---------------------------------------------------------------------------
# Trade Simulator (converts model predictions → trade results)
# ---------------------------------------------------------------------------


class TradeSimulator:
    """Simulates trade execution from model prediction tensors.

    Converts ``(N, C)`` class probability tensors and ``(N, seq_len, feat)``
    OHLCV tensors into a list of ``TradeResult`` objects.

    Args:
        pip_size: Instrument pip size (e.g. 0.0001 for EURUSD).
        spread_pips: Fixed spread cost per trade.
        sl_pips: Stop loss distance in pips (default 20).
        tp_pips: Take profit distance in pips (default 40 = 2:1 RR).
        direction_threshold: Min probability to enter a directional trade.
    """

    def __init__(
        self,
        pip_size: float = 0.0001,
        spread_pips: float = 1.0,
        sl_pips: float = 20.0,
        tp_pips: float = 40.0,
        direction_threshold: float = 0.50,
    ) -> None:
        self._pip = pip_size
        self._spread = spread_pips
        self._sl = sl_pips
        self._tp = tp_pips
        self._dir_thresh = direction_threshold

    def simulate(
        self,
        probs: np.ndarray,
        ohlcv: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        slippage_pips: Optional[np.ndarray] = None,
    ) -> list[TradeResult]:
        """Simulate trades from class probabilities and price data.

        Args:
            probs: ``(N, C)`` class probabilities — class 0 = BUY, 1 = SELL, 2 = no-trade.
            ohlcv: ``(N, 5)`` OHLCV data aligned with probs.
            timestamps: Optional ``(N,)`` timestamps.
            slippage_pips: Optional ``(N,)`` per-bar slippage.

        Returns:
            List of ``TradeResult`` objects.
        """
        n = min(len(probs), len(ohlcv))
        trades: list[TradeResult] = []

        for i in range(n):
            p = probs[i]
            direction_class = int(p.argmax())
            confidence = float(p.max())

            if confidence < self._dir_thresh or direction_class == 2:
                continue  # no-trade

            direction = 1 if direction_class == 0 else -1
            entry = float(ohlcv[i, 3])  # close price
            slip = float(slippage_pips[i]) if slippage_pips is not None else 0.0
            actual_entry = entry + direction * (self._spread + slip) * self._pip
            sl_level = actual_entry - direction * self._sl * self._pip
            tp_level = actual_entry + direction * self._tp * self._pip

            # Forward simulate to find exit
            mfe = 0.0
            mae = 0.0
            exit_price = actual_entry
            holding = 0
            tp_hit = False
            sl_hit = False

            for j in range(i + 1, min(i + 200, n)):
                bar_high = float(ohlcv[j, 1])
                bar_low = float(ohlcv[j, 2])
                holding += 1

                if direction == 1:
                    mfe = max(mfe, (bar_high - actual_entry) / self._pip)
                    mae = max(mae, (actual_entry - bar_low) / self._pip)
                    if bar_low <= sl_level:
                        exit_price = sl_level
                        sl_hit = True
                        break
                    if bar_high >= tp_level:
                        exit_price = tp_level
                        tp_hit = True
                        break
                else:
                    mfe = max(mfe, (actual_entry - bar_low) / self._pip)
                    mae = max(mae, (bar_high - actual_entry) / self._pip)
                    if bar_high >= sl_level:
                        exit_price = sl_level
                        sl_hit = True
                        break
                    if bar_low <= tp_level:
                        exit_price = tp_level
                        tp_hit = True
                        break

            pnl = direction * (exit_price - actual_entry) / self._pip

            trades.append(TradeResult(
                direction=direction,
                entry_price=actual_entry,
                exit_price=exit_price,
                stop_loss=sl_level,
                take_profit=tp_level,
                holding_bars=holding,
                pnl_pips=pnl,
                mfe_pips=mfe,
                mae_pips=mae,
                tp_hit=tp_hit,
                sl_hit=sl_hit,
                confidence=confidence,
                timestamp=float(timestamps[i]) if timestamps is not None else 0.0,
            ))

        return trades
