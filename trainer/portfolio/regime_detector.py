"""Phase 20 — Regime Detector.

Detects current market regime from price/return series and adjusts
portfolio recommendations accordingly.

Regimes: TRENDING, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY,
         RISK_ON, RISK_OFF, HIGH_LIQUIDITY, LOW_LIQUIDITY
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class MarketRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    HIGH_LIQUIDITY = "high_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    UNKNOWN = "unknown"


# Regime → portfolio adjustment multipliers
REGIME_ADJUSTMENTS: Dict[MarketRegime, Dict[str, float]] = {
    MarketRegime.TRENDING:        {"momentum_boost": 1.3, "position_size": 1.1, "diversification_min": 0.6},
    MarketRegime.RANGING:         {"momentum_boost": 0.7, "position_size": 0.9, "diversification_min": 0.7},
    MarketRegime.HIGH_VOLATILITY: {"momentum_boost": 0.8, "position_size": 0.6, "diversification_min": 0.8},
    MarketRegime.LOW_VOLATILITY:  {"momentum_boost": 1.1, "position_size": 1.2, "diversification_min": 0.5},
    MarketRegime.RISK_ON:         {"momentum_boost": 1.2, "position_size": 1.1, "diversification_min": 0.5},
    MarketRegime.RISK_OFF:        {"momentum_boost": 0.7, "position_size": 0.6, "diversification_min": 0.9},
    MarketRegime.HIGH_LIQUIDITY:  {"momentum_boost": 1.1, "position_size": 1.1, "diversification_min": 0.5},
    MarketRegime.LOW_LIQUIDITY:   {"momentum_boost": 0.8, "position_size": 0.7, "diversification_min": 0.8},
    MarketRegime.UNKNOWN:         {"momentum_boost": 1.0, "position_size": 1.0, "diversification_min": 0.7},
}


@dataclass
class RegimeSignal:
    regime: MarketRegime
    confidence: float
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


@dataclass
class RegimeAssessment:
    primary_regime: MarketRegime
    secondary_regimes: List[RegimeSignal]
    confidence: float
    adjustment_factors: Dict[str, float]
    timestamp: str = ""

    def adjustment(self, key: str, default: float = 1.0) -> float:
        return self.adjustment_factors.get(key, default)

    def to_dict(self) -> dict:
        return {
            "primary_regime": self.primary_regime.value,
            "confidence": round(self.confidence, 4),
            "secondary_regimes": [r.to_dict() for r in self.secondary_regimes],
            "adjustment_factors": {k: round(v, 4) for k, v in self.adjustment_factors.items()},
            "timestamp": self.timestamp,
        }


class RegimeDetector:
    """Detects market regime from return series and indicator data."""

    def __init__(self,
                 vol_window: int = 20,
                 trend_window: int = 40,
                 high_vol_percentile: float = 0.75,
                 low_vol_percentile: float = 0.25) -> None:
        self._vol_window = vol_window
        self._trend_window = trend_window
        self._high_vol_pct = high_vol_percentile
        self._low_vol_pct = low_vol_percentile
        self._return_series: Dict[str, List[float]] = {}
        self._vol_history: List[float] = []
        self._regime_history: List[RegimeAssessment] = []

    def add_returns(self, instrument: str, returns: List[float]) -> None:
        series = self._return_series.setdefault(instrument, [])
        series.extend(returns)
        if len(series) > 500:
            self._return_series[instrument] = series[-500:]

    def add_return(self, instrument: str, ret: float) -> None:
        self.add_returns(instrument, [ret])

    def _portfolio_returns(self, window: int) -> List[float]:
        all_series = list(self._return_series.values())
        if not all_series:
            return []
        min_len = min(len(s) for s in all_series)
        w = min(window, min_len)
        if w == 0:
            return []
        avg = []
        for i in range(w):
            idx = min_len - w + i
            vals = [s[idx] for s in all_series]
            avg.append(sum(vals) / len(vals))
        return avg

    @staticmethod
    def _rolling_vol(series: List[float], window: int) -> float:
        if len(series) < 2:
            return 0.0
        sub = series[-window:]
        n = len(sub)
        mean = sum(sub) / n
        var = sum((x - mean) ** 2 for x in sub) / max(n - 1, 1)
        return math.sqrt(var)

    @staticmethod
    def _trend_strength(series: List[float]) -> float:
        n = len(series)
        if n < 4:
            return 0.0
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(series) / n
        cov = sum((xs[i] - mx) * (series[i] - my) for i in range(n))
        var = sum((xs[i] - mx) ** 2 for i in range(n))
        if var < 1e-12:
            return 0.0
        slope = cov / var
        ss_res = sum((series[i] - (my + slope * (xs[i] - mx))) ** 2 for i in range(n))
        ss_tot = sum((v - my) ** 2 for v in series) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return max(0.0, r2)

    def detect(self, timestamp: str = "",
               risk_on_instruments: Optional[List[str]] = None,
               safe_haven_instruments: Optional[List[str]] = None) -> RegimeAssessment:
        signals: List[RegimeSignal] = []

        portfolio_rets = self._portfolio_returns(self._trend_window)

        # Volatility regime
        if len(portfolio_rets) >= 10:
            current_vol = self._rolling_vol(portfolio_rets, self._vol_window)
            self._vol_history.append(current_vol)
            if len(self._vol_history) > 200:
                self._vol_history = self._vol_history[-200:]

            if len(self._vol_history) >= 10:
                sorted_vols = sorted(self._vol_history)
                hi_threshold = sorted_vols[int(len(sorted_vols) * self._high_vol_pct)]
                lo_threshold = sorted_vols[int(len(sorted_vols) * self._low_vol_pct)]
                if current_vol >= hi_threshold:
                    conf = min(1.0, (current_vol / max(hi_threshold, 1e-9) - 1.0) * 3 + 0.6)
                    signals.append(RegimeSignal(
                        MarketRegime.HIGH_VOLATILITY, min(conf, 0.95),
                        [f"Current vol {current_vol:.4f} >= {self._high_vol_pct:.0%} percentile"],
                    ))
                elif current_vol <= lo_threshold:
                    conf = min(1.0, (lo_threshold / max(current_vol, 1e-9) - 1.0) * 3 + 0.6)
                    signals.append(RegimeSignal(
                        MarketRegime.LOW_VOLATILITY, min(conf, 0.95),
                        [f"Current vol {current_vol:.4f} <= {self._low_vol_pct:.0%} percentile"],
                    ))

        # Trend regime
        if len(portfolio_rets) >= self._trend_window // 2:
            ts = self._trend_strength(portfolio_rets)
            if ts >= 0.5:
                signals.append(RegimeSignal(
                    MarketRegime.TRENDING, min(ts, 0.9),
                    [f"Portfolio R² = {ts:.2f} indicates trending"],
                ))
            else:
                signals.append(RegimeSignal(
                    MarketRegime.RANGING, min(1.0 - ts, 0.8),
                    [f"Portfolio R² = {ts:.2f} indicates ranging"],
                ))

        # Risk-on / Risk-off
        risk_signal = self._risk_sentiment(risk_on_instruments, safe_haven_instruments)
        if risk_signal is not None:
            signals.append(risk_signal)

        if not signals:
            primary = MarketRegime.UNKNOWN
            primary_conf = 0.5
        else:
            top = max(signals, key=lambda s: s.confidence)
            primary = top.regime
            primary_conf = top.confidence

        adj = dict(REGIME_ADJUSTMENTS[primary])
        assessment = RegimeAssessment(
            primary_regime=primary,
            secondary_regimes=sorted(signals, key=lambda s: s.confidence, reverse=True),
            confidence=primary_conf,
            adjustment_factors=adj,
            timestamp=timestamp,
        )
        self._regime_history.append(assessment)
        if len(self._regime_history) > 200:
            self._regime_history = self._regime_history[-200:]
        return assessment

    def _risk_sentiment(self,
                        risk_on: Optional[List[str]],
                        safe_haven: Optional[List[str]]) -> Optional[RegimeSignal]:
        risk_on = risk_on or ["NAS100", "SPX500", "BTCUSD", "AUDUSD", "NZDUSD"]
        safe_haven = safe_haven or ["XAUUSD", "USDJPY", "USDCHF"]

        risk_on_rets = []
        safe_rets = []
        for sym in risk_on:
            s = self._return_series.get(sym, [])
            if s:
                risk_on_rets.append(sum(s[-10:]) / max(len(s[-10:]), 1))
        for sym in safe_haven:
            s = self._return_series.get(sym, [])
            if s:
                safe_rets.append(sum(s[-10:]) / max(len(s[-10:]), 1))

        if not risk_on_rets or not safe_rets:
            return None

        avg_risk = sum(risk_on_rets) / len(risk_on_rets)
        avg_safe = sum(safe_rets) / len(safe_rets)
        spread = avg_risk - avg_safe

        if spread > 0:
            conf = min(0.9, abs(spread) * 50)
            return RegimeSignal(MarketRegime.RISK_ON, max(conf, 0.4),
                                [f"Risk-on spread = {spread:.4f}"])
        else:
            conf = min(0.9, abs(spread) * 50)
            return RegimeSignal(MarketRegime.RISK_OFF, max(conf, 0.4),
                                [f"Risk-off spread = {spread:.4f}"])

    def regime_history(self, n: int = 20) -> List[dict]:
        return [a.to_dict() for a in self._regime_history[-n:]]

    def dominant_regime_last_n(self, n: int = 10) -> MarketRegime:
        recent = self._regime_history[-n:]
        if not recent:
            return MarketRegime.UNKNOWN
        counts: Dict[str, int] = {}
        for a in recent:
            k = a.primary_regime.value
            counts[k] = counts.get(k, 0) + 1
        best = max(counts, key=lambda k: counts[k])
        return MarketRegime(best)
