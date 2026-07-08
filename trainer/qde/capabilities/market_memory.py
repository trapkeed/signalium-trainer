"""Phase 28.3 — Capability 9: Market Memory.

Maintains historical pattern library and regime performance history.
Matches current market conditions against past observations to provide
context-rich priors for the QDE's reasoning.

CONTRACT
--------
- Returns MemoryContext (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from trainer.qde.capabilities.cap_models import (
    MarketContext,
    MarketSnapshot,
    MemoryContext,
    MemoryPattern,
    RegimeType,
    SetupCandidate,
)


@dataclass
class HistoricalOutcome:
    """A stored historical trade outcome for pattern matching."""
    outcome_id: str
    setup_type: str
    regime: str
    direction: str
    result: str          # WIN | LOSS | NEUTRAL
    pnl_r: float
    rsi_at_entry: float
    adx_at_entry: float
    atr_ratio: float     # atr_14 / atr_50
    timestamp: float = field(default_factory=time.time)


class MarketMemoryCapability:
    """Stores and retrieves historical market patterns.

    This capability contributes to the QDE's Market Understanding and
    Regime Understanding responsibilities by providing calibrated historical
    priors based on pattern similarity.

    Pattern matching is done via lightweight feature vector similarity —
    no neural network, no database. Memory is in-process; a database
    adapter can replace self._outcomes for production.
    """

    CAPABILITY_NAME = "market_memory"

    # Maximum patterns to store before evicting oldest
    MAX_MEMORY = 1000

    # Minimum similarity score to include a match
    MIN_SIMILARITY = 0.60

    def __init__(self) -> None:
        self._outcomes: list[HistoricalOutcome] = []
        self._regime_stats: dict[str, dict[str, float]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def recall(
        self,
        snapshot: MarketSnapshot,
        candidate: SetupCandidate | None,
        market_context: MarketContext,
        top_k: int = 5,
    ) -> MemoryContext:
        """Retrieve historical patterns similar to the current market state.

        Args:
            snapshot: Current market snapshot.
            candidate: Optional setup candidate being evaluated.
            market_context: Current market context.
            top_k: Number of best matches to return.

        Returns:
            MemoryContext — advisory only, not a trade instruction.
        """
        warnings: list[str] = []
        matched: list[MemoryPattern] = []

        if not self._outcomes:
            return MemoryContext(
                capability=self.CAPABILITY_NAME,
                snapshot_id=snapshot.snapshot_id,
                confidence=0.1,
                warnings=["Memory is empty — no historical patterns available"],
                matched_patterns=[],
                pattern_match_quality=0.0,
                regime_has_been_seen_before=False,
            )

        current_vector = self._feature_vector(snapshot, market_context, candidate)
        regime_key = market_context.regime.value

        # Filter to same setup type if candidate is provided
        pool = self._outcomes
        if candidate:
            pool = [o for o in pool if o.setup_type == candidate.setup_type] or self._outcomes

        # Score all outcomes by similarity
        scored = []
        for outcome in pool:
            sim = self._similarity(current_vector, outcome)
            if sim >= self.MIN_SIMILARITY:
                scored.append((sim, outcome))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        for sim, outcome in top:
            matched.append(MemoryPattern(
                pattern_id=outcome.outcome_id,
                similarity_score=sim,
                pattern_type=outcome.setup_type,
                historical_outcome=outcome.result,
                historical_pnl_r=outcome.pnl_r,
                occurred_at=outcome.timestamp,
                regime_at_occurrence=outcome.regime,
            ))

        match_quality = sum(m.similarity_score for m in matched) / len(matched) if matched else 0.0

        # Historical win rate for this setup type in current regime
        hist_wr = self._regime_win_rate(regime_key, candidate.setup_type if candidate else "")
        hist_ev = self._regime_ev(regime_key, candidate.setup_type if candidate else "")

        # Anomaly detection: current ATR ratio is way outside historical range
        recent_ratios = [o.atr_ratio for o in self._outcomes[-100:]]
        current_ratio = snapshot.atr_14 / (snapshot.atr_50 + 1e-10)
        anomaly = False
        anomaly_desc = ""
        if recent_ratios:
            mean_ratio = sum(recent_ratios) / len(recent_ratios)
            if current_ratio > mean_ratio * 2.5:
                anomaly = True
                anomaly_desc = f"ATR ratio {current_ratio:.2f} is {current_ratio/mean_ratio:.1f}× the recent mean — possible unusual volatility"

        regime_seen = any(o.regime == regime_key for o in self._outcomes)
        regime_hist = self._regime_stats.get(regime_key, {})
        confidence = min(1.0, match_quality * 0.6 + (0.4 if regime_seen else 0.1))

        return MemoryContext(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            matched_patterns=matched,
            pattern_match_quality=match_quality,
            historical_win_rate_this_setup=hist_wr,
            historical_ev_this_setup=hist_ev,
            regime_has_been_seen_before=regime_seen,
            regime_historical_performance=regime_hist,
            anomaly_detected=anomaly,
            anomaly_description=anomaly_desc,
        )

    def store(self, outcome: HistoricalOutcome) -> None:
        """Store a completed trade outcome for future pattern matching."""
        self._outcomes.append(outcome)
        if len(self._outcomes) > self.MAX_MEMORY:
            self._outcomes.pop(0)
        self._update_regime_stats(outcome)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _feature_vector(
        self,
        snap: MarketSnapshot,
        ctx: MarketContext,
        candidate: SetupCandidate | None,
    ) -> dict[str, float]:
        atr_ratio = snap.atr_14 / (snap.atr_50 + 1e-10)
        return {
            "rsi": snap.rsi_14 / 100.0,
            "adx": snap.adx_14 / 100.0,
            "atr_ratio": min(3.0, atr_ratio) / 3.0,
            "bb_width": min(2.0, snap.bb_width) / 2.0,
            "vol_ratio": min(3.0, snap.volume_ratio) / 3.0,
        }

    def _similarity(
        self,
        current: dict[str, float],
        outcome: HistoricalOutcome,
    ) -> float:
        """Compute cosine-style similarity between current vector and stored outcome."""
        historical = {
            "rsi": outcome.rsi_at_entry / 100.0,
            "adx": outcome.adx_at_entry / 100.0,
            "atr_ratio": min(3.0, outcome.atr_ratio) / 3.0,
            "bb_width": 0.5,   # not stored per-outcome; neutral
            "vol_ratio": 1.0,
        }
        # Normalised Euclidean distance → similarity
        sq_dist = sum((current.get(k, 0.0) - historical.get(k, 0.0)) ** 2 for k in current)
        distance = sq_dist ** 0.5
        return max(0.0, 1.0 - distance)

    def _regime_win_rate(self, regime: str, setup_type: str) -> float | None:
        pool = [o for o in self._outcomes if o.regime == regime]
        if setup_type:
            pool = [o for o in pool if o.setup_type == setup_type] or pool
        if not pool:
            return None
        wins = sum(1 for o in pool if o.result == "WIN")
        return wins / len(pool)

    def _regime_ev(self, regime: str, setup_type: str) -> float | None:
        pool = [o for o in self._outcomes if o.regime == regime]
        if setup_type:
            pool = [o for o in pool if o.setup_type == setup_type] or pool
        if not pool:
            return None
        return sum(o.pnl_r for o in pool) / len(pool)

    def _update_regime_stats(self, outcome: HistoricalOutcome) -> None:
        if outcome.regime not in self._regime_stats:
            self._regime_stats[outcome.regime] = {
                "n": 0, "wins": 0, "total_r": 0.0,
                "win_rate": 0.0, "avg_r": 0.0,
            }
        s = self._regime_stats[outcome.regime]
        s["n"] += 1
        if outcome.result == "WIN":
            s["wins"] += 1
        s["total_r"] += outcome.pnl_r
        s["win_rate"] = s["wins"] / s["n"]
        s["avg_r"] = s["total_r"] / s["n"]
