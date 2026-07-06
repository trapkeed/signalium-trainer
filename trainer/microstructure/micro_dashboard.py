"""Phase 22 — Microstructure Dashboard.

7 dashboard views:
1. Microstructure Timeline
2. Market State Timeline
3. Momentum Evolution
4. Volatility Evolution
5. Liquidity Estimates
6. Microstructure Heatmaps
7. Embedding Explorer
"""

from __future__ import annotations

import collections
import time
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import (
    MarketStateRecord,
    MicrostructureEmbedding,
    MicrostructureFeatures,
    MicrostructureScore,
)
from trainer.microstructure.micro_types import MarketState


def _bucket(ts: float, granularity: str = "day") -> str:
    t = time.gmtime(ts)
    if granularity == "hour":
        return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d} {t.tm_hour:02d}:00"
    if granularity == "month":
        return f"{t.tm_year}-{t.tm_mon:02d}"
    return f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}"


class MicrostructureDashboard:
    """Generates all 7 microstructure dashboard views."""

    # ── 1. Microstructure Timeline ────────────────────────────────────────────

    def microstructure_timeline(
        self,
        feature_list: List[MicrostructureFeatures],
        scores: List[MicrostructureScore],
        granularity: str = "day",
    ) -> Dict:
        if not feature_list:
            return {"view": "MICROSTRUCTURE_TIMELINE", "periods": [], "data": {}}

        data: Dict[str, Dict] = {}
        for f, s in zip(feature_list, scores):
            b = _bucket(f.timestamp, granularity)
            if b not in data:
                data[b] = {
                    "trend_scores": [],
                    "momentum_scores": [],
                    "efficiency_scores": [],
                    "volatility_scores": [],
                    "overall_scores": [],
                }
            data[b]["trend_scores"].append(s.trend_score)
            data[b]["momentum_scores"].append(s.momentum_score)
            data[b]["efficiency_scores"].append(s.efficiency_score)
            data[b]["volatility_scores"].append(s.volatility_score)
            data[b]["overall_scores"].append(s.overall_score)

        result = {}
        for b, d in data.items():
            result[b] = {k: round(sum(v) / len(v), 4) for k, v in d.items() if v}

        return {
            "view": "MICROSTRUCTURE_TIMELINE",
            "granularity": granularity,
            "periods": sorted(result.keys()),
            "data": result,
            "total": len(feature_list),
        }

    # ── 2. Market State Timeline ──────────────────────────────────────────────

    def market_state_timeline(
        self,
        state_records: List[MarketStateRecord],
        granularity: str = "day",
    ) -> Dict:
        if not state_records:
            return {"view": "MARKET_STATE_TIMELINE", "periods": [], "data": {}}

        data: Dict[str, Dict] = {}
        for r in state_records:
            b = _bucket(r.timestamp, granularity)
            if b not in data:
                data[b] = {}
            st = r.state.value
            data[b][st] = data[b].get(st, 0) + 1

        return {
            "view": "MARKET_STATE_TIMELINE",
            "granularity": granularity,
            "periods": sorted(data.keys()),
            "data": data,
            "total_states": len(state_records),
            "states": [s.value for s in MarketState],
        }

    # ── 3. Momentum Evolution ─────────────────────────────────────────────────

    def momentum_evolution(
        self,
        feature_list: List[MicrostructureFeatures],
    ) -> Dict:
        if not feature_list:
            return {"view": "MOMENTUM_EVOLUTION", "series": {}}

        series = {
            "trend_energy": [],
            "momentum_acceleration": [],
            "momentum_deceleration": [],
            "momentum_exhaustion": [],
            "directional_persistence": [],
            "impulse_decay": [],
        }
        timestamps = []

        for f in feature_list:
            timestamps.append(round(f.timestamp, 3))
            for key in series:
                series[key].append(round(getattr(f, key, 0.0), 4))

        return {
            "view": "MOMENTUM_EVOLUTION",
            "timestamps": timestamps,
            "series": series,
            "total": len(feature_list),
        }

    # ── 4. Volatility Evolution ───────────────────────────────────────────────

    def volatility_evolution(
        self,
        feature_list: List[MicrostructureFeatures],
    ) -> Dict:
        if not feature_list:
            return {"view": "VOLATILITY_EVOLUTION", "series": {}}

        series = {
            "atr_expansion": [],
            "atr_compression": [],
            "volatility_clustering": [],
            "volatility_persistence": [],
            "volatility_shock": [],
            "noise_ratio": [],
        }
        timestamps = []

        for f in feature_list:
            timestamps.append(round(f.timestamp, 3))
            for key in series:
                series[key].append(round(getattr(f, key, 0.0), 4))

        return {
            "view": "VOLATILITY_EVOLUTION",
            "timestamps": timestamps,
            "series": series,
            "total": len(feature_list),
        }

    # ── 5. Liquidity Estimates ────────────────────────────────────────────────

    def liquidity_estimates(
        self,
        feature_list: List[MicrostructureFeatures],
        state_records: Optional[List[MarketStateRecord]] = None,
    ) -> Dict:
        if not feature_list:
            return {"view": "LIQUIDITY_ESTIMATES", "series": {}}

        series = {
            "spread_proxy": [],
            "gap_frequency": [],
            "large_candle_frequency": [],
            "breakout_strength": [],
            "reversal_strength": [],
            "liquidity_vacuum_prob": [],
            "compression_zone_score": [],
        }
        timestamps = []

        for f in feature_list:
            timestamps.append(round(f.timestamp, 3))
            for key in series:
                series[key].append(round(getattr(f, key, 0.0), 4))

        liq_states = {}
        if state_records:
            for r in state_records:
                liq_states[round(r.timestamp, 3)] = r.liquidity_state.value

        return {
            "view": "LIQUIDITY_ESTIMATES",
            "timestamps": timestamps,
            "series": series,
            "liq_states": liq_states,
            "total": len(feature_list),
        }

    # ── 6. Microstructure Heatmaps ────────────────────────────────────────────

    def microstructure_heatmaps(
        self,
        feature_list: List[MicrostructureFeatures],
        state_records: List[MarketStateRecord],
    ) -> Dict:
        """Score heatmap: market state × score dimension."""

        if not state_records:
            return {"view": "MICROSTRUCTURE_HEATMAPS", "grid": {}}

        # Map timestamp → state for fast lookup
        state_by_ts: Dict[float, str] = {
            round(r.timestamp, 0): r.state.value for r in state_records
        }

        grid: Dict[str, Dict[str, list]] = {}
        score_dims = [
            "trend_score",
            "momentum_score",
            "efficiency_score",
            "volatility_score",
            "liquidity_score",
        ]

        for f in feature_list:
            ts = round(f.timestamp, 0)
            state = state_by_ts.get(ts, "UNKNOWN")
            if state not in grid:
                grid[state] = {d: [] for d in score_dims}

        # Re-scan with scores not attached; use feature proxies
        for f in feature_list:
            ts = round(f.timestamp, 0)
            state = state_by_ts.get(ts, "UNKNOWN")
            if state not in grid:
                grid[state] = {d: [] for d in score_dims}
            grid[state]["trend_score"].append(f.trend_energy)
            grid[state]["momentum_score"].append(f.directional_persistence)
            grid[state]["efficiency_score"].append(f.trend_efficiency)
            grid[state]["volatility_score"].append(f.atr_expansion)
            grid[state]["liquidity_score"].append(1.0 - f.spread_proxy)

        avg_grid: Dict[str, Dict[str, float]] = {}
        for state, dims in grid.items():
            avg_grid[state] = {d: round(sum(v) / len(v), 4) if v else 0.0 for d, v in dims.items()}

        return {
            "view": "MICROSTRUCTURE_HEATMAPS",
            "states": list(avg_grid.keys()),
            "dimensions": score_dims,
            "grid": avg_grid,
            "total": len(feature_list),
        }

    # ── 7. Embedding Explorer ─────────────────────────────────────────────────

    def embedding_explorer(
        self,
        embeddings: List[MicrostructureEmbedding],
    ) -> Dict:
        """Summarise embedding vectors for exploration."""
        if not embeddings:
            return {"view": "EMBEDDING_EXPLORER", "total": 0}

        dim = embeddings[0].dim if embeddings else 0
        n = len(embeddings)

        # Per-dimension mean and std
        if n > 0:
            dim_means = []
            for i in range(dim):
                vals = [e.embedding[i] for e in embeddings if i < len(e.embedding)]
                dim_means.append(round(sum(vals) / len(vals), 5) if vals else 0.0)

            sub_means = {
                "price_action": [
                    round(sum(e.price_action_state[i] for e in embeddings) / n, 5)
                    for i in range(min(8, len(embeddings[0].price_action_state)))
                ],
                "liquidity": [
                    round(sum(e.liquidity_state[i] for e in embeddings) / n, 5)
                    for i in range(min(6, len(embeddings[0].liquidity_state)))
                ],
                "momentum": [
                    round(sum(e.momentum_state[i] for e in embeddings) / n, 5)
                    for i in range(min(6, len(embeddings[0].momentum_state)))
                ],
                "volatility": [
                    round(sum(e.volatility_state[i] for e in embeddings) / n, 5)
                    for i in range(min(6, len(embeddings[0].volatility_state)))
                ],
                "efficiency": [
                    round(sum(e.efficiency_state[i] for e in embeddings) / n, 5)
                    for i in range(min(6, len(embeddings[0].efficiency_state)))
                ],
            }
        else:
            dim_means = []
            sub_means = {}

        return {
            "view": "EMBEDDING_EXPLORER",
            "total": n,
            "dim": dim,
            "dim_means": dim_means,
            "sub_means": sub_means,
        }

    def all_dashboards(
        self,
        feature_list: List[MicrostructureFeatures],
        scores: List[MicrostructureScore],
        state_records: List[MarketStateRecord],
        embeddings: List[MicrostructureEmbedding],
    ) -> Dict:
        return {
            "microstructure_timeline": self.microstructure_timeline(feature_list, scores),
            "market_state_timeline": self.market_state_timeline(state_records),
            "momentum_evolution": self.momentum_evolution(feature_list),
            "volatility_evolution": self.volatility_evolution(feature_list),
            "liquidity_estimates": self.liquidity_estimates(feature_list, state_records),
            "microstructure_heatmaps": self.microstructure_heatmaps(feature_list, state_records),
            "embedding_explorer": self.embedding_explorer(embeddings),
        }
