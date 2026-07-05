"""Phase 22 — Microstructure API.

Endpoints:
 1. analyse         — full OHLC analysis pipeline (features + score + state + embedding)
 2. current_state   — latest market state classification
 3. scores          — current microstructure scores
 4. embedding       — current embedding vector
 5. historical      — historical feature analysis
 6. get_dashboard   — specific or all dashboard views
 7. system_stats    — database stats
 8. order_flow_status — order flow data availability report
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from trainer.microstructure.micro_models import (
    OHLCBar,
    MicrostructureFeatures,
    MarketStateRecord,
    MicrostructureScore,
    MicrostructureEmbedding,
)
from trainer.microstructure.price_action    import PriceActionAnalyzer
from trainer.microstructure.volatility_micro import VolatilityMicrostructure
from trainer.microstructure.momentum_micro   import MomentumMicrostructure
from trainer.microstructure.liquidity_proxy  import LiquidityProxy
from trainer.microstructure.market_efficiency import MarketEfficiency
from trainer.microstructure.micro_scorer     import MicrostructureScorer
from trainer.microstructure.state_classifier import MarketStateClassifier
from trainer.microstructure.micro_embeddings import MicrostructureEmbedder
from trainer.microstructure.micro_database   import MicrostructureDatabase
from trainer.microstructure.micro_dashboard  import MicrostructureDashboard
from trainer.microstructure.tick_interface   import TickInterface, TickBar
from trainer.microstructure.order_flow_interface import OrderFlowInterface
from trainer.microstructure.micro_types import DataGranularity


class MicrostructureAPI:
    """Unified API for the Market Microstructure Intelligence Engine."""

    def __init__(self, db_path: str = ":memory:"):
        self.price_action  = PriceActionAnalyzer()
        self.volatility    = VolatilityMicrostructure()
        self.momentum      = MomentumMicrostructure()
        self.liquidity     = LiquidityProxy()
        self.efficiency    = MarketEfficiency()
        self.scorer        = MicrostructureScorer()
        self.classifier    = MarketStateClassifier()
        self.embedder      = MicrostructureEmbedder()
        self.tick          = TickInterface()
        self.order_flow    = OrderFlowInterface()
        self.db            = MicrostructureDatabase(db_path)
        self.dashboard_eng = MicrostructureDashboard()

        # In-memory history
        self._features:  List[MicrostructureFeatures]  = []
        self._scores:    List[MicrostructureScore]      = []
        self._states:    List[MarketStateRecord]        = []
        self._embeddings: List[MicrostructureEmbedding] = []

    # ── 1. Full analysis ──────────────────────────────────────────────────────

    def analyse(
        self,
        bars:       List[OHLCBar],
        instrument: str = "",
        timeframe:  str = "H1",
        ticks:      Optional[List[TickBar]] = None,
        persist:    bool = True,
    ) -> Dict:
        """Run the full OHLC microstructure analysis pipeline.

        Returns: features, score, market_state, embedding.
        """
        if not bars:
            return {"error": "No bars provided"}

        # Build feature object
        f = MicrostructureFeatures(
            timestamp  = bars[-1].timestamp,
            instrument = instrument or bars[-1].instrument,
            timeframe  = timeframe,
            granularity = DataGranularity.OHLC.value,
        )

        # Run all analyzers
        self.price_action.analyse_window(bars, f)
        self.volatility.analyse_window(bars, f)
        self.momentum.analyse_window(bars, f)
        self.liquidity.analyse_window(bars, f)
        self.efficiency.analyse_window(bars, f)

        # Optional tick enrichment
        if ticks:
            self.tick.enrich_features(f, ticks)

        # Score
        score = self.scorer.score(f)

        # Classify state
        state_record = self.classifier.classify(f, score)

        # Embed
        emb = self.embedder.embed(f, instrument, timeframe)

        # Track history
        self._features.append(f)
        self._scores.append(score)
        self._states.append(state_record)
        self._embeddings.append(emb)

        # Persist
        if persist:
            self.db.save_features(f)
            self.db.save_score(score, f.timestamp, instrument, timeframe)
            self.db.save_market_state(state_record, instrument, timeframe)
            self.db.save_embedding(emb)
            self.db.save_liquidity(f, state_record.liquidity_state.value)
            self.db.save_momentum(f, state_record.momentum_state.value)

        logger.info(
            f"MicrostructureAPI.analyse: {instrument}/{timeframe} "
            f"state={state_record.state.value} score={score.overall_score:.3f}"
        )

        return {
            "features":    f.to_dict(),
            "score":       score.to_dict(),
            "market_state": state_record.to_dict(),
            "embedding":   emb.to_dict(),
        }

    # ── 2. Current state ──────────────────────────────────────────────────────

    def current_state(self) -> Optional[Dict]:
        if not self._states:
            return None
        return self._states[-1].to_dict()

    # ── 3. Scores ─────────────────────────────────────────────────────────────

    def scores(self, last_n: int = 1) -> List[Dict]:
        return [s.to_dict() for s in self._scores[-last_n:]]

    # ── 4. Embedding ──────────────────────────────────────────────────────────

    def embedding(self) -> Optional[Dict]:
        if not self._embeddings:
            return None
        return self._embeddings[-1].to_dict()

    # ── 5. Historical analysis ────────────────────────────────────────────────

    def historical(
        self,
        instrument: Optional[str] = None,
        limit:      int           = 100,
    ) -> Dict:
        features = self._features[-limit:]
        scores   = self._scores[-limit:]
        states   = self._states[-limit:]
        return {
            "total":        len(features),
            "features":     [f.to_dict() for f in features],
            "scores":       [s.to_dict() for s in scores],
            "market_states": [s.to_dict() for s in states],
            "state_distribution": dict(
                (s.state.value, 0) for s in states
            ) if states else {},
        }

    # ── 6. Dashboard ──────────────────────────────────────────────────────────

    def get_dashboard(self, name: str = "all") -> Dict:
        fns = {
            "microstructure_timeline": lambda: self.dashboard_eng.microstructure_timeline(
                self._features, self._scores
            ),
            "market_state_timeline": lambda: self.dashboard_eng.market_state_timeline(
                self._states
            ),
            "momentum_evolution": lambda: self.dashboard_eng.momentum_evolution(
                self._features
            ),
            "volatility_evolution": lambda: self.dashboard_eng.volatility_evolution(
                self._features
            ),
            "liquidity_estimates": lambda: self.dashboard_eng.liquidity_estimates(
                self._features, self._states
            ),
            "microstructure_heatmaps": lambda: self.dashboard_eng.microstructure_heatmaps(
                self._features, self._states
            ),
            "embedding_explorer": lambda: self.dashboard_eng.embedding_explorer(
                self._embeddings
            ),
        }
        if name == "all":
            return self.dashboard_eng.all_dashboards(
                self._features, self._scores, self._states, self._embeddings
            )
        fn = fns.get(name)
        if fn:
            return fn()
        return {"error": f"Unknown dashboard '{name}'"}

    # ── 7. System stats ───────────────────────────────────────────────────────

    def system_stats(self) -> Dict:
        return {
            "in_memory": {
                "features":    len(self._features),
                "scores":      len(self._scores),
                "states":      len(self._states),
                "embeddings":  len(self._embeddings),
            },
            "database": self.db.stats(),
            "embedding_dim": self.embedder.embedding_dim(),
            "tick_available":       self.tick.is_available(),
            "order_flow_available": self.order_flow.is_available(),
        }

    # ── 8. Order flow status ──────────────────────────────────────────────────

    def order_flow_status(self) -> Dict:
        return self.order_flow.data_quality_report()
