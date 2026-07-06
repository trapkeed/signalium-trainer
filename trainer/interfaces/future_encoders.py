"""Future encoder interfaces — Phase 14.9C reserved stubs.

==========================================================================
IMPORTANT: This module contains INTERFACE STUBS ONLY.
           No implementation is provided.
           Do NOT instantiate any class from this module.
           Do NOT implement reinforcement learning here.
           Do NOT implement graph neural networks here.
           Do NOT implement macroeconomic models here.
==========================================================================

Each interface below defines the exact contract that a future concrete
implementation must satisfy to be accepted by the PluginRegistry and the
HybridEncoder fusion layer.

Interfaces defined here:
  BaseReinforcementLearningInterface  — RL agent / policy interface
  BaseCrossAssetEncoder               — Cross-asset correlation encoder
  BaseNewsEncoder                     — NLP news embedding encoder
  BaseMacroEncoder                    — Macroeconomic data encoder
  BaseAlternativeDataEncoder          — Alternative data encoder
  BaseOrderBookEncoder                — Level 2 order book encoder
  BaseOptionsDataEncoder              — Options chain encoder
  BaseSentimentEncoder                — Market sentiment encoder
  BaseGraphNeuralNetworkInterface     — GNN graph encoder interface
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from trainer.interfaces.base import BaseExternalEncoder

# ---------------------------------------------------------------------------
# Reinforcement Learning Interface
# ---------------------------------------------------------------------------


class BaseReinforcementLearningInterface(BaseExternalEncoder):
    """RESERVED — Future reinforcement learning agent interface.

    Prepared for future integration where a trained RL policy is used to
    generate state representations that augment the model's latent space.
    The RL agent output (value estimate or policy embedding) is passed as an
    additional stream to the fusion layer.

    PLANNED INPUTS:
        state_tensor: Current market state representation.
        reward_history: Optional historical reward signal.
        action_history: Optional historical action sequence.

    PLANNED OUTPUTS:
        (batch, output_dim) — RL policy or value embedding.

    Do NOT implement RL training here.  This interface connects an
    already-trained external RL policy to the hybrid encoder.
    """

    @property
    def name(self) -> str:
        return "reinforcement_learning"

    @abstractmethod
    def get_policy_embedding(
        self,
        state_tensor: torch.Tensor,
        reward_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract a policy embedding from the RL agent state.

        RESERVED — not yet implemented.

        Args:
            state_tensor: ``(batch, state_dim)`` current market state.
            reward_history: Optional ``(batch, T)`` historical rewards.

        Returns:
            ``(batch, output_dim)`` policy embedding.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {
            "version": "0.0.0",
            "status": "future",
            "phase": "reserved",
            "note": "RL interface — not yet implemented.",
        }


# ---------------------------------------------------------------------------
# Cross-Asset Encoder
# ---------------------------------------------------------------------------


class BaseCrossAssetEncoder(BaseExternalEncoder):
    """RESERVED — Cross-asset correlation encoder interface.

    Encodes price and correlation data from related assets (equity indices,
    FX pairs, commodities, crypto) into a latent vector that conditions the
    model's prediction on inter-asset dynamics.

    PLANNED INPUTS:
        asset_sequences: Dict mapping asset symbol → ``(batch, seq_len, feat)``
        correlation_matrix: Optional ``(batch, n_assets, n_assets)``

    PLANNED OUTPUTS:
        (batch, output_dim) — cross-asset context embedding.
    """

    @property
    def name(self) -> str:
        return "cross_asset"

    @abstractmethod
    def encode_assets(
        self,
        asset_sequences: dict[str, torch.Tensor],
        correlation_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode multi-asset data into a single latent vector.

        RESERVED — not yet implemented.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {"version": "0.0.0", "status": "future", "phase": "reserved"}


# ---------------------------------------------------------------------------
# News Encoder
# ---------------------------------------------------------------------------


class BaseNewsEncoder(BaseExternalEncoder):
    """RESERVED — NLP news embedding encoder interface.

    Encodes financial news headlines and articles into a sentiment-conditioned
    embedding using a pre-trained language model (e.g. FinBERT).

    PLANNED INPUTS:
        token_ids: ``(batch, n_articles, seq_len)`` tokenised news texts.
        attention_mask: Optional ``(batch, n_articles, seq_len)`` mask.
        timestamps: Optional ``(batch, n_articles)`` article timestamps.

    PLANNED OUTPUTS:
        (batch, output_dim) — aggregated news embedding.
    """

    @property
    def name(self) -> str:
        return "news"

    @abstractmethod
    def encode_articles(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode financial news articles into a latent representation.

        RESERVED — not yet implemented.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {"version": "0.0.0", "status": "future", "phase": "reserved"}


# ---------------------------------------------------------------------------
# Macro Encoder
# ---------------------------------------------------------------------------


class BaseMacroEncoder(BaseExternalEncoder):
    """RESERVED — Macroeconomic data encoder interface.

    Do NOT implement macroeconomic models here.  This interface connects
    pre-computed macroeconomic indicators (interest rates, CPI, GDP,
    employment data) to the hybrid encoder as an additional context stream.

    PLANNED INPUTS:
        macro_indicators: ``(batch, n_indicators)`` standardised values.
        indicator_names: List of indicator names for interpretability.
        release_timestamps: Optional ``(batch, n_indicators)`` release times.

    PLANNED OUTPUTS:
        (batch, output_dim) — macro-economic context embedding.
    """

    @property
    def name(self) -> str:
        return "macro"

    @abstractmethod
    def encode_indicators(
        self,
        macro_indicators: torch.Tensor,
        indicator_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Encode macroeconomic indicators into a latent vector.

        RESERVED — not yet implemented.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {"version": "0.0.0", "status": "future", "phase": "reserved"}


# ---------------------------------------------------------------------------
# Alternative Data Encoder
# ---------------------------------------------------------------------------


class BaseAlternativeDataEncoder(BaseExternalEncoder):
    """RESERVED — Alternative data encoder interface.

    Encodes non-traditional market signals (satellite imagery analytics,
    credit card transaction flows, social media activity, web scraping
    signals, foot traffic data, …) into a latent representation.

    PLANNED INPUTS:
        alt_data: ``(batch, n_signals)`` normalised alternative signals.
        source_ids: Optional ``(batch, n_signals)`` signal source identifiers.

    PLANNED OUTPUTS:
        (batch, output_dim) — alternative data embedding.
    """

    @property
    def name(self) -> str:
        return "alternative_data"

    @abstractmethod
    def encode_signals(
        self,
        alt_data: torch.Tensor,
        source_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode alternative data signals.

        RESERVED — not yet implemented.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {"version": "0.0.0", "status": "future", "phase": "reserved"}


# ---------------------------------------------------------------------------
# Order Book Encoder
# ---------------------------------------------------------------------------


class BaseOrderBookEncoder(BaseExternalEncoder):
    """RESERVED — Level 2 / Level 3 order book encoder interface.

    Encodes bid/ask depth data into a latent representation capturing
    liquidity conditions, order imbalance, and microstructure signals.

    PLANNED INPUTS:
        bid_prices: ``(batch, n_levels)`` bid price levels.
        bid_sizes:  ``(batch, n_levels)`` bid size at each level.
        ask_prices: ``(batch, n_levels)`` ask price levels.
        ask_sizes:  ``(batch, n_levels)`` ask size at each level.
        timestamps: Optional ``(batch,)`` snapshot timestamps.

    PLANNED OUTPUTS:
        (batch, output_dim) — microstructure embedding.
    """

    @property
    def name(self) -> str:
        return "order_book"

    @abstractmethod
    def encode_book(
        self,
        bid_prices: torch.Tensor,
        bid_sizes: torch.Tensor,
        ask_prices: torch.Tensor,
        ask_sizes: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode order book depth into a microstructure embedding.

        RESERVED — not yet implemented.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {"version": "0.0.0", "status": "future", "phase": "reserved"}


# ---------------------------------------------------------------------------
# Options Data Encoder
# ---------------------------------------------------------------------------


class BaseOptionsDataEncoder(BaseExternalEncoder):
    """RESERVED — Options chain data encoder interface.

    Encodes the options market surface (IV smile, put/call ratio, Greeks,
    open interest distribution) into a latent representation capturing
    implied volatility regime and institutional positioning signals.

    PLANNED INPUTS:
        strikes:     ``(batch, n_strikes)`` strike prices.
        expiries:    ``(batch, n_expiries)`` expiry dates (days to expiry).
        ivs:         ``(batch, n_strikes, n_expiries)`` implied vols.
        greeks:      Optional ``(batch, n_strikes, n_expiries, n_greeks)``.
        open_interest: Optional ``(batch, n_strikes, n_expiries)``.

    PLANNED OUTPUTS:
        (batch, output_dim) — options surface embedding.
    """

    @property
    def name(self) -> str:
        return "options_data"

    @abstractmethod
    def encode_surface(
        self,
        strikes: torch.Tensor,
        expiries: torch.Tensor,
        ivs: torch.Tensor,
        greeks: torch.Tensor | None = None,
        open_interest: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the options volatility surface.

        RESERVED — not yet implemented.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {"version": "0.0.0", "status": "future", "phase": "reserved"}


# ---------------------------------------------------------------------------
# Sentiment Encoder
# ---------------------------------------------------------------------------


class BaseSentimentEncoder(BaseExternalEncoder):
    """RESERVED — Market sentiment encoder interface.

    Aggregates sentiment signals from multiple sources (social media,
    analyst ratings, put/call ratio, fear & greed index, insider
    transactions) into a unified market sentiment embedding.

    PLANNED INPUTS:
        sentiment_scores: ``(batch, n_sources)`` normalised sentiment values.
        source_weights:   Optional ``(n_sources,)`` source reliability weights.
        timestamps:       Optional ``(batch, n_sources)`` signal timestamps.

    PLANNED OUTPUTS:
        (batch, output_dim) — aggregated sentiment embedding.
    """

    @property
    def name(self) -> str:
        return "sentiment"

    @abstractmethod
    def encode_sentiment(
        self,
        sentiment_scores: torch.Tensor,
        source_weights: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode multi-source sentiment into a latent representation.

        RESERVED — not yet implemented.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {"version": "0.0.0", "status": "future", "phase": "reserved"}


# ---------------------------------------------------------------------------
# Graph Neural Network Interface
# ---------------------------------------------------------------------------


class BaseGraphNeuralNetworkInterface(BaseExternalEncoder):
    """RESERVED — Graph Neural Network encoder interface.

    ==========================================================================
    IMPORTANT: Do NOT implement graph neural networks in this module.
    This is a PURE INTERFACE STUB for future research.
    ==========================================================================

    Prepared for future integration where a GNN models relationships between
    market participants, instruments, sectors, or supply-chain nodes.

    The GNN encoder produces a node embedding or graph-level embedding that
    is passed as an additional stream to the fusion layer.

    PLANNED INPUTS:
        node_features: ``(batch, n_nodes, node_feat_dim)`` node attributes.
        edge_index:    ``(2, n_edges)`` COO-format adjacency list.
        edge_weights:  Optional ``(n_edges,)`` edge weights.

    PLANNED OUTPUTS:
        (batch, output_dim) — graph-level pooled embedding.
    """

    @property
    def name(self) -> str:
        return "graph_neural_network"

    @abstractmethod
    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a market relationship graph into a latent representation.

        RESERVED — not yet implemented.

        Args:
            node_features: ``(batch, n_nodes, feat_dim)`` node attributes.
            edge_index: ``(2, n_edges)`` sparse adjacency (COO format).
            edge_weights: Optional ``(n_edges,)`` edge weights.

        Returns:
            ``(batch, output_dim)`` graph embedding.
        """

    def get_metadata(self) -> dict[str, Any]:
        return {
            "version": "0.0.0",
            "status": "future",
            "phase": "reserved",
            "note": "GNN interface — no implementation. Requires torch_geometric or similar.",
        }
