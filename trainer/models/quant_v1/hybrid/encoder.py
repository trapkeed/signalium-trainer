"""HybridEncoder — top-level Phase 14.9A multi-stream encoder.

Ties together all five streams, the Dynamic Feature Gate, and the Adaptive
Fusion Layer into a single ``nn.Module`` that produces the shared latent
representation consumed by all prediction heads.

Output shape: ``(batch, fusion.latent_dim)``

This MUST equal the backbone's context_dim so the 22 existing prediction
heads work without modification.

Architecture::

    HybridInput
        │
        ├── RawMarketStream       → (batch, raw_out_dim)        ── Stream 1
        ├── MultiTimeframeStream  → (batch, tf_out_dim)         ── Stream 2
        ├── EngineeredFeatureStream──[DynamicFeatureGate]─→ (batch, feat_out_dim) ── Stream 3+6
        ├── ContextEmbeddingStream→ (batch, ctx_out_dim)        ── Stream 4
        ├── MarketMemoryStream    → (batch, mem_out_dim)        ── Stream 5
        │
        └── AdaptiveFusionLayer → (batch, latent_dim)           ── Stream 7
                │
                └── Shared Latent Representation → all prediction heads
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from trainer.models.quant_v1.hybrid.config import HybridEncoderConfig
from trainer.models.quant_v1.hybrid.embeddings import ContextEmbeddingStream
from trainer.models.quant_v1.hybrid.fusion import AdaptiveFusionLayer
from trainer.models.quant_v1.hybrid.gate import DynamicFeatureGate
from trainer.models.quant_v1.hybrid.input import HybridInput
from trainer.models.quant_v1.hybrid.memory import MarketMemoryStream
from trainer.models.quant_v1.hybrid.streams import (
    EngineeredFeatureStream,
    MultiTimeframeStream,
    RawMarketStream,
)


class HybridEncoder(nn.Module):
    """Multi-stream hybrid encoder producing the shared latent representation.

    Args:
        config: ``HybridEncoderConfig`` with all stream and fusion settings.
                ``config.fusion.latent_dim`` MUST match the backbone's
                context_dim for the prediction heads to work unchanged.
    """

    def __init__(self, config: HybridEncoderConfig) -> None:
        super().__init__()
        self._config = config

        # ── Stream 1: Raw Market Sequence ────────────────────────────────
        self.raw_market: RawMarketStream | None = None
        if config.raw_market.enabled:
            self.raw_market = RawMarketStream(config.raw_market)

        # ── Stream 2: Multi-Timeframe ─────────────────────────────────────
        self.multi_timeframe: MultiTimeframeStream | None = None
        if config.multi_timeframe.enabled:
            self.multi_timeframe = MultiTimeframeStream(config.multi_timeframe)

        # ── Stream 3: Engineered Features ────────────────────────────────
        self.engineered_features: EngineeredFeatureStream | None = None
        if config.engineered_features.enabled:
            self.engineered_features = EngineeredFeatureStream(config.engineered_features)

        # ── Stream 4: Context Embeddings ─────────────────────────────────
        self.context_embeddings: ContextEmbeddingStream | None = None
        if config.context_embeddings.enabled:
            self.context_embeddings = ContextEmbeddingStream(config.context_embeddings)

        # ── Stream 5: Market Memory ───────────────────────────────────────
        self.market_memory: MarketMemoryStream | None = None
        if config.market_memory.enabled:
            self.market_memory = MarketMemoryStream(config.market_memory)

        # ── Stream 6: Dynamic Feature Gate ───────────────────────────────
        self.feature_gate: DynamicFeatureGate | None = None
        if config.feature_gate.enabled and config.engineered_features.enabled:
            self.feature_gate = DynamicFeatureGate(
                feature_dim=config.engineered_features.input_dim,
                config=config.feature_gate,
            )

        # ── Stream 7: Adaptive Fusion ─────────────────────────────────────
        stream_dims = self._enumerate_stream_dims()
        self.fusion = AdaptiveFusionLayer(
            stream_dims=stream_dims,
            config=config.fusion,
        )

        # Training statistics
        self._last_gate_stats: dict[str, float] = {}
        self._last_contributions: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def output_dim(self) -> int:
        """Dimension of the shared latent representation."""
        return self._config.fusion.latent_dim

    @property
    def enabled_streams(self) -> list[str]:
        """Names of the streams that are currently active."""
        return [name for name, _ in self._enumerate_stream_dims()]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, hybrid_input: HybridInput) -> torch.Tensor:
        """Encode all available streams and fuse into the shared latent.

        Args:
            hybrid_input: ``HybridInput`` container with optional field tensors.

        Returns:
            ``(batch, latent_dim)`` — the shared latent representation.
        """
        batch_size = hybrid_input.batch_size
        stream_outputs: list[torch.Tensor] = []
        gate_context: torch.Tensor | None = None

        # Stream 1 — Raw Market
        if self.raw_market is not None and hybrid_input.raw_sequence is not None:
            raw_out = self.raw_market(hybrid_input.raw_sequence)
            stream_outputs.append(raw_out)
            gate_context = raw_out

        # Stream 2 — Multi-Timeframe
        if self.multi_timeframe is not None and hybrid_input.multi_timeframe is not None:
            tf_out = self.multi_timeframe(hybrid_input.multi_timeframe, batch_size=batch_size)
            stream_outputs.append(tf_out)
            if gate_context is None:
                gate_context = tf_out

        # Stream 3 (+6) — Engineered Features with optional gate
        if self.engineered_features is not None and hybrid_input.engineered_features is not None:
            feats = hybrid_input.engineered_features
            if self.feature_gate is not None and gate_context is not None:
                # Ensure gate_context is the right dimension
                ctx_for_gate = gate_context
                if ctx_for_gate.shape[-1] != self._config.feature_gate.context_dim:
                    # Use a zero context when dims mismatch (safe fallback)
                    ctx_for_gate = torch.zeros(
                        batch_size,
                        self._config.feature_gate.context_dim,
                        device=feats.device,
                        dtype=feats.dtype,
                    )
                feats, gate_weights = self.feature_gate(feats, ctx_for_gate)
                self._last_gate_stats = self.feature_gate.gate_statistics(gate_weights)
            feat_out = self.engineered_features(feats)
            stream_outputs.append(feat_out)

        # Stream 4 — Context Embeddings
        if self.context_embeddings is not None and hybrid_input.context_ids is not None:
            ctx_out = self.context_embeddings(hybrid_input.context_ids, batch_size=batch_size)
            stream_outputs.append(ctx_out)

        # Stream 5 — Market Memory
        if self.market_memory is not None and hybrid_input.memory_state is not None:
            mem_out = self.market_memory(hybrid_input.memory_state)
            stream_outputs.append(mem_out)

        if not stream_outputs:
            # All streams empty — return zeros of the correct shape
            dev = hybrid_input.device
            return torch.zeros(batch_size, self.output_dim, device=dev)

        # Adaptive Fusion → Shared Latent Representation
        latent = self.fusion(stream_outputs)
        self._last_contributions = self.fusion.contribution_summary()
        return latent

    # ------------------------------------------------------------------
    # Training monitoring
    # ------------------------------------------------------------------

    def get_stream_statistics(self) -> dict[str, Any]:
        """Return the most recent gate and contribution statistics.

        Useful for tracking per-stream contributions and gate behavior
        in the training loop.
        """
        stats: dict[str, Any] = {}
        for key, val in self._last_gate_stats.items():
            stats[f"gate/{key}"] = val
        for name, contrib in self._last_contributions.items():
            stats[f"stream_contribution/{name}"] = contrib
        return stats

    def gradient_norms(self) -> dict[str, float]:
        """Compute gradient L2 norms per stream module (call after backward).

        Returns a dict mapping ``stream/<name>`` to its gradient norm.
        """
        norms: dict[str, float] = {}
        _stream_map = {
            "raw_market": self.raw_market,
            "multi_timeframe": self.multi_timeframe,
            "engineered_features": self.engineered_features,
            "context_embeddings": self.context_embeddings,
            "market_memory": self.market_memory,
            "feature_gate": self.feature_gate,
            "fusion": self.fusion,
        }
        for name, module in _stream_map.items():
            if module is None:
                continue
            total = 0.0
            count = 0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.norm(2).item() ** 2
                    count += 1
            norms[f"grad_norm/{name}"] = total**0.5 if count > 0 else 0.0
        return norms

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enumerate_stream_dims(self) -> list[tuple[str, int]]:
        """Return list of (stream_name, out_dim) for every enabled stream."""
        dims: list[tuple[str, int]] = []
        if self._config.raw_market.enabled:
            dims.append(("raw_market", self._config.raw_market.out_dim))
        if self._config.multi_timeframe.enabled:
            dims.append(("multi_timeframe", self._config.multi_timeframe.out_dim))
        if self._config.engineered_features.enabled:
            dims.append(("engineered_features", self._config.engineered_features.out_dim))
        if self._config.context_embeddings.enabled:
            dims.append(("context_embeddings", self._config.context_embeddings.out_dim))
        if self._config.market_memory.enabled:
            dims.append(("market_memory", self._config.market_memory.out_dim))
        return dims if dims else [("raw_market", self._config.raw_market.out_dim)]
