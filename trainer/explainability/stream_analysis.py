"""Stream contribution analysis for the Phase 14.9A Hybrid Encoder.

Analyses how much each of the five processing streams contributes to the
shared latent representation that drives all prediction heads.

Methods used:
  1. Fusion weights — contribution scores recorded by AdaptiveFusionLayer
     during every forward pass (available via ``_last_contributions``).
  2. Activation norms — L2 norm of each stream encoder's parameters, used
     as a proxy for how much the encoder has learned.
  3. Gate statistics — Dynamic Feature Gate signal (if active).

For non-hybrid models, a degenerate ``StreamContribution`` with a single
"backbone" stream is returned so callers never need to branch.

Usage::

    # After a forward pass:
    with torch.no_grad():
        _ = model(hybrid_input)
    contribution = StreamContributionAnalyzer.from_model(model)
    print(contribution.ranked_streams)

    # Or one-shot:
    contribution = StreamContributionAnalyzer.analyze(model, input_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class StreamContribution:
    """Per-stream contribution scores for a single forward pass.

    Attributes:
        fusion_weights: Normalised fusion weight per stream name (sums ≈ 1).
            Populated from the AdaptiveFusionLayer's
            ``contribution_summary()`` recorded during the forward pass.
        activation_norms: L2 norm of each stream encoder's parameters.
        gate_statistics: Dynamic Feature Gate statistics (empty dict when
            the gate is disabled or no hybrid encoder is present).
        most_influential_stream: Name of the highest-weight stream.
        is_hybrid: Whether the model uses the Phase 14.9A hybrid encoder.
    """

    fusion_weights: dict[str, float] = field(default_factory=dict)
    activation_norms: dict[str, float] = field(default_factory=dict)
    gate_statistics: dict[str, float] = field(default_factory=dict)
    most_influential_stream: str = "unknown"
    is_hybrid: bool = False

    @property
    def ranked_streams(self) -> list[tuple[str, float]]:
        """Streams sorted by fusion weight, highest first."""
        return sorted(
            self.fusion_weights.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "is_hybrid": self.is_hybrid,
            "fusion_weights": self.fusion_weights,
            "activation_norms": self.activation_norms,
            "gate_statistics": self.gate_statistics,
            "most_influential_stream": self.most_influential_stream,
            "ranked_streams": self.ranked_streams,
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class StreamContributionAnalyzer:
    """Analyzes stream-level contributions in the HybridEncoder.

    Reads ``_last_contributions`` and ``_last_gate_stats`` that
    ``HybridEncoder`` records during every forward pass (Phase 14.9A).
    For non-hybrid models a degenerate ``StreamContribution`` with a single
    "backbone" stream is returned so callers don't need to branch.

    All methods are static — no instance state is needed.
    """

    @staticmethod
    def from_model(model: nn.Module) -> StreamContribution:
        """Extract stream contributions recorded during the last forward pass.

        Call this AFTER running a forward pass so that ``_last_contributions``
        and ``_last_gate_stats`` have been updated by the encoder.

        Args:
            model: Any model — works with ``QuantModelV1`` (or any model
                exposing a ``hybrid_encoder`` sub-module).

        Returns:
            ``StreamContribution`` populated with the most recent statistics.
        """
        hybrid_enc = getattr(model, "hybrid_encoder", None)
        if hybrid_enc is None:
            # Non-hybrid — degenerate result with a single backbone stream
            return StreamContribution(
                fusion_weights={"backbone": 1.0},
                activation_norms={"backbone": 1.0},
                most_influential_stream="backbone",
                is_hybrid=False,
            )

        # Fusion weights — recorded by AdaptiveFusionLayer during forward()
        fusion_weights: dict[str, float] = dict(getattr(hybrid_enc, "_last_contributions", {}))
        gate_stats: dict[str, float] = dict(getattr(hybrid_enc, "_last_gate_stats", {}))

        # Activation norms per stream (proxy for learned representation size)
        norms: dict[str, float] = {}
        stream_map: dict[str, nn.Module | None] = {
            "raw_market": getattr(hybrid_enc, "raw_market", None),
            "multi_timeframe": getattr(hybrid_enc, "multi_timeframe", None),
            "engineered_features": getattr(hybrid_enc, "engineered_features", None),
            "context_embeddings": getattr(hybrid_enc, "context_embeddings", None),
            "market_memory": getattr(hybrid_enc, "market_memory", None),
        }
        for stream_name, module in stream_map.items():
            if module is None:
                continue
            total_sq = sum(
                float(p.data.norm(2).item()) ** 2 for p in module.parameters() if p.requires_grad
            )
            norms[stream_name] = float(total_sq**0.5)

        most_influential = (
            max(fusion_weights, key=lambda k: fusion_weights[k]) if fusion_weights else "unknown"
        )
        return StreamContribution(
            fusion_weights=fusion_weights,
            activation_norms=norms,
            gate_statistics=gate_stats,
            most_influential_stream=most_influential,
            is_hybrid=True,
        )

    @staticmethod
    def analyze(model: nn.Module, input_data: Any) -> StreamContribution:
        """Run a forward pass then extract stream contributions.

        Puts the model in eval mode, runs ``forward()`` with no_grad,
        then calls :meth:`from_model`.

        Args:
            model: ``QuantModelV1`` instance (or compatible).
            input_data: Input tensor or ``HybridInput``.

        Returns:
            ``StreamContribution`` from the completed forward pass.
        """
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(input_data)
        except Exception as exc:
            logger.debug(f"StreamContributionAnalyzer.analyze: forward failed: {exc}")
        finally:
            if was_training:
                model.train()
        return StreamContributionAnalyzer.from_model(model)
