"""Attention weight extraction and visualization for Transformer backbones.

Works with TransformerBackbone and HybridLSTMTransformerBackbone (Phase 14.9B).
Registers non-destructive forward hooks on ``nn.MultiheadAttention`` modules to
capture attention weights during inference.

Design invariants:
  - Hooks are always removed cleanly, even on exception (context manager).
  - Never modifies model weights or architecture.
  - Returns an empty AttentionMap for non-Transformer models rather than
    raising an error — callers can check `is_empty`.

Usage::

    extractor = AttentionExtractor()
    with extractor.register(model):
        _ = model(x)
        attn_map = extractor.get_last_attention_map()

    # Or use the convenience class method:
    attn_map = AttentionExtractor.extract(model, x)
    print(attn_map.most_attended_timestep())
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# AttentionMap — result container
# ---------------------------------------------------------------------------


@dataclass
class AttentionMap:
    """Captured multi-head attention weights from all Transformer layers.

    Attributes:
        weights: List of tensors — one per encoder layer.  Each tensor has
            shape ``(batch, n_heads, seq_len, seq_len)`` where rows = query
            positions and columns = key positions.
        layer_names: Names of the sub-modules whose attention was captured.
        seq_len: Sequence length (inferred from first captured weight).
        n_heads: Number of attention heads (from the first layer).
    """

    weights: list[torch.Tensor]
    layer_names: list[str] = field(default_factory=list)
    seq_len: int = 0
    n_heads: int = 0

    @property
    def n_layers(self) -> int:
        """Number of Transformer layers captured."""
        return len(self.weights)

    @property
    def is_empty(self) -> bool:
        """True when no attention weights were captured."""
        return len(self.weights) == 0

    def mean_attention(self, head_reduction: str = "mean") -> torch.Tensor:
        """Average attention across layers and optionally across heads.

        Args:
            head_reduction: ``"mean"`` | ``"max"`` — how to reduce heads.

        Returns:
            ``(batch, seq_len, seq_len)`` mean attention matrix.

        Raises:
            RuntimeError: When the map is empty (non-Transformer model).
        """
        if self.is_empty:
            raise RuntimeError(
                "No attention weights captured — model may not use a Transformer backbone."
            )
        stacked = torch.stack(self.weights, dim=0)  # (L, B, H, S, S)
        if head_reduction == "mean":
            reduced = stacked.mean(dim=2)  # (L, B, S, S)
        elif head_reduction == "max":
            reduced = stacked.max(dim=2).values
        else:
            raise ValueError(f"head_reduction must be 'mean' or 'max', got {head_reduction!r}")
        return reduced.mean(dim=0)  # (B, S, S)

    def most_attended_timestep(self, sample_idx: int = 0) -> int:
        """Return the timestep (query position) with highest mean attention.

        Collapses across layers and heads, then sums over the key dimension
        to get a single importance score per query position.

        Args:
            sample_idx: Batch index (default 0).

        Returns:
            0-indexed position of the most attended query timestep.
        """
        mean = self.mean_attention()  # (B, S, S)
        query_importance = mean[sample_idx].sum(dim=-1)  # (S,)
        return int(query_importance.argmax().item())

    def layer_entropy(self) -> list[float]:
        """Compute per-layer mean attention entropy.

        Lower entropy = more focused (attends to fewer positions).

        Returns:
            Float per layer — lower is more focused.
        """
        scores = []
        for w in self.weights:
            mean_w = w.mean(dim=(0, 1))  # (S, S)
            eps = 1e-9
            entropy = -(mean_w * (mean_w + eps).log()).sum(dim=-1).mean()
            scores.append(float(entropy.item()))
        return scores

    def to_dict(self) -> dict[str, Any]:
        """Serialise metadata to a JSON-safe dict (shapes, not tensors)."""
        out: dict[str, Any] = {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "seq_len": self.seq_len,
            "layer_names": self.layer_names,
            "is_empty": self.is_empty,
        }
        if not self.is_empty:
            out["mean_attention_shape"] = list(self.mean_attention().shape)
            out["layer_entropy"] = self.layer_entropy()
        return out


# ---------------------------------------------------------------------------
# AttentionExtractor — hook-based weight capture
# ---------------------------------------------------------------------------


class AttentionExtractor:
    """Extracts attention weights from Transformer backbones via forward hooks.

    Registers hooks on every ``nn.MultiheadAttention`` sub-module found in the
    model.  Hooks are removed cleanly via the context manager even on exception.

    Usage::

        extractor = AttentionExtractor()
        with extractor.register(model):
            _ = model(input_tensor)
            attn_map = extractor.get_last_attention_map()

        # One-shot convenience:
        attn_map = AttentionExtractor.extract(model, input_tensor)
    """

    def __init__(self) -> None:
        self._captured: list[tuple[str, torch.Tensor]] = []
        self._hooks: list[Any] = []

    def _hook_fn(self, name: str):
        """Build a hook closure that captures attention weights by layer name."""

        def fn(module, inp, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    self._captured.append((name, attn_weights.detach().cpu()))

        return fn

    @contextlib.contextmanager
    def register(self, model: nn.Module):
        """Context manager: register hooks, yield, then remove hooks.

        Usage::

            with extractor.register(model):
                output = model(x)
        """
        self._captured.clear()
        self._hooks.clear()

        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._hook_fn(name))
                self._hooks.append(hook)

        if not self._hooks:
            logger.debug(
                "AttentionExtractor: no nn.MultiheadAttention modules found. "
                "Attention maps will be empty (non-Transformer backbone)."
            )
        try:
            yield self
        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()

    def get_last_attention_map(self) -> AttentionMap:
        """Return the AttentionMap captured during the last forward pass.

        Returns:
            ``AttentionMap`` populated with captured weights, or an empty
            ``AttentionMap`` when no Transformer layers were found.
        """
        if not self._captured:
            return AttentionMap(weights=[], seq_len=0, n_heads=0)

        weights = [w for _, w in self._captured]
        names = [n for n, _ in self._captured]

        # Infer seq_len and n_heads from the first captured tensor
        first = weights[0]
        if first.dim() == 4:
            n_heads = int(first.shape[1])
            seq_len = int(first.shape[-1])
        elif first.dim() == 3:
            # Some MHA implementations return (batch, seq, seq)
            n_heads = 1
            seq_len = int(first.shape[-1])
            weights = [w.unsqueeze(1) for w in weights]  # add head dim
        else:
            n_heads = 0
            seq_len = 0

        return AttentionMap(
            weights=weights,
            layer_names=names,
            seq_len=seq_len,
            n_heads=n_heads,
        )

    @classmethod
    def extract(
        cls,
        model: nn.Module,
        input_tensor: torch.Tensor,
    ) -> AttentionMap:
        """One-shot attention extraction convenience method.

        Puts the model in eval mode, runs a no-grad forward pass, captures
        attention weights, and restores the original training mode.

        Args:
            model: Model with (optionally) Transformer layers.
            input_tensor: ``(batch, seq_len, n_features)`` input.

        Returns:
            ``AttentionMap`` with captured weights (may be empty).
        """
        extractor = cls()
        was_training = model.training
        model.eval()
        try:
            with extractor.register(model):
                with torch.no_grad():
                    model(input_tensor)
            return extractor.get_last_attention_map()
        finally:
            if was_training:
                model.train()
