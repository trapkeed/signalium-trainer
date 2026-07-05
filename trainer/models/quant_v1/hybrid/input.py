"""HybridInput — structured input container for the Hybrid Encoder.

When the hybrid encoder is enabled, callers pass a ``HybridInput`` to the
model instead of a plain ``torch.Tensor``.  All fields are optional; fields
not provided simply skip the corresponding stream.

Backward-compatible usage: existing callers passing a plain ``torch.Tensor``
continue to work unchanged — the model's ``forward()`` detects the type and
routes accordingly.

Example::

    from trainer.models.quant_v1.hybrid.input import HybridInput

    x = HybridInput(
        raw_sequence=torch.randn(batch, seq_len, 7),
        engineered_features=torch.randn(batch, 64),
        context_ids={
            "instrument": torch.zeros(batch, dtype=torch.long),
            "session": torch.ones(batch, dtype=torch.long),
            "hour": torch.randint(0, 24, (batch,)),
        },
    )
    logits = model(x)  # still returns (batch, 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class HybridInput:
    """Structured multi-stream input container for Phase 14.9A.

    All fields are optional.  The ``HybridEncoder`` enables each stream only
    when its corresponding tensor is provided AND that stream's config has
    ``enabled=True``.

    Attributes:
        raw_sequence: Raw OHLCV-Spread-Timestamp tensor.
            Shape: ``(batch, seq_len, raw_input_dim)``.
        multi_timeframe: Per-timeframe feature tensors.
            Keys are timeframe strings (e.g. "1m", "5m", "1h").
            Values are ``(batch, tf_seq_len, feat_dim)`` tensors.
        engineered_features: Pre-computed feature vector.
            Shape: ``(batch, num_features)``.
        context_ids: Categorical context identifiers (integer IDs).
            Keys match ``ContextEmbeddingStreamConfig.embeddings`` names.
            Values are ``(batch,)`` long tensors.
        memory_state: Market Memory Engine state vector.
            Shape: ``(batch, memory_dim)``.
    """

    raw_sequence: Optional[torch.Tensor] = None
    multi_timeframe: Optional[dict[str, torch.Tensor]] = None
    engineered_features: Optional[torch.Tensor] = None
    context_ids: Optional[dict[str, torch.Tensor]] = None
    memory_state: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        """Infer batch size from whichever field is available first."""
        for candidate in (
            self.raw_sequence,
            self.engineered_features,
            self.memory_state,
        ):
            if candidate is not None:
                return candidate.shape[0]
        if self.multi_timeframe:
            return next(iter(self.multi_timeframe.values())).shape[0]
        if self.context_ids:
            return next(iter(self.context_ids.values())).shape[0]
        raise ValueError("HybridInput is empty — at least one field must be provided.")

    @property
    def device(self) -> torch.device:
        """Infer device from whichever field is available first."""
        for candidate in (
            self.raw_sequence,
            self.engineered_features,
            self.memory_state,
        ):
            if candidate is not None:
                return candidate.device
        if self.multi_timeframe:
            return next(iter(self.multi_timeframe.values())).device
        if self.context_ids:
            return next(iter(self.context_ids.values())).device
        return torch.device("cpu")
