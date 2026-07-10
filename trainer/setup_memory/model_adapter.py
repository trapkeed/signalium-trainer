"""S32 — Setup Memory Engine model-input adapter.

Bridges ``SetupMemoryEngine.feature_vector()`` into the Hybrid Encoder's
existing ``HybridInput.memory_state`` slot (see
``trainer/models/quant_v1/hybrid/input.py``), so setup state reaches the
model as an input feature without adding a new stream or touching the
encoder itself.

This module has an optional torch dependency: it is only imported by
callers that actually build ``HybridInput`` tensors, keeping
``trainer.setup_memory`` itself usable without torch installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trainer.setup_memory.engine import SetupMemoryEngine

if TYPE_CHECKING:
    import torch

#: Fixed feature layout appended after the one-hot state (7) —
#: [state_confidence, state_age_normalized, expected_direction_scalar,
#:  expected_move_size, expected_holding_period_s]. Keep in sync with
#: ``setup_state_feature_dim()`` below.
_SCALAR_FIELDS = (
    "state_confidence",
    "state_age_normalized",
    "expected_direction_scalar",
    "expected_move_size",
    "expected_holding_period_s",
)


def setup_state_feature_dim() -> int:
    """Width of the flat feature vector produced by ``setup_state_features``."""
    return len(SetupMemoryEngine.STATE_ORDER) + len(_SCALAR_FIELDS)


def setup_state_features(engine: SetupMemoryEngine, symbol: str, timeframe: str) -> list[float]:
    """Flatten one (symbol, timeframe) setup memory into a fixed-width list
    of floats: one-hot state (7) + scalar fields (see ``_SCALAR_FIELDS``).
    Safe to call for a symbol/timeframe that has never been seen — returns
    the neutral IDLE feature vector."""
    fv: dict[str, Any] = engine.feature_vector(symbol, timeframe)
    return list(fv["state_onehot"]) + [float(fv[field]) for field in _SCALAR_FIELDS]


def batch_setup_state_tensor(
    engine: SetupMemoryEngine, symbol_timeframe_pairs: list[tuple[str, str]]
) -> "torch.Tensor":
    """Build a ``(batch, setup_state_feature_dim())`` tensor suitable for
    ``HybridInput.memory_state``, one row per (symbol, timeframe) pair in
    the batch (same order)."""
    import torch

    rows = [
        setup_state_features(engine, symbol, timeframe)
        for symbol, timeframe in symbol_timeframe_pairs
    ]
    return torch.tensor(rows, dtype=torch.float32)
