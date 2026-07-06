"""MultiTaskParquetDataset — Phase 14.6 Institutional Analytics labels.

Returns ``(x, labels_dict)`` tuples so that MultiTaskTrainingEngine can
route each label to the correct head's loss term.

Required column:
    label (or ``direction_column``) — int class label (0=BUY, 1=SELL, 2=NO_TRADE)

Phase 14.5 optional scalar columns:
    expected_move   — float: expected price movement (signed)
    stop_loss       — float: optimal SL distance (≥ 0)
    take_profit     — float: optimal TP distance (≥ 0)
    holding_time    — float: expected trade duration (≥ 0)
    tp_probability  — float: P(TP reached before SL) ∈ [0, 1]
    confidence      — float: target confidence ∈ [0, 1]

Phase 14.6 optional scalar columns:
    mfe                  — float: max favorable excursion (≥ 0)
    mae_excursion        — float: max adverse excursion (≥ 0)
    trade_efficiency     — float: % of move captured ∈ [0, 100]
    reward_risk          — float: expected R:R ratio (≥ 0)
    sl_resilience        — float: P(SL survives noise) ∈ [0, 1]
    tp_achievability     — float: P(TP achieved) ∈ [0, 1]
    entry_quality        — float: entry quality score ∈ [0, 100]
    exit_quality         — float: exit quality score ∈ [0, 100]
    market_stability     — int:   0-3 (Stable/ModStable/Unstable/HighlyUnstable)
    position_size_quality— float: capital allocation quality ∈ [0, 100]

Phase 14.6 optional vector columns (stored as list/array in Parquet):
    trade_survival      — float[K]: survival probs at K bar horizons
    trade_decay         — float[D+1]: [decay_prob, decay curve at D steps]
    profit_distribution — float[6]: [mean, median, var, skew, ci_lo, ci_hi]
    loss_distribution   — float[4]: [mean, median, var, tail_risk]
    uncertainty         — float[2]: [epistemic, aleatoric]

Missing columns produce no key in the labels dict, which MultiTaskLoss
silently skips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from trainer.datasets.base import BaseDataset

# ---------------------------------------------------------------------------
# Column → label key mappings
# ---------------------------------------------------------------------------

# Scalar columns (value → LongTensor or FloatTensor per row)
_SCALAR_FLOAT_COLS: dict[str, str] = {
    # Phase 14.5
    "expected_move": "y_move",
    "stop_loss": "y_sl",
    "take_profit": "y_tp",
    "holding_time": "y_holding_time",
    "tp_probability": "y_tp_prob",
    "confidence": "y_confidence",
    # Phase 14.6
    "mfe": "y_mfe",
    "mae_excursion": "y_mae_excursion",
    "trade_efficiency": "y_trade_efficiency",
    "reward_risk": "y_reward_risk",
    "sl_resilience": "y_sl_resilience",
    "tp_achievability": "y_tp_achievability",
    "entry_quality": "y_entry_quality",
    "exit_quality": "y_exit_quality",
    "position_size_quality": "y_position_size_quality",
}

# Scalar int column (direction label for market stability)
_SCALAR_INT_COLS: dict[str, str] = {
    "market_stability": "y_market_stability",
}

# Vector columns — each cell is a list/array; shape (K,) per row
_VECTOR_COLS: dict[str, str] = {
    "trade_survival": "y_trade_survival",
    "trade_decay": "y_trade_decay",
    "profit_distribution": "y_profit_distribution",
    "loss_distribution": "y_loss_distribution",
    "uncertainty": "y_uncertainty",
}


class MultiTaskParquetDataset(Dataset):
    """Parquet-backed dataset that yields (x, labels_dict) for multi-task training.

    Args:
        root: Directory containing split Parquet files.
        split: "train", "val", or "test".
        feature_columns: Columns to use as features (auto-detected if None).
        direction_column: Column holding the direction label (default "label").
        dtype: torch dtype for feature tensors.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        feature_columns: Optional[list[str]] = None,
        direction_column: str = "label",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._root = Path(root)
        self._split = split
        self._direction_column = direction_column
        self._dtype = dtype

        parquet_path = self._root / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df: pd.DataFrame = pd.read_parquet(parquet_path)

        # Determine which columns are labels (exclude from features)
        known_label_cols: set[str] = (
            {direction_column}
            | set(_SCALAR_FLOAT_COLS.keys())
            | set(_SCALAR_INT_COLS.keys())
            | set(_VECTOR_COLS.keys())
        )

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in known_label_cols]
        self._feature_columns = feature_columns

        self._features: np.ndarray = df[feature_columns].to_numpy(dtype=np.float32)
        self._direction: np.ndarray = df[direction_column].to_numpy(dtype=np.int64)

        # Scalar float labels
        self._scalar_float: dict[str, np.ndarray] = {}
        for col, key in _SCALAR_FLOAT_COLS.items():
            if col in df.columns:
                self._scalar_float[key] = df[col].to_numpy(dtype=np.float32)

        # Scalar int labels
        self._scalar_int: dict[str, np.ndarray] = {}
        for col, key in _SCALAR_INT_COLS.items():
            if col in df.columns:
                self._scalar_int[key] = df[col].to_numpy(dtype=np.int64)

        # Vector labels — each row is an array/list
        self._vector: dict[str, np.ndarray] = {}
        for col, key in _VECTOR_COLS.items():
            if col in df.columns:
                arr = np.stack(df[col].to_numpy())
                self._vector[key] = arr.astype(np.float32)

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self._direction.max()) + 1

    @property
    def available_label_keys(self) -> list[str]:
        """All label keys that will appear in every sample's labels dict."""
        return (
            ["y_direction"]
            + list(self._scalar_float.keys())
            + list(self._scalar_int.keys())
            + list(self._vector.keys())
        )

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.tensor(self._features[index], dtype=self._dtype)
        labels: dict[str, torch.Tensor] = {
            "y_direction": torch.tensor(self._direction[index], dtype=torch.long),
        }
        for key, arr in self._scalar_float.items():
            labels[key] = torch.tensor(arr[index], dtype=torch.float32)
        for key, arr in self._scalar_int.items():
            labels[key] = torch.tensor(arr[index], dtype=torch.long)
        for key, arr in self._vector.items():
            labels[key] = torch.tensor(arr[index], dtype=torch.float32)
        return x, labels


def collate_multi_task(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Custom collate function for MultiTaskParquetDataset batches.

    Stacks feature tensors and assembles a dict of batched label tensors.
    Drop-in replacement for the DataLoader ``collate_fn``.

    Args:
        batch: List of (x, labels_dict) tuples from the dataset.

    Returns:
        (x_batch, labels_batch) where labels_batch maps each key to a
        batched tensor of shape (B, ...).
    """
    xs, label_list = zip(*batch)
    x_batch = torch.stack(xs)
    keys = label_list[0].keys()
    labels_batch: dict[str, torch.Tensor] = {
        k: torch.stack([d[k] for d in label_list]) for k in keys
    }
    return x_batch, labels_batch
