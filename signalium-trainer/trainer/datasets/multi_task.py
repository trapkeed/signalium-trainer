"""MultiTaskParquetDataset — extends ParquetDataset for Phase 14.5 multi-task labels.

The dataset returns ``(x, labels_dict)`` tuples instead of ``(x, y)`` so that the
MultiTaskTrainingEngine can route each label to the correct head's loss function.

Required column:
    label (or ``direction_column``) — int class label (0=BUY, 1=SELL, 2=NO_TRADE)

Optional columns (provide any subset):
    expected_move   — float: expected price movement
    stop_loss       — float: optimal SL distance (≥ 0)
    take_profit     — float: optimal TP distance (≥ 0)
    holding_time    — float: expected trade duration
    tp_probability  — float: P(TP reached before SL), in [0, 1]
    confidence      — float: target confidence [0, 1]

Missing optional columns produce no key in the labels dict, which the
MultiTaskLoss silently skips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from trainer.datasets.base import BaseDataset


_OPTIONAL_LABEL_COLUMNS: dict[str, str] = {
    "expected_move": "y_move",
    "stop_loss": "y_sl",
    "take_profit": "y_tp",
    "holding_time": "y_holding_time",
    "tp_probability": "y_tp_prob",
    "confidence": "y_confidence",
}


class MultiTaskParquetDataset(Dataset):
    """Parquet-backed dataset that yields multi-task label dicts.

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

        known_label_cols = {direction_column} | set(_OPTIONAL_LABEL_COLUMNS.keys())

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in known_label_cols]
        self._feature_columns = feature_columns

        self._features: np.ndarray = df[feature_columns].to_numpy(dtype=np.float32)
        self._direction: np.ndarray = df[direction_column].to_numpy(dtype=np.int64)

        self._optional: dict[str, np.ndarray] = {}
        for col, key in _OPTIONAL_LABEL_COLUMNS.items():
            if col in df.columns:
                self._optional[key] = df[col].to_numpy(dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self._direction.max()) + 1

    @property
    def available_label_keys(self) -> list[str]:
        """Keys that will appear in the labels dict for every sample."""
        return ["y_direction"] + list(self._optional.keys())

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.tensor(self._features[index], dtype=self._dtype)
        labels: dict[str, torch.Tensor] = {
            "y_direction": torch.tensor(self._direction[index], dtype=torch.long),
        }
        for key, arr in self._optional.items():
            labels[key] = torch.tensor(arr[index], dtype=torch.float32)
        return x, labels


def collate_multi_task(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Custom collate function for MultiTaskParquetDataset batches.

    Stacks feature tensors and assembles a dict of batched label tensors.
    Drop-in replacement for the default DataLoader collate_fn.

    Args:
        batch: List of (x, labels_dict) tuples from the dataset.

    Returns:
        (x_batch, labels_batch) where x_batch is (B, ...) and
        labels_batch maps each key to a batched tensor of shape (B, ...).
    """
    xs, label_list = zip(*batch)
    x_batch = torch.stack(xs)
    keys = label_list[0].keys()
    labels_batch: dict[str, torch.Tensor] = {
        k: torch.stack([d[k] for d in label_list]) for k in keys
    }
    return x_batch, labels_batch
