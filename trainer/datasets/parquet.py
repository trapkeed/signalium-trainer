"""PyTorch Dataset backed by Parquet files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from trainer.datasets.base import BaseDataset


class ParquetDataset(BaseDataset):
    """Dataset that reads feature/label data from Parquet files.

    Parquet files are loaded eagerly into memory at construction time.
    For very large datasets use :class:`StreamingParquetDataset`.

    Args:
        root: Directory containing split Parquet files.
        split: "train", "val", or "test".
        feature_columns: Column names to use as features.
        label_column: Column name to use as the label.
        transform: Optional callable applied to each sample tensor.
        dtype: torch dtype for the feature tensor.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        feature_columns: list[str] | None = None,
        label_column: str = "label",
        transform: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(root=root, split=split, transform=transform)
        self._label_column = label_column
        self._dtype = dtype

        parquet_path = self._root / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df: pd.DataFrame = pd.read_parquet(parquet_path)

        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != label_column]
        self._feature_columns = feature_columns

        self._features: np.ndarray = df[feature_columns].to_numpy(dtype=np.float32)
        self._labels: np.ndarray = df[label_column].to_numpy()
        self._num_classes = int(self._labels.max()) + 1 if self._labels.dtype.kind in "iuf" else 0

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self._features[index], dtype=self._dtype)
        y = torch.tensor(self._labels[index])
        if self._transform is not None:
            x = self._transform(x)
        return x, y


class StreamingParquetDataset(BaseDataset):
    """Lazy-loading Dataset that reads row groups on demand.

    Suitable for datasets that do not fit in RAM.

    Args:
        root: Directory containing split Parquet files.
        split: "train", "val", or "test".
        feature_columns: Column names to use as features.
        label_column: Column name to use as the label.
        transform: Optional callable applied to each sample tensor.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        feature_columns: list[str] | None = None,
        label_column: str = "label",
        transform: Any = None,
    ) -> None:
        import pyarrow.parquet as pq

        super().__init__(root=root, split=split, transform=transform)
        self._label_column = label_column

        parquet_path = self._root / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self._pf = pq.ParquetFile(parquet_path)
        schema = self._pf.schema_arrow
        all_cols = schema.names

        self._feature_columns = feature_columns or [c for c in all_cols if c != label_column]
        self._length = self._pf.metadata.num_rows

        # Read first row group to infer dimensions
        first_batch = self._pf.read_row_group(0, columns=self._feature_columns)
        self._feature_dim = first_batch.num_columns
        self._num_classes = 0  # Requires full scan; set externally if known.

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        # Locate row group and row within group
        row_group_sizes = [
            self._pf.metadata.row_group(i).num_rows for i in range(self._pf.metadata.num_row_groups)
        ]
        cumulative = 0
        for rg_idx, rg_size in enumerate(row_group_sizes):
            if cumulative + rg_size > index:
                local_idx = index - cumulative
                batch = self._pf.read_row_group(
                    rg_idx, columns=self._feature_columns + [self._label_column]
                )
                row = batch.slice(local_idx, 1)
                x_arr = row.select(self._feature_columns).to_pydict()
                y_arr = row.column(self._label_column)[0].as_py()
                x = torch.tensor([x_arr[c][0] for c in self._feature_columns], dtype=torch.float32)
                y = torch.tensor(y_arr)
                if self._transform is not None:
                    x = self._transform(x)
                return x, y
            cumulative += rg_size
        raise IndexError(f"Index {index} out of range for dataset of length {self._length}")
