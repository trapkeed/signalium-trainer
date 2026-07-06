"""Dataset package — Parquet-based PyTorch Dataset implementations."""

from trainer.datasets.base import BaseDataset
from trainer.datasets.multi_task import MultiTaskParquetDataset, collate_multi_task
from trainer.datasets.parquet import ParquetDataset
from trainer.datasets.validation import DatasetValidator
from trainer.datasets.versioning import DatasetVersion

__all__ = [
    "BaseDataset",
    "ParquetDataset",
    "DatasetValidator",
    "DatasetVersion",
    "MultiTaskParquetDataset",
    "collate_multi_task",
]
