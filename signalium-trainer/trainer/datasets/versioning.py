"""Dataset versioning utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class DatasetVersion:
    """Immutable record of a dataset version.

    Args:
        dataset_id: Signalium dataset identifier.
        version: Semantic version string (e.g. "1.0.0").
        feature_version: Version of the feature pipeline used.
        label_version: Version of the label pipeline used.
        created_at: ISO timestamp when the version was produced.
        metadata: Arbitrary key-value metadata.
    """

    dataset_id: str
    version: str
    feature_version: Optional[str] = None
    label_version: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "DatasetVersion":
        """Load a DatasetVersion from a manifest.json file.

        Args:
            manifest_path: Path to the manifest file.

        Returns:
            DatasetVersion parsed from the manifest.
        """
        with manifest_path.open("r") as fh:
            data = json.load(fh)
        return cls(
            dataset_id=data["dataset_id"],
            version=data["version"],
            feature_version=data.get("feature_version"),
            label_version=data.get("label_version"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def save(self, path: Path) -> None:
        """Write the version record to a JSON file.

        Args:
            path: Destination path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
