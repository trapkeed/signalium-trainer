"""ExperimentTracker — persistent store for ExperimentRecords."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trainer.research.experiment.record import ExperimentRecord


class ExperimentTracker:
    """Thread-safe experiment store that persists records to a JSON file.

    All writes are atomic via a lock. The JSON file grows as experiments
    are added; call :meth:`load_all` to restore state from disk.

    Usage::

        tracker = ExperimentTracker(store_path=Path("experiments/records.json"))

        record = ExperimentRecord(name="lstm-baseline", model_name="quant_v1")
        tracker.save(record)

        # ... training ...
        record.complete(metrics={"accuracy": 0.72, "f1_macro": 0.68})
        tracker.save(record)

        completed = tracker.filter(status="completed")
        best = tracker.best_by("f1_macro")

    Args:
        store_path: Path to the JSON storage file. Created if it does not exist.
    """

    def __init__(self, store_path: Path = Path("experiments/records.json")) -> None:
        self._store_path = Path(store_path)
        self._lock = threading.Lock()
        self._records: dict[str, ExperimentRecord] = {}
        self._load_from_disk()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, record: ExperimentRecord) -> None:
        """Insert or update an experiment record.

        Args:
            record: ExperimentRecord to persist.
        """
        with self._lock:
            self._records[record.experiment_id] = record
            self._flush()

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        """Retrieve a single record by ID.

        Args:
            experiment_id: UUID string of the experiment.

        Returns:
            ExperimentRecord or None if not found.
        """
        return self._records.get(experiment_id)

    def delete(self, experiment_id: str) -> bool:
        """Remove a record from the store.

        Args:
            experiment_id: UUID string to remove.

        Returns:
            True if the record existed and was deleted; False otherwise.
        """
        with self._lock:
            if experiment_id in self._records:
                del self._records[experiment_id]
                self._flush()
                return True
            return False

    def load_all(self) -> list[ExperimentRecord]:
        """Return all stored records sorted by start time (oldest first).

        Returns:
            List of ExperimentRecords.
        """
        return sorted(self._records.values(), key=lambda r: r.start_time)

    # ------------------------------------------------------------------
    # Filtering & ranking
    # ------------------------------------------------------------------

    def filter(
        self,
        model_name: str | None = None,
        status: str | None = None,
        dataset_version: str | None = None,
        predicate: Callable[[ExperimentRecord], bool] | None = None,
    ) -> list[ExperimentRecord]:
        """Return records matching all provided filters.

        Args:
            model_name: Match exact model name.
            status: Match exact status (``"completed"``, ``"running"``, ``"failed"``).
            dataset_version: Match exact dataset version.
            predicate: Optional callable for custom filtering.

        Returns:
            Filtered list sorted by start time (oldest first).
        """
        results = list(self._records.values())
        if model_name is not None:
            results = [r for r in results if r.model_name == model_name]
        if status is not None:
            results = [r for r in results if r.status == status]
        if dataset_version is not None:
            results = [r for r in results if r.dataset_version == dataset_version]
        if predicate is not None:
            results = [r for r in results if predicate(r)]
        return sorted(results, key=lambda r: r.start_time)

    def best_by(
        self,
        metric: str,
        model_name: str | None = None,
        higher_is_better: bool = True,
    ) -> ExperimentRecord | None:
        """Return the experiment with the best value for a given metric.

        Args:
            metric: Metric key (e.g. ``"f1_macro"``).
            model_name: Optional filter by model name.
            higher_is_better: Set False for loss-like metrics.

        Returns:
            Best ExperimentRecord, or None if no completed records match.
        """
        candidates = self.filter(model_name=model_name, status="completed")
        candidates = [r for r in candidates if metric in r.metrics]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: (r.metrics[metric] if higher_is_better else -r.metrics[metric]),
        )

    def summary(self) -> list[dict[str, Any]]:
        """Return a concise table summary of all experiments.

        Returns:
            List of dicts with id, name, model, status, accuracy, f1, duration.
        """
        rows = []
        for rec in self.load_all():
            rows.append(
                {
                    "experiment_id": rec.experiment_id[:8],
                    "name": rec.name,
                    "model_name": rec.model_name,
                    "status": rec.status,
                    "accuracy": rec.metrics.get("accuracy"),
                    "f1_macro": rec.metrics.get("f1_macro"),
                    "duration_s": rec.training_duration_seconds,
                    "git": rec.git_commit_hash[:7] if rec.git_commit_hash else None,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Write all records to the JSON store (called with lock held)."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {eid: rec.to_dict() for eid, rec in self._records.items()}
        self._store_path.write_text(json.dumps(data, indent=2, default=str))

    def _load_from_disk(self) -> None:
        """Load records from the JSON store on initialisation."""
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text())
            self._records = {
                eid: ExperimentRecord.from_dict(rec_dict) for eid, rec_dict in data.items()
            }
        except Exception:
            self._records = {}
