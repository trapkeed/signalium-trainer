"""ChampionRegistry — tracks lifecycle state of models in the promotion framework."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from trainer.research.champion.status import ModelStatus


@dataclass
class ModelEntry:
    """Metadata record for a model in the ChampionRegistry.

    Attributes:
        model_name: Registry name of the model.
        model_version: Semantic version string.
        status: Current lifecycle status.
        experiment_id: ID of the experiment that produced this model.
        metrics: Snapshot of the key metrics at promotion time.
        promoted_at: UTC timestamp of the most recent status change.
        notes: Free-text reason for promotion/demotion/retirement.
    """

    model_name: str
    model_version: str
    status: ModelStatus = ModelStatus.EXPERIMENTAL
    experiment_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    promoted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status.value,
            "experiment_id": self.experiment_id,
            "metrics": self.metrics,
            "promoted_at": self.promoted_at.isoformat(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelEntry":
        entry = cls.__new__(cls)
        entry.model_name = d["model_name"]
        entry.model_version = d["model_version"]
        entry.status = ModelStatus(d.get("status", "experimental"))
        entry.experiment_id = d.get("experiment_id", "")
        entry.metrics = d.get("metrics", {})
        raw_ts = d.get("promoted_at")
        entry.promoted_at = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
        entry.notes = d.get("notes", "")
        return entry


class ChampionRegistry:
    """Tracks model lifecycle states in the Champion/Challenger promotion framework.

    Only one model per ``model_name`` may hold CHAMPION status at a time.
    Promotion decisions are always manual — this registry records state
    transitions but never triggers them automatically.

    Thread-safe. Persists state to a JSON file.

    Usage::

        registry = ChampionRegistry(store_path=Path("experiments/champions.json"))

        registry.register("quant_v1", "1.0.0", experiment_id="abc123",
                          metrics={"f1_macro": 0.72})
        registry.promote("quant_v1", "1.0.0", ModelStatus.CANDIDATE)
        registry.promote("quant_v1", "1.0.0", ModelStatus.CHALLENGER)
        registry.promote("quant_v1", "1.0.0", ModelStatus.CHAMPION,
                         notes="Beat baseline on F1 by +4pp.")
        champion = registry.get_champion("quant_v1")

    Args:
        store_path: JSON file path. Created on first write.
    """

    def __init__(
        self,
        store_path: Path = Path("experiments/champions.json"),
    ) -> None:
        self._store_path = Path(store_path)
        self._lock = threading.Lock()
        self._entries: dict[str, list[ModelEntry]] = {}  # model_name → entries
        self._load()

    def register(
        self,
        model_name: str,
        model_version: str,
        experiment_id: str = "",
        metrics: Optional[dict[str, float]] = None,
        notes: str = "",
    ) -> ModelEntry:
        """Add a new model entry with EXPERIMENTAL status.

        Args:
            model_name: Registry name of the model.
            model_version: Semantic version.
            experiment_id: Linked experiment UUID.
            metrics: Initial metric snapshot.
            notes: Optional notes.

        Returns:
            The newly created ModelEntry.
        """
        entry = ModelEntry(
            model_name=model_name,
            model_version=model_version,
            status=ModelStatus.EXPERIMENTAL,
            experiment_id=experiment_id,
            metrics=metrics or {},
            notes=notes,
        )
        with self._lock:
            self._entries.setdefault(model_name, []).append(entry)
            self._flush()
        return entry

    def promote(
        self,
        model_name: str,
        model_version: str,
        new_status: ModelStatus,
        notes: str = "",
    ) -> ModelEntry:
        """Transition a model to a new lifecycle status.

        If ``new_status`` is CHAMPION, any existing champion for the same
        ``model_name`` is automatically demoted to RETIRED.

        Args:
            model_name: Registry name.
            model_version: Semantic version.
            new_status: Target ModelStatus.
            notes: Reason for the transition.

        Returns:
            Updated ModelEntry.

        Raises:
            KeyError: If the model/version is not registered.
            ValueError: If the target is RETIRED via this method (use :meth:`retire`).
        """
        if new_status == ModelStatus.RETIRED:
            return self.retire(model_name, model_version, notes=notes)

        with self._lock:
            entry = self._find(model_name, model_version)
            if entry is None:
                raise KeyError(f"Model '{model_name}' v{model_version} is not registered.")

            if new_status == ModelStatus.CHAMPION:
                for other in self._entries.get(model_name, []):
                    if other.status == ModelStatus.CHAMPION and other is not entry:
                        other.status = ModelStatus.RETIRED
                        other.promoted_at = datetime.now(timezone.utc)
                        other.notes = f"Auto-retired when {model_version} became champion."

            entry.status = new_status
            entry.promoted_at = datetime.now(timezone.utc)
            entry.notes = notes
            self._flush()
        return entry

    def retire(
        self,
        model_name: str,
        model_version: str,
        notes: str = "",
    ) -> ModelEntry:
        """Retire a model (terminal state — cannot be promoted after this).

        Args:
            model_name: Registry name.
            model_version: Semantic version.
            notes: Reason for retirement.

        Returns:
            Updated ModelEntry.

        Raises:
            KeyError: If the model/version is not registered.
        """
        with self._lock:
            entry = self._find(model_name, model_version)
            if entry is None:
                raise KeyError(f"Model '{model_name}' v{model_version} is not registered.")
            entry.status = ModelStatus.RETIRED
            entry.promoted_at = datetime.now(timezone.utc)
            entry.notes = notes
            self._flush()
        return entry

    def get_champion(self, model_name: str) -> Optional[ModelEntry]:
        """Return the current CHAMPION entry for a model name.

        Args:
            model_name: Registry name.

        Returns:
            ModelEntry with CHAMPION status, or None.
        """
        for entry in self._entries.get(model_name, []):
            if entry.status == ModelStatus.CHAMPION:
                return entry
        return None

    def get_challengers(self, model_name: str) -> list[ModelEntry]:
        """Return all CHALLENGER entries for a model name."""
        return [e for e in self._entries.get(model_name, []) if e.status == ModelStatus.CHALLENGER]

    def list_all(self, model_name: Optional[str] = None) -> list[ModelEntry]:
        """Return all entries, optionally filtered by model name.

        Args:
            model_name: If provided, only entries for this model are returned.

        Returns:
            List of ModelEntry objects.
        """
        if model_name is not None:
            return list(self._entries.get(model_name, []))
        return [e for entries in self._entries.values() for e in entries]

    def summary(self) -> list[dict[str, Any]]:
        """Return a summary table of all entries."""
        return [
            {
                "model_name": e.model_name,
                "version": e.model_version,
                "status": e.status.value,
                "f1_macro": e.metrics.get("f1_macro"),
                "accuracy": e.metrics.get("accuracy"),
                "promoted_at": e.promoted_at.isoformat(),
                "notes": e.notes[:60] if e.notes else "",
            }
            for e in self.list_all()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find(self, model_name: str, model_version: str) -> Optional[ModelEntry]:
        """Find a specific model/version entry (call with lock held)."""
        for entry in self._entries.get(model_name, []):
            if entry.model_version == model_version:
                return entry
        return None

    def _flush(self) -> None:
        """Persist to JSON (call with lock held)."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: [e.to_dict() for e in entries] for name, entries in self._entries.items()}
        self._store_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self) -> None:
        """Load from JSON on initialisation."""
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text())
            self._entries = {
                name: [ModelEntry.from_dict(e) for e in entries] for name, entries in data.items()
            }
        except Exception:
            self._entries = {}
