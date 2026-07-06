"""Phase 15 — Institutional Model Leaderboard.

A permanent, sortable, filterable leaderboard tracking every model that
has passed through the validation platform.

Stores all fields required by the institutional research specification:
  Model ID, Version, Training Date, Dataset Version, Feature Version,
  Label Version, Architecture, Backbone, Parameter Count, Training Time,
  Validation Score, Profit Factor, Sharpe, Sortino, Max Drawdown,
  Calibration Score, Promotion Status, Champion Status, Deployment History,
  Retirement Status, Notes.

Supports:
  - Sort by any field (ascending / descending)
  - Filter by status, architecture, date range
  - Search by model name or notes
  - Compare arbitrary entries
  - Persist to / load from JSON

Usage::

    lb = InstitutionalLeaderboard()
    lb.add(entry)
    top = lb.top(10, sort_by="sharpe_ratio")
    lb.save("leaderboard.json")
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Leaderboard entry
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """A single row in the institutional leaderboard.

    All fields are optional with safe defaults to allow partial data entry.
    """

    model_id: str = ""
    model_version: str = "1.0.0"
    model_name: str = ""
    training_date: str = ""
    architecture: str = ""
    backbone: str = ""
    phase: str = ""
    n_parameters: int = 0
    training_time_seconds: float = 0.0
    dataset_version: str = ""
    feature_version: str = ""
    label_version: str = ""
    # Statistical metrics
    validation_accuracy: float = 0.0
    validation_f1: float = 0.0
    ece: float = 0.0
    brier_score: float = 0.0
    roc_auc: float = 0.0
    # Trading metrics
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    # Infrastructure
    inference_latency_ms: float = 0.0
    model_size_mb: float = 0.0
    # Lifecycle
    promotion_status: str = (
        "experimental"  # experimental | candidate | challenger | champion | retired
    )
    is_champion: bool = False
    deployment_history: list[dict[str, str]] = field(default_factory=list)
    is_retired: bool = False
    # Scoring
    overall_score: float = 0.0
    # Meta
    experiment_id: str = ""
    git_commit: str = ""
    n_validation_folds: int = 0
    notes: str = ""
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "training_date": self.training_date,
            "architecture": self.architecture,
            "backbone": self.backbone,
            "phase": self.phase,
            "n_parameters": self.n_parameters,
            "training_time_seconds": round(self.training_time_seconds, 1),
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "validation_accuracy": round(self.validation_accuracy, 4),
            "validation_f1": round(self.validation_f1, 4),
            "ece": round(self.ece, 4),
            "brier_score": round(self.brier_score, 4),
            "roc_auc": round(self.roc_auc, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "expectancy": round(self.expectancy, 4),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "model_size_mb": round(self.model_size_mb, 2),
            "promotion_status": self.promotion_status,
            "is_champion": self.is_champion,
            "is_retired": self.is_retired,
            "overall_score": round(self.overall_score, 4),
            "experiment_id": self.experiment_id,
            "git_commit": self.git_commit,
            "n_validation_folds": self.n_validation_folds,
            "notes": self.notes,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LeaderboardEntry":
        entry = cls()
        for k, v in d.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        return entry


# ---------------------------------------------------------------------------
# Institutional Leaderboard
# ---------------------------------------------------------------------------


class InstitutionalLeaderboard:
    """Permanent institutional model leaderboard.

    Thread-safe. Persists to JSON.

    Usage::

        lb = InstitutionalLeaderboard(store_path=Path("leaderboard.json"))
        lb.add(entry)
        top = lb.top(10, sort_by="sharpe_ratio")
        champions = lb.filter(is_champion=True)
        print(lb.format_table())
    """

    def __init__(
        self,
        store_path: Optional[Path] = None,
        score_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self._path = Path(store_path) if store_path else None
        self._lock = threading.Lock()
        self._entries: list[LeaderboardEntry] = []
        self._weights = score_weights or {
            "validation_accuracy": 0.15,
            "validation_f1": 0.15,
            "sharpe_ratio": 0.20,
            "profit_factor": 0.15,
            "max_drawdown": -0.10,
            "ece": -0.10,
            "win_rate": 0.15,
        }
        if self._path and self._path.exists():
            self._load()

    # ── CRUD ─────────────────────────────────────────────────────────────

    def add(self, entry: LeaderboardEntry) -> LeaderboardEntry:
        """Add a new entry, computing the overall score automatically."""
        entry.overall_score = self._compute_score(entry)
        with self._lock:
            self._entries.append(entry)
            self._persist()
        logger.info(
            f"Leaderboard: added {entry.model_name or entry.model_id!r} "
            f"score={entry.overall_score:.4f}"
        )
        return entry

    def update(self, model_id: str, updates: dict[str, Any]) -> Optional[LeaderboardEntry]:
        """Update fields on an existing entry by model_id."""
        with self._lock:
            for e in self._entries:
                if e.model_id == model_id:
                    for k, v in updates.items():
                        if hasattr(e, k):
                            setattr(e, k, v)
                    e.overall_score = self._compute_score(e)
                    self._persist()
                    return e
        return None

    def remove(self, model_id: str) -> bool:
        """Remove an entry by model_id."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.model_id != model_id]
            self._persist()
        return len(self._entries) < before

    # ── Querying ──────────────────────────────────────────────────────────

    def top(
        self,
        k: int = 10,
        sort_by: str = "overall_score",
        ascending: bool = False,
        filter_fn: Optional[Any] = None,
    ) -> list[LeaderboardEntry]:
        """Return top-k entries sorted by a field.

        Args:
            k: Number of entries.
            sort_by: Field name to sort by.
            ascending: Sort ascending when True.
            filter_fn: Optional callable ``(entry) → bool``.

        Returns:
            Sorted list of ``LeaderboardEntry``.
        """
        entries = [e for e in self._entries if (filter_fn is None or filter_fn(e))]
        entries.sort(
            key=lambda e: float(getattr(e, sort_by, 0.0)),
            reverse=not ascending,
        )
        return entries[:k]

    def filter(
        self,
        promotion_status: Optional[str] = None,
        architecture: Optional[str] = None,
        is_champion: Optional[bool] = None,
        is_retired: Optional[bool] = None,
        phase: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        max_drawdown_limit: Optional[float] = None,
    ) -> list[LeaderboardEntry]:
        """Filter entries by multiple criteria."""
        results = list(self._entries)
        if promotion_status:
            results = [e for e in results if e.promotion_status == promotion_status]
        if architecture:
            results = [e for e in results if e.architecture == architecture]
        if is_champion is not None:
            results = [e for e in results if e.is_champion == is_champion]
        if is_retired is not None:
            results = [e for e in results if e.is_retired == is_retired]
        if phase:
            results = [e for e in results if e.phase == phase]
        if min_accuracy is not None:
            results = [e for e in results if e.validation_accuracy >= min_accuracy]
        if max_drawdown_limit is not None:
            results = [e for e in results if e.max_drawdown <= max_drawdown_limit]
        return results

    def search(self, query: str) -> list[LeaderboardEntry]:
        """Full-text search across model_name, model_id, notes, and architecture."""
        q = query.lower()
        return [
            e
            for e in self._entries
            if (
                q in e.model_name.lower()
                or q in e.model_id.lower()
                or q in e.notes.lower()
                or q in e.architecture.lower()
                or q in e.backbone.lower()
            )
        ]

    def get(self, model_id: str) -> Optional[LeaderboardEntry]:
        """Retrieve an entry by model_id."""
        for e in self._entries:
            if e.model_id == model_id:
                return e
        return None

    def champion(self) -> Optional[LeaderboardEntry]:
        """Return the current champion entry."""
        for e in self._entries:
            if e.is_champion and not e.is_retired:
                return e
        return None

    def compare(self, model_id_a: str, model_id_b: str) -> dict[str, Any]:
        """Compare two entries side by side."""
        a = self.get(model_id_a)
        b = self.get(model_id_b)
        if not a or not b:
            return {"error": "One or both model IDs not found."}
        metrics = [
            "validation_accuracy",
            "validation_f1",
            "sharpe_ratio",
            "profit_factor",
            "max_drawdown",
            "ece",
            "win_rate",
            "expectancy",
            "overall_score",
        ]
        return {
            "model_a": model_id_a,
            "model_b": model_id_b,
            "comparison": {
                m: {"a": round(getattr(a, m, 0.0), 4), "b": round(getattr(b, m, 0.0), 4)}
                for m in metrics
            },
        }

    # ── Formatting ─────────────────────────────────────────────────────────

    def format_table(
        self,
        k: int = 20,
        sort_by: str = "overall_score",
    ) -> str:
        """Return an ASCII table of the top-k entries."""
        entries = self.top(k=k, sort_by=sort_by)
        header = (
            f"{'#':>3s}  {'Model':<28s}  {'Version':<8s}  "
            f"{'Acc':>6s}  {'F1':>6s}  {'Sharpe':>7s}  "
            f"{'PF':>6s}  {'DD':>6s}  {'Score':>7s}  {'Status':<12s}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for i, e in enumerate(entries, 1):
            champion_marker = "*" if e.is_champion else " "
            lines.append(
                f"{champion_marker}{i:>2d}  {(e.model_name or e.model_id):<28.28s}  "
                f"{e.model_version:<8.8s}  "
                f"{e.validation_accuracy:>6.4f}  {e.validation_f1:>6.4f}  "
                f"{e.sharpe_ratio:>7.3f}  {e.profit_factor:>6.3f}  "
                f"{e.max_drawdown:>6.3f}  {e.overall_score:>7.4f}  "
                f"{e.promotion_status:<12.12s}"
            )
        lines.append(sep)
        lines.append(f"  (* = Champion)  {len(self._entries)} total entries  sorted by {sort_by}")
        return "\n".join(lines)

    def all_entries(self) -> list[LeaderboardEntry]:
        return list(self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [e.to_dict() for e in self._entries]}

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Save leaderboard to JSON."""
        out_path = Path(path) if path else self._path
        if out_path is None:
            raise ValueError("No path provided and no default store_path set.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        logger.info(f"Leaderboard saved to {out_path}")
        return out_path

    @classmethod
    def load(cls, path: Path) -> "InstitutionalLeaderboard":
        """Load leaderboard from JSON."""
        lb = cls(store_path=path)
        return lb

    # ── Private ────────────────────────────────────────────────────────────

    def _compute_score(self, e: LeaderboardEntry) -> float:
        """Compute weighted composite score (clamped to [0, 1])."""
        score = 0.0
        for metric, weight in self._weights.items():
            val = float(getattr(e, metric, 0.0))
            score += weight * val
        return float(max(0.0, min(1.0, score)))

    def _persist(self) -> None:
        if self._path:
            try:
                self.save()
            except Exception as exc:
                logger.warning(f"Leaderboard persist failed: {exc}")

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
            self._entries = [LeaderboardEntry.from_dict(e) for e in data.get("entries", [])]
            logger.info(f"Leaderboard loaded {len(self._entries)} entries from {self._path}")
        except Exception as exc:
            logger.warning(f"Leaderboard load failed: {exc}")
            self._entries = []
