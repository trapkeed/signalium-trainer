"""Phase 17 — AEIP Ensemble Leaderboard.

Tracks performance rankings for:
  - Individual experts
  - The ensemble
  - The meta model
  - Champion
  - Retired models
  - Experimental models

Usage::

    lb = EnsembleLeaderboard()
    lb.update_expert("expert_01", {"accuracy": 0.62, "sharpe": 1.5})
    lb.update_ensemble({"accuracy": 0.68, "sharpe": 1.8})
    rankings = lb.get_rankings()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Leaderboard entry
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """One entry in the ensemble leaderboard.

    Attributes:
        entry_id: Unique UUID.
        entity_id: Expert ID, "ensemble", or "meta_model".
        entity_type: "expert" | "ensemble" | "meta_model".
        name: Human-readable name.
        status: "active" | "champion" | "retired" | "experimental".
        rank: Current rank (1 = best).
        score: Composite leaderboard score.
        accuracy: Prediction accuracy.
        sharpe_ratio: Rolling Sharpe.
        win_rate: Win rate.
        profit_factor: Profit factor.
        calibration_error: ECE.
        n_predictions: Total predictions.
        model_family: Architecture family.
        specializations: Expert specializations.
        updated_at: ISO-8601 last update timestamp.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = ""
    entity_type: str = "expert"
    name: str = ""
    status: str = "active"
    rank: int = 0
    score: float = 0.0
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    calibration_error: float = 0.0
    n_predictions: int = 0
    model_family: str = ""
    specializations: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "status": self.status,
            "rank": self.rank,
            "score": self.score,
            "accuracy": self.accuracy,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calibration_error": self.calibration_error,
            "n_predictions": self.n_predictions,
            "model_family": self.model_family,
            "specializations": self.specializations,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------------
# Composite leaderboard score
# ---------------------------------------------------------------------------


def _composite_score(metrics: dict[str, float]) -> float:
    """Compute composite leaderboard score from metrics dict."""
    acc = metrics.get("accuracy", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    pf = metrics.get("profit_factor", 1.0)
    cal_err = metrics.get("calibration_error", 0.5)
    norm_sharpe = min(1.0, max(0.0, (sharpe + 2.0) / 7.0))
    norm_pf = min(1.0, max(0.0, (pf - 0.5) / 4.5))
    norm_cal = 1.0 - min(1.0, cal_err)
    return 0.35 * acc + 0.25 * norm_sharpe + 0.15 * win_rate + 0.15 * norm_pf + 0.10 * norm_cal


# ---------------------------------------------------------------------------
# Ensemble leaderboard
# ---------------------------------------------------------------------------


class EnsembleLeaderboard:
    """Tracks and ranks all entities in the AEIP.

    Entities: individual experts, the ensemble, and the meta model.
    """

    def __init__(self) -> None:
        self._entries: dict[str, LeaderboardEntry] = {}
        self._metrics: dict[str, dict[str, float]] = {}

    def update_expert(
        self,
        expert_id: str,
        metrics: dict[str, float],
        name: str = "",
        status: str = "active",
        model_family: str = "",
        specializations: list[str] | None = None,
    ) -> LeaderboardEntry:
        """Update or create a leaderboard entry for an expert."""
        self._metrics[expert_id] = metrics
        score = _composite_score(metrics)
        if expert_id not in self._entries:
            self._entries[expert_id] = LeaderboardEntry(entity_id=expert_id, entity_type="expert")
        e = self._entries[expert_id]
        e.name = name or e.name or f"Expert {expert_id[:8]}"
        e.status = status
        e.score = round(score, 6)
        e.accuracy = metrics.get("accuracy", e.accuracy)
        e.sharpe_ratio = metrics.get("sharpe_ratio", e.sharpe_ratio)
        e.win_rate = metrics.get("win_rate", e.win_rate)
        e.profit_factor = metrics.get("profit_factor", e.profit_factor)
        e.calibration_error = metrics.get("calibration_error", e.calibration_error)
        e.n_predictions = metrics.get("n_predictions", e.n_predictions)
        e.model_family = model_family or e.model_family
        e.specializations = specializations or e.specializations
        e.updated_at = datetime.now(UTC).isoformat()
        self._rerank()
        return e

    def update_ensemble(
        self, metrics: dict[str, float], strategy: str = "ensemble"
    ) -> LeaderboardEntry:
        """Update the ensemble leaderboard entry."""
        entity_id = "ensemble"
        self._metrics[entity_id] = metrics
        score = _composite_score(metrics)
        if entity_id not in self._entries:
            self._entries[entity_id] = LeaderboardEntry(
                entity_id=entity_id, entity_type="ensemble", name=f"Ensemble ({strategy})"
            )
        e = self._entries[entity_id]
        e.score = round(score, 6)
        e.accuracy = metrics.get("accuracy", e.accuracy)
        e.sharpe_ratio = metrics.get("sharpe_ratio", e.sharpe_ratio)
        e.win_rate = metrics.get("win_rate", e.win_rate)
        e.profit_factor = metrics.get("profit_factor", e.profit_factor)
        e.calibration_error = metrics.get("calibration_error", e.calibration_error)
        e.n_predictions = metrics.get("n_predictions", e.n_predictions)
        e.updated_at = datetime.now(UTC).isoformat()
        self._rerank()
        return e

    def update_meta_model(self, metrics: dict[str, float]) -> LeaderboardEntry:
        """Update the meta model leaderboard entry."""
        entity_id = "meta_model"
        self._metrics[entity_id] = metrics
        score = _composite_score(metrics)
        if entity_id not in self._entries:
            self._entries[entity_id] = LeaderboardEntry(
                entity_id=entity_id, entity_type="meta_model", name="Ensemble Meta Model"
            )
        e = self._entries[entity_id]
        e.score = round(score, 6)
        e.accuracy = metrics.get("accuracy", e.accuracy)
        e.n_predictions = metrics.get("n_training_samples", e.n_predictions)
        e.updated_at = datetime.now(UTC).isoformat()
        self._rerank()
        return e

    def _rerank(self) -> None:
        """Re-compute ranks for all non-retired entries."""
        active = [e for e in self._entries.values() if e.status != "retired"]
        sorted_entries = sorted(active, key=lambda e: e.score, reverse=True)
        for rank, entry in enumerate(sorted_entries, start=1):
            entry.rank = rank
        for e in self._entries.values():
            if e.status == "retired":
                e.rank = 0

    def get_rankings(
        self, include_retired: bool = False, entity_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Return sorted leaderboard entries."""
        entries = list(self._entries.values())
        if not include_retired:
            entries = [e for e in entries if e.status != "retired"]
        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]
        return [e.to_dict() for e in sorted(entries, key=lambda e: e.rank)]

    def get_champion(self) -> dict[str, Any] | None:
        champions = [e for e in self._entries.values() if e.status == "champion"]
        if champions:
            return champions[0].to_dict()
        active = [
            e for e in self._entries.values() if e.entity_type == "expert" and e.status != "retired"
        ]
        if active:
            return max(active, key=lambda e: e.score).to_dict()
        return None

    def promote_champion(self, entity_id: str) -> None:
        """Promote entity to champion status."""
        for e in self._entries.values():
            if e.status == "champion":
                e.status = "active"
        if entity_id in self._entries:
            self._entries[entity_id].status = "champion"
        self._rerank()

    def retire(self, entity_id: str) -> None:
        if entity_id in self._entries:
            self._entries[entity_id].status = "retired"
            self._entries[entity_id].rank = 0

    def experts(self) -> list[dict[str, Any]]:
        return self.get_rankings(entity_type="expert")

    def retired(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries.values() if e.status == "retired"]

    def experimental(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries.values() if e.status == "experimental"]

    def summary(self) -> dict[str, Any]:
        all_entries = list(self._entries.values())
        return {
            "total_entries": len(all_entries),
            "n_experts": sum(1 for e in all_entries if e.entity_type == "expert"),
            "n_active": sum(1 for e in all_entries if e.status == "active"),
            "n_champion": sum(1 for e in all_entries if e.status == "champion"),
            "n_retired": sum(1 for e in all_entries if e.status == "retired"),
            "n_experimental": sum(1 for e in all_entries if e.status == "experimental"),
            "champion": self.get_champion(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {eid: e.to_dict() for eid, e in self._entries.items()},
        }
