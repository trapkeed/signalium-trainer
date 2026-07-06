"""Phase 19.1 — Research Cost Tracker.

Estimates and accumulates the cost of every experiment:

  Per-experiment:
    - GPU hours
    - CPU hours
    - Training cost (USD)
    - Storage cost (USD)
    - Inference cost (USD)
    - Validation cost (USD)

  Platform totals:
    - Total GPU hours
    - Total CPU hours
    - Total cost by category
    - Cost per project / branch / researcher

Default GPU rate: $2.00 / GPU-hour (configurable)
Default CPU rate: $0.05 / CPU-hour (configurable)
Default storage rate: $0.023 / GB-month (AWS S3-equivalent)

Usage::

    tracker = CostTracker(gpu_rate_per_hour=2.00, cpu_rate_per_hour=0.05)
    cost = tracker.record(
        entity_id="exp_001",
        gpu_hours=4.5,
        cpu_hours=2.0,
        storage_gb_months=10.0,
        validation_hours=0.5,
    )
    print(cost.total_cost_usd)
    report = tracker.cost_report()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# ExperimentCost
# ---------------------------------------------------------------------------


@dataclass
class ExperimentCost:
    """Cost breakdown for a single experiment or entity."""

    cost_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_id: str = ""
    entity_type: str = "experiment"
    project: str = ""
    branch: str = ""
    researcher: str = "system"
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Resource consumption
    gpu_hours: float = 0.0
    cpu_hours: float = 0.0
    storage_gb_months: float = 0.0
    validation_gpu_hours: float = 0.0
    inference_hours: float = 0.0

    # USD costs
    training_cost_usd: float = 0.0
    cpu_cost_usd: float = 0.0
    storage_cost_usd: float = 0.0
    validation_cost_usd: float = 0.0
    inference_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        return (
            self.training_cost_usd
            + self.cpu_cost_usd
            + self.storage_cost_usd
            + self.validation_cost_usd
            + self.inference_cost_usd
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cost_id": self.cost_id,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "project": self.project,
            "branch": self.branch,
            "researcher": self.researcher,
            "recorded_at": self.recorded_at,
            "gpu_hours": self.gpu_hours,
            "cpu_hours": self.cpu_hours,
            "storage_gb_months": self.storage_gb_months,
            "validation_gpu_hours": self.validation_gpu_hours,
            "inference_hours": self.inference_hours,
            "training_cost_usd": self.training_cost_usd,
            "cpu_cost_usd": self.cpu_cost_usd,
            "storage_cost_usd": self.storage_cost_usd,
            "validation_cost_usd": self.validation_cost_usd,
            "inference_cost_usd": self.inference_cost_usd,
            "total_cost_usd": self.total_cost_usd,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentCost":
        return cls(
            cost_id=d.get("cost_id", str(uuid.uuid4())),
            entity_id=d.get("entity_id", ""),
            entity_type=d.get("entity_type", "experiment"),
            project=d.get("project", ""),
            branch=d.get("branch", ""),
            researcher=d.get("researcher", "system"),
            recorded_at=d.get("recorded_at", datetime.now(timezone.utc).isoformat()),
            gpu_hours=d.get("gpu_hours", 0.0),
            cpu_hours=d.get("cpu_hours", 0.0),
            storage_gb_months=d.get("storage_gb_months", 0.0),
            validation_gpu_hours=d.get("validation_gpu_hours", 0.0),
            inference_hours=d.get("inference_hours", 0.0),
            training_cost_usd=d.get("training_cost_usd", 0.0),
            cpu_cost_usd=d.get("cpu_cost_usd", 0.0),
            storage_cost_usd=d.get("storage_cost_usd", 0.0),
            validation_cost_usd=d.get("validation_cost_usd", 0.0),
            inference_cost_usd=d.get("inference_cost_usd", 0.0),
        )


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Accumulates and reports research costs across all experiments."""

    # Default rates
    DEFAULT_GPU_RATE = 2.00  # USD / GPU-hour
    DEFAULT_CPU_RATE = 0.05  # USD / CPU-hour
    DEFAULT_STORAGE_RATE = 0.023  # USD / GB-month
    DEFAULT_INFERENCE_RATE = 0.50  # USD / GPU-hour

    def __init__(
        self,
        gpu_rate_per_hour: float = DEFAULT_GPU_RATE,
        cpu_rate_per_hour: float = DEFAULT_CPU_RATE,
        storage_rate_per_gb_month: float = DEFAULT_STORAGE_RATE,
        inference_rate_per_hour: float = DEFAULT_INFERENCE_RATE,
    ) -> None:
        self.gpu_rate = gpu_rate_per_hour
        self.cpu_rate = cpu_rate_per_hour
        self.storage_rate = storage_rate_per_gb_month
        self.inference_rate = inference_rate_per_hour
        self._costs: List[ExperimentCost] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        entity_id: str,
        *,
        entity_type: str = "experiment",
        project: str = "",
        branch: str = "",
        researcher: str = "system",
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
        storage_gb_months: float = 0.0,
        validation_gpu_hours: float = 0.0,
        inference_hours: float = 0.0,
    ) -> ExperimentCost:
        """Record resource consumption and compute USD cost estimate."""
        training_cost = gpu_hours * self.gpu_rate
        cpu_cost = cpu_hours * self.cpu_rate
        storage_cost = storage_gb_months * self.storage_rate
        val_cost = validation_gpu_hours * self.gpu_rate
        inf_cost = inference_hours * self.inference_rate

        cost = ExperimentCost(
            entity_id=entity_id,
            entity_type=entity_type,
            project=project,
            branch=branch,
            researcher=researcher,
            gpu_hours=gpu_hours,
            cpu_hours=cpu_hours,
            storage_gb_months=storage_gb_months,
            validation_gpu_hours=validation_gpu_hours,
            inference_hours=inference_hours,
            training_cost_usd=round(training_cost, 6),
            cpu_cost_usd=round(cpu_cost, 6),
            storage_cost_usd=round(storage_cost, 6),
            validation_cost_usd=round(val_cost, 6),
            inference_cost_usd=round(inf_cost, 6),
        )
        self._costs.append(cost)
        logger.debug(f"[CostTracker] {entity_id}: ${cost.total_cost_usd:.4f}")
        return cost

    def estimate_from_duration(
        self,
        entity_id: str,
        training_seconds: float,
        *,
        gpu_count: int = 1,
        cpu_cores: int = 8,
        storage_gb: float = 5.0,
        storage_months: float = 1.0,
        project: str = "",
        branch: str = "",
        researcher: str = "system",
    ) -> ExperimentCost:
        """Estimate cost from a training duration in seconds."""
        gpu_hours = (training_seconds / 3600.0) * gpu_count
        cpu_hours = (training_seconds / 3600.0) * (cpu_cores / 8.0)
        storage_gb_months = storage_gb * storage_months
        return self.record(
            entity_id,
            entity_type="experiment",
            project=project,
            branch=branch,
            researcher=researcher,
            gpu_hours=gpu_hours,
            cpu_hours=cpu_hours,
            storage_gb_months=storage_gb_months,
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get(self, entity_id: str) -> Optional[ExperimentCost]:
        for c in reversed(self._costs):
            if c.entity_id == entity_id:
                return c
        return None

    def all_costs(self) -> List[ExperimentCost]:
        return list(self._costs)

    def by_project(self, project: str) -> List[ExperimentCost]:
        return [c for c in self._costs if c.project == project]

    def by_branch(self, branch: str) -> List[ExperimentCost]:
        return [c for c in self._costs if c.branch == branch]

    def by_researcher(self, researcher: str) -> List[ExperimentCost]:
        return [c for c in self._costs if c.researcher == researcher]

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def total_cost_usd(self) -> float:
        return sum(c.total_cost_usd for c in self._costs)

    def total_gpu_hours(self) -> float:
        return sum(c.gpu_hours for c in self._costs)

    def total_cpu_hours(self) -> float:
        return sum(c.cpu_hours for c in self._costs)

    def cost_by_project(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for c in self._costs:
            key = c.project or "unassigned"
            result[key] = result.get(key, 0.0) + c.total_cost_usd
        return {k: round(v, 4) for k, v in result.items()}

    def cost_by_researcher(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for c in self._costs:
            key = c.researcher or "unknown"
            result[key] = result.get(key, 0.0) + c.total_cost_usd
        return {k: round(v, 4) for k, v in result.items()}

    def cost_by_category(self) -> Dict[str, float]:
        return {
            "training": round(sum(c.training_cost_usd for c in self._costs), 4),
            "cpu": round(sum(c.cpu_cost_usd for c in self._costs), 4),
            "storage": round(sum(c.storage_cost_usd for c in self._costs), 4),
            "validation": round(sum(c.validation_cost_usd for c in self._costs), 4),
            "inference": round(sum(c.inference_cost_usd for c in self._costs), 4),
        }

    def most_expensive(self, n: int = 5) -> List[ExperimentCost]:
        return sorted(self._costs, key=lambda c: c.total_cost_usd, reverse=True)[:n]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def cost_report(self) -> Dict[str, Any]:
        return {
            "total_experiments": len(self._costs),
            "total_cost_usd": round(self.total_cost_usd(), 4),
            "total_gpu_hours": round(self.total_gpu_hours(), 4),
            "total_cpu_hours": round(self.total_cpu_hours(), 4),
            "cost_by_category": self.cost_by_category(),
            "cost_by_project": self.cost_by_project(),
            "cost_by_researcher": self.cost_by_researcher(),
            "most_expensive_experiments": [c.to_dict() for c in self.most_expensive(5)],
            "rates": {
                "gpu_rate_per_hour_usd": self.gpu_rate,
                "cpu_rate_per_hour_usd": self.cpu_rate,
                "storage_rate_per_gb_month_usd": self.storage_rate,
                "inference_rate_per_hour_usd": self.inference_rate,
            },
        }

    def load_cost(self, cost: ExperimentCost) -> None:
        """Load a pre-built cost record (used by persistence layer)."""
        self._costs.append(cost)
