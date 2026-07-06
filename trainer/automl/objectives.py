"""Phase 16 — AutoML Objective Functions.

Defines all objective functions used to score and rank experiments.

Supports:
  - Single-objective optimization
  - Multi-objective optimization (Pareto front)
  - Composite scoring

Standard objectives (all return higher-is-better scores after normalization):
  profit_factor, sharpe_ratio, sortino_ratio, max_drawdown (inverted),
  expected_calibration_error (inverted), validation_score, latency (inverted),
  memory_usage (inverted), composite_research_score, win_rate, accuracy, f1.

Usage::

    objectives = [
        Objective("sharpe_ratio", direction="maximize", weight=0.4),
        Objective("max_drawdown", direction="minimize", weight=0.3),
        Objective("profit_factor", direction="maximize", weight=0.3),
    ]
    scorer = ObjectiveScorer(objectives)
    score = scorer.composite_score(metrics)
    pareto = scorer.pareto_front([metrics_a, metrics_b, metrics_c])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Objective definition
# ---------------------------------------------------------------------------


@dataclass
class Objective:
    """A single optimization objective.

    Attributes:
        name: Metric key in the results dict.
        direction: "maximize" or "minimize".
        weight: Weight for composite scoring (normalized internally).
        clip_min: Optional lower clip before normalization.
        clip_max: Optional upper clip before normalization.
        label: Human-readable label (defaults to name).
    """

    name: str
    direction: str = "maximize"
    weight: float = 1.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {self.direction!r}")
        if not self.label:
            self.label = self.name

    def is_better(self, a: float, b: float) -> bool:
        """Return True if value a is better than value b."""
        if self.direction == "maximize":
            return a > b
        return a < b

    def normalize(self, value: float, low: float, high: float) -> float:
        """Normalize value to [0, 1] (1 = best direction)."""
        if high == low:
            return 0.5
        v = value
        if self.clip_min is not None:
            v = max(v, self.clip_min)
        if self.clip_max is not None:
            v = min(v, self.clip_max)
        norm = (v - low) / (high - low)
        if self.direction == "minimize":
            norm = 1.0 - norm
        return max(0.0, min(1.0, norm))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "weight": self.weight,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Objective":
        return Objective(
            name=d["name"],
            direction=d.get("direction", "maximize"),
            weight=d.get("weight", 1.0),
            clip_min=d.get("clip_min"),
            clip_max=d.get("clip_max"),
            label=d.get("label", ""),
        )


# ---------------------------------------------------------------------------
# Predefined objectives
# ---------------------------------------------------------------------------


STANDARD_OBJECTIVES: Dict[str, Objective] = {
    "profit_factor": Objective("profit_factor", "maximize", clip_min=0.0, clip_max=5.0),
    "sharpe_ratio": Objective("sharpe_ratio", "maximize", clip_min=-3.0, clip_max=5.0),
    "sortino_ratio": Objective("sortino_ratio", "maximize", clip_min=-3.0, clip_max=5.0),
    "max_drawdown": Objective("max_drawdown", "minimize", clip_min=0.0, clip_max=1.0),
    "expected_calibration_error": Objective(
        "expected_calibration_error", "minimize", clip_min=0.0, clip_max=1.0
    ),
    "validation_score": Objective("validation_score", "maximize", clip_min=0.0, clip_max=1.0),
    "latency_ms": Objective("latency_ms", "minimize", clip_min=0.0),
    "memory_mb": Objective("memory_mb", "minimize", clip_min=0.0),
    "win_rate": Objective("win_rate", "maximize", clip_min=0.0, clip_max=1.0),
    "accuracy": Objective("accuracy", "maximize", clip_min=0.0, clip_max=1.0),
    "f1_score": Objective("f1_score", "maximize", clip_min=0.0, clip_max=1.0),
    "composite_research_score": Objective(
        "composite_research_score", "maximize", clip_min=0.0, clip_max=1.0
    ),
    "calmar_ratio": Objective("calmar_ratio", "maximize", clip_min=-5.0, clip_max=10.0),
    "omega_ratio": Objective("omega_ratio", "maximize", clip_min=0.0, clip_max=5.0),
}


def get_objective(name: str) -> Objective:
    """Return a standard objective by name."""
    if name not in STANDARD_OBJECTIVES:
        raise KeyError(
            f"Unknown standard objective: {name!r}. Available: {list(STANDARD_OBJECTIVES)}"
        )
    return STANDARD_OBJECTIVES[name]


# ---------------------------------------------------------------------------
# Objective scorer
# ---------------------------------------------------------------------------


class ObjectiveScorer:
    """Computes composite and per-objective scores for a set of metrics.

    Args:
        objectives: List of Objective definitions. Weights are normalized.
    """

    def __init__(self, objectives: List[Objective]) -> None:
        if not objectives:
            raise ValueError("At least one objective is required.")
        self.objectives = objectives
        total_weight = sum(o.weight for o in objectives)
        self._normalized_weights = [o.weight / total_weight for o in objectives]

    def extract(self, metrics: Dict[str, Any], obj: Objective) -> Optional[float]:
        """Extract objective metric from a results dict (supports nested keys)."""
        val = metrics.get(obj.name)
        if val is None:
            for v in metrics.values():
                if isinstance(v, dict):
                    val = v.get(obj.name)
                    if val is not None:
                        break
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def score_single(
        self, metrics: Dict[str, Any], obj: Objective, low: float = 0.0, high: float = 1.0
    ) -> float:
        """Return normalized [0, 1] score for a single objective."""
        val = self.extract(metrics, obj)
        if val is None:
            return 0.0
        return obj.normalize(val, low, high)

    def composite_score(
        self, metrics: Dict[str, Any], reference_sets: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Return weighted composite score in [0, 1].

        If reference_sets is provided, normalization bounds are derived from
        the population (including current metrics). Otherwise raw values are
        used with each objective's clip_min/clip_max as bounds.
        """
        total = 0.0
        for obj, w in zip(self.objectives, self._normalized_weights):
            val = self.extract(metrics, obj)
            if val is None:
                continue
            if reference_sets:
                all_vals = [self.extract(m, obj) for m in reference_sets + [metrics]]
                valid = [v for v in all_vals if v is not None]
                low = min(valid)
                high = max(valid)
            else:
                low = obj.clip_min if obj.clip_min is not None else val - 1e-9
                high = obj.clip_max if obj.clip_max is not None else val + 1e-9
            norm = obj.normalize(val, low, high)
            total += w * norm
        return round(total, 6)

    def per_objective_scores(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Return dict of {objective_name: raw_value} for all objectives."""
        return {obj.name: self.extract(metrics, obj) or 0.0 for obj in self.objectives}

    def dominates(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Return True if solution A dominates B in Pareto sense."""
        a_better_any = False
        for obj in self.objectives:
            va = self.extract(a, obj) or 0.0
            vb = self.extract(b, obj) or 0.0
            if obj.is_better(vb, va):
                return False
            if obj.is_better(va, vb):
                a_better_any = True
        return a_better_any

    def pareto_front(self, solutions: List[Dict[str, Any]]) -> List[int]:
        """Return indices of Pareto-optimal solutions in the input list."""
        n = len(solutions)
        dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(solutions[j], solutions[i]):
                    dominated[i] = True
                    break
        return [i for i in range(n) if not dominated[i]]

    def rank(
        self,
        metrics_list: List[Dict[str, Any]],
        reference_sets: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[int, float]]:
        """Return sorted list of (original_index, composite_score) desc."""
        ref = reference_sets or metrics_list
        scored = [(i, self.composite_score(m, ref)) for i, m in enumerate(metrics_list)]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {"objectives": [o.to_dict() for o in self.objectives]}


# ---------------------------------------------------------------------------
# Composite research score
# ---------------------------------------------------------------------------


def compute_composite_research_score(metrics: Dict[str, Any]) -> float:
    """Compute the Signalium composite research score from a metrics dict.

    Weights:
      - validation_score:  25%
      - sharpe_ratio:      20%
      - profit_factor:     20%
      - max_drawdown:      20% (inverted)
      - calibration score: 15% (inverted ECE)

    Returns a value in [0, 1].
    """

    def safe(key: str, default: float = 0.0) -> float:
        val = metrics.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    val_score = min(max(safe("validation_score", 0.5), 0.0), 1.0)
    sharpe = min(max(safe("sharpe_ratio", 0.0) / 5.0, 0.0), 1.0)
    pf = min(max((safe("profit_factor", 1.0) - 1.0) / 4.0, 0.0), 1.0)
    drawdown = 1.0 - min(max(safe("max_drawdown", 0.0), 0.0), 1.0)
    ece = 1.0 - min(max(safe("expected_calibration_error", 0.1), 0.0), 1.0)

    composite = 0.25 * val_score + 0.20 * sharpe + 0.20 * pf + 0.20 * drawdown + 0.15 * ece
    return round(composite, 6)


def build_research_scorer() -> ObjectiveScorer:
    """Return the default multi-objective scorer for research experiments."""
    return ObjectiveScorer(
        [
            Objective("composite_research_score", "maximize", weight=0.25),
            Objective("sharpe_ratio", "maximize", weight=0.20, clip_min=-2.0, clip_max=5.0),
            Objective("profit_factor", "maximize", weight=0.20, clip_min=0.5, clip_max=5.0),
            Objective("max_drawdown", "minimize", weight=0.20, clip_min=0.0, clip_max=0.5),
            Objective(
                "expected_calibration_error", "minimize", weight=0.15, clip_min=0.0, clip_max=0.5
            ),
        ]
    )
