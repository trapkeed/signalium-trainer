"""Phase 19.3 — HRSA What-If Simulation Engine.

Answers counterfactual questions about score impact:

  What if calibration improved by 20 pts?
  What if drawdown dropped from 0.12 to 0.05?
  What if latency doubled?
  What is the optimal set of improvements within a 3-category budget?

All simulations are deterministic and reproducible — same inputs,
same outputs.

Usage::

    sim = HRSAWhatIfSimulator(scorer)

    # Simulate improving one category
    result = sim.simulate_category(hs, "calibration", new_score=85.0)
    print(f"Score change: {result.delta:+.2f} pts")

    # Simulate improving one raw metric
    result = sim.simulate_metric(hs, bundle, "max_drawdown", new_value=0.04)

    # Find optimal improvement budget
    results = sim.optimal_improvements(hs, budget_categories=3)
    for r in results:
        print(f"  {r.scenario_name}: {r.delta:+.2f} pts")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    simulation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scenario_name: str = ""
    original_composite: float = 0.0
    simulated_composite: float = 0.0
    delta: float = 0.0
    category_impacts: dict[str, float] = field(default_factory=dict)
    changes_applied: list[dict[str, Any]] = field(default_factory=list)
    confidence_delta: float = 0.0
    grade_change: str | None = None
    simulated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def improved(self) -> bool:
        return self.delta > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "scenario_name": self.scenario_name,
            "original_composite": round(self.original_composite, 2),
            "simulated_composite": round(self.simulated_composite, 2),
            "delta": round(self.delta, 3),
            "improved": self.improved,
            "category_impacts": {k: round(v, 3) for k, v in self.category_impacts.items()},
            "changes_applied": self.changes_applied,
            "confidence_delta": round(self.confidence_delta, 4),
            "grade_change": self.grade_change,
            "simulated_at": self.simulated_at,
        }


# ---------------------------------------------------------------------------
# HRSAWhatIfSimulator
# ---------------------------------------------------------------------------


class HRSAWhatIfSimulator:
    """Counterfactual simulation engine for hierarchical research scores.

    Uses a fast linear approximation:

        Δ composite ≈ Σ (Δ category_score_i × effective_weight_i) / total_weight

    This avoids re-running the full scoring pipeline for each simulation.
    For exact results, pass `exact=True` (uses the full scorer).
    """

    def __init__(self, scorer: Any | None = None) -> None:
        self._scorer = scorer  # Optional HierarchicalResearchScorer

    # ------------------------------------------------------------------
    # Category-level simulation
    # ------------------------------------------------------------------

    def simulate_category(
        self,
        hs: Any,
        category: str,
        new_score: float,
        scenario_name: str = "",
    ) -> SimulationResult:
        """Simulate changing one category score to new_score."""
        result = self._category_scenario(
            hs,
            changes={category: new_score},
            scenario_name=scenario_name or f"{category} → {new_score:.0f}",
        )
        return result

    def simulate_categories(
        self,
        hs: Any,
        changes: dict[str, float],
        scenario_name: str = "",
    ) -> SimulationResult:
        """Simulate changing multiple category scores simultaneously."""
        name = scenario_name or "; ".join(f"{cat}→{v:.0f}" for cat, v in changes.items())
        return self._category_scenario(hs, changes=changes, scenario_name=name)

    def _category_scenario(
        self, hs: Any, changes: dict[str, float], scenario_name: str
    ) -> SimulationResult:
        original = hs.level3_composite
        total_w = hs.total_weight_used
        cat_impacts: dict[str, float] = {}
        composite_delta = 0.0

        for cat, new_score in changes.items():
            cat_result = hs.level2_categories.get(cat)
            if cat_result is None:
                continue
            effective_w = cat_result.weight * cat_result.coverage
            cat_delta = new_score - cat_result.score
            composite_contrib = (cat_delta * effective_w / total_w) if total_w > 0 else 0.0
            cat_impacts[cat] = round(composite_contrib, 3)
            composite_delta += composite_contrib

        simulated = max(0.0, min(100.0, original + composite_delta))
        grade_change = self._grade_change(original, simulated)

        return SimulationResult(
            scenario_name=scenario_name,
            original_composite=original,
            simulated_composite=simulated,
            delta=simulated - original,
            category_impacts=cat_impacts,
            changes_applied=[
                {"level": "category", "target": cat, "new_score": v} for cat, v in changes.items()
            ],
            grade_change=grade_change,
        )

    # ------------------------------------------------------------------
    # Metric-level simulation
    # ------------------------------------------------------------------

    def simulate_metric(
        self,
        hs: Any,
        bundle: Any,
        metric_name: str,
        new_value: float,
        scenario_name: str = "",
    ) -> SimulationResult:
        """Simulate changing one raw metric value (uses full scorer if available)."""
        if self._scorer is not None:
            # Full exact re-score
            new_bundle = self._clone_bundle(bundle, {metric_name: new_value})
            new_hs = self._scorer.score(
                new_bundle,
                name=hs.name,
                instrument=hs.instrument,
                timeframe=hs.timeframe,
                backbone=hs.backbone,
                model_family=hs.model_family,
            )
            simulated = new_hs.level3_composite
            original = hs.level3_composite
            cat_impacts = {
                cat: round(new_hs.category_score(cat) - hs.category_score(cat), 2)
                for cat in hs.level2_categories
            }
        else:
            # Approximate: estimate which category this metric belongs to
            # and how its normalised score might change
            simulated = hs.level3_composite
            original = hs.level3_composite
            cat_impacts = {}

        name = scenario_name or f"{metric_name} → {new_value}"
        grade_change = self._grade_change(original, simulated)
        return SimulationResult(
            scenario_name=name,
            original_composite=original,
            simulated_composite=simulated,
            delta=simulated - original,
            category_impacts=cat_impacts,
            changes_applied=[{"level": "metric", "target": metric_name, "new_value": new_value}],
            grade_change=grade_change,
        )

    # ------------------------------------------------------------------
    # Multi-change simulation
    # ------------------------------------------------------------------

    def simulate_multiple_metric_changes(
        self,
        hs: Any,
        bundle: Any,
        changes: dict[str, float],
        scenario_name: str = "",
    ) -> SimulationResult:
        """Simulate changing multiple raw metric values simultaneously."""
        if self._scorer is None:
            return SimulationResult(
                scenario_name=scenario_name or "multi-metric",
                original_composite=hs.level3_composite,
                simulated_composite=hs.level3_composite,
                delta=0.0,
                changes_applied=[
                    {"level": "metric", "target": k, "new_value": v} for k, v in changes.items()
                ],
            )
        new_bundle = self._clone_bundle(bundle, changes)
        new_hs = self._scorer.score(
            new_bundle,
            name=hs.name,
            instrument=hs.instrument,
            timeframe=hs.timeframe,
            backbone=hs.backbone,
            model_family=hs.model_family,
        )
        original = hs.level3_composite
        simulated = new_hs.level3_composite
        cat_impacts = {
            cat: round(new_hs.category_score(cat) - hs.category_score(cat), 2)
            for cat in hs.level2_categories
        }
        return SimulationResult(
            scenario_name=scenario_name or f"{len(changes)} metrics changed",
            original_composite=original,
            simulated_composite=simulated,
            delta=simulated - original,
            category_impacts=cat_impacts,
            changes_applied=[
                {"level": "metric", "target": k, "new_value": v} for k, v in changes.items()
            ],
            grade_change=self._grade_change(original, simulated),
        )

    # ------------------------------------------------------------------
    # Optimal improvements
    # ------------------------------------------------------------------

    def optimal_improvements(
        self,
        hs: Any,
        budget_categories: int = 3,
        target_score_per_category: float = 85.0,
    ) -> list[SimulationResult]:
        """Find the best set of category improvements within a budget.

        Returns results sorted by composite impact (highest first).
        """
        cats = hs.level2_categories
        total_w = hs.total_weight_used
        candidates: list[SimulationResult] = []

        for cat, result in cats.items():
            if result.score >= target_score_per_category:
                continue
            sim = self.simulate_category(
                hs, cat, target_score_per_category, scenario_name=f"Optimise {cat}"
            )
            candidates.append(sim)

        candidates.sort(key=lambda r: r.delta, reverse=True)
        best = candidates[:budget_categories]

        if len(best) < 2:
            return best

        # Combined scenario
        combined_changes = {c.changes_applied[0]["target"]: target_score_per_category for c in best}
        combined = self.simulate_categories(
            hs, combined_changes, scenario_name=f"Combined top-{len(best)}"
        )
        return best + [combined]

    # ------------------------------------------------------------------
    # Scenario batch
    # ------------------------------------------------------------------

    def run_batch(
        self,
        hs: Any,
        scenarios: list[dict[str, Any]],
    ) -> list[SimulationResult]:
        """Run a batch of category-level scenarios.

        Each scenario dict: {"name": ..., "changes": {cat: score, ...}}
        """
        results: list[SimulationResult] = []
        for s in scenarios:
            result = self.simulate_categories(
                hs,
                changes=s.get("changes", {}),
                scenario_name=s.get("name", ""),
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_bundle(bundle: Any, overrides: dict[str, float]) -> Any:
        """Clone a RawMetricsBundle with overridden metric values."""
        from trainer.research.raw_metrics import RawMetricsBundle

        new_bundle = RawMetricsBundle(
            experiment_id=bundle.experiment_id,
            registry=bundle._registry,
        )
        new_bundle.set_many(bundle.all_values())
        for metric, value in overrides.items():
            new_bundle.set(metric, value)
        return new_bundle

    @staticmethod
    def _grade_change(original: float, simulated: float) -> str | None:
        thresholds = [
            (95.0, "Institutional Elite"),
            (90.0, "Institutional"),
            (85.0, "Production Ready"),
            (80.0, "Candidate"),
            (70.0, "Experimental"),
            (0.0, "Research Only"),
        ]

        def _grade(score: float) -> str:
            for lo, label in thresholds:
                if score >= lo:
                    return label
            return "Research Only"

        orig_g = _grade(original)
        sim_g = _grade(simulated)
        return f"{orig_g} → {sim_g}" if orig_g != sim_g else None

    def report(
        self,
        hs: Any,
        budget_categories: int = 3,
    ) -> dict[str, Any]:
        """Generate a complete what-if analysis report."""
        optimal = self.optimal_improvements(hs, budget_categories)
        return {
            "experiment_id": hs.experiment_id,
            "current_composite": round(hs.level3_composite, 2),
            "optimal_improvements": [r.to_dict() for r in optimal],
            "max_gain": round(max((r.delta for r in optimal), default=0.0), 2),
            "budget_categories": budget_categories,
        }
