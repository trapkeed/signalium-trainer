"""Phase 29 — Learning Loop Actuators.

The six modules below are OBSERVATION-ONLY: they analyse decision history
and emit read-only reports. Nothing in the system previously consumed those
reports, so their insights never reached any downstream process.

    - SelfEvaluationEngine  -> SelfAssessment
    - PatternDiscovery      -> PatternReport
    - CounterfactualEngine  -> list[CounterfactualResult]
    - LifecycleOptimizer    -> OptimisationReport
    - PortfolioOptimizer    -> OptimizationResult
    - OutcomeTracker        -> OutcomeSummary

This module is the ACTUATOR layer: it reads those reports and turns their
findings into concrete requests against the two systems that are allowed to
change the model's future behaviour:

    - ResearchLabAPI      (trainer.research_lab.research_lab_api)
    - RetrainingAdvisor   (trainer.research.retraining_advisor)

Every actuator method maps observations onto exactly one of six action
kinds:

    1. FEATURE_DISCOVERY      -> ResearchLabAPI.record_feature
    2. RETRAINING_RECOMMENDATION -> RetrainingAdvisor.evaluate (advisory only)
    3. DATASET_ENRICHMENT     -> ResearchLabAPI.create_dataset
    4. EXPERIMENT_PROPOSAL    -> ResearchLabAPI.create_experiment
    5. PARAMETER_SEARCH       -> ResearchLabAPI.create_search
    6. INVESTIGATION          -> ResearchLabAPI.record_failure

CRITICAL CONSTRAINTS (do not weaken these):
    - No actuator ever mutates trading logic, model weights, coefficients,
      or live thresholds directly. Every action produced here is a request
      or proposal that lands in an existing, human-gated workflow
      (ResearchLab admin review, deployment gates, retraining advisor
      acknowledgement, etc.).
    - The model is only ever allowed to improve via retraining. Retraining
      is always requested through RetrainingAdvisor.evaluate(), which is
      hard-pinned to advisory-only (auto_retrain=False, raises otherwise).
    - This layer never talks to the retraining advisor's internal automatic
      trigger machinery, and never calls anything that would deploy or
      promote a model — approval remains a separate, explicit, human step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

from trainer.lifecycle.lifecycle_optimizer import OptimisationFinding, OptimisationReport
from trainer.memory.decision_models import CounterfactualResult
from trainer.memory.outcome_tracker import OutcomeSummary
from trainer.memory.pattern_discovery import DiscoveredPattern, PatternReport
from trainer.memory.self_evaluation import SelfAssessment
from trainer.portfolio.optimizer import OptimizationResult
from trainer.research.retraining_advisor import RetrainingAdvisor, build_default_advisor
from trainer.research_lab.research_lab_api import ResearchLabAPI

# ── Action kinds ────────────────────────────────────────────────────────────


class ActionType:
    FEATURE_DISCOVERY = "FEATURE_DISCOVERY"
    RETRAINING_RECOMMENDATION = "RETRAINING_RECOMMENDATION"
    DATASET_ENRICHMENT = "DATASET_ENRICHMENT"
    EXPERIMENT_PROPOSAL = "EXPERIMENT_PROPOSAL"
    PARAMETER_SEARCH = "PARAMETER_SEARCH"
    INVESTIGATION = "INVESTIGATION"

    ALL = [
        FEATURE_DISCOVERY,
        RETRAINING_RECOMMENDATION,
        DATASET_ENRICHMENT,
        EXPERIMENT_PROPOSAL,
        PARAMETER_SEARCH,
        INVESTIGATION,
    ]


@dataclass
class ActuatorAction:
    """A single concrete action taken (or attempted) by the learning loop."""

    action_type: str
    source: str  # which observation module/report triggered this action
    success: bool
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "source": self.source,
            "success": self.success,
            "summary": self.summary,
            "detail": self.detail,
        }


@dataclass
class ActuatorCycleResult:
    """Aggregate result of running the actuator over one or more reports."""

    actions: list[ActuatorAction] = field(default_factory=list)

    def add(self, action: ActuatorAction) -> None:
        self.actions.append(action)

    def extend(self, actions: list[ActuatorAction]) -> None:
        self.actions.extend(actions)

    @property
    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {t: 0 for t in ActionType.ALL}
        for a in self.actions:
            counts[a.action_type] = counts.get(a.action_type, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_actions": len(self.actions),
            "by_type": self.by_type,
            "actions": [a.to_dict() for a in self.actions],
        }


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len] or "unnamed"


# ── Learning Loop Actuator ───────────────────────────────────────────────────


class LearningLoopActuator:
    """Turns observation reports from the six learning-loop modules into
    concrete requests against ResearchLabAPI and RetrainingAdvisor.

    Every ``act_on_*`` method is independent and idempotent-ish (repeated
    identical inputs will create new proposal/request records — the
    downstream systems, not this class, are responsible for deduplication
    and cooldown; RetrainingAdvisor rules already enforce per-trigger
    cooldowns).
    """

    # -- Tunable thresholds (heuristics — documented, not magic) -----------
    QUALITY_TARGET = 0.60
    WEAK_WIN_RATE = 0.45
    STRONG_PATTERN_MIN_SAMPLE = 10
    ENRICHMENT_SAMPLE_TARGET = 20
    SL_INVESTIGATION_RATIO = 0.40
    COUNTERFACTUAL_MIN_SAMPLE = 5
    COUNTERFACTUAL_SYSTEMATIC_RATIO = 0.60
    LIFECYCLE_PARAM_SEARCH_CONFIDENCE = 0.60
    LIFECYCLE_EXPERIMENT_MIN_TRADES = 20
    PORTFOLIO_IMPROVEMENT_RATIO = 0.10
    PORTFOLIO_LOW_DIVERSIFICATION = 0.30

    # Counterfactual alternative_action -> candidate parameter search space.
    # These describe SEARCH SPACES for the retraining/experiment pipeline —
    # never a direct write to a live trading parameter.
    _COUNTERFACTUAL_PARAM_SPACE = {
        "TIGHTER_SL": {"sl_atr_multiplier": [0.5, 1.0]},
        "LARGER_TP": {"tp_atr_multiplier": [1.0, 2.0]},
        "EARLY_TRAIL": {"trailing_activation_pct": [0.2, 0.6]},
        "DELAYED_ENTRY": {"entry_delay_atr": [0.5, 2.0]},
    }

    _LIFECYCLE_PARAM_SPACE = {
        "BREAKEVEN": {"breakeven_trigger_pct": [0.3, 0.7]},
        "TRAILING": {
            "trailing_activation_pct": [0.3, 0.7],
            "trailing_distance_atr": [0.5, 2.0],
        },
        "SCALING": {"scale_in_confidence_threshold": [0.70, 0.90]},
        "TIMING": {"entry_timeout_minutes": [15, 120]},
    }

    def __init__(
        self,
        research_lab: ResearchLabAPI | None = None,
        retraining_advisor: RetrainingAdvisor | None = None,
        model_id: str = "qde-default",
        qde_version: str = "unversioned",
    ) -> None:
        self._lab = research_lab or ResearchLabAPI()
        self._advisor = retraining_advisor or build_default_advisor()
        self.model_id = model_id
        self.qde_version = qde_version
        self._history: list[ActuatorAction] = []

    # -- Introspection -------------------------------------------------------

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._history[-limit:]]

    @property
    def retraining_advisor(self) -> RetrainingAdvisor:
        return self._advisor

    @property
    def research_lab(self) -> ResearchLabAPI:
        return self._lab

    def _record(self, action: ActuatorAction) -> ActuatorAction:
        self._history.append(action)
        return action

    def _request_retraining(
        self, source: str, context: dict[str, Any]
    ) -> list[ActuatorAction]:
        """Route a retraining signal through the advisory-only advisor.

        Never triggers retraining directly — only produces
        RetrainingRecommendation objects that require explicit human action.
        """
        if not context:
            return []
        recs = self._advisor.evaluate(self.model_id, context)
        actions = []
        for rec in recs:
            actions.append(
                self._record(
                    ActuatorAction(
                        action_type=ActionType.RETRAINING_RECOMMENDATION,
                        source=source,
                        success=True,
                        summary=f"Retraining recommended ({rec.trigger}): {rec.reason}",
                        detail=rec.to_dict(),
                    )
                )
            )
        return actions

    def _apply(self, action_type: str, source: str, response, summary: str) -> ActuatorAction:
        action = ActuatorAction(
            action_type=action_type,
            source=source,
            success=response.success,
            summary=summary if response.success else f"{summary} (FAILED: {response.error})",
            detail=response.to_dict(),
        )
        return self._record(action)

    # -- 1. SelfEvaluationEngine → SelfAssessment ----------------------------

    def act_on_self_assessment(self, assessment: SelfAssessment) -> list[ActuatorAction]:
        """Turn a self-assessment into retraining signals and investigations.

        - Low overall quality / win rate  -> retraining recommendation
          (performance_degradation).
        - Overconfidence weakness detected -> retraining recommendation
          (confidence_degradation).
        - Recurring failure modes          -> ResearchLab investigations.
        """
        actions: list[ActuatorAction] = []
        source = f"SelfEvaluationEngine[{assessment.period}]"

        context: dict[str, Any] = {}
        if assessment.overall_quality_score < self.QUALITY_TARGET:
            context["performance_degradation"] = round(
                0.05 + (self.QUALITY_TARGET - assessment.overall_quality_score), 4
            )
        if assessment.overall_win_rate < self.WEAK_WIN_RATE:
            existing = context.get("performance_degradation", 0.0)
            context["performance_degradation"] = max(
                existing, round(0.05 + (self.WEAK_WIN_RATE - assessment.overall_win_rate), 4)
            )
        if any("overconfidence" in w.lower() for w in assessment.weaknesses):
            context["confidence_degradation"] = 0.15

        actions.extend(self._request_retraining(source, context))

        placeholder = "No recurring failure modes identified"
        for fm in assessment.failure_modes:
            if fm == placeholder:
                continue
            category = (
                "MARKET_CONDITION_FAILURE" if "missed opportunit" in fm.lower() else
                "REGIME_FAILURE" if "unexpected market event" in fm.lower() else
                "EXECUTION_FAILURE"
            )
            severity = "HIGH" if assessment.grade in ("POOR", "REJECT") else "MEDIUM"
            resp = self._lab.record_failure(
                qde_version=self.qde_version,
                category=category,
                severity=severity,
                description=fm,
                notes=f"Raised by SelfEvaluationEngine period={assessment.period}",
            )
            actions.append(
                self._apply(
                    ActionType.INVESTIGATION,
                    source,
                    resp,
                    f"Investigation opened for failure mode: {fm}",
                )
            )

        return actions

    # -- 2. PatternDiscovery → PatternReport ---------------------------------

    def act_on_pattern_report(self, report: PatternReport) -> list[ActuatorAction]:
        """Turn discovered patterns into feature requests, dataset enrichment
        requests, and market-evolution retraining signals.

        - STRONG positive patterns  -> feature discovery request (the
          relationship is durable enough to encode as an engineered feature).
        - STRONG negative patterns in a REGIME category -> regime-shift
          retraining signal.
        - WEAK patterns with a small sample -> dataset enrichment request
          (more data needed before the pattern can be trusted either way).
        """
        actions: list[ActuatorAction] = []
        source = "PatternDiscovery"
        context: dict[str, Any] = {}

        for pattern in report.patterns:
            if pattern.strength == "STRONG" and pattern.direction == "POSITIVE":
                importance = round(min(1.0, abs(pattern.win_rate - 0.5) * 2), 4)
                resp = self._lab.record_feature(
                    name=f"pattern_{_slug(pattern.category)}_{_slug(pattern.description)}",
                    source="pattern_discovery",
                    importance_score=importance,
                    method="rule_mined",
                    notes=pattern.evidence,
                )
                actions.append(
                    self._apply(
                        ActionType.FEATURE_DISCOVERY,
                        source,
                        resp,
                        f"Feature discovery request from pattern {pattern.pattern_id}: "
                        f"{pattern.description}",
                    )
                )
            elif pattern.strength == "STRONG" and pattern.direction == "NEGATIVE":
                if pattern.category == "REGIME":
                    context["regime_shift_detected"] = True
                else:
                    context["market_evolution_detected"] = True
            elif pattern.strength == "WEAK" and pattern.sample_size < self.ENRICHMENT_SAMPLE_TARGET:
                resp = self._lab.create_dataset(
                    name=f"enrichment-{_slug(pattern.category)}-{_slug(pattern.description)}",
                    version="proposed",
                    source="learning_loop_actuator",
                    instruments=[],
                    start_date="",
                    end_date="",
                    row_count=0,
                    tags=["enrichment_request", pattern.category.lower()],
                    notes=(
                        f"Pattern {pattern.pattern_id} is WEAK with only "
                        f"{pattern.sample_size} samples ({pattern.evidence}); "
                        "more data requested before drawing conclusions."
                    ),
                )
                actions.append(
                    self._apply(
                        ActionType.DATASET_ENRICHMENT,
                        source,
                        resp,
                        f"Dataset enrichment requested for weak pattern {pattern.pattern_id}: "
                        f"{pattern.description}",
                    )
                )

        actions.extend(self._request_retraining(source, context))
        return actions

    # -- 3. CounterfactualEngine → list[CounterfactualResult] ----------------

    def act_on_counterfactual_batch(
        self, results_by_decision: list[list[CounterfactualResult]]
    ) -> list[ActuatorAction]:
        """Aggregate counterfactuals across many decisions and propose
        parameter searches for systematic biases.

        A single decision's counterfactuals are estimates, not evidence.
        Only when an alternative_action (e.g. TIGHTER_SL) is systematically
        better than what actually happened, across enough decisions, do we
        propose a parameter search — never a direct parameter change.
        """
        actions: list[ActuatorAction] = []
        source = "CounterfactualEngine"

        by_action: dict[str, list[CounterfactualResult]] = {}
        for results in results_by_decision:
            for r in results:
                by_action.setdefault(r.alternative_action, []).append(r)

        for alt_action, results in by_action.items():
            n = len(results)
            if n < self.COUNTERFACTUAL_MIN_SAMPLE:
                continue
            better_ratio = sum(1 for r in results if r.better_than_actual) / n
            avg_delta = sum(r.delta_pnl for r in results) / n
            if better_ratio < self.COUNTERFACTUAL_SYSTEMATIC_RATIO:
                continue

            param_space = self._COUNTERFACTUAL_PARAM_SPACE.get(alt_action)
            if param_space:
                resp = self._lab.create_search(
                    name=f"counterfactual-{_slug(alt_action)}-search",
                    qde_version=self.qde_version,
                    parameter_space=param_space,
                    strategy="BAYESIAN",
                    max_trials=30,
                    notes=(
                        f"{alt_action} was better than actual in {better_ratio:.0%} of "
                        f"{n} decisions (avg delta_pnl={avg_delta:.5f})."
                    ),
                )
                actions.append(
                    self._apply(
                        ActionType.PARAMETER_SEARCH,
                        source,
                        resp,
                        f"Parameter search proposed for systematic {alt_action} edge "
                        f"({better_ratio:.0%} better across {n} decisions)",
                    )
                )
            elif alt_action in ("WAIT", "NO_TRADE"):
                # The model is systematically entering trades it shouldn't —
                # advisory retraining signal, not a direct entry-rule edit.
                actions.extend(
                    self._request_retraining(
                        source,
                        {"confidence_degradation": round(better_ratio, 4)},
                    )
                )

        return actions

    # -- 4. LifecycleOptimizer → OptimisationReport --------------------------

    def act_on_lifecycle_report(self, report: OptimisationReport) -> list[ActuatorAction]:
        """Turn lifecycle findings into parameter searches, investigations,
        and (when the case is strong enough) an experiment proposal.
        """
        actions: list[ActuatorAction] = []
        source = "LifecycleOptimizer"

        strong_positive: list[OptimisationFinding] = []
        for finding in report.findings:
            if finding.category == "RISK":
                severity = "HIGH" if finding.confidence >= 0.75 else "MEDIUM"
                resp = self._lab.record_failure(
                    qde_version=self.qde_version,
                    category="MARKET_CONDITION_FAILURE",
                    severity=severity,
                    description=finding.recommendation,
                    notes=finding.evidence,
                )
                actions.append(
                    self._apply(
                        ActionType.INVESTIGATION,
                        source,
                        resp,
                        f"Investigation opened for lifecycle risk finding {finding.finding_id}",
                    )
                )
                continue

            if (
                finding.expected_impact == "POSITIVE"
                and finding.confidence >= self.LIFECYCLE_PARAM_SEARCH_CONFIDENCE
            ):
                strong_positive.append(finding)
                param_space = self._LIFECYCLE_PARAM_SPACE.get(finding.category)
                if param_space:
                    resp = self._lab.create_search(
                        name=f"lifecycle-{_slug(finding.category)}-search",
                        qde_version=self.qde_version,
                        parameter_space=param_space,
                        strategy="BAYESIAN",
                        max_trials=25,
                        notes=f"{finding.recommendation} ({finding.evidence})",
                    )
                    actions.append(
                        self._apply(
                            ActionType.PARAMETER_SEARCH,
                            source,
                            resp,
                            f"Parameter search proposed for lifecycle finding "
                            f"{finding.finding_id}: {finding.recommendation}",
                        )
                    )

        if (
            report.total_analysed >= self.LIFECYCLE_EXPERIMENT_MIN_TRADES
            and any(f.priority == 1 for f in strong_positive)
        ):
            resp = self._lab.create_experiment(
                name=f"lifecycle-candidate-{self.qde_version}",
                qde_version=self.qde_version,
                training_dataset="latest",
                researcher="learning_loop_actuator",
                tags=["lifecycle_optimizer"],
                notes=report.summary,
            )
            actions.append(
                self._apply(
                    ActionType.EXPERIMENT_PROPOSAL,
                    source,
                    resp,
                    f"Experiment proposed to validate bundled lifecycle changes: "
                    f"{report.top_priority}",
                )
            )

        return actions

    # -- 5. PortfolioOptimizer → OptimizationResult --------------------------

    def act_on_portfolio_result(
        self,
        result: OptimizationResult,
        baseline_objective_value: float | None = None,
    ) -> list[ActuatorAction]:
        """Turn a portfolio optimization result into an experiment proposal
        (if it beats the current baseline), an investigation (if it failed
        to converge), or a dataset enrichment request (if diversification is
        structurally weak).
        """
        actions: list[ActuatorAction] = []
        source = f"PortfolioOptimizer[{result.objective}]"

        if not result.converged:
            resp = self._lab.record_failure(
                qde_version=self.qde_version,
                category="EXECUTION_FAILURE",
                severity="LOW",
                description=(
                    f"Portfolio optimizer failed to converge for objective "
                    f"'{result.objective}' after {result.iterations} iterations"
                ),
            )
            actions.append(
                self._apply(
                    ActionType.INVESTIGATION,
                    source,
                    resp,
                    "Investigation opened for non-converged portfolio optimization",
                )
            )
            return actions

        if (
            baseline_objective_value is not None
            and baseline_objective_value > 0
            and result.objective_value
            > baseline_objective_value * (1 + self.PORTFOLIO_IMPROVEMENT_RATIO)
        ):
            improvement = result.objective_value / baseline_objective_value - 1
            resp = self._lab.create_experiment(
                name=f"portfolio-reweight-{_slug(result.objective)}",
                qde_version=self.qde_version,
                training_dataset="portfolio_optimizer",
                researcher="learning_loop_actuator",
                hyperparameters={
                    "objective": result.objective,
                    "optimal_weights": result.optimal_weights,
                },
                notes=(
                    f"Portfolio optimizer found a {improvement:.1%} improvement in "
                    f"{result.objective} over the current baseline."
                ),
            )
            actions.append(
                self._apply(
                    ActionType.EXPERIMENT_PROPOSAL,
                    source,
                    resp,
                    f"Experiment proposed for improved portfolio weights "
                    f"({improvement:.1%} better {result.objective})",
                )
            )

        metrics = result.optimal_metrics
        if (
            getattr(metrics, "diversification_score", 1.0) < self.PORTFOLIO_LOW_DIVERSIFICATION
        ):
            resp = self._lab.create_dataset(
                name=f"enrichment-diversification-{_slug(result.objective)}",
                version="proposed",
                source="learning_loop_actuator",
                instruments=list(result.optimal_weights.keys()),
                start_date="",
                end_date="",
                row_count=0,
                tags=["enrichment_request", "diversification_gap"],
                notes=(
                    f"Diversification score {metrics.diversification_score:.2f} is low — "
                    "requesting data on additional instruments to widen the opportunity set."
                ),
            )
            actions.append(
                self._apply(
                    ActionType.DATASET_ENRICHMENT,
                    source,
                    resp,
                    "Dataset enrichment requested to address low diversification",
                )
            )

        return actions

    # -- 6. OutcomeTracker → OutcomeSummary -----------------------------------

    def act_on_outcome_summary(self, summary: OutcomeSummary) -> list[ActuatorAction]:
        """Turn an outcome summary into retraining signals, investigations,
        and dataset enrichment requests for under-sampled segments.
        """
        actions: list[ActuatorAction] = []
        source = "OutcomeTracker"

        if summary.with_outcome == 0:
            return actions

        context: dict[str, Any] = {}
        if summary.win_rate < self.WEAK_WIN_RATE:
            context["performance_degradation"] = round(
                0.05 + (self.WEAK_WIN_RATE - summary.win_rate), 4
            )
        actions.extend(self._request_retraining(source, context))

        sl_ratio = summary.hit_sl_count / summary.with_outcome
        if sl_ratio > self.SL_INVESTIGATION_RATIO:
            severity = "HIGH" if sl_ratio > 0.60 else "MEDIUM"
            resp = self._lab.record_failure(
                qde_version=self.qde_version,
                category="UNEXPECTED_DRAWDOWN",
                severity=severity,
                description=f"Stop-loss hit rate of {sl_ratio:.0%} exceeds acceptable threshold",
                loss_pct=None,
                notes=f"{summary.hit_sl_count}/{summary.with_outcome} decisions hit SL",
            )
            actions.append(
                self._apply(
                    ActionType.INVESTIGATION,
                    source,
                    resp,
                    f"Investigation opened for elevated SL hit rate ({sl_ratio:.0%})",
                )
            )

        for scope_name, breakdown in (
            ("instrument", summary.by_instrument),
            ("session", summary.by_session),
            ("regime", summary.by_regime),
        ):
            for key, stats in breakdown.items():
                if stats.get("count", 0) < self.ENRICHMENT_SAMPLE_TARGET:
                    resp = self._lab.create_dataset(
                        name=f"enrichment-{scope_name}-{_slug(str(key))}",
                        version="proposed",
                        source="learning_loop_actuator",
                        instruments=[key] if scope_name == "instrument" else [],
                        start_date="",
                        end_date="",
                        row_count=0,
                        tags=["enrichment_request", scope_name],
                        notes=(
                            f"Only {stats.get('count', 0)} decisions recorded for "
                            f"{scope_name}={key}; requesting more data."
                        ),
                    )
                    actions.append(
                        self._apply(
                            ActionType.DATASET_ENRICHMENT,
                            source,
                            resp,
                            f"Dataset enrichment requested for under-sampled {scope_name} "
                            f"'{key}' ({stats.get('count', 0)} decisions)",
                        )
                    )

        return actions

    # -- Orchestration --------------------------------------------------------

    def run_cycle(
        self,
        *,
        self_assessment: SelfAssessment | None = None,
        pattern_report: PatternReport | None = None,
        counterfactual_batches: list[list[CounterfactualResult]] | None = None,
        lifecycle_report: OptimisationReport | None = None,
        portfolio_result: OptimizationResult | None = None,
        portfolio_baseline_objective: float | None = None,
        outcome_summary: OutcomeSummary | None = None,
    ) -> ActuatorCycleResult:
        """Run whichever observation reports are provided through their
        corresponding actuator(s) and return the aggregated result.

        Any report left as None is simply skipped — callers are not required
        to run all six observation modules on every cycle.
        """
        result = ActuatorCycleResult()

        if self_assessment is not None:
            result.extend(self.act_on_self_assessment(self_assessment))
        if pattern_report is not None:
            result.extend(self.act_on_pattern_report(pattern_report))
        if counterfactual_batches is not None:
            result.extend(self.act_on_counterfactual_batch(counterfactual_batches))
        if lifecycle_report is not None:
            result.extend(self.act_on_lifecycle_report(lifecycle_report))
        if portfolio_result is not None:
            result.extend(
                self.act_on_portfolio_result(portfolio_result, portfolio_baseline_objective)
            )
        if outcome_summary is not None:
            result.extend(self.act_on_outcome_summary(outcome_summary))

        logger.info(
            f"LearningLoopActuator.run_cycle: {len(result.actions)} actions "
            f"({result.by_type})"
        )
        return result
