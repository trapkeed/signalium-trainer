"""Phase 17 — AEIP Expert Failure Analysis.

Tracks and analyzes expert prediction failures:

  - Experts that consistently disagree with the ensemble
  - Experts that consistently outperform
  - Experts that fail during specific regimes
  - Experts with persistently high uncertainty
  - Agreement/disagreement matrices across expert pairs

Usage::

    analyzer = ExpertFailureAnalyzer()
    analyzer.record(
        expert_id="expert_01",
        regime="trending_up",
        predicted_class="buy",
        ensemble_class="sell",
        true_class="sell",
        confidence=0.8,
        uncertainty=0.1,
    )
    report = analyzer.get_report("expert_01")
    matrix = analyzer.agreement_matrix(["expert_01", "expert_02"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Failure event
# ---------------------------------------------------------------------------


@dataclass
class FailureEvent:
    """A single prediction failure or success record.

    Attributes:
        expert_id: The expert being evaluated.
        regime: Market regime at prediction time.
        instrument: Instrument.
        timeframe: Timeframe.
        predicted_class: Expert's prediction.
        ensemble_class: Ensemble's final prediction.
        true_class: Ground truth class (None = not yet known).
        confidence: Expert confidence.
        uncertainty: Expert uncertainty.
        agrees_with_ensemble: Whether expert matched ensemble output.
        is_correct: Whether expert was correct (if true_class known).
    """

    expert_id: str
    regime: str = "unknown"
    instrument: str = ""
    timeframe: str = ""
    predicted_class: str = "no_trade"
    ensemble_class: str = "no_trade"
    true_class: Optional[str] = None
    confidence: float = 0.5
    uncertainty: float = 0.5
    agrees_with_ensemble: bool = True
    is_correct: Optional[bool] = None


# ---------------------------------------------------------------------------
# Expert failure report
# ---------------------------------------------------------------------------


@dataclass
class ExpertFailureReport:
    """Aggregated failure analysis for one expert.

    Attributes:
        expert_id: The expert analyzed.
        n_predictions: Total predictions recorded.
        n_correct: Correct predictions (where true class known).
        n_incorrect: Incorrect predictions.
        accuracy: Overall accuracy.
        n_disagree_ensemble: Times expert disagreed with ensemble.
        ensemble_agreement_rate: Fraction of agreement with ensemble.
        high_uncertainty_rate: Fraction of predictions with uncertainty > 0.6.
        failure_by_regime: Dict of regime → failure count.
        success_by_regime: Dict of regime → success count.
        regime_accuracy: Dict of regime → accuracy.
        is_consistent_disagree: Expert disagrees more than 40% of the time.
        is_consistent_outperformer: Expert accuracy 10%+ above ensemble accuracy.
        is_regime_specific_failure: Any regime with accuracy < 0.40.
        is_high_uncertainty: Avg uncertainty > 0.6.
        notes: List of automated observations.
    """

    expert_id: str
    n_predictions: int = 0
    n_correct: int = 0
    n_incorrect: int = 0
    accuracy: float = 0.0
    n_disagree_ensemble: int = 0
    ensemble_agreement_rate: float = 1.0
    high_uncertainty_rate: float = 0.0
    failure_by_regime: Dict[str, int] = field(default_factory=dict)
    success_by_regime: Dict[str, int] = field(default_factory=dict)
    regime_accuracy: Dict[str, float] = field(default_factory=dict)
    is_consistent_disagree: bool = False
    is_consistent_outperformer: bool = False
    is_regime_specific_failure: bool = False
    is_high_uncertainty: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_id": self.expert_id,
            "n_predictions": self.n_predictions,
            "n_correct": self.n_correct,
            "n_incorrect": self.n_incorrect,
            "accuracy": self.accuracy,
            "n_disagree_ensemble": self.n_disagree_ensemble,
            "ensemble_agreement_rate": self.ensemble_agreement_rate,
            "high_uncertainty_rate": self.high_uncertainty_rate,
            "failure_by_regime": self.failure_by_regime,
            "success_by_regime": self.success_by_regime,
            "regime_accuracy": self.regime_accuracy,
            "is_consistent_disagree": self.is_consistent_disagree,
            "is_consistent_outperformer": self.is_consistent_outperformer,
            "is_regime_specific_failure": self.is_regime_specific_failure,
            "is_high_uncertainty": self.is_high_uncertainty,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Failure analyzer
# ---------------------------------------------------------------------------


class ExpertFailureAnalyzer:
    """Analyzes expert failure patterns across regimes and instruments.

    Records expert predictions and computes failure statistics including
    agreement/disagreement with the ensemble, accuracy by regime,
    and automatic pattern detection.
    """

    def __init__(self, uncertainty_threshold: float = 0.6,
                 disagree_threshold: float = 0.40,
                 outperform_margin: float = 0.10,
                 regime_failure_threshold: float = 0.40) -> None:
        self.uncertainty_threshold = uncertainty_threshold
        self.disagree_threshold = disagree_threshold
        self.outperform_margin = outperform_margin
        self.regime_failure_threshold = regime_failure_threshold
        self._events: Dict[str, List[FailureEvent]] = {}
        self._pair_agree: Dict[Tuple[str, str], List[bool]] = {}

    def record(self, expert_id: str, regime: str = "unknown",
               instrument: str = "", timeframe: str = "",
               predicted_class: str = "no_trade",
               ensemble_class: str = "no_trade",
               true_class: Optional[str] = None,
               confidence: float = 0.5,
               uncertainty: float = 0.5) -> FailureEvent:
        """Record one prediction event for an expert."""
        agrees = (predicted_class == ensemble_class)
        is_correct = None
        if true_class is not None:
            is_correct = (predicted_class == true_class)

        event = FailureEvent(
            expert_id=expert_id,
            regime=regime,
            instrument=instrument,
            timeframe=timeframe,
            predicted_class=predicted_class,
            ensemble_class=ensemble_class,
            true_class=true_class,
            confidence=confidence,
            uncertainty=uncertainty,
            agrees_with_ensemble=agrees,
            is_correct=is_correct,
        )
        if expert_id not in self._events:
            self._events[expert_id] = []
        self._events[expert_id].append(event)
        return event

    def record_pair_agreement(self, expert_a: str, expert_b: str,
                               class_a: str, class_b: str) -> None:
        """Record pairwise agreement between two experts."""
        key = (min(expert_a, expert_b), max(expert_a, expert_b))
        if key not in self._pair_agree:
            self._pair_agree[key] = []
        self._pair_agree[key].append(class_a == class_b)

    def get_report(self, expert_id: str,
                   ensemble_accuracy: float = 0.0) -> ExpertFailureReport:
        """Compute failure analysis report for one expert."""
        events = self._events.get(expert_id, [])
        if not events:
            return ExpertFailureReport(expert_id=expert_id)

        n = len(events)
        labeled = [e for e in events if e.is_correct is not None]
        n_correct = sum(1 for e in labeled if e.is_correct)
        n_incorrect = len(labeled) - n_correct
        accuracy = n_correct / len(labeled) if labeled else 0.0

        n_disagree = sum(1 for e in events if not e.agrees_with_ensemble)
        agreement_rate = 1.0 - (n_disagree / n)

        high_unc = sum(1 for e in events
                       if e.uncertainty > self.uncertainty_threshold)
        high_unc_rate = high_unc / n

        failure_by_regime: Dict[str, int] = {}
        success_by_regime: Dict[str, int] = {}
        for e in labeled:
            r = e.regime
            if e.is_correct:
                success_by_regime[r] = success_by_regime.get(r, 0) + 1
            else:
                failure_by_regime[r] = failure_by_regime.get(r, 0) + 1

        all_regimes = set(failure_by_regime) | set(success_by_regime)
        regime_accuracy: Dict[str, float] = {}
        for r in all_regimes:
            s = success_by_regime.get(r, 0)
            f = failure_by_regime.get(r, 0)
            tot = s + f
            regime_accuracy[r] = s / tot if tot > 0 else 0.0

        is_disagree = agreement_rate < (1.0 - self.disagree_threshold)
        is_outperformer = (ensemble_accuracy > 0 and
                           accuracy > ensemble_accuracy + self.outperform_margin)
        is_regime_fail = any(
            acc < self.regime_failure_threshold
            for acc in regime_accuracy.values()
        )
        is_high_unc = high_unc_rate > 0.5

        notes = []
        if is_disagree:
            notes.append(
                f"Expert disagrees with ensemble {(1 - agreement_rate):.1%} of the time."
            )
        if is_outperformer:
            notes.append(
                f"Expert outperforms ensemble by {accuracy - ensemble_accuracy:.1%}."
            )
        if is_regime_fail:
            bad_regimes = [r for r, acc in regime_accuracy.items()
                           if acc < self.regime_failure_threshold]
            notes.append(
                f"Poor regime accuracy in: {', '.join(bad_regimes)}."
            )
        if is_high_unc:
            notes.append(
                f"High uncertainty rate: {high_unc_rate:.1%}."
            )

        return ExpertFailureReport(
            expert_id=expert_id,
            n_predictions=n,
            n_correct=n_correct,
            n_incorrect=n_incorrect,
            accuracy=round(accuracy, 4),
            n_disagree_ensemble=n_disagree,
            ensemble_agreement_rate=round(agreement_rate, 4),
            high_uncertainty_rate=round(high_unc_rate, 4),
            failure_by_regime=failure_by_regime,
            success_by_regime=success_by_regime,
            regime_accuracy={r: round(a, 4) for r, a in regime_accuracy.items()},
            is_consistent_disagree=is_disagree,
            is_consistent_outperformer=is_outperformer,
            is_regime_specific_failure=is_regime_fail,
            is_high_uncertainty=is_high_unc,
            notes=notes,
        )

    def all_reports(self, ensemble_accuracy: float = 0.0
                    ) -> Dict[str, ExpertFailureReport]:
        return {
            eid: self.get_report(eid, ensemble_accuracy)
            for eid in self._events
        }

    def agreement_matrix(self, expert_ids: List[str]) -> Dict[str, Any]:
        """Return pairwise agreement matrix as a chart-ready dict."""
        matrix: Dict[str, Dict[str, float]] = {}
        for a in expert_ids:
            matrix[a] = {}
            for b in expert_ids:
                if a == b:
                    matrix[a][b] = 1.0
                else:
                    key = (min(a, b), max(a, b))
                    agreements = self._pair_agree.get(key, [])
                    if agreements:
                        matrix[a][b] = sum(agreements) / len(agreements)
                    else:
                        matrix[a][b] = None
        return {
            "chart_type": "heatmap",
            "title": "Expert Agreement Matrix",
            "expert_ids": expert_ids,
            "matrix": matrix,
        }

    def disagreement_matrix(self, expert_ids: List[str]) -> Dict[str, Any]:
        """Return pairwise disagreement matrix (1 - agreement)."""
        agree = self.agreement_matrix(expert_ids)
        for a in expert_ids:
            for b in expert_ids:
                v = agree["matrix"][a][b]
                agree["matrix"][a][b] = (1.0 - v) if v is not None else None
        agree["title"] = "Expert Disagreement Matrix"
        return agree

    def top_performers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return top N experts by accuracy."""
        reports = self.all_reports()
        sorted_reports = sorted(
            reports.values(),
            key=lambda r: r.accuracy,
            reverse=True,
        )
        return [r.to_dict() for r in sorted_reports[:n]]

    def consistent_disagreers(self) -> List[str]:
        """Return expert IDs that consistently disagree with the ensemble."""
        return [
            eid for eid, r in self.all_reports().items()
            if r.is_consistent_disagree
        ]
