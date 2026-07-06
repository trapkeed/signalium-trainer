"""Phase 16 — AutoML Model Ranking.

Generates configurable rankings for completed experiments.

Produces:
  - Overall ranking (composite score)
  - Per-instrument ranking
  - Per-timeframe ranking
  - Per-regime ranking

Rankings can be computed across a single search job or across all jobs
in the AutoML database.

Usage::

    ranker = ExperimentRanker(scorer=build_research_scorer())
    rankings = ranker.rank_all(experiments)
    top5 = ranker.top_n(experiments, n=5)
    by_instrument = ranker.rank_by_instrument(experiments, "EURUSD")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from trainer.automl.objectives import ObjectiveScorer, build_research_scorer

# ---------------------------------------------------------------------------
# Ranking entry
# ---------------------------------------------------------------------------


@dataclass
class RankingEntry:
    """A single model's ranking in a dimension.

    Attributes:
        trial_id: The ranked trial.
        record_id: Experiment record ID.
        job_id: Parent search job.
        rank: Rank (1 = best).
        score: Score used for ranking.
        backbone: Backbone name.
        model_version: Model version string.
        metrics: Key metrics snapshot.
        dimension: What was ranked on ("overall", instrument name, etc.).
        dimension_type: "overall" | "instrument" | "timeframe" | "regime".
    """

    trial_id: str
    record_id: str = ""
    job_id: str = ""
    rank: int = 0
    score: float = 0.0
    backbone: str = ""
    model_version: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    dimension: str = "overall"
    dimension_type: str = "overall"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "record_id": self.record_id,
            "job_id": self.job_id,
            "rank": self.rank,
            "score": self.score,
            "backbone": self.backbone,
            "model_version": self.model_version,
            "metrics": self.metrics,
            "dimension": self.dimension,
            "dimension_type": self.dimension_type,
        }


# ---------------------------------------------------------------------------
# Full ranking result
# ---------------------------------------------------------------------------


@dataclass
class RankingResult:
    """Complete ranking output for a set of experiments.

    Attributes:
        overall: Sorted list of RankingEntry by composite score.
        by_instrument: Dict of instrument → sorted RankingEntry list.
        by_timeframe: Dict of timeframe → sorted RankingEntry list.
        by_regime: Dict of regime → sorted RankingEntry list.
        champion: The overall #1 ranked trial.
        n_experiments: Total number of experiments ranked.
    """

    overall: List[RankingEntry] = field(default_factory=list)
    by_instrument: Dict[str, List[RankingEntry]] = field(default_factory=dict)
    by_timeframe: Dict[str, List[RankingEntry]] = field(default_factory=dict)
    by_regime: Dict[str, List[RankingEntry]] = field(default_factory=dict)
    champion: Optional[RankingEntry] = None
    n_experiments: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_experiments": self.n_experiments,
            "champion": self.champion.to_dict() if self.champion else None,
            "overall": [e.to_dict() for e in self.overall],
            "by_instrument": {k: [e.to_dict() for e in v] for k, v in self.by_instrument.items()},
            "by_timeframe": {k: [e.to_dict() for e in v] for k, v in self.by_timeframe.items()},
            "by_regime": {k: [e.to_dict() for e in v] for k, v in self.by_regime.items()},
        }


# ---------------------------------------------------------------------------
# Experiment ranker
# ---------------------------------------------------------------------------


class ExperimentRanker:
    """Ranks experiments across multiple dimensions.

    Args:
        scorer: ObjectiveScorer for composite score computation.
            Defaults to the standard research scorer.
    """

    INSTRUMENTS = [
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "USDCHF",
        "AUDUSD",
        "USDCAD",
        "NZDUSD",
        "EURGBP",
        "EURJPY",
        "GBPJPY",
        "XAUUSD",
        "XAGUSD",
        "USOIL",
        "US30",
        "NAS100",
    ]

    TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]

    REGIMES = ["trending_up", "trending_down", "ranging", "volatile", "crisis"]

    def __init__(self, scorer: Optional[ObjectiveScorer] = None) -> None:
        self.scorer = scorer or build_research_scorer()

    def _score_experiment(
        self, experiment: Dict[str, Any], reference: List[Dict[str, Any]]
    ) -> float:
        metrics = experiment.get("metrics") or experiment.get("validation_results", {})
        return self.scorer.composite_score(metrics, reference)

    def _extract_metrics(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        return experiment.get("metrics") or experiment.get("validation_results", {})

    def _make_entry(
        self,
        experiment: Dict[str, Any],
        rank: int,
        score: float,
        dimension: str = "overall",
        dimension_type: str = "overall",
    ) -> RankingEntry:
        return RankingEntry(
            trial_id=experiment.get("trial_id", ""),
            record_id=experiment.get("record_id", ""),
            job_id=experiment.get("job_id", ""),
            rank=rank,
            score=score,
            backbone=experiment.get("backbone", ""),
            model_version=experiment.get("model_version", ""),
            metrics=self._extract_metrics(experiment),
            dimension=dimension,
            dimension_type=dimension_type,
        )

    def rank_all(self, experiments: List[Dict[str, Any]]) -> List[RankingEntry]:
        """Return experiments sorted by composite score descending."""
        if not experiments:
            return []
        all_metrics = [self._extract_metrics(e) for e in experiments]
        scored: List[Tuple[int, float]] = []
        for i, exp in enumerate(experiments):
            s = self._score_experiment(exp, all_metrics)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            self._make_entry(experiments[idx], rank + 1, score)
            for rank, (idx, score) in enumerate(scored)
        ]

    def top_n(self, experiments: List[Dict[str, Any]], n: int = 10) -> List[RankingEntry]:
        """Return top-N ranked experiments."""
        return self.rank_all(experiments)[:n]

    def rank_by_instrument(
        self, experiments: List[Dict[str, Any]], instrument: str
    ) -> List[RankingEntry]:
        """Rank experiments by their per-instrument metrics."""
        filtered = []
        for exp in experiments:
            metrics = self._extract_metrics(exp)
            instr_metrics = (
                metrics.get("per_instrument", {}).get(instrument)
                or metrics.get(instrument)
                or metrics
            )
            filtered.append({**exp, "_ranked_metrics": instr_metrics})

        if not filtered:
            return []

        ref = [f.get("_ranked_metrics", {}) for f in filtered]
        scored: List[Tuple[int, float]] = []
        for i, exp in enumerate(filtered):
            metrics = exp.get("_ranked_metrics", {})
            s = self.scorer.composite_score(metrics, ref)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            self._make_entry(
                experiments[idx], rank + 1, score, dimension=instrument, dimension_type="instrument"
            )
            for rank, (idx, score) in enumerate(scored)
        ]

    def rank_by_timeframe(
        self, experiments: List[Dict[str, Any]], timeframe: str
    ) -> List[RankingEntry]:
        """Rank experiments by their per-timeframe metrics."""
        filtered = []
        for exp in experiments:
            metrics = self._extract_metrics(exp)
            tf_metrics = (
                metrics.get("per_timeframe", {}).get(timeframe) or metrics.get(timeframe) or metrics
            )
            filtered.append({**exp, "_ranked_metrics": tf_metrics})

        if not filtered:
            return []

        ref = [f.get("_ranked_metrics", {}) for f in filtered]
        scored: List[Tuple[int, float]] = []
        for i, exp in enumerate(filtered):
            metrics = exp.get("_ranked_metrics", {})
            s = self.scorer.composite_score(metrics, ref)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            self._make_entry(
                experiments[idx], rank + 1, score, dimension=timeframe, dimension_type="timeframe"
            )
            for rank, (idx, score) in enumerate(scored)
        ]

    def rank_by_regime(self, experiments: List[Dict[str, Any]], regime: str) -> List[RankingEntry]:
        """Rank experiments by their per-regime metrics."""
        filtered = []
        for exp in experiments:
            metrics = self._extract_metrics(exp)
            regime_metrics = (
                metrics.get("per_regime", {}).get(regime) or metrics.get(regime) or metrics
            )
            filtered.append({**exp, "_ranked_metrics": regime_metrics})

        if not filtered:
            return []

        ref = [f.get("_ranked_metrics", {}) for f in filtered]
        scored: List[Tuple[int, float]] = []
        for i, exp in enumerate(filtered):
            metrics = exp.get("_ranked_metrics", {})
            s = self.scorer.composite_score(metrics, ref)
            scored.append((i, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            self._make_entry(
                experiments[idx], rank + 1, score, dimension=regime, dimension_type="regime"
            )
            for rank, (idx, score) in enumerate(scored)
        ]

    def full_ranking(
        self,
        experiments: List[Dict[str, Any]],
        instruments: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        regimes: Optional[List[str]] = None,
    ) -> RankingResult:
        """Compute full multi-dimensional ranking.

        Args:
            experiments: List of experiment record dicts.
            instruments: Instrument list (defaults to INSTRUMENTS).
            timeframes: Timeframe list (defaults to TIMEFRAMES).
            regimes: Regime list (defaults to REGIMES).

        Returns:
            RankingResult with overall, per-instrument, per-timeframe,
            per-regime rankings.
        """
        instr_list = instruments or self.INSTRUMENTS
        tf_list = timeframes or self.TIMEFRAMES
        regime_list = regimes or self.REGIMES

        overall = self.rank_all(experiments)

        by_instrument: Dict[str, List[RankingEntry]] = {}
        for instr in instr_list:
            by_instrument[instr] = self.rank_by_instrument(experiments, instr)

        by_timeframe: Dict[str, List[RankingEntry]] = {}
        for tf in tf_list:
            by_timeframe[tf] = self.rank_by_timeframe(experiments, tf)

        by_regime: Dict[str, List[RankingEntry]] = {}
        for regime in regime_list:
            by_regime[regime] = self.rank_by_regime(experiments, regime)

        champion = overall[0] if overall else None

        return RankingResult(
            overall=overall,
            by_instrument=by_instrument,
            by_timeframe=by_timeframe,
            by_regime=by_regime,
            champion=champion,
            n_experiments=len(experiments),
        )
