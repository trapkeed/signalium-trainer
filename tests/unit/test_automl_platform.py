"""Phase 16 — AutoML Platform Unit Tests.

Comprehensive test suite covering all modules of the Institutional AutoML
and Research Optimization Platform.

Tests:
  - SearchSpace: all parameter types, sampling, grid, YAML round-trip
  - Objectives: scoring, normalization, Pareto front, composite research score
  - Pruning: median pruner, successive halving, no-op, build_pruner registry
  - Strategies: manual, grid, random, bayesian suggestion and reporting
  - Experiment: Trial lifecycle, SearchJob tracking, ExperimentRecord
  - ResourceManager: slot acquisition, usage tracking, throttle detection
  - BackboneCatalog: spec retrieval, variant generation, from_config
  - ExperimentRanker: overall, per-instrument, per-timeframe, per-regime
  - AutoMLDatabase: CRUD for jobs, records, pruning, promotion, analytics
  - Visualization: all chart types, full dashboard
  - AutoMLAPI: all endpoints, error handling, champion/challenger
  - AutoMLEngine: mock evaluation, full run, budget, promotion
"""

from __future__ import annotations

import math
import os
import tempfile
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# SearchSpace Tests
# ===========================================================================


class TestFloatParameter:
    def test_sample_in_range(self):
        from trainer.automl.search_space import FloatParameter

        p = FloatParameter("lr", low=0.001, high=0.1)
        for _ in range(20):
            v = p.sample()
            assert 0.001 <= v <= 0.1

    def test_log_scale_sample(self):
        from trainer.automl.search_space import FloatParameter

        p = FloatParameter("lr", low=1e-5, high=1e-2, log=True)
        for _ in range(20):
            v = p.sample()
            assert 1e-5 <= v <= 1e-2

    def test_step_quantization(self):
        from trainer.automl.search_space import FloatParameter

        p = FloatParameter("dropout", low=0.0, high=0.5, step=0.1)
        for _ in range(20):
            v = p.sample()
            assert abs(v - round(v / 0.1) * 0.1) < 1e-9

    def test_grid_values(self):
        from trainer.automl.search_space import FloatParameter

        p = FloatParameter("x", low=0.0, high=1.0)
        vals = p.grid_values(5)
        assert len(vals) == 5
        assert vals[0] == pytest.approx(0.0)
        assert vals[-1] == pytest.approx(1.0)

    def test_to_from_dict(self):
        from trainer.automl.search_space import FloatParameter, BaseParameter

        p = FloatParameter("lr", low=1e-5, high=1e-2, log=True, step=None)
        d = p.to_dict()
        p2 = BaseParameter.from_dict(d)
        assert p2.name == "lr"
        assert p2.log is True
        assert p2.low == pytest.approx(1e-5)


class TestIntParameter:
    def test_sample_in_range(self):
        from trainer.automl.search_space import IntParameter

        p = IntParameter("n_layers", low=1, high=6)
        for _ in range(30):
            v = p.sample()
            assert 1 <= v <= 6
            assert isinstance(v, int)

    def test_step(self):
        from trainer.automl.search_space import IntParameter

        p = IntParameter("batch_size", low=16, high=128, step=16)
        for _ in range(30):
            v = p.sample()
            assert v % 16 == 0

    def test_grid_values(self):
        from trainer.automl.search_space import IntParameter

        p = IntParameter("n", low=1, high=5)
        vals = p.grid_values()
        assert 1 in vals and 5 in vals

    def test_to_from_dict(self):
        from trainer.automl.search_space import IntParameter, BaseParameter

        p = IntParameter("n", low=2, high=8, step=2)
        p2 = BaseParameter.from_dict(p.to_dict())
        assert p2.low == 2 and p2.step == 2


class TestCategoricalParameter:
    def test_sample_from_choices(self):
        from trainer.automl.search_space import CategoricalParameter

        p = CategoricalParameter("opt", choices=["adam", "sgd", "adamw"])
        for _ in range(30):
            assert p.sample() in ["adam", "sgd", "adamw"]

    def test_weighted_sampling(self):
        from trainer.automl.search_space import CategoricalParameter
        import random

        p = CategoricalParameter("opt", choices=["a", "b"], weights=[0.99, 0.01])
        rng = random.Random(0)
        counts = {"a": 0, "b": 0}
        for _ in range(1000):
            counts[p.sample(rng)] += 1
        assert counts["a"] > counts["b"] * 5

    def test_to_from_dict(self):
        from trainer.automl.search_space import CategoricalParameter, BaseParameter

        p = CategoricalParameter("bb", choices=["lstm", "gru"])
        p2 = BaseParameter.from_dict(p.to_dict())
        assert p2.choices == ["lstm", "gru"]


class TestConditionalParameter:
    def test_active_when_parent_matches(self):
        from trainer.automl.search_space import ConditionalParameter, FloatParameter

        inner = FloatParameter("momentum", low=0.8, high=0.99)
        cond = ConditionalParameter(
            name="momentum",
            parent="optimizer",
            parent_value="sgd",
            parameter=inner,
            default=None,
        )
        assert cond.is_active({"optimizer": "sgd"})
        assert not cond.is_active({"optimizer": "adam"})

    def test_to_from_dict(self):
        from trainer.automl.search_space import ConditionalParameter, FloatParameter, BaseParameter

        inner = FloatParameter("m", low=0.8, high=0.99)
        cond = ConditionalParameter(
            name="m",
            parent="opt",
            parent_value="sgd",
            parameter=inner,
            default=0.9,
        )
        d = cond.to_dict()
        cond2 = BaseParameter.from_dict(d)
        assert cond2.parent == "opt"
        assert cond2.parent_value == "sgd"


class TestSearchSpace:
    def test_sample_returns_all_params(self):
        from trainer.automl.search_space import SearchSpace, FloatParameter, IntParameter

        space = SearchSpace("test")
        space.add(FloatParameter("lr", 1e-4, 1e-2))
        space.add(IntParameter("n", 1, 5))
        sample = space.sample()
        assert "lr" in sample
        assert "n" in sample

    def test_grid_returns_list(self):
        from trainer.automl.search_space import SearchSpace, CategoricalParameter

        space = SearchSpace("g")
        space.add(CategoricalParameter("bb", choices=["lstm", "gru"]))
        grid = space.grid()
        assert len(grid) == 2
        assert all("bb" in c for c in grid)

    def test_to_from_dict(self):
        from trainer.automl.search_space import SearchSpace, FloatParameter

        space = SearchSpace("s1")
        space.add(FloatParameter("lr", 1e-5, 1e-2, log=True))
        d = space.to_dict()
        space2 = SearchSpace.from_dict(d)
        assert space2.name == "s1"
        assert len(space2) == 1

    def test_build_standard_space(self):
        from trainer.automl.search_space import build_standard_space

        space = build_standard_space()
        assert len(space) > 10
        sample = space.sample()
        assert "learning_rate" in sample
        assert "backbone" in sample

    def test_len(self):
        from trainer.automl.search_space import SearchSpace, FloatParameter, IntParameter

        space = SearchSpace()
        space.add(FloatParameter("a", 0, 1))
        space.add(IntParameter("b", 1, 5))
        assert len(space) == 2


# ===========================================================================
# Objectives Tests
# ===========================================================================


class TestObjective:
    def test_is_better_maximize(self):
        from trainer.automl.objectives import Objective

        obj = Objective("sharpe", "maximize")
        assert obj.is_better(1.5, 1.0)
        assert not obj.is_better(0.5, 1.0)

    def test_is_better_minimize(self):
        from trainer.automl.objectives import Objective

        obj = Objective("drawdown", "minimize")
        assert obj.is_better(0.1, 0.2)
        assert not obj.is_better(0.3, 0.2)

    def test_normalize_maximize(self):
        from trainer.automl.objectives import Objective

        obj = Objective("score", "maximize")
        assert obj.normalize(1.0, 0.0, 1.0) == pytest.approx(1.0)
        assert obj.normalize(0.0, 0.0, 1.0) == pytest.approx(0.0)
        assert obj.normalize(0.5, 0.0, 1.0) == pytest.approx(0.5)

    def test_normalize_minimize(self):
        from trainer.automl.objectives import Objective

        obj = Objective("drawdown", "minimize")
        assert obj.normalize(0.0, 0.0, 1.0) == pytest.approx(1.0)
        assert obj.normalize(1.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_invalid_direction(self):
        from trainer.automl.objectives import Objective

        with pytest.raises(ValueError):
            Objective("x", "sideways")

    def test_to_from_dict(self):
        from trainer.automl.objectives import Objective

        obj = Objective("pf", "maximize", weight=2.0, clip_min=0.0, clip_max=5.0)
        obj2 = Objective.from_dict(obj.to_dict())
        assert obj2.name == "pf" and obj2.weight == pytest.approx(2.0)


class TestObjectiveScorer:
    def setup_method(self):
        from trainer.automl.objectives import Objective, ObjectiveScorer

        self.scorer = ObjectiveScorer(
            [
                Objective("sharpe_ratio", "maximize", weight=0.5, clip_min=-2.0, clip_max=5.0),
                Objective("max_drawdown", "minimize", weight=0.5, clip_min=0.0, clip_max=0.5),
            ]
        )

    def test_composite_score_range(self):
        score = self.scorer.composite_score(
            {
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.1,
            }
        )
        assert 0.0 <= score <= 1.0

    def test_composite_better_sharpe_scores_higher(self):
        good = self.scorer.composite_score(
            {"sharpe_ratio": 3.0, "max_drawdown": 0.15},
            [
                {"sharpe_ratio": 0.0, "max_drawdown": 0.40},
                {"sharpe_ratio": 3.0, "max_drawdown": 0.15},
            ],
        )
        bad = self.scorer.composite_score(
            {"sharpe_ratio": 0.0, "max_drawdown": 0.40},
            [
                {"sharpe_ratio": 0.0, "max_drawdown": 0.40},
                {"sharpe_ratio": 3.0, "max_drawdown": 0.15},
            ],
        )
        assert good > bad

    def test_per_objective_scores(self):
        scores = self.scorer.per_objective_scores(
            {
                "sharpe_ratio": 1.5,
                "max_drawdown": 0.2,
            }
        )
        assert "sharpe_ratio" in scores
        assert "max_drawdown" in scores

    def test_pareto_front(self):
        solutions = [
            {"sharpe_ratio": 3.0, "max_drawdown": 0.2},
            {"sharpe_ratio": 0.5, "max_drawdown": 0.5},
            {"sharpe_ratio": 1.0, "max_drawdown": 0.15},
        ]
        front = self.scorer.pareto_front(solutions)
        assert isinstance(front, list)
        assert 0 in front  # highest sharpe should be on Pareto front

    def test_rank_returns_sorted(self):
        metrics_list = [
            {"sharpe_ratio": 0.5, "max_drawdown": 0.4},
            {"sharpe_ratio": 3.0, "max_drawdown": 0.1},
        ]
        ranked = self.scorer.rank(metrics_list)
        assert ranked[0][1] > ranked[1][1]  # first has higher score

    def test_missing_metric_defaults_to_zero(self):
        score = self.scorer.composite_score({})
        assert score == pytest.approx(0.0)

    def test_empty_objectives_raises(self):
        from trainer.automl.objectives import ObjectiveScorer

        with pytest.raises(ValueError):
            ObjectiveScorer([])

    def test_dominates(self):
        a = {"sharpe_ratio": 3.0, "max_drawdown": 0.1}
        b = {"sharpe_ratio": 1.0, "max_drawdown": 0.3}
        assert self.scorer.dominates(a, b)
        assert not self.scorer.dominates(b, a)


class TestCompositeResearchScore:
    def test_score_range(self):
        from trainer.automl.objectives import compute_composite_research_score

        score = compute_composite_research_score(
            {
                "validation_score": 0.6,
                "sharpe_ratio": 1.2,
                "profit_factor": 1.8,
                "max_drawdown": 0.15,
                "expected_calibration_error": 0.05,
            }
        )
        assert 0.0 <= score <= 1.0

    def test_better_metrics_higher_score(self):
        from trainer.automl.objectives import compute_composite_research_score

        good = compute_composite_research_score(
            {
                "validation_score": 0.75,
                "sharpe_ratio": 2.5,
                "profit_factor": 2.0,
                "max_drawdown": 0.05,
                "expected_calibration_error": 0.02,
            }
        )
        bad = compute_composite_research_score(
            {
                "validation_score": 0.50,
                "sharpe_ratio": 0.1,
                "profit_factor": 0.8,
                "max_drawdown": 0.40,
                "expected_calibration_error": 0.30,
            }
        )
        assert good > bad

    def test_empty_metrics(self):
        from trainer.automl.objectives import compute_composite_research_score

        score = compute_composite_research_score({})
        assert 0.0 <= score <= 1.0


class TestStandardObjectives:
    def test_all_standard_objectives_present(self):
        from trainer.automl.objectives import STANDARD_OBJECTIVES

        expected = [
            "profit_factor",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "expected_calibration_error",
            "validation_score",
        ]
        for e in expected:
            assert e in STANDARD_OBJECTIVES

    def test_get_objective(self):
        from trainer.automl.objectives import get_objective

        obj = get_objective("sharpe_ratio")
        assert obj.direction == "maximize"

    def test_get_objective_invalid(self):
        from trainer.automl.objectives import get_objective

        with pytest.raises(KeyError):
            get_objective("nonexistent_metric")

    def test_build_research_scorer(self):
        from trainer.automl.objectives import build_research_scorer

        scorer = build_research_scorer()
        assert len(scorer.objectives) >= 3


# ===========================================================================
# Pruning Tests
# ===========================================================================


class TestNoPruner:
    def test_never_prunes(self):
        from trainer.automl.pruning import NoPruner

        pruner = NoPruner()
        history = {"t1": [(1, 0.9), (2, 0.8)], "t2": [(1, 0.7)]}
        for step in range(10):
            decision = pruner.should_prune("t3", step, 0.1, history)
            assert decision.prune is False


class TestMedianPruner:
    def setup_method(self):
        from trainer.automl.pruning import MedianPruner

        self.pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)

    def test_no_prune_during_startup(self):
        history = {"t1": [(5, 0.9)], "t2": [(5, 0.8)]}
        decision = self.pruner.should_prune("t3", 5, 0.1, history)
        assert decision.prune is False

    def test_prunes_below_median(self):
        history = {
            "t1": [(5, 0.9)],
            "t2": [(5, 0.8)],
            "t3": [(5, 0.7)],
            "t4": [(5, 0.6)],
        }
        decision = self.pruner.should_prune("t5", 5, 0.1, history)
        assert decision.prune is True

    def test_keeps_above_median(self):
        history = {
            "t1": [(5, 0.2)],
            "t2": [(5, 0.3)],
            "t3": [(5, 0.1)],
            "t4": [(5, 0.4)],
        }
        decision = self.pruner.should_prune("t5", 5, 0.9, history)
        assert decision.prune is False

    def test_decision_has_trial_id(self):
        history = {f"t{i}": [(5, 0.5)] for i in range(5)}
        decision = self.pruner.should_prune("my_trial", 5, 0.9, history)
        assert decision.trial_id == "my_trial"

    def test_to_dict(self):
        d = self.pruner.to_dict()
        assert d["type"] == "median"
        assert "n_startup_trials" in d


class TestSuccessiveHalvingPruner:
    def setup_method(self):
        from trainer.automl.pruning import SuccessiveHalvingPruner

        self.pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)

    def test_below_min_resource_not_pruned(self):
        decision = self.pruner.should_prune("t1", 0, 0.1, {})
        assert decision.prune is False

    def test_no_peers_not_pruned(self):
        decision = self.pruner.should_prune("t1", 3, 0.1, {})
        assert decision.prune is False

    def test_to_dict(self):
        d = self.pruner.to_dict()
        assert d["type"] == "successive_halving"


class TestBuildPruner:
    def test_none_pruner(self):
        from trainer.automl.pruning import build_pruner, NoPruner

        p = build_pruner({"type": "none"})
        assert isinstance(p, NoPruner)

    def test_median_pruner(self):
        from trainer.automl.pruning import build_pruner, MedianPruner

        p = build_pruner({"type": "median", "n_startup_trials": 3})
        assert isinstance(p, MedianPruner)
        assert p.n_startup_trials == 3

    def test_successive_halving_pruner(self):
        from trainer.automl.pruning import build_pruner, SuccessiveHalvingPruner

        p = build_pruner({"type": "successive_halving", "reduction_factor": 4})
        assert isinstance(p, SuccessiveHalvingPruner)
        assert p.reduction_factor == 4

    def test_unknown_type_raises(self):
        from trainer.automl.pruning import build_pruner

        with pytest.raises(ValueError):
            build_pruner({"type": "hyperband"})


# ===========================================================================
# Strategy Tests
# ===========================================================================


def _make_space():
    from trainer.automl.search_space import (
        SearchSpace,
        FloatParameter,
        IntParameter,
        CategoricalParameter,
    )

    space = SearchSpace("test")
    space.add(FloatParameter("lr", 1e-4, 1e-2, log=True))
    space.add(IntParameter("n_layers", 1, 4))
    space.add(CategoricalParameter("backbone", choices=["lstm", "gru"]))
    return space


class TestManualStrategy:
    def test_returns_configs_in_order(self):
        from trainer.automl.strategies import ManualStrategy

        configs = [{"lr": 0.001}, {"lr": 0.01}]
        s = ManualStrategy(_make_space(), configs)
        s1 = s.suggest("t1")
        s2 = s.suggest("t2")
        assert s1.config == {"lr": 0.001}
        assert s2.config == {"lr": 0.01}

    def test_raises_when_exhausted(self):
        from trainer.automl.strategies import ManualStrategy

        s = ManualStrategy(_make_space(), [{"lr": 0.001}])
        s.suggest("t1")
        with pytest.raises(StopIteration):
            s.suggest("t2")


class TestGridStrategy:
    def test_generates_grid(self):
        from trainer.automl.strategies import GridStrategy

        space = _make_space()
        s = GridStrategy(space, max_per_param=3)
        assert s.n_trials > 0

    def test_exhaustion(self):
        from trainer.automl.strategies import GridStrategy

        space = _make_space()
        s = GridStrategy(space, max_per_param=2)
        n = s.n_trials
        for i in range(n):
            s.suggest(f"t{i}")
        with pytest.raises(StopIteration):
            s.suggest("overflow")

    def test_is_exhausted(self):
        from trainer.automl.strategies import GridStrategy

        space = _make_space()
        s = GridStrategy(space, max_per_param=2)
        while not s.is_exhausted():
            s.suggest(f"t{s._iteration}")
        assert s.is_exhausted()


class TestRandomStrategy:
    def test_samples_within_space(self):
        from trainer.automl.strategies import RandomStrategy

        s = RandomStrategy(_make_space(), n_trials=10, seed=42)
        for i in range(10):
            sug = s.suggest(f"t{i}")
            assert "lr" in sug.config
            assert "backbone" in sug.config

    def test_reproducible_with_seed(self):
        from trainer.automl.strategies import RandomStrategy

        space = _make_space()
        s1 = RandomStrategy(space, n_trials=5, seed=99)
        s2 = RandomStrategy(space, n_trials=5, seed=99)
        configs1 = [s1.suggest(f"t{i}").config for i in range(5)]
        configs2 = [s2.suggest(f"t{i}").config for i in range(5)]
        assert configs1 == configs2

    def test_exhaustion(self):
        from trainer.automl.strategies import RandomStrategy

        s = RandomStrategy(_make_space(), n_trials=3, seed=0)
        for i in range(3):
            s.suggest(f"t{i}")
        with pytest.raises(StopIteration):
            s.suggest("overflow")

    def test_iteration_increments(self):
        from trainer.automl.strategies import RandomStrategy

        s = RandomStrategy(_make_space(), n_trials=5, seed=0)
        s.suggest("t0")
        assert s._iteration == 1


class TestBayesianStrategy:
    def test_initial_random_phase(self):
        from trainer.automl.strategies import BayesianStrategy

        s = BayesianStrategy(_make_space(), n_trials=10, n_initial_random=5, seed=1)
        sug = s.suggest("t0")
        assert "lr" in sug.config

    def test_reports_accepted(self):
        from trainer.automl.strategies import BayesianStrategy

        s = BayesianStrategy(_make_space(), n_trials=10, seed=2)
        sug = s.suggest("t0")
        s.report("t0", 0.75)
        assert "t0" in s._scores

    def test_bayesian_phase_after_warmup(self):
        from trainer.automl.strategies import BayesianStrategy

        s = BayesianStrategy(_make_space(), n_trials=10, n_initial_random=3, seed=3)
        for i in range(3):
            sug = s.suggest(f"t{i}")
            s.report(f"t{i}", 0.5 + i * 0.1)
        sug = s.suggest("t3")
        assert "lr" in sug.config

    def test_exhaustion(self):
        from trainer.automl.strategies import BayesianStrategy

        s = BayesianStrategy(_make_space(), n_trials=2, seed=4)
        for i in range(2):
            s.suggest(f"t{i}")
        with pytest.raises(StopIteration):
            s.suggest("overflow")

    def test_to_dict(self):
        from trainer.automl.strategies import BayesianStrategy

        s = BayesianStrategy(_make_space(), n_trials=5, seed=0)
        d = s.to_dict()
        assert d["type"] == "bayesian"


class TestBuildStrategy:
    def test_random(self):
        from trainer.automl.strategies import build_strategy, RandomStrategy

        s = build_strategy({"type": "random", "n_trials": 5}, _make_space())
        assert isinstance(s, RandomStrategy)

    def test_bayesian(self):
        from trainer.automl.strategies import build_strategy, BayesianStrategy

        s = build_strategy({"type": "bayesian", "n_trials": 10}, _make_space())
        assert isinstance(s, BayesianStrategy)

    def test_unknown_raises(self):
        from trainer.automl.strategies import build_strategy

        with pytest.raises(ValueError):
            build_strategy({"type": "evolutionary"}, _make_space())


# ===========================================================================
# Experiment Tests
# ===========================================================================


class TestTrial:
    def test_initial_status_pending(self):
        from trainer.automl.experiment import Trial, TrialStatus

        t = Trial()
        assert t.status == TrialStatus.PENDING

    def test_start_sets_running(self):
        from trainer.automl.experiment import Trial, TrialStatus

        t = Trial()
        t.start()
        assert t.status == TrialStatus.RUNNING
        assert t.start_time != ""

    def test_complete_sets_score(self):
        from trainer.automl.experiment import Trial, TrialStatus

        t = Trial()
        t.start()
        t.complete(score=0.82, metrics={"sharpe_ratio": 1.5})
        assert t.status == TrialStatus.COMPLETED
        assert t.score == pytest.approx(0.82)
        assert t.metrics["sharpe_ratio"] == pytest.approx(1.5)

    def test_prune_sets_pruned(self):
        from trainer.automl.experiment import Trial, TrialStatus

        t = Trial()
        t.start()
        t.prune(step=3, reason="below median")
        assert t.status == TrialStatus.PRUNED
        assert t.pruning_step == 3

    def test_fail_sets_failed(self):
        from trainer.automl.experiment import Trial, TrialStatus

        t = Trial()
        t.start()
        t.fail("OOM error")
        assert t.status == TrialStatus.FAILED
        assert "OOM" in t.error_message

    def test_to_from_dict(self):
        from trainer.automl.experiment import Trial

        t = Trial(backbone="lstm", config={"lr": 0.001})
        t.complete(0.7, {"sharpe": 1.0})
        d = t.to_dict()
        t2 = Trial.from_dict(d)
        assert t2.score == pytest.approx(0.7)
        assert t2.backbone == "lstm"

    def test_intermediate_scores(self):
        from trainer.automl.experiment import Trial

        t = Trial()
        t.add_intermediate(1, 0.5)
        t.add_intermediate(2, 0.7)
        assert len(t.intermediate_scores) == 2

    def test_is_terminal(self):
        from trainer.automl.experiment import Trial

        t = Trial()
        assert not t.is_terminal
        t.complete(0.5)
        assert t.is_terminal


class TestSearchJob:
    def test_initial_state(self):
        from trainer.automl.experiment import SearchJob

        job = SearchJob(name="test", n_trials=10)
        assert job.n_completed == 0
        assert job.best_score is None

    def test_add_trial_updates_champion(self):
        from trainer.automl.experiment import SearchJob, Trial

        job = SearchJob()
        t = Trial()
        t.complete(0.85)
        job.add_trial(t)
        assert job.best_score == pytest.approx(0.85)
        assert job.champion_trial_id == t.trial_id

    def test_champion_trial_improves(self):
        from trainer.automl.experiment import SearchJob, Trial

        job = SearchJob()
        for score in [0.5, 0.7, 0.6, 0.9, 0.8]:
            t = Trial()
            t.complete(score)
            job.add_trial(t)
        assert job.best_score == pytest.approx(0.9)

    def test_job_lifecycle(self):
        from trainer.automl.experiment import SearchJob

        job = SearchJob()
        job.start()
        assert job.status == "running"
        job.complete()
        assert job.status == "completed"
        assert job.completed_at != ""

    def test_summary_dict(self):
        from trainer.automl.experiment import SearchJob

        job = SearchJob(name="myJob")
        d = job.summary()
        assert d["name"] == "myJob"
        assert "n_completed" in d


class TestExperimentRecord:
    def test_from_trial(self):
        from trainer.automl.experiment import Trial, SearchJob, ExperimentRecord

        trial = Trial(backbone="gru", config={"lr": 0.001})
        trial.complete(0.75, {"sharpe_ratio": 1.8})
        job = SearchJob(name="test")
        record = ExperimentRecord.from_trial(trial, job)
        assert record.trial_id == trial.trial_id
        assert record.backbone == "gru"
        assert record.primary_score == pytest.approx(0.75)

    def test_to_dict(self):
        from trainer.automl.experiment import Trial, SearchJob, ExperimentRecord

        trial = Trial()
        trial.complete(0.6)
        record = ExperimentRecord.from_trial(trial, SearchJob())
        d = record.to_dict()
        assert "trial_id" in d
        assert "config" in d


# ===========================================================================
# Resource Manager Tests
# ===========================================================================


class TestResourceManager:
    def test_acquire_release_slot(self):
        from trainer.automl.resource_manager import ResourceManager

        rm = ResourceManager(max_workers=2)
        token = rm.acquire_slot("t1")
        assert token is not None
        assert rm.active_workers == 1
        rm.release_slot(token)
        assert rm.active_workers == 0

    def test_max_workers_respected(self):
        from trainer.automl.resource_manager import ResourceManager

        rm = ResourceManager(max_workers=1)
        token = rm.acquire_slot("t1")
        assert token is not None
        token2 = rm.acquire_slot("t2")
        assert token2 is None

    def test_record_usage(self):
        from trainer.automl.resource_manager import ResourceManager

        rm = ResourceManager()
        snap = rm.record_usage("t1", cpu_percent=45.0, ram_mb=8192.0)
        assert snap.trial_id == "t1"
        assert snap.cpu_percent == pytest.approx(45.0)

    def test_get_trial_usage(self):
        from trainer.automl.resource_manager import ResourceManager

        rm = ResourceManager()
        rm.record_usage("t1", cpu_percent=30.0, ram_mb=4096.0)
        rm.record_usage("t1", cpu_percent=50.0, ram_mb=6144.0)
        usage = rm.get_trial_usage("t1")
        assert usage["n_snapshots"] == 2
        assert usage["peak_cpu_percent"] == pytest.approx(50.0)

    def test_usage_report(self):
        from trainer.automl.resource_manager import ResourceManager

        rm = ResourceManager(max_workers=4)
        rm.record_usage("t1", cpu_percent=60.0, ram_mb=2048.0)
        report = rm.usage_report()
        assert report["max_workers"] == 4
        assert "peak_cpu_percent" in report

    def test_throttle_high_ram(self):
        from trainer.automl.resource_manager import ResourceManager

        rm = ResourceManager(max_ram_gb=8.0)
        rm.record_usage("t1", ram_mb=8000.0)  # ~98% of 8GB
        assert rm.should_throttle()

    def test_to_dict(self):
        from trainer.automl.resource_manager import ResourceManager

        rm = ResourceManager(max_workers=4, gpu_memory_gb=40.0)
        d = rm.to_dict()
        assert d["max_workers"] == 4
        assert d["gpu_memory_gb"] == pytest.approx(40.0)


# ===========================================================================
# Backbone Search Tests
# ===========================================================================


class TestBackboneCatalog:
    def test_get_default_backbone(self):
        from trainer.automl.backbone_search import BackboneCatalog

        cat = BackboneCatalog()
        spec = cat.get("lstm")
        assert spec.name == "lstm"

    def test_all_default_backbones_present(self):
        from trainer.automl.backbone_search import BackboneCatalog

        cat = BackboneCatalog()
        for name in [
            "lstm",
            "gru",
            "transformer_encoder",
            "temporal_cnn",
            "hybrid_lstm_cnn",
            "hybrid_transformer_cnn",
        ]:
            assert cat.get(name).name == name

    def test_register_custom_backbone(self):
        from trainer.automl.backbone_search import BackboneCatalog, BackboneSpec

        cat = BackboneCatalog()
        spec = BackboneSpec(name="custom_rnn", hidden_size=64)
        cat.register(spec)
        assert cat.get("custom_rnn").name == "custom_rnn"

    def test_get_unknown_raises(self):
        from trainer.automl.backbone_search import BackboneCatalog

        cat = BackboneCatalog()
        with pytest.raises(KeyError):
            cat.get("nonexistent_backbone")

    def test_generate_variants(self):
        from trainer.automl.backbone_search import BackboneCatalog

        cat = BackboneCatalog()
        variants = cat.generate_variants("lstm", hidden_sizes=[64, 128])
        assert len(variants) == 2
        sizes = {v.hidden_size for v in variants}
        assert sizes == {64, 128}

    def test_from_config(self):
        from trainer.automl.backbone_search import BackboneCatalog

        cat = BackboneCatalog()
        spec = cat.from_config({"name": "gru", "hidden_size": 256})
        assert spec.hidden_size == 256
        assert spec.name == "gru"

    def test_list_names(self):
        from trainer.automl.backbone_search import BackboneCatalog

        cat = BackboneCatalog()
        names = cat.list_names()
        assert "lstm" in names
        assert "transformer_encoder" in names


class TestBackboneSpec:
    def test_to_from_dict(self):
        from trainer.automl.backbone_search import BackboneSpec

        spec = BackboneSpec(name="lstm", hidden_size=128, num_layers=3)
        d = spec.to_dict()
        spec2 = BackboneSpec.from_dict(d)
        assert spec2.hidden_size == 128
        assert spec2.num_layers == 3

    def test_display_name_auto_generated(self):
        from trainer.automl.backbone_search import BackboneSpec

        spec = BackboneSpec(name="temporal_cnn")
        assert spec.display_name == "Temporal Cnn"


class TestBackboneSearcher:
    def test_rank_results(self):
        from trainer.automl.backbone_search import BackboneSearcher, BackboneEvalResult

        searcher = BackboneSearcher()
        results = [
            BackboneEvalResult("lstm", {}, score=0.7),
            BackboneEvalResult("gru", {}, score=0.9),
            BackboneEvalResult("transformer_encoder", {}, score=0.8),
        ]
        ranked = searcher.rank_results(results)
        assert ranked[0].backbone_name == "gru"
        assert ranked[0].rank == 1

    def test_best(self):
        from trainer.automl.backbone_search import BackboneSearcher, BackboneEvalResult

        searcher = BackboneSearcher()
        results = [
            BackboneEvalResult("lstm", {}, score=0.5),
            BackboneEvalResult("gru", {}, score=0.95),
        ]
        best = searcher.best(results)
        assert best.backbone_name == "gru"

    def test_summary(self):
        from trainer.automl.backbone_search import BackboneSearcher, BackboneEvalResult

        searcher = BackboneSearcher()
        results = [BackboneEvalResult("lstm", {}, score=0.6)]
        s = searcher.summary(results)
        assert s["best_backbone"] == "lstm"


# ===========================================================================
# Ranking Tests
# ===========================================================================


def _make_experiment_dicts(n: int, base_score: float = 0.5) -> List[Dict]:
    return [
        {
            "trial_id": str(uuid.uuid4()),
            "record_id": str(uuid.uuid4()),
            "job_id": "job-1",
            "backbone": "lstm",
            "model_version": "1.0.0",
            "status": "completed",
            "primary_score": base_score + i * 0.05,
            "validation_results": {
                "sharpe_ratio": 1.0 + i * 0.1,
                "max_drawdown": 0.2 - i * 0.01,
                "profit_factor": 1.5 + i * 0.05,
                "expected_calibration_error": 0.1,
                "validation_score": base_score + i * 0.05,
            },
            "metrics": {},
        }
        for i in range(n)
    ]


class TestExperimentRanker:
    def test_rank_all_sorted(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        exps = _make_experiment_dicts(5)
        ranked = ranker.rank_all(exps)
        assert len(ranked) == 5
        for i in range(len(ranked) - 1):
            assert ranked[i].score >= ranked[i + 1].score

    def test_rank_1_is_best(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        exps = _make_experiment_dicts(3)
        ranked = ranker.rank_all(exps)
        assert ranked[0].rank == 1

    def test_top_n(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        exps = _make_experiment_dicts(10)
        top3 = ranker.top_n(exps, 3)
        assert len(top3) == 3

    def test_rank_by_instrument(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        exps = _make_experiment_dicts(3)
        ranked = ranker.rank_by_instrument(exps, "EURUSD")
        assert all(e.dimension_type == "instrument" for e in ranked)
        assert all(e.dimension == "EURUSD" for e in ranked)

    def test_rank_by_timeframe(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        exps = _make_experiment_dicts(3)
        ranked = ranker.rank_by_timeframe(exps, "H1")
        assert all(e.dimension_type == "timeframe" for e in ranked)

    def test_rank_by_regime(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        exps = _make_experiment_dicts(3)
        ranked = ranker.rank_by_regime(exps, "trending_up")
        assert all(e.dimension_type == "regime" for e in ranked)

    def test_full_ranking(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        exps = _make_experiment_dicts(4)
        result = ranker.full_ranking(
            exps,
            instruments=["EURUSD"],
            timeframes=["H1"],
            regimes=["trending_up"],
        )
        assert result.champion is not None
        assert result.n_experiments == 4
        assert "EURUSD" in result.by_instrument
        assert "H1" in result.by_timeframe

    def test_empty_experiments(self):
        from trainer.automl.ranking import ExperimentRanker

        ranker = ExperimentRanker()
        assert ranker.rank_all([]) == []


class TestRankingEntry:
    def test_to_dict(self):
        from trainer.automl.ranking import RankingEntry

        entry = RankingEntry(trial_id="t1", rank=1, score=0.9, backbone="lstm", dimension="overall")
        d = entry.to_dict()
        assert d["rank"] == 1
        assert d["trial_id"] == "t1"


# ===========================================================================
# Database Tests
# ===========================================================================


class TestAutoMLDatabase:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_automl.db")

    def _make_db(self):
        from trainer.automl.database import AutoMLDatabase

        return AutoMLDatabase(self.db_path)

    def _make_job(self):
        from trainer.automl.experiment import SearchJob

        return SearchJob(name="test_job", strategy="random", n_trials=10)

    def _make_record(self, job_id: str = ""):
        from trainer.automl.experiment import Trial, SearchJob, ExperimentRecord

        trial = Trial(backbone="lstm", config={"lr": 0.001})
        trial.complete(0.75, {"sharpe_ratio": 1.5})
        job = SearchJob()
        job.job_id = job_id or job.job_id
        return ExperimentRecord.from_trial(trial, job)

    def test_save_and_get_job(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        retrieved = db.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved["name"] == "test_job"
        db.close()

    def test_list_jobs(self):
        db = self._make_db()
        for i in range(3):
            job = self._make_job()
            job.name = f"job_{i}"
            db.save_job(job)
        jobs = db.list_jobs()
        assert len(jobs) == 3
        db.close()

    def test_list_jobs_by_status(self):
        db = self._make_db()
        job = self._make_job()
        job.status = "running"
        db.save_job(job)
        running = db.list_jobs(status="running")
        assert len(running) == 1
        pending = db.list_jobs(status="pending")
        assert len(pending) == 0
        db.close()

    def test_update_job_status(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        db.update_job_status(job.job_id, "completed", best_score=0.85)
        retrieved = db.get_job(job.job_id)
        assert retrieved["status"] == "completed"
        assert retrieved["best_score"] == pytest.approx(0.85)
        db.close()

    def test_save_and_get_record(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        record = self._make_record(job.job_id)
        db.save_record(record)
        retrieved = db.get_record(record.record_id)
        assert retrieved is not None
        assert retrieved["backbone"] == "lstm"
        db.close()

    def test_list_records(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        for _ in range(5):
            db.save_record(self._make_record(job.job_id))
        records = db.list_records(job_id=job.job_id)
        assert len(records) == 5
        db.close()

    def test_save_pruning_decision(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        did = db.save_pruning_decision(
            "t1", job.job_id, step=5, score=0.3, threshold=0.6, prune=True, reason="below median"
        )
        assert did is not None
        history = db.get_pruning_history("t1")
        assert len(history) == 1
        assert history[0]["prune"] == 1
        db.close()

    def test_save_promotion_decision(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        did = db.save_promotion_decision(
            trial_id="t1",
            job_id=job.job_id,
            model_version="2.0.0",
            is_eligible=True,
            auto_approved=True,
            manually_approved=False,
            champion_status=True,
            rollback_eligible=True,
            rules=[],
            notes="auto",
        )
        assert did is not None
        history = db.get_promotion_history()
        assert len(history) == 1
        assert history[0]["is_eligible"] is True
        db.close()

    def test_best_record(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        for i in range(3):
            from trainer.automl.experiment import Trial, SearchJob, ExperimentRecord

            trial = Trial(backbone="lstm", config={})
            trial.complete(float(i) * 0.3)
            record = ExperimentRecord.from_trial(trial, job)
            record.primary_score = float(i) * 0.3
            db.save_record(record)
        best = db.best_record(job.job_id)
        assert best is not None
        assert best["primary_score"] == pytest.approx(0.6)
        db.close()

    def test_backbone_summary(self):
        db = self._make_db()
        job = self._make_job()
        db.save_job(job)
        for bb in ["lstm", "lstm", "gru"]:
            from trainer.automl.experiment import Trial, SearchJob, ExperimentRecord

            trial = Trial(backbone=bb, config={})
            trial.complete(0.5)
            record = ExperimentRecord.from_trial(trial, job)
            record.status = "completed"
            record.primary_score = 0.5
            db.save_record(record)
        summary = db.backbone_summary(job.job_id)
        names = [s["backbone"] for s in summary]
        assert "lstm" in names
        db.close()

    def test_stats(self):
        db = self._make_db()
        stats = db.stats()
        assert "total_jobs" in stats
        assert "total_records" in stats
        db.close()

    def test_get_nonexistent_job(self):
        db = self._make_db()
        assert db.get_job("nonexistent") is None
        db.close()


# ===========================================================================
# Visualization Tests
# ===========================================================================


def _make_completed_records(n: int = 5) -> List[Dict]:
    records = []
    for i in range(n):
        records.append(
            {
                "trial_id": str(uuid.uuid4()),
                "record_id": str(uuid.uuid4()),
                "backbone": "lstm",
                "status": "completed",
                "primary_score": 0.5 + i * 0.05,
                "composite_research_score": 0.5 + i * 0.05,
                "created_at": f"2024-01-{i+1:02d}T00:00:00Z",
                "training_duration_seconds": 10.0 * i,
                "validation_duration_seconds": 2.0,
                "pruned": False,
                "config": {"learning_rate": 0.001 * (i + 1), "dropout": 0.1},
                "validation_results": {
                    "sharpe_ratio": 1.0 + i * 0.2,
                    "max_drawdown": 0.20 - i * 0.01,
                },
            }
        )
    return records


class TestAutoMLVisualizationBuilder:
    def setup_method(self):
        from trainer.automl.visualization import AutoMLVisualizationBuilder

        self.viz = AutoMLVisualizationBuilder()
        self.records = _make_completed_records(5)

    def test_optimization_history(self):
        chart = self.viz.optimization_history(self.records)
        assert chart["chart_type"] == "line"
        assert len(chart["series"][0]["data"]) == 5

    def test_search_convergence(self):
        chart = self.viz.search_convergence(self.records)
        assert chart["chart_type"] == "line"
        data = chart["series"][0]["data"]
        scores = [p["running_best"] for p in data]
        assert scores == sorted(scores)  # running best is non-decreasing

    def test_parameter_importance(self):
        chart = self.viz.parameter_importance(self.records)
        assert chart["chart_type"] == "bar"

    def test_leaderboard_progression(self):
        chart = self.viz.leaderboard_progression(self.records)
        assert chart["chart_type"] == "bar"
        assert len(chart["series"][0]["data"]) <= 5

    def test_experiment_timeline(self):
        chart = self.viz.experiment_timeline(self.records)
        assert chart["chart_type"] == "gantt"
        assert len(chart["entries"]) == 5

    def test_resource_utilization(self):
        usage = {
            "active_workers": 1,
            "max_workers": 4,
            "peak_cpu_percent": 45.0,
            "peak_ram_mb": 8192.0,
            "peak_gpu_percent": 0.0,
            "peak_vram_mb": 0.0,
        }
        chart = self.viz.resource_utilization(usage)
        assert chart["chart_type"] == "gauge"

    def test_objective_tradeoff(self):
        chart = self.viz.objective_tradeoff(self.records, "sharpe_ratio", "max_drawdown")
        assert chart["chart_type"] == "scatter"
        assert len(chart["series"][0]["data"]) == 5

    def test_score_distribution(self):
        chart = self.viz.score_distribution(self.records, n_bins=10)
        assert chart["chart_type"] == "histogram"
        assert sum(b["count"] for b in chart["bins"]) == 5

    def test_full_dashboard_has_all_charts(self):
        dashboard = self.viz.full_dashboard(self.records)
        expected_keys = [
            "optimization_history",
            "search_convergence",
            "parameter_importance",
            "leaderboard_progression",
            "experiment_timeline",
            "objective_tradeoff",
            "score_distribution",
        ]
        for k in expected_keys:
            assert k in dashboard

    def test_empty_records(self):
        chart = self.viz.optimization_history([])
        assert chart["series"][0]["data"] == []

    def test_parameter_importance_needs_min_trials(self):
        chart = self.viz.parameter_importance(self.records[:2])
        assert "note" in chart


# ===========================================================================
# API Tests
# ===========================================================================


class TestAutoMLAPI:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "api_test.db")
        from trainer.automl.database import AutoMLDatabase
        from trainer.automl.api import AutoMLAPI

        self.db = AutoMLDatabase(db_path)
        self.api = AutoMLAPI(self.db)

    def teardown_method(self):
        self.db.close()

    def _create_job(self, name: str = "test") -> Dict:
        return self.api.create_job(
            {
                "name": name,
                "strategy": "random",
                "n_trials": 10,
                "objective": "composite_research_score",
            }
        )

    def test_create_job_ok(self):
        resp = self._create_job("my_job")
        assert resp["status"] == "ok"
        assert resp["data"]["name"] == "my_job"

    def test_get_job_ok(self):
        resp = self._create_job()
        job_id = resp["data"]["job_id"]
        get_resp = self.api.get_job(job_id)
        assert get_resp["status"] == "ok"
        assert get_resp["data"]["job_id"] == job_id

    def test_get_job_not_found(self):
        resp = self.api.get_job("nonexistent-id")
        assert resp["status"] == "error"
        assert resp["code"] == "NOT_FOUND"

    def test_list_jobs_empty(self):
        resp = self.api.list_jobs()
        assert resp["status"] == "ok"
        assert resp["data"]["count"] == 0

    def test_list_jobs_with_entries(self):
        self._create_job("j1")
        self._create_job("j2")
        resp = self.api.list_jobs()
        assert resp["data"]["count"] == 2

    def test_pause_and_resume_job(self):
        resp = self._create_job()
        job_id = resp["data"]["job_id"]
        pause_resp = self.api.pause_job(job_id)
        assert pause_resp["status"] == "ok"
        resume_resp = self.api.resume_job(job_id)
        assert resume_resp["status"] == "ok"

    def test_list_trials_empty(self):
        resp = self._create_job()
        job_id = resp["data"]["job_id"]
        resp2 = self.api.list_trials(job_id=job_id)
        assert resp2["status"] == "ok"
        assert resp2["data"]["count"] == 0

    def test_get_leaderboard_empty(self):
        resp = self.api.get_leaderboard()
        assert resp["status"] == "ok"
        assert resp["data"]["total"] == 0

    def test_get_pareto_front_empty(self):
        resp = self.api.get_pareto_front()
        assert resp["status"] == "ok"
        assert resp["data"]["n_pareto"] == 0

    def test_get_promotion_candidates_empty(self):
        resp = self.api.get_promotion_candidates()
        assert resp["status"] == "ok"
        assert resp["data"]["total_candidates"] == 0

    def test_approve_promotion(self):
        job_resp = self._create_job()
        job_id = job_resp["data"]["job_id"]
        resp = self.api.approve_promotion(
            trial_id="t-test", job_id=job_id, model_version="2.0.0", notes="Test approval"
        )
        assert resp["status"] == "ok"
        assert resp["data"]["champion_status"] is True

    def test_get_stats(self):
        resp = self.api.get_stats()
        assert resp["status"] == "ok"
        assert "total_jobs" in resp["data"]

    def test_champion_challenger_not_found(self):
        resp = self.api.champion_challenger_compare("bad1", "bad2")
        assert resp["status"] == "error"

    def test_get_dashboard(self):
        resp = self.api.get_dashboard()
        assert resp["status"] == "ok"
        assert "charts" in resp["data"]

    def test_get_optimization_history(self):
        job_resp = self._create_job()
        job_id = job_resp["data"]["job_id"]
        resp = self.api.get_optimization_history(job_id)
        assert resp["status"] == "ok"


# ===========================================================================
# Engine Tests
# ===========================================================================


class TestAutoMLConfig:
    def test_defaults(self):
        from trainer.automl.engine import AutoMLConfig

        cfg = AutoMLConfig()
        assert cfg.n_trials == 20
        assert cfg.strategy == "random"
        assert cfg.objective == "composite_research_score"

    def test_from_dict(self):
        from trainer.automl.engine import AutoMLConfig

        cfg = AutoMLConfig.from_dict(
            {
                "name": "test_search",
                "strategy": "bayesian",
                "n_trials": 30,
            }
        )
        assert cfg.name == "test_search"
        assert cfg.strategy == "bayesian"
        assert cfg.n_trials == 30

    def test_to_dict(self):
        from trainer.automl.engine import AutoMLConfig

        cfg = AutoMLConfig(name="test", n_trials=15)
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert d["n_trials"] == 15


class TestAutoMLEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def _make_engine(self, n_trials: int = 5, strategy: str = "random") -> "AutoMLEngine":
        from trainer.automl.engine import AutoMLEngine, AutoMLConfig

        cfg = AutoMLConfig(
            name="unit_test_search",
            strategy=strategy,
            n_trials=n_trials,
            seed=42,
            db_path=os.path.join(self.tmpdir, "engine_test.db"),
        )
        return AutoMLEngine(config=cfg)

    def test_engine_initialization(self):
        engine = self._make_engine()
        assert engine.config.n_trials == 5
        assert engine.search_space is not None

    def test_mock_evaluate_returns_metrics(self):
        from trainer.automl.backbone_search import BackboneSpec

        engine = self._make_engine()
        spec = BackboneSpec(name="lstm")
        from trainer.automl.experiment import Trial

        trial = Trial(config={"learning_rate": 0.001, "dropout": 0.1, "hidden_size": 128})
        metrics = engine._mock_evaluate(trial.config, spec)
        assert "validation_score" in metrics
        assert "sharpe_ratio" in metrics
        assert 0.0 < metrics["validation_score"] < 1.0

    def test_check_promotion_defaults(self):
        engine = self._make_engine()
        good = {
            "validation_score": 0.65,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.15,
            "profit_factor": 1.5,
            "win_rate": 0.55,
        }
        assert engine._check_promotion(good, {}) is True

    def test_check_promotion_fails_threshold(self):
        engine = self._make_engine()
        bad = {
            "validation_score": 0.40,
            "sharpe_ratio": -0.5,
            "max_drawdown": 0.60,
            "profit_factor": 0.7,
            "win_rate": 0.30,
        }
        assert (
            engine._check_promotion(
                bad,
                {
                    "min_validation_score": 0.55,
                },
            )
            is False
        )

    def test_full_run_random(self):
        from trainer.automl.engine import AutoMLResult

        engine = self._make_engine(n_trials=5, strategy="random")
        result = engine.run(model_id="unit_test")
        assert isinstance(result, AutoMLResult)
        assert result.job.status == "completed"
        assert result.job.n_completed > 0
        assert result.job.best_score is not None

    def test_full_run_with_custom_train_fn(self):
        engine = self._make_engine(n_trials=3)

        def my_train_fn(config, backbone):
            return {
                "validation_score": 0.65,
                "sharpe_ratio": 1.5,
                "profit_factor": 1.8,
                "max_drawdown": 0.12,
                "win_rate": 0.55,
                "expected_calibration_error": 0.05,
            }

        result = engine.run(train_fn=my_train_fn)
        assert result.job.n_completed == 3
        assert result.job.best_score is not None

    def test_result_has_champion(self):
        engine = self._make_engine(n_trials=5)
        result = engine.run()
        assert result.champion_trial is not None
        assert result.champion_trial.score == result.job.best_score

    def test_result_summary(self):
        engine = self._make_engine(n_trials=3)
        result = engine.run()
        summary = result.summary()
        assert "AutoML Search" in summary
        assert "completed" in summary

    def test_from_dict(self):
        from trainer.automl.engine import AutoMLEngine

        engine = AutoMLEngine.from_dict(
            {
                "name": "dict_test",
                "strategy": "random",
                "n_trials": 3,
                "db_path": os.path.join(self.tmpdir, "dict_test.db"),
            }
        )
        assert engine.config.name == "dict_test"

    def test_bayesian_strategy_run(self):
        engine = self._make_engine(n_trials=6, strategy="bayesian")
        result = engine.run()
        assert result.job.status == "completed"
        assert result.job.n_completed >= 1

    def test_ranking_in_result(self):
        engine = self._make_engine(n_trials=5)
        result = engine.run()
        assert result.ranking is not None
        assert result.ranking.n_experiments >= 0

    def test_dashboard_in_result(self):
        engine = self._make_engine(n_trials=3)
        result = engine.run()
        assert result.dashboard is not None
        assert "optimization_history" in result.dashboard

    def test_report_in_result(self):
        engine = self._make_engine(n_trials=3)
        result = engine.run()
        assert result.report["report_type"] == "automl_platform"
        assert "n_trials_total" in result.report


# ===========================================================================
# __init__ public API
# ===========================================================================


class TestPublicAPI:
    def test_all_exports_importable(self):
        from trainer.automl import (
            FloatParameter,
            IntParameter,
            CategoricalParameter,
            ConditionalParameter,
            SearchSpace,
            build_standard_space,
            Objective,
            ObjectiveScorer,
            STANDARD_OBJECTIVES,
            get_objective,
            compute_composite_research_score,
            build_research_scorer,
            BasePruner,
            NoPruner,
            MedianPruner,
            SuccessiveHalvingPruner,
            PruningDecision,
            build_pruner,
            BaseStrategy,
            Suggestion,
            ManualStrategy,
            GridStrategy,
            RandomStrategy,
            BayesianStrategy,
            build_strategy,
            TrialStatus,
            Trial,
            SearchJob,
            ExperimentRecord,
            ResourceSnapshot,
            WorkerSlot,
            ResourceManager,
            BackboneSpec,
            BackboneCatalog,
            BackboneSearcher,
            BackboneEvalResult,
            RankingEntry,
            RankingResult,
            ExperimentRanker,
            AutoMLDatabase,
            AutoMLVisualizationBuilder,
            AutoMLAPI,
            AutoMLConfig,
            AutoMLResult,
            AutoMLEngine,
        )

    def test_all_names_in_all(self):
        import trainer.automl as m

        for name in m.__all__:
            assert hasattr(m, name), f"Missing: {name}"
