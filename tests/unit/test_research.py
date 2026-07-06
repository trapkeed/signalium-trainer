"""Unit tests for the Signalium Research & Benchmarking Framework.

Covers: ExperimentRecord, ExperimentTracker, HyperparameterRecord,
BenchmarkConditions, Leaderboard, WalkForwardValidator, RollingWindowValidator,
ChampionRegistry, ModelStatus, ReportGenerator, and exporters.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from trainer.research import (
    BenchmarkConditions,
    BenchmarkResult,
    ChampionRegistry,
    ExperimentRecord,
    ExperimentTracker,
    HyperparameterRecord,
    Leaderboard,
    ModelRunResult,
    ModelStatus,
    ReportConfig,
    ReportGenerator,
    RollingWindowValidator,
    ScoringWeights,
    ValidationFold,
    WalkForwardValidator,
)
from trainer.research.benchmark.runner import BenchmarkResult, ModelRunResult
from trainer.research.leaderboard.leaderboard import LeaderboardEntry

# -----------------------------------------------------------------------
# ExperimentRecord
# -----------------------------------------------------------------------


class TestExperimentRecord:
    def test_creates_with_uuid(self) -> None:
        rec = ExperimentRecord()
        assert len(rec.experiment_id) == 36  # UUID4 length

    def test_default_status_is_running(self) -> None:
        rec = ExperimentRecord()
        assert rec.status == "running"

    def test_complete_sets_status_and_metrics(self) -> None:
        rec = ExperimentRecord()
        rec.complete({"accuracy": 0.75, "f1_macro": 0.70})
        assert rec.status == "completed"
        assert rec.metrics["accuracy"] == pytest.approx(0.75)
        assert rec.end_time is not None
        assert rec.training_duration_seconds is not None
        assert rec.training_duration_seconds >= 0

    def test_fail_sets_status(self) -> None:
        rec = ExperimentRecord()
        rec.fail("GPU OOM")
        assert rec.status == "failed"
        assert "GPU OOM" in rec.notes

    def test_to_dict_round_trip(self) -> None:
        rec = ExperimentRecord(
            name="test",
            model_name="quant_v1",
            model_version="1.0.0",
            random_seed=99,
        )
        rec.complete({"f1_macro": 0.68})
        d = rec.to_dict()
        restored = ExperimentRecord.from_dict(d)
        assert restored.experiment_id == rec.experiment_id
        assert restored.model_name == "quant_v1"
        assert restored.metrics["f1_macro"] == pytest.approx(0.68)
        assert restored.status == "completed"

    def test_to_dict_json_serialisable(self) -> None:
        rec = ExperimentRecord(name="x")
        rec.complete({"loss": 0.42})
        json.dumps(rec.to_dict(), default=str)


# -----------------------------------------------------------------------
# ExperimentTracker
# -----------------------------------------------------------------------


class TestExperimentTracker:
    @pytest.fixture()
    def tracker(self, tmp_path: Path) -> ExperimentTracker:
        return ExperimentTracker(store_path=tmp_path / "records.json")

    def test_save_and_retrieve(self, tracker: ExperimentTracker) -> None:
        rec = ExperimentRecord(name="run-1", model_name="quant_v1")
        tracker.save(rec)
        retrieved = tracker.get(rec.experiment_id)
        assert retrieved is not None
        assert retrieved.name == "run-1"

    def test_filter_by_status(self, tracker: ExperimentTracker) -> None:
        r1 = ExperimentRecord(name="ok")
        r1.complete({"accuracy": 0.8})
        r2 = ExperimentRecord(name="fail")
        r2.fail()
        tracker.save(r1)
        tracker.save(r2)
        completed = tracker.filter(status="completed")
        assert len(completed) == 1
        assert completed[0].name == "ok"

    def test_best_by_metric(self, tracker: ExperimentTracker) -> None:
        for score in [0.65, 0.80, 0.72]:
            r = ExperimentRecord(model_name="quant_v1")
            r.complete({"f1_macro": score})
            tracker.save(r)
        best = tracker.best_by("f1_macro", model_name="quant_v1")
        assert best is not None
        assert best.metrics["f1_macro"] == pytest.approx(0.80)

    def test_delete(self, tracker: ExperimentTracker) -> None:
        rec = ExperimentRecord()
        tracker.save(rec)
        assert tracker.delete(rec.experiment_id)
        assert tracker.get(rec.experiment_id) is None

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        p = tmp_path / "records.json"
        t1 = ExperimentTracker(store_path=p)
        rec = ExperimentRecord(name="persisted")
        rec.complete({"accuracy": 0.9})
        t1.save(rec)

        t2 = ExperimentTracker(store_path=p)
        retrieved = t2.get(rec.experiment_id)
        assert retrieved is not None
        assert retrieved.name == "persisted"

    def test_summary_returns_list(self, tracker: ExperimentTracker) -> None:
        rec = ExperimentRecord(name="s", model_name="quant_v1")
        rec.complete({"accuracy": 0.7, "f1_macro": 0.65})
        tracker.save(rec)
        rows = tracker.summary()
        assert isinstance(rows, list)
        assert rows[0]["model_name"] == "quant_v1"


# -----------------------------------------------------------------------
# HyperparameterRecord
# -----------------------------------------------------------------------


class TestHyperparameterRecord:
    def test_defaults(self) -> None:
        h = HyperparameterRecord()
        assert h.learning_rate == pytest.approx(1e-3)
        assert h.epochs == 100

    def test_to_dict_round_trip(self) -> None:
        h = HyperparameterRecord(learning_rate=5e-4, hidden_size=512, loss_type="focal")
        d = h.to_dict()
        h2 = HyperparameterRecord.from_dict(d)
        assert h2.learning_rate == pytest.approx(5e-4)
        assert h2.hidden_size == 512
        assert h2.loss_type == "focal"

    def test_diff_detects_changes(self) -> None:
        h1 = HyperparameterRecord(learning_rate=1e-3, hidden_size=256)
        h2 = HyperparameterRecord(learning_rate=5e-4, hidden_size=256)
        diffs = h1.diff(h2)
        assert "learning_rate" in diffs
        assert "hidden_size" not in diffs

    def test_extra_fields_preserved(self) -> None:
        h = HyperparameterRecord.from_dict({"learning_rate": 1e-3, "custom_param": "x"})
        assert h.extra["custom_param"] == "x"


# -----------------------------------------------------------------------
# BenchmarkConditions
# -----------------------------------------------------------------------


class TestBenchmarkConditions:
    def test_validate_passes(self) -> None:
        BenchmarkConditions(train_ratio=0.7, val_ratio=0.15).validate()

    def test_validate_fails_sum_too_large(self) -> None:
        with pytest.raises(ValueError, match="< 1.0"):
            BenchmarkConditions(train_ratio=0.7, val_ratio=0.31).validate()

    def test_validate_fails_zero_ratio(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BenchmarkConditions(train_ratio=0.0).validate()

    def test_to_dict_serialisable(self) -> None:
        d = BenchmarkConditions().to_dict()
        json.dumps(d)


# -----------------------------------------------------------------------
# Leaderboard
# -----------------------------------------------------------------------


def _make_benchmark_result(n_models: int = 3) -> BenchmarkResult:
    """Build a synthetic BenchmarkResult for testing."""
    import uuid

    runs = []
    for i in range(n_models):
        runs.append(
            ModelRunResult(
                model_name=f"model_{i}",
                metrics={
                    "accuracy": 0.60 + i * 0.05,
                    "f1_macro": 0.58 + i * 0.05,
                    "f1_weighted": 0.59 + i * 0.05,
                    "calibration_error": 0.10 - i * 0.02,
                    "inference_latency_ms": 5.0 - i * 1.0,
                },
                training_time_seconds=30.0 + i * 10,
                model_size_mb=1.5 + i * 0.5,
                status="completed",
            )
        )
    return BenchmarkResult(
        benchmark_id=str(uuid.uuid4()),
        conditions=BenchmarkConditions(),
        runs=runs,
        total_duration_seconds=90.0,
    )


class TestLeaderboard:
    @pytest.fixture()
    def leaderboard(self) -> Leaderboard:
        lb = Leaderboard()
        lb.ingest(_make_benchmark_result(3))
        return lb

    def test_ranked_entries_count(self, leaderboard: Leaderboard) -> None:
        entries = leaderboard.ranked_entries()
        assert len(entries) == 3

    def test_rank_1_is_best(self, leaderboard: Leaderboard) -> None:
        entries = leaderboard.ranked_entries()
        assert entries[0].rank == 1
        # model_2 has highest scores in synthetic data
        assert entries[0].accuracy >= entries[1].accuracy

    def test_champion_is_rank_1(self, leaderboard: Leaderboard) -> None:
        champ = leaderboard.champion()
        assert champ is not None
        assert champ.rank == 1

    def test_top_n(self, leaderboard: Leaderboard) -> None:
        top2 = leaderboard.top_n(2)
        assert len(top2) == 2
        assert top2[0].rank == 1
        assert top2[1].rank == 2

    def test_format_table_returns_string(self, leaderboard: Leaderboard) -> None:
        table = leaderboard.format_table()
        assert isinstance(table, str)
        assert "Rank" in table
        assert "Model" in table

    def test_to_dict_serialisable(self, leaderboard: Leaderboard) -> None:
        json.dumps(leaderboard.to_dict(), default=str)

    def test_failed_runs_appear_last(self) -> None:
        import uuid

        result = BenchmarkResult(
            benchmark_id=str(uuid.uuid4()),
            conditions=BenchmarkConditions(),
            runs=[
                ModelRunResult(
                    "good",
                    metrics={"accuracy": 0.8, "f1_macro": 0.75, "f1_weighted": 0.76},
                    status="completed",
                ),
                ModelRunResult("bad", status="failed", error="OOM"),
            ],
        )
        lb = Leaderboard()
        lb.ingest(result)
        entries = lb.ranked_entries()
        assert entries[-1].status == "failed"

    def test_custom_weights(self) -> None:
        weights = ScoringWeights(
            accuracy=1.0, f1_macro=0.0, f1_weighted=0.0, calibration_error=0.0, inference_speed=0.0
        )
        lb = Leaderboard(weights=weights)
        lb.ingest(_make_benchmark_result(3))
        # model_2 has highest accuracy so must be rank 1
        assert lb.champion().model_name == "model_2"


# -----------------------------------------------------------------------
# WalkForwardValidator
# -----------------------------------------------------------------------


class TestWalkForwardValidator:
    @pytest.fixture()
    def data(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = 200
        x = torch.randn(n, 10, 8)
        y = torch.randint(0, 3, (n,))
        return x, y

    def test_generates_correct_number_of_folds(self, data) -> None:
        v = WalkForwardValidator(min_train_size=100, val_size=20, step_size=20)
        folds = list(v.split(*data))
        assert len(folds) == v.num_folds(200)
        assert len(folds) > 0

    def test_fold_shapes_correct(self, data) -> None:
        v = WalkForwardValidator(min_train_size=100, val_size=20, step_size=20)
        for fold in v.split(*data):
            assert fold.x_train.shape[0] == fold.train_size
            assert fold.x_val.shape[0] == fold.val_size
            assert fold.y_train.shape[0] == fold.train_size
            assert fold.y_val.shape[0] == fold.val_size

    def test_train_always_before_val(self, data) -> None:
        v = WalkForwardValidator(min_train_size=80, val_size=20, step_size=20)
        for fold in v.split(*data):
            assert fold.train_end <= fold.val_start

    def test_max_folds_respected(self, data) -> None:
        v = WalkForwardValidator(min_train_size=80, val_size=20, step_size=10, max_folds=3)
        folds = list(v.split(*data))
        assert len(folds) <= 3

    def test_dataset_too_small_raises(self) -> None:
        v = WalkForwardValidator(min_train_size=100, val_size=50, step_size=10)
        x = torch.randn(120, 5, 3)
        y = torch.randint(0, 3, (120,))
        with pytest.raises(ValueError, match="too small"):
            list(v.split(x, y))

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardValidator(min_train_size=0, val_size=10, step_size=5)
        with pytest.raises(ValueError):
            WalkForwardValidator(min_train_size=10, val_size=0, step_size=5)
        with pytest.raises(ValueError):
            WalkForwardValidator(min_train_size=10, val_size=10, step_size=0)


# -----------------------------------------------------------------------
# RollingWindowValidator
# -----------------------------------------------------------------------


class TestRollingWindowValidator:
    @pytest.fixture()
    def data(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = 150
        x = torch.randn(n, 10, 8)
        y = torch.randint(0, 3, (n,))
        return x, y

    def test_generates_folds(self, data) -> None:
        v = RollingWindowValidator(train_size=60, val_size=20, step_size=20)
        folds = list(v.split(*data))
        assert len(folds) == v.num_folds(150)
        assert len(folds) > 0

    def test_fixed_window_size(self, data) -> None:
        v = RollingWindowValidator(train_size=60, val_size=20, step_size=20)
        for fold in v.split(*data):
            assert fold.x_train.shape[0] == 60
            assert fold.x_val.shape[0] == 20

    def test_step_slides_window(self, data) -> None:
        v = RollingWindowValidator(train_size=60, val_size=20, step_size=20)
        starts = [f.window_start for f in v.split(*data)]
        diffs = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        assert all(d == 20 for d in diffs)

    def test_dataset_too_small_raises(self) -> None:
        v = RollingWindowValidator(train_size=100, val_size=50, step_size=10)
        x = torch.randn(120, 5, 3)
        y = torch.randint(0, 3, (120,))
        with pytest.raises(ValueError, match="rolling window"):
            list(v.split(x, y))


# -----------------------------------------------------------------------
# ChampionRegistry and ModelStatus
# -----------------------------------------------------------------------


class TestModelStatus:
    def test_champion_is_production_eligible(self) -> None:
        assert ModelStatus.CHAMPION.is_production_eligible

    def test_challenger_is_production_eligible(self) -> None:
        assert ModelStatus.CHALLENGER.is_production_eligible

    def test_experimental_not_production_eligible(self) -> None:
        assert not ModelStatus.EXPERIMENTAL.is_production_eligible

    def test_retired_not_active(self) -> None:
        assert not ModelStatus.RETIRED.is_active

    def test_rank_ordering(self) -> None:
        assert ModelStatus.CHAMPION.rank > ModelStatus.CHALLENGER.rank
        assert ModelStatus.CHALLENGER.rank > ModelStatus.CANDIDATE.rank
        assert ModelStatus.CANDIDATE.rank > ModelStatus.EXPERIMENTAL.rank
        assert ModelStatus.RETIRED.rank == 0


class TestChampionRegistry:
    @pytest.fixture()
    def registry(self, tmp_path: Path) -> ChampionRegistry:
        return ChampionRegistry(store_path=tmp_path / "champions.json")

    def test_register_creates_experimental(self, registry: ChampionRegistry) -> None:
        entry = registry.register("quant_v1", "1.0.0")
        assert entry.status == ModelStatus.EXPERIMENTAL

    def test_promote_through_lifecycle(self, registry: ChampionRegistry) -> None:
        registry.register("quant_v1", "1.0.0")
        registry.promote("quant_v1", "1.0.0", ModelStatus.CANDIDATE)
        registry.promote("quant_v1", "1.0.0", ModelStatus.CHALLENGER)
        registry.promote("quant_v1", "1.0.0", ModelStatus.CHAMPION)
        champ = registry.get_champion("quant_v1")
        assert champ is not None
        assert champ.model_version == "1.0.0"

    def test_new_champion_retires_old(self, registry: ChampionRegistry) -> None:
        registry.register("quant_v1", "1.0.0")
        registry.promote("quant_v1", "1.0.0", ModelStatus.CHAMPION)
        registry.register("quant_v1", "2.0.0")
        registry.promote("quant_v1", "2.0.0", ModelStatus.CHAMPION)

        champ = registry.get_champion("quant_v1")
        assert champ.model_version == "2.0.0"

        all_entries = registry.list_all("quant_v1")
        old = next(e for e in all_entries if e.model_version == "1.0.0")
        assert old.status == ModelStatus.RETIRED

    def test_retire_terminates(self, registry: ChampionRegistry) -> None:
        registry.register("quant_v1", "0.9.0")
        registry.retire("quant_v1", "0.9.0", notes="superseded")
        entries = registry.list_all("quant_v1")
        assert entries[0].status == ModelStatus.RETIRED

    def test_promote_unknown_raises(self, registry: ChampionRegistry) -> None:
        with pytest.raises(KeyError):
            registry.promote("nonexistent", "1.0.0", ModelStatus.CANDIDATE)

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        p = tmp_path / "champs.json"
        r1 = ChampionRegistry(store_path=p)
        r1.register("quant_v1", "1.0.0", metrics={"f1_macro": 0.78})
        r1.promote("quant_v1", "1.0.0", ModelStatus.CHAMPION)

        r2 = ChampionRegistry(store_path=p)
        champ = r2.get_champion("quant_v1")
        assert champ is not None
        assert champ.model_version == "1.0.0"

    def test_summary_returns_list(self, registry: ChampionRegistry) -> None:
        registry.register("quant_v1", "1.0.0", metrics={"f1_macro": 0.72})
        rows = registry.summary()
        assert isinstance(rows, list)
        assert rows[0]["model_name"] == "quant_v1"


# -----------------------------------------------------------------------
# Report exporters
# -----------------------------------------------------------------------


class TestReportExporters:
    @pytest.fixture()
    def sample_data(self) -> dict:
        return {
            "title": "Test Report",
            "generated_at": "2024-07-04T12:00:00",
            "leaderboard": [
                {
                    "rank": 1,
                    "model_name": "quant_v1",
                    "architecture": "LSTM",
                    "accuracy": 0.75,
                    "f1_macro": 0.72,
                    "f1_weighted": 0.73,
                    "calibration_error": 0.08,
                    "inference_latency_ms": 3.2,
                    "training_time_seconds": 45.0,
                    "model_size_mb": 1.8,
                    "overall_score": 0.81,
                    "gpu_memory_mb": 0.0,
                    "status": "completed",
                },
            ],
            "conditions": {"random_seed": 42, "max_epochs": 10},
            "failure_analysis": [],
        }

    def test_json_export(self, sample_data, tmp_path: Path) -> None:
        from trainer.research.report.exporters import export_json

        p = tmp_path / "report.json"
        export_json(sample_data, p)
        assert p.exists()
        d = json.loads(p.read_text())
        assert d["title"] == "Test Report"

    def test_markdown_export(self, sample_data, tmp_path: Path) -> None:
        from trainer.research.report.exporters import export_markdown

        p = tmp_path / "report.md"
        export_markdown(sample_data, p)
        assert p.exists()
        text = p.read_text()
        assert "# Test Report" in text
        assert "Leaderboard" in text

    def test_html_export(self, sample_data, tmp_path: Path) -> None:
        from trainer.research.report.exporters import export_html

        p = tmp_path / "report.html"
        export_html(sample_data, p)
        assert p.exists()
        text = p.read_text()
        assert "<!DOCTYPE html>" in text
        assert "Test Report" in text

    def test_csv_export(self, sample_data, tmp_path: Path) -> None:
        from trainer.research.report.exporters import export_csv

        p = tmp_path / "report.csv"
        export_csv(sample_data, p)
        assert p.exists()
        text = p.read_text()
        assert "quant_v1" in text


# -----------------------------------------------------------------------
# ReportGenerator
# -----------------------------------------------------------------------


class TestReportGenerator:
    def test_generates_benchmark_report(self, tmp_path: Path) -> None:
        gen = ReportGenerator(
            ReportConfig(
                title="Unit Test Report",
                formats=["json", "markdown"],
                output_dir=tmp_path,
            )
        )
        result = _make_benchmark_result(2)
        lb = Leaderboard()
        lb.ingest(result)
        paths = gen.generate_benchmark_report(result, lb)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_generates_experiment_report(self, tmp_path: Path) -> None:
        gen = ReportGenerator(ReportConfig(formats=["json"], output_dir=tmp_path))
        rec = ExperimentRecord(name="unit-test", model_name="quant_v1")
        rec.complete({"accuracy": 0.78})
        paths = gen.generate_experiment_report(rec)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_generates_leaderboard_report(self, tmp_path: Path) -> None:
        lb = Leaderboard()
        lb.ingest(_make_benchmark_result(3))
        gen = ReportGenerator(ReportConfig(formats=["json", "csv"], output_dir=tmp_path))
        paths = gen.generate_leaderboard_report(lb)
        assert any(p.suffix == ".json" for p in paths)
        assert any(p.suffix == ".csv" for p in paths)
