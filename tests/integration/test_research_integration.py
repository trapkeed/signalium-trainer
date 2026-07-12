"""Integration tests — Research & Benchmarking Framework end-to-end.

Verifies:
- BenchmarkRunner trains multiple models and collects metrics
- Leaderboard ranks correctly after a real benchmark run
- ReportGenerator produces all output formats
- Full lifecycle: register → train → record → rank → promote
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
pytestmark = pytest.mark.requires_torch
import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.framework import ModelFactory
from trainer.models.quant_v1.config import (
    BackboneConfig,
    BackboneType,
    ClassificationHeadConfig,
    ConfidenceHeadConfig,
    InputEncoderConfig,
    QuantV1Config,
)
from trainer.research import (
    BenchmarkConditions,
    BenchmarkRunner,
    ChampionRegistry,
    ExperimentTracker,
    Leaderboard,
    ModelStatus,
    ReportConfig,
    ReportGenerator,
    RollingWindowValidator,
    ScoringWeights,
    WalkForwardValidator,
)

BATCH = 64
SEQ_LEN = 20
INPUT_DIM = 16
NUM_CLASSES = 3
N_TRAIN = 200
N_VAL = 60


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_quant_v1_cfg() -> QuantV1Config:
    return QuantV1Config(
        encoder=InputEncoderConfig(input_dim=INPUT_DIM, hidden_dim=32, projection_dropout=0.0),
        backbone=BackboneConfig(
            type=BackboneType.LSTM,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            window_length=SEQ_LEN,
        ),
        classification_head=ClassificationHeadConfig(
            num_classes=NUM_CLASSES, hidden_dim=32, dropout=0.0
        ),
        confidence_head=ConfidenceHeadConfig(hidden_dim=16, dropout=0.0),
    )


@pytest.fixture(scope="module")
def synthetic_train() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    return torch.randn(N_TRAIN, SEQ_LEN, INPUT_DIM), torch.randint(0, NUM_CLASSES, (N_TRAIN,))


@pytest.fixture(scope="module")
def synthetic_val() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(99)
    return torch.randn(N_VAL, SEQ_LEN, INPUT_DIM), torch.randint(0, NUM_CLASSES, (N_VAL,))


# -----------------------------------------------------------------------
# BenchmarkRunner integration
# -----------------------------------------------------------------------


class TestBenchmarkRunnerIntegration:
    def test_single_model_run_completes(
        self,
        small_quant_v1_cfg: QuantV1Config,
        synthetic_train,
        synthetic_val,
    ) -> None:
        conditions = BenchmarkConditions(random_seed=42, max_epochs=3, batch_size=32, device="cpu")
        runner = BenchmarkRunner(conditions=conditions)
        runner.register_model(
            name="quant_v1_lstm",
            model_factory=lambda: ModelFactory.create(
                "quant_v1", TrainerConfig(), small_quant_v1_cfg
            ),
        )
        result = runner.run(synthetic_train, synthetic_val)
        assert len(result.runs) == 1
        run = result.runs[0]
        assert run.status == "completed"
        assert "accuracy" in run.metrics
        assert run.training_time_seconds > 0
        assert run.inference_latency_ms > 0
        assert run.model_size_mb > 0

    def test_two_model_comparison(
        self,
        small_quant_v1_cfg: QuantV1Config,
        synthetic_train,
        synthetic_val,
    ) -> None:
        gru_cfg = QuantV1Config(
            encoder=InputEncoderConfig(input_dim=INPUT_DIM, hidden_dim=32),
            backbone=BackboneConfig(
                type=BackboneType.GRU,
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
                window_length=SEQ_LEN,
            ),
            classification_head=ClassificationHeadConfig(num_classes=NUM_CLASSES, hidden_dim=32),
            confidence_head=ConfidenceHeadConfig(hidden_dim=16),
        )

        conditions = BenchmarkConditions(random_seed=42, max_epochs=2, batch_size=32)
        runner = BenchmarkRunner(conditions=conditions)
        runner.register_model(
            "quant_v1_lstm",
            lambda: ModelFactory.create("quant_v1", TrainerConfig(), small_quant_v1_cfg),
        )
        runner.register_model(
            "quant_v1_gru",
            lambda: ModelFactory.create("quant_v1", TrainerConfig(), gru_cfg),
        )

        result = runner.run(synthetic_train, synthetic_val)
        assert len(result.completed_runs()) == 2
        assert result.total_duration_seconds > 0
        assert result.to_dict()["benchmark_id"] == result.benchmark_id

    def test_failed_model_does_not_crash_runner(self, synthetic_train, synthetic_val) -> None:
        conditions = BenchmarkConditions(max_epochs=1, batch_size=32)
        runner = BenchmarkRunner(conditions=conditions)

        def bad_factory():
            raise RuntimeError("Intentional failure")

        runner.register_model("broken_model", bad_factory)
        result = runner.run(synthetic_train, synthetic_val)
        assert len(result.runs) == 1
        assert result.runs[0].status == "failed"
        assert result.runs[0].error is not None


# -----------------------------------------------------------------------
# Leaderboard integration
# -----------------------------------------------------------------------


class TestLeaderboardIntegration:
    def test_leaderboard_from_benchmark_run(
        self,
        small_quant_v1_cfg,
        synthetic_train,
        synthetic_val,
    ) -> None:
        conditions = BenchmarkConditions(max_epochs=2, batch_size=32)
        runner = BenchmarkRunner(conditions=conditions)
        runner.register_model(
            "quant_v1_lstm",
            lambda: ModelFactory.create("quant_v1", TrainerConfig(), small_quant_v1_cfg),
        )
        result = runner.run(synthetic_train, synthetic_val)

        lb = Leaderboard()
        lb.ingest(result)

        assert lb.champion() is not None
        assert lb.champion().rank == 1
        table = lb.format_table()
        assert "quant_v1_lstm" in table

    def test_custom_weights_change_ranking(self, synthetic_train, synthetic_val) -> None:
        import uuid

        from trainer.research.benchmark.runner import BenchmarkResult, ModelRunResult

        result = BenchmarkResult(
            benchmark_id=str(uuid.uuid4()),
            conditions=BenchmarkConditions(),
            runs=[
                ModelRunResult(
                    "fast_model",
                    metrics={
                        "accuracy": 0.70,
                        "f1_macro": 0.68,
                        "f1_weighted": 0.69,
                        "inference_latency_ms": 1.0,
                    },
                    training_time_seconds=5.0,
                    status="completed",
                ),
                ModelRunResult(
                    "accurate_model",
                    metrics={
                        "accuracy": 0.85,
                        "f1_macro": 0.83,
                        "f1_weighted": 0.84,
                        "inference_latency_ms": 50.0,
                    },
                    training_time_seconds=300.0,
                    status="completed",
                ),
            ],
        )

        lb_speed = Leaderboard(
            ScoringWeights(
                accuracy=0.0,
                f1_macro=0.0,
                f1_weighted=0.0,
                calibration_error=0.0,
                inference_speed=1.0,
            )
        )
        lb_speed.ingest(result)
        assert lb_speed.champion().model_name == "fast_model"

        lb_acc = Leaderboard(
            ScoringWeights(
                accuracy=1.0,
                f1_macro=0.0,
                f1_weighted=0.0,
                calibration_error=0.0,
                inference_speed=0.0,
            )
        )
        lb_acc.ingest(result)
        assert lb_acc.champion().model_name == "accurate_model"


# -----------------------------------------------------------------------
# Full lifecycle integration
# -----------------------------------------------------------------------


class TestFullLifecycleIntegration:
    def test_train_record_promote(
        self,
        tmp_path: Path,
        small_quant_v1_cfg,
        synthetic_train,
        synthetic_val,
    ) -> None:
        tracker = ExperimentTracker(store_path=tmp_path / "records.json")
        champion_registry = ChampionRegistry(store_path=tmp_path / "champions.json")

        conditions = BenchmarkConditions(max_epochs=2, batch_size=32)
        runner = BenchmarkRunner(conditions=conditions, experiment_tracker=tracker)
        runner.register_model(
            "quant_v1",
            lambda: ModelFactory.create("quant_v1", TrainerConfig(), small_quant_v1_cfg),
        )
        result = runner.run(synthetic_train, synthetic_val)

        assert len(tracker.load_all()) == 1
        run = result.runs[0]
        assert run.status == "completed"

        champion_registry.register(
            "quant_v1",
            "1.0.0",
            experiment_id=run.experiment_id,
            metrics=run.metrics,
        )
        champion_registry.promote("quant_v1", "1.0.0", ModelStatus.CANDIDATE)
        champion_registry.promote(
            "quant_v1", "1.0.0", ModelStatus.CHAMPION, notes="Baseline established"
        )

        champ = champion_registry.get_champion("quant_v1")
        assert champ is not None
        assert champ.status == ModelStatus.CHAMPION

    def test_full_report_generation(
        self,
        tmp_path: Path,
        small_quant_v1_cfg,
        synthetic_train,
        synthetic_val,
    ) -> None:
        conditions = BenchmarkConditions(max_epochs=2, batch_size=32)
        runner = BenchmarkRunner(conditions=conditions)
        runner.register_model(
            "quant_v1",
            lambda: ModelFactory.create("quant_v1", TrainerConfig(), small_quant_v1_cfg),
        )
        result = runner.run(synthetic_train, synthetic_val)

        lb = Leaderboard()
        lb.ingest(result)

        gen = ReportGenerator(
            ReportConfig(
                title="Integration Test",
                formats=["json", "markdown", "html", "csv"],
                output_dir=tmp_path / "reports",
            )
        )
        paths = gen.generate_benchmark_report(result, lb)
        assert len(paths) == 4
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

        json_path = next(p for p in paths if p.suffix == ".json")
        d = json.loads(json_path.read_text())
        assert "leaderboard" in d
        assert "conditions" in d


# -----------------------------------------------------------------------
# Cross-validation integration
# -----------------------------------------------------------------------


class TestCrossValidationIntegration:
    def test_walk_forward_trains_on_each_fold(self, small_quant_v1_cfg) -> None:
        n = 300
        x = torch.randn(n, SEQ_LEN, INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, (n,))

        validator = WalkForwardValidator(min_train_size=100, val_size=50, step_size=50, max_folds=3)
        fold_metrics = []

        for fold in validator.split(x, y):
            model = ModelFactory.create("quant_v1", TrainerConfig(), small_quant_v1_cfg)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for _ in range(2):
                optimizer.zero_grad()
                logits = model(fold.x_train[:32])
                loss = criterion(logits, fold.y_train[:32])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(fold.x_val)
                preds = val_logits.argmax(dim=-1)
                acc = (preds == fold.y_val).float().mean().item()
            fold_metrics.append({"fold": fold.fold_index, "accuracy": acc})

        assert len(fold_metrics) == 3
        assert all("accuracy" in m for m in fold_metrics)

    def test_rolling_window_correct_sizes(self) -> None:
        n = 200
        x = torch.randn(n, SEQ_LEN, INPUT_DIM)
        y = torch.randint(0, NUM_CLASSES, (n,))

        validator = RollingWindowValidator(train_size=80, val_size=20, step_size=20)
        for fold in validator.split(x, y):
            assert fold.x_train.shape[0] == 80
            assert fold.x_val.shape[0] == 20
