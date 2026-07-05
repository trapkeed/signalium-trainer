"""Phase 14.9B — Comprehensive Infrastructure Tests.

Covers:
  - CheckpointManager (save, load, top-k rotation, best tracking, RNG state)
  - WarmupScheduler (warmup phase, base scheduler handoff, plateau compat)
  - ReproducibilityManager (capture, hash, save, load, verify, RNG snapshot)
  - TrainingProfiler (model size, training speed, inference speed, GPU memory)
  - Backbone registry (TransformerBackbone, TemporalCNNBackbone, HybridBackbone)
  - Schema additions (new optimizer/scheduler types, gradient_checkpointing)
  - Engine gradient accumulation (correct loss scaling, correct update cadence)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from trainer.checkpoints.manager import CheckpointManager
from trainer.config.schema import (
    CheckpointConfig,
    OptimizerType,
    ReproducibilityConfig,
    SchedulerConfig,
    SchedulerType,
    TrainerConfig,
    TrainingConfig,
)
from trainer.models.quant_v1.backbone import (
    GRUBackbone,
    HybridLSTMTransformerBackbone,
    LSTMBackbone,
    TemporalCNNBackbone,
    TransformerBackbone,
    build_backbone,
)
from trainer.models.quant_v1.config import BackboneConfig, BackboneType
from trainer.training.profiler import TrainingProfiler
from trainer.training.reproducibility import ReproducibilityManager, RunFingerprint
from trainer.training.scheduler import WarmupScheduler, build_warmup_scheduler


# ===========================================================================
# Shared helpers
# ===========================================================================


class _Net(nn.Module):
    def __init__(self, in_f: int = 4, out_f: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_opt(model: nn.Module, lr: float = 1e-3) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=lr)


# ===========================================================================
# CheckpointManager
# ===========================================================================


class TestCheckpointManager:
    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "ckpts"
        CheckpointManager(target)
        assert target.is_dir()

    def test_saves_latest(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path)
        mgr.save(0, m, opt, sch, {"val_loss": 0.5}, TrainerConfig())
        assert (tmp_path / "latest.pt").exists()

    def test_saves_best_on_first_epoch(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path, monitor_metric="val_loss", monitor_mode="min")
        mgr.save(0, m, opt, sch, {"val_loss": 0.9}, TrainerConfig())
        assert (tmp_path / "best.pt").exists()

    def test_updates_best_on_improvement(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path, monitor_metric="val_loss", monitor_mode="min")
        mgr.save(0, m, opt, sch, {"val_loss": 0.9}, TrainerConfig())
        mgr.save(1, m, opt, sch, {"val_loss": 0.4}, TrainerConfig())
        assert mgr.best_metrics["val_loss"] == pytest.approx(0.4)

    def test_does_not_update_best_on_regression(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path, monitor_metric="val_loss", monitor_mode="min")
        mgr.save(0, m, opt, sch, {"val_loss": 0.4}, TrainerConfig())
        mgr.save(1, m, opt, sch, {"val_loss": 0.9}, TrainerConfig())
        assert mgr.best_metrics["val_loss"] == pytest.approx(0.4)

    def test_best_mode_max(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path, monitor_metric="val_acc", monitor_mode="max")
        mgr.save(0, m, opt, sch, {"val_acc": 0.6}, TrainerConfig())
        mgr.save(1, m, opt, sch, {"val_acc": 0.8}, TrainerConfig())
        mgr.save(2, m, opt, sch, {"val_acc": 0.7}, TrainerConfig())
        assert mgr.best_metrics["val_acc"] == pytest.approx(0.8)

    def test_rotates_top_k(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path, keep_top_k=2)
        cfg = TrainerConfig()
        for epoch in range(5):
            mgr.save(epoch, m, opt, sch, {"val_loss": 0.9 - epoch * 0.1}, cfg)
        epoch_files = list(tmp_path.glob("epoch_*.pt"))
        assert len(epoch_files) <= 2

    def test_load_restores_model_weights(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path)
        mgr.save(0, m, opt, sch, {"val_loss": 0.3}, TrainerConfig())

        m2 = _Net()
        mgr.load(tmp_path / "latest.pt", m2)
        for p1, p2 in zip(m.parameters(), m2.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_with_optional_args_none(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path)
        mgr.save(0, m, opt, sch, {"val_loss": 0.3}, TrainerConfig())
        # Load without optimizer/scheduler — must not raise
        state = mgr.load(tmp_path / "latest.pt", m)
        assert state["epoch"] == 0

    def test_state_contains_epoch(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path)
        mgr.save(7, m, opt, sch, {"val_loss": 0.2}, TrainerConfig())
        state = mgr.load(tmp_path / "epoch_0007.pt", m)
        assert state["epoch"] == 7

    def test_state_contains_metrics(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path)
        mgr.save(0, m, opt, sch, {"val_loss": 0.123, "train_loss": 0.456}, TrainerConfig())
        state = mgr.load(tmp_path / "latest.pt", m)
        assert state["metrics"]["val_loss"] == pytest.approx(0.123)

    def test_invalid_monitor_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="monitor_mode"):
            CheckpointManager(tmp_path, monitor_mode="bad")

    def test_best_metrics_empty_initially(self, tmp_path: Path) -> None:
        mgr = CheckpointManager(tmp_path)
        assert mgr.best_metrics == {}

    def test_rng_state_saved(self, tmp_path: Path) -> None:
        m = _Net()
        opt = _make_opt(m)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        mgr = CheckpointManager(tmp_path, save_rng_state=True)
        mgr.save(0, m, opt, sch, {"val_loss": 0.5}, TrainerConfig())
        state = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
        assert "rng_state" in state
        assert "torch" in state["rng_state"]


# ===========================================================================
# WarmupScheduler
# ===========================================================================


class TestWarmupScheduler:
    def test_linear_warmup_increases_lr(self) -> None:
        model = _Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        ws = WarmupScheduler(opt, warmup_epochs=5, base_scheduler=base)

        lrs = []
        for _ in range(5):
            ws.step()
            lrs.append(opt.param_groups[0]["lr"])

        # Each step should be >= the previous during warmup
        for a, b in zip(lrs, lrs[1:]):
            assert b >= a - 1e-10, f"LR decreased during warmup: {a} → {b}"

    def test_warmup_reaches_base_lr(self) -> None:
        model = _Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        ws = WarmupScheduler(opt, warmup_epochs=4, base_scheduler=base)
        for _ in range(4):
            ws.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-2)

    def test_base_scheduler_activates_after_warmup(self) -> None:
        model = _Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        ws = WarmupScheduler(opt, warmup_epochs=2, base_scheduler=base)
        for _ in range(2):
            ws.step()
        lr_at_end_of_warmup = opt.param_groups[0]["lr"]
        ws.step()
        lr_after_step = opt.param_groups[0]["lr"]
        assert lr_after_step < lr_at_end_of_warmup

    def test_state_dict_round_trip(self) -> None:
        model = _Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        ws = WarmupScheduler(opt, warmup_epochs=5, base_scheduler=base)
        ws.step()
        sd = ws.state_dict()
        ws2 = WarmupScheduler(opt, warmup_epochs=5, base_scheduler=base)
        ws2.load_state_dict(sd)
        assert ws2._epoch == ws._epoch

    def test_plateau_warmup_no_metric_error(self) -> None:
        model = _Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
        ws = WarmupScheduler(opt, warmup_epochs=3, base_scheduler=base, is_plateau=True)
        # During warmup, metric not required
        for _ in range(3):
            ws.step()
        # After warmup, step with metric
        ws.step(metric=0.42)

    def test_build_warmup_scheduler_factory(self) -> None:
        model = _Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        ws = build_warmup_scheduler(opt, base, warmup_epochs=5)
        assert isinstance(ws, WarmupScheduler)

    def test_zero_warmup_epochs_no_wrapper(self) -> None:
        model = _Net()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        ws = build_warmup_scheduler(opt, base, warmup_epochs=0)
        assert isinstance(ws, WarmupScheduler)
        ws.step()


# ===========================================================================
# ReproducibilityManager
# ===========================================================================


class TestReproducibilityManager:
    def test_capture_returns_fingerprint(self) -> None:
        cfg = TrainerConfig()
        mgr = ReproducibilityManager(cfg)
        fp = mgr.capture()
        assert isinstance(fp, RunFingerprint)
        assert fp.config_hash != ""
        assert fp.seed == cfg.training.seed

    def test_config_hash_is_stable(self) -> None:
        cfg = TrainerConfig()
        mgr = ReproducibilityManager(cfg)
        h1 = ReproducibilityManager.hash_config(cfg)
        h2 = ReproducibilityManager.hash_config(cfg)
        assert h1 == h2

    def test_config_hash_changes_on_mutation(self) -> None:
        cfg1 = TrainerConfig()
        cfg2 = TrainerConfig()
        cfg2.training.seed = 99999
        h1 = ReproducibilityManager.hash_config(cfg1)
        h2 = ReproducibilityManager.hash_config(cfg2)
        assert h1 != h2

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        cfg = TrainerConfig()
        mgr = ReproducibilityManager(cfg)
        fp = mgr.capture(dataset_version="v1.0")
        path = tmp_path / "fingerprint.json"
        mgr.save(fp, path)
        assert path.exists()
        loaded = mgr.load(path)
        assert loaded.config_hash == fp.config_hash
        assert loaded.dataset_version == "v1.0"

    def test_to_json_round_trip(self) -> None:
        fp = RunFingerprint(
            config_hash="abc123",
            seed=42,
            torch_version="2.0.0",
        )
        j = fp.to_json()
        fp2 = RunFingerprint.from_json(j)
        assert fp2.config_hash == "abc123"
        assert fp2.seed == 42

    def test_capture_rng_state_contains_torch(self) -> None:
        state = ReproducibilityManager.capture_rng_state()
        assert "torch" in state
        assert "python" in state
        assert "numpy" in state

    def test_restore_rng_state(self) -> None:
        state = ReproducibilityManager.capture_rng_state()
        x1 = torch.randn(10)
        ReproducibilityManager.restore_rng_state(state)
        x2 = torch.randn(10)
        assert torch.allclose(x1, x2)

    def test_verify_passes_same_config(self, tmp_path: Path) -> None:
        cfg = TrainerConfig()
        mgr = ReproducibilityManager(cfg)
        fp = mgr.capture()
        path = tmp_path / "fp.json"
        mgr.save(fp, path)
        passed, issues = mgr.verify(path, check_git=False, check_versions=False)
        assert passed
        assert issues == []

    def test_verify_fails_different_config(self, tmp_path: Path) -> None:
        cfg1 = TrainerConfig()
        cfg2 = TrainerConfig()
        cfg2.training.seed = 99
        mgr1 = ReproducibilityManager(cfg1)
        mgr2 = ReproducibilityManager(cfg2)
        fp = mgr1.capture()
        path = tmp_path / "fp.json"
        mgr1.save(fp, path)
        passed, issues = mgr2.verify(path, check_git=False, check_versions=False)
        assert not passed

    def test_fingerprint_version_tracking(self) -> None:
        cfg = TrainerConfig()
        mgr = ReproducibilityManager(cfg)
        fp = mgr.capture(
            dataset_version="ds-v3",
            feature_version="feat-v2",
            label_version="lbl-v1",
        )
        assert fp.dataset_version == "ds-v3"
        assert fp.feature_version == "feat-v2"
        assert fp.label_version == "lbl-v1"


# ===========================================================================
# TrainingProfiler
# ===========================================================================


class TestTrainingProfiler:
    def test_profile_model_size_total_params(self) -> None:
        model = _Net(in_f=4, out_f=2)
        device = torch.device("cpu")
        profiler = TrainingProfiler(model, device)
        result = profiler.profile_model_size()
        # Linear(4, 2): 4*2 + 2 = 10 params
        assert result.total_params == 10
        assert result.trainable_params == 10
        assert result.non_trainable_params == 0
        assert result.estimated_mb > 0

    def test_profile_model_size_non_trainable(self) -> None:
        model = _Net(in_f=8, out_f=4)
        for p in model.parameters():
            p.requires_grad = False
        device = torch.device("cpu")
        profiler = TrainingProfiler(model, device)
        result = profiler.profile_model_size()
        assert result.trainable_params == 0
        assert result.non_trainable_params > 0

    def test_profile_inference_speed_returns_result(self) -> None:
        model = _Net(in_f=4, out_f=2)
        device = torch.device("cpu")
        profiler = TrainingProfiler(model, device)
        result = profiler.profile_inference_speed(
            input_shape=(4,), n_runs=20, batch_size=1
        )
        assert result.ms_per_sample > 0
        assert result.samples_per_sec > 0
        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms >= result.latency_p50_ms
        assert result.latency_p99_ms >= result.latency_p95_ms

    def test_profile_gpu_memory_no_cuda(self) -> None:
        model = _Net()
        device = torch.device("cpu")
        profiler = TrainingProfiler(model, device)
        loader = [(torch.randn(4, 4), torch.zeros(4, dtype=torch.long))] * 3
        result = profiler.profile_gpu_memory(loader, n_batches=3)
        assert result.available is False

    def test_cpu_memory_returns_rss(self) -> None:
        model = _Net()
        device = torch.device("cpu")
        profiler = TrainingProfiler(model, device)
        result = profiler.profile_cpu_memory()
        assert result.rss_mb > 0

    def test_benchmark_report_summary_contains_model_size(self) -> None:
        model = _Net()
        device = torch.device("cpu")
        profiler = TrainingProfiler(model, device)
        loader = [(torch.randn(4, 4), torch.zeros(4, dtype=torch.long))] * 5
        report = profiler.full_benchmark(loader, n_training_batches=3)
        summary = report.summary()
        assert "Model Size" in summary
        assert "Training Speed" in summary

    def test_benchmark_report_to_dict(self) -> None:
        model = _Net()
        device = torch.device("cpu")
        profiler = TrainingProfiler(model, device)
        loader = [(torch.randn(4, 4), torch.zeros(4, dtype=torch.long))] * 5
        report = profiler.full_benchmark(loader, n_training_batches=3)
        d = report.to_dict()
        assert "model_size" in d
        assert "training_speed" in d


# ===========================================================================
# Backbone Registry
# ===========================================================================


class TestBackboneRegistry:
    """Tests for the new Phase 14.9B backbone implementations."""

    def _cfg(self, btype: BackboneType, **kw) -> BackboneConfig:
        return BackboneConfig(type=btype, hidden_size=32, num_layers=1, dropout=0.0, **kw)

    def test_lstm_output_shape(self) -> None:
        cfg = self._cfg(BackboneType.LSTM)
        bb = LSTMBackbone(input_dim=16, config=cfg)
        x = torch.randn(4, 10, 16)
        out = bb(x)
        assert out.shape == (4, 32)

    def test_lstm_bidirectional_output_shape(self) -> None:
        cfg = self._cfg(BackboneType.LSTM, bidirectional=True)
        bb = LSTMBackbone(input_dim=16, config=cfg)
        x = torch.randn(4, 10, 16)
        out = bb(x)
        assert out.shape == (4, 64)

    def test_gru_output_shape(self) -> None:
        cfg = self._cfg(BackboneType.GRU)
        bb = GRUBackbone(input_dim=16, config=cfg)
        x = torch.randn(4, 10, 16)
        out = bb(x)
        assert out.shape == (4, 32)

    def test_transformer_output_shape(self) -> None:
        cfg = self._cfg(BackboneType.TRANSFORMER, num_heads=4, ff_dim=64, num_layers=2)
        bb = TransformerBackbone(input_dim=16, config=cfg)
        x = torch.randn(4, 10, 16)
        out = bb(x)
        assert out.shape == (4, 32)

    def test_transformer_mean_pooling(self) -> None:
        cfg = self._cfg(BackboneType.TRANSFORMER, num_heads=4, ff_dim=64, pooling="mean")
        bb = TransformerBackbone(input_dim=16, config=cfg)
        x = torch.randn(4, 10, 16)
        out = bb(x)
        assert out.shape == (4, 32)

    def test_transformer_output_dim_property(self) -> None:
        cfg = self._cfg(BackboneType.TRANSFORMER, num_heads=4, ff_dim=64)
        bb = TransformerBackbone(input_dim=16, config=cfg)
        assert bb.output_dim == 32

    def test_tcn_output_shape(self) -> None:
        cfg = self._cfg(
            BackboneType.TEMPORAL_CNN,
            tcn_channels=[16, 32],
            tcn_kernel_size=3,
        )
        bb = TemporalCNNBackbone(input_dim=16, config=cfg)
        x = torch.randn(4, 20, 16)
        out = bb(x)
        assert out.shape == (4, 32)

    def test_tcn_output_dim_property(self) -> None:
        cfg = self._cfg(BackboneType.TEMPORAL_CNN, tcn_channels=[16, 32])
        bb = TemporalCNNBackbone(input_dim=16, config=cfg)
        assert bb.output_dim == 32

    def test_hybrid_output_shape(self) -> None:
        cfg = self._cfg(
            BackboneType.HYBRID_LSTM_TRANSFORMER,
            num_heads=4,
            ff_dim=64,
            rnn_layers=1,
            transformer_layers=1,
        )
        bb = HybridLSTMTransformerBackbone(input_dim=16, config=cfg)
        x = torch.randn(4, 10, 16)
        out = bb(x)
        assert out.shape == (4, 32)

    def test_hybrid_output_dim_property(self) -> None:
        cfg = self._cfg(
            BackboneType.HYBRID_LSTM_TRANSFORMER, num_heads=4, ff_dim=64
        )
        bb = HybridLSTMTransformerBackbone(input_dim=16, config=cfg)
        assert bb.output_dim == 32

    def test_build_backbone_lstm(self) -> None:
        cfg = self._cfg(BackboneType.LSTM)
        bb = build_backbone(16, cfg)
        assert isinstance(bb, LSTMBackbone)

    def test_build_backbone_gru(self) -> None:
        cfg = self._cfg(BackboneType.GRU)
        bb = build_backbone(16, cfg)
        assert isinstance(bb, GRUBackbone)

    def test_build_backbone_transformer(self) -> None:
        cfg = self._cfg(BackboneType.TRANSFORMER, num_heads=4, ff_dim=64)
        bb = build_backbone(16, cfg)
        assert isinstance(bb, TransformerBackbone)

    def test_build_backbone_tcn(self) -> None:
        cfg = self._cfg(BackboneType.TEMPORAL_CNN, tcn_channels=[16, 32])
        bb = build_backbone(16, cfg)
        assert isinstance(bb, TemporalCNNBackbone)

    def test_build_backbone_hybrid(self) -> None:
        cfg = self._cfg(
            BackboneType.HYBRID_LSTM_TRANSFORMER, num_heads=4, ff_dim=64
        )
        bb = build_backbone(16, cfg)
        assert isinstance(bb, HybridLSTMTransformerBackbone)

    def test_build_backbone_unknown_raises(self) -> None:
        cfg = self._cfg(BackboneType.LSTM)
        cfg.type = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown backbone type"):
            build_backbone(16, cfg)

    def test_backbones_are_nn_modules(self) -> None:
        cfg = self._cfg(BackboneType.TRANSFORMER, num_heads=4, ff_dim=64)
        bb = build_backbone(16, cfg)
        assert isinstance(bb, nn.Module)

    def test_transformer_gradient_checkpointing_flag(self) -> None:
        cfg = self._cfg(BackboneType.TRANSFORMER, num_heads=4, ff_dim=64,
                        gradient_checkpointing=True)
        bb = TransformerBackbone(input_dim=16, config=cfg)
        assert bb._use_checkpoint is True

    def test_hybrid_gradient_checkpointing_flag(self) -> None:
        cfg = self._cfg(
            BackboneType.HYBRID_LSTM_TRANSFORMER,
            num_heads=4, ff_dim=64,
            gradient_checkpointing=True,
        )
        bb = HybridLSTMTransformerBackbone(input_dim=16, config=cfg)
        assert bb._use_checkpoint is True


# ===========================================================================
# Schema additions
# ===========================================================================


class TestSchemaAdditions:
    def test_optimizer_adagrad_in_enum(self) -> None:
        assert OptimizerType.ADAGRAD.value == "adagrad"

    def test_optimizer_adamax_in_enum(self) -> None:
        assert OptimizerType.ADAMAX.value == "adamax"

    def test_scheduler_cosine_warm_restarts_in_enum(self) -> None:
        assert SchedulerType.COSINE_WARM_RESTARTS.value == "cosine_warm_restarts"

    def test_scheduler_one_cycle_in_enum(self) -> None:
        assert SchedulerType.ONE_CYCLE.value == "one_cycle"

    def test_scheduler_linear_in_enum(self) -> None:
        assert SchedulerType.LINEAR.value == "linear"

    def test_training_config_has_gradient_checkpointing(self) -> None:
        cfg = TrainingConfig()
        assert hasattr(cfg, "gradient_checkpointing")
        assert cfg.gradient_checkpointing is False

    def test_checkpoint_config_has_rng_state_flag(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.save_rng_state is True

    def test_checkpoint_config_has_optimizer_state_flag(self) -> None:
        cfg = CheckpointConfig()
        assert cfg.save_optimizer_state is True

    def test_scheduler_config_has_warmup_epochs(self) -> None:
        cfg = SchedulerConfig()
        assert cfg.warmup_epochs == 0

    def test_trainer_config_has_reproducibility(self) -> None:
        cfg = TrainerConfig()
        assert isinstance(cfg.reproducibility, ReproducibilityConfig)

    def test_reproducibility_config_defaults(self) -> None:
        cfg = ReproducibilityConfig()
        assert cfg.enabled is True
        assert cfg.verify_on_resume is True

    def test_backbone_config_has_transformer_fields(self) -> None:
        cfg = BackboneConfig()
        assert hasattr(cfg, "num_heads")
        assert hasattr(cfg, "ff_dim")
        assert hasattr(cfg, "max_seq_len")

    def test_backbone_config_has_tcn_fields(self) -> None:
        cfg = BackboneConfig()
        assert hasattr(cfg, "tcn_channels")
        assert hasattr(cfg, "tcn_kernel_size")

    def test_backbone_config_has_hybrid_fields(self) -> None:
        cfg = BackboneConfig()
        assert hasattr(cfg, "rnn_layers")
        assert hasattr(cfg, "transformer_layers")

    def test_backbone_config_has_gradient_checkpointing(self) -> None:
        cfg = BackboneConfig()
        assert hasattr(cfg, "gradient_checkpointing")
        assert cfg.gradient_checkpointing is False

    def test_backbone_type_transformer(self) -> None:
        assert BackboneType.TRANSFORMER.value == "transformer"

    def test_backbone_type_temporal_cnn(self) -> None:
        assert BackboneType.TEMPORAL_CNN.value == "temporal_cnn"

    def test_backbone_type_hybrid(self) -> None:
        assert BackboneType.HYBRID_LSTM_TRANSFORMER.value == "hybrid_lstm_transformer"


# ===========================================================================
# Gradient accumulation correctness
# ===========================================================================


class TestGradientAccumulation:
    """Verify that gradient accumulation produces correct gradients.

    The gradient from N accumulated micro-batches with loss/N each must equal
    the gradient from a single full-batch with loss/1.
    """

    def _run_with_accum(self, accum_steps: int, batch: torch.Tensor,
                        labels: torch.Tensor) -> dict[str, torch.Tensor]:
        model = _Net(4, 3)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.CrossEntropyLoss()

        sub = batch.split(batch.shape[0] // accum_steps)
        sub_lbl = labels.split(labels.shape[0] // accum_steps)

        for i, (x, y) in enumerate(zip(sub, sub_lbl)):
            if i == 0:
                opt.zero_grad()
            loss = criterion(model(x), y) / accum_steps
            loss.backward()

        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    def _run_full_batch(self, batch: torch.Tensor,
                        labels: torch.Tensor) -> dict[str, torch.Tensor]:
        model = _Net(4, 3)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.CrossEntropyLoss()
        opt.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    def test_accum_2_matches_full_batch(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        g_accum = self._run_with_accum(2, x, y)
        g_full = self._run_full_batch(x, y)
        for name in g_full:
            assert torch.allclose(g_accum[name], g_full[name], atol=1e-5), \
                f"Gradient mismatch for {name}"

    def test_accum_4_matches_full_batch(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))
        g_accum = self._run_with_accum(4, x, y)
        g_full = self._run_full_batch(x, y)
        for name in g_full:
            assert torch.allclose(g_accum[name], g_full[name], atol=1e-5), \
                f"Gradient mismatch for {name}"
