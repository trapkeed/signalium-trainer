"""Unit tests for Phase 14.6 Institutional Trade Analytics heads.

Tests cover:
  - All 15 new prediction heads
  - Output shapes and value ranges
  - Non-negativity constraints (Softplus-activated heads)
  - Bounded outputs (Sigmoid-based heads in [0, 1] or [0, 100])
  - Gradient flow through every head
  - MultiTaskLoss with all Phase 14.6 terms
  - Multi-task forward pass completeness
  - Backward compatibility (forward() still returns logits)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from trainer.losses.advanced import (
    DistributionLoss,
    DynamicLossWeighting,
    GaussianNLLLoss,
    QuantileLoss,
)
from trainer.losses.multi_task import MultiTaskLoss, MultiTaskLossOutput
from trainer.models.quant_v1.config import (
    DistributionHeadConfig,
    ExcursionHeadConfig,
    InstitutionalHeadConfig,
    MarketStabilityHeadConfig,
    MultiTaskLossConfig,
    QuantV1Config,
    RegressionUnit,
    TradeDecayHeadConfig,
    TradeSurvivalHeadConfig,
    UncertaintyHeadConfig,
)
from trainer.models.quant_v1.head import (
    EntryQualityHead,
    ExitQualityHead,
    LossDistributionHead,
    MAEExcursionHead,
    MarketStabilityHead,
    MFEHead,
    PositionSizeQualityHead,
    ProfitDistributionHead,
    RewardRiskHead,
    SLResilienceHead,
    TPAchievabilityHead,
    TradeDecayHead,
    TradeEfficiencyHead,
    TradeSurvivalHead,
    UncertaintyHead,
)

BATCH = 8
CTX_DIM = 64


@pytest.fixture
def context() -> torch.Tensor:
    return torch.randn(BATCH, CTX_DIM)


# ---------------------------------------------------------------------------
# Phase 14.6 Head output shape and range tests
# ---------------------------------------------------------------------------


class TestMFEHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = MFEHead(CTX_DIM, ExcursionHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_nonnegative(self, context: torch.Tensor) -> None:
        head = MFEHead(CTX_DIM, ExcursionHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all(), "MFE must be non-negative (Softplus)"

    def test_gradient(self, context: torch.Tensor) -> None:
        ctx = context.requires_grad_(True)
        head = MFEHead(CTX_DIM, ExcursionHeadConfig(enabled=True))
        loss = head(ctx).sum()
        loss.backward()
        assert ctx.grad is not None


class TestMAEExcursionHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = MAEExcursionHead(CTX_DIM, ExcursionHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_nonnegative(self, context: torch.Tensor) -> None:
        head = MAEExcursionHead(CTX_DIM, ExcursionHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all(), "MAE excursion must be non-negative"


class TestTradeEfficiencyHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = TradeEfficiencyHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_range(self, context: torch.Tensor) -> None:
        head = TradeEfficiencyHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all() and (out <= 100).all(), "Efficiency must be in [0, 100]"


class TestRewardRiskHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = RewardRiskHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_nonnegative(self, context: torch.Tensor) -> None:
        head = RewardRiskHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all(), "R:R ratio must be non-negative"


class TestTradeSurvivalHead:
    def test_shape_default_bars(self, context: torch.Tensor) -> None:
        cfg = TradeSurvivalHeadConfig(enabled=True, survival_bars=[5, 10, 20])
        head = TradeSurvivalHead(CTX_DIM, cfg)
        out = head(context)
        assert out.shape == (BATCH, 3)

    def test_shape_custom_bars(self, context: torch.Tensor) -> None:
        cfg = TradeSurvivalHeadConfig(enabled=True, survival_bars=[5, 10, 15, 20, 30])
        head = TradeSurvivalHead(CTX_DIM, cfg)
        out = head(context)
        assert out.shape == (BATCH, 5)

    def test_probability_range(self, context: torch.Tensor) -> None:
        cfg = TradeSurvivalHeadConfig(enabled=True, survival_bars=[5, 10, 20])
        head = TradeSurvivalHead(CTX_DIM, cfg)
        out = head(context)
        assert (out >= 0).all() and (out <= 1).all(), "Survival probs must be in [0, 1]"

    def test_survival_bars_property(self) -> None:
        cfg = TradeSurvivalHeadConfig(enabled=True, survival_bars=[5, 10, 20])
        head = TradeSurvivalHead(CTX_DIM, cfg)
        assert head.survival_bars == [5, 10, 20]


class TestSLResilienceHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = SLResilienceHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_probability_range(self, context: torch.Tensor) -> None:
        head = SLResilienceHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all() and (out <= 1).all()


class TestTPAchievabilityHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = TPAchievabilityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_probability_range(self, context: torch.Tensor) -> None:
        head = TPAchievabilityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all() and (out <= 1).all()


class TestEntryExitQualityHeads:
    def test_entry_shape(self, context: torch.Tensor) -> None:
        head = EntryQualityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_entry_range(self, context: torch.Tensor) -> None:
        head = EntryQualityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all() and (out <= 100).all(), "Entry quality must be in [0, 100]"

    def test_exit_shape(self, context: torch.Tensor) -> None:
        head = ExitQualityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_exit_range(self, context: torch.Tensor) -> None:
        head = ExitQualityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all() and (out <= 100).all(), "Exit quality must be in [0, 100]"


class TestMarketStabilityHead:
    def test_logit_shape(self, context: torch.Tensor) -> None:
        head = MarketStabilityHead(CTX_DIM, MarketStabilityHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 4)

    def test_proba_shape(self, context: torch.Tensor) -> None:
        head = MarketStabilityHead(CTX_DIM, MarketStabilityHeadConfig(enabled=True))
        probs = head.predict_proba(context)
        assert probs.shape == (BATCH, 4)

    def test_proba_sums_to_one(self, context: torch.Tensor) -> None:
        head = MarketStabilityHead(CTX_DIM, MarketStabilityHeadConfig(enabled=True))
        probs = head.predict_proba(context)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_label_names(self) -> None:
        head = MarketStabilityHead(CTX_DIM, MarketStabilityHeadConfig(enabled=True))
        assert len(head.label_names) == 4


class TestTradeDecayHead:
    def test_shape_default_steps(self, context: torch.Tensor) -> None:
        cfg = TradeDecayHeadConfig(enabled=True, decay_steps=5)
        head = TradeDecayHead(CTX_DIM, cfg)
        out = head(context)
        assert out.shape == (BATCH, 6)  # 5 steps + 1 overall prob

    def test_shape_custom_steps(self, context: torch.Tensor) -> None:
        cfg = TradeDecayHeadConfig(enabled=True, decay_steps=10)
        head = TradeDecayHead(CTX_DIM, cfg)
        out = head(context)
        assert out.shape == (BATCH, 11)

    def test_probability_range(self, context: torch.Tensor) -> None:
        cfg = TradeDecayHeadConfig(enabled=True, decay_steps=5)
        head = TradeDecayHead(CTX_DIM, cfg)
        out = head(context)
        assert (out >= 0).all() and (out <= 1).all(), "Decay outputs must be in [0, 1]"


class TestProfitDistributionHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = ProfitDistributionHead(CTX_DIM, DistributionHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 6)

    def test_variance_nonnegative(self, context: torch.Tensor) -> None:
        head = ProfitDistributionHead(CTX_DIM, DistributionHeadConfig(enabled=True))
        out = head(context)
        variance = out[:, ProfitDistributionHead.VAR_IDX]
        assert (variance >= 0).all(), "Variance must be non-negative (Softplus)"

    def test_gradient(self, context: torch.Tensor) -> None:
        ctx = context.requires_grad_(True)
        head = ProfitDistributionHead(CTX_DIM, DistributionHeadConfig(enabled=True))
        loss = head(ctx).sum()
        loss.backward()
        assert ctx.grad is not None


class TestLossDistributionHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = LossDistributionHead(CTX_DIM, DistributionHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 4)

    def test_variance_nonnegative(self, context: torch.Tensor) -> None:
        head = LossDistributionHead(CTX_DIM, DistributionHeadConfig(enabled=True))
        out = head(context)
        variance = out[:, LossDistributionHead.VAR_IDX]
        assert (variance >= 0).all()

    def test_tail_risk_nonnegative(self, context: torch.Tensor) -> None:
        head = LossDistributionHead(CTX_DIM, DistributionHeadConfig(enabled=True))
        out = head(context)
        tail_risk = out[:, LossDistributionHead.TAIL_IDX]
        assert (tail_risk >= 0).all(), "Tail risk must be non-negative (Softplus)"


class TestPositionSizeQualityHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = PositionSizeQualityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 1)

    def test_range(self, context: torch.Tensor) -> None:
        head = PositionSizeQualityHead(CTX_DIM, InstitutionalHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all() and (out <= 100).all(), "PS quality must be in [0, 100]"


class TestUncertaintyHead:
    def test_shape(self, context: torch.Tensor) -> None:
        head = UncertaintyHead(CTX_DIM, UncertaintyHeadConfig(enabled=True))
        out = head(context)
        assert out.shape == (BATCH, 2)

    def test_nonnegative(self, context: torch.Tensor) -> None:
        head = UncertaintyHead(CTX_DIM, UncertaintyHeadConfig(enabled=True))
        out = head(context)
        assert (out >= 0).all(), "Uncertainty estimates must be non-negative (Softplus)"

    def test_output_indices(self, context: torch.Tensor) -> None:
        head = UncertaintyHead(CTX_DIM, UncertaintyHeadConfig(enabled=True))
        out = head(context)
        epistemic = out[:, UncertaintyHead.EPISTEMIC_IDX]
        aleatoric = out[:, UncertaintyHead.ALEATORIC_IDX]
        assert epistemic.shape == (BATCH,)
        assert aleatoric.shape == (BATCH,)


# ---------------------------------------------------------------------------
# Advanced loss functions
# ---------------------------------------------------------------------------


class TestQuantileLoss:
    def test_median_loss(self) -> None:
        loss_fn = QuantileLoss(quantile=0.5)
        pred = torch.zeros(BATCH)
        target = torch.ones(BATCH)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.5, abs=1e-5)

    def test_zero_loss_on_perfect(self) -> None:
        loss_fn = QuantileLoss(quantile=0.5)
        pred = torch.ones(BATCH) * 3.14
        target = torch.ones(BATCH) * 3.14
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_asymmetry(self) -> None:
        q = 0.9
        loss_fn = QuantileLoss(quantile=q)
        pred = torch.zeros(1)
        target_up = torch.ones(1)
        target_down = -torch.ones(1)
        loss_up = loss_fn(pred, target_up)
        loss_down = loss_fn(pred, target_down)
        # Under-prediction is more costly at high quantile
        assert loss_up.item() > loss_down.item()

    def test_invalid_quantile(self) -> None:
        with pytest.raises(ValueError):
            QuantileLoss(quantile=1.5)


class TestGaussianNLLLoss:
    def test_scalar_output(self) -> None:
        loss_fn = GaussianNLLLoss()
        mu = torch.zeros(BATCH, 1)
        log_var = torch.zeros(BATCH, 1)
        target = torch.randn(BATCH, 1)
        loss = loss_fn(mu, log_var, target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flows(self) -> None:
        loss_fn = GaussianNLLLoss()
        mu = torch.zeros(BATCH, 1, requires_grad=True)
        log_var = torch.zeros(BATCH, 1, requires_grad=True)
        target = torch.ones(BATCH, 1)
        loss = loss_fn(mu, log_var, target)
        loss.backward()
        assert mu.grad is not None
        assert log_var.grad is not None


class TestDistributionLoss:
    def test_profit_dist_loss(self) -> None:
        loss_fn = DistributionLoss(n_outputs=6)
        pred = torch.randn(BATCH, 6)
        target = torch.randn(BATCH, 6)
        # Force non-negative variance in target
        target[:, 2] = target[:, 2].abs()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_loss_dist_loss(self) -> None:
        loss_fn = DistributionLoss(n_outputs=4)
        pred = torch.randn(BATCH, 4)
        target = torch.randn(BATCH, 4)
        target[:, 2] = target[:, 2].abs()
        target[:, 3] = target[:, 3].abs()
        loss = loss_fn(pred, target)
        assert loss.shape == ()

    def test_invalid_n_outputs(self) -> None:
        with pytest.raises(ValueError):
            DistributionLoss(n_outputs=5)


class TestDynamicLossWeighting:
    def test_weighted_output_shape(self) -> None:
        dwl = DynamicLossWeighting(n_tasks=3)
        losses = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.8)]
        total = dwl(losses)
        assert total.shape == ()

    def test_task_weights_positive(self) -> None:
        dwl = DynamicLossWeighting(n_tasks=5)
        weights = dwl.task_weights
        assert (weights > 0).all()
        assert weights.shape == (5,)

    def test_wrong_n_losses(self) -> None:
        dwl = DynamicLossWeighting(n_tasks=3)
        with pytest.raises(ValueError):
            dwl([torch.tensor(0.5), torch.tensor(0.3)])

    def test_gradients_flow_to_log_vars(self) -> None:
        dwl = DynamicLossWeighting(n_tasks=2)
        losses = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)]
        total = dwl(losses)
        total.backward()
        assert dwl.log_vars.grad is not None


# ---------------------------------------------------------------------------
# Full model integration
# ---------------------------------------------------------------------------


def _make_full_config() -> QuantV1Config:
    """Create a QuantV1Config with all Phase 14.6 heads enabled."""
    from trainer.models.quant_v1.config import (
        BackboneConfig,
        ClassificationHeadConfig,
        ConfidenceHeadConfig,
        InputEncoderConfig,
        RegressionHeadConfig,
        TPProbabilityHeadConfig,
    )

    cfg = QuantV1Config()
    cfg.encoder = InputEncoderConfig(input_dim=16, hidden_dim=32)
    cfg.backbone = BackboneConfig(hidden_size=32, num_layers=1, window_length=4)
    cfg.classification_head = ClassificationHeadConfig(hidden_dim=32, dropout=0.0)
    cfg.confidence_head = ConfidenceHeadConfig(hidden_dim=16, dropout=0.0)
    # Phase 14.5
    cfg.expected_move_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.stop_loss_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.take_profit_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.holding_time_head = RegressionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.tp_probability_head = TPProbabilityHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    # Phase 14.6
    cfg.mfe_head = ExcursionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.mae_excursion_head = ExcursionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.trade_efficiency_head = InstitutionalHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.reward_risk_head = InstitutionalHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.trade_survival_head = TradeSurvivalHeadConfig(
        enabled=True, hidden_dim=16, dropout=0.0, survival_bars=[5, 10, 20]
    )
    cfg.sl_resilience_head = InstitutionalHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.tp_achievability_head = InstitutionalHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.entry_quality_head = InstitutionalHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.exit_quality_head = InstitutionalHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.market_stability_head = MarketStabilityHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.trade_decay_head = TradeDecayHeadConfig(
        enabled=True, hidden_dim=16, dropout=0.0, decay_steps=5
    )
    cfg.profit_distribution_head = DistributionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.loss_distribution_head = DistributionHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    cfg.position_size_quality_head = InstitutionalHeadConfig(
        enabled=True, hidden_dim=16, dropout=0.0
    )
    cfg.uncertainty_head = UncertaintyHeadConfig(enabled=True, hidden_dim=16, dropout=0.0)
    return cfg


@pytest.fixture
def full_model():
    """QuantModelV1 with all heads enabled and minimal size."""
    from unittest.mock import MagicMock

    from trainer.models.quant_v1.model import QuantModelV1

    trainer_config = MagicMock()
    model_config = _make_full_config()
    model = QuantModelV1(trainer_config, model_config)
    model.eval()
    return model


class TestFullModelPhase146:
    SEQ_LEN = 4
    INPUT_DIM = 16

    def _input(self, model) -> torch.Tensor:
        return torch.randn(BATCH, self.SEQ_LEN, model.feature_dim)

    def test_forward_returns_logits_only(self, full_model) -> None:
        x = self._input(full_model)
        out = full_model(x)
        assert out.shape == (BATCH, 3), "forward() must return (batch, 3) logits"

    def test_enabled_heads_count(self, full_model) -> None:
        enabled = full_model.enabled_heads
        # 2 always-on + 5 Phase14.5 + 15 Phase14.6 = 22
        assert len(enabled) == 22

    def test_forward_multi_task_keys(self, full_model) -> None:
        x = self._input(full_model)
        out = full_model.forward_multi_task(x)
        expected_keys = {
            "direction_logits",
            "confidence",
            "expected_move",
            "stop_loss",
            "take_profit",
            "holding_time",
            "tp_probability",
            "mfe",
            "mae_excursion",
            "trade_efficiency",
            "reward_risk",
            "trade_survival",
            "sl_resilience",
            "tp_achievability",
            "entry_quality",
            "exit_quality",
            "market_stability",
            "trade_decay",
            "profit_distribution",
            "loss_distribution",
            "position_size_quality",
            "uncertainty",
        }
        assert expected_keys == set(out.keys())

    def test_predict_output_fields(self, full_model) -> None:
        x = self._input(full_model)
        out = full_model.predict(x)
        # Phase 14.6 fields all populated
        assert out.mfe is not None
        assert out.mae_excursion is not None
        assert out.trade_efficiency is not None
        assert out.reward_risk is not None
        assert out.trade_survival is not None
        assert out.sl_resilience is not None
        assert out.tp_achievability is not None
        assert out.entry_quality is not None
        assert out.exit_quality is not None
        assert out.market_stability is not None
        assert out.trade_decay is not None
        assert out.profit_distribution is not None
        assert out.loss_distribution is not None
        assert out.position_size_quality is not None
        assert out.uncertainty is not None

    def test_output_shapes(self, full_model) -> None:
        x = self._input(full_model)
        out = full_model.predict(x)
        assert out.mfe.shape == (BATCH, 1)
        assert out.mae_excursion.shape == (BATCH, 1)
        assert out.trade_efficiency.shape == (BATCH, 1)
        assert out.reward_risk.shape == (BATCH, 1)
        assert out.trade_survival.shape == (BATCH, 3)
        assert out.sl_resilience.shape == (BATCH, 1)
        assert out.tp_achievability.shape == (BATCH, 1)
        assert out.entry_quality.shape == (BATCH, 1)
        assert out.exit_quality.shape == (BATCH, 1)
        assert out.market_stability.shape == (BATCH, 4)
        assert out.trade_decay.shape == (BATCH, 6)  # 5 steps + 1
        assert out.profit_distribution.shape == (BATCH, 6)
        assert out.loss_distribution.shape == (BATCH, 4)
        assert out.position_size_quality.shape == (BATCH, 1)
        assert out.uncertainty.shape == (BATCH, 2)

    def test_output_value_ranges(self, full_model) -> None:
        x = self._input(full_model)
        out = full_model.predict(x)
        assert (out.mfe >= 0).all()
        assert (out.mae_excursion >= 0).all()
        assert (out.trade_efficiency >= 0).all() and (out.trade_efficiency <= 100).all()
        assert (out.reward_risk >= 0).all()
        assert (out.trade_survival >= 0).all() and (out.trade_survival <= 1).all()
        assert (out.sl_resilience >= 0).all() and (out.sl_resilience <= 1).all()
        assert (out.tp_achievability >= 0).all() and (out.tp_achievability <= 1).all()
        assert (out.entry_quality >= 0).all() and (out.entry_quality <= 100).all()
        assert (out.exit_quality >= 0).all() and (out.exit_quality <= 100).all()
        stability_sum = out.market_stability.sum(dim=-1)
        assert torch.allclose(stability_sum, torch.ones(BATCH), atol=1e-5)
        assert (out.trade_decay >= 0).all() and (out.trade_decay <= 1).all()
        assert (out.uncertainty >= 0).all()
        assert (out.position_size_quality >= 0).all() and (out.position_size_quality <= 100).all()

    def test_backward_compat_no_heads(self) -> None:
        from unittest.mock import MagicMock

        from trainer.models.quant_v1.model import QuantModelV1

        trainer_config = MagicMock()
        model = QuantModelV1(trainer_config, QuantV1Config())
        assert model.enabled_heads == ["direction", "confidence"]

    def test_gradient_flows_through_all_heads(self, full_model) -> None:
        full_model.train()
        x = self._input(full_model)
        out = full_model.forward_multi_task(x)
        total = sum(v.sum() for v in out.values())
        total.backward()
        for name, param in full_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestMultiTaskLossPhase146:
    SEQ_LEN = 4
    INPUT_DIM = 16

    def test_loss_with_all_labels(self, full_model) -> None:
        full_model.train()
        x = torch.randn(BATCH, self.SEQ_LEN, full_model.feature_dim)
        preds = full_model.forward_multi_task(x)

        labels = {
            "y_direction": torch.randint(0, 3, (BATCH,)),
            "y_confidence": torch.rand(BATCH),
            "y_move": torch.randn(BATCH),
            "y_sl": torch.rand(BATCH).abs(),
            "y_tp": torch.rand(BATCH).abs(),
            "y_holding_time": torch.rand(BATCH).abs(),
            "y_tp_prob": torch.rand(BATCH),
            "y_mfe": torch.rand(BATCH).abs(),
            "y_mae_excursion": torch.rand(BATCH).abs(),
            "y_trade_efficiency": torch.rand(BATCH) * 100,
            "y_reward_risk": torch.rand(BATCH).abs(),
            "y_trade_survival": torch.rand(BATCH, 3).round(),
            "y_sl_resilience": torch.rand(BATCH),
            "y_tp_achievability": torch.rand(BATCH),
            "y_entry_quality": torch.rand(BATCH) * 100,
            "y_exit_quality": torch.rand(BATCH) * 100,
            "y_market_stability": torch.randint(0, 4, (BATCH,)),
            "y_trade_decay": torch.rand(BATCH, 6),
            "y_profit_distribution": torch.randn(BATCH, 6),
            "y_loss_distribution": torch.randn(BATCH, 4),
            "y_position_size_quality": torch.rand(BATCH) * 100,
            "y_uncertainty": torch.rand(BATCH, 2),
        }

        loss_fn = MultiTaskLoss(config=MultiTaskLossConfig())
        result = loss_fn(preds, labels)

        assert isinstance(result, MultiTaskLossOutput)
        assert result.total.item() > 0
        assert result.total.requires_grad
        result.total.backward()

    def test_loss_breakdown_dict(self, full_model) -> None:
        x = torch.randn(BATCH, self.SEQ_LEN, full_model.feature_dim)
        full_model.eval()
        with torch.no_grad():
            preds = full_model.forward_multi_task(x)
        labels = {
            "y_direction": torch.randint(0, 3, (BATCH,)),
            "y_market_stability": torch.randint(0, 4, (BATCH,)),
        }
        loss_fn = MultiTaskLoss(config=MultiTaskLossConfig())
        result = loss_fn(preds, labels)
        breakdown = result.to_dict()
        assert "loss_total" in breakdown
        assert "loss_direction" in breakdown
        assert all(isinstance(v, float) for v in breakdown.values())

    def test_partial_labels_no_error(self, full_model) -> None:
        x = torch.randn(BATCH, self.SEQ_LEN, full_model.feature_dim)
        preds = full_model.forward_multi_task(x)
        labels = {"y_direction": torch.randint(0, 3, (BATCH,))}
        loss_fn = MultiTaskLoss(config=MultiTaskLossConfig())
        result = loss_fn(preds, labels)
        assert result.total.item() > 0
