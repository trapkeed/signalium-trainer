"""MultiTaskLoss — weighted combined loss for Phase 14.6 multi-task training.

Phase 14.5 terms (preserved):
    direction, confidence, expected_move, stop_loss, take_profit,
    holding_time, tp_probability

Phase 14.6 terms (new):
    mfe, mae_excursion, trade_efficiency, reward_risk,
    trade_survival, sl_resilience, tp_achievability,
    entry_quality, exit_quality, market_stability,
    trade_decay, profit_distribution, loss_distribution,
    position_size_quality, uncertainty

L_total = Σ_i  w_i * L_i   (only over heads that have both prediction and label)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.losses.advanced import DistributionLoss, MultiStepBCELoss
from trainer.models.quant_v1.config import MultiTaskLossConfig, RegressionLossType


# ---------------------------------------------------------------------------
# Regression loss helper
# ---------------------------------------------------------------------------


def _regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: RegressionLossType = RegressionLossType.HUBER,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Compute a scalar regression loss between prediction and target.

    Args:
        pred: Predictions, shape (batch, 1) or (batch,).
        target: Ground-truth, same rank after broadcasting.
        loss_type: Which regression loss to apply.
        huber_delta: Delta for Huber loss.

    Returns:
        Scalar mean loss.
    """
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    if pred.dim() == 1:
        pred = pred.unsqueeze(-1)
    if loss_type == RegressionLossType.MSE:
        return F.mse_loss(pred, target)
    if loss_type == RegressionLossType.MAE:
        return F.l1_loss(pred, target)
    return F.huber_loss(pred, target, delta=huber_delta)


# ---------------------------------------------------------------------------
# Loss output container
# ---------------------------------------------------------------------------


@dataclass
class MultiTaskLossOutput:
    """Per-head loss breakdown returned by ``MultiTaskLoss.forward()``.

    Only populated for heads that had both a prediction key and a label key.

    Attributes:
        total:               Weighted combined scalar loss (call .backward() on this).
        direction:           CrossEntropy direction loss.
        confidence:          MSE confidence calibration loss.
        expected_move:       Regression move-size loss.
        stop_loss:           Regression SL loss.
        take_profit:         Regression TP loss.
        holding_time:        Regression holding-time loss.
        tp_probability:      BCE TP-before-SL loss.

        mfe:                 Regression MFE loss.
        mae_excursion:       Regression MAE excursion loss.
        trade_efficiency:    MSE trade efficiency loss.
        reward_risk:         Regression R:R loss.
        trade_survival:      Multi-step BCE survival loss.
        sl_resilience:       BCE SL resilience loss.
        tp_achievability:    BCE TP achievability loss.
        entry_quality:       MSE entry quality loss.
        exit_quality:        MSE exit quality loss.
        market_stability:    CrossEntropy market stability loss.
        trade_decay:         Multi-step BCE decay loss.
        profit_distribution: Composite distribution loss.
        loss_distribution:   Composite distribution loss.
        position_size_quality: MSE position size quality loss.
        uncertainty:         MSE uncertainty estimation loss.
    """

    total: torch.Tensor
    direction: torch.Tensor

    # Phase 14.5
    confidence: Optional[torch.Tensor] = None
    expected_move: Optional[torch.Tensor] = None
    stop_loss: Optional[torch.Tensor] = None
    take_profit: Optional[torch.Tensor] = None
    holding_time: Optional[torch.Tensor] = None
    tp_probability: Optional[torch.Tensor] = None

    # Phase 14.6
    mfe: Optional[torch.Tensor] = None
    mae_excursion: Optional[torch.Tensor] = None
    trade_efficiency: Optional[torch.Tensor] = None
    reward_risk: Optional[torch.Tensor] = None
    trade_survival: Optional[torch.Tensor] = None
    sl_resilience: Optional[torch.Tensor] = None
    tp_achievability: Optional[torch.Tensor] = None
    entry_quality: Optional[torch.Tensor] = None
    exit_quality: Optional[torch.Tensor] = None
    market_stability: Optional[torch.Tensor] = None
    trade_decay: Optional[torch.Tensor] = None
    profit_distribution: Optional[torch.Tensor] = None
    loss_distribution: Optional[torch.Tensor] = None
    position_size_quality: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None

    def to_dict(self) -> dict[str, float]:
        """Convert to plain dict of scalar floats for logging / MLflow."""
        result: dict[str, float] = {"loss_total": self.total.item()}
        result["loss_direction"] = self.direction.item()
        _fields = [
            "confidence", "expected_move", "stop_loss", "take_profit",
            "holding_time", "tp_probability", "mfe", "mae_excursion",
            "trade_efficiency", "reward_risk", "trade_survival",
            "sl_resilience", "tp_achievability", "entry_quality",
            "exit_quality", "market_stability", "trade_decay",
            "profit_distribution", "loss_distribution",
            "position_size_quality", "uncertainty",
        ]
        for f in _fields:
            val: Optional[torch.Tensor] = getattr(self, f)
            if val is not None:
                result[f"loss_{f}"] = val.item()
        return result

    def to_series(self) -> dict[str, Any]:
        """Return all losses including total for tracking / charting."""
        return self.to_dict()


# ---------------------------------------------------------------------------
# MultiTaskLoss
# ---------------------------------------------------------------------------


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss for QuantModelV1 Phase 14.6.

    Accepts the output dict from ``QuantModelV1.forward_multi_task()`` and a
    corresponding labels dict.  Computes each active head's loss and returns a
    weighted sum along with the per-head breakdown.

    Labels dict keys (all optional except ``y_direction``):
        Phase 14.5:
            y_direction    — LongTensor (batch,)
            y_confidence   — FloatTensor (batch,) or (batch, 1)
            y_move         — FloatTensor (batch,) or (batch, 1)
            y_sl           — FloatTensor (batch,) or (batch, 1)
            y_tp           — FloatTensor (batch,) or (batch, 1)
            y_holding_time — FloatTensor (batch,) or (batch, 1)
            y_tp_prob      — FloatTensor (batch,) or (batch, 1)

        Phase 14.6:
            y_mfe                — FloatTensor (batch,) or (batch, 1)
            y_mae_excursion      — FloatTensor (batch,) or (batch, 1)
            y_trade_efficiency   — FloatTensor (batch,) or (batch, 1)
            y_reward_risk        — FloatTensor (batch,) or (batch, 1)
            y_trade_survival     — FloatTensor (batch, K)  — K survival horizons
            y_sl_resilience      — FloatTensor (batch,) or (batch, 1)
            y_tp_achievability   — FloatTensor (batch,) or (batch, 1)
            y_entry_quality      — FloatTensor (batch,) or (batch, 1)
            y_exit_quality       — FloatTensor (batch,) or (batch, 1)
            y_market_stability   — LongTensor (batch,) — class index 0-3
            y_trade_decay        — FloatTensor (batch, D+1)
            y_profit_distribution— FloatTensor (batch, 6)
            y_loss_distribution  — FloatTensor (batch, 4)
            y_position_size_quality — FloatTensor (batch,) or (batch, 1)
            y_uncertainty        — FloatTensor (batch, 2) — [epistemic, aleatoric]

    Args:
        config: MultiTaskLossConfig with all per-head weights.
        direction_class_weights: Optional class weight tensor.
        label_smoothing: Direction CrossEntropyLoss label smoothing.
        regression_loss_type: Default regression loss type for scalar heads.
        huber_delta: Delta for Huber regression losses.
    """

    def __init__(
        self,
        config: MultiTaskLossConfig,
        direction_class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        regression_loss_type: RegressionLossType = RegressionLossType.HUBER,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self._config = config
        self._huber_delta = huber_delta
        self._reg_loss_type = regression_loss_type

        self._direction_loss = nn.CrossEntropyLoss(
            weight=direction_class_weights,
            label_smoothing=label_smoothing,
        )
        self._market_stability_loss = nn.CrossEntropyLoss()
        self._profit_dist_loss = DistributionLoss(n_outputs=6, huber_delta=huber_delta)
        self._loss_dist_loss = DistributionLoss(n_outputs=4, huber_delta=huber_delta)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> MultiTaskLossOutput:
        """Compute the combined multi-task loss.

        Args:
            predictions: Output from ``QuantModelV1.forward_multi_task()``.
            labels: Dict of target tensors (see class docstring for keys).

        Returns:
            MultiTaskLossOutput with individual and total losses.
        """
        cfg = self._config

        # ── Direction (always) ────────────────────────────────────────────
        direction_loss = self._direction_loss(
            predictions["direction_logits"],
            labels["y_direction"].long(),
        )
        total = cfg.direction_weight * direction_loss

        # ── Helper for scalar regression ─────────────────────────────────
        def _reg(pred_key: str, label_key: str) -> Optional[torch.Tensor]:
            if pred_key in predictions and label_key in labels:
                return _regression_loss(
                    predictions[pred_key],
                    labels[label_key].float(),
                    self._reg_loss_type,
                    self._huber_delta,
                )
            return None

        # ── Helper for binary cross-entropy (scalar) ─────────────────────
        def _bce(pred_key: str, label_key: str) -> Optional[torch.Tensor]:
            if pred_key in predictions and label_key in labels:
                pred = predictions[pred_key]
                tgt = labels[label_key].float()
                if tgt.dim() == 1:
                    tgt = tgt.unsqueeze(-1)
                return F.binary_cross_entropy(pred, tgt)
            return None

        # ── Helper for MSE (scalar) ───────────────────────────────────────
        def _mse(pred_key: str, label_key: str) -> Optional[torch.Tensor]:
            if pred_key in predictions and label_key in labels:
                pred = predictions[pred_key]
                tgt = labels[label_key].float()
                if tgt.dim() == 1:
                    tgt = tgt.unsqueeze(-1)
                return F.mse_loss(pred, tgt)
            return None

        # ── Phase 14.5 losses ─────────────────────────────────────────────

        conf_loss: Optional[torch.Tensor] = None
        if "confidence" in predictions and "y_confidence" in labels:
            y_conf = labels["y_confidence"].float()
            if y_conf.dim() == 1:
                y_conf = y_conf.unsqueeze(-1)
            conf_loss = F.mse_loss(predictions["confidence"], y_conf)
            total = total + cfg.confidence_weight * conf_loss

        move_loss = _reg("expected_move", "y_move")
        if move_loss is not None:
            total = total + cfg.move_weight * move_loss

        sl_loss = _reg("stop_loss", "y_sl")
        if sl_loss is not None:
            total = total + cfg.sl_weight * sl_loss

        tp_loss = _reg("take_profit", "y_tp")
        if tp_loss is not None:
            total = total + cfg.tp_weight * tp_loss

        ht_loss = _reg("holding_time", "y_holding_time")
        if ht_loss is not None:
            total = total + cfg.holding_time_weight * ht_loss

        tp_prob_loss = _bce("tp_probability", "y_tp_prob")
        if tp_prob_loss is not None:
            total = total + cfg.tp_probability_weight * tp_prob_loss

        # ── Phase 14.6 losses ─────────────────────────────────────────────

        mfe_loss = _reg("mfe", "y_mfe")
        if mfe_loss is not None:
            total = total + cfg.mfe_weight * mfe_loss

        mae_exc_loss = _reg("mae_excursion", "y_mae_excursion")
        if mae_exc_loss is not None:
            total = total + cfg.mae_excursion_weight * mae_exc_loss

        te_loss = _mse("trade_efficiency", "y_trade_efficiency")
        if te_loss is not None:
            total = total + cfg.trade_efficiency_weight * te_loss

        rr_loss = _reg("reward_risk", "y_reward_risk")
        if rr_loss is not None:
            total = total + cfg.reward_risk_weight * rr_loss

        # Trade survival — multi-step BCE
        ts_loss: Optional[torch.Tensor] = None
        if "trade_survival" in predictions and "y_trade_survival" in labels:
            ts_loss = F.binary_cross_entropy(
                predictions["trade_survival"],
                labels["y_trade_survival"].float(),
            )
            total = total + cfg.trade_survival_weight * ts_loss

        sl_res_loss = _bce("sl_resilience", "y_sl_resilience")
        if sl_res_loss is not None:
            total = total + cfg.sl_resilience_weight * sl_res_loss

        tp_ach_loss = _bce("tp_achievability", "y_tp_achievability")
        if tp_ach_loss is not None:
            total = total + cfg.tp_achievability_weight * tp_ach_loss

        eq_loss = _mse("entry_quality", "y_entry_quality")
        if eq_loss is not None:
            total = total + cfg.entry_quality_weight * eq_loss

        exq_loss = _mse("exit_quality", "y_exit_quality")
        if exq_loss is not None:
            total = total + cfg.exit_quality_weight * exq_loss

        # Market stability — 4-class CrossEntropy
        ms_loss: Optional[torch.Tensor] = None
        if "market_stability" in predictions and "y_market_stability" in labels:
            ms_loss = self._market_stability_loss(
                predictions["market_stability"],
                labels["y_market_stability"].long(),
            )
            total = total + cfg.market_stability_weight * ms_loss

        # Trade decay — multi-step BCE (prob + curve)
        td_loss: Optional[torch.Tensor] = None
        if "trade_decay" in predictions and "y_trade_decay" in labels:
            td_loss = F.binary_cross_entropy(
                predictions["trade_decay"],
                labels["y_trade_decay"].float(),
            )
            total = total + cfg.trade_decay_weight * td_loss

        # Profit distribution — composite loss
        pd_loss: Optional[torch.Tensor] = None
        if "profit_distribution" in predictions and "y_profit_distribution" in labels:
            pd_loss = self._profit_dist_loss(
                predictions["profit_distribution"],
                labels["y_profit_distribution"].float(),
            )
            total = total + cfg.profit_distribution_weight * pd_loss

        # Loss distribution — composite loss
        ld_loss: Optional[torch.Tensor] = None
        if "loss_distribution" in predictions and "y_loss_distribution" in labels:
            ld_loss = self._loss_dist_loss(
                predictions["loss_distribution"],
                labels["y_loss_distribution"].float(),
            )
            total = total + cfg.loss_distribution_weight * ld_loss

        ps_loss = _mse("position_size_quality", "y_position_size_quality")
        if ps_loss is not None:
            total = total + cfg.position_size_quality_weight * ps_loss

        # Uncertainty — MSE on both components
        unc_loss: Optional[torch.Tensor] = None
        if "uncertainty" in predictions and "y_uncertainty" in labels:
            unc_loss = F.mse_loss(
                predictions["uncertainty"],
                labels["y_uncertainty"].float(),
            )
            total = total + cfg.uncertainty_weight * unc_loss

        return MultiTaskLossOutput(
            total=total,
            direction=direction_loss,
            # Phase 14.5
            confidence=conf_loss,
            expected_move=move_loss,
            stop_loss=sl_loss,
            take_profit=tp_loss,
            holding_time=ht_loss,
            tp_probability=tp_prob_loss,
            # Phase 14.6
            mfe=mfe_loss,
            mae_excursion=mae_exc_loss,
            trade_efficiency=te_loss,
            reward_risk=rr_loss,
            trade_survival=ts_loss,
            sl_resilience=sl_res_loss,
            tp_achievability=tp_ach_loss,
            entry_quality=eq_loss,
            exit_quality=exq_loss,
            market_stability=ms_loss,
            trade_decay=td_loss,
            profit_distribution=pd_loss,
            loss_distribution=ld_loss,
            position_size_quality=ps_loss,
            uncertainty=unc_loss,
        )
