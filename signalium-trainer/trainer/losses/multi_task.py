"""MultiTaskLoss — weighted combined loss for Phase 14.5 multi-task training.

Computes a single differentiable scalar loss from all enabled head losses:

    L_total = w_dir   * L_direction
            + w_move  * L_expected_move    (if enabled)
            + w_sl    * L_stop_loss        (if enabled)
            + w_tp    * L_take_profit      (if enabled)
            + w_time  * L_holding_time     (if enabled)
            + w_conf  * L_confidence       (if enabled)
            + w_tpprob* L_tp_probability   (if enabled)

Each individual loss is also tracked for per-head diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.models.quant_v1.config import MultiTaskLossConfig, RegressionLossType


@dataclass
class MultiTaskLossOutput:
    """Per-head loss breakdown returned by MultiTaskLoss.forward().

    Attributes:
        total:          Combined weighted scalar loss (used for .backward()).
        direction:      CrossEntropy direction classification loss.
        confidence:     BCE confidence calibration loss (if targets supplied).
        expected_move:  Regression loss for move size head (if enabled).
        stop_loss:      Regression loss for stop loss head (if enabled).
        take_profit:    Regression loss for take profit head (if enabled).
        holding_time:   Regression loss for holding time head (if enabled).
        tp_probability: BCE loss for TP-before-SL head (if enabled).
    """

    total: torch.Tensor
    direction: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    expected_move: Optional[torch.Tensor] = None
    stop_loss: Optional[torch.Tensor] = None
    take_profit: Optional[torch.Tensor] = None
    holding_time: Optional[torch.Tensor] = None
    tp_probability: Optional[torch.Tensor] = None

    def to_dict(self) -> dict[str, float]:
        """Convert to a plain dict of scalar floats for logging."""
        result: dict[str, float] = {"loss_total": self.total.item()}
        result["loss_direction"] = self.direction.item()
        if self.confidence is not None:
            result["loss_confidence"] = self.confidence.item()
        if self.expected_move is not None:
            result["loss_expected_move"] = self.expected_move.item()
        if self.stop_loss is not None:
            result["loss_stop_loss"] = self.stop_loss.item()
        if self.take_profit is not None:
            result["loss_take_profit"] = self.take_profit.item()
        if self.holding_time is not None:
            result["loss_holding_time"] = self.holding_time.item()
        if self.tp_probability is not None:
            result["loss_tp_probability"] = self.tp_probability.item()
        return result


def _regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: RegressionLossType,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Compute a regression loss between prediction and target tensors.

    Args:
        pred: Predicted values, shape (batch, 1).
        target: Ground-truth values, shape (batch,) or (batch, 1).
        loss_type: Which regression loss to use.
        huber_delta: Delta for Huber loss (ignored for MSE / MAE).

    Returns:
        Scalar mean loss.
    """
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    if loss_type == RegressionLossType.MSE:
        return F.mse_loss(pred, target)
    if loss_type == RegressionLossType.MAE:
        return F.l1_loss(pred, target)
    return F.huber_loss(pred, target, delta=huber_delta)


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss for QuantModelV1 Phase 14.5.

    Accepts the output dict from ``QuantModelV1.forward_multi_task()`` and a
    corresponding labels dict, computes each head's loss, and returns a
    weighted sum along with the per-head breakdown.

    Labels dict keys (all optional except ``y_direction``):
        y_direction    — LongTensor (batch,), class index 0/1/2
        y_confidence   — FloatTensor (batch,) or (batch,1), target confidence [0,1]
        y_move         — FloatTensor (batch,) or (batch,1), expected move
        y_sl           — FloatTensor (batch,) or (batch,1), SL distance
        y_tp           — FloatTensor (batch,) or (batch,1), TP distance
        y_holding_time — FloatTensor (batch,) or (batch,1), holding time
        y_tp_prob      — FloatTensor (batch,) or (batch,1), P(TP before SL)

    Args:
        config: MultiTaskLossConfig with per-head weights.
        direction_class_weights: Optional class weight tensor for direction loss.
        label_smoothing: Label smoothing for direction CrossEntropyLoss.
        expected_move_loss_type: Regression loss type for expected move head.
        sl_loss_type: Regression loss type for stop loss head.
        tp_loss_type: Regression loss type for take profit head.
        holding_time_loss_type: Regression loss type for holding time head.
    """

    def __init__(
        self,
        config: MultiTaskLossConfig,
        direction_class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        expected_move_loss_type: RegressionLossType = RegressionLossType.HUBER,
        sl_loss_type: RegressionLossType = RegressionLossType.HUBER,
        tp_loss_type: RegressionLossType = RegressionLossType.HUBER,
        holding_time_loss_type: RegressionLossType = RegressionLossType.HUBER,
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self._config = config
        self._huber_delta = huber_delta
        self._move_loss_type = expected_move_loss_type
        self._sl_loss_type = sl_loss_type
        self._tp_loss_type = tp_loss_type
        self._ht_loss_type = holding_time_loss_type

        self._direction_loss = nn.CrossEntropyLoss(
            weight=direction_class_weights,
            label_smoothing=label_smoothing,
        )

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
            MultiTaskLossOutput with individual and combined losses.
        """
        cfg = self._config

        # --- Direction loss (always computed) ---
        direction_loss = self._direction_loss(
            predictions["direction_logits"],
            labels["y_direction"].long(),
        )
        total = cfg.direction_weight * direction_loss

        # --- Confidence loss (MSE against target confidence) ---
        conf_loss: Optional[torch.Tensor] = None
        if "confidence" in predictions and "y_confidence" in labels:
            y_conf = labels["y_confidence"]
            if y_conf.dim() == 1:
                y_conf = y_conf.unsqueeze(-1)
            conf_loss = F.mse_loss(predictions["confidence"], y_conf.float())
            total = total + cfg.confidence_weight * conf_loss

        # --- Expected move loss ---
        move_loss: Optional[torch.Tensor] = None
        if "expected_move" in predictions and "y_move" in labels:
            move_loss = _regression_loss(
                predictions["expected_move"],
                labels["y_move"].float(),
                self._move_loss_type,
                self._huber_delta,
            )
            total = total + cfg.move_weight * move_loss

        # --- Stop loss distance loss ---
        sl_loss: Optional[torch.Tensor] = None
        if "stop_loss" in predictions and "y_sl" in labels:
            sl_loss = _regression_loss(
                predictions["stop_loss"],
                labels["y_sl"].float(),
                self._sl_loss_type,
                self._huber_delta,
            )
            total = total + cfg.sl_weight * sl_loss

        # --- Take profit distance loss ---
        tp_loss: Optional[torch.Tensor] = None
        if "take_profit" in predictions and "y_tp" in labels:
            tp_loss = _regression_loss(
                predictions["take_profit"],
                labels["y_tp"].float(),
                self._tp_loss_type,
                self._huber_delta,
            )
            total = total + cfg.tp_weight * tp_loss

        # --- Holding time loss ---
        ht_loss: Optional[torch.Tensor] = None
        if "holding_time" in predictions and "y_holding_time" in labels:
            ht_loss = _regression_loss(
                predictions["holding_time"],
                labels["y_holding_time"].float(),
                self._ht_loss_type,
                self._huber_delta,
            )
            total = total + cfg.holding_time_weight * ht_loss

        # --- TP probability loss (BCE) ---
        tp_prob_loss: Optional[torch.Tensor] = None
        if "tp_probability" in predictions and "y_tp_prob" in labels:
            y_tp_prob = labels["y_tp_prob"].float()
            if y_tp_prob.dim() == 1:
                y_tp_prob = y_tp_prob.unsqueeze(-1)
            tp_prob_loss = F.binary_cross_entropy(
                predictions["tp_probability"], y_tp_prob
            )
            total = total + cfg.tp_probability_weight * tp_prob_loss

        return MultiTaskLossOutput(
            total=total,
            direction=direction_loss,
            confidence=conf_loss,
            expected_move=move_loss,
            stop_loss=sl_loss,
            take_profit=tp_loss,
            holding_time=ht_loss,
            tp_probability=tp_prob_loss,
        )
