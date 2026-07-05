"""QuantModelV1 — Phase 14.5 Multi-Task Quant Prediction Model.

Extends the original direction classifier with six additional prediction
heads that share the same learned market representation.  A single
forward pass now optionally produces:

  • Market direction (BUY / SELL / NO TRADE)
  • Expected move size
  • Suggested stop loss distance
  • Suggested take profit distance
  • Expected holding time
  • Confidence score
  • Probability that Take Profit is reached before Stop Loss

Backward compatibility is fully preserved:
  - forward()  → returns direction logits only (same as before)
  - predict()  → returns MultiTaskQuantModelOutput (superset of original)
  - All new heads are disabled by default; existing callers see no change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.models.base import BaseQuantModel
from trainer.models.quant_v1.backbone import SequenceEncoder, build_backbone
from trainer.models.quant_v1.config import QuantV1Config
from trainer.models.quant_v1.encoder import InputEncoder
from trainer.models.quant_v1.head import (
    ClassificationHead,
    ConfidenceHead,
    ExpectedMoveHead,
    HoldingTimeHead,
    StopLossHead,
    TakeProfitHead,
    TPProbabilityHead,
)
from trainer.models.quant_v1.output import (
    MultiTaskQuantModelOutput,
    OutputLayer,
    QuantModelOutput,
)
from trainer.registry.registry import ModelRegistry


@ModelRegistry.register("quant_v1")
class QuantModelV1(BaseQuantModel):
    """Signalium Quant Model V1 — Phase 14.5 Multi-Task.

    Architecture:
        Input (batch, seq_len, input_dim)
            ↓
        InputEncoder  [LayerNorm → Linear → GELU → Dropout]
            ↓
        SequenceEncoder  [LSTM / GRU]
            ↓  context vector (batch, backbone_output_dim)
            ├── ClassificationHead → logits (batch, 3)          [always on]
            ├── ConfidenceHead     → confidence (batch, 1)      [always on]
            ├── ExpectedMoveHead   → move size (batch, 1)       [optional]
            ├── StopLossHead       → SL distance (batch, 1)     [optional]
            ├── TakeProfitHead     → TP distance (batch, 1)     [optional]
            ├── HoldingTimeHead    → duration (batch, 1)        [optional]
            └── TPProbabilityHead  → P(TP before SL) (batch, 1)[optional]
            ↓
        OutputLayer  [Softmax + MultiTaskQuantModelOutput assembly]

    New heads are disabled by default; existing callers see identical
    behaviour from forward() and a superset from predict().

    Args:
        trainer_config: Global TrainerConfig.
        model_config: QuantV1Config with all hyper-parameters.
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        model_config: Optional[QuantV1Config] = None,
    ) -> None:
        super().__init__(trainer_config)
        self._model_config = model_config or QuantV1Config()

        enc_cfg = self._model_config.encoder
        bb_cfg = self._model_config.backbone
        cls_cfg = self._model_config.classification_head
        conf_cfg = self._model_config.confidence_head

        # --- Core layers (unchanged) ---
        self.input_encoder = InputEncoder(enc_cfg)
        self.backbone: SequenceEncoder = build_backbone(
            input_dim=self.input_encoder.output_dim,
            config=bb_cfg,
        )
        ctx_dim = self.backbone.output_dim

        self.classification_head = ClassificationHead(
            context_dim=ctx_dim, config=cls_cfg
        )
        self.confidence_head = ConfidenceHead(
            context_dim=ctx_dim, config=conf_cfg
        )
        self.output_layer = OutputLayer(label_names=cls_cfg.label_names)

        # --- Phase 14.5 optional heads ---
        mt = self._model_config

        self.expected_move_head: Optional[ExpectedMoveHead] = (
            ExpectedMoveHead(ctx_dim, mt.expected_move_head)
            if mt.expected_move_head.enabled
            else None
        )
        self.stop_loss_head: Optional[StopLossHead] = (
            StopLossHead(ctx_dim, mt.stop_loss_head)
            if mt.stop_loss_head.enabled
            else None
        )
        self.take_profit_head: Optional[TakeProfitHead] = (
            TakeProfitHead(ctx_dim, mt.take_profit_head)
            if mt.take_profit_head.enabled
            else None
        )
        self.holding_time_head: Optional[HoldingTimeHead] = (
            HoldingTimeHead(ctx_dim, mt.holding_time_head)
            if mt.holding_time_head.enabled
            else None
        )
        self.tp_probability_head: Optional[TPProbabilityHead] = (
            TPProbabilityHead(ctx_dim, mt.tp_probability_head)
            if mt.tp_probability_head.enabled
            else None
        )

        self._initialize_weights()

    # ------------------------------------------------------------------
    # BaseQuantModel interface
    # ------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Expected input feature dimensionality."""
        return self._model_config.encoder.input_dim

    @property
    def num_classes(self) -> int:
        """Number of output classes (BUY=0, SELL=1, NO_TRADE=2)."""
        return self._model_config.classification_head.num_classes

    # ------------------------------------------------------------------
    # Forward pass — backward compatible
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass — returns direction logits for loss computation.

        This method is IDENTICAL to the original V1 behaviour.  Downstream
        training loops using CrossEntropyLoss(logits, targets) require no
        changes.

        For the full multi-task output call :meth:`predict` or
        :meth:`forward_multi_task`.

        Args:
            x: Feature tensor of shape (batch, seq_len, input_dim).

        Returns:
            Direction logit tensor of shape (batch, num_classes).
        """
        _, logits, _ = self._encode(x)
        return logits

    def predict(self, x: torch.Tensor) -> MultiTaskQuantModelOutput:
        """Full inference pass — returns MultiTaskQuantModelOutput.

        Includes all enabled prediction head outputs.  Compatible with
        existing consumers of the original QuantModelOutput via
        ``output.to_base_output()``.

        Args:
            x: Feature tensor of shape (batch, seq_len, input_dim).

        Returns:
            MultiTaskQuantModelOutput with all enabled fields populated.
        """
        return self._forward_full(x)

    def forward_multi_task(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Multi-task forward pass for the multi-task training engine.

        Returns a dict of raw tensors (pre-activation for classification,
        post-activation for regression/binary heads) keyed by head name.
        The multi-task loss function consumes this dict directly.

        Keys always present:
            direction_logits  — (batch, num_classes) raw CE logits
            confidence        — (batch, 1) calibrated confidence [0, 1]

        Keys present when head is enabled:
            expected_move     — (batch, 1) move size (signed)
            stop_loss         — (batch, 1) SL distance (≥ 0)
            take_profit       — (batch, 1) TP distance (≥ 0)
            holding_time      — (batch, 1) duration (≥ 0)
            tp_probability    — (batch, 1) P(TP before SL) in [0, 1]

        Args:
            x: Feature tensor of shape (batch, seq_len, input_dim).

        Returns:
            Dict of named tensors.
        """
        context, logits, confidence = self._encode(x)

        out: dict[str, torch.Tensor] = {
            "direction_logits": logits,
            "confidence": confidence,
        }
        if self.expected_move_head is not None:
            out["expected_move"] = self.expected_move_head(context)
        if self.stop_loss_head is not None:
            out["stop_loss"] = self.stop_loss_head(context)
        if self.take_profit_head is not None:
            out["take_profit"] = self.take_profit_head(context)
        if self.holding_time_head is not None:
            out["holding_time"] = self.holding_time_head(context)
        if self.tp_probability_head is not None:
            out["tp_probability"] = self.tp_probability_head(context)
        return out

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    @property
    def architecture_name(self) -> str:
        bb_type = self._model_config.backbone.type.value.upper()
        return f"QuantModelV1-{bb_type}"

    @property
    def model_version(self) -> str:
        return self._model_config.model_version

    @property
    def label_names(self) -> list[str]:
        return self._model_config.classification_head.label_names

    @property
    def enabled_heads(self) -> list[str]:
        """Names of all currently enabled prediction heads."""
        heads = ["direction", "confidence"]
        if self.expected_move_head is not None:
            heads.append("expected_move")
        if self.stop_loss_head is not None:
            heads.append("stop_loss")
        if self.take_profit_head is not None:
            heads.append("take_profit")
        if self.holding_time_head is not None:
            heads.append("holding_time")
        if self.tp_probability_head is not None:
            heads.append("tp_probability")
        return heads

    def get_architecture_summary(self) -> dict[str, Any]:
        """Return a dict describing the model architecture for logging/export."""
        enc_cfg = self._model_config.encoder
        bb_cfg = self._model_config.backbone
        cls_cfg = self._model_config.classification_head
        mt = self._model_config
        return {
            "architecture": self.architecture_name,
            "version": self.model_version,
            "input_dim": enc_cfg.input_dim,
            "encoder_hidden_dim": enc_cfg.hidden_dim,
            "backbone_type": bb_cfg.type.value,
            "backbone_hidden_size": bb_cfg.hidden_size,
            "backbone_num_layers": bb_cfg.num_layers,
            "backbone_bidirectional": bb_cfg.bidirectional,
            "window_length": bb_cfg.window_length,
            "num_classes": cls_cfg.num_classes,
            "label_names": cls_cfg.label_names,
            "enabled_heads": self.enabled_heads,
            "expected_move_unit": mt.expected_move_head.unit.value if mt.expected_move_head.enabled else None,
            "stop_loss_unit": mt.stop_loss_head.unit.value if mt.stop_loss_head.enabled else None,
            "take_profit_unit": mt.take_profit_head.unit.value if mt.take_profit_head.enabled else None,
            "holding_time_unit": mt.holding_time_head.unit.value if mt.holding_time_head.enabled else None,
            **self.get_param_count(),
        }

    def get_export_metadata(self) -> dict[str, Any]:
        """Return structured metadata for export manifest embedding."""
        mt = self._model_config
        return {
            "model_id": "signalium_quant_v1",
            "model_version": self.model_version,
            "architecture": self.architecture_name,
            "phase": "14.5",
            "supported_outputs": self.enabled_heads,
            "prediction_schema": {
                "buy_prob": "float [0,1] — P(BUY)",
                "sell_prob": "float [0,1] — P(SELL)",
                "no_trade_prob": "float [0,1] — P(NO TRADE)",
                "confidence": "float [0,1] — model certainty",
                "expected_move": f"float — move size in {mt.expected_move_head.unit.value}" if mt.expected_move_head.enabled else None,
                "stop_loss": f"float ≥0 — SL distance in {mt.stop_loss_head.unit.value}" if mt.stop_loss_head.enabled else None,
                "take_profit": f"float ≥0 — TP distance in {mt.take_profit_head.unit.value}" if mt.take_profit_head.enabled else None,
                "holding_time": f"float ≥0 — duration in {mt.holding_time_head.unit.value}" if mt.holding_time_head.enabled else None,
                "tp_probability": "float [0,1] — P(TP before SL)" if mt.tp_probability_head.enabled else None,
            },
            "output_version": "2.0.0",
            "loss_config": {
                "direction_weight": mt.multi_task_loss.direction_weight,
                "move_weight": mt.multi_task_loss.move_weight,
                "sl_weight": mt.multi_task_loss.sl_weight,
                "tp_weight": mt.multi_task_loss.tp_weight,
                "holding_time_weight": mt.multi_task_loss.holding_time_weight,
                "confidence_weight": mt.multi_task_loss.confidence_weight,
                "tp_probability_weight": mt.multi_task_loss.tp_probability_weight,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared encoder pass — returns (context, logits, confidence).

        This is the hot path shared by forward(), predict(), and
        forward_multi_task(). Each head then consumes context independently
        without re-running the expensive backbone.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            context:    (batch, backbone_output_dim)
            logits:     (batch, num_classes)
            confidence: (batch, 1)
        """
        encoded = self.input_encoder(x)            # (batch, seq_len, hidden_dim)
        context = self.backbone(encoded)            # (batch, backbone_output_dim)
        logits = self.classification_head(context)  # (batch, num_classes)
        confidence = self.confidence_head(context)  # (batch, 1)
        return context, logits, confidence

    def _forward_full(self, x: torch.Tensor) -> MultiTaskQuantModelOutput:
        """Internal full forward — shared by predict()."""
        context, logits, confidence = self._encode(x)
        base = self.output_layer(logits, confidence)

        return MultiTaskQuantModelOutput(
            logits=base.logits,
            probabilities=base.probabilities,
            confidence=base.confidence,
            predicted_class=base.predicted_class,
            buy_prob=base.buy_prob,
            sell_prob=base.sell_prob,
            no_trade_prob=base.no_trade_prob,
            expected_move=self.expected_move_head(context) if self.expected_move_head else None,
            stop_loss=self.stop_loss_head(context) if self.stop_loss_head else None,
            take_profit=self.take_profit_head(context) if self.take_profit_head else None,
            holding_time=self.holding_time_head(context) if self.holding_time_head else None,
            tp_probability=self.tp_probability_head(context) if self.tp_probability_head else None,
            prediction_timestamp=datetime.now(tz=timezone.utc).isoformat(),
            model_version=self.model_version,
        )

    def _initialize_weights(self) -> None:
        """Apply sensible weight initialisation across all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
