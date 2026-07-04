"""QuantModelV1 — Signalium's first proprietary supervised trading signal classifier.

Predicts BUY / SELL / NO TRADE from engineered feature windows.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.models.base import BaseQuantModel
from trainer.models.quant_v1.backbone import SequenceEncoder, build_backbone
from trainer.models.quant_v1.config import QuantV1Config
from trainer.models.quant_v1.encoder import InputEncoder
from trainer.models.quant_v1.head import ClassificationHead, ConfidenceHead
from trainer.models.quant_v1.output import OutputLayer, QuantModelOutput
from trainer.registry.registry import ModelRegistry


@ModelRegistry.register("quant_v1")
class QuantModelV1(BaseQuantModel):
    """Signalium Quant Model V1 — LSTM-backbone trading signal classifier.

    Architecture:
        Input (batch, seq_len, input_dim)
            ↓
        InputEncoder  [LayerNorm → Linear → GELU → Dropout]
            ↓
        SequenceEncoder  [LSTM / GRU / future backbone]
            ↓  context vector (batch, hidden_size)
            ├── ClassificationHead → logits (batch, 3)
            └── ConfidenceHead    → confidence (batch, 1)
            ↓
        OutputLayer  [Softmax + QuantModelOutput assembly]
            ↓
        QuantModelOutput
            • buy_prob      — P(BUY)
            • sell_prob     — P(SELL)
            • no_trade_prob — P(NO TRADE)
            • confidence    — model certainty in [0, 1]
            • predicted_class — argmax class index
            • probabilities — full softmax distribution

    The backbone is swappable without touching the training pipeline.
    Register alternative backbones in backbone.py and set backbone.type
    in the config.

    Args:
        trainer_config: Global TrainerConfig (used for device, mixed precision, etc.)
        model_config: QuantV1Config with all model-specific hyper-parameters.
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

        # --- Layers ---
        self.input_encoder = InputEncoder(enc_cfg)
        self.backbone: SequenceEncoder = build_backbone(
            input_dim=self.input_encoder.output_dim,
            config=bb_cfg,
        )
        self.classification_head = ClassificationHead(
            context_dim=self.backbone.output_dim,
            config=cls_cfg,
        )
        self.confidence_head = ConfidenceHead(
            context_dim=self.backbone.output_dim,
            config=conf_cfg,
        )
        self.output_layer = OutputLayer(label_names=cls_cfg.label_names)

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
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass — returns logits for loss computation.

        To obtain the full structured output (probabilities, confidence,
        predicted_class, etc.) call :meth:`predict` instead.

        Args:
            x: Feature tensor of shape (batch, seq_len, input_dim).

        Returns:
            Class logit tensor of shape (batch, num_classes).
        """
        output = self._forward_full(x)
        return output.logits

    def predict(self, x: torch.Tensor) -> QuantModelOutput:
        """Full inference pass — returns structured QuantModelOutput.

        Args:
            x: Feature tensor of shape (batch, seq_len, input_dim).

        Returns:
            QuantModelOutput with all fields populated.
        """
        return self._forward_full(x)

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    @property
    def architecture_name(self) -> str:
        """Human-readable architecture identifier."""
        bb_type = self._model_config.backbone.type.value.upper()
        return f"QuantModelV1-{bb_type}"

    @property
    def model_version(self) -> str:
        """Semantic version of this model architecture."""
        return self._model_config.model_version

    @property
    def label_names(self) -> list[str]:
        """Ordered list of class label names."""
        return self._model_config.classification_head.label_names

    def get_architecture_summary(self) -> dict[str, Any]:
        """Return a dict describing the model architecture for logging/export."""
        enc_cfg = self._model_config.encoder
        bb_cfg = self._model_config.backbone
        cls_cfg = self._model_config.classification_head
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
            **self.get_param_count(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_full(self, x: torch.Tensor) -> QuantModelOutput:
        """Internal full forward pass shared by forward() and predict()."""
        # x: (batch, seq_len, input_dim)
        encoded = self.input_encoder(x)            # (batch, seq_len, hidden_dim)
        context = self.backbone(encoded)            # (batch, backbone_output_dim)
        logits = self.classification_head(context)  # (batch, num_classes)
        confidence = self.confidence_head(context)  # (batch, 1)
        return self.output_layer(logits, confidence)

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
                        # Set forget gate bias to 1 for better gradient flow
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
