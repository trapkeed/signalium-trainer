"""QuantModelV1 — Phase 14.9A Hybrid Representation Learning Model.

Single forward pass produces the complete quantitative trade profile:

  Phase 14.5 (preserved):
    Direction (BUY/SELL/NO TRADE) · Confidence · Expected Move
    Stop Loss · Take Profit · Holding Time · TP Probability

  Phase 14.6 (preserved):
    MFE · MAE Excursion · Trade Efficiency · Reward/Risk
    Trade Survival · SL Resilience · TP Achievability
    Entry Quality · Exit Quality · Market Stability
    Trade Decay · Profit Distribution · Loss Distribution
    Position Size Quality · Uncertainty (Epistemic + Aleatoric)

  Phase 14.9A (new — optional):
    Hybrid Encoder: 5 independent streams fused into the shared latent:
      Stream 1 — Raw Market Sequence (OHLCV via LSTM/GRU)
      Stream 2 — Multi-Timeframe (7 independent timeframe encoders)
      Stream 3 — Engineered Features (MLP + independent normalisation)
      Stream 4 — Context Embeddings (instrument, session, hour, regime …)
      Stream 5 — Market Memory (Market Memory Engine state)
      Stream 6 — Dynamic Feature Gate (context-conditioned gating)
      Stream 7 — Adaptive Fusion (concat / weighted / attention)

Backward compatibility is fully preserved:
  - forward(tensor)     → returns direction logits only (unchanged)
  - forward(HybridInput)→ returns direction logits via hybrid encoder
  - predict()           → returns MultiTaskQuantModelOutput (superset)
  - forward_multi_task()→ returns dict of all enabled head tensors
  - All heads disabled by default; existing callers see zero change.
  - hybrid_encoder disabled by default; plain tensor path is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from trainer.config.schema import TrainerConfig
from trainer.models.base import BaseQuantModel
from trainer.models.quant_v1.backbone import SequenceEncoder, build_backbone
from trainer.models.quant_v1.config import QuantV1Config
from trainer.models.quant_v1.encoder import InputEncoder
from trainer.models.quant_v1.head import (  # Phase 14.5; Phase 14.6
    ClassificationHead,
    ConfidenceHead,
    EntryQualityHead,
    ExitQualityHead,
    ExpectedMoveHead,
    HoldingTimeHead,
    LossDistributionHead,
    MAEExcursionHead,
    MarketStabilityHead,
    MFEHead,
    PositionSizeQualityHead,
    ProfitDistributionHead,
    RewardRiskHead,
    SLResilienceHead,
    StopLossHead,
    TakeProfitHead,
    TPAchievabilityHead,
    TPProbabilityHead,
    TradeDecayHead,
    TradeEfficiencyHead,
    TradeSurvivalHead,
    UncertaintyHead,
)
from trainer.models.quant_v1.hybrid.encoder import HybridEncoder
from trainer.models.quant_v1.hybrid.input import HybridInput
from trainer.models.quant_v1.output import (
    MultiTaskQuantModelOutput,
    OutputLayer,
    QuantModelOutput,
)
from trainer.registry.registry import ModelRegistry


@ModelRegistry.register("quant_v1")
class QuantModelV1(BaseQuantModel):
    """Signalium Quant Model V1 — Phase 14.6 Institutional Analytics.

    Architecture (all heads share the same backbone context vector):

        Input (batch, seq_len, input_dim)
            ↓
        InputEncoder  [LayerNorm → Linear → GELU → Dropout]
            ↓
        SequenceEncoder  [LSTM / GRU]
            ↓  context vector (batch, backbone_output_dim)
            │
            ├─ Phase 14.5 ──────────────────────────────────────────────
            ├── ClassificationHead  → logits (batch, 3)          [always]
            ├── ConfidenceHead      → confidence (batch, 1)      [always]
            ├── ExpectedMoveHead    → move (batch, 1)            [opt]
            ├── StopLossHead        → SL (batch, 1)              [opt]
            ├── TakeProfitHead      → TP (batch, 1)              [opt]
            ├── HoldingTimeHead     → time (batch, 1)            [opt]
            ├── TPProbabilityHead   → P(TP|SL) (batch, 1)        [opt]
            │
            ├─ Phase 14.6 Institutional ────────────────────────────────
            ├── MFEHead             → MFE (batch, 1)             [opt]
            ├── MAEExcursionHead    → MAE (batch, 1)             [opt]
            ├── TradeEfficiencyHead → efficiency (batch, 1)      [opt]
            ├── RewardRiskHead      → R:R (batch, 1)             [opt]
            ├── TradeSurvivalHead   → survival (batch, K)        [opt]
            ├── SLResilienceHead    → SL resilience (batch, 1)   [opt]
            ├── TPAchievabilityHead → TP achievability (batch,1) [opt]
            ├── EntryQualityHead    → entry score (batch, 1)     [opt]
            ├── ExitQualityHead     → exit score (batch, 1)      [opt]
            ├── MarketStabilityHead → stability (batch, 4)       [opt]
            ├── TradeDecayHead      → decay (batch, D+1)         [opt]
            ├── ProfitDistributionHead → profit dist (batch, 6)  [opt]
            ├── LossDistributionHead   → loss dist (batch, 4)    [opt]
            ├── PositionSizeQualityHead→ PS quality (batch, 1)   [opt]
            └── UncertaintyHead     → [ep, al] (batch, 2)        [opt]

    All heads are disabled by default.  Enable via config flags.
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
        mt = self._model_config

        # ── Core layers (unchanged) ──────────────────────────────────────
        self.input_encoder = InputEncoder(enc_cfg)
        self.backbone: SequenceEncoder = build_backbone(
            input_dim=self.input_encoder.output_dim,
            config=bb_cfg,
        )
        ctx_dim = self.backbone.output_dim

        self.classification_head = ClassificationHead(context_dim=ctx_dim, config=cls_cfg)
        self.confidence_head = ConfidenceHead(context_dim=ctx_dim, config=conf_cfg)
        self.output_layer = OutputLayer(label_names=cls_cfg.label_names)

        # ── Phase 14.5 optional heads ────────────────────────────────────
        self.expected_move_head: Optional[ExpectedMoveHead] = (
            ExpectedMoveHead(ctx_dim, mt.expected_move_head)
            if mt.expected_move_head.enabled
            else None
        )
        self.stop_loss_head: Optional[StopLossHead] = (
            StopLossHead(ctx_dim, mt.stop_loss_head) if mt.stop_loss_head.enabled else None
        )
        self.take_profit_head: Optional[TakeProfitHead] = (
            TakeProfitHead(ctx_dim, mt.take_profit_head) if mt.take_profit_head.enabled else None
        )
        self.holding_time_head: Optional[HoldingTimeHead] = (
            HoldingTimeHead(ctx_dim, mt.holding_time_head) if mt.holding_time_head.enabled else None
        )
        self.tp_probability_head: Optional[TPProbabilityHead] = (
            TPProbabilityHead(ctx_dim, mt.tp_probability_head)
            if mt.tp_probability_head.enabled
            else None
        )

        # ── Phase 14.6 optional heads ────────────────────────────────────
        self.mfe_head: Optional[MFEHead] = (
            MFEHead(ctx_dim, mt.mfe_head) if mt.mfe_head.enabled else None
        )
        self.mae_excursion_head: Optional[MAEExcursionHead] = (
            MAEExcursionHead(ctx_dim, mt.mae_excursion_head)
            if mt.mae_excursion_head.enabled
            else None
        )
        self.trade_efficiency_head: Optional[TradeEfficiencyHead] = (
            TradeEfficiencyHead(ctx_dim, mt.trade_efficiency_head)
            if mt.trade_efficiency_head.enabled
            else None
        )
        self.reward_risk_head: Optional[RewardRiskHead] = (
            RewardRiskHead(ctx_dim, mt.reward_risk_head) if mt.reward_risk_head.enabled else None
        )
        self.trade_survival_head: Optional[TradeSurvivalHead] = (
            TradeSurvivalHead(ctx_dim, mt.trade_survival_head)
            if mt.trade_survival_head.enabled
            else None
        )
        self.sl_resilience_head: Optional[SLResilienceHead] = (
            SLResilienceHead(ctx_dim, mt.sl_resilience_head)
            if mt.sl_resilience_head.enabled
            else None
        )
        self.tp_achievability_head: Optional[TPAchievabilityHead] = (
            TPAchievabilityHead(ctx_dim, mt.tp_achievability_head)
            if mt.tp_achievability_head.enabled
            else None
        )
        self.entry_quality_head: Optional[EntryQualityHead] = (
            EntryQualityHead(ctx_dim, mt.entry_quality_head)
            if mt.entry_quality_head.enabled
            else None
        )
        self.exit_quality_head: Optional[ExitQualityHead] = (
            ExitQualityHead(ctx_dim, mt.exit_quality_head) if mt.exit_quality_head.enabled else None
        )
        self.market_stability_head: Optional[MarketStabilityHead] = (
            MarketStabilityHead(ctx_dim, mt.market_stability_head)
            if mt.market_stability_head.enabled
            else None
        )
        self.trade_decay_head: Optional[TradeDecayHead] = (
            TradeDecayHead(ctx_dim, mt.trade_decay_head) if mt.trade_decay_head.enabled else None
        )
        self.profit_distribution_head: Optional[ProfitDistributionHead] = (
            ProfitDistributionHead(ctx_dim, mt.profit_distribution_head)
            if mt.profit_distribution_head.enabled
            else None
        )
        self.loss_distribution_head: Optional[LossDistributionHead] = (
            LossDistributionHead(ctx_dim, mt.loss_distribution_head)
            if mt.loss_distribution_head.enabled
            else None
        )
        self.position_size_quality_head: Optional[PositionSizeQualityHead] = (
            PositionSizeQualityHead(ctx_dim, mt.position_size_quality_head)
            if mt.position_size_quality_head.enabled
            else None
        )
        self.uncertainty_head: Optional[UncertaintyHead] = (
            UncertaintyHead(ctx_dim, mt.uncertainty_head) if mt.uncertainty_head.enabled else None
        )

        self._initialize_weights()

        # ── Phase 14.9A — Hybrid Encoder (disabled by default) ───────────
        self.hybrid_encoder: Optional[HybridEncoder] = None
        hybrid_cfg = getattr(self._model_config, "hybrid_encoder", None)
        if hybrid_cfg is not None and hybrid_cfg.enabled:
            self.hybrid_encoder = HybridEncoder(hybrid_cfg)

    # ── BaseQuantModel interface ─────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        return self._model_config.encoder.input_dim

    @property
    def num_classes(self) -> int:
        return self._model_config.classification_head.num_classes

    # ── Forward pass — backward compatible ──────────────────────────────

    def forward(self, x: Union[torch.Tensor, HybridInput]) -> torch.Tensor:
        """Standard forward pass — returns direction logits only.

        Accepts either a plain ``torch.Tensor`` (original path, unchanged) or
        a ``HybridInput`` (Phase 14.9A multi-stream path).

        Args:
            x: Feature tensor ``(batch, seq_len, input_dim)`` **or**
               a ``HybridInput`` container when hybrid encoder is enabled.

        Returns:
            Direction logit tensor of shape ``(batch, num_classes)``.
        """
        _, logits, _ = self._encode(x)
        return logits

    def predict(self, x: Union[torch.Tensor, HybridInput]) -> MultiTaskQuantModelOutput:
        """Full inference pass — returns complete MultiTaskQuantModelOutput.

        All enabled head outputs are populated.  Compatible with existing
        consumers via ``output.to_base_output()``.

        Args:
            x: Feature tensor or ``HybridInput``.

        Returns:
            MultiTaskQuantModelOutput with all enabled fields populated.
        """
        return self._forward_full(x)

    def forward_multi_task(self, x: Union[torch.Tensor, HybridInput]) -> dict[str, torch.Tensor]:
        """Multi-task forward for the training engine — returns dict of tensors.

        Always present:
            direction_logits — (batch, num_classes) raw CE logits
            confidence       — (batch, 1) ∈ [0, 1]

        Present when head enabled (Phase 14.5):
            expected_move, stop_loss, take_profit, holding_time, tp_probability

        Present when head enabled (Phase 14.6):
            mfe, mae_excursion, trade_efficiency, reward_risk,
            trade_survival, sl_resilience, tp_achievability,
            entry_quality, exit_quality, market_stability,
            trade_decay, profit_distribution, loss_distribution,
            position_size_quality, uncertainty

        Args:
            x: Feature tensor of shape (batch, seq_len, input_dim).

        Returns:
            Dict of named tensors consumed by MultiTaskLoss.
        """
        context, logits, confidence = self._encode(x)

        out: dict[str, torch.Tensor] = {
            "direction_logits": logits,
            "confidence": confidence,
        }

        # ── Phase 14.5 ───────────────────────────────────────────────────
        _p14_5: list[tuple[Optional[nn.Module], str]] = [
            (self.expected_move_head, "expected_move"),
            (self.stop_loss_head, "stop_loss"),
            (self.take_profit_head, "take_profit"),
            (self.holding_time_head, "holding_time"),
            (self.tp_probability_head, "tp_probability"),
        ]

        # ── Phase 14.6 ───────────────────────────────────────────────────
        _p14_6: list[tuple[Optional[nn.Module], str]] = [
            (self.mfe_head, "mfe"),
            (self.mae_excursion_head, "mae_excursion"),
            (self.trade_efficiency_head, "trade_efficiency"),
            (self.reward_risk_head, "reward_risk"),
            (self.trade_survival_head, "trade_survival"),
            (self.sl_resilience_head, "sl_resilience"),
            (self.tp_achievability_head, "tp_achievability"),
            (self.entry_quality_head, "entry_quality"),
            (self.exit_quality_head, "exit_quality"),
            (self.market_stability_head, "market_stability"),
            (self.trade_decay_head, "trade_decay"),
            (self.profit_distribution_head, "profit_distribution"),
            (self.loss_distribution_head, "loss_distribution"),
            (self.position_size_quality_head, "position_size_quality"),
            (self.uncertainty_head, "uncertainty"),
        ]

        for head, key in [*_p14_5, *_p14_6]:
            if head is not None:
                out[key] = head(context)

        return out

    # ── Model metadata ───────────────────────────────────────────────────

    @property
    def is_hybrid(self) -> bool:
        """True when the Phase 14.9A hybrid encoder is active."""
        return self.hybrid_encoder is not None

    @property
    def architecture_name(self) -> str:
        bb_type = self._model_config.backbone.type.value.upper()
        if self.is_hybrid:
            return f"QuantModelV1-Hybrid-{bb_type}"
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
        _all_optional: list[tuple[Optional[nn.Module], str]] = [
            (self.expected_move_head, "expected_move"),
            (self.stop_loss_head, "stop_loss"),
            (self.take_profit_head, "take_profit"),
            (self.holding_time_head, "holding_time"),
            (self.tp_probability_head, "tp_probability"),
            (self.mfe_head, "mfe"),
            (self.mae_excursion_head, "mae_excursion"),
            (self.trade_efficiency_head, "trade_efficiency"),
            (self.reward_risk_head, "reward_risk"),
            (self.trade_survival_head, "trade_survival"),
            (self.sl_resilience_head, "sl_resilience"),
            (self.tp_achievability_head, "tp_achievability"),
            (self.entry_quality_head, "entry_quality"),
            (self.exit_quality_head, "exit_quality"),
            (self.market_stability_head, "market_stability"),
            (self.trade_decay_head, "trade_decay"),
            (self.profit_distribution_head, "profit_distribution"),
            (self.loss_distribution_head, "loss_distribution"),
            (self.position_size_quality_head, "position_size_quality"),
            (self.uncertainty_head, "uncertainty"),
        ]
        for head, name in _all_optional:
            if head is not None:
                heads.append(name)
        return heads

    def get_architecture_summary(self) -> dict[str, Any]:
        """Return a dict describing the model architecture for logging/export."""
        enc_cfg = self._model_config.encoder
        bb_cfg = self._model_config.backbone
        cls_cfg = self._model_config.classification_head
        summary: dict[str, Any] = {
            "architecture": self.architecture_name,
            "version": self.model_version,
            "phase": "14.9A" if self.is_hybrid else "14.6",
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
            **self.get_param_count(),
        }
        if self.is_hybrid and self.hybrid_encoder is not None:
            summary["hybrid_enabled_streams"] = self.hybrid_encoder.enabled_streams
        return summary

    def get_export_metadata(self) -> dict[str, Any]:
        """Return structured metadata for export manifest embedding.

        Updated in Phase 14.9C to include:
          - phase (now "14.9C")
          - active_streams — list of active hybrid encoder stream names
          - backbone_type  — backbone architecture identifier
          - fusion_strategy — adaptive fusion mode (hybrid only)
          - encoder_versions — per-encoder component version map
        """
        mt = self._model_config
        bb_cfg = self._model_config.backbone

        # Derive phase from active components
        if self.is_hybrid:
            phase = "14.9C"
        elif any(
            [
                self.mfe_head,
                self.mae_excursion_head,
                self.trade_efficiency_head,
            ]
        ):
            phase = "14.6"
        else:
            phase = "14.5"

        meta: dict[str, Any] = {
            "model_id": "signalium_quant_v1",
            "model_version": self.model_version,
            "architecture": self.architecture_name,
            "phase": phase,
            "supported_outputs": self.enabled_heads,
            "output_version": "3.0.0",
            "prediction_schema": self._build_prediction_schema(),
            "loss_config": {k: v for k, v in mt.multi_task_loss.__dict__.items()},
            # Phase 14.9C — architecture readiness fields
            "backbone_type": bb_cfg.type.value,
            "active_streams": [],
            "fusion_strategy": None,
            "encoder_versions": {
                "input_encoder": "1.0.0",
                "backbone": "14.9B",
                "classification_head": "14.5",
                "confidence_head": "14.5",
            },
        }

        if self.is_hybrid and self.hybrid_encoder is not None:
            fusion_strategy = mt.hybrid_encoder.fusion.strategy.value
            meta["hybrid_encoder"] = {
                "enabled": True,
                "enabled_streams": self.hybrid_encoder.enabled_streams,
                "output_dim": self.hybrid_encoder.output_dim,
                "fusion_strategy": fusion_strategy,
            }
            meta["active_streams"] = self.hybrid_encoder.enabled_streams
            meta["fusion_strategy"] = fusion_strategy
            # Extend encoder_versions with hybrid stream components
            meta["encoder_versions"].update(
                {
                    "hybrid_encoder": "14.9A",
                    "raw_market_stream": "14.9A",
                    "multi_timeframe_stream": "14.9A",
                    "engineered_feature_stream": "14.9A",
                    "context_embedding_stream": "14.9A",
                    "market_memory_stream": "14.9A",
                    "adaptive_fusion_layer": "14.9A",
                }
            )
        return meta

    # ── Private helpers ──────────────────────────────────────────────────

    def _encode(
        self, x: Union[torch.Tensor, HybridInput]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared encoder pass — (context, logits, confidence).

        Accepts either a plain ``torch.Tensor`` or a ``HybridInput``.

        Routing:
          - HybridInput + hybrid_encoder enabled → HybridEncoder path
          - HybridInput + no hybrid_encoder       → raw_sequence fallback
          - torch.Tensor                          → original path (unchanged)

        Single backbone pass shared by forward(), predict(), and
        forward_multi_task().  All heads consume context independently.
        """
        if isinstance(x, HybridInput):
            if self.hybrid_encoder is not None:
                context = self.hybrid_encoder(x)
            else:
                # HybridInput passed without hybrid encoder — use raw_sequence
                if x.raw_sequence is None:
                    raise ValueError(
                        "HybridInput.raw_sequence is required when hybrid_encoder "
                        "is disabled (set hybrid_encoder.enabled=True or provide "
                        "a plain torch.Tensor)."
                    )
                encoded = self.input_encoder(x.raw_sequence)
                context = self.backbone(encoded)
        else:
            # Original path — completely unchanged
            encoded = self.input_encoder(x)
            context = self.backbone(encoded)

        logits = self.classification_head(context)
        confidence = self.confidence_head(context)
        return context, logits, confidence

    def _forward_full(self, x: Union[torch.Tensor, HybridInput]) -> MultiTaskQuantModelOutput:
        """Internal full forward — shared by predict()."""
        context, logits, confidence = self._encode(x)
        base = self.output_layer(logits, confidence)

        def _call(head: Optional[nn.Module]) -> Optional[torch.Tensor]:
            return head(context) if head is not None else None

        # For MarketStabilityHead, return softmax probabilities at inference
        market_stability_out: Optional[torch.Tensor] = None
        if self.market_stability_head is not None:
            market_stability_out = self.market_stability_head.predict_proba(context)

        return MultiTaskQuantModelOutput(
            logits=base.logits,
            probabilities=base.probabilities,
            confidence=base.confidence,
            predicted_class=base.predicted_class,
            buy_prob=base.buy_prob,
            sell_prob=base.sell_prob,
            no_trade_prob=base.no_trade_prob,
            # Phase 14.5
            expected_move=_call(self.expected_move_head),
            stop_loss=_call(self.stop_loss_head),
            take_profit=_call(self.take_profit_head),
            holding_time=_call(self.holding_time_head),
            tp_probability=_call(self.tp_probability_head),
            # Phase 14.6
            mfe=_call(self.mfe_head),
            mae_excursion=_call(self.mae_excursion_head),
            trade_efficiency=_call(self.trade_efficiency_head),
            reward_risk=_call(self.reward_risk_head),
            trade_survival=_call(self.trade_survival_head),
            sl_resilience=_call(self.sl_resilience_head),
            tp_achievability=_call(self.tp_achievability_head),
            entry_quality=_call(self.entry_quality_head),
            exit_quality=_call(self.exit_quality_head),
            market_stability=market_stability_out,
            trade_decay=_call(self.trade_decay_head),
            profit_distribution=_call(self.profit_distribution_head),
            loss_distribution=_call(self.loss_distribution_head),
            position_size_quality=_call(self.position_size_quality_head),
            uncertainty=_call(self.uncertainty_head),
            prediction_timestamp=datetime.now(tz=timezone.utc).isoformat(),
            model_version=self.model_version,
        )

    def _build_prediction_schema(self) -> dict[str, Any]:
        """Build the inference output schema for export metadata."""
        mt = self._model_config
        schema: dict[str, Any] = {
            "buy_prob": "float [0,1] — P(BUY)",
            "sell_prob": "float [0,1] — P(SELL)",
            "no_trade_prob": "float [0,1] — P(NO TRADE)",
            "confidence": "float [0,1] — model certainty",
        }

        def _add(enabled: bool, key: str, desc: str) -> None:
            if enabled:
                schema[key] = desc

        _add(
            mt.expected_move_head.enabled,
            "expected_move",
            f"float — move in {mt.expected_move_head.unit.value}",
        )
        _add(
            mt.stop_loss_head.enabled,
            "stop_loss",
            f"float ≥0 — SL distance in {mt.stop_loss_head.unit.value}",
        )
        _add(
            mt.take_profit_head.enabled,
            "take_profit",
            f"float ≥0 — TP distance in {mt.take_profit_head.unit.value}",
        )
        _add(
            mt.holding_time_head.enabled,
            "holding_time",
            f"float ≥0 — duration in {mt.holding_time_head.unit.value}",
        )
        _add(mt.tp_probability_head.enabled, "tp_probability", "float [0,1] — P(TP before SL)")
        _add(mt.mfe_head.enabled, "mfe", f"float ≥0 — MFE in {mt.mfe_head.unit.value}")
        _add(
            mt.mae_excursion_head.enabled,
            "mae_excursion",
            f"float ≥0 — MAE in {mt.mae_excursion_head.unit.value}",
        )
        _add(
            mt.trade_efficiency_head.enabled, "trade_efficiency", "float [0,100] — % move captured"
        )
        _add(mt.reward_risk_head.enabled, "reward_risk", "float ≥0 — expected R:R ratio")
        _add(
            mt.trade_survival_head.enabled,
            "trade_survival",
            f"float[K] [0,1] — survival probs at bars {mt.trade_survival_head.survival_bars}",
        )
        _add(mt.sl_resilience_head.enabled, "sl_resilience", "float [0,1] — P(SL survives noise)")
        _add(
            mt.tp_achievability_head.enabled,
            "tp_achievability",
            "float [0,1] — P(TP achieved before invalidation)",
        )
        _add(mt.entry_quality_head.enabled, "entry_quality", "float [0,100] — entry quality score")
        _add(mt.exit_quality_head.enabled, "exit_quality", "float [0,100] — exit quality score")
        _add(
            mt.market_stability_head.enabled,
            "market_stability",
            "float[4] — [stable, mod_stable, unstable, highly_unstable] probs",
        )
        _add(
            mt.trade_decay_head.enabled,
            "trade_decay",
            f"float[{mt.trade_decay_head.decay_steps + 1}] — [decay_prob, curve…]",
        )
        _add(
            mt.profit_distribution_head.enabled,
            "profit_distribution",
            "float[6] — [mean, median, variance, skewness, ci_lower, ci_upper]",
        )
        _add(
            mt.loss_distribution_head.enabled,
            "loss_distribution",
            "float[4] — [mean, median, variance, tail_risk]",
        )
        _add(
            mt.position_size_quality_head.enabled,
            "position_size_quality",
            "float [0,100] — capital allocation quality (NOT lot size)",
        )
        _add(mt.uncertainty_head.enabled, "uncertainty", "float[2] — [epistemic ≥0, aleatoric ≥0]")
        return schema

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
                        param.data[n // 4 : n // 2].fill_(1.0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
