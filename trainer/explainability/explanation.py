"""High-level Explainability Engine — orchestrates all Phase 14.9C methods.

Produces a ``PredictionExplanation`` from a single model and input tensor.
Never modifies the model or its predictions.

Components orchestrated:
  - GradientAttribution / IntegratedGradients  → top contributing features
  - AttentionExtractor                          → most influential timeframe
  - StreamContributionAnalyzer                  → hybrid stream analysis
  - confidence / uncertainty head inspection    → natural-language explanation

Usage::

    engine = ExplainabilityEngine(attribution_method="integrated_gradients")
    explanation = engine.explain(
        model=model,
        input_tensor=x,
        feature_names=["open", "high", "low", "close", "volume"],
        target_class=0,  # 0=BUY, 1=SELL, 2=NO_TRADE
    )
    print(explanation.summary())
    data = explanation.to_dict()   # JSON-safe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from trainer.explainability.attention import AttentionExtractor, AttentionMap
from trainer.explainability.attribution import (
    AttributionResult,
    GradientAttribution,
    IntegratedGradients,
)
from trainer.explainability.stream_analysis import StreamContribution, StreamContributionAnalyzer

# ---------------------------------------------------------------------------
# PredictionExplanation — result container
# ---------------------------------------------------------------------------


@dataclass
class PredictionExplanation:
    """Complete explanation for a single model prediction.

    Attributes:
        top_features: Ranked list of ``(feature_name, attribution_score)``
            pairs — most impactful first.
        most_influential_timeframe: Human-readable label of the most-attended
            timestep (e.g. ``"t-3 (step 7/10)"``); None for non-Transformer
            models.
        most_influential_memory: Description of the most active market memory
            stream weight; None when hybrid encoder is not active.
        confidence_explanation: Human-readable confidence interpretation.
        uncertainty_explanation: Human-readable uncertainty interpretation;
            None when the uncertainty head is not enabled.
        stream_contributions: Per-stream contribution breakdown (hybrid only).
        attention_map: Captured attention weights; None / empty for non-
            Transformer backbones.
        attribution_result: Full ``AttributionResult`` with raw tensors.
        target_class: Class index that was explained.
        method: Attribution method used (``"gradient_attribution"`` |
            ``"integrated_gradients"``).
    """

    top_features: list[tuple[str, float]] = field(default_factory=list)
    most_influential_timeframe: str | None = None
    most_influential_memory: str | None = None
    confidence_explanation: str = ""
    uncertainty_explanation: str | None = None
    stream_contributions: StreamContribution | None = None
    attention_map: AttentionMap | None = None
    attribution_result: AttributionResult | None = None
    target_class: int = 0
    method: str = "gradient_attribution"

    def summary(self) -> str:
        """Return a human-readable explanation summary.

        Uses plain ASCII so the output is safe for any terminal or log file.
        """
        label = (
            ["BUY", "SELL", "NO_TRADE"][self.target_class]
            if self.target_class < 3
            else str(self.target_class)
        )
        lines = [
            f"=== Prediction Explanation  class={self.target_class} ({label})"
            f"  method={self.method} ===",
            "",
            f"Confidence : {self.confidence_explanation}",
        ]
        if self.uncertainty_explanation:
            lines.append(f"Uncertainty: {self.uncertainty_explanation}")

        lines += ["", "Top Contributing Features:"]
        for rank, (name, score) in enumerate(self.top_features[:10], 1):
            bar_len = max(1, int(abs(score) * 25))
            bar = "#" * bar_len
            sign = "+" if score >= 0 else "-"
            lines.append(f"  {rank:2d}. {name:<32s} {sign}{bar}  ({score:+.4f})")

        if self.most_influential_timeframe:
            lines.append(f"\nMost Influential Timeframe : {self.most_influential_timeframe}")
        if self.most_influential_memory:
            lines.append(f"Most Influential Memory    : {self.most_influential_memory}")

        if self.stream_contributions and self.stream_contributions.is_hybrid:
            lines += ["", "Stream Contributions (Hybrid Encoder):"]
            for stream, weight in self.stream_contributions.ranked_streams:
                bar_len = max(1, int(weight * 30))
                bar = "#" * bar_len
                lines.append(f"  {stream:<30s} {bar}  ({weight:.3f})")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a nested JSON-safe dict (no tensors)."""
        d: dict[str, Any] = {
            "target_class": self.target_class,
            "method": self.method,
            "top_features": [{"name": n, "score": s} for n, s in self.top_features],
            "most_influential_timeframe": self.most_influential_timeframe,
            "most_influential_memory": self.most_influential_memory,
            "confidence_explanation": self.confidence_explanation,
            "uncertainty_explanation": self.uncertainty_explanation,
        }
        if self.stream_contributions:
            d["stream_contributions"] = self.stream_contributions.to_dict()
        if self.attention_map and not self.attention_map.is_empty:
            d["attention_map"] = self.attention_map.to_dict()
        if self.attribution_result:
            d["attribution"] = self.attribution_result.to_dict()
        return d


# ---------------------------------------------------------------------------
# ExplainabilityEngine — orchestrator
# ---------------------------------------------------------------------------


class ExplainabilityEngine:
    """Orchestrates all Phase 14.9C explainability methods.

    Produces a ``PredictionExplanation`` from a single call.  Never modifies
    the model — all analysis is purely read-only.

    Args:
        attribution_method: ``"integrated_gradients"`` (default, more accurate)
            or ``"gradient"`` (faster).
        ig_steps: Integration steps for Integrated Gradients (ignored when
            ``attribution_method="gradient"``).
        top_k_features: Number of top features to include.
        extract_attention: Whether to attempt attention weight extraction
            (effective only for Transformer-based backbones; gracefully
            skipped for LSTM/GRU/TCN).

    Usage::

        engine = ExplainabilityEngine()
        explanation = engine.explain(model, x, feature_names=names)
        print(explanation.summary())
    """

    _LABEL_NAMES = ["BUY", "SELL", "NO_TRADE"]

    def __init__(
        self,
        attribution_method: str = "integrated_gradients",
        ig_steps: int = 50,
        top_k_features: int = 10,
        extract_attention: bool = True,
    ) -> None:
        if attribution_method not in ("gradient", "integrated_gradients"):
            raise ValueError(
                f"attribution_method must be 'gradient' or 'integrated_gradients', "
                f"got {attribution_method!r}"
            )
        self._method = attribution_method
        self._ig_steps = ig_steps
        self._top_k = top_k_features
        self._extract_attention = extract_attention

    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        feature_names: list[str] | None = None,
        target_class: int = 0,
    ) -> PredictionExplanation:
        """Generate a complete prediction explanation.

        Args:
            model: ``QuantModelV1`` (or any model accepting
                ``(batch, seq_len, n_features)`` → logits).
            input_tensor: ``(batch, seq_len, n_features)`` input.
            feature_names: Optional list of feature column names.
            target_class: Class index to explain (0=BUY, 1=SELL, 2=NO_TRADE).

        Returns:
            ``PredictionExplanation`` with all available fields populated.
        """
        # ── Feature attribution ──────────────────────────────────────────
        attribution = self._compute_attribution(model, input_tensor, feature_names, target_class)
        attribution.aggregate(reduction="abs_mean")
        top_features = attribution.top_features(k=self._top_k, sample_idx=0)

        # ── Attention (Transformer backbones only) ───────────────────────
        attn_map: AttentionMap | None = None
        most_influential_timeframe: str | None = None
        if self._extract_attention:
            try:
                attn_map = AttentionExtractor.extract(model, input_tensor)
                if not attn_map.is_empty:
                    ts = attn_map.most_attended_timestep(sample_idx=0)
                    seq_len = int(input_tensor.shape[1])
                    offset = ts - seq_len + 1
                    most_influential_timeframe = f"t{offset:+d} (step {ts}/{seq_len - 1})"
            except Exception as exc:
                logger.debug(f"Attention extraction skipped: {exc}")

        # ── Stream contributions (Hybrid encoder only) ───────────────────
        stream_contrib: StreamContribution | None = None
        most_influential_memory: str | None = None
        try:
            stream_contrib = StreamContributionAnalyzer.analyze(model, input_tensor)
            if stream_contrib.is_hybrid and "market_memory" in stream_contrib.fusion_weights:
                w = stream_contrib.fusion_weights["market_memory"]
                most_influential_memory = f"market_memory (fusion weight={w:.3f})"
        except Exception as exc:
            logger.debug(f"Stream contribution analysis skipped: {exc}")

        # ── Confidence explanation ───────────────────────────────────────
        confidence_text = self._explain_confidence(model, input_tensor, target_class)

        # ── Uncertainty explanation ──────────────────────────────────────
        uncertainty_text = self._explain_uncertainty(model, input_tensor)

        return PredictionExplanation(
            top_features=top_features,
            most_influential_timeframe=most_influential_timeframe,
            most_influential_memory=most_influential_memory,
            confidence_explanation=confidence_text,
            uncertainty_explanation=uncertainty_text,
            stream_contributions=stream_contrib,
            attention_map=attn_map,
            attribution_result=attribution,
            target_class=target_class,
            method="gradient_attribution" if self._method == "gradient" else self._method,
        )

    def explain_stream_contributions(
        self,
        model: nn.Module,
        input_data: Any,
    ) -> StreamContribution:
        """Stream-only analysis convenience wrapper.

        Args:
            model: ``QuantModelV1`` instance.
            input_data: Input tensor or ``HybridInput``.

        Returns:
            ``StreamContribution`` with fusion weights and activation norms.
        """
        return StreamContributionAnalyzer.analyze(model, input_data)

    # ── Private helpers ───────────────────────────────────────────────────

    def _compute_attribution(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        feature_names: list[str] | None,
        target_class: int,
    ) -> AttributionResult:
        if self._method == "integrated_gradients":
            return IntegratedGradients().attribute(
                model,
                input_tensor,
                target_class=target_class,
                steps=self._ig_steps,
                feature_names=feature_names,
            )
        return GradientAttribution().attribute(
            model,
            input_tensor,
            target_class=target_class,
            feature_names=feature_names,
        )

    @torch.no_grad()
    def _explain_confidence(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> str:
        try:
            was_training = model.training
            model.eval()
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            target_prob = float(probs[0, target_class].item())
            if was_training:
                model.train()
            label = (
                self._LABEL_NAMES[target_class]
                if target_class < len(self._LABEL_NAMES)
                else str(target_class)
            )
            if target_prob >= 0.80:
                level = "very high"
            elif target_prob >= 0.65:
                level = "high"
            elif target_prob >= 0.50:
                level = "moderate"
            elif target_prob >= 0.35:
                level = "low"
            else:
                level = "very low"
            return (
                f"{level.capitalize()} confidence in {label} "
                f"(softmax probability = {target_prob:.1%})"
            )
        except Exception:
            return "Confidence explanation unavailable."

    @torch.no_grad()
    def _explain_uncertainty(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
    ) -> str | None:
        try:
            unc_head = getattr(model, "uncertainty_head", None)
            if unc_head is None:
                return None
            was_training = model.training
            model.eval()
            if not hasattr(model, "_encode"):
                return None
            context, _, _ = model._encode(input_tensor)
            unc = unc_head(context)  # (batch, 2) — [epistemic, aleatoric]
            ep = float(unc[0, 0].item())
            al = float(unc[0, 1].item())
            if was_training:
                model.train()
            level = "high" if (ep + al) > 0.5 else "low"
            parts = [f"Prediction uncertainty is {level} (epistemic={ep:.3f}, aleatoric={al:.3f})."]
            if ep > 0.3:
                parts.append(
                    "Model has limited knowledge of this region — consider reducing position size."
                )
            if al > 0.3:
                parts.append(
                    "High aleatoric noise — market conditions are inherently unpredictable here."
                )
            return " ".join(parts)
        except Exception:
            return None
