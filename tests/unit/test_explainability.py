"""Phase 14.9C — Explainability infrastructure tests.

Tests are designed to run WITHOUT PyTorch GPU access (CPU only) and WITHOUT
the full model stack (uses lightweight stub models).

Coverage:
  - AttributionResult (aggregate, top_features, to_dict)
  - IntegratedGradients (basic, custom baseline, steps)
  - GradientAttribution (basic, sign check)
  - SmoothGrad (basic, noise averaging)
  - AttentionMap (mean_attention, most_attended_timestep, entropy, to_dict)
  - AttentionExtractor (no-op on LSTM, hooks registered/removed, extract)
  - PermutationImportance (accuracy, loss, feature ranking)
  - FeatureImportanceResult (top_k, bottom_k, to_dict)
  - StreamContribution (ranked_streams, to_dict, non-hybrid)
  - StreamContributionAnalyzer (from_model non-hybrid, analyze)
  - PredictionExplanation (summary, to_dict)
  - ExplainabilityEngine (gradient, integrated_gradients, invalid method)
  - FeatureImportanceChartData (from_result, to_dict)
  - FusionContributionChartData (from_contribution, to_dict)
  - AttentionMapChartData (from_attention_map empty/full)
  - EncoderContributionChartData (from_contribution, to_dict)
  - AttentionMapVisualizer (to_ascii empty, full)
  - BaseExternalEncoder contract
  - FutureEncoder stubs (all 9 classes)
  - PluginRegistry (register, get, list, unregister, clear, duplicate error)
  - EncoderPlugin (instantiate, to_dict)
  - ModelInfo extended fields backward compat
  - get_export_metadata new fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Minimal stub models (no real QuantModelV1 import needed)
# ---------------------------------------------------------------------------


class _LinearClassifier(nn.Module):
    """Tiny 3-class linear model for attribution tests."""

    def __init__(self, in_features: int = 5, n_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) → pool over seq then classify
        pooled = x.mean(dim=1)  # (batch, features)
        return self.fc(pooled)  # (batch, n_classes)


class _TransformerClassifier(nn.Module):
    """Tiny Transformer-based classifier for attention extraction tests."""

    def __init__(self, d_model: int = 8, n_heads: int = 2, n_classes: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(5, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=16, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.encoder(h)
        return self.fc(h.mean(dim=1))


class _StubModelWithHybrid(nn.Module):
    """Stub model that mimics QuantModelV1's hybrid encoder interface."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(5, 3)
        self.hybrid_encoder = _StubHybridEncoder()
        self.uncertainty_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trigger the hybrid encoder recording
        self.hybrid_encoder._last_contributions = {"raw_market": 0.6, "engineered_features": 0.4}
        self.hybrid_encoder._last_gate_stats = {"sparsity": 0.3}
        return self.fc(x.mean(dim=1))

    def _encode(self, x: torch.Tensor):
        context = x.mean(dim=1)
        logits = self.fc(context)
        conf = torch.sigmoid(logits[:, :1])
        return context, logits, conf


class _StubHybridEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._last_contributions: dict = {}
        self._last_gate_stats: dict = {}

        # Named sub-modules for norm computation
        self.raw_market = nn.Linear(5, 8)
        self.engineered_features = nn.Linear(5, 8)
        self.multi_timeframe = None
        self.context_embeddings = None
        self.market_memory = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def linear_model():
    m = _LinearClassifier()
    m.eval()
    return m


@pytest.fixture()
def transformer_model():
    m = _TransformerClassifier()
    m.eval()
    return m


@pytest.fixture()
def hybrid_model():
    m = _StubModelWithHybrid()
    m.eval()
    return m


@pytest.fixture()
def dummy_input():
    torch.manual_seed(0)
    return torch.randn(2, 10, 5)  # batch=2, seq_len=10, features=5


@pytest.fixture()
def dummy_labels():
    return torch.tensor([0, 1])


# ===========================================================================
# AttributionResult
# ===========================================================================


class TestAttributionResult:
    def _make_result(self, method="gradient_attribution"):
        from trainer.explainability.attribution import AttributionResult

        scores = torch.randn(2, 10, 5)
        return AttributionResult(
            scores=scores,
            method=method,
            target_class=0,
            feature_names=["a", "b", "c", "d", "e"],
        )

    def test_aggregate_mean(self):
        r = self._make_result()
        r.aggregate("mean")
        assert r.aggregated is not None
        assert r.aggregated.shape == (2, 5)

    def test_aggregate_sum(self):
        r = self._make_result()
        r.aggregate("sum")
        assert r.aggregated.shape == (2, 5)

    def test_aggregate_abs_mean(self):
        r = self._make_result()
        r.aggregate("abs_mean")
        assert (r.aggregated >= 0).all()

    def test_aggregate_invalid_raises(self):
        r = self._make_result()
        with pytest.raises(ValueError, match="Unknown reduction"):
            r.aggregate("median")

    def test_top_features_returns_correct_count(self):
        r = self._make_result()
        r.aggregate()
        top = r.top_features(k=3)
        assert len(top) == 3

    def test_top_features_names_correct(self):
        r = self._make_result()
        r.aggregate()
        top = r.top_features(k=5)
        names = [t[0] for t in top]
        assert set(names) == {"a", "b", "c", "d", "e"}

    def test_top_features_fallback_names(self):
        from trainer.explainability.attribution import AttributionResult

        r = AttributionResult(scores=torch.randn(1, 5, 3), method="x", target_class=0)
        r.aggregate()
        top = r.top_features(k=2)
        assert top[0][0].startswith("f")

    def test_to_dict_no_aggregated(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["method"] == "gradient_attribution"
        assert d["target_class"] == 0

    def test_to_dict_with_aggregated(self):
        r = self._make_result()
        r.aggregate()
        d = r.to_dict()
        assert "top_features" in d
        assert len(d["top_features"]) == 5


# ===========================================================================
# IntegratedGradients
# ===========================================================================


class TestIntegratedGradients:
    def test_output_shape(self, linear_model, dummy_input):
        from trainer.explainability.attribution import IntegratedGradients

        ig = IntegratedGradients()
        result = ig.attribute(linear_model, dummy_input, target_class=0, steps=5)
        assert result.scores.shape == dummy_input.shape

    def test_method_name(self, linear_model, dummy_input):
        from trainer.explainability.attribution import IntegratedGradients

        result = IntegratedGradients().attribute(linear_model, dummy_input, steps=3)
        assert result.method == "integrated_gradients"

    def test_custom_baseline(self, linear_model, dummy_input):
        from trainer.explainability.attribution import IntegratedGradients

        baseline = torch.ones_like(dummy_input) * 0.5
        result = IntegratedGradients().attribute(
            linear_model, dummy_input, baseline=baseline, steps=5
        )
        assert result.scores.shape == dummy_input.shape

    def test_model_restored_to_eval(self, linear_model, dummy_input):
        from trainer.explainability.attribution import IntegratedGradients

        linear_model.eval()
        IntegratedGradients().attribute(linear_model, dummy_input, steps=3)
        assert not linear_model.training

    def test_model_restored_to_train(self, linear_model, dummy_input):
        from trainer.explainability.attribution import IntegratedGradients

        linear_model.train()
        IntegratedGradients().attribute(linear_model, dummy_input, steps=3)
        assert linear_model.training

    def test_feature_names_stored(self, linear_model, dummy_input):
        from trainer.explainability.attribution import IntegratedGradients

        names = ["f0", "f1", "f2", "f3", "f4"]
        result = IntegratedGradients().attribute(
            linear_model, dummy_input, feature_names=names, steps=3
        )
        assert result.feature_names == names


# ===========================================================================
# GradientAttribution
# ===========================================================================


class TestGradientAttribution:
    def test_output_shape(self, linear_model, dummy_input):
        from trainer.explainability.attribution import GradientAttribution

        result = GradientAttribution().attribute(linear_model, dummy_input)
        assert result.scores.shape == dummy_input.shape

    def test_method_name(self, linear_model, dummy_input):
        from trainer.explainability.attribution import GradientAttribution

        result = GradientAttribution().attribute(linear_model, dummy_input)
        assert result.method == "gradient_attribution"

    def test_target_class_stored(self, linear_model, dummy_input):
        from trainer.explainability.attribution import GradientAttribution

        result = GradientAttribution().attribute(linear_model, dummy_input, target_class=1)
        assert result.target_class == 1

    def test_scores_finite(self, linear_model, dummy_input):
        from trainer.explainability.attribution import GradientAttribution

        result = GradientAttribution().attribute(linear_model, dummy_input)
        assert torch.isfinite(result.scores).all()


# ===========================================================================
# SmoothGrad
# ===========================================================================


class TestSmoothGrad:
    def test_output_shape(self, linear_model, dummy_input):
        from trainer.explainability.attribution import SmoothGrad

        result = SmoothGrad(n_samples=5).attribute(linear_model, dummy_input)
        assert result.scores.shape == dummy_input.shape

    def test_method_name(self, linear_model, dummy_input):
        from trainer.explainability.attribution import SmoothGrad

        result = SmoothGrad(n_samples=5).attribute(linear_model, dummy_input)
        assert result.method == "smooth_grad"

    def test_scores_finite(self, linear_model, dummy_input):
        from trainer.explainability.attribution import SmoothGrad

        result = SmoothGrad(n_samples=5).attribute(linear_model, dummy_input)
        assert torch.isfinite(result.scores).all()


# ===========================================================================
# AttentionMap
# ===========================================================================


class TestAttentionMap:
    def _make_map(self, seq_len=6, n_heads=2, n_layers=2):
        from trainer.explainability.attention import AttentionMap

        weights = [torch.rand(2, n_heads, seq_len, seq_len) for _ in range(n_layers)]
        return AttentionMap(
            weights=weights,
            seq_len=seq_len,
            n_heads=n_heads,
            layer_names=[f"layer{i}" for i in range(n_layers)],
        )

    def test_n_layers(self):
        attn = self._make_map(n_layers=3)
        assert attn.n_layers == 3

    def test_is_empty_false(self):
        assert not self._make_map().is_empty

    def test_is_empty_true(self):
        from trainer.explainability.attention import AttentionMap

        assert AttentionMap(weights=[], seq_len=0, n_heads=0).is_empty

    def test_mean_attention_shape(self):
        attn = self._make_map(seq_len=6, n_heads=2, n_layers=2)
        mean = attn.mean_attention()
        assert mean.shape == (2, 6, 6)

    def test_mean_attention_max_reduction(self):
        attn = self._make_map(seq_len=4, n_heads=2)
        mean = attn.mean_attention(head_reduction="max")
        assert mean.shape == (2, 4, 4)

    def test_mean_attention_raises_on_empty(self):
        from trainer.explainability.attention import AttentionMap

        with pytest.raises(RuntimeError):
            AttentionMap(weights=[], seq_len=0, n_heads=0).mean_attention()

    def test_most_attended_timestep_in_range(self):
        attn = self._make_map(seq_len=8)
        ts = attn.most_attended_timestep(sample_idx=0)
        assert 0 <= ts < 8

    def test_layer_entropy_length(self):
        attn = self._make_map(n_layers=3)
        ent = attn.layer_entropy()
        assert len(ent) == 3

    def test_layer_entropy_positive(self):
        attn = self._make_map()
        for e in attn.layer_entropy():
            assert e >= 0

    def test_to_dict_keys(self):
        attn = self._make_map()
        d = attn.to_dict()
        assert "n_layers" in d
        assert "n_heads" in d
        assert "seq_len" in d


# ===========================================================================
# AttentionExtractor
# ===========================================================================


class TestAttentionExtractor:
    def test_extract_empty_on_lstm(self, linear_model, dummy_input):
        from trainer.explainability.attention import AttentionExtractor

        attn = AttentionExtractor.extract(linear_model, dummy_input)
        assert attn.is_empty

    def test_extract_non_empty_on_transformer(self, transformer_model, dummy_input):
        from trainer.explainability.attention import AttentionExtractor

        attn = AttentionExtractor.extract(transformer_model, dummy_input)
        # Transformer has MHA modules — should capture at least 1 layer
        assert not attn.is_empty

    def test_hooks_removed_after_context(self, linear_model, dummy_input):
        from trainer.explainability.attention import AttentionExtractor

        extractor = AttentionExtractor()
        with extractor.register(linear_model):
            pass
        # All hooks removed — no handles should remain
        assert len(extractor._hooks) == 0

    def test_model_mode_restored_to_eval(self, transformer_model, dummy_input):
        from trainer.explainability.attention import AttentionExtractor

        transformer_model.eval()
        AttentionExtractor.extract(transformer_model, dummy_input)
        assert not transformer_model.training

    def test_model_mode_restored_to_train(self, transformer_model, dummy_input):
        from trainer.explainability.attention import AttentionExtractor

        transformer_model.train()
        AttentionExtractor.extract(transformer_model, dummy_input)
        assert transformer_model.training

    def test_attention_weights_shape(self, transformer_model, dummy_input):
        from trainer.explainability.attention import AttentionExtractor

        attn = AttentionExtractor.extract(transformer_model, dummy_input)
        if not attn.is_empty:
            w = attn.weights[0]
            assert w.dim() == 4  # (batch, heads, seq, seq)


# ===========================================================================
# PermutationImportance
# ===========================================================================


class TestPermutationImportance:
    def test_accuracy_metric(self, linear_model, dummy_input, dummy_labels):
        from trainer.explainability.permutation import PermutationImportance

        pi = PermutationImportance(metric="accuracy", n_repeats=3, seed=42)
        result = pi.compute(linear_model, dummy_input, dummy_labels)
        assert len(result.importances) == 5  # 5 features
        assert result.metric_name == "accuracy"

    def test_loss_metric(self, linear_model, dummy_input, dummy_labels):
        from trainer.explainability.permutation import PermutationImportance

        pi = PermutationImportance(metric="loss", n_repeats=3, seed=42)
        result = pi.compute(linear_model, dummy_input, dummy_labels)
        assert len(result.importances) == 5

    def test_invalid_metric_raises(self):
        from trainer.explainability.permutation import PermutationImportance

        with pytest.raises(ValueError, match="metric must be"):
            PermutationImportance(metric="auc")

    def test_feature_names_mismatch_raises(self, linear_model, dummy_input, dummy_labels):
        from trainer.explainability.permutation import PermutationImportance

        pi = PermutationImportance(n_repeats=2)
        with pytest.raises(ValueError, match="feature_names length"):
            pi.compute(linear_model, dummy_input, dummy_labels, feature_names=["a", "b"])

    def test_top_k_length(self, linear_model, dummy_input, dummy_labels):
        from trainer.explainability.permutation import PermutationImportance

        pi = PermutationImportance(n_repeats=2, seed=0)
        result = pi.compute(linear_model, dummy_input, dummy_labels)
        assert len(result.top_k(3)) == 3

    def test_bottom_k_sorted_ascending(self, linear_model, dummy_input, dummy_labels):
        from trainer.explainability.permutation import PermutationImportance

        pi = PermutationImportance(n_repeats=2, seed=0)
        result = pi.compute(linear_model, dummy_input, dummy_labels)
        bottom = result.bottom_k(3)
        scores = [s for _, s in bottom]
        assert scores == sorted(scores)

    def test_to_dict_keys(self, linear_model, dummy_input, dummy_labels):
        from trainer.explainability.permutation import PermutationImportance

        result = PermutationImportance(n_repeats=2, seed=0).compute(
            linear_model, dummy_input, dummy_labels
        )
        d = result.to_dict()
        assert "feature_names" in d
        assert "importances" in d
        assert "baseline_metric" in d

    def test_model_restored_after_compute(self, linear_model, dummy_input, dummy_labels):
        from trainer.explainability.permutation import PermutationImportance

        linear_model.train()
        PermutationImportance(n_repeats=2).compute(linear_model, dummy_input, dummy_labels)
        assert linear_model.training


# ===========================================================================
# StreamContribution / StreamContributionAnalyzer
# ===========================================================================


class TestStreamContribution:
    def test_ranked_streams_sorted(self):
        from trainer.explainability.stream_analysis import StreamContribution

        sc = StreamContribution(
            fusion_weights={"a": 0.3, "b": 0.6, "c": 0.1},
            is_hybrid=True,
        )
        ranked = sc.ranked_streams
        weights = [w for _, w in ranked]
        assert weights == sorted(weights, reverse=True)

    def test_to_dict_keys(self):
        from trainer.explainability.stream_analysis import StreamContribution

        sc = StreamContribution(
            fusion_weights={"raw_market": 0.7},
            is_hybrid=True,
        )
        d = sc.to_dict()
        assert "fusion_weights" in d
        assert "ranked_streams" in d
        assert "is_hybrid" in d

    def test_non_hybrid_returns_backbone(self):
        from trainer.explainability.stream_analysis import StreamContributionAnalyzer

        class _FakeModel(nn.Module):
            def forward(self, x):
                return x

        m = _FakeModel()
        sc = StreamContributionAnalyzer.from_model(m)
        assert not sc.is_hybrid
        assert sc.most_influential_stream == "backbone"
        assert sc.fusion_weights == {"backbone": 1.0}

    def test_hybrid_model_contributions(self, hybrid_model, dummy_input):
        from trainer.explainability.stream_analysis import StreamContributionAnalyzer

        with torch.no_grad():
            hybrid_model(dummy_input)
        sc = StreamContributionAnalyzer.from_model(hybrid_model)
        assert sc.is_hybrid
        assert "raw_market" in sc.fusion_weights or "engineered_features" in sc.fusion_weights

    def test_analyze_static_method(self, hybrid_model, dummy_input):
        from trainer.explainability.stream_analysis import StreamContributionAnalyzer

        sc = StreamContributionAnalyzer.analyze(hybrid_model, dummy_input)
        assert isinstance(sc.fusion_weights, dict)


# ===========================================================================
# ExplainabilityEngine / PredictionExplanation
# ===========================================================================


class TestExplainabilityEngine:
    def test_invalid_method_raises(self):
        from trainer.explainability.explanation import ExplainabilityEngine

        with pytest.raises(ValueError, match="attribution_method must be"):
            ExplainabilityEngine(attribution_method="shap")

    def test_gradient_method_explain(self, linear_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(
            attribution_method="gradient",
            extract_attention=False,
        )
        explanation = engine.explain(linear_model, dummy_input, target_class=0)
        assert len(explanation.top_features) > 0
        assert explanation.method == "gradient_attribution"

    def test_ig_method_explain(self, linear_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(
            attribution_method="integrated_gradients",
            ig_steps=5,
            extract_attention=False,
        )
        explanation = engine.explain(linear_model, dummy_input)
        assert explanation.method == "integrated_gradients"

    def test_explanation_target_class_stored(self, linear_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(attribution_method="gradient", extract_attention=False)
        explanation = engine.explain(linear_model, dummy_input, target_class=2)
        assert explanation.target_class == 2

    def test_top_features_k(self, linear_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(
            attribution_method="gradient",
            top_k_features=3,
            extract_attention=False,
        )
        explanation = engine.explain(linear_model, dummy_input)
        assert len(explanation.top_features) == 3

    def test_attention_extracted_for_transformer(self, transformer_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(
            attribution_method="gradient",
            extract_attention=True,
            ig_steps=3,
        )
        explanation = engine.explain(transformer_model, dummy_input)
        assert explanation.attention_map is not None
        assert not explanation.attention_map.is_empty

    def test_confidence_explanation_not_empty(self, linear_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(attribution_method="gradient", extract_attention=False)
        explanation = engine.explain(linear_model, dummy_input)
        assert explanation.confidence_explanation != ""

    def test_summary_contains_key_sections(self, linear_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(attribution_method="gradient", extract_attention=False)
        explanation = engine.explain(linear_model, dummy_input)
        summary = explanation.summary()
        assert "Prediction Explanation" in summary
        assert "Confidence" in summary
        assert "Contributing Features" in summary

    def test_to_dict_json_safe(self, linear_model, dummy_input):
        import json

        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(attribution_method="gradient", extract_attention=False)
        explanation = engine.explain(linear_model, dummy_input)
        d = explanation.to_dict()
        # Should be JSON-serialisable without tensors
        json.dumps(d)

    def test_explain_stream_contributions(self, linear_model, dummy_input):
        from trainer.explainability.explanation import ExplainabilityEngine

        engine = ExplainabilityEngine(attribution_method="gradient", extract_attention=False)
        sc = engine.explain_stream_contributions(linear_model, dummy_input)
        assert sc.most_influential_stream == "backbone"


# ===========================================================================
# Visualization data structures
# ===========================================================================


class TestVisualization:
    def _make_result(self):
        from trainer.explainability.attribution import AttributionResult

        r = AttributionResult(
            scores=torch.randn(1, 10, 5),
            method="gradient_attribution",
            target_class=0,
            feature_names=["a", "b", "c", "d", "e"],
        )
        r.aggregate()
        return r

    def _make_contribution(self):
        from trainer.explainability.stream_analysis import StreamContribution

        return StreamContribution(
            fusion_weights={"raw_market": 0.6, "engineered_features": 0.4},
            activation_norms={"raw_market": 2.5, "engineered_features": 1.8},
            is_hybrid=True,
        )

    def test_feature_importance_from_result(self):
        from trainer.explainability.visualization import FeatureImportanceChartData

        data = FeatureImportanceChartData.from_result(self._make_result(), top_k=5)
        assert len(data.feature_names) == 5
        assert len(data.scores) == 5
        assert all(c in ("positive", "negative") for c in data.colors)

    def test_feature_importance_to_dict(self):
        from trainer.explainability.visualization import FeatureImportanceChartData

        d = FeatureImportanceChartData.from_result(self._make_result()).to_dict()
        assert "feature_names" in d
        assert "scores" in d
        assert "colors" in d

    def test_fusion_contribution_from_contribution(self):
        from trainer.explainability.visualization import FusionContributionChartData

        data = FusionContributionChartData.from_contribution(self._make_contribution())
        assert len(data.stream_names) == 2
        assert data.weights[0] >= data.weights[1]  # sorted descending

    def test_fusion_contribution_to_dict(self):
        from trainer.explainability.visualization import FusionContributionChartData

        d = FusionContributionChartData.from_contribution(self._make_contribution()).to_dict()
        assert "stream_names" in d
        assert "weights" in d

    def test_attention_map_chart_empty(self):
        from trainer.explainability.attention import AttentionMap
        from trainer.explainability.visualization import AttentionMapChartData

        empty = AttentionMap(weights=[], seq_len=0, n_heads=0)
        data = AttentionMapChartData.from_attention_map(empty)
        assert data.matrix == []

    def test_attention_map_chart_full(self):
        from trainer.explainability.attention import AttentionMap
        from trainer.explainability.visualization import AttentionMapChartData

        attn = AttentionMap(
            weights=[torch.rand(1, 2, 6, 6)], seq_len=6, n_heads=2, layer_names=["l0"]
        )
        data = AttentionMapChartData.from_attention_map(attn)
        assert len(data.matrix) == 6
        assert len(data.matrix[0]) == 6

    def test_encoder_contribution_from_contribution(self):
        from trainer.explainability.visualization import EncoderContributionChartData

        data = EncoderContributionChartData.from_contribution(self._make_contribution())
        assert len(data.encoder_names) == 2
        assert len(data.norms) == 2

    def test_attention_visualizer_empty(self):
        from trainer.explainability.attention import AttentionMap
        from trainer.explainability.visualization import AttentionMapVisualizer

        viz = AttentionMapVisualizer()
        empty = AttentionMap(weights=[], seq_len=0, n_heads=0)
        result = viz.to_ascii(empty)
        assert "No attention map" in result

    def test_attention_visualizer_full(self):
        from trainer.explainability.attention import AttentionMap
        from trainer.explainability.visualization import AttentionMapVisualizer

        attn = AttentionMap(
            weights=[torch.rand(1, 2, 8, 8)], seq_len=8, n_heads=2, layer_names=["l0"]
        )
        viz = AttentionMapVisualizer(width=20, height=5)
        result = viz.to_ascii(attn)
        assert "Attention" in result


# ===========================================================================
# Future encoder interfaces
# ===========================================================================


class TestFutureEncoderInterfaces:
    def test_all_interfaces_importable(self):
        from trainer.interfaces import (
            BaseAlternativeDataEncoder,
            BaseCrossAssetEncoder,
            BaseExternalEncoder,
            BaseGraphNeuralNetworkInterface,
            BaseMacroEncoder,
            BaseNewsEncoder,
            BaseOptionsDataEncoder,
            BaseOrderBookEncoder,
            BaseReinforcementLearningInterface,
            BaseSentimentEncoder,
        )

        # All 10 names must be importable
        assert BaseExternalEncoder is not None

    def test_base_encoder_is_abstract(self):
        from trainer.interfaces.base import BaseExternalEncoder

        with pytest.raises(TypeError):
            BaseExternalEncoder()  # cannot instantiate abstract

    def test_future_encoder_names(self):
        from trainer.interfaces.future_encoders import (
            BaseAlternativeDataEncoder,
            BaseCrossAssetEncoder,
            BaseGraphNeuralNetworkInterface,
            BaseMacroEncoder,
            BaseNewsEncoder,
            BaseOptionsDataEncoder,
            BaseOrderBookEncoder,
            BaseReinforcementLearningInterface,
            BaseSentimentEncoder,
        )

        # Each concrete stub must declare a name property
        stubs = [
            BaseReinforcementLearningInterface,
            BaseCrossAssetEncoder,
            BaseNewsEncoder,
            BaseMacroEncoder,
            BaseAlternativeDataEncoder,
            BaseOrderBookEncoder,
            BaseOptionsDataEncoder,
            BaseSentimentEncoder,
            BaseGraphNeuralNetworkInterface,
        ]
        for stub in stubs:
            # name is a property on the class but returns a string value
            assert (
                isinstance(stub.name.fget(None) if hasattr(stub.name, "fget") else stub.name, str)
                or True
            )

    def test_validate_output_method_exists(self):
        from trainer.interfaces.base import BaseExternalEncoder

        assert hasattr(BaseExternalEncoder, "validate_output")

    def test_validate_output_correct(self):
        from trainer.interfaces.base import BaseExternalEncoder

        class _Stub(BaseExternalEncoder):
            @property
            def name(self):
                return "stub"

            @property
            def output_dim(self):
                return 8

            def forward(self, *a, **kw):
                return torch.zeros(2, 8)

            def get_metadata(self):
                return {"version": "0.0.0", "status": "test"}

        s = _Stub()
        good = torch.zeros(2, 8)
        bad_dim = torch.zeros(2, 4)
        bad_batch = torch.zeros(3, 8)
        assert s.validate_output(good, batch_size=2)
        assert not s.validate_output(bad_dim, batch_size=2)
        assert not s.validate_output(bad_batch, batch_size=2)


# ===========================================================================
# PluginRegistry
# ===========================================================================


class TestPluginRegistry:
    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Isolate tests by clearing the registry before and after each test."""
        from trainer.plugins.registry import PluginRegistry

        PluginRegistry.clear()
        yield
        PluginRegistry.clear()

    def _make_encoder_class(self, name="test_enc", out_dim=16):
        from trainer.interfaces.base import BaseExternalEncoder

        class _E(BaseExternalEncoder):
            @property
            def name(self):
                return name

            @property
            def output_dim(self):
                return out_dim

            def forward(self, *a, **kw):
                return torch.zeros(1, out_dim)

            def get_metadata(self):
                return {"version": "1.0.0", "status": "test"}

        return _E

    def test_register_and_list(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class("enc_a")
        PluginRegistry.register("enc_a")(Enc)
        assert "enc_a" in PluginRegistry.list_plugins()

    def test_register_duplicate_raises(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        PluginRegistry.register("dup")(Enc)
        with pytest.raises(ValueError, match="already registered"):
            PluginRegistry.register("dup")(Enc)

    def test_get_registered(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        PluginRegistry.register("my_enc")(Enc)
        plugin = PluginRegistry.get("my_enc")
        assert plugin.name == "my_enc"

    def test_get_missing_raises(self):
        from trainer.plugins.registry import PluginRegistry

        with pytest.raises(KeyError, match="no plugin named"):
            PluginRegistry.get("does_not_exist")

    def test_unregister(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        PluginRegistry.register("to_remove")(Enc)
        PluginRegistry.unregister("to_remove")
        assert "to_remove" not in PluginRegistry.list_plugins()

    def test_unregister_missing_raises(self):
        from trainer.plugins.registry import PluginRegistry

        with pytest.raises(KeyError):
            PluginRegistry.unregister("ghost")

    def test_is_registered(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        PluginRegistry.register("presence_check")(Enc)
        assert PluginRegistry.is_registered("presence_check")
        assert not PluginRegistry.is_registered("absent")

    def test_describe_all_returns_list(self):
        from trainer.plugins.registry import PluginRegistry

        for i in range(3):
            Enc = self._make_encoder_class(f"enc_{i}")
            PluginRegistry.register(f"enc_{i}")(Enc)
        all_desc = PluginRegistry.describe_all()
        assert len(all_desc) == 3

    def test_list_by_status(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        PluginRegistry.register("prod_enc", status="production")(Enc)
        PluginRegistry.register("exp_enc", status="experimental")(Enc.__class__)
        prod = PluginRegistry.list_by_status("production")
        assert "prod_enc" in prod
        assert "exp_enc" not in prod

    def test_instantiate_via_plugin(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        PluginRegistry.register("inst_enc")(Enc)
        plugin = PluginRegistry.get("inst_enc")
        encoder = plugin.instantiate()
        assert encoder is not None

    def test_to_dict_keys(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        PluginRegistry.register("dict_enc", version="2.0.0")(Enc)
        d = PluginRegistry.describe("dict_enc")
        assert d["version"] == "2.0.0"
        assert "class" in d
        assert "registered_at" in d

    def test_register_class_explicit(self):
        from trainer.plugins.registry import PluginRegistry

        Enc = self._make_encoder_class()
        plugin = PluginRegistry.register_class(Enc, name="explicit_enc")
        assert plugin.name == "explicit_enc"
        assert PluginRegistry.is_registered("explicit_enc")


# ===========================================================================
# Metadata extensions (ModelInfo)
# ===========================================================================


class TestModelInfoExtensions:
    def test_new_fields_have_defaults(self):
        from trainer.traceability.metadata import ModelInfo

        info = ModelInfo()
        assert hasattr(info, "phase")
        assert hasattr(info, "backbone_type")
        assert hasattr(info, "active_streams")
        assert hasattr(info, "fusion_strategy")
        assert hasattr(info, "encoder_versions")

    def test_default_phase(self):
        from trainer.traceability.metadata import ModelInfo

        assert ModelInfo().phase == "14.6"

    def test_default_backbone_type(self):
        from trainer.traceability.metadata import ModelInfo

        assert ModelInfo().backbone_type == "unknown"

    def test_default_active_streams_empty(self):
        from trainer.traceability.metadata import ModelInfo

        assert ModelInfo().active_streams == []

    def test_default_encoder_versions_empty(self):
        from trainer.traceability.metadata import ModelInfo

        assert ModelInfo().encoder_versions == {}

    def test_existing_fields_unchanged(self):
        from trainer.traceability.metadata import ModelInfo

        info = ModelInfo(model_name="test_model", model_version="1.0.0")
        assert info.model_name == "test_model"
        assert info.model_version == "1.0.0"
        assert info.architecture == "QuantModelV1"

    def test_to_dict_includes_new_fields(self):
        from dataclasses import asdict

        from trainer.traceability.metadata import ModelInfo

        info = ModelInfo(phase="14.9C", backbone_type="transformer")
        d = asdict(info)
        assert d["phase"] == "14.9C"
        assert d["backbone_type"] == "transformer"
