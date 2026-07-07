"""Phase 16 — AutoML Backbone Search.

Defines and evaluates multiple model backbone configurations.

Supported backbones (configuration-driven, no training):
  - lstm
  - gru
  - transformer_encoder
  - temporal_cnn
  - hybrid_lstm_cnn
  - hybrid_transformer_cnn

Architecture selection is fully configuration-driven via YAML.
No architecture-specific code changes are needed to add backbones.

Usage::

    catalog = BackboneCatalog()
    cfg = catalog.get("transformer_encoder")
    variants = catalog.generate_variants("lstm", hidden_sizes=[64, 128, 256])
    for variant in variants:
        print(variant.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Backbone specification
# ---------------------------------------------------------------------------


@dataclass
class BackboneSpec:
    """Specification for a model backbone configuration.

    Attributes:
        name: Backbone identifier (e.g. "lstm", "transformer_encoder").
        display_name: Human-readable name.
        hidden_size: Hidden state / embedding dimension.
        num_layers: Number of stacked layers.
        dropout: Dropout rate (0–0.5).
        attention_heads: Multi-head attention heads (transformer only).
        kernel_size: Conv kernel size (CNN backbones).
        bidirectional: Use bidirectional LSTM/GRU.
        fusion_strategy: How to combine modality features.
        activation: Activation function name.
        normalization: Normalization strategy.
        extra: Any additional backbone-specific parameters.
    """

    name: str
    display_name: str = ""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    attention_heads: int = 4
    kernel_size: int = 3
    bidirectional: bool = False
    fusion_strategy: str = "concat"
    activation: str = "gelu"
    normalization: str = "layer_norm"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "attention_heads": self.attention_heads,
            "kernel_size": self.kernel_size,
            "bidirectional": self.bidirectional,
            "fusion_strategy": self.fusion_strategy,
            "activation": self.activation,
            "normalization": self.normalization,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> BackboneSpec:
        return BackboneSpec(
            name=d["name"],
            display_name=d.get("display_name", ""),
            hidden_size=d.get("hidden_size", 128),
            num_layers=d.get("num_layers", 2),
            dropout=d.get("dropout", 0.1),
            attention_heads=d.get("attention_heads", 4),
            kernel_size=d.get("kernel_size", 3),
            bidirectional=d.get("bidirectional", False),
            fusion_strategy=d.get("fusion_strategy", "concat"),
            activation=d.get("activation", "gelu"),
            normalization=d.get("normalization", "layer_norm"),
            extra=d.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# Backbone catalog
# ---------------------------------------------------------------------------


# Default backbone specifications
_DEFAULT_BACKBONES: dict[str, BackboneSpec] = {
    "lstm": BackboneSpec(
        name="lstm",
        display_name="LSTM",
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        normalization="layer_norm",
    ),
    "gru": BackboneSpec(
        name="gru",
        display_name="GRU",
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
        normalization="layer_norm",
    ),
    "transformer_encoder": BackboneSpec(
        name="transformer_encoder",
        display_name="Transformer Encoder",
        hidden_size=128,
        num_layers=4,
        attention_heads=4,
        dropout=0.1,
        normalization="layer_norm",
        activation="gelu",
    ),
    "temporal_cnn": BackboneSpec(
        name="temporal_cnn",
        display_name="Temporal CNN",
        hidden_size=128,
        num_layers=4,
        kernel_size=3,
        dropout=0.1,
        activation="relu",
        normalization="batch_norm",
    ),
    "hybrid_lstm_cnn": BackboneSpec(
        name="hybrid_lstm_cnn",
        display_name="Hybrid LSTM + CNN",
        hidden_size=128,
        num_layers=2,
        kernel_size=3,
        dropout=0.1,
        fusion_strategy="concat",
        normalization="layer_norm",
        extra={"cnn_layers": 2, "lstm_layers": 2},
    ),
    "hybrid_transformer_cnn": BackboneSpec(
        name="hybrid_transformer_cnn",
        display_name="Hybrid Transformer + CNN",
        hidden_size=128,
        num_layers=4,
        attention_heads=4,
        kernel_size=3,
        dropout=0.1,
        fusion_strategy="attention",
        normalization="layer_norm",
        extra={"cnn_layers": 2, "transformer_layers": 2},
    ),
}


class BackboneCatalog:
    """Registry of available backbone configurations.

    Supports adding custom backbones and generating search variants.
    """

    def __init__(self) -> None:
        self._backbones: dict[str, BackboneSpec] = dict(_DEFAULT_BACKBONES)

    def register(self, spec: BackboneSpec) -> None:
        """Register a new backbone specification."""
        self._backbones[spec.name] = spec

    def get(self, name: str) -> BackboneSpec:
        """Get a backbone spec by name."""
        if name not in self._backbones:
            raise KeyError(
                f"Backbone {name!r} not in catalog. " f"Available: {list(self._backbones)}"
            )
        return self._backbones[name]

    def list_names(self) -> list[str]:
        """Return sorted list of registered backbone names."""
        return sorted(self._backbones)

    def all_specs(self) -> list[BackboneSpec]:
        """Return all registered backbone specs."""
        return list(self._backbones.values())

    def generate_variants(
        self,
        base_name: str,
        hidden_sizes: list[int] | None = None,
        num_layers_list: list[int] | None = None,
        dropout_values: list[float] | None = None,
    ) -> list[BackboneSpec]:
        """Generate parameter variants for a base backbone.

        Returns one BackboneSpec per unique combination of overrides.
        """
        base = self.get(base_name)
        sizes = hidden_sizes or [base.hidden_size]
        layers = num_layers_list or [base.num_layers]
        dropouts = dropout_values or [base.dropout]
        variants = []
        for h in sizes:
            for l in layers:
                for d in dropouts:
                    import copy

                    variant = copy.deepcopy(base)
                    variant.hidden_size = h
                    variant.num_layers = l
                    variant.dropout = d
                    variant.name = f"{base_name}_h{h}_l{l}_d{int(d*100)}"
                    variant.display_name = f"{base.display_name} (H={h}, L={l}, D={d:.2f})"
                    variants.append(variant)
        return variants

    def from_config(self, config: dict[str, Any]) -> BackboneSpec:
        """Build a BackboneSpec from a config dict, merging with defaults."""
        name = config.get("name", "lstm")
        base = self.get(name)
        import copy

        spec = copy.deepcopy(base)
        for k, v in config.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {"backbones": {k: v.to_dict() for k, v in self._backbones.items()}}


# ---------------------------------------------------------------------------
# Backbone evaluator
# ---------------------------------------------------------------------------


@dataclass
class BackboneEvalResult:
    """Result of evaluating a backbone configuration.

    Attributes:
        backbone_name: Name of the evaluated backbone.
        config: Full backbone configuration.
        score: Primary objective score.
        metrics: All evaluation metrics.
        latency_ms: Estimated inference latency.
        memory_mb: Estimated memory footprint.
        parameter_count: Number of model parameters.
        rank: Rank among evaluated backbones (1 = best).
    """

    backbone_name: str
    config: dict[str, Any]
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    parameter_count: int = 0
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone_name": self.backbone_name,
            "config": self.config,
            "score": self.score,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "parameter_count": self.parameter_count,
            "rank": self.rank,
        }


class BackboneSearcher:
    """Evaluates and ranks multiple backbone configurations.

    Args:
        catalog: BackboneCatalog with available backbones.
        backbones: List of backbone names to evaluate (None = all).
    """

    def __init__(
        self, catalog: BackboneCatalog | None = None, backbones: list[str] | None = None
    ) -> None:
        self.catalog = catalog or BackboneCatalog()
        self.backbones = backbones or self.catalog.list_names()

    def rank_results(self, results: list[BackboneEvalResult]) -> list[BackboneEvalResult]:
        """Rank results by score descending. Mutates rank field in-place."""
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(sorted_results):
            r.rank = i + 1
        return sorted_results

    def best(self, results: list[BackboneEvalResult]) -> BackboneEvalResult | None:
        """Return the highest-scoring result."""
        if not results:
            return None
        return max(results, key=lambda r: r.score)

    def summary(self, results: list[BackboneEvalResult]) -> dict[str, Any]:
        """Return a summary dict of backbone search results."""
        ranked = self.rank_results(results)
        return {
            "n_evaluated": len(results),
            "best_backbone": ranked[0].backbone_name if ranked else None,
            "best_score": ranked[0].score if ranked else None,
            "rankings": [
                {"rank": r.rank, "backbone": r.backbone_name, "score": r.score} for r in ranked
            ],
        }
