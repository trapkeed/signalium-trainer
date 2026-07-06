"""Visualization data structures for Phase 14.9C explainability outputs.

All classes produce chart-ready data dictionaries WITHOUT any matplotlib,
plotly, or seaborn dependency.  Downstream consumers pass the data to
whichever plotting library they prefer.

Classes:
    FeatureImportanceChartData  — horizontal bar chart of attribution scores
    FusionContributionChartData — pie / donut chart of hybrid stream weights
    AttentionMapChartData       — heatmap of attention weight matrix
    EncoderContributionChartData— bar chart of encoder parameter norms
    AttentionMapVisualizer      — ASCII heatmap (no dependencies)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# FeatureImportanceChartData
# ---------------------------------------------------------------------------


@dataclass
class FeatureImportanceChartData:
    """Data for a horizontal bar chart of feature attribution scores.

    Attributes:
        feature_names: Ordered feature names (largest bar first).
        scores: Corresponding attribution scores.
        colors: Suggested bar colors — ``"positive"`` | ``"negative"``.
        title: Suggested chart title.
        x_label: Suggested x-axis label.
    """

    feature_names: list[str]
    scores: list[float]
    colors: list[str]
    title: str = "Feature Attribution Scores"
    x_label: str = "Attribution Score"

    @classmethod
    def from_result(
        cls,
        result: Any,
        top_k: int = 20,
        sample_idx: int = 0,
    ) -> "FeatureImportanceChartData":
        """Build chart data from an ``AttributionResult``.

        Args:
            result: ``AttributionResult`` (must have ``aggregated`` populated).
            top_k: Number of top features to include.
            sample_idx: Batch index.

        Returns:
            Ready-to-plot ``FeatureImportanceChartData``.
        """
        top = result.top_features(k=top_k, sample_idx=sample_idx)
        names = [t[0] for t in top]
        scores = [t[1] for t in top]
        colors = ["positive" if s >= 0 else "negative" for s in scores]
        return cls(feature_names=names, scores=scores, colors=colors)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "title": self.title,
            "x_label": self.x_label,
            "feature_names": self.feature_names,
            "scores": self.scores,
            "colors": self.colors,
        }


# ---------------------------------------------------------------------------
# FusionContributionChartData
# ---------------------------------------------------------------------------


@dataclass
class FusionContributionChartData:
    """Data for a pie / donut chart of hybrid stream fusion weights.

    Attributes:
        stream_names: Stream labels.
        weights: Normalised contribution weights (sum ≈ 1.0).
        title: Suggested chart title.
    """

    stream_names: list[str]
    weights: list[float]
    title: str = "Hybrid Encoder Stream Contributions"

    @classmethod
    def from_contribution(cls, contribution: Any) -> "FusionContributionChartData":
        """Build chart data from a ``StreamContribution``.

        Args:
            contribution: ``StreamContribution`` result.

        Returns:
            Ready-to-plot ``FusionContributionChartData``.
        """
        ranked = contribution.ranked_streams
        return cls(
            stream_names=[r[0] for r in ranked],
            weights=[r[1] for r in ranked],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "stream_names": self.stream_names,
            "weights": self.weights,
        }


# ---------------------------------------------------------------------------
# AttentionMapChartData
# ---------------------------------------------------------------------------


@dataclass
class AttentionMapChartData:
    """Data for a 2-D heatmap of attention weights (queries × keys).

    Attributes:
        x_labels: Key position labels (columns = positions attended TO).
        y_labels: Query position labels (rows = positions attending FROM).
        matrix: 2-D list of floats — mean attention over layers and heads,
            shape ``(seq_len, seq_len)``.
        title: Suggested chart title.
    """

    x_labels: list[str]
    y_labels: list[str]
    matrix: list[list[float]]
    title: str = "Attention Map (Mean over Layers and Heads)"

    @classmethod
    def from_attention_map(cls, attn_map: Any) -> "AttentionMapChartData":
        """Build chart data from an ``AttentionMap``.

        Args:
            attn_map: ``AttentionMap`` from ``AttentionExtractor``.

        Returns:
            Ready-to-plot ``AttentionMapChartData``.
        """
        if attn_map.is_empty:
            return cls(x_labels=[], y_labels=[], matrix=[])
        mean = attn_map.mean_attention()  # (batch, seq, seq) — take sample 0
        mat = mean[0].cpu().tolist()
        seq_len = len(mat)
        labels = [f"t-{seq_len - 1 - i}" if i < seq_len - 1 else "t" for i in range(seq_len)]
        return cls(x_labels=labels, y_labels=labels, matrix=mat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "x_labels": self.x_labels,
            "y_labels": self.y_labels,
            "matrix": self.matrix,
        }


# ---------------------------------------------------------------------------
# EncoderContributionChartData
# ---------------------------------------------------------------------------


@dataclass
class EncoderContributionChartData:
    """Data for a bar chart of encoder-level parameter norms.

    Shows the L2 norm of each stream encoder's parameters as a proxy for
    how much the encoder has learned during training.

    Attributes:
        encoder_names: Encoder names.
        norms: L2 norm per encoder.
        title: Suggested chart title.
    """

    encoder_names: list[str]
    norms: list[float]
    title: str = "Encoder Parameter Norms"

    @classmethod
    def from_contribution(cls, contribution: Any) -> "EncoderContributionChartData":
        """Build chart data from a ``StreamContribution``.

        Args:
            contribution: ``StreamContribution`` result.

        Returns:
            Ready-to-plot ``EncoderContributionChartData``.
        """
        names = list(contribution.activation_norms.keys())
        norms = [contribution.activation_norms[n] for n in names]
        return cls(encoder_names=names, norms=norms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "encoder_names": self.encoder_names,
            "norms": self.norms,
        }


# ---------------------------------------------------------------------------
# AttentionMapVisualizer — ASCII (no dependencies)
# ---------------------------------------------------------------------------


class AttentionMapVisualizer:
    """Renders attention maps as ASCII block heatmaps (no dependencies).

    Uses Unicode block characters (░▒▓█) to represent attention weights so
    the output is printable to any terminal or log file.

    Args:
        width: Number of columns in the rendered output.
        height: Number of rows in the rendered output.

    Usage::

        viz = AttentionMapVisualizer(width=40, height=10)
        print(viz.to_ascii(attn_map))
    """

    _SHADES = " ░▒▓█"

    def __init__(self, width: int = 40, height: int = 10) -> None:
        self._width = width
        self._height = height

    def to_ascii(self, attn_map: Any, sample_idx: int = 0) -> str:
        """Render a text-based heatmap of the mean attention matrix.

        Args:
            attn_map: ``AttentionMap`` from ``AttentionExtractor``.
            sample_idx: Batch index (default 0).

        Returns:
            Multi-line ASCII string suitable for printing or logging.
        """
        if attn_map.is_empty:
            return "[No attention map available — " "model uses a non-Transformer backbone]"

        try:
            import torch
            import torch.nn.functional as F

            mean = attn_map.mean_attention()[sample_idx]  # (seq, seq)
            seq_len = int(mean.shape[0])

            grid = mean.unsqueeze(0).unsqueeze(0).float()  # (1, 1, S, S)
            resized = (
                F.interpolate(
                    grid,
                    size=(self._height, self._width),
                    mode="nearest",
                )
                .squeeze()
                .cpu()
            )

            lo = float(resized.min().item())
            hi = float(resized.max().item())
            rng = max(hi - lo, 1e-9)

            header = (
                f"Attention Heatmap  seq_len={seq_len}  "
                f"{attn_map.n_layers} layer(s)  "
                f"{attn_map.n_heads} head(s)"
            )
            lines = [header, "+" + "-" * self._width + "+"]
            for row in resized.tolist():
                chars = "".join(self._SHADES[min(4, int((v - lo) / rng * 5))] for v in row)
                lines.append("|" + chars + "|")
            lines.append("+" + "-" * self._width + "+")
            lines.append("  darker = higher attention weight")
            return "\n".join(lines)

        except Exception as exc:
            logger.debug(f"AttentionMapVisualizer.to_ascii failed: {exc}")
            return "[Attention map rendering failed — check logs for details]"
