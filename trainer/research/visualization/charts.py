"""Chart generators for the Signalium Research & Benchmarking Framework.

All chart functions return a Path to the saved image file.
matplotlib is an optional dependency; all functions degrade gracefully
if it is not installed by returning None.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

_MPL_AVAILABLE = False
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend — safe for server environments
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    pass


def _require_matplotlib(fn_name: str) -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            f"{fn_name}() requires matplotlib. " "Install it with: pip install matplotlib"
        )


def plot_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
    title: str = "Training Curves",
) -> Optional[Path]:
    """Plot train/val loss and accuracy curves side-by-side.

    Args:
        history: Dict mapping metric name → list of per-epoch values.
                 Expected keys: ``train_loss``, ``loss`` (val), ``accuracy``.
        output_path: Path to save the PNG.
        title: Plot title.

    Returns:
        Path to the saved PNG, or None if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    epochs = range(1, len(next(iter(history.values()))) + 1) if history else range(1)

    ax_loss, ax_acc = axes[0], axes[1]

    if "train_loss" in history:
        ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    if "loss" in history:
        ax_loss.plot(epochs, history["loss"], label="Val Loss", color="tomato", linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    if "accuracy" in history:
        ax_acc.plot(epochs, history["accuracy"], label="Val Accuracy", color="mediumseagreen")
    if "f1_macro" in history:
        ax_acc.plot(
            epochs, history["f1_macro"], label="Val F1 (macro)", color="goldenrod", linestyle="--"
        )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Score")
    ax_acc.set_title("Accuracy & F1")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    class_labels: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> Optional[Path]:
    """Plot and save a confusion matrix.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        class_labels: Ordered class label strings.
        output_path: Path to save the PNG.
        title: Plot title.

    Returns:
        Path to the saved PNG, or None if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_labels)
    cm = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(cm[i, j].item()),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curves(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    class_labels: list[str],
    output_path: Path,
    title: str = "ROC Curves",
) -> Optional[Path]:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_probs: Predicted class probabilities (N, num_classes).
        class_labels: Ordered class label strings.
        output_path: Path to save the PNG.
        title: Plot title.

    Returns:
        Path to the saved PNG, or None if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_labels)
    colors = ["steelblue", "tomato", "mediumseagreen", "goldenrod", "orchid"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5)", alpha=0.5)

    for c in range(n_classes):
        binary_true = (y_true == c).float().numpy()
        scores = y_probs[:, c].numpy()

        # Compute ROC via sorted thresholds
        sorted_idx = scores.argsort()[::-1]
        sorted_true = binary_true[sorted_idx]
        n_pos = binary_true.sum()
        n_neg = len(binary_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            continue

        tp, fp = 0.0, 0.0
        tprs, fprs = [0.0], [0.0]
        for label in sorted_true:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tprs.append(tp / n_pos)
            fprs.append(fp / n_neg)
        tprs.append(1.0)
        fprs.append(1.0)

        auc = sum((fprs[i + 1] - fprs[i]) * tprs[i] for i in range(len(tprs) - 1))
        color = colors[c % len(colors)]
        ax.plot(fprs, tprs, color=color, label=f"{class_labels[c]} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_calibration_curve(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    class_label: str,
    class_idx: int,
    output_path: Path,
    n_bins: int = 10,
    title: str = "Calibration Curve",
) -> Optional[Path]:
    """Plot reliability diagram for one class.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_probs: Predicted probabilities (N, num_classes).
        class_label: Class name for the title.
        class_idx: Index of the class to plot.
        output_path: Path to save the PNG.
        n_bins: Number of confidence bins.
        title: Plot title.

    Returns:
        Path to the saved PNG, or None if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    binary_true = (y_true == class_idx).float().numpy()
    probs = y_probs[:, class_idx].numpy()

    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    frac_pos, mean_conf = [], []

    for low, high in bins:
        mask = (probs >= low) & (probs < high)
        if mask.sum() > 0:
            frac_pos.append(binary_true[mask].mean())
            mean_conf.append(probs[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.6)
    ax.plot(mean_conf, frac_pos, "s-", color="steelblue", label=f"{class_label}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{title} — {class_label}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pr_curve(
    y_true: torch.Tensor,
    y_probs: torch.Tensor,
    class_labels: list[str],
    output_path: Path,
    title: str = "Precision-Recall Curves",
) -> Optional[Path]:
    """Plot precision-recall curves for each class.

    Args:
        y_true: Ground truth integer class indices (N,).
        y_probs: Predicted class probabilities (N, num_classes).
        class_labels: Ordered class label strings.
        output_path: Path to save the PNG.
        title: Plot title.

    Returns:
        Path to the saved PNG, or None if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["steelblue", "tomato", "mediumseagreen", "goldenrod", "orchid"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for c in range(len(class_labels)):
        binary_true = (y_true == c).float().numpy()
        scores = y_probs[:, c].numpy()
        sorted_idx = scores.argsort()[::-1]
        sorted_true = binary_true[sorted_idx]
        n_pos = int(binary_true.sum())
        if n_pos == 0:
            continue

        tp, fp = 0, 0
        precisions, recalls = [1.0], [0.0]
        for label in sorted_true:
            if label == 1:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_pos)

        color = colors[c % len(colors)]
        ax.plot(recalls, precisions, color=color, label=class_labels[c])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_leaderboard(
    leaderboard_dicts: list[dict[str, Any]],
    output_path: Path,
    title: str = "Model Leaderboard",
    metric: str = "overall_score",
) -> Optional[Path]:
    """Plot a horizontal bar chart of the leaderboard.

    Args:
        leaderboard_dicts: List of dicts from Leaderboard.to_dict().
        output_path: Path to save the PNG.
        title: Plot title.
        metric: Metric key to plot (default: ``"overall_score"``).

    Returns:
        Path to the saved PNG, or None if matplotlib is not available.
    """
    if not _MPL_AVAILABLE:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = [r for r in leaderboard_dicts if r.get("status") == "completed"]
    if not completed:
        return None

    names = [r["model_name"] for r in completed]
    scores = [r.get(metric, 0) for r in completed]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.7)))
    bars = ax.barh(names[::-1], scores[::-1], color="steelblue", edgecolor="white")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlim(0, max(scores) * 1.15 if scores else 1.0)
    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
