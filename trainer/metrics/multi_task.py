"""Multi-task evaluation metrics for Phase 14.5 QuantModelV1.

Provides per-head metric computation:
  - Direction accuracy
  - Move size MAE / RMSE
  - Stop Loss MAE
  - Take Profit MAE
  - Holding Time MAE
  - TP Probability ROC-AUC
  - Confidence calibration error (ECE)
  - Confidence reliability
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MultiTaskMetrics:
    """Per-head evaluation metrics for a single evaluation pass.

    All fields are None when the corresponding head is disabled or labels
    are not available in the evaluation dataset.
    """

    direction_accuracy: float | None = None

    move_mae: float | None = None
    move_rmse: float | None = None

    sl_mae: float | None = None
    sl_rmse: float | None = None

    tp_mae: float | None = None
    tp_rmse: float | None = None

    holding_time_mae: float | None = None
    holding_time_rmse: float | None = None

    tp_prob_roc_auc: float | None = None

    confidence_ece: float | None = None
    confidence_reliability: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Return non-None metrics as a plain dict for logging/export."""
        result: dict[str, float] = {}
        for field_name, value in self.__dict__.items():
            if value is not None:
                result[field_name] = value
        return result


def compute_direction_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute direction classification accuracy.

    Args:
        logits: Raw logits of shape (N, num_classes).
        targets: Integer class labels of shape (N,).

    Returns:
        Accuracy in [0, 1].
    """
    preds = logits.argmax(dim=-1)
    return float((preds == targets.long()).float().mean().item())


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Compute MAE and RMSE for a regression head.

    Args:
        preds: Predicted values of shape (N,) or (N, 1).
        targets: Ground-truth values of shape (N,) or (N, 1).

    Returns:
        Tuple of (mae, rmse).
    """
    p = preds.squeeze(-1).float()
    t = targets.squeeze(-1).float()
    mae = (p - t).abs().mean().item()
    rmse = ((p - t) ** 2).mean().sqrt().item()
    return mae, rmse


def compute_tp_prob_roc_auc(preds: torch.Tensor, targets: torch.Tensor) -> float | None:
    """Compute ROC-AUC for the TP-before-SL binary head.

    Requires scikit-learn.  Returns None if only one class is present in
    targets (degenerate case) or if scikit-learn is unavailable.

    Args:
        preds: Predicted probabilities of shape (N,) or (N, 1), in [0, 1].
        targets: Binary labels of shape (N,) or (N, 1), values 0 or 1.

    Returns:
        ROC-AUC score, or None.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None

    p_arr = preds.squeeze(-1).float().numpy()
    t_arr = targets.squeeze(-1).float().numpy()
    unique = set(t_arr.tolist())
    if len(unique) < 2:
        return None
    try:
        return float(roc_auc_score(t_arr, p_arr))
    except Exception:
        return None


def compute_confidence_ece(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) for the confidence head.

    ECE measures the gap between predicted confidence and observed accuracy
    across calibration bins.

    Args:
        confidence: Predicted confidence values of shape (N,) or (N, 1).
        correct: Binary correctness indicators of shape (N,), 1=correct, 0=wrong.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE in [0, 1].
    """
    conf = confidence.squeeze(-1).float()
    correct = correct.float()
    n = conf.shape[0]
    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        avg_conf = conf[mask].mean().item()
        avg_acc = correct[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(avg_conf - avg_acc)

    return ece


def compute_confidence_reliability(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Compute a reliability score: fraction of bins where confidence ≥ accuracy.

    A perfectly calibrated model has reliability 1.0.  Overconfident models
    have reliability < 1.0.

    Args:
        confidence: Predicted confidence values of shape (N,) or (N, 1).
        correct: Binary correctness indicators of shape (N,).
        n_bins: Number of equal-width bins.

    Returns:
        Fraction of bins where confidence ≥ accuracy (in [0, 1]).
    """
    conf = confidence.squeeze(-1).float()
    correct = correct.float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    reliable = 0
    total_bins = 0

    for i in range(n_bins):
        lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        avg_conf = conf[mask].mean().item()
        avg_acc = correct[mask].mean().item()
        if avg_conf >= avg_acc:
            reliable += 1
        total_bins += 1

    return reliable / max(total_bins, 1)


class MultiTaskMetricsTracker:
    """Accumulates per-batch multi-task predictions for epoch-level metrics.

    Usage::

        tracker = MultiTaskMetricsTracker()
        for x, labels in val_loader:
            predictions = model.forward_multi_task(x)
            tracker.update(predictions, labels)
        metrics = tracker.compute()
    """

    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._dir_logits: list[torch.Tensor] = []
        self._dir_targets: list[torch.Tensor] = []
        self._conf: list[torch.Tensor] = []
        self._move_preds: list[torch.Tensor] = []
        self._move_tgt: list[torch.Tensor] = []
        self._sl_preds: list[torch.Tensor] = []
        self._sl_tgt: list[torch.Tensor] = []
        self._tp_preds: list[torch.Tensor] = []
        self._tp_tgt: list[torch.Tensor] = []
        self._ht_preds: list[torch.Tensor] = []
        self._ht_tgt: list[torch.Tensor] = []
        self._tp_prob_preds: list[torch.Tensor] = []
        self._tp_prob_tgt: list[torch.Tensor] = []

    def update(
        self,
        predictions: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> None:
        """Accumulate one batch of predictions and labels."""
        cpu = torch.device("cpu")

        self._dir_logits.append(predictions["direction_logits"].detach().to(cpu))
        self._dir_targets.append(labels["y_direction"].detach().to(cpu))

        if "confidence" in predictions:
            self._conf.append(predictions["confidence"].detach().to(cpu))

        def _pair(pred_key: str, lbl_key: str, p_list: list, t_list: list) -> None:
            if pred_key in predictions and lbl_key in labels:
                p_list.append(predictions[pred_key].detach().to(cpu))
                t_list.append(labels[lbl_key].detach().to(cpu))

        _pair("expected_move", "y_move", self._move_preds, self._move_tgt)
        _pair("stop_loss", "y_sl", self._sl_preds, self._sl_tgt)
        _pair("take_profit", "y_tp", self._tp_preds, self._tp_tgt)
        _pair("holding_time", "y_holding_time", self._ht_preds, self._ht_tgt)
        _pair("tp_probability", "y_tp_prob", self._tp_prob_preds, self._tp_prob_tgt)

    def compute(self) -> MultiTaskMetrics:
        """Compute epoch-level metrics from accumulated batches.

        Returns:
            MultiTaskMetrics with all available metric values populated.
        """
        metrics = MultiTaskMetrics()

        if self._dir_logits:
            all_logits = torch.cat(self._dir_logits)
            all_targets = torch.cat(self._dir_targets)
            metrics.direction_accuracy = compute_direction_accuracy(all_logits, all_targets)

            if self._conf:
                all_conf = torch.cat(self._conf)
                preds_cls = all_logits.argmax(dim=-1)
                correct = (preds_cls == all_targets.long()).float()
                metrics.confidence_ece = compute_confidence_ece(all_conf, correct)
                metrics.confidence_reliability = compute_confidence_reliability(all_conf, correct)

        def _reg(preds: list, tgts: list, prefix: str) -> None:
            if preds:
                p = torch.cat(preds)
                t = torch.cat(tgts)
                mae, rmse = compute_regression_metrics(p, t)
                setattr(metrics, f"{prefix}_mae", mae)
                setattr(metrics, f"{prefix}_rmse", rmse)

        _reg(self._move_preds, self._move_tgt, "move")
        _reg(self._sl_preds, self._sl_tgt, "sl")
        _reg(self._tp_preds, self._tp_tgt, "tp")
        _reg(self._ht_preds, self._ht_tgt, "holding_time")

        if self._tp_prob_preds:
            p = torch.cat(self._tp_prob_preds)
            t = torch.cat(self._tp_prob_tgt)
            metrics.tp_prob_roc_auc = compute_tp_prob_roc_auc(p, t)

        self._reset()
        return metrics
