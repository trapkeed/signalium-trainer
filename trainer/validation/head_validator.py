"""Phase 15 — Prediction Head Validator.

Evaluates every prediction head independently on the validation set.

Heads supported (auto-detected from model):
  - classification_head      → accuracy, F1, ROC AUC
  - confidence_head          → calibration, ECE
  - expected_move_head       → MAE, RMSE, direction accuracy
  - holding_time_head        → MAE, RMSE
  - stop_loss_head           → MAE, RMSE, hit rate
  - take_profit_head         → MAE, RMSE, hit rate
  - tp_probability_head      → Brier score, AUC
  - mfe_head                 → MAE, RMSE, correlation
  - mae_excursion_head       → MAE, RMSE, correlation
  - trade_survival_head      → AUC, accuracy
  - entry_quality_head       → accuracy, correlation
  - exit_quality_head        → accuracy, correlation
  - trade_efficiency_head    → MAE, correlation
  - market_stability_head    → MAE, correlation
  - position_quality_head    → accuracy, correlation
  - uncertainty_head         → calibration quality
  Future heads → automatically supported via duck-typing.

Usage::

    validator = PredictionHeadValidator()
    results = validator.validate(model, x_val, y_val, probs)
    for head_name, metrics in results.items():
        print(f"{head_name}: {metrics}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Head result
# ---------------------------------------------------------------------------


@dataclass
class HeadResult:
    """Evaluation result for a single prediction head.

    Attributes:
        head_name: Name of the prediction head.
        head_type: "classification" | "regression" | "probability".
        accuracy: For classification heads.
        mae: Mean absolute error (regression heads).
        rmse: Root mean squared error.
        correlation: Pearson correlation (regression heads).
        auc: ROC AUC (classification / probability heads).
        ece: Expected calibration error (probability heads).
        brier_score: Brier score (probability heads).
        n_samples: Number of samples evaluated.
        is_active: False when head is not present or raised an error.
        error: Error message if is_active is False.
    """

    head_name: str
    head_type: str = "classification"
    accuracy: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    correlation: float = 0.0
    auc: float = 0.0
    ece: float = 0.0
    brier_score: float = 0.0
    n_samples: int = 0
    is_active: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_name": self.head_name,
            "head_type": self.head_type,
            "accuracy": round(self.accuracy, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "correlation": round(self.correlation, 4),
            "auc": round(self.auc, 4),
            "ece": round(self.ece, 4),
            "brier_score": round(self.brier_score, 4),
            "n_samples": self.n_samples,
            "is_active": self.is_active,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Prediction Head Validator
# ---------------------------------------------------------------------------


class PredictionHeadValidator:
    """Evaluates every active prediction head independently.

    Uses duck-typing to detect which heads are present on the model.
    Skips (marks as inactive) any head that is None or raises an error.

    Args:
        n_calibration_bins: Bins for ECE / calibration metrics.
    """

    # Head registry: head_attr → (head_type, description)
    _HEAD_REGISTRY: dict[str, tuple[str, str]] = {
        "classification_head": ("classification", "Direction prediction (BUY/SELL/HOLD)"),
        "confidence_head": ("probability", "Prediction confidence calibration"),
        "expected_move_head": ("regression", "Expected price move magnitude"),
        "holding_time_head": ("regression", "Predicted holding bars"),
        "stop_loss_head": ("regression", "Stop loss level prediction"),
        "take_profit_head": ("regression", "Take profit level prediction"),
        "tp_probability_head": ("probability", "TP-before-SL probability"),
        "mfe_head": ("regression", "Maximum Favourable Excursion"),
        "mae_excursion_head": ("regression", "Maximum Adverse Excursion"),
        "trade_survival_head": ("probability", "Trade survival (no SL hit) probability"),
        "entry_quality_head": ("regression", "Entry quality score"),
        "exit_quality_head": ("regression", "Exit quality score"),
        "trade_efficiency_head": ("regression", "Trade efficiency score"),
        "market_stability_head": ("regression", "Market stability score"),
        "position_quality_head": ("regression", "Position quality score"),
        "uncertainty_head": ("probability", "Epistemic + aleatoric uncertainty"),
    }

    def __init__(self, n_calibration_bins: int = 15) -> None:
        self._bins = n_calibration_bins

    def validate(
        self,
        model: nn.Module,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        probs: torch.Tensor | None = None,
    ) -> dict[str, HeadResult]:
        """Evaluate all detected heads on the validation set.

        Args:
            model: ``QuantModelV1`` instance.
            x_val: ``(N, seq_len, feat)`` validation features.
            y_val: ``(N,)`` classification labels.
            probs: Optional pre-computed ``(N, C)`` softmax probabilities.

        Returns:
            Dict mapping head_name → ``HeadResult``.
        """
        results: dict[str, HeadResult] = {}
        was_training = model.training
        model.eval()

        try:
            # Run the model to get all head outputs
            with torch.no_grad():
                try:
                    all_outputs = model.forward_multi_task(x_val)
                except Exception:
                    all_outputs = {}

            if probs is None:
                with torch.no_grad():
                    try:
                        logits = model(x_val)
                        probs = torch.softmax(logits, dim=-1)
                    except Exception:
                        probs = torch.zeros(len(x_val), 3)

            # Evaluate main classification head
            results["classification_head"] = self._eval_classification(
                "classification_head", probs, y_val
            )

            # Evaluate confidence head
            conf_out = all_outputs.get("confidence")
            if conf_out is not None:
                results["confidence_head"] = self._eval_probability_head(
                    "confidence_head", conf_out, y_val
                )
            else:
                results["confidence_head"] = self._inactive("confidence_head", "Not active")

            # Evaluate all other registered heads
            for head_name, (head_type, _) in self._HEAD_REGISTRY.items():
                if head_name in ("classification_head", "confidence_head"):
                    continue
                head_key = head_name.replace("_head", "")
                output = all_outputs.get(head_key) or all_outputs.get(head_name)
                if output is None:
                    results[head_name] = self._inactive(head_name, "Not in model outputs")
                    continue
                try:
                    if head_type == "regression":
                        results[head_name] = self._eval_regression_head(head_name, output, y_val)
                    elif head_type == "probability":
                        results[head_name] = self._eval_probability_head(head_name, output, y_val)
                    else:
                        results[head_name] = self._eval_classification(head_name, output, y_val)
                except Exception as exc:
                    results[head_name] = self._inactive(head_name, str(exc))

        finally:
            if was_training:
                model.train()

        logger.debug(
            f"PredictionHeadValidator: evaluated {sum(1 for r in results.values() if r.is_active)} "
            f"active heads of {len(results)} total."
        )
        return results

    # ── Head evaluators ───────────────────────────────────────────────────

    def _eval_classification(
        self, name: str, probs: torch.Tensor, labels: torch.Tensor
    ) -> HeadResult:
        p = probs.cpu().float().numpy()
        y = labels.cpu().long().numpy()
        n = len(y)
        preds = p.argmax(axis=1)
        acc = float((preds == y).mean())
        C = p.shape[1] if p.ndim == 2 else 1
        auc = self._roc_auc(p, y, C)
        conf = p.max(axis=1)
        ece = self._ece(conf, preds, y, n)
        return HeadResult(
            head_name=name,
            head_type="classification",
            accuracy=acc,
            auc=auc,
            ece=ece,
            n_samples=n,
        )

    def _eval_regression_head(
        self, name: str, output: torch.Tensor, labels: torch.Tensor
    ) -> HeadResult:
        y_pred = output.cpu().float().numpy().flatten()
        y_true = labels.cpu().float().numpy()
        n = min(len(y_pred), len(y_true))
        y_pred, y_true = y_pred[:n], y_true[:n]
        mae = float(np.abs(y_pred - y_true).mean())
        rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))
        if y_pred.std() > 0 and y_true.std() > 0:
            corr = float(np.corrcoef(y_pred, y_true)[0, 1])
        else:
            corr = 0.0
        return HeadResult(
            head_name=name,
            head_type="regression",
            mae=mae,
            rmse=rmse,
            correlation=corr,
            n_samples=n,
        )

    def _eval_probability_head(
        self, name: str, output: torch.Tensor, labels: torch.Tensor
    ) -> HeadResult:
        y_pred = output.cpu().float().numpy().flatten()
        y_true = labels.cpu().float().numpy()
        n = min(len(y_pred), len(y_true))
        y_pred, y_true = y_pred[:n], y_true[:n]
        y_bin = (y_true > 0).astype(float)
        auc = self._roc_auc_binary(y_pred, y_bin)
        brier = float(((y_pred - y_bin) ** 2).mean())
        ece = self._ece_binary(y_pred, y_bin)
        return HeadResult(
            head_name=name,
            head_type="probability",
            auc=auc,
            brier_score=brier,
            ece=ece,
            n_samples=n,
        )

    @staticmethod
    def _inactive(name: str, reason: str) -> HeadResult:
        return HeadResult(head_name=name, is_active=False, error=reason)

    # ── Stat helpers ──────────────────────────────────────────────────────

    def _ece(self, conf, preds, y, n):
        correct = (preds == y).astype(float)
        bins = np.linspace(0, 1, self._bins + 1)
        ece = 0.0
        for i in range(self._bins):
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
        return float(ece)

    def _ece_binary(self, probs, y):
        bins = np.linspace(0, 1, self._bins + 1)
        n = len(probs)
        ece = 0.0
        for i in range(self._bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n) * abs(y[mask].mean() - probs[mask].mean())
        return float(ece)

    @staticmethod
    def _roc_auc(p, y, C):
        aucs = []
        for c in range(C):
            y_bin = (y == c).astype(int)
            scores = p[:, c] if p.ndim == 2 else p
            aucs.append(PredictionHeadValidator._roc_auc_binary(scores, y_bin))
        return float(np.mean(aucs))

    @staticmethod
    def _roc_auc_binary(scores, labels):
        order = np.argsort(-scores)
        labels_sorted = labels[order]
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = fp = 0
        tpr, fpr = [0.0], [0.0]
        for lbl in labels_sorted:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
        _trapz = getattr(np, "trapezoid", np.trapz)
        return float(_trapz(tpr, fpr))
