"""Phase 15 — Statistical Metrics Engine.

Computes all 17 institutional statistical metrics from model predictions.

Metrics:
  1.  Accuracy
  2.  Precision (macro)
  3.  Recall (macro)
  4.  F1 (macro)
  5.  ROC AUC (one-vs-rest, macro)
  6.  PR AUC (average precision, macro)
  7.  Log Loss
  8.  Cross Entropy
  9.  MAE (on class probabilities vs one-hot)
  10. RMSE (same)
  11. MAPE (percentage error on confidence)
  12. R² (explained variance of confidence)
  13. Expected Calibration Error (ECE)
  14. Brier Score
  15. Prediction Entropy (mean)
  16. Confidence Distribution statistics
  17. Maximum Calibration Error (MCE)

All computations are vectorised in numpy — no external sklearn dependency.

Usage::

    engine = StatisticalMetricsEngine(n_classes=3)
    metrics = engine.compute(probs, labels)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class StatisticalMetrics:
    """All 17 institutional statistical metrics.

    Attributes mirror the metric names in the spec.
    """

    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    cross_entropy: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    r_squared: float = 0.0
    ece: float = 0.0
    brier_score: float = 0.0
    prediction_entropy: float = 0.0
    confidence_mean: float = 0.0
    confidence_std: float = 0.0
    confidence_p10: float = 0.0
    confidence_p50: float = 0.0
    confidence_p90: float = 0.0
    mce: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision_macro": round(self.precision_macro, 4),
            "recall_macro": round(self.recall_macro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "log_loss": round(self.log_loss, 4),
            "cross_entropy": round(self.cross_entropy, 4),
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4),
            "r_squared": round(self.r_squared, 4),
            "ece": round(self.ece, 4),
            "brier_score": round(self.brier_score, 4),
            "prediction_entropy": round(self.prediction_entropy, 4),
            "confidence_mean": round(self.confidence_mean, 4),
            "confidence_std": round(self.confidence_std, 4),
            "confidence_p10": round(self.confidence_p10, 4),
            "confidence_p50": round(self.confidence_p50, 4),
            "confidence_p90": round(self.confidence_p90, 4),
            "mce": round(self.mce, 4),
        }


class StatisticalMetricsEngine:
    """Computes all 17 institutional statistical metrics.

    Args:
        n_classes: Number of prediction classes.
        n_calibration_bins: Bins for ECE / MCE computation.

    Usage::

        engine = StatisticalMetricsEngine(n_classes=3)
        probs = torch.softmax(logits, dim=-1)   # (N, 3)
        metrics = engine.compute(probs, labels)  # labels: (N,)
    """

    def __init__(self, n_classes: int = 3, n_calibration_bins: int = 15) -> None:
        self._C = n_classes
        self._bins = n_calibration_bins

    def compute(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> StatisticalMetrics:
        """Compute all statistical metrics.

        Args:
            probs: ``(N, n_classes)`` softmax probabilities.
            labels: ``(N,)`` integer class labels.

        Returns:
            ``StatisticalMetrics`` with all 17 metrics.
        """
        p = probs.cpu().float().numpy()
        y = labels.cpu().long().numpy()
        n = len(y)
        C = self._C

        preds = p.argmax(axis=1)
        conf = p.max(axis=1)

        # 1. Accuracy
        accuracy = float((preds == y).mean())

        # 2-4. Precision / Recall / F1 (macro)
        precision_macro, recall_macro, f1_macro = self._prf1_macro(preds, y, C)

        # 5. ROC AUC (one-vs-rest, macro)
        roc_auc = self._roc_auc_macro(p, y, C)

        # 6. PR AUC (average precision, macro)
        pr_auc = self._pr_auc_macro(p, y, C)

        # 7-8. Log Loss / Cross Entropy (equivalent for softmax outputs)
        eps = 1e-9
        y_oh = np.eye(C)[y]
        log_loss = -float(np.mean(np.sum(y_oh * np.log(p + eps), axis=1)))
        cross_entropy = log_loss  # identical for one-hot targets

        # 9. MAE (on predicted probabilities vs one-hot targets)
        mae = float(np.mean(np.abs(p - y_oh)))

        # 10. RMSE
        rmse = float(np.sqrt(np.mean((p - y_oh) ** 2)))

        # 11. MAPE (on confidence vs correct probability)
        correct_probs = p[np.arange(n), y]
        with np.errstate(divide="ignore", invalid="ignore"):
            mape_vals = np.abs(1.0 - correct_probs) / np.where(
                correct_probs > 0, correct_probs, 1.0
            )
        mape = float(np.nanmean(mape_vals) * 100)

        # 12. R² (explained variance of confidence)
        var_total = float(conf.var()) if conf.var() > 0 else 1e-9
        correct_mask = (preds == y).astype(float)
        residuals = correct_mask - conf
        r_squared = max(-1.0, 1.0 - float(residuals.var() / var_total))

        # 13. ECE
        ece = self._ece(conf, preds, y, n)

        # 14. Brier Score (multi-class)
        brier_score = float(np.mean(np.sum((p - y_oh) ** 2, axis=1)))

        # 15. Prediction Entropy
        entropy = -np.sum(p * np.log(p + eps), axis=1)
        prediction_entropy = float(entropy.mean())

        # 16. Confidence distribution
        confidence_mean = float(conf.mean())
        confidence_std = float(conf.std())
        confidence_p10 = float(np.percentile(conf, 10))
        confidence_p50 = float(np.percentile(conf, 50))
        confidence_p90 = float(np.percentile(conf, 90))

        # 17. MCE (Maximum Calibration Error)
        mce = self._mce(conf, preds, y)

        return StatisticalMetrics(
            accuracy=accuracy,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            log_loss=log_loss,
            cross_entropy=cross_entropy,
            mae=mae,
            rmse=rmse,
            mape=mape,
            r_squared=r_squared,
            ece=ece,
            brier_score=brier_score,
            prediction_entropy=prediction_entropy,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            confidence_p10=confidence_p10,
            confidence_p50=confidence_p50,
            confidence_p90=confidence_p90,
            mce=mce,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _prf1_macro(preds: np.ndarray, labels: np.ndarray, C: int):
        precisions, recalls, f1s = [], [], []
        for c in range(C):
            tp = int(((preds == c) & (labels == c)).sum())
            fp = int(((preds == c) & (labels != c)).sum())
            fn = int(((preds != c) & (labels == c)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

    @staticmethod
    def _roc_auc_macro(p: np.ndarray, y: np.ndarray, C: int) -> float:
        aucs = []
        for c in range(C):
            y_bin = (y == c).astype(int)
            scores = p[:, c]
            auc = StatisticalMetricsEngine._auc_binary(scores, y_bin)
            aucs.append(auc)
        return float(np.mean(aucs))

    @staticmethod
    def _pr_auc_macro(p: np.ndarray, y: np.ndarray, C: int) -> float:
        aucs = []
        for c in range(C):
            y_bin = (y == c).astype(int)
            scores = p[:, c]
            aucs.append(StatisticalMetricsEngine._average_precision(scores, y_bin))
        return float(np.mean(aucs))

    @staticmethod
    def _auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
        """Trapezoidal AUC without sklearn."""
        order = np.argsort(-scores)
        labels_sorted = labels[order]
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tpr, fpr = [0.0], [0.0]
        tp = fp = 0
        for lbl in labels_sorted:
            if lbl == 1:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
        _trapz = getattr(np, "trapezoid", np.trapz)
        return float(_trapz(tpr, fpr))

    @staticmethod
    def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
        order = np.argsort(-scores)
        labels_sorted = labels[order]
        n_pos = labels.sum()
        if n_pos == 0:
            return 0.0
        precisions, recalls = [], []
        tp = 0
        for i, lbl in enumerate(labels_sorted):
            if lbl == 1:
                tp += 1
            precisions.append(tp / (i + 1))
            recalls.append(tp / n_pos)
        _trapz = getattr(np, "trapezoid", np.trapz)
        return float(_trapz(precisions, recalls)) if len(recalls) > 1 else 0.0

    def _ece(self, conf: np.ndarray, preds: np.ndarray, labels: np.ndarray, n: int) -> float:
        correct = (preds == labels).astype(float)
        bins = np.linspace(0, 1, self._bins + 1)
        ece = 0.0
        for i in range(self._bins):
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return float(ece)

    def _mce(self, conf: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> float:
        correct = (preds == labels).astype(float)
        bins = np.linspace(0, 1, self._bins + 1)
        max_ce = 0.0
        for i in range(self._bins):
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            max_ce = max(max_ce, abs(bin_acc - bin_conf))
        return float(max_ce)
