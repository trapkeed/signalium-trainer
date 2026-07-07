"""Phase 15 — Drift Analysis.

Detects distributional shifts that can degrade model performance over time.

Detects:
  - Feature Drift      — input distribution shift between windows
  - Prediction Drift   — shift in class probability distribution
  - Label Drift        — shift in ground-truth label distribution
  - Regime Drift       — shift in detected market regime frequencies
  - Performance Drift  — degradation in accuracy / trading metrics over time
  - Input Distribution Drift — mean, variance, and percentile shifts

Uses Population Stability Index (PSI) and Jensen-Shannon Divergence (JSD)
as drift statistics — no external dependencies beyond numpy.

Drift history is stored and can be queried across multiple validation runs.

Usage::

    analyzer = DriftAnalyzer(psi_threshold=0.2)
    report = analyzer.detect(
        x_ref=x_train, x_current=x_val,
        y_ref=y_train, y_current=y_val,
        probs_ref=probs_train, probs_current=probs_val,
    )
    print(report.summary())
    history = analyzer.history()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftStats:
    """Drift statistics for a single feature."""

    feature_idx: int
    feature_name: str
    psi: float
    jsd: float
    mean_shift: float
    std_shift: float
    is_drifted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "psi": round(self.psi, 4),
            "jsd": round(self.jsd, 4),
            "mean_shift": round(self.mean_shift, 4),
            "std_shift": round(self.std_shift, 4),
            "is_drifted": self.is_drifted,
        }


@dataclass
class DriftReport:
    """Full drift analysis report.

    Attributes:
        feature_drift: Per-feature drift statistics.
        prediction_drift_jsd: JSD between ref and current prediction distributions.
        label_drift_jsd: JSD between ref and current label distributions.
        performance_drift_pct: Accuracy change (current - reference) as %.
        regime_drift_jsd: JSD between regime distributions (if provided).
        n_drifted_features: Number of features classified as drifted.
        drift_severity: "none" | "minor" | "moderate" | "severe".
        detected_at: ISO-8601 timestamp.
    """

    feature_drift: list[FeatureDriftStats] = field(default_factory=list)
    prediction_drift_jsd: float = 0.0
    label_drift_jsd: float = 0.0
    performance_drift_pct: float = 0.0
    regime_drift_jsd: float = 0.0
    n_drifted_features: int = 0
    drift_severity: str = "none"
    detected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def summary(self) -> str:
        lines = [
            f"=== Drift Report ({self.detected_at}) ===",
            f"  Severity             : {self.drift_severity.upper()}",
            f"  Drifted Features     : {self.n_drifted_features}/{len(self.feature_drift)}",
            f"  Prediction Drift JSD : {self.prediction_drift_jsd:.4f}",
            f"  Label Drift JSD      : {self.label_drift_jsd:.4f}",
            f"  Performance Drift    : {self.performance_drift_pct:+.2f}%",
        ]
        if self.regime_drift_jsd > 0:
            lines.append(f"  Regime Drift JSD     : {self.regime_drift_jsd:.4f}")
        if self.n_drifted_features > 0:
            lines.append("  Most Drifted Features:")
            top = sorted(self.feature_drift, key=lambda f: f.psi, reverse=True)[:5]
            for f in top:
                lines.append(f"    {f.feature_name:<30s} PSI={f.psi:.3f}  JSD={f.jsd:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_severity": self.drift_severity,
            "n_drifted_features": self.n_drifted_features,
            "prediction_drift_jsd": round(self.prediction_drift_jsd, 4),
            "label_drift_jsd": round(self.label_drift_jsd, 4),
            "performance_drift_pct": round(self.performance_drift_pct, 4),
            "regime_drift_jsd": round(self.regime_drift_jsd, 4),
            "detected_at": self.detected_at,
            "feature_drift": [f.to_dict() for f in self.feature_drift],
        }


# ---------------------------------------------------------------------------
# Drift Analyzer
# ---------------------------------------------------------------------------


class DriftAnalyzer:
    """Detects distributional drift across validation windows.

    Args:
        psi_threshold: PSI threshold for feature drift classification.
            PSI < 0.1 = no drift, 0.1–0.25 = minor, > 0.25 = significant.
        jsd_threshold: JSD threshold for prediction/label drift.
        n_bins: Number of histogram bins for PSI.
        max_history: Maximum drift reports to keep in memory.

    PSI interpretation:
        < 0.10 : No drift
        0.10–0.25 : Minor drift
        > 0.25 : Significant drift (retrain signal)
    """

    PSI_THRESHOLDS = {0.10: "none", 0.25: "minor", float("inf"): "significant"}

    def __init__(
        self,
        psi_threshold: float = 0.20,
        jsd_threshold: float = 0.10,
        n_bins: int = 10,
        max_history: int = 100,
    ) -> None:
        self._psi_thresh = psi_threshold
        self._jsd_thresh = jsd_threshold
        self._bins = n_bins
        self._max_history = max_history
        self._history: list[DriftReport] = []

    def detect(
        self,
        x_ref: torch.Tensor,
        x_current: torch.Tensor,
        y_ref: torch.Tensor | None = None,
        y_current: torch.Tensor | None = None,
        probs_ref: torch.Tensor | None = None,
        probs_current: torch.Tensor | None = None,
        acc_ref: float = 0.0,
        acc_current: float = 0.0,
        regime_dist_ref: dict[str, float] | None = None,
        regime_dist_current: dict[str, float] | None = None,
        feature_names: list[str] | None = None,
    ) -> DriftReport:
        """Detect all drift types.

        Args:
            x_ref: Reference period features (train).
            x_current: Current period features (val).
            y_ref: Reference labels (optional).
            y_current: Current labels (optional).
            probs_ref: Reference prediction probabilities (optional).
            probs_current: Current prediction probabilities (optional).
            acc_ref: Reference accuracy (optional, for performance drift).
            acc_current: Current accuracy (optional).
            regime_dist_ref: Reference regime distribution dict.
            regime_dist_current: Current regime distribution dict.
            feature_names: Optional feature column names.

        Returns:
            ``DriftReport`` with all drift statistics.
        """
        # Flatten features to 2D
        x_r = x_ref.reshape(x_ref.shape[0], -1).cpu().float().numpy()
        x_c = x_current.reshape(x_current.shape[0], -1).cpu().float().numpy()
        n_feat = min(x_r.shape[1], x_c.shape[1])

        # Feature drift
        feature_drifts: list[FeatureDriftStats] = []
        for i in range(n_feat):
            psi = self._psi(x_r[:, i], x_c[:, i])
            jsd = self._jsd_histograms(x_r[:, i], x_c[:, i])
            mean_shift = float(x_c[:, i].mean() - x_r[:, i].mean())
            std_shift = float(x_c[:, i].std() - x_r[:, i].std())
            name = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
            feature_drifts.append(
                FeatureDriftStats(
                    feature_idx=i,
                    feature_name=name,
                    psi=psi,
                    jsd=jsd,
                    mean_shift=mean_shift,
                    std_shift=std_shift,
                    is_drifted=psi > self._psi_thresh,
                )
            )

        n_drifted = sum(1 for f in feature_drifts if f.is_drifted)

        # Prediction drift
        pred_jsd = 0.0
        if probs_ref is not None and probs_current is not None:
            p_r = probs_ref.cpu().float().numpy().mean(axis=0)
            p_c = probs_current.cpu().float().numpy().mean(axis=0)
            pred_jsd = float(self._jsd(p_r, p_c))

        # Label drift
        label_jsd = 0.0
        if y_ref is not None and y_current is not None:
            n_classes = max(int(y_ref.max().item()), int(y_current.max().item())) + 1
            dist_r = np.bincount(y_ref.cpu().numpy(), minlength=n_classes).astype(float)
            dist_c = np.bincount(y_current.cpu().numpy(), minlength=n_classes).astype(float)
            dist_r /= dist_r.sum() + 1e-9
            dist_c /= dist_c.sum() + 1e-9
            label_jsd = float(self._jsd(dist_r, dist_c))

        # Performance drift
        perf_drift_pct = (acc_current - acc_ref) * 100.0

        # Regime drift
        regime_jsd = 0.0
        if regime_dist_ref and regime_dist_current:
            keys = sorted(set(regime_dist_ref) | set(regime_dist_current))
            r_r = np.array([regime_dist_ref.get(k, 0.0) for k in keys])
            r_c = np.array([regime_dist_current.get(k, 0.0) for k in keys])
            eps = 1e-9
            r_r = (r_r + eps) / (r_r + eps).sum()
            r_c = (r_c + eps) / (r_c + eps).sum()
            regime_jsd = float(self._jsd(r_r, r_c))

        # Determine severity
        max_psi = max((f.psi for f in feature_drifts), default=0.0)
        severity = self._classify_severity(max_psi, pred_jsd, label_jsd)

        report = DriftReport(
            feature_drift=feature_drifts,
            prediction_drift_jsd=pred_jsd,
            label_drift_jsd=label_jsd,
            performance_drift_pct=perf_drift_pct,
            regime_drift_jsd=regime_jsd,
            n_drifted_features=n_drifted,
            drift_severity=severity,
        )

        self._history.append(report)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return report

    def history(self) -> list[DriftReport]:
        """Return all stored drift reports."""
        return list(self._history)

    def trend_summary(self) -> dict[str, Any]:
        """Compute drift trends across the stored history."""
        if not self._history:
            return {}
        pred_jsds = [r.prediction_drift_jsd for r in self._history]
        perf_drifts = [r.performance_drift_pct for r in self._history]
        return {
            "n_reports": len(self._history),
            "prediction_drift_trend": "increasing" if pred_jsds[-1] > pred_jsds[0] else "stable",
            "avg_prediction_drift_jsd": round(float(np.mean(pred_jsds)), 4),
            "avg_performance_drift_pct": round(float(np.mean(perf_drifts)), 4),
            "severities": [r.drift_severity for r in self._history],
        }

    # ── Statistical measures ──────────────────────────────────────────────

    def _psi(self, ref: np.ndarray, current: np.ndarray) -> float:
        """Population Stability Index."""
        eps = 1e-9
        bins = np.linspace(
            min(ref.min(), current.min()) - eps,
            max(ref.max(), current.max()) + eps,
            self._bins + 1,
        )
        r_hist, _ = np.histogram(ref, bins=bins)
        c_hist, _ = np.histogram(current, bins=bins)
        r_pct = (r_hist / max(r_hist.sum(), 1)) + eps
        c_pct = (c_hist / max(c_hist.sum(), 1)) + eps
        return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))

    def _jsd_histograms(self, ref: np.ndarray, current: np.ndarray) -> float:
        """JSD from histograms."""
        eps = 1e-9
        bins = np.linspace(
            min(ref.min(), current.min()) - eps,
            max(ref.max(), current.max()) + eps,
            self._bins + 1,
        )
        r, _ = np.histogram(ref, bins=bins)
        c, _ = np.histogram(current, bins=bins)
        r = (r / max(r.sum(), 1)).astype(float) + eps
        c = (c / max(c.sum(), 1)).astype(float) + eps
        return float(self._jsd(r / r.sum(), c / c.sum()))

    @staticmethod
    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon Divergence (symmetric, bounded [0, 1])."""
        eps = 1e-9
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
        jsd = 0.5 * (kl_pm + kl_qm)
        return float(np.clip(jsd, 0, 1))

    @staticmethod
    def _classify_severity(max_psi: float, pred_jsd: float, label_jsd: float) -> str:
        if max_psi > 0.25 or pred_jsd > 0.20 or label_jsd > 0.15:
            return "severe"
        if max_psi > 0.10 or pred_jsd > 0.10 or label_jsd > 0.08:
            return "moderate"
        if max_psi > 0.05 or pred_jsd > 0.05:
            return "minor"
        return "none"
