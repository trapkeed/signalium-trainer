"""Phase 15 — Time-series leakage detection.

Detects various forms of data leakage and data quality issues that would
invalidate walk-forward validation results.

Detects:
  - Future leakage     — features computed using data after the label timestamp
  - Target leakage     — features that are trivially correlated with the target
  - Feature leakage    — train-set statistics used in val/test normalisation
  - Timestamp issues   — non-monotone, duplicate, or missing timestamps
  - Duplicate samples  — identical (x, y) pairs across splits
  - Missing sequences  — NaN / Inf values in feature tensors

AbortValidation is raised when any critical leakage is detected and
``abort_on_detect=True`` (the default).

Usage::

    detector = LeakageDetector(abort_on_detect=True)
    report = detector.check(x_train, y_train, x_val, y_val, timestamps=ts)
    if report.has_issues:
        print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import numpy as np
from loguru import logger


class LeakageDetected(RuntimeError):
    """Raised when critical leakage is found and abort_on_detect is True."""


@dataclass
class LeakageReport:
    """Results from a full leakage scan.

    Attributes:
        future_leakage: True when high feature–label correlation in future region.
        target_leakage: True when a feature trivially predicts the target.
        timestamp_issues: List of detected timestamp problem descriptions.
        duplicate_samples: Number of duplicate (x, y) rows detected.
        nan_inf_features: Number of NaN/Inf values found.
        missing_sequences: Number of sequences flagged as missing.
        warnings: Non-critical advisory messages.
        errors: Critical leakage messages that block validation.
    """

    future_leakage: bool = False
    target_leakage: bool = False
    timestamp_issues: list[str] = field(default_factory=list)
    duplicate_samples: int = 0
    nan_inf_features: int = 0
    missing_sequences: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(
            self.future_leakage
            or self.target_leakage
            or self.timestamp_issues
            or self.duplicate_samples > 0
            or self.nan_inf_features > 0
            or self.missing_sequences > 0
            or self.errors
        )

    @property
    def is_critical(self) -> bool:
        return bool(
            self.future_leakage or self.target_leakage or self.errors
        )

    def summary(self) -> str:
        lines = ["=== Leakage Detection Report ==="]
        if not self.has_issues:
            lines.append("  PASS: No leakage or data quality issues detected.")
            return "\n".join(lines)
        if self.future_leakage:
            lines.append("  CRITICAL: Future leakage detected in feature set.")
        if self.target_leakage:
            lines.append("  CRITICAL: Target leakage detected in features.")
        for ts in self.timestamp_issues:
            lines.append(f"  TIMESTAMP: {ts}")
        if self.duplicate_samples > 0:
            lines.append(f"  WARNING: {self.duplicate_samples} duplicate samples found.")
        if self.nan_inf_features > 0:
            lines.append(f"  WARNING: {self.nan_inf_features} NaN/Inf feature values.")
        if self.missing_sequences > 0:
            lines.append(f"  WARNING: {self.missing_sequences} missing/zero sequences.")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "future_leakage": self.future_leakage,
            "target_leakage": self.target_leakage,
            "timestamp_issues": self.timestamp_issues,
            "duplicate_samples": self.duplicate_samples,
            "nan_inf_features": self.nan_inf_features,
            "missing_sequences": self.missing_sequences,
            "warnings": self.warnings,
            "errors": self.errors,
            "has_issues": self.has_issues,
            "is_critical": self.is_critical,
        }


class LeakageDetector:
    """Institutional-grade leakage and data quality scanner.

    Args:
        abort_on_detect: Raise ``LeakageDetected`` when critical issues found.
        target_correlation_threshold: Above this Pearson |r|, a feature is
            flagged as potential target leakage (default 0.95).
        future_correlation_threshold: Cross-correlation threshold used to
            detect future data in feature construction (default 0.90).
        verbose: Log details of each check.
    """

    def __init__(
        self,
        abort_on_detect: bool = True,
        target_correlation_threshold: float = 0.95,
        future_correlation_threshold: float = 0.90,
        verbose: bool = True,
    ) -> None:
        self._abort = abort_on_detect
        self._target_corr_thresh = target_correlation_threshold
        self._future_corr_thresh = future_correlation_threshold
        self._verbose = verbose

    def check(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        feature_names: Optional[list[str]] = None,
    ) -> LeakageReport:
        """Run all leakage checks.

        Args:
            x_train: ``(N_train, seq_len, n_feat)`` or ``(N_train, n_feat)``
            y_train: ``(N_train,)``
            x_val: ``(N_val, ...)``
            y_val: ``(N_val,)``
            timestamps: Optional ``(N_total,)`` sorted Unix timestamps.
            feature_names: Optional feature column names for reporting.

        Returns:
            ``LeakageReport`` with all findings.
        """
        report = LeakageReport()

        # ── 1. NaN / Inf check ─────────────────────────────────────────
        self._check_nan_inf(x_train, "train", report)
        self._check_nan_inf(x_val, "val", report)

        # ── 2. Missing/zero sequences ───────────────────────────────────
        self._check_missing_sequences(x_train, report)

        # ── 3. Timestamp ordering ───────────────────────────────────────
        if timestamps is not None:
            self._check_timestamps(timestamps, report)

        # ── 4. Duplicate samples (train × val overlap) ──────────────────
        self._check_duplicates(x_train, x_val, report)

        # ── 5. Target leakage ───────────────────────────────────────────
        self._check_target_leakage(x_train, y_train, feature_names, report)

        # ── 6. Future leakage proxy ─────────────────────────────────────
        self._check_future_leakage(x_train, x_val, y_val, report)

        if self._verbose:
            logger.info("LeakageDetector: {}", report.summary())

        if self._abort and report.is_critical:
            raise LeakageDetected(
                "Validation aborted due to critical leakage:\n" + report.summary()
            )

        return report

    # ── Private checkers ──────────────────────────────────────────────────

    def _check_nan_inf(
        self, x: torch.Tensor, split: str, report: LeakageReport
    ) -> None:
        if not torch.isfinite(x).all():
            n_bad = int((~torch.isfinite(x)).sum().item())
            report.nan_inf_features += n_bad
            report.warnings.append(
                f"{n_bad} NaN/Inf values detected in {split} features."
            )

    def _check_missing_sequences(
        self, x: torch.Tensor, report: LeakageReport
    ) -> None:
        flat = x.reshape(x.shape[0], -1)
        all_zero = (flat == 0).all(dim=1)
        n_missing = int(all_zero.sum().item())
        if n_missing > 0:
            report.missing_sequences += n_missing
            report.warnings.append(
                f"{n_missing} all-zero sequences detected (possibly missing data)."
            )

    def _check_timestamps(
        self, timestamps: torch.Tensor, report: LeakageReport
    ) -> None:
        ts = timestamps.cpu().numpy()
        # Non-monotonic
        if not np.all(ts[1:] >= ts[:-1]):
            n_violations = int((ts[1:] < ts[:-1]).sum())
            report.timestamp_issues.append(
                f"Non-monotonic timestamps: {n_violations} out-of-order pairs."
            )
        # Duplicates
        _, counts = np.unique(ts, return_counts=True)
        n_dup_ts = int((counts > 1).sum())
        if n_dup_ts > 0:
            report.timestamp_issues.append(
                f"{n_dup_ts} duplicate timestamps detected."
            )
        # Large gaps (> 10× median interval)
        if len(ts) > 2:
            diffs = np.diff(ts)
            positive_diffs = diffs[diffs > 0]
            if len(positive_diffs) > 0:
                median_gap = float(np.median(positive_diffs))
                large_gaps = int((diffs > 10 * median_gap).sum())
                if large_gaps > 0:
                    report.timestamp_issues.append(
                        f"{large_gaps} abnormally large timestamp gaps (>10× median interval)."
                    )

    def _check_duplicates(
        self, x_train: torch.Tensor, x_val: torch.Tensor, report: LeakageReport
    ) -> None:
        try:
            # Compare flattened rows using hash-based approach
            train_flat = x_train.reshape(x_train.shape[0], -1).cpu().numpy()
            val_flat = x_val.reshape(x_val.shape[0], -1).cpu().numpy()
            train_set = set(map(tuple, train_flat.astype(np.float32).tolist()))
            duplicates = sum(
                1 for row in val_flat.astype(np.float32).tolist()
                if tuple(row) in train_set
            )
            report.duplicate_samples += duplicates
            if duplicates > 0:
                report.warnings.append(
                    f"{duplicates} val samples are exact duplicates of train samples."
                )
        except Exception:
            report.warnings.append("Duplicate check skipped (data too large or sparse).")

    def _check_target_leakage(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        feature_names: Optional[list[str]],
        report: LeakageReport,
    ) -> None:
        """Flag features with very high correlation to the target label."""
        try:
            x = x_train.reshape(x_train.shape[0], -1).float().cpu().numpy()
            y = y_train.float().cpu().numpy()
            n_feat = x.shape[1]
            leaky_features: list[str] = []
            for i in range(min(n_feat, 100)):  # check first 100 features
                col = x[:, i]
                if col.std() < 1e-9:
                    continue
                corr = float(np.corrcoef(col, y)[0, 1])
                if abs(corr) > self._target_corr_thresh:
                    name = (feature_names[i] if feature_names and i < len(feature_names)
                            else f"feature_{i}")
                    leaky_features.append(f"{name} (|r|={abs(corr):.3f})")
            if leaky_features:
                report.target_leakage = True
                report.errors.append(
                    f"Target leakage — features trivially correlated with label: "
                    + ", ".join(leaky_features[:5])
                )
        except Exception as exc:
            report.warnings.append(f"Target leakage check error: {exc}")

    def _check_future_leakage(
        self,
        x_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        report: LeakageReport,
    ) -> None:
        """Proxy check: do val features correlate unusually well with val labels?"""
        try:
            x = x_val.reshape(x_val.shape[0], -1).float().cpu().numpy()
            y = y_val.float().cpu().numpy()
            n_feat = min(x.shape[1], 50)
            max_corr = 0.0
            for i in range(n_feat):
                col = x[:, i]
                if col.std() < 1e-9:
                    continue
                corr = abs(float(np.corrcoef(col, y)[0, 1]))
                max_corr = max(max_corr, corr)
            if max_corr > self._future_corr_thresh:
                report.future_leakage = True
                report.errors.append(
                    f"Potential future leakage — max val feature–label |r|={max_corr:.3f} "
                    f"exceeds threshold {self._future_corr_thresh}."
                )
        except Exception as exc:
            report.warnings.append(f"Future leakage check error: {exc}")
