"""Phase 15 — Visualization Data Builder.

Generates all chart data structures required by the institutional validation
platform.  All outputs are JSON-safe dicts — no plotting library required.

Charts generated:
  - Walk Forward Timeline
  - Equity Curves (with Monte Carlo bands)
  - Performance Heat Maps (instrument × timeframe)
  - Confusion Matrices
  - Residual Plots
  - Calibration Curves (reliability diagrams)
  - Feature Drift Charts
  - Performance Trends
  - Leaderboard Chart
  - Regime Performance
  - Trade Distribution
  - Risk Distribution

Usage::

    builder = VisualizationDataBuilder()
    data = builder.build_equity_curves(mc_result)
    data = builder.build_heatmap(multi_asset_result)
    data = builder.build_calibration_curve(calibration_report)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class VisualizationDataBuilder:
    """Builds all Phase 15 chart data structures.

    All methods return JSON-safe dicts ready for use with any frontend
    charting library (e.g. Recharts, Chart.js, Plotly).
    """

    # ── Walk Forward Timeline ─────────────────────────────────────────────

    @staticmethod
    def build_walk_forward_timeline(folds: list[Any]) -> dict[str, Any]:
        """Build walk-forward fold timeline data.

        Args:
            folds: List of ``WindowFold`` objects.

        Returns:
            Dict with fold start/end indices for timeline rendering.
        """
        series = []
        for fold in folds:
            series.append(
                {
                    "fold": fold.fold_index,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "val_start": fold.val_start,
                    "val_end": fold.val_end,
                    "test_start": fold.test_start if fold.has_test else None,
                    "test_end": fold.test_end if fold.has_test else None,
                    "strategy": fold.strategy,
                }
            )
        return {
            "chart_type": "walk_forward_timeline",
            "folds": series,
            "n_folds": len(folds),
        }

    # ── Equity Curves ─────────────────────────────────────────────────────

    @staticmethod
    def build_equity_curves(mc_result: Any) -> dict[str, Any]:
        """Build equity curve chart data with Monte Carlo percentile bands.

        Args:
            mc_result: ``MonteCarloResult`` from ``MonteCarloValidator``.

        Returns:
            Chart data with P5, P25, P50, P75, P95 equity curves.
        """
        if mc_result.equity_curves.ndim < 2:
            return {"chart_type": "equity_curves", "error": "No data"}

        n_steps = mc_result.equity_curves.shape[1]
        x = list(range(n_steps))

        return {
            "chart_type": "equity_curves",
            "x_axis": x,
            "p5": mc_result.get_percentile_curve(5).tolist(),
            "p25": mc_result.get_percentile_curve(25).tolist(),
            "p50": mc_result.get_percentile_curve(50).tolist(),
            "p75": mc_result.get_percentile_curve(75).tolist(),
            "p95": mc_result.get_percentile_curve(95).tolist(),
            "initial_capital": mc_result.initial_capital,
            "ruin_probability": round(mc_result.ruin_probability, 4),
            "n_simulations": mc_result.n_simulations,
        }

    # ── Performance Heat Map ──────────────────────────────────────────────

    @staticmethod
    def build_heatmap(
        multi_asset_result: Any,
        metric: str = "mean_accuracy",
    ) -> dict[str, Any]:
        """Build instrument × timeframe performance heat map.

        Args:
            multi_asset_result: ``MultiAssetValidationResult``.
            metric: The metric to display (e.g. "mean_accuracy", "mean_sharpe").

        Returns:
            Heat map chart data dict.
        """
        instruments = sorted(multi_asset_result.results.keys())
        timeframes = sorted(
            set(tf for tf_map in multi_asset_result.results.values() for tf in tf_map),
            key=lambda tf: {
                "1M": 1,
                "5M": 2,
                "15M": 3,
                "30M": 4,
                "1H": 5,
                "4H": 6,
                "1D": 7,
                "1W": 8,
            }.get(tf, 99),
        )

        matrix: list[list[float | None]] = []
        for instr in instruments:
            row: list[float | None] = []
            for tf in timeframes:
                r = multi_asset_result.results.get(instr, {}).get(tf)
                if r and r.status == "ok":
                    val = round(float(getattr(r, metric, 0.0)), 4)
                else:
                    val = None
                row.append(val)
            matrix.append(row)

        return {
            "chart_type": "heatmap",
            "metric": metric,
            "x_labels": timeframes,
            "y_labels": instruments,
            "matrix": matrix,
        }

    # ── Confusion Matrix ──────────────────────────────────────────────────

    @staticmethod
    def build_confusion_matrix(
        probs: Any,
        labels: Any,
        class_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build confusion matrix chart data.

        Args:
            probs: ``(N, C)`` torch tensor or numpy array.
            labels: ``(N,)`` integer labels.
            class_names: Optional class labels.

        Returns:
            Normalised confusion matrix dict.
        """
        try:

            if hasattr(probs, "cpu"):
                p = probs.cpu().float().numpy()
                y = labels.cpu().long().numpy()
            else:
                p = np.array(probs, dtype=float)
                y = np.array(labels, dtype=int)
        except Exception:
            p = np.array(probs, dtype=float)
            y = np.array(labels, dtype=int)

        C = p.shape[1]
        preds = p.argmax(axis=1)
        cm = np.zeros((C, C), dtype=int)
        for true, pred in zip(y, preds):
            cm[true, pred] += 1

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = (cm / np.where(row_sums > 0, row_sums, 1)).round(4)
        names = class_names or [str(i) for i in range(C)]

        return {
            "chart_type": "confusion_matrix",
            "class_names": names,
            "matrix_raw": cm.tolist(),
            "matrix_normalised": cm_norm.tolist(),
            "n_samples": len(y),
        }

    # ── Calibration Curve ─────────────────────────────────────────────────

    @staticmethod
    def build_calibration_curve(calibration_report: Any) -> dict[str, Any]:
        """Build reliability diagram / calibration curve data.

        Args:
            calibration_report: ``CalibrationReport``.

        Returns:
            Chart data with ideal diagonal and actual calibration line.
        """
        return {
            "chart_type": "calibration_curve",
            "bin_midpoints": calibration_report.bin_midpoints,
            "actual_accuracy": calibration_report.bin_accuracies,
            "predicted_confidence": calibration_report.bin_confidences,
            "ideal_diagonal": calibration_report.bin_midpoints,
            "bin_fractions": calibration_report.bin_fractions,
            "ece": round(calibration_report.ece, 4),
            "mce": round(calibration_report.mce, 4),
        }

    # ── Feature Drift Chart ───────────────────────────────────────────────

    @staticmethod
    def build_drift_chart(drift_report: Any) -> dict[str, Any]:
        """Build feature drift bar chart data.

        Args:
            drift_report: ``DriftReport``.

        Returns:
            Bar chart data sorted by PSI descending.
        """
        top_features = sorted(
            drift_report.feature_drift,
            key=lambda f: f.psi,
            reverse=True,
        )[:20]
        return {
            "chart_type": "feature_drift",
            "feature_names": [f.feature_name for f in top_features],
            "psi_values": [round(f.psi, 4) for f in top_features],
            "jsd_values": [round(f.jsd, 4) for f in top_features],
            "is_drifted": [f.is_drifted for f in top_features],
            "psi_minor_threshold": 0.10,
            "psi_major_threshold": 0.25,
            "severity": drift_report.drift_severity,
        }

    # ── Performance Trend ─────────────────────────────────────────────────

    @staticmethod
    def build_performance_trend(
        fold_metrics: list[dict[str, float]],
        metric: str = "accuracy",
    ) -> dict[str, Any]:
        """Build performance-over-time chart from walk-forward fold results.

        Args:
            fold_metrics: List of fold metric dicts (from InstrumentResult.fold_results).
            metric: Metric to plot.

        Returns:
            Line chart data.
        """
        values = [float(f.get(metric, 0.0)) for f in fold_metrics]
        n = len(values)
        trend_direction = (
            "improving"
            if n > 2 and values[-1] > values[0]
            else "declining" if n > 2 and values[-1] < values[0] else "stable"
        )
        return {
            "chart_type": "performance_trend",
            "metric": metric,
            "fold_indices": list(range(n)),
            "values": [round(v, 4) for v in values],
            "mean": round(float(np.mean(values)) if values else 0.0, 4),
            "std": round(float(np.std(values)) if values else 0.0, 4),
            "trend_direction": trend_direction,
        }

    # ── Regime Performance ────────────────────────────────────────────────

    @staticmethod
    def build_regime_chart(
        regime_performance: dict[str, dict[str, float]],
        metric: str = "accuracy",
    ) -> dict[str, Any]:
        """Build regime performance bar chart.

        Args:
            regime_performance: Dict mapping regime → metric_dict.
            metric: Metric to display.

        Returns:
            Bar chart data sorted by metric value.
        """
        sorted_regimes = sorted(
            regime_performance.items(),
            key=lambda x: x[1].get(metric, 0.0),
            reverse=True,
        )
        return {
            "chart_type": "regime_performance",
            "metric": metric,
            "regime_names": [r[0] for r in sorted_regimes],
            "values": [round(float(r[1].get(metric, 0.0)), 4) for r in sorted_regimes],
        }

    # ── Trade Distribution ────────────────────────────────────────────────

    @staticmethod
    def build_trade_distribution(trades: list[Any]) -> dict[str, Any]:
        """Build trade P&L distribution histogram.

        Args:
            trades: List of ``TradeResult`` objects.

        Returns:
            Histogram data for trade P&L distribution.
        """
        if not trades:
            return {"chart_type": "trade_distribution", "error": "No trades"}

        pnls = np.array([t.pnl_pips for t in trades if t.direction != 0])
        if len(pnls) == 0:
            return {"chart_type": "trade_distribution", "error": "No active trades"}

        counts, bin_edges = np.histogram(pnls, bins=30)
        midpoints = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(counts))]

        return {
            "chart_type": "trade_distribution",
            "bin_midpoints": [round(float(m), 2) for m in midpoints],
            "counts": counts.tolist(),
            "mean_pnl": round(float(pnls.mean()), 2),
            "median_pnl": round(float(np.median(pnls)), 2),
            "win_rate": round(float((pnls > 0).mean()), 4),
            "n_trades": len(pnls),
        }

    # ── Risk Distribution ─────────────────────────────────────────────────

    @staticmethod
    def build_risk_distribution(
        trading_metrics: Any,
        mc_result: Any | None = None,
    ) -> dict[str, Any]:
        """Build risk distribution chart data.

        Args:
            trading_metrics: ``TradingMetrics`` object.
            mc_result: Optional ``MonteCarloResult`` for confidence bands.

        Returns:
            Risk distribution data dict.
        """
        data: dict[str, Any] = {
            "chart_type": "risk_distribution",
            "max_drawdown": round(trading_metrics.max_drawdown, 4),
            "ulcer_index": round(trading_metrics.ulcer_index, 4),
            "tail_risk_cvar5": round(trading_metrics.tail_risk_cvar5, 4),
            "risk_of_ruin": round(trading_metrics.risk_of_ruin, 4),
            "sharpe_ratio": round(trading_metrics.sharpe_ratio, 4),
            "sortino_ratio": round(trading_metrics.sortino_ratio, 4),
        }
        if mc_result is not None:
            data["mc_max_drawdown_p50"] = round(mc_result.max_drawdown_p50, 4)
            data["mc_max_drawdown_p95"] = round(mc_result.max_drawdown_p95, 4)
            data["mc_ruin_probability"] = round(mc_result.ruin_probability, 4)
        return data
