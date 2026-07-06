"""Phase 19.3 — HRSA Trend Engine.

Tracks the evolution of hierarchical scores over time at three levels:

  Metric trends     — individual raw metric trajectories
  Category trends   — Level 2 category score trajectories
  Composite trends  — Level 3 score trajectory

Generates:
  - Historical score series
  - Linear regression trend per category/metric
  - Improvement rate (avg pts gained per period)
  - Regression detection (score drop from recent peak)
  - Score stability (coefficient of variation)

Usage::

    engine = HRSATrendEngine()
    engine.record(hs)   # from a HierarchicalScore

    cat_trends = engine.category_trends()
    improvement = engine.improvement_rate(window=5)
    regressions = engine.detect_regressions(threshold=5.0)
    stability = engine.score_stability()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# TrendPoint — one data point in a trend series
# ---------------------------------------------------------------------------


@dataclass
class TrendPoint:
    recorded_at: str
    value: float
    experiment_id: str = ""
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_at": self.recorded_at,
            "value": round(self.value, 3),
            "experiment_id": self.experiment_id,
            "name": self.name,
        }


# ---------------------------------------------------------------------------
# TrendResult — regression analysis of a series
# ---------------------------------------------------------------------------


@dataclass
class TrendResult:
    series_name: str
    n: int
    slope: float  # pts per period (positive = improving)
    intercept: float
    trend_label: str  # "improving" | "stable" | "declining"
    r_squared: float
    mean: float
    std: float
    latest: float
    peak: float
    trough: float
    points: list[TrendPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_name": self.series_name,
            "n": self.n,
            "slope": round(self.slope, 4),
            "trend_label": self.trend_label,
            "r_squared": round(self.r_squared, 4),
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "latest": round(self.latest, 2),
            "peak": round(self.peak, 2),
            "trough": round(self.trough, 2),
            "points": [p.to_dict() for p in self.points],
        }


# ---------------------------------------------------------------------------
# RegressionAlert
# ---------------------------------------------------------------------------


@dataclass
class HRSARegressionAlert:
    series_name: str
    current_value: float
    peak_value: float
    drop: float
    severity: str
    detected_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_name": self.series_name,
            "current_value": round(self.current_value, 2),
            "peak_value": round(self.peak_value, 2),
            "drop": round(self.drop, 2),
            "severity": self.severity,
            "detected_at": self.detected_at,
        }


# ---------------------------------------------------------------------------
# Linear regression helper
# ---------------------------------------------------------------------------


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    ss_xx = sum((xi - x_mean) ** 2 for xi in xs)
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(xs, ys))
    slope = ss_xy / ss_xx if ss_xx != 0 else 0.0
    intercept = y_mean - slope * x_mean
    ss_tot = sum((yi - y_mean) ** 2 for yi in ys)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var**0.5


def _trend_label(slope: float, threshold: float = 0.5) -> str:
    if slope > threshold:
        return "improving"
    if slope < -threshold:
        return "declining"
    return "stable"


def _severity(drop: float) -> str:
    if drop >= 15.0:
        return "critical"
    if drop >= 10.0:
        return "high"
    if drop >= 5.0:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# HRSATrendEngine
# ---------------------------------------------------------------------------


class HRSATrendEngine:
    """Tracks score evolution at all three HRSA levels."""

    def __init__(self) -> None:
        # composite series: list of TrendPoints
        self._composite_series: list[TrendPoint] = []
        # category series: Dict[category_name, List[TrendPoint]]
        self._category_series: dict[str, list[TrendPoint]] = {}
        # metric series: Dict[metric_name, List[TrendPoint]]
        self._metric_series: dict[str, list[TrendPoint]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, hs: Any) -> None:
        """Record a HierarchicalScore into all trend series."""
        ts = datetime.now(UTC).isoformat()
        pt = TrendPoint(
            recorded_at=ts,
            value=hs.level3_composite,
            experiment_id=hs.experiment_id,
            name=hs.name,
        )
        self._composite_series.append(pt)

        for cat, result in hs.level2_categories.items():
            if result.metric_count > 0:
                self._category_series.setdefault(cat, []).append(
                    TrendPoint(ts, result.score, hs.experiment_id, cat)
                )
                # Level 1 metric trends
                for metric_name, metric_score in result.metric_scores.items():
                    self._metric_series.setdefault(metric_name, []).append(
                        TrendPoint(ts, metric_score, hs.experiment_id, metric_name)
                    )

    def record_raw(
        self,
        series_name: str,
        value: float,
        experiment_id: str = "",
        level: str = "composite",
    ) -> None:
        """Record a raw value into a named series."""
        ts = datetime.now(UTC).isoformat()
        pt = TrendPoint(ts, value, experiment_id, series_name)
        if level == "composite":
            self._composite_series.append(pt)
        elif level == "category":
            self._category_series.setdefault(series_name, []).append(pt)
        elif level == "metric":
            self._metric_series.setdefault(series_name, []).append(pt)

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    def _analyse_series(self, name: str, points: list[TrendPoint], window: int = 0) -> TrendResult:
        pts = points[-window:] if window > 0 else points
        values = [p.value for p in pts]
        if not values:
            return TrendResult(name, 0, 0.0, 0.0, "stable", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        xs = list(range(len(values)))
        slope, intercept, r2 = _linreg(xs, values)
        mean = sum(values) / len(values)
        sd = _std(values)
        return TrendResult(
            series_name=name,
            n=len(values),
            slope=slope,
            intercept=intercept,
            trend_label=_trend_label(slope),
            r_squared=r2,
            mean=mean,
            std=sd,
            latest=values[-1],
            peak=max(values),
            trough=min(values),
            points=pts,
        )

    def composite_trend(self, window: int = 0) -> TrendResult:
        return self._analyse_series("composite", self._composite_series, window)

    def category_trends(self, window: int = 0) -> dict[str, TrendResult]:
        return {
            cat: self._analyse_series(cat, pts, window)
            for cat, pts in self._category_series.items()
        }

    def category_trend(self, category: str, window: int = 0) -> TrendResult | None:
        pts = self._category_series.get(category)
        if pts is None:
            return None
        return self._analyse_series(category, pts, window)

    def metric_trends(self, window: int = 0) -> dict[str, TrendResult]:
        return {
            name: self._analyse_series(name, pts, window)
            for name, pts in self._metric_series.items()
        }

    def metric_trend(self, metric_name: str, window: int = 0) -> TrendResult | None:
        pts = self._metric_series.get(metric_name)
        if pts is None:
            return None
        return self._analyse_series(metric_name, pts, window)

    def improvement_rate(self, window: int = 10) -> float:
        """Average composite score improvement per period (last `window` records)."""
        pts = self._composite_series[-window:]
        if len(pts) < 2:
            return 0.0
        values = [p.value for p in pts]
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        return sum(diffs) / len(diffs)

    def detect_regressions(
        self, window: int = 5, threshold: float = 5.0
    ) -> list[HRSARegressionAlert]:
        """Detect regressions at composite and category level."""
        alerts: list[HRSARegressionAlert] = []

        # Composite regression
        comp_pts = self._composite_series[-window:]
        if len(comp_pts) >= 2:
            comp_vals = [p.value for p in comp_pts]
            peak = max(comp_vals)
            latest = comp_vals[-1]
            drop = peak - latest
            if drop >= threshold:
                alerts.append(HRSARegressionAlert("composite", latest, peak, drop, _severity(drop)))

        # Category regressions
        for cat, pts in self._category_series.items():
            recent = pts[-window:]
            if len(recent) < 2:
                continue
            vals = [p.value for p in recent]
            peak = max(vals)
            latest = vals[-1]
            drop = peak - latest
            if drop >= threshold:
                alerts.append(HRSARegressionAlert(cat, latest, peak, drop, _severity(drop)))

        return alerts

    def score_stability(self, window: int = 0) -> dict[str, float]:
        """Coefficient of variation (std / mean) for composite and categories.
        Lower = more stable."""
        result: dict[str, float] = {}

        def _cv(pts: list[TrendPoint], w: int) -> float:
            vals = [p.value for p in (pts[-w:] if w else pts)]
            if not vals or sum(vals) == 0:
                return 0.0
            mean = sum(vals) / len(vals)
            sd = _std(vals)
            return sd / mean if mean > 0 else 0.0

        result["composite"] = round(_cv(self._composite_series, window), 4)
        for cat, pts in self._category_series.items():
            result[cat] = round(_cv(pts, window), 4)
        return result

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def trend_report(self, window: int = 0) -> dict[str, Any]:
        comp = self.composite_trend(window)
        cat_trends = self.category_trends(window)
        regressions = self.detect_regressions()
        improvement = self.improvement_rate()
        stability = self.score_stability(window)

        return {
            "composite_trend": comp.to_dict(),
            "improvement_rate_per_period": round(improvement, 4),
            "stability": stability,
            "category_trends": {cat: t.to_dict() for cat, t in cat_trends.items()},
            "regression_alerts": [r.to_dict() for r in regressions],
            "total_recorded": len(self._composite_series),
            "categories_tracked": len(self._category_series),
            "metrics_tracked": len(self._metric_series),
        }

    def history(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._composite_series]

    def category_history(self, category: str) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._category_series.get(category, [])]


# Backward-compat alias
RegressionEvent = HRSARegressionAlert
