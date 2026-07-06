"""Phase 15 — Failure Analysis.

Automatically identifies where the model performs worst and generates
improvement recommendations.

Identifies:
  - Worst instruments
  - Worst timeframes
  - Worst sessions (time-of-day buckets)
  - Worst market regimes
  - Worst prediction heads
  - Worst confidence ranges
  - Worst market conditions

All analysis is read-only — no model or data modification.

Usage::

    analyzer = FailureAnalyzer()
    report = analyzer.analyze(
        instrument_results=multi_asset_result,
        regime_performance=regime_perf,
        head_results=head_results,
    )
    print(report.summary())
    print(report.recommendations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Failure report
# ---------------------------------------------------------------------------


@dataclass
class WorstPerformer:
    """A single worst-performing dimension entry.

    Attributes:
        dimension: What was measured (instrument, timeframe, regime, etc.).
        identifier: Specific value (e.g. "BTCUSD", "1M", "high_volatility").
        metric: Metric name (e.g. "accuracy", "sharpe_ratio").
        value: Metric value.
        rank: 1-indexed rank among worst performers (1 = worst).
    """

    dimension: str
    identifier: str
    metric: str
    value: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "identifier": self.identifier,
            "metric": self.metric,
            "value": round(self.value, 4),
            "rank": self.rank,
        }


@dataclass
class FailureReport:
    """Complete failure analysis report.

    Attributes:
        worst_instruments: Ranked worst instruments.
        worst_timeframes: Ranked worst timeframes.
        worst_sessions: Ranked worst time-of-day sessions.
        worst_regimes: Ranked worst market regimes.
        worst_heads: Ranked worst prediction heads.
        worst_confidence_ranges: Ranges where confidence is least reliable.
        worst_conditions: General worst-condition summary.
        recommendations: Actionable improvement recommendations.
    """

    worst_instruments: list[WorstPerformer] = field(default_factory=list)
    worst_timeframes: list[WorstPerformer] = field(default_factory=list)
    worst_sessions: list[WorstPerformer] = field(default_factory=list)
    worst_regimes: list[WorstPerformer] = field(default_factory=list)
    worst_heads: list[WorstPerformer] = field(default_factory=list)
    worst_confidence_ranges: list[dict[str, Any]] = field(default_factory=list)
    worst_conditions: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=== Failure Analysis Report ===", ""]
        sections = [
            ("Worst Instruments", self.worst_instruments[:5]),
            ("Worst Timeframes", self.worst_timeframes[:5]),
            ("Worst Sessions", self.worst_sessions[:5]),
            ("Worst Market Regimes", self.worst_regimes[:5]),
            ("Worst Prediction Heads", self.worst_heads[:5]),
        ]
        for title, performers in sections:
            if performers:
                lines.append(f"  {title}:")
                for p in performers:
                    lines.append(f"    #{p.rank} {p.identifier:<25s} {p.metric}={p.value:.4f}")
        if self.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    - {r}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worst_instruments": [w.to_dict() for w in self.worst_instruments],
            "worst_timeframes": [w.to_dict() for w in self.worst_timeframes],
            "worst_sessions": [w.to_dict() for w in self.worst_sessions],
            "worst_regimes": [w.to_dict() for w in self.worst_regimes],
            "worst_heads": [w.to_dict() for w in self.worst_heads],
            "worst_confidence_ranges": self.worst_confidence_ranges,
            "worst_conditions": self.worst_conditions,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Failure Analyzer
# ---------------------------------------------------------------------------


class FailureAnalyzer:
    """Automatically identifies failure patterns and generates recommendations.

    Args:
        top_k: Number of worst performers to surface per dimension.
        accuracy_threshold: Below this accuracy, an item is flagged as failing.
        sharpe_threshold: Below this Sharpe, a condition is flagged.
    """

    def __init__(
        self,
        top_k: int = 5,
        accuracy_threshold: float = 0.45,
        sharpe_threshold: float = 0.5,
    ) -> None:
        self._k = top_k
        self._acc_thresh = accuracy_threshold
        self._sharpe_thresh = sharpe_threshold

    def analyze(
        self,
        instrument_results: Any | None = None,
        regime_performance: dict[str, dict[str, float]] | None = None,
        head_results: dict[str, dict[str, float]] | None = None,
        session_performance: dict[str, dict[str, float]] | None = None,
        confidence_bins: list[dict[str, Any]] | None = None,
    ) -> FailureReport:
        """Run full failure analysis.

        Args:
            instrument_results: ``MultiAssetValidationResult`` from Phase 15
                validation (or None).
            regime_performance: Dict mapping regime_name → metric_dict.
            head_results: Dict mapping head_name → metric_dict.
            session_performance: Dict mapping session_label → metric_dict.
            confidence_bins: List of dicts with "range", "accuracy", "n_samples".

        Returns:
            ``FailureReport`` with ranked worst performers and recommendations.
        """
        worst_instruments: list[WorstPerformer] = []
        worst_timeframes: list[WorstPerformer] = []
        worst_sessions: list[WorstPerformer] = []
        worst_regimes: list[WorstPerformer] = []
        worst_heads: list[WorstPerformer] = []
        worst_confidence: list[dict[str, Any]] = []
        worst_conditions: list[str] = []

        # ── Instrument analysis ─────────────────────────────────────────
        if instrument_results is not None:
            all_results = instrument_results.all_results()
            ok = [r for r in all_results if r.status == "ok"]
            by_instr: dict[str, list[float]] = {}
            by_tf: dict[str, list[float]] = {}
            for r in ok:
                by_instr.setdefault(r.instrument, []).append(r.mean_accuracy)
                by_tf.setdefault(r.timeframe, []).append(r.mean_accuracy)

            instr_accs = {k: float(np.mean(v)) for k, v in by_instr.items()}
            ranked = sorted(instr_accs.items(), key=lambda x: x[1])
            for i, (instr, acc) in enumerate(ranked[: self._k]):
                worst_instruments.append(
                    WorstPerformer(
                        dimension="instrument",
                        identifier=instr,
                        metric="accuracy",
                        value=acc,
                        rank=i + 1,
                    )
                )

            tf_accs = {k: float(np.mean(v)) for k, v in by_tf.items()}
            ranked_tf = sorted(tf_accs.items(), key=lambda x: x[1])
            for i, (tf, acc) in enumerate(ranked_tf[: self._k]):
                worst_timeframes.append(
                    WorstPerformer(
                        dimension="timeframe",
                        identifier=tf,
                        metric="accuracy",
                        value=acc,
                        rank=i + 1,
                    )
                )

        # ── Regime analysis ─────────────────────────────────────────────
        if regime_performance:
            ranked_r = sorted(
                regime_performance.items(),
                key=lambda x: x[1].get("accuracy", 0.0),
            )
            for i, (regime, metrics) in enumerate(ranked_r[: self._k]):
                worst_regimes.append(
                    WorstPerformer(
                        dimension="regime",
                        identifier=regime,
                        metric="accuracy",
                        value=metrics.get("accuracy", 0.0),
                        rank=i + 1,
                    )
                )

        # ── Prediction head analysis ────────────────────────────────────
        if head_results:
            ranked_h = sorted(
                head_results.items(),
                key=lambda x: x[1].get("accuracy", 0.0),
            )
            for i, (head, metrics) in enumerate(ranked_h[: self._k]):
                worst_heads.append(
                    WorstPerformer(
                        dimension="head",
                        identifier=head,
                        metric="accuracy",
                        value=metrics.get("accuracy", 0.0),
                        rank=i + 1,
                    )
                )

        # ── Session analysis ────────────────────────────────────────────
        if session_performance:
            ranked_s = sorted(
                session_performance.items(),
                key=lambda x: x[1].get("accuracy", 0.0),
            )
            for i, (session, metrics) in enumerate(ranked_s[: self._k]):
                worst_sessions.append(
                    WorstPerformer(
                        dimension="session",
                        identifier=session,
                        metric="accuracy",
                        value=metrics.get("accuracy", 0.0),
                        rank=i + 1,
                    )
                )

        # ── Confidence range analysis ───────────────────────────────────
        if confidence_bins:
            for b in confidence_bins:
                if b.get("accuracy", 1.0) < self._acc_thresh:
                    worst_confidence.append(b)

        # ── Generate worst conditions list ──────────────────────────────
        if worst_instruments:
            worst_conditions.append(
                f"Instrument: {worst_instruments[0].identifier} "
                f"(accuracy={worst_instruments[0].value:.3f})"
            )
        if worst_timeframes:
            worst_conditions.append(
                f"Timeframe: {worst_timeframes[0].identifier} "
                f"(accuracy={worst_timeframes[0].value:.3f})"
            )
        if worst_regimes:
            worst_conditions.append(
                f"Regime: {worst_regimes[0].identifier} " f"(accuracy={worst_regimes[0].value:.3f})"
            )

        # ── Generate recommendations ────────────────────────────────────
        recommendations = self._generate_recommendations(
            worst_instruments,
            worst_timeframes,
            worst_regimes,
            worst_heads,
            worst_sessions,
            worst_confidence,
        )

        return FailureReport(
            worst_instruments=worst_instruments,
            worst_timeframes=worst_timeframes,
            worst_sessions=worst_sessions,
            worst_regimes=worst_regimes,
            worst_heads=worst_heads,
            worst_confidence_ranges=worst_confidence,
            worst_conditions=worst_conditions,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        worst_instruments,
        worst_timeframes,
        worst_regimes,
        worst_heads,
        worst_sessions,
        worst_confidence,
    ) -> list[str]:
        recs: list[str] = []

        # Instrument-specific recommendations
        if worst_instruments:
            w = worst_instruments[0]
            if w.value < self._acc_thresh:
                if "BTC" in w.identifier or "ETH" in w.identifier:
                    recs.append(
                        f"Crypto instrument {w.identifier} shows poor accuracy ({w.value:.3f}). "
                        "Consider adding regime-specific training data for crypto markets "
                        "or training a dedicated crypto model."
                    )
                elif "XAU" in w.identifier or "XAG" in w.identifier:
                    recs.append(
                        f"Commodity {w.identifier} accuracy is low ({w.value:.3f}). "
                        "Consider adding macro and geopolitical event features."
                    )
                else:
                    recs.append(
                        f"{w.identifier} accuracy is low ({w.value:.3f}). "
                        "Consider instrument-specific feature engineering."
                    )

        # Timeframe recommendations
        if worst_timeframes:
            w = worst_timeframes[0]
            if w.value < self._acc_thresh:
                if w.identifier in ("1M", "5M"):
                    recs.append(
                        f"Short timeframe {w.identifier} accuracy is poor ({w.value:.3f}). "
                        "Microstructure noise may be overwhelming the model. "
                        "Consider adding bid-ask spread and order flow features, "
                        "or reducing prediction frequency on this timeframe."
                    )
                elif w.identifier in ("1D", "1W"):
                    recs.append(
                        f"Long timeframe {w.identifier} accuracy is poor ({w.value:.3f}). "
                        "Consider adding macro indicators and earnings calendar features."
                    )
                else:
                    recs.append(
                        f"Timeframe {w.identifier} accuracy is below threshold ({w.value:.3f}). "
                        "Review feature construction for this timeframe."
                    )

        # Regime recommendations
        if worst_regimes:
            w = worst_regimes[0]
            if w.value < self._acc_thresh:
                regime_recs = {
                    "ranging": "Ranging markets are harder to predict. Consider adding "
                    "volatility-normalised features and a dedicated ranging-regime classifier.",
                    "high_volatility": "High volatility accuracy is poor. Consider reducing "
                    "confidence thresholds and using wider stop-losses in this regime.",
                    "news_period": "News-period performance is poor. Add a news-aware feature "
                    "or disable trading during high-impact news.",
                    "breakout": "Breakout detection accuracy is low. Consider adding "
                    "volume profile and order book features.",
                    "compression": "Compression regime accuracy is low. The model may benefit "
                    "from compression-specific training examples.",
                }
                rec = regime_recs.get(
                    w.identifier,
                    f"Regime '{w.identifier}' accuracy ({w.value:.3f}) is below threshold. "
                    "Consider regime-specific feature engineering.",
                )
                recs.append(rec)

        # Head-specific recommendations
        if worst_heads:
            w = worst_heads[0]
            if w.value < self._acc_thresh:
                recs.append(
                    f"Prediction head '{w.identifier}' accuracy is low ({w.value:.3f}). "
                    "Review head-specific training data quality and loss weighting."
                )

        # Confidence calibration
        if worst_confidence:
            recs.append(
                f"{len(worst_confidence)} confidence ranges show poor calibration. "
                "Apply temperature scaling or Platt calibration before deployment."
            )

        # Catch-all
        if not recs:
            recs.append(
                "No specific failure patterns identified. "
                "Model performance is broadly consistent across dimensions."
            )

        return recs
