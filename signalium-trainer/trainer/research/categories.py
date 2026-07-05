"""Phase 19.2 — IRSE Category Scorers.

Eight scoring categories, each producing a sub-score in [0, 100].

Categories:
  ValidationQualityScorer     — walk-forward, regime/instrument/timeframe stability
  TradingPerformanceScorer    — profit factor, Sharpe, Sortino, Calmar, drawdown
  RobustnessScorer            — Monte Carlo, stress tests, noise/gap/spread tolerance
  CalibrationScorer           — ECE, Brier score, confidence reliability
  GeneralizationScorer        — unseen instruments/timeframes/regimes, OOS performance
  EfficiencyScorer            — latency, training time, GPU/RAM, model size
  ExplainabilityScorer        — feature attribution, attention quality, documentation
  OperationalReadinessScorer  — deployment readiness, monitoring, drift, rollback

All inputs are dataclasses with sensible defaults so partial data never raises.
All scorers are pure functions — no side effects, fully deterministic.

Usage::

    from trainer.research.categories import (
        ValidationQualityInputs, ValidationQualityScorer,
        TradingPerformanceInputs, TradingPerformanceScorer,
    )
    inputs = ValidationQualityInputs(walk_forward_accuracy=0.69, n_folds=12)
    score, sub = ValidationQualityScorer().score(inputs)
    print(f"Validation quality: {score:.1f}/100")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _linear(value: float, lo: float, hi: float) -> float:
    """Map value from [lo, hi] → [0, 100]."""
    if hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo) * 100.0)


def _inverse_linear(value: float, lo: float, hi: float) -> float:
    """Inverse: lower value → higher score (e.g. drawdown, ECE)."""
    if hi <= lo:
        return 100.0
    return _clamp((1.0 - (value - lo) / (hi - lo)) * 100.0)


def _threshold(value: float, threshold: float, above: bool = True) -> float:
    """Binary threshold: full score if above (or below) threshold."""
    if above:
        return 100.0 if value >= threshold else (value / threshold) * 100.0
    return 100.0 if value <= threshold else (threshold / value) * 100.0 if value > 0 else 0.0


def _weighted_avg(pairs: List[Tuple[float, float]]) -> float:
    """Weighted average of (score, weight) pairs."""
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return sum(s * w for s, w in pairs) / total_w


# ===========================================================================
# 1. Validation Quality
# ===========================================================================

@dataclass
class ValidationQualityInputs:
    walk_forward_accuracy:       float = 0.0   # 0–1
    walk_forward_f1_macro:       float = 0.0   # 0–1
    n_folds:                     int   = 0
    cross_regime_stability:      float = 0.0   # std dev of fold scores (lower = better)
    cross_instrument_stability:  float = 0.0   # std dev across instruments (lower = better)
    cross_timeframe_stability:   float = 0.0   # std dev across timeframes (lower = better)
    consistency_score:           float = 0.0   # fraction of folds above threshold (0–1)
    prediction_reliability:      float = 0.0   # fraction of high-confidence predictions correct (0–1)


class ValidationQualityScorer:
    """Score validation quality from walk-forward and stability metrics."""

    # Targets: accuracy ≥ 0.65 is 80 pts, ≥ 0.75 is 100 pts
    _ACC_LO, _ACC_HI = 0.50, 0.75

    def score(
        self, inputs: ValidationQualityInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {}

        sub["walk_forward_accuracy"] = _linear(
            inputs.walk_forward_accuracy, self._ACC_LO, self._ACC_HI
        )
        sub["walk_forward_f1_macro"] = _linear(
            inputs.walk_forward_f1_macro, self._ACC_LO, self._ACC_HI
        )
        # Fold count: 12+ folds → 100, fewer scales linearly
        sub["fold_coverage"] = _clamp(inputs.n_folds / 12.0 * 100.0)

        # Stability: lower std is better (0.0 → 100, 0.10+ → 0)
        sub["cross_regime_stability"] = _inverse_linear(
            inputs.cross_regime_stability, 0.0, 0.10
        )
        sub["cross_instrument_stability"] = _inverse_linear(
            inputs.cross_instrument_stability, 0.0, 0.10
        )
        sub["cross_timeframe_stability"] = _inverse_linear(
            inputs.cross_timeframe_stability, 0.0, 0.10
        )

        sub["consistency"] = _clamp(inputs.consistency_score * 100.0)
        sub["prediction_reliability"] = _clamp(inputs.prediction_reliability * 100.0)

        composite = _weighted_avg([
            (sub["walk_forward_accuracy"],       0.25),
            (sub["walk_forward_f1_macro"],        0.15),
            (sub["fold_coverage"],                0.10),
            (sub["cross_regime_stability"],       0.15),
            (sub["cross_instrument_stability"],   0.10),
            (sub["cross_timeframe_stability"],    0.10),
            (sub["consistency"],                  0.10),
            (sub["prediction_reliability"],       0.05),
        ])
        return _clamp(composite), sub


# ===========================================================================
# 2. Trading Performance
# ===========================================================================

@dataclass
class TradingPerformanceInputs:
    profit_factor:           float = 0.0   # ≥ 1 is good; 1.5+ is strong
    expectancy:              float = 0.0   # average R per trade
    sharpe_ratio:            float = 0.0   # 1.5+ is strong
    sortino_ratio:           float = 0.0   # 2.0+ is strong
    calmar_ratio:            float = 0.0   # 1.5+ is strong
    max_drawdown:            float = 0.0   # fraction (lower is better; 0.10 = -10%)
    win_rate:                float = 0.0   # fraction (0–1)
    average_r_multiple:      float = 0.0   # average reward/risk
    tp_before_sl_accuracy:   float = 0.0   # fraction of trades TP before SL
    capital_efficiency:      float = 0.0   # 0–1


class TradingPerformanceScorer:
    def score(
        self, inputs: TradingPerformanceInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {}

        # Profit factor: 1.0 → 0, 1.3 → 60, 2.0+ → 100
        sub["profit_factor"] = _clamp(
            (inputs.profit_factor - 1.0) / 1.0 * 100.0
        ) if inputs.profit_factor > 1.0 else 0.0

        # Expectancy: 0.0 → 0, 0.5+ → 100
        sub["expectancy"] = _linear(inputs.expectancy, 0.0, 0.5)

        # Sharpe: 0.5 → 0, 2.0+ → 100
        sub["sharpe_ratio"] = _linear(inputs.sharpe_ratio, 0.5, 2.5)

        # Sortino: 0.5 → 0, 3.0+ → 100
        sub["sortino_ratio"] = _linear(inputs.sortino_ratio, 0.5, 3.0)

        # Calmar: 0.5 → 0, 3.0+ → 100
        sub["calmar_ratio"] = _linear(inputs.calmar_ratio, 0.5, 3.0)

        # Max drawdown: 0.20 → 0, 0.05 → 100 (lower is better)
        sub["max_drawdown"] = _inverse_linear(inputs.max_drawdown, 0.03, 0.25)

        # Win rate: 40% → 0, 65%+ → 100
        sub["win_rate"] = _linear(inputs.win_rate, 0.40, 0.65)

        # Average R multiple: 0.5 → 0, 2.0+ → 100
        sub["average_r_multiple"] = _linear(inputs.average_r_multiple, 0.3, 2.0)

        # TP before SL: 0.5 → 0, 0.75+ → 100
        sub["tp_before_sl_accuracy"] = _linear(
            inputs.tp_before_sl_accuracy, 0.45, 0.75
        )

        # Capital efficiency: 0 → 0, 1 → 100
        sub["capital_efficiency"] = _clamp(inputs.capital_efficiency * 100.0)

        composite = _weighted_avg([
            (sub["profit_factor"],        0.15),
            (sub["expectancy"],           0.10),
            (sub["sharpe_ratio"],         0.20),
            (sub["sortino_ratio"],        0.10),
            (sub["calmar_ratio"],         0.10),
            (sub["max_drawdown"],         0.15),
            (sub["win_rate"],             0.08),
            (sub["average_r_multiple"],   0.05),
            (sub["tp_before_sl_accuracy"],0.04),
            (sub["capital_efficiency"],   0.03),
        ])
        return _clamp(composite), sub


# ===========================================================================
# 3. Robustness
# ===========================================================================

@dataclass
class RobustnessInputs:
    monte_carlo_stability:    float = 0.0   # 0–1 (fraction of MC runs profitable)
    stress_test_pass_rate:    float = 0.0   # 0–1 (fraction of stress scenarios passed)
    noise_resistance:         float = 0.0   # 0–1 (performance retention under noise)
    data_gap_tolerance:       float = 0.0   # 0–1 (performance retention with gaps)
    spread_expansion_tolerance: float = 0.0 # 0–1 (performance retention with 2x spread)
    execution_delay_tolerance: float = 0.0  # 0–1 (performance retention with delay)
    volatility_shift_tolerance: float = 0.0 # 0–1 (performance retention in vol shift)


class RobustnessScorer:
    def score(
        self, inputs: RobustnessInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {
            "monte_carlo_stability":     _clamp(inputs.monte_carlo_stability * 100.0),
            "stress_test_pass_rate":     _clamp(inputs.stress_test_pass_rate * 100.0),
            "noise_resistance":          _clamp(inputs.noise_resistance * 100.0),
            "data_gap_tolerance":        _clamp(inputs.data_gap_tolerance * 100.0),
            "spread_expansion_tolerance":_clamp(inputs.spread_expansion_tolerance * 100.0),
            "execution_delay_tolerance": _clamp(inputs.execution_delay_tolerance * 100.0),
            "volatility_shift_tolerance":_clamp(inputs.volatility_shift_tolerance * 100.0),
        }
        composite = _weighted_avg([
            (sub["monte_carlo_stability"],      0.20),
            (sub["stress_test_pass_rate"],      0.25),
            (sub["noise_resistance"],           0.15),
            (sub["data_gap_tolerance"],         0.10),
            (sub["spread_expansion_tolerance"], 0.15),
            (sub["execution_delay_tolerance"],  0.07),
            (sub["volatility_shift_tolerance"], 0.08),
        ])
        return _clamp(composite), sub


# ===========================================================================
# 4. Calibration
# ===========================================================================

@dataclass
class CalibrationInputs:
    expected_calibration_error: float = 0.0   # ECE (lower is better; 0.05 → 100)
    brier_score:                float = 0.0   # (lower is better; 0.20 → 100)
    confidence_reliability:     float = 0.0   # 0–1 (fraction of confident preds correct)
    prediction_entropy:         float = 0.0   # bits (lower = more decisive; 0 → 100)
    probability_calibration:    float = 0.0   # 0–1 (from calibration curve R²)


class CalibrationScorer:
    def score(
        self, inputs: CalibrationInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {
            # ECE: 0.0 → 100, 0.10+ → 0
            "expected_calibration_error": _inverse_linear(
                inputs.expected_calibration_error, 0.0, 0.10
            ),
            # Brier: 0.10 → 100, 0.30+ → 0 (for 3-class: random ≈ 0.67)
            "brier_score": _inverse_linear(inputs.brier_score, 0.10, 0.30),
            "confidence_reliability": _clamp(inputs.confidence_reliability * 100.0),
            # Entropy: 0.0 → 100, 1.5+ → 0 (for 3-class max ≈ 1.585)
            "prediction_entropy": _inverse_linear(inputs.prediction_entropy, 0.0, 1.5),
            "probability_calibration": _clamp(inputs.probability_calibration * 100.0),
        }
        composite = _weighted_avg([
            (sub["expected_calibration_error"], 0.35),
            (sub["brier_score"],                0.25),
            (sub["confidence_reliability"],     0.20),
            (sub["prediction_entropy"],         0.10),
            (sub["probability_calibration"],    0.10),
        ])
        return _clamp(composite), sub


# ===========================================================================
# 5. Generalization
# ===========================================================================

@dataclass
class GeneralizationInputs:
    unseen_instrument_performance: float = 0.0  # 0–1 accuracy on new instruments
    unseen_timeframe_performance:  float = 0.0  # 0–1 accuracy on new timeframes
    unseen_regime_performance:     float = 0.0  # 0–1 accuracy in new regimes
    out_of_sample_performance:     float = 0.0  # 0–1 accuracy on true OOS
    recent_data_performance:       float = 0.0  # 0–1 accuracy on most recent 6mo
    historical_stability:          float = 0.0  # 0–1 (low variance across time periods)


class GeneralizationScorer:
    _LO, _HI = 0.50, 0.75

    def score(
        self, inputs: GeneralizationInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {
            "unseen_instrument_performance": _linear(
                inputs.unseen_instrument_performance, self._LO, self._HI
            ),
            "unseen_timeframe_performance": _linear(
                inputs.unseen_timeframe_performance, self._LO, self._HI
            ),
            "unseen_regime_performance": _linear(
                inputs.unseen_regime_performance, self._LO, self._HI
            ),
            "out_of_sample_performance": _linear(
                inputs.out_of_sample_performance, self._LO, self._HI
            ),
            "recent_data_performance": _linear(
                inputs.recent_data_performance, self._LO, self._HI
            ),
            "historical_stability": _clamp(inputs.historical_stability * 100.0),
        }
        composite = _weighted_avg([
            (sub["unseen_instrument_performance"], 0.15),
            (sub["unseen_timeframe_performance"],  0.15),
            (sub["unseen_regime_performance"],     0.20),
            (sub["out_of_sample_performance"],     0.25),
            (sub["recent_data_performance"],       0.15),
            (sub["historical_stability"],          0.10),
        ])
        return _clamp(composite), sub


# ===========================================================================
# 6. Efficiency
# ===========================================================================

@dataclass
class EfficiencyInputs:
    inference_latency_ms:   float = 0.0  # ms per sample (lower is better; 20ms → 100)
    training_time_hours:    float = 0.0  # hours (lower is better; 24h → 0)
    gpu_utilization:        float = 0.0  # 0–1 (higher is better)
    cpu_utilization:        float = 0.0  # 0–1 (higher = better used resources)
    ram_usage_gb:           float = 0.0  # GB (lower is better; 32GB+ → 0)
    model_size_mb:          float = 0.0  # MB (lower is better; 1000MB+ → 0)
    storage_footprint_gb:   float = 0.0  # GB (lower is better; 50GB+ → 0)


class EfficiencyScorer:
    def score(
        self, inputs: EfficiencyInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {
            # Latency: 5ms → 100, 100ms → 0
            "inference_latency": _inverse_linear(
                inputs.inference_latency_ms, 5.0, 100.0
            ),
            # Training time: 1h → 100, 48h → 0
            "training_time": _inverse_linear(
                inputs.training_time_hours, 1.0, 48.0
            ),
            "gpu_utilization":  _clamp(inputs.gpu_utilization * 100.0),
            "cpu_utilization":  _clamp(inputs.cpu_utilization * 100.0),
            # RAM: 2GB → 100, 64GB → 0
            "ram_usage": _inverse_linear(inputs.ram_usage_gb, 2.0, 64.0),
            # Model size: 50MB → 100, 2000MB → 0
            "model_size": _inverse_linear(inputs.model_size_mb, 50.0, 2000.0),
            # Storage: 1GB → 100, 100GB → 0
            "storage_footprint": _inverse_linear(
                inputs.storage_footprint_gb, 1.0, 100.0
            ),
        }
        composite = _weighted_avg([
            (sub["inference_latency"],  0.30),
            (sub["training_time"],      0.15),
            (sub["gpu_utilization"],    0.15),
            (sub["cpu_utilization"],    0.05),
            (sub["ram_usage"],          0.15),
            (sub["model_size"],         0.10),
            (sub["storage_footprint"],  0.10),
        ])
        return _clamp(composite), sub


# ===========================================================================
# 7. Explainability
# ===========================================================================

@dataclass
class ExplainabilityInputs:
    feature_attribution_coverage: float = 0.0  # 0–1 (fraction of features attributed)
    attention_quality:            float = 0.0  # 0–1 (quality score from attention maps)
    prediction_traceability:      float = 0.0  # 0–1 (fraction of predictions traceable)
    decision_transparency:        float = 0.0  # 0–1 (human-assigned score)
    documentation_completeness:   float = 0.0  # 0–1 (fraction of required docs present)


class ExplainabilityScorer:
    def score(
        self, inputs: ExplainabilityInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {
            "feature_attribution_coverage": _clamp(inputs.feature_attribution_coverage * 100.0),
            "attention_quality":            _clamp(inputs.attention_quality * 100.0),
            "prediction_traceability":      _clamp(inputs.prediction_traceability * 100.0),
            "decision_transparency":        _clamp(inputs.decision_transparency * 100.0),
            "documentation_completeness":   _clamp(inputs.documentation_completeness * 100.0),
        }
        composite = _weighted_avg([
            (sub["feature_attribution_coverage"], 0.30),
            (sub["attention_quality"],            0.20),
            (sub["prediction_traceability"],      0.20),
            (sub["decision_transparency"],        0.15),
            (sub["documentation_completeness"],   0.15),
        ])
        return _clamp(composite), sub


# ===========================================================================
# 8. Operational Readiness
# ===========================================================================

@dataclass
class OperationalReadinessInputs:
    deployment_readiness:    float = 0.0  # 0–1 (checklist completion)
    monitoring_coverage:     float = 0.0  # 0–1 (fraction of required monitors active)
    health_checks:           float = 0.0  # 0–1 (fraction of health checks passing)
    drift_monitoring:        float = 0.0  # 0–1 (drift monitors configured and active)
    rollback_readiness:      float = 0.0  # 0–1 (rollback plan tested)
    audit_completeness:      float = 0.0  # 0–1 (fraction of audit requirements met)


class OperationalReadinessScorer:
    def score(
        self, inputs: OperationalReadinessInputs
    ) -> Tuple[float, Dict[str, float]]:
        sub: Dict[str, float] = {
            "deployment_readiness":  _clamp(inputs.deployment_readiness * 100.0),
            "monitoring_coverage":   _clamp(inputs.monitoring_coverage * 100.0),
            "health_checks":         _clamp(inputs.health_checks * 100.0),
            "drift_monitoring":      _clamp(inputs.drift_monitoring * 100.0),
            "rollback_readiness":    _clamp(inputs.rollback_readiness * 100.0),
            "audit_completeness":    _clamp(inputs.audit_completeness * 100.0),
        }
        composite = _weighted_avg([
            (sub["deployment_readiness"],  0.25),
            (sub["monitoring_coverage"],   0.20),
            (sub["health_checks"],         0.20),
            (sub["drift_monitoring"],      0.15),
            (sub["rollback_readiness"],    0.10),
            (sub["audit_completeness"],    0.10),
        ])
        return _clamp(composite), sub


# ===========================================================================
# Convenience: all scorer instances
# ===========================================================================

_SCORERS = {
    "validation_quality":    ValidationQualityScorer(),
    "trading_performance":   TradingPerformanceScorer(),
    "robustness":            RobustnessScorer(),
    "calibration":           CalibrationScorer(),
    "generalization":        GeneralizationScorer(),
    "efficiency":            EfficiencyScorer(),
    "explainability":        ExplainabilityScorer(),
    "operational_readiness": OperationalReadinessScorer(),
}
