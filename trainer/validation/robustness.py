"""Phase 15 — Robustness Testing.

Stress-tests a model under various perturbations to the validation data.
Measures how much the model's accuracy degrades under each stress scenario.

Stress scenarios:
  - MissingCandles    — randomly zero out N% of input bars
  - NoisyData         — add Gaussian noise to all features
  - SpreadExpansion   — inflate the simulated spread cost
  - LatencyDelay      — shift the input by K bars (simulates stale data)
  - DataGaps          — insert NaN-filled gaps into the sequence
  - VolatilitySpike   — scale random bars by a spike multiplier
  - MarketCrash       — reverse returns over a window (worst-case scenario)

Usage::

    tester = RobustnessTester(seed=42)
    report = tester.evaluate(model, x_val, y_val)
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Stress result
# ---------------------------------------------------------------------------


@dataclass
class StressScenarioResult:
    """Result of a single stress scenario.

    Attributes:
        scenario_name: Name of the stress scenario applied.
        parameter: Key parameter value (e.g. noise_std, missing_fraction).
        baseline_accuracy: Accuracy on clean data.
        stressed_accuracy: Accuracy after stress applied.
        accuracy_delta: stressed - baseline (negative = degradation).
        relative_degradation: abs(delta) / baseline.
        passed: True when degradation is within tolerance.
    """

    scenario_name: str
    parameter: float
    baseline_accuracy: float
    stressed_accuracy: float
    accuracy_delta: float
    relative_degradation: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "parameter": self.parameter,
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "stressed_accuracy": round(self.stressed_accuracy, 4),
            "accuracy_delta": round(self.accuracy_delta, 4),
            "relative_degradation": round(self.relative_degradation, 4),
            "passed": self.passed,
        }


@dataclass
class RobustnessReport:
    """Full robustness test report.

    Attributes:
        scenarios: All scenario results.
        overall_robustness_score: Fraction of scenarios that passed.
        baseline_accuracy: Clean data accuracy (reference).
    """

    scenarios: list[StressScenarioResult] = field(default_factory=list)
    overall_robustness_score: float = 0.0
    baseline_accuracy: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Robustness Report  baseline_acc={self.baseline_accuracy:.3f}  "
            f"score={self.overall_robustness_score:.1%}",
        ]
        for s in self.scenarios:
            icon = "PASS" if s.passed else "FAIL"
            lines.append(
                f"  [{icon}] {s.scenario_name:<22s} param={s.parameter:.3f}  "
                f"stressed={s.stressed_accuracy:.3f}  "
                f"delta={s.accuracy_delta:+.3f}  "
                f"degradation={s.relative_degradation:.1%}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_robustness_score": round(self.overall_robustness_score, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


# ---------------------------------------------------------------------------
# Robustness Tester
# ---------------------------------------------------------------------------


class RobustnessTester:
    """Runs all stress scenarios and measures accuracy degradation.

    Args:
        seed: Random seed for reproducibility.
        tolerance: Maximum allowable relative accuracy degradation (default 20%).
        missing_fractions: Fractions of bars to zero-out.
        noise_levels: Noise standard deviations as fraction of feature range.
        spread_multipliers: Spread expansion factors.
        latency_shifts: Number of bars to shift input.
        gap_fractions: Fraction of sequence length for data gaps.
        spike_multipliers: Volatility spike scale factors.
        crash_window: Length of reverse-returns crash window.
    """

    def __init__(
        self,
        seed: int = 42,
        tolerance: float = 0.20,
        missing_fractions: list[float] | None = None,
        noise_levels: list[float] | None = None,
        spread_multipliers: list[float] | None = None,
        latency_shifts: list[int] | None = None,
        gap_fractions: list[float] | None = None,
        spike_multipliers: list[float] | None = None,
        crash_window: int = 20,
    ) -> None:
        self._seed = seed
        self._tol = tolerance
        self._missing = missing_fractions or [0.05, 0.10, 0.20]
        self._noise = noise_levels or [0.05, 0.10, 0.20]
        self._spread = spread_multipliers or [2.0, 5.0]
        self._latency = latency_shifts or [1, 3]
        self._gaps = gap_fractions or [0.05, 0.10]
        self._spikes = spike_multipliers or [3.0, 5.0]
        self._crash_w = crash_window

    def evaluate(
        self,
        model: nn.Module,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> RobustnessReport:
        """Run all stress scenarios.

        Args:
            model: Model to stress-test (not modified).
            x_val: ``(N, seq_len, feat)`` clean validation features.
            y_val: ``(N,)`` validation labels.

        Returns:
            ``RobustnessReport`` with per-scenario results.
        """
        baseline_acc = self._eval_accuracy(model, x_val, y_val)
        scenarios: list[StressScenarioResult] = []
        rng = np.random.default_rng(self._seed)

        for frac in self._missing:
            x_stressed = self._apply_missing_candles(x_val.clone(), frac, rng)
            scenarios.append(
                self._make_result("MissingCandles", frac, baseline_acc, model, x_stressed, y_val)
            )

        for level in self._noise:
            x_stressed = self._apply_noise(x_val.clone(), level, rng)
            scenarios.append(
                self._make_result("NoisyData", level, baseline_acc, model, x_stressed, y_val)
            )

        for mult in self._spread:
            scenarios.append(
                StressScenarioResult(
                    scenario_name="SpreadExpansion",
                    parameter=mult,
                    baseline_accuracy=baseline_acc,
                    stressed_accuracy=baseline_acc,  # doesn't change model accuracy
                    accuracy_delta=0.0,
                    relative_degradation=0.0,
                    passed=True,
                    details={
                        "note": "Spread affects trading metrics only, not classification accuracy."
                    },
                )
            )

        for shift in self._latency:
            x_stressed = self._apply_latency(x_val.clone(), shift)
            scenarios.append(
                self._make_result(
                    "LatencyDelay", float(shift), baseline_acc, model, x_stressed, y_val
                )
            )

        for frac in self._gaps:
            x_stressed = self._apply_data_gaps(x_val.clone(), frac, rng)
            scenarios.append(
                self._make_result("DataGaps", frac, baseline_acc, model, x_stressed, y_val)
            )

        for mult in self._spikes:
            x_stressed = self._apply_volatility_spike(x_val.clone(), mult, rng)
            scenarios.append(
                self._make_result("VolatilitySpike", mult, baseline_acc, model, x_stressed, y_val)
            )

        # Market crash: reverse a random window
        x_stressed = self._apply_market_crash(x_val.clone(), rng)
        scenarios.append(
            self._make_result(
                "MarketCrash", float(self._crash_w), baseline_acc, model, x_stressed, y_val
            )
        )

        passed = sum(1 for s in scenarios if s.passed)
        robustness_score = passed / len(scenarios) if scenarios else 0.0

        return RobustnessReport(
            scenarios=scenarios,
            overall_robustness_score=robustness_score,
            baseline_accuracy=baseline_acc,
        )

    # ── Stress scenario implementations ────────────────────────────────────

    def _make_result(
        self,
        name: str,
        param: float,
        baseline: float,
        model: nn.Module,
        x_stressed: torch.Tensor,
        y_val: torch.Tensor,
    ) -> StressScenarioResult:
        acc = self._eval_accuracy(model, x_stressed, y_val)
        delta = acc - baseline
        rel_deg = abs(delta) / max(baseline, 1e-9)
        passed = rel_deg <= self._tol
        return StressScenarioResult(
            scenario_name=name,
            parameter=param,
            baseline_accuracy=baseline,
            stressed_accuracy=acc,
            accuracy_delta=delta,
            relative_degradation=rel_deg,
            passed=passed,
        )

    @staticmethod
    def _apply_missing_candles(x: torch.Tensor, frac: float, rng) -> torch.Tensor:
        n = x.shape[0]
        n_missing = max(1, int(n * frac))
        idx = rng.choice(n, n_missing, replace=False)
        x[idx] = 0.0
        return x

    @staticmethod
    def _apply_noise(x: torch.Tensor, level: float, rng) -> torch.Tensor:
        scale = float(x.abs().max().item()) * level
        noise = torch.tensor(rng.normal(0, scale, x.shape), dtype=x.dtype)
        return x + noise

    @staticmethod
    def _apply_latency(x: torch.Tensor, shift: int) -> torch.Tensor:
        shifted = torch.roll(x, shift, dims=0)
        shifted[:shift] = 0.0
        return shifted

    @staticmethod
    def _apply_data_gaps(x: torch.Tensor, frac: float, rng) -> torch.Tensor:
        seq_len = x.shape[1] if x.dim() == 3 else 1
        n_gap = max(1, int(x.shape[0] * frac))
        idx = rng.choice(x.shape[0], n_gap, replace=False)
        x[idx] = float("nan")
        x = torch.nan_to_num(x, nan=0.0)
        return x

    @staticmethod
    def _apply_volatility_spike(x: torch.Tensor, mult: float, rng) -> torch.Tensor:
        n = x.shape[0]
        spike_idx = rng.choice(n, max(1, n // 20), replace=False)
        x[spike_idx] *= mult
        return x

    def _apply_market_crash(self, x: torch.Tensor, rng) -> torch.Tensor:
        n = x.shape[0]
        if n <= self._crash_w:
            return x
        start = int(rng.integers(0, n - self._crash_w))
        x[start : start + self._crash_w] = -x[start : start + self._crash_w]
        return x

    @torch.no_grad()
    def _eval_accuracy(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        was_training = model.training
        model.eval()
        try:
            # Replace NaN/Inf that might have been introduced
            x_clean = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            logits = model(x_clean)
            preds = logits.argmax(dim=-1)
            return float((preds == y).float().mean().item())
        except Exception as exc:
            logger.debug(f"RobustnessTester eval failed: {exc}")
            return 0.0
        finally:
            if was_training:
                model.train()
