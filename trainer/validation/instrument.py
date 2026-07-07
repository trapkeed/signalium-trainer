"""Phase 15 — Multi-asset and multi-timeframe validation infrastructure.

Defines the canonical instrument and timeframe registries and the
``MultiAssetValidator`` that runs a validation strategy across every
instrument × timeframe combination.

Instruments: 14 canonical + unlimited future registrations.
Timeframes: 8 canonical + unlimited future registrations.

Usage::

    validator = MultiAssetValidator(
        instruments=["EURUSD", "GBPUSD", "XAUUSD"],
        timeframes=["1H", "4H"],
        strategy=ExpandingWindowValidator(min_train_size=500, val_size=100),
    )
    results = validator.validate(model, dataset_map)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

# ---------------------------------------------------------------------------
# Canonical instrument registry
# ---------------------------------------------------------------------------


INSTRUMENTS: dict[str, dict[str, Any]] = {
    "EURUSD": {"type": "forex", "pip_size": 0.0001, "description": "Euro / US Dollar"},
    "GBPUSD": {"type": "forex", "pip_size": 0.0001, "description": "British Pound / US Dollar"},
    "USDJPY": {"type": "forex", "pip_size": 0.01, "description": "US Dollar / Japanese Yen"},
    "AUDUSD": {"type": "forex", "pip_size": 0.0001, "description": "Australian Dollar / US Dollar"},
    "USDCAD": {"type": "forex", "pip_size": 0.0001, "description": "US Dollar / Canadian Dollar"},
    "USDCHF": {"type": "forex", "pip_size": 0.0001, "description": "US Dollar / Swiss Franc"},
    "NZDUSD": {
        "type": "forex",
        "pip_size": 0.0001,
        "description": "New Zealand Dollar / US Dollar",
    },
    "XAUUSD": {"type": "commodity", "pip_size": 0.01, "description": "Gold / US Dollar"},
    "XAGUSD": {"type": "commodity", "pip_size": 0.001, "description": "Silver / US Dollar"},
    "NAS100": {"type": "index", "pip_size": 0.01, "description": "NASDAQ 100 Index"},
    "US30": {"type": "index", "pip_size": 0.01, "description": "Dow Jones Industrial Average"},
    "SPX500": {"type": "index", "pip_size": 0.01, "description": "S&P 500 Index"},
    "BTCUSD": {"type": "crypto", "pip_size": 1.0, "description": "Bitcoin / US Dollar"},
    "ETHUSD": {"type": "crypto", "pip_size": 0.01, "description": "Ethereum / US Dollar"},
}

# User-extensible registry (not just the 14 canonical above)
_CUSTOM_INSTRUMENTS: dict[str, dict[str, Any]] = {}


def register_instrument(
    symbol: str,
    instrument_type: str = "custom",
    pip_size: float = 0.0001,
    description: str = "",
) -> None:
    """Register a custom instrument for validation.

    Args:
        symbol: Instrument symbol (e.g. "USDZAR").
        instrument_type: One of "forex", "crypto", "commodity", "index", "custom".
        pip_size: Minimum price increment.
        description: Human-readable description.
    """
    _CUSTOM_INSTRUMENTS[symbol.upper()] = {
        "type": instrument_type,
        "pip_size": pip_size,
        "description": description,
    }
    logger.debug(f"Registered custom instrument: {symbol.upper()}")


def get_all_instruments() -> dict[str, dict[str, Any]]:
    """Return all registered instruments (canonical + custom)."""
    return {**INSTRUMENTS, **_CUSTOM_INSTRUMENTS}


# ---------------------------------------------------------------------------
# Canonical timeframe registry
# ---------------------------------------------------------------------------


TIMEFRAMES: dict[str, int] = {
    "1M": 1,  # 1 minute in minutes
    "5M": 5,
    "15M": 15,
    "30M": 30,
    "1H": 60,
    "4H": 240,
    "1D": 1440,  # Daily
    "1W": 10080,  # Weekly
}

_CUSTOM_TIMEFRAMES: dict[str, int] = {}


def register_timeframe(label: str, minutes: int) -> None:
    """Register a custom timeframe.

    Args:
        label: Human-readable label (e.g. "2H").
        minutes: Duration in minutes.
    """
    _CUSTOM_TIMEFRAMES[label] = minutes
    logger.debug(f"Registered custom timeframe: {label} ({minutes}m)")


def get_all_timeframes() -> dict[str, int]:
    """Return all registered timeframes."""
    return {**TIMEFRAMES, **_CUSTOM_TIMEFRAMES}


# ---------------------------------------------------------------------------
# Per-instrument result
# ---------------------------------------------------------------------------


@dataclass
class InstrumentResult:
    """Validation results for a single instrument × timeframe combination.

    Attributes:
        instrument: Instrument symbol (e.g. "EURUSD").
        timeframe: Timeframe label (e.g. "1H").
        n_folds: Number of walk-forward folds evaluated.
        mean_accuracy: Mean accuracy across folds.
        mean_f1: Mean F1 across folds.
        mean_profit_factor: Mean profit factor across folds.
        mean_sharpe: Mean Sharpe ratio across folds.
        max_drawdown: Maximum drawdown observed across folds.
        fold_results: List of per-fold metric dicts.
        status: "ok" | "insufficient_data" | "error".
        error_message: Set when status == "error".
    """

    instrument: str
    timeframe: str
    n_folds: int = 0
    mean_accuracy: float = 0.0
    mean_f1: float = 0.0
    mean_profit_factor: float = 0.0
    mean_sharpe: float = 0.0
    max_drawdown: float = 0.0
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "n_folds": self.n_folds,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_profit_factor": round(self.mean_profit_factor, 4),
            "mean_sharpe": round(self.mean_sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class MultiAssetValidationResult:
    """Aggregated results across all instrument × timeframe combinations.

    Attributes:
        results: Nested dict: instrument → timeframe → InstrumentResult.
        n_instruments: Number of instruments validated.
        n_timeframes: Number of timeframes validated.
        global_mean_accuracy: Mean accuracy across all combinations.
        global_mean_sharpe: Mean Sharpe across all combinations.
        worst_instrument: Instrument with lowest mean accuracy.
        worst_timeframe: Timeframe with lowest mean accuracy.
    """

    results: dict[str, dict[str, InstrumentResult]] = field(default_factory=dict)
    n_instruments: int = 0
    n_timeframes: int = 0
    global_mean_accuracy: float = 0.0
    global_mean_sharpe: float = 0.0
    worst_instrument: str = ""
    worst_timeframe: str = ""

    def get(self, instrument: str, timeframe: str) -> InstrumentResult | None:
        return self.results.get(instrument, {}).get(timeframe)

    def all_results(self) -> list[InstrumentResult]:
        return [r for tf_map in self.results.values() for r in tf_map.values()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_instruments": self.n_instruments,
            "n_timeframes": self.n_timeframes,
            "global_mean_accuracy": round(self.global_mean_accuracy, 4),
            "global_mean_sharpe": round(self.global_mean_sharpe, 4),
            "worst_instrument": self.worst_instrument,
            "worst_timeframe": self.worst_timeframe,
            "results": {
                instr: {tf: r.to_dict() for tf, r in tf_map.items()}
                for instr, tf_map in self.results.items()
            },
        }


# ---------------------------------------------------------------------------
# MultiAssetValidator
# ---------------------------------------------------------------------------


class MultiAssetValidator:
    """Runs validation across multiple instruments and timeframes.

    Dataset lookup:
        ``dataset_map`` must be a dict of ``(instrument, timeframe) → (x, y)``
        where ``x: (N, seq_len, n_feat)`` and ``y: (N,)``.

    Args:
        instruments: Instrument symbols to validate (must be in dataset_map).
        timeframes: Timeframe labels to validate (must be in dataset_map).
        strategy: Walk-forward strategy (any object with ``.split(x, y)``).
        eval_fn: Callable ``(model, x_val, y_val) → dict[str, float]``.
            Defaults to accuracy-only evaluator.
        min_folds_required: Minimum folds before an instrument-TF combo is
            flagged as ``insufficient_data``.
    """

    def __init__(
        self,
        instruments: list[str],
        timeframes: list[str],
        strategy: Any,
        eval_fn: Callable | None = None,
        min_folds_required: int = 3,
    ) -> None:
        self._instruments = [i.upper() for i in instruments]
        self._timeframes = timeframes
        self._strategy = strategy
        self._eval_fn = eval_fn or _default_eval_fn
        self._min_folds = min_folds_required

    def validate(
        self,
        model: nn.Module,
        dataset_map: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]],
    ) -> MultiAssetValidationResult:
        """Run validation across all instrument × timeframe combinations.

        Args:
            model: Trained model (eval mode, not modified).
            dataset_map: ``(instrument, timeframe) → (x, y)`` dict.

        Returns:
            ``MultiAssetValidationResult`` with nested per-combination results.
        """
        out: dict[str, dict[str, InstrumentResult]] = {}
        all_accs: list[float] = []
        all_sharpes: list[float] = []

        for instrument in self._instruments:
            out[instrument] = {}
            for timeframe in self._timeframes:
                key = (instrument, timeframe)
                if key not in dataset_map:
                    logger.debug(
                        f"MultiAssetValidator: no data for {instrument}/{timeframe} — skipping."
                    )
                    out[instrument][timeframe] = InstrumentResult(
                        instrument=instrument,
                        timeframe=timeframe,
                        status="insufficient_data",
                        error_message="Not present in dataset_map.",
                    )
                    continue

                x, y = dataset_map[key]
                result = self._validate_one(model, instrument, timeframe, x, y)
                out[instrument][timeframe] = result
                if result.status == "ok":
                    all_accs.append(result.mean_accuracy)
                    all_sharpes.append(result.mean_sharpe)

        global_acc = float(np.mean(all_accs)) if all_accs else 0.0
        global_sharpe = float(np.mean(all_sharpes)) if all_sharpes else 0.0

        # Worst instrument
        instr_accs = {
            instr: float(
                np.mean([r.mean_accuracy for r in out[instr].values() if r.status == "ok"] or [0.0])
            )
            for instr in self._instruments
        }
        worst_instr = min(instr_accs, key=lambda k: instr_accs[k]) if instr_accs else ""

        # Worst timeframe
        tf_accs = {
            tf: float(
                np.mean(
                    [
                        out[instr].get(tf, InstrumentResult("", "")).mean_accuracy
                        for instr in self._instruments
                        if out.get(instr, {}).get(tf, InstrumentResult("", "")).status == "ok"
                    ]
                    or [0.0]
                )
            )
            for tf in self._timeframes
        }
        worst_tf = min(tf_accs, key=lambda k: tf_accs[k]) if tf_accs else ""

        return MultiAssetValidationResult(
            results=out,
            n_instruments=len(self._instruments),
            n_timeframes=len(self._timeframes),
            global_mean_accuracy=global_acc,
            global_mean_sharpe=global_sharpe,
            worst_instrument=worst_instr,
            worst_timeframe=worst_tf,
        )

    def _validate_one(
        self,
        model: nn.Module,
        instrument: str,
        timeframe: str,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> InstrumentResult:
        try:
            fold_results: list[dict[str, float]] = []
            for fold in self._strategy.split(x, y):
                metrics = self._eval_fn(model, fold.x_val, fold.y_val)
                metrics["fold"] = float(fold.fold_index)
                fold_results.append(metrics)

            if len(fold_results) < self._min_folds:
                return InstrumentResult(
                    instrument=instrument,
                    timeframe=timeframe,
                    n_folds=len(fold_results),
                    status="insufficient_data",
                    error_message=f"Only {len(fold_results)} folds < minimum {self._min_folds}.",
                )

            accs = [r.get("accuracy", 0.0) for r in fold_results]
            f1s = [r.get("f1", 0.0) for r in fold_results]
            pfs = [r.get("profit_factor", 1.0) for r in fold_results]
            sharpes = [r.get("sharpe", 0.0) for r in fold_results]
            drawdowns = [r.get("max_drawdown", 0.0) for r in fold_results]

            return InstrumentResult(
                instrument=instrument,
                timeframe=timeframe,
                n_folds=len(fold_results),
                mean_accuracy=float(np.mean(accs)),
                mean_f1=float(np.mean(f1s)),
                mean_profit_factor=float(np.mean(pfs)),
                mean_sharpe=float(np.mean(sharpes)),
                max_drawdown=float(np.max(drawdowns)),
                fold_results=fold_results,
                status="ok",
            )
        except Exception as exc:
            logger.warning(f"MultiAssetValidator: {instrument}/{timeframe} failed: {exc}")
            return InstrumentResult(
                instrument=instrument,
                timeframe=timeframe,
                status="error",
                error_message=str(exc),
            )


@torch.no_grad()
def _default_eval_fn(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> dict[str, float]:
    """Minimal evaluator: accuracy only."""
    was_training = model.training
    model.eval()
    try:
        logits = model(x_val)
        preds = logits.argmax(dim=-1)
        acc = float((preds == y_val).float().mean().item())
    finally:
        if was_training:
            model.train()
    return {"accuracy": acc}
