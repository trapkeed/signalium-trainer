"""Phase S4 — hindsight-optimal stop-loss labels.

Explicitly NOT ATR-based (unlike `trainer.execution.stop_loss_engine`, which
computes a live ATR-multiple recommendation for runtime use). This module
looks *backward* at a fully-realized `TradeOutcomePath` and asks: what is the
tightest stop-loss distance that would not have been triggered before the
position reached its best achievable exit? That distance, not a volatility
multiple, is the label a model should be trained to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

from trainer.training_signals.outcome_path import TradeOutcomePath


@dataclass
class HindsightStopLossLabel:
    instrument: str
    direction: int
    entry_price: float
    optimal_stop_price: float
    optimal_stop_distance: float
    survives_to_index: int
    realized_return_pct: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 5),
            "optimal_stop_price": round(self.optimal_stop_price, 5),
            "optimal_stop_distance": round(self.optimal_stop_distance, 5),
            "survives_to_index": self.survives_to_index,
            "realized_return_pct": round(self.realized_return_pct, 5),
            "rationale": self.rationale,
        }


class HindsightStopLossLabeler:
    """Derives the hindsight-optimal stop-loss distance for one realized trade.

    Algorithm: find the bar at which the trade reached its maximum favorable
    excursion (MFE) — the best point at which the position could have been
    exited profitably. The optimal stop is the smallest distance that would
    have survived (i.e. not been breached) up to and including that bar. If
    the path never moved favorably, the "best achievable exit" degenerates
    to the final bar, and the label is simply the realized max adverse
    excursion — i.e. "this is exactly how much room the trade needed, and no
    more, to reach the best outcome it in fact reached."
    """

    def label(self, path: TradeOutcomePath) -> HindsightStopLossLabel:
        if not path.future_bars:
            return HindsightStopLossLabel(
                instrument=path.instrument,
                direction=path.direction,
                entry_price=path.entry_price,
                optimal_stop_price=path.entry_price,
                optimal_stop_distance=0.0,
                survives_to_index=-1,
                realized_return_pct=0.0,
                rationale="No future bars available — no stop distance could be derived.",
            )

        mfe, mfe_idx = path.max_favorable_excursion()
        survives_to_index = mfe_idx if mfe_idx >= 0 else len(path.future_bars) - 1
        optimal_distance = path.max_adverse_excursion_through(survives_to_index)

        if path.direction >= 0:
            optimal_stop_price = path.entry_price - optimal_distance
        else:
            optimal_stop_price = path.entry_price + optimal_distance

        if mfe_idx >= 0:
            rationale = (
                f"MFE of {mfe:.5f} reached at bar {mfe_idx}; the tightest stop that "
                f"would have survived to that bar is {optimal_distance:.5f} away from entry."
            )
        else:
            rationale = (
                "Trade never moved favorably; optimal stop is the realized max "
                f"adverse excursion of {optimal_distance:.5f} (the exact room the "
                "path used before its worst point)."
            )

        return HindsightStopLossLabel(
            instrument=path.instrument,
            direction=path.direction,
            entry_price=path.entry_price,
            optimal_stop_price=optimal_stop_price,
            optimal_stop_distance=optimal_distance,
            survives_to_index=survives_to_index,
            realized_return_pct=path.realized_return_pct(),
            rationale=rationale,
        )
