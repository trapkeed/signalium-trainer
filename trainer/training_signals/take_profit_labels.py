"""Phase S4 — hindsight-optimal take-profit labels.

Explicitly NOT a fixed reward:risk ratio (unlike
`trainer.execution.take_profit_engine`, which computes live TP1-TP3 levels
as multiples of the stop distance). This module derives the label directly
from the realized future maximum favorable excursion (MFE), per the S4 spec:
"Train using future MFE."
"""

from __future__ import annotations

from dataclasses import dataclass

from trainer.training_signals.outcome_path import TradeOutcomePath


@dataclass
class HindsightTakeProfitLabel:
    instrument: str
    direction: int
    entry_price: float
    optimal_target_price: float
    optimal_target_distance: float
    bars_to_reach: int
    realized_return_pct: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction,
            "entry_price": round(self.entry_price, 5),
            "optimal_target_price": round(self.optimal_target_price, 5),
            "optimal_target_distance": round(self.optimal_target_distance, 5),
            "bars_to_reach": self.bars_to_reach,
            "realized_return_pct": round(self.realized_return_pct, 5),
            "rationale": self.rationale,
        }


class HindsightTakeProfitLabeler:
    """The optimal take-profit label is the price level at the trade's actual
    maximum favorable excursion — the best exit the path in fact offered.
    If the path never moved favorably, the label distance is 0 (there was no
    profit to target), which itself is informative training signal (this
    setup should have produced a near-zero/negative size or confidence
    label, not a heuristic non-zero target).
    """

    def label(self, path: TradeOutcomePath) -> HindsightTakeProfitLabel:
        if not path.future_bars:
            return HindsightTakeProfitLabel(
                instrument=path.instrument,
                direction=path.direction,
                entry_price=path.entry_price,
                optimal_target_price=path.entry_price,
                optimal_target_distance=0.0,
                bars_to_reach=0,
                realized_return_pct=0.0,
                rationale="No future bars available — no MFE could be observed.",
            )

        mfe, mfe_idx = path.max_favorable_excursion()

        if path.direction >= 0:
            optimal_target_price = path.entry_price + mfe
        else:
            optimal_target_price = path.entry_price - mfe

        bars_to_reach = mfe_idx + 1 if mfe_idx >= 0 else 0
        rationale = (
            f"Actual MFE of {mfe:.5f} reached after {bars_to_reach} bar(s) — "
            "used directly as the optimal target distance, no RR-ratio heuristic applied."
            if mfe_idx >= 0
            else "Path never moved favorably from entry — optimal target distance is 0."
        )

        return HindsightTakeProfitLabel(
            instrument=path.instrument,
            direction=path.direction,
            entry_price=path.entry_price,
            optimal_target_price=optimal_target_price,
            optimal_target_distance=mfe,
            bars_to_reach=bars_to_reach,
            realized_return_pct=path.realized_return_pct(),
            rationale=rationale,
        )
