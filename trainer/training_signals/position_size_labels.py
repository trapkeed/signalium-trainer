"""Phase S4 — hindsight-optimal position-size labels.

Explicitly NOT fixed-fractional / plain-Kelly / ATR-volatility-adjusted
sizing (unlike `trainer.execution.position_sizer` and
`trainer.icari.position_sizing_engine`, both of which apply a live sizing
*rule*). Per the S4 spec: "Train using risk adjusted returns." The label is
derived purely from what the trade actually returned relative to the risk it
actually carried (realized max adverse excursion), then mapped onto a
`[0.0, 1.0]` size-signal scale so it is directly usable as a training target
for `QDEOutputHeads.size_signal`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from trainer.training_signals.outcome_path import TradeOutcomePath

# Risk-adjusted-return magnitude at which the size-signal label saturates
# towards 1.0. This is a scaling constant for the label's shape, not a sizing
# rule — it does not decide how much capital to risk on any live trade.
_SATURATION_RAR = 3.0


@dataclass
class HindsightPositionSizeLabel:
    instrument: str
    realized_return_pct: float
    realized_risk_pct: float
    risk_adjusted_return: float
    size_signal_label: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "realized_return_pct": round(self.realized_return_pct, 5),
            "realized_risk_pct": round(self.realized_risk_pct, 5),
            "risk_adjusted_return": round(self.risk_adjusted_return, 4),
            "size_signal_label": round(self.size_signal_label, 4),
            "rationale": self.rationale,
        }


class HindsightPositionSizeLabeler:
    """Computes a `[0.0, 1.0]` position-size training label from a realized
    trade outcome's risk-adjusted return, i.e. `realized_return_pct /
    realized_risk_pct` where risk is the trade's own realized max adverse
    excursion (as a fraction of entry price) — never a heuristic account-risk
    percentage.

    A trade that returned a lot for the drawdown it actually endured should
    be labeled as "should have been sized up"; a trade that endured a large
    drawdown for little or negative return should be labeled as "should have
    been sized down or skipped." The mapping to `[0, 1]` uses a smooth
    saturating curve (`tanh`) rather than a linear or hard-coded step
    function so the label degrades gracefully for extreme realized outcomes.
    """

    def label(self, path: TradeOutcomePath) -> HindsightPositionSizeLabel:
        realized_return_pct = path.realized_return_pct()
        realized_risk_abs = path.max_adverse_excursion()
        realized_risk_pct = (
            realized_risk_abs / path.entry_price if path.entry_price else 0.0
        )

        if realized_risk_pct <= 1e-9:
            # No adverse excursion at all was realized — the trade carried
            # effectively zero realized risk. A positive return here is
            # capped at full size; a non-positive return is treated as
            # carrying no signal (0.5, neutral) rather than divide-by-zero.
            if realized_return_pct > 0:
                risk_adjusted_return = _SATURATION_RAR
            else:
                risk_adjusted_return = 0.0
        else:
            risk_adjusted_return = realized_return_pct / realized_risk_pct

        size_signal_label = 0.5 * (1.0 + math.tanh(risk_adjusted_return / _SATURATION_RAR))

        rationale = (
            f"risk_adjusted_return={risk_adjusted_return:.3f} "
            f"(return={realized_return_pct:.4%} / risk={realized_risk_pct:.4%}) "
            f"-> size_signal_label={size_signal_label:.3f}"
        )

        return HindsightPositionSizeLabel(
            instrument=path.instrument,
            realized_return_pct=realized_return_pct,
            realized_risk_pct=realized_risk_pct,
            risk_adjusted_return=risk_adjusted_return,
            size_signal_label=size_signal_label,
            rationale=rationale,
        )
