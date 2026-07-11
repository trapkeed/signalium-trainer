"""Phase S4 — the shared "what actually happened" input type.

Every labeler in `trainer.training_signals` consumes a `TradeOutcomePath`
rather than a live market snapshot. This is the deliberate boundary between
Phase S3 (`trainer.qde.qde_position_manager`, which reasons over *live*
telemetry to make a real-time call) and Phase S4 (which reasons over
*historical, fully-realized* telemetry to produce a training label). Reuses
`OHLCBar` (`trainer.microstructure.micro_models`) for the future price path
rather than inventing a second OHLC type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trainer.microstructure.micro_models import OHLCBar


@dataclass
class TradeOutcomePath:
    """The realized future price path following a trade entry.

    `direction` is `1` for long, `-1` for short — matching the convention
    already used by `trainer.validation.trading.TradeResult`.

    ``instrument`` **must** be a canonical symbol (e.g. ``"EURUSD"``, not
    ``"EURUSD.r"``).  The constructor enforces this via
    ``trainer.symbols.assert_canonical`` so broker-decorated strings never
    reach the label-generation pipeline.
    """

    instrument: str
    direction: int
    entry_price: float
    entry_time: float = 0.0
    future_bars: list[OHLCBar] = field(default_factory=list)

    def __post_init__(self) -> None:
        from trainer.symbols import assert_canonical

        assert_canonical(self.instrument)

    def favorable_excursion(self, bar: OHLCBar) -> float:
        """Non-negative favorable price distance reached within `bar`."""
        if self.direction >= 0:
            return max(0.0, bar.high - self.entry_price)
        return max(0.0, self.entry_price - bar.low)

    def adverse_excursion(self, bar: OHLCBar) -> float:
        """Non-negative adverse price distance reached within `bar`."""
        if self.direction >= 0:
            return max(0.0, self.entry_price - bar.low)
        return max(0.0, bar.high - self.entry_price)

    def final_price(self) -> float:
        return self.future_bars[-1].close if self.future_bars else self.entry_price

    def realized_return_pct(self) -> float:
        """Realized return as a fraction of entry price (signed)."""
        if self.entry_price == 0:
            return 0.0
        final = self.final_price()
        if self.direction >= 0:
            return (final - self.entry_price) / self.entry_price
        return (self.entry_price - final) / self.entry_price

    def max_favorable_excursion(self) -> tuple[float, int]:
        """(MFE distance, index of the bar where it occurred). Index is -1
        if there are no future bars."""
        mfe = 0.0
        mfe_idx = -1
        for i, bar in enumerate(self.future_bars):
            fe = self.favorable_excursion(bar)
            if fe > mfe:
                mfe = fe
                mfe_idx = i
        return mfe, mfe_idx

    def max_adverse_excursion_through(self, up_to_index: int) -> float:
        """Largest adverse excursion in `future_bars[:up_to_index + 1]`."""
        mae = 0.0
        for bar in self.future_bars[: up_to_index + 1]:
            ae = self.adverse_excursion(bar)
            if ae > mae:
                mae = ae
        return mae

    def max_adverse_excursion(self) -> float:
        return self.max_adverse_excursion_through(len(self.future_bars) - 1)
