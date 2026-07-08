"""Phase 28.3 — Capability 5: Exposure Reasoning.

Evaluates current portfolio exposure across all open trades and determines
whether it is safe to add more exposure.

CONTRACT
--------
- Returns ExposureReport (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from collections import defaultdict

from trainer.qde.capabilities.cap_models import (
    Direction,
    ExposureReport,
    MarketSnapshot,
    OpenTrade,
)


class ExposureReasoningCapability:
    """Evaluates total portfolio exposure and correlation risk.

    This capability contributes to the QDE's Exposure responsibility.
    It never instructs trades — it only reports current exposure state
    and whether headroom exists for additional positions.
    """

    CAPABILITY_NAME = "exposure_reasoning"

    # Maximum gross exposure as fraction of equity
    MAX_GROSS_EXPOSURE = 0.06  # 6% of equity at risk across all trades

    # Maximum single-instrument concentration
    MAX_INSTRUMENT_CONCENTRATION = 0.03  # 3% per instrument

    # Intra-group correlation coefficient assumption
    SAME_GROUP_CORRELATION = 0.80

    # Currency groups for correlation estimation
    CURRENCY_GROUPS: dict[str, str] = {
        "EURUSD": "EUR", "EURGBP": "EUR", "EURJPY": "EUR",
        "GBPUSD": "GBP", "GBPJPY": "GBP", "GBPCHF": "GBP",
        "USDJPY": "USD", "USDCAD": "USD", "USDCHF": "USD",
        "AUDUSD": "AUD", "NZDUSD": "NZD", "USDMXN": "USD",
        "XAUUSD": "GOLD", "BTCUSD": "CRYPTO",
    }

    def analyse(
        self,
        snapshot: MarketSnapshot,
        equity: float,
    ) -> ExposureReport:
        """Evaluate current portfolio exposure.

        Args:
            snapshot: Current market snapshot (all_open_trades used).
            equity: Current account equity for exposure percentage calculation.

        Returns:
            ExposureReport — advisory only, not a trade instruction.
        """
        trades = snapshot.all_open_trades
        warnings: list[str] = []

        if equity <= 0:
            return ExposureReport(
                capability=self.CAPABILITY_NAME,
                snapshot_id=snapshot.snapshot_id,
                confidence=0.0,
                warnings=["Equity is zero or negative — cannot compute exposure"],
                adding_exposure_is_safe=False,
            )

        # Compute per-instrument risk (initial risk in equity fraction)
        instrument_risks: dict[str, float] = defaultdict(float)
        long_exposure = 0.0
        short_exposure = 0.0

        for trade in trades:
            risk = trade.risk_at_open / equity
            instrument_risks[trade.instrument] += risk
            if trade.direction == Direction.LONG:
                long_exposure += risk
            else:
                short_exposure += risk

        gross = long_exposure + short_exposure
        net = long_exposure - short_exposure
        headroom = max(0.0, self.MAX_GROSS_EXPOSURE - gross) / self.MAX_GROSS_EXPOSURE

        # Concentration risk: max single-instrument exposure / gross
        concentration = max(instrument_risks.values()) / gross if gross > 0 else 0.0

        # Correlation risk: count instruments in same group
        groups: dict[str, list[str]] = defaultdict(list)
        for instr in instrument_risks:
            group = self.CURRENCY_GROUPS.get(instr, instr)
            groups[group].append(instr)
        correlated_pairs = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
        corr_risk = min(1.0, correlated_pairs * self.SAME_GROUP_CORRELATION / max(1, len(trades)))

        adding_safe = (
            gross < self.MAX_GROSS_EXPOSURE
            and concentration < 1.0
        )

        if gross >= self.MAX_GROSS_EXPOSURE:
            warnings.append(f"Gross exposure {gross:.1%} at or above maximum {self.MAX_GROSS_EXPOSURE:.1%}")
        if concentration > 0.8:
            warnings.append(f"High concentration: one instrument = {concentration:.1%} of gross risk")
        if corr_risk > 0.5:
            warnings.append(f"High correlation risk: {corr_risk:.1%}")

        confidence = max(0.4, 1.0 - corr_risk * 0.3 - concentration * 0.3)

        return ExposureReport(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            gross_exposure_pct=gross,
            net_directional_exposure=net,
            instrument_exposures=dict(instrument_risks),
            correlation_risk=corr_risk,
            concentration_risk=concentration,
            exposure_headroom_pct=headroom,
            adding_exposure_is_safe=adding_safe,
            notes=f"Gross={gross:.1%} Net={net:+.1%} Headroom={headroom:.1%}",
        )
