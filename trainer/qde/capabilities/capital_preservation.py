"""Phase 28.3 — Capability 8: Capital Preservation.

Evaluates account state against configurable circuit breakers and determines
whether new entries or scaling are permitted.

CONTRACT
--------
- Returns PreservationStatus (advisory only).
- Does NOT produce a QDEDecision.
- Does NOT instruct any trade action.
- May NOT import QDEDecision or QDEIntelligence.
"""

from __future__ import annotations

from trainer.qde.capabilities.cap_models import (
    AccountState,
    CircuitBreakerStatus,
    MarketSnapshot,
    PreservationStatus,
)


class CapitalPreservationCapability:
    """Evaluates circuit breakers and capital preservation status.

    This capability is the sole owner of the QDE's Capital Preservation
    responsibility. It reports whether any preservation limits are breached.
    The QDE uses this output to determine whether to block entries,
    block scaling, or force exits.

    It never forces anything — it reports. The QDE decides.
    """

    CAPABILITY_NAME = "capital_preservation"

    def __init__(
        self,
        daily_loss_limit_pct: float = 0.03,
        drawdown_limit_pct: float = 0.10,
        consecutive_loss_limit: int = 5,
        warning_daily_pct: float = 0.02,
        warning_drawdown_pct: float = 0.07,
    ) -> None:
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.drawdown_limit_pct = drawdown_limit_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.warning_daily_pct = warning_daily_pct
        self.warning_drawdown_pct = warning_drawdown_pct

    def evaluate(self, snapshot: MarketSnapshot) -> PreservationStatus:
        """Evaluate capital preservation status against all circuit breakers.

        Args:
            snapshot: Current market snapshot (account state required).

        Returns:
            PreservationStatus — advisory only, not a trade instruction.
        """
        account = snapshot.account
        warnings: list[str] = []

        if account is None:
            return PreservationStatus(
                capability=self.CAPABILITY_NAME,
                snapshot_id=snapshot.snapshot_id,
                confidence=0.5,
                warnings=["No account state in snapshot — preservation check skipped"],
                circuit_breaker=CircuitBreakerStatus.CLEAR,
                new_entries_allowed=True,
                scaling_allowed=True,
            )

        daily_pct = abs(min(account.daily_pnl_pct, 0.0))
        drawdown_pct = account.drawdown_from_peak
        consec_losses = account.consecutive_losses

        breaches: list[str] = []
        forced_exit = False
        forced_reason = ""

        # ── Daily loss limit ──────────────────────────────────────────────────
        daily_headroom = max(0.0, self.daily_loss_limit_pct - daily_pct)
        if daily_pct >= self.daily_loss_limit_pct:
            breaches.append(f"Daily loss {daily_pct:.2%} ≥ limit {self.daily_loss_limit_pct:.2%}")
            forced_exit = True
            forced_reason = f"Daily loss limit breached ({daily_pct:.2%})"
        elif daily_pct >= self.warning_daily_pct:
            warnings.append(f"Daily loss {daily_pct:.2%} approaching limit {self.daily_loss_limit_pct:.2%}")

        # ── Drawdown limit ────────────────────────────────────────────────────
        drawdown_headroom = max(0.0, self.drawdown_limit_pct - drawdown_pct)
        if drawdown_pct >= self.drawdown_limit_pct:
            breaches.append(f"Drawdown {drawdown_pct:.2%} ≥ limit {self.drawdown_limit_pct:.2%}")
            forced_exit = True
            forced_reason = forced_reason or f"Drawdown limit breached ({drawdown_pct:.2%})"
        elif drawdown_pct >= self.warning_drawdown_pct:
            warnings.append(f"Drawdown {drawdown_pct:.2%} approaching limit {self.drawdown_limit_pct:.2%}")

        # ── Consecutive loss limit ────────────────────────────────────────────
        if consec_losses >= self.consecutive_loss_limit:
            breaches.append(f"Consecutive losses {consec_losses} ≥ limit {self.consecutive_loss_limit}")

        # ── Determine circuit breaker status ──────────────────────────────────
        if breaches:
            cb = CircuitBreakerStatus.TRIGGERED
        elif warnings:
            cb = CircuitBreakerStatus.WARNING
        else:
            cb = CircuitBreakerStatus.CLEAR

        new_entries_allowed = cb != CircuitBreakerStatus.TRIGGERED
        scaling_allowed = cb == CircuitBreakerStatus.CLEAR
        confidence = 1.0 if account is not None else 0.5

        return PreservationStatus(
            capability=self.CAPABILITY_NAME,
            snapshot_id=snapshot.snapshot_id,
            confidence=confidence,
            warnings=warnings,
            circuit_breaker=cb,
            daily_loss_limit_pct=self.daily_loss_limit_pct,
            daily_loss_current_pct=daily_pct,
            daily_loss_headroom_pct=daily_headroom,
            drawdown_limit_pct=self.drawdown_limit_pct,
            drawdown_current_pct=drawdown_pct,
            drawdown_headroom_pct=drawdown_headroom,
            consecutive_loss_limit=self.consecutive_loss_limit,
            consecutive_losses_current=consec_losses,
            new_entries_allowed=new_entries_allowed,
            scaling_allowed=scaling_allowed,
            forced_exit_required=forced_exit,
            forced_exit_reason=forced_reason,
        )
