"""QDE Brain — exposure: rolling store for concentration-outcome learning.

``ExposureMemory`` accumulates ``(ExposureObservation → ExposureOutcome)``
pairs as trades complete, forming the empirical distribution from which
``TradeSaturationEngine`` later estimates whether the current portfolio
concentration level has historically produced good or bad outcomes.

Design mirrors ``trainer.qde.brain.projection.projection_memory``:
  * Bounded FIFO per-portfolio store (no per-symbol bucketing — exposure
    is a portfolio-level concept, not a per-symbol one).
  * ``to_snapshot()`` / ``load_snapshot()`` for cross-session persistence.
  * No business logic — data store only.

What ``ExposureObservation`` captures
--------------------------------------
The portfolio concentration state immediately *before* a candidate trade
is added, plus the marginal change the trade would introduce.  This lets
the engine later ask: "given that the portfolio looked like X and I was
considering a trade that would increase currency_hhi by ΔY, how did
things turn out?"

What ``ExposureOutcome`` captures
-----------------------------------
The realized portfolio quality over the candidate trade's lifetime.
``outcome_quality`` (0-1) is the primary learning target — a composite
of portfolio return, drawdown ratio, and exposure P&L contribution.
Callers compute it from live account data once the trade closes.

No hardcoded thresholds
------------------------
Neither ``ExposureObservation`` nor ``ExposureOutcome`` contains any
concentration *limit*.  ``TradeSaturationEngine`` learns what level of
concentration is appropriate from the outcome distribution — the data
says when concentration hurts, not the code.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field

# Bounded FIFO capacity — once exceeded the oldest record is evicted.
_DEFAULT_MAX_RECORDS = 2000

# Concentration-state vector dimension (must match as_vector() below).
_N_DIMS = 5


# ── ExposureObservation ───────────────────────────────────────────────────────


@dataclass
class ExposureObservation:
    """Portfolio concentration state and marginal change at trade-entry time.

    All HHI values (Herfindahl-Hirschman Index) are in [0, 1]:
      0 = perfectly spread across many currencies/sectors/regions
      1 = all exposure in one currency/sector/region

    ``n_positions_norm`` is n_open_positions / 10, capped at 1.0, so
    the vector remains scale-comparable across observation times.

    The five-dimensional ``as_vector()`` is the input to the L2-distance
    nearest-neighbour lookup in ``TradeSaturationEngine``.
    """

    # ── Portfolio state BEFORE this trade ────────────────────────────────
    currency_hhi: float = 0.0
    sector_hhi: float = 0.0
    correlation_concentration: float = 0.0
    regional_hhi: float = 0.0
    n_positions_norm: float = 0.0  # n_open / 10, capped at 1.0

    # ── Candidate trade being considered ─────────────────────────────────
    candidate_instrument: str = ""
    candidate_direction: int = 1  # +1 long, −1 short

    # ── Marginal deltas introduced by this trade ─────────────────────────
    marginal_currency_hhi: float = 0.0
    marginal_sector_hhi: float = 0.0
    marginal_correlation_risk: float = 0.0
    marginal_regional_hhi: float = 0.0
    marginal_composite: float = 0.0  # weighted blend of deltas

    timestamp: float = field(default_factory=time.time)

    def as_vector(self) -> list[float]:
        """Five-dimensional concentration state for nearest-neighbour lookup.

        Uses the BEFORE-state (pre-trade portfolio), not the marginals,
        so the engine can find historical records where the portfolio
        looked similar to now — and see what happened when a trade was
        added to such a portfolio.
        """
        return [
            self.currency_hhi,
            self.sector_hhi,
            self.correlation_concentration,
            self.regional_hhi,
            self.n_positions_norm,
        ]

    def marginal_vector(self) -> list[float]:
        """Four-dimensional marginal change vector for more precise lookup."""
        return [
            self.marginal_currency_hhi,
            self.marginal_sector_hhi,
            self.marginal_correlation_risk,
            self.marginal_regional_hhi,
        ]

    def to_dict(self) -> dict:
        return {
            "currency_hhi": round(self.currency_hhi, 4),
            "sector_hhi": round(self.sector_hhi, 4),
            "correlation_concentration": round(self.correlation_concentration, 4),
            "regional_hhi": round(self.regional_hhi, 4),
            "n_positions_norm": round(self.n_positions_norm, 4),
            "candidate_instrument": self.candidate_instrument,
            "candidate_direction": self.candidate_direction,
            "marginal_currency_hhi": round(self.marginal_currency_hhi, 4),
            "marginal_sector_hhi": round(self.marginal_sector_hhi, 4),
            "marginal_correlation_risk": round(self.marginal_correlation_risk, 4),
            "marginal_regional_hhi": round(self.marginal_regional_hhi, 4),
            "marginal_composite": round(self.marginal_composite, 4),
            "timestamp": round(self.timestamp, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExposureObservation:
        return cls(
            currency_hhi=d.get("currency_hhi", 0.0),
            sector_hhi=d.get("sector_hhi", 0.0),
            correlation_concentration=d.get("correlation_concentration", 0.0),
            regional_hhi=d.get("regional_hhi", 0.0),
            n_positions_norm=d.get("n_positions_norm", 0.0),
            candidate_instrument=d.get("candidate_instrument", ""),
            candidate_direction=d.get("candidate_direction", 1),
            marginal_currency_hhi=d.get("marginal_currency_hhi", 0.0),
            marginal_sector_hhi=d.get("marginal_sector_hhi", 0.0),
            marginal_correlation_risk=d.get("marginal_correlation_risk", 0.0),
            marginal_regional_hhi=d.get("marginal_regional_hhi", 0.0),
            marginal_composite=d.get("marginal_composite", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )


# ── ExposureOutcome ───────────────────────────────────────────────────────────


@dataclass
class ExposureOutcome:
    """Portfolio-level outcome over the lifetime of the recorded trade.

    ``outcome_quality`` (0-1) is the primary learning target.  A
    composite of return quality, drawdown cost, and contribution balance:

    Callers compute it as:
        quality = clamp(0.5 + portfolio_return / reference_vol, 0, 1)

    where ``reference_vol`` is the historical portfolio volatility
    (or 0.01 as a floor).  This normalises quality relative to the
    portfolio's own risk scale — no hardcoded return threshold.

    ``portfolio_return``
        Realized portfolio return (all positions) during this trade's
        duration.  Positive = portfolio made money while this trade was
        open.

    ``exposure_pnl_contribution``
        Estimated contribution of this specific concentration to the
        portfolio's P&L.  Positive = concentration helped; negative =
        it hurt (e.g. all USD-short positions moved against each other).

    ``duration_seconds``
        How long the recorded trade was open, in seconds.
    """

    portfolio_return: float = 0.0  # signed fraction
    portfolio_max_drawdown: float = 0.0  # non-negative fraction
    outcome_quality: float = 0.5  # 0-1, 0.5 = neutral
    exposure_pnl_contribution: float = 0.0  # signed, relative to portfolio
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "portfolio_return": round(self.portfolio_return, 6),
            "portfolio_max_drawdown": round(self.portfolio_max_drawdown, 4),
            "outcome_quality": round(self.outcome_quality, 4),
            "exposure_pnl_contribution": round(self.exposure_pnl_contribution, 6),
            "duration_seconds": round(self.duration_seconds, 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExposureOutcome:
        return cls(
            portfolio_return=d.get("portfolio_return", 0.0),
            portfolio_max_drawdown=d.get("portfolio_max_drawdown", 0.0),
            outcome_quality=d.get("outcome_quality", 0.5),
            exposure_pnl_contribution=d.get("exposure_pnl_contribution", 0.0),
            duration_seconds=d.get("duration_seconds", 0.0),
        )

    @staticmethod
    def compute_quality(
        portfolio_return: float,
        portfolio_max_drawdown: float,
        reference_vol: float = 0.01,
    ) -> float:
        """Derive ``outcome_quality`` from raw portfolio metrics.

        Normalises return relative to portfolio volatility so the quality
        signal is comparable across different instruments and periods.
        No hardcoded return target — "good" is relative to the
        portfolio's own risk level.
        """
        ref = max(reference_vol, 1e-4)
        raw = 0.5 + portfolio_return / ref - 0.5 * portfolio_max_drawdown / ref
        return max(0.0, min(1.0, raw))


# ── ExposureRecord ────────────────────────────────────────────────────────────


@dataclass
class ExposureRecord:
    """One completed (observation → outcome) pair."""

    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    observation: ExposureObservation = field(default_factory=ExposureObservation)
    outcome: ExposureOutcome = field(default_factory=ExposureOutcome)
    created_at: float = field(default_factory=time.time)

    def recency_weight(self, now: float, tau_days: float = 30.0) -> float:
        """Exponential recency weight: 1.0 when fresh, decays over tau_days."""
        elapsed_days = max(0.0, now - self.created_at) / 86400.0
        return max(0.0, 1.0 / (1.0 + elapsed_days / max(tau_days, 1e-9)))

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "observation": self.observation.to_dict(),
            "outcome": self.outcome.to_dict(),
            "created_at": round(self.created_at, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExposureRecord:
        return cls(
            record_id=d.get("record_id", str(uuid.uuid4())[:12]),
            observation=ExposureObservation.from_dict(d.get("observation", {})),
            outcome=ExposureOutcome.from_dict(d.get("outcome", {})),
            created_at=d.get("created_at", 0.0),
        )


# ── ExposureMemory ────────────────────────────────────────────────────────────


class ExposureMemory:
    """Bounded rolling store of exposure-outcome records.

    Portfolio-level (not per-symbol): one deque for the entire portfolio.

    Responsibilities
    ----------------
    * ``record()`` — append a new (observation, outcome) pair, evicting
      the oldest when the store is full.
    * ``records()`` — return a snapshot of current records (newest last).
    * ``to_snapshot()`` / ``load_snapshot()`` — persistence.

    No business logic — that lives in ``TradeSaturationEngine``.
    """

    def __init__(self, max_records: int = _DEFAULT_MAX_RECORDS) -> None:
        self._max_records = max(1, max_records)
        self._store: deque[ExposureRecord] = deque(maxlen=self._max_records)

    # ── Write ──────────────────────────────────────────────────────────────

    def record(
        self,
        observation: ExposureObservation,
        outcome: ExposureOutcome,
        now: float | None = None,
    ) -> ExposureRecord:
        """Append one (observation → outcome) pair.

        Called once per completed trade from the learning loop.
        The oldest record is automatically evicted when capacity is
        exceeded.
        """
        now = now if now is not None else time.time()
        rec = ExposureRecord(observation=observation, outcome=outcome, created_at=now)
        self._store.append(rec)
        return rec

    # ── Read ───────────────────────────────────────────────────────────────

    def records(self) -> list[ExposureRecord]:
        """All current records, oldest first."""
        return list(self._store)

    def __len__(self) -> int:
        return len(self._store)

    # ── Persistence ────────────────────────────────────────────────────────

    def to_snapshot(self) -> dict:
        return {
            "max_records": self._max_records,
            "records": [r.to_dict() for r in self._store],
        }

    @classmethod
    def load_snapshot(cls, snapshot: dict) -> ExposureMemory:
        mem = cls(max_records=snapshot.get("max_records", _DEFAULT_MAX_RECORDS))
        for d in snapshot.get("records", []):
            try:
                rec = ExposureRecord.from_dict(d)
                mem._store.append(rec)
            except Exception:
                pass
        return mem
