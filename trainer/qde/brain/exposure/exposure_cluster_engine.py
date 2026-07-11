"""QDE Brain — exposure: hidden portfolio concentration analysis.

``ExposureClusterEngine`` answers the question the per-symbol engines
cannot: given all currently open positions, what is the *portfolio's*
true exposure across four hidden dimensions?

Why "hidden"
------------
A portfolio of EURUSD long, GBPUSD long, XAUUSD long, and USDCHF short
looks like four independent instruments.  Each has its own conviction,
uncertainty, and projection score.  But all four express the same
underlying view: *short USD*.  The portfolio is not diversified across
four opportunities — it has one massive USD bet distributed across four
instruments.  The per-symbol engines cannot see this; only a cross-
instrument computation can.

Four concentration dimensions
------------------------------
1. **Currency** — net currency exposure, derived from
   ``InstrumentInfo.currency_exposures()`` (the existing authoritative
   source, not redefined here).  EURUSD long gives EUR+1, USD−1.
   Weighted by position size.  Net USD exposure across the portfolio is
   the sum of all USD contributions, signed.

2. **Sector / asset class** — net weight per asset class (FOREX,
   COMMODITY, INDEX, CRYPTO).  A portfolio full of FOREX positions is
   sector-concentrated even if it spans many currency pairs.

3. **Structural correlation** — inferred from ``InstrumentInfo.tags``
   without a live correlation matrix.  Instruments sharing tags like
   ``safe_haven``, ``risk_on``, ``commodity_currency`` tend to move
   together in stress regimes.  The correlation concentration is the HHI
   of the net weight across these structural groups.  An optional live
   ``CorrelationMatrix`` (from ``trainer.portfolio.correlation_engine``)
   can be supplied to replace the structural fallback with realized
   pairwise correlations.

4. **Regional / macro** — net weight per ``InstrumentInfo.region``
   (NORTH_AMERICA, EUROPE, ASIA_PACIFIC, GLOBAL).  A portfolio of only
   European FX pairs is regionally concentrated.

HHI formula
-----------
Herfindahl-Hirschman Index: ``Σ (|net_i| / total)²``.

  * 1/n = perfectly diversified across n groups
  * 1.0 = all exposure in one group

This is the same mathematical concept as ``PortfolioScorer``'s
``_concentration_score`` but applied to different data (net currency /
sector / regional exposure, not position weights).  It is NOT imported
from there — the inputs and interpretation differ.

Design rules
------------
* Stateless — takes positions + registry, returns an ``ExposureCluster``.
* Reuses ``InstrumentInfo.currency_exposures()`` — never re-derives it.
* Accepts an optional live ``CorrelationMatrix`` for the correlation
  dimension; falls back to structural-tag-based grouping when absent.
* Does not import ``QDEDecision`` or any execution-layer type.
* Does not set concentration *limits* — only measures and quantifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from trainer.portfolio.portfolio_instruments import AssetClass, InstrumentInfo, InstrumentRegistry

if TYPE_CHECKING:
    # Soft import so this module can be used without the full portfolio package.
    from trainer.portfolio.correlation_engine import CorrelationMatrix


# ── Structural correlation groups (from InstrumentInfo.tags) ──────────────────

# Priority-ordered: first matching tag wins.
_CORRELATION_TAG_GROUPS = [
    "safe_haven",
    "risk_on",
    "commodity_currency",
    "commodity",
    "crypto",
]
_FOREX_MAJOR_GROUP = "forex_major"
_FOREX_CROSS_GROUP = "forex_cross"
_UNCORRELATED      = "uncorrelated"

# Composite concentration weights for the four dimensions.
_W_CURRENCY     = 0.35
_W_SECTOR       = 0.20
_W_CORRELATION  = 0.30
_W_REGIONAL     = 0.15

assert abs(_W_CURRENCY + _W_SECTOR + _W_CORRELATION + _W_REGIONAL - 1.0) < 1e-9

# Cap on the normalized position count (n_open / _N_POS_CAP).
_N_POS_CAP = 10.0


# ── Internal helpers ──────────────────────────────────────────────────────────


def _hhi(net: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index on absolute net values → [0, 1].

    0 = perfectly spread; 1 = all exposure in one bucket.
    Returns 0 on an empty or all-zero dict.
    """
    if not net:
        return 0.0
    total = sum(abs(v) for v in net.values())
    if total < 1e-12:
        return 0.0
    return sum((abs(v) / total) ** 2 for v in net.values())


def _correlation_group(info: InstrumentInfo) -> str:
    """Infer the structural correlation group from instrument tags."""
    for tag in _CORRELATION_TAG_GROUPS:
        if tag in info.tags:
            return tag
    if info.asset_class == AssetClass.FOREX:
        return _FOREX_MAJOR_GROUP if "major" in info.tags else _FOREX_CROSS_GROUP
    return _UNCORRELATED


def _pairwise_corr_concentration(
    instruments: list[str],
    weights:     list[float],
    matrix:      "CorrelationMatrix",
) -> float:
    """Average |pairwise correlation| weighted by position sizes.

    Mirrors ``PortfolioScorer._correlation_risk`` but weights pairs by
    the *product* of their weights (larger positions contribute more to
    the concentration reading).
    """
    n = len(instruments)
    if n < 2:
        return 0.0
    total_weight = 0.0
    weighted_corr = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pair_w = abs(weights[i]) * abs(weights[j])
            rho = abs(matrix.get(instruments[i], instruments[j]))
            weighted_corr += pair_w * rho
            total_weight  += pair_w
    return weighted_corr / max(total_weight, 1e-12)


# ── PositionExposure ──────────────────────────────────────────────────────────


@dataclass
class PositionExposure:
    """Input descriptor for one currently open position.

    ``weight`` is the fraction of total portfolio notional this position
    represents (0-1).  Callers derive it from position size / account
    balance.  Positive = long, negative = short (encoded in
    ``direction``); ``weight`` is always non-negative.

    The ``direction`` field (+1 long, −1 short) signs all exposure
    contributions: a long EURUSD gives EUR +weight, USD −weight, while a
    short EURUSD gives EUR −weight, USD +weight.
    """

    instrument: str   = ""
    direction:  int   = 1     # +1 long, −1 short
    weight:     float = 0.0   # 0-1 fraction of portfolio notional

    def signed_weight(self) -> float:
        return self.direction * self.weight

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction":  self.direction,
            "weight":     round(self.weight, 4),
        }


# ── ExposureCluster ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExposureCluster:
    """Four-dimensional portfolio concentration snapshot.

    All HHI values are in [0, 1]:
      0 = perfectly spread, 1 = totally concentrated.

    ``dominant_currency``
        The currency with the largest absolute net exposure.  Empty
        string when no positions are open.

    ``composite_concentration``
        Weighted blend of the four HHIs; the overall concentration
        signal used by ``TradeSaturationEngine``.
    """

    # ── Currency dimension ─────────────────────────────────────────────────
    currency_concentrations:   dict = field(default_factory=dict)  # {currency: net}
    currency_hhi:              float = 0.0
    max_single_currency_abs:   float = 0.0   # largest |net| currency exposure
    dominant_currency:         str   = ""

    # ── Sector / asset-class dimension ────────────────────────────────────
    sector_concentrations:     dict = field(default_factory=dict)  # {sector: net}
    sector_hhi:                float = 0.0

    # ── Structural correlation dimension ──────────────────────────────────
    correlation_groups:        dict = field(default_factory=dict)  # {group: net}
    correlation_concentration: float = 0.0

    # ── Regional / macro dimension ────────────────────────────────────────
    regional_concentrations:   dict = field(default_factory=dict)  # {region: net}
    regional_hhi:              float = 0.0

    # ── Summary ───────────────────────────────────────────────────────────
    composite_concentration:   float = 0.0
    n_positions:               int   = 0

    def to_dict(self) -> dict:
        return {
            "currency_concentrations":   {k: round(v, 4)
                                          for k, v in self.currency_concentrations.items()},
            "currency_hhi":              round(self.currency_hhi, 4),
            "max_single_currency_abs":   round(self.max_single_currency_abs, 4),
            "dominant_currency":         self.dominant_currency,
            "sector_concentrations":     {k: round(v, 4)
                                          for k, v in self.sector_concentrations.items()},
            "sector_hhi":                round(self.sector_hhi, 4),
            "correlation_groups":        {k: round(v, 4)
                                          for k, v in self.correlation_groups.items()},
            "correlation_concentration": round(self.correlation_concentration, 4),
            "regional_concentrations":   {k: round(v, 4)
                                          for k, v in self.regional_concentrations.items()},
            "regional_hhi":              round(self.regional_hhi, 4),
            "composite_concentration":   round(self.composite_concentration, 4),
            "n_positions":               self.n_positions,
        }


# ── ExposureClusterEngine ─────────────────────────────────────────────────────


class ExposureClusterEngine:
    """Stateless four-dimensional portfolio concentration engine.

    ``compute()`` takes the full set of currently open positions and
    returns one ``ExposureCluster`` in a single pass.

    ``marginal()`` returns the (before, after) cluster pair when a
    candidate trade is considered, giving the caller the exact
    incremental concentration cost of the decision.
    """

    def __init__(self, registry: InstrumentRegistry | None = None) -> None:
        self._registry = registry or InstrumentRegistry.default()

    # ── Public API ────────────────────────────────────────────────────────

    def compute(
        self,
        positions:   list[PositionExposure],
        corr_matrix: "CorrelationMatrix | None" = None,
    ) -> ExposureCluster:
        """Compute the four concentration dimensions for a position set.

        ``positions`` may be empty (returns a zero-concentration cluster).
        ``corr_matrix`` is optional; when absent, structural-tag grouping
        is used for the correlation dimension.
        """
        if not positions:
            return ExposureCluster()

        # ── Normalise weights to sum to 1.0 ───────────────────────────────
        total_w = sum(abs(p.weight) for p in positions)
        if total_w < 1e-12:
            return ExposureCluster(n_positions=len(positions))

        # ── Accumulate per-dimension net exposures ────────────────────────
        currency_net:  dict[str, float] = {}
        sector_net:    dict[str, float] = {}
        corr_net:      dict[str, float] = {}
        regional_net:  dict[str, float] = {}

        corr_instruments: list[str]   = []
        corr_weights:     list[float] = []

        for pos in positions:
            info = self._registry.get(pos.instrument)
            if info is None:
                # Unknown instrument: treat as uncorrelated, no currency data
                _add(corr_net, _UNCORRELATED, pos.signed_weight() / total_w)
                continue

            norm_w = pos.signed_weight() / total_w

            # Currency — use signed norm_w so direction flips the sign of
            # every currency leg: a short USDCHF (norm_w < 0) correctly
            # yields negative USD and positive CHF exposure.
            for currency, sign in info.currency_exposures().items():
                _add(currency_net, currency, sign * norm_w)

            # Sector
            _add(sector_net, info.asset_class.value, norm_w)

            # Correlation group
            _add(corr_net, _correlation_group(info), norm_w)
            corr_instruments.append(pos.instrument)
            corr_weights.append(norm_w)

            # Region
            _add(regional_net, info.region.value, norm_w)

        # ── Concentration HHIs ────────────────────────────────────────────
        currency_hhi = _hhi(currency_net)
        sector_hhi   = _hhi(sector_net)
        regional_hhi = _hhi(regional_net)

        # Correlation: prefer live matrix over structural grouping
        if corr_matrix is not None and len(corr_instruments) >= 2:
            corr_conc = _pairwise_corr_concentration(
                corr_instruments, corr_weights, corr_matrix
            )
        else:
            corr_conc = _hhi(corr_net)

        composite = (
            _W_CURRENCY    * currency_hhi
            + _W_SECTOR    * sector_hhi
            + _W_CORRELATION * corr_conc
            + _W_REGIONAL  * regional_hhi
        )

        # Dominant currency
        dom_ccy = ""
        max_abs = 0.0
        for ccy, net in currency_net.items():
            if abs(net) > max_abs:
                max_abs = abs(net)
                dom_ccy = ccy

        return ExposureCluster(
            currency_concentrations=dict(currency_net),
            currency_hhi=round(currency_hhi, 6),
            max_single_currency_abs=round(max_abs, 6),
            dominant_currency=dom_ccy,
            sector_concentrations=dict(sector_net),
            sector_hhi=round(sector_hhi, 6),
            correlation_groups=dict(corr_net),
            correlation_concentration=round(corr_conc, 6),
            regional_concentrations=dict(regional_net),
            regional_hhi=round(regional_hhi, 6),
            composite_concentration=round(composite, 6),
            n_positions=len(positions),
        )

    def marginal(
        self,
        candidate:          PositionExposure,
        current_positions:  list[PositionExposure],
        corr_matrix:        "CorrelationMatrix | None" = None,
    ) -> tuple[ExposureCluster, ExposureCluster]:
        """Return (cluster_before, cluster_after) for a candidate trade.

        The delta between the two clusters is the marginal concentration
        cost of adding this trade to the portfolio.
        """
        before = self.compute(current_positions, corr_matrix=corr_matrix)
        after  = self.compute(current_positions + [candidate], corr_matrix=corr_matrix)
        return before, after

    def marginal_deltas(
        self,
        candidate:         PositionExposure,
        current_positions: list[PositionExposure],
        corr_matrix:       "CorrelationMatrix | None" = None,
    ) -> dict[str, float]:
        """Return a dict of concentration deltas introduced by the candidate.

        Keys: ``delta_currency_hhi``, ``delta_sector_hhi``,
        ``delta_correlation``, ``delta_regional_hhi``,
        ``delta_composite``.  All values are after − before; positive
        means adding the candidate increases concentration in that
        dimension.
        """
        before, after = self.marginal(candidate, current_positions, corr_matrix)
        return {
            "delta_currency_hhi":  round(after.currency_hhi - before.currency_hhi, 6),
            "delta_sector_hhi":    round(after.sector_hhi   - before.sector_hhi,   6),
            "delta_correlation":   round(after.correlation_concentration
                                         - before.correlation_concentration, 6),
            "delta_regional_hhi":  round(after.regional_hhi - before.regional_hhi, 6),
            "delta_composite":     round(after.composite_concentration
                                         - before.composite_concentration, 6),
        }


# ── Internal accumulator ──────────────────────────────────────────────────────


def _add(d: dict[str, float], key: str, value: float) -> None:
    d[key] = d.get(key, 0.0) + value
