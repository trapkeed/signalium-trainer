"""S30-spec — Universal Symbol Normalization Engine.

Maps every broker/provider-specific spelling of an instrument (suffixes,
prefixes, separators, casing, vendor aliases) to one canonical symbol, e.g.
``EUR/USD``, ``EURUSD.r``, ``EURUSDm``, ``EURUSD.pro``, ``EURUSD_raw`` all
normalize to ``EURUSD``.

Design:
  - A fixed set of canonical symbols with admin-manageable enable/disable.
  - A fixed set of built-in vendor aliases (``GOLD`` -> ``XAUUSD`` etc.).
  - Admin-defined aliases, layered on top of the built-ins, editable at
    runtime (``add_alias`` / ``remove_alias``).
  - "Runtime learning": ``learn_from_observation`` records a
    provider-supplied spelling the normalizer hasn't seen paired with the
    canonical symbol the caller asserts it resolves to (e.g. from a
    successful trade confirmation), so future occurrences of that exact
    string normalize without needing an admin to hand-register it.
  - Structural stripping of common broker suffixes/prefixes/separators
    happens *before* alias lookup, so aliases only need to cover
    vendor-specific renamings (GOLD, DJI30), not every suffix variant.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


#: Canonical symbols this engine knows about out of the box, grouped by
#: asset class per the S30 spec. Crypto is intentionally absent from the
#: default-enabled set but the registry supports adding it (architecturally
#: supported, disabled by default).
DEFAULT_SYMBOLS: dict[str, str] = {
    "EURUSD": "forex", "GBPUSD": "forex", "USDJPY": "forex", "AUDUSD": "forex",
    "USDCHF": "forex", "USDCAD": "forex", "NZDUSD": "forex",
    "XAUUSD": "metal", "XAGUSD": "metal",
    "US30": "index", "NAS100": "index", "SPX500": "index", "GER40": "index", "UK100": "index",
}

#: Built-in vendor aliases that don't fit the structural
#: suffix/prefix/separator-stripping rule (e.g. "GOLD" for XAUUSD is a
#: different word entirely, not a decorated spelling of XAUUSD).
BUILTIN_ALIASES: dict[str, str] = {
    "GOLD": "XAUUSD", "XAU/USD": "XAUUSD",
    "SILVER": "XAGUSD", "XAG/USD": "XAGUSD",
    "DJI30": "US30", "DOW": "US30", "DOW30": "US30",
    "NASDAQ100": "NAS100", "NDX": "NAS100",
    "SP500": "SPX500", "SPX": "SPX500",
    "DAX": "GER40", "DAX40": "GER40",
    "FTSE": "UK100", "FTSE100": "UK100",
    "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY",
    "AUD/USD": "AUDUSD", "USD/CHF": "USDCHF", "USD/CAD": "USDCAD", "NZD/USD": "NZDUSD",
}

#: Broker decoration patterns stripped structurally before alias lookup.
#: Order matters: separators first, then suffix/prefix noise.
_SEPARATOR_RE = re.compile(r"[/_\-\s]")
_SUFFIX_RE = re.compile(r"\.(r|pro|raw|cash|m|micro|ecn)$", re.IGNORECASE)
_TRAILING_LETTER_RE = re.compile(r"(m|c|i)$", re.IGNORECASE)  # e.g. EURUSDm, US30m — trailing broker-type letter


@dataclass
class NormalizationResult:
    input_symbol: str
    canonical: str | None
    matched_via: str  # "exact" | "builtin_alias" | "admin_alias" | "learned" | "structural" | "unresolved"
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_symbol": self.input_symbol, "canonical": self.canonical,
            "matched_via": self.matched_via, "enabled": self.enabled,
        }


class UnknownSymbolError(Exception):
    pass


class SymbolNormalizer:
    def __init__(self, symbols: dict[str, str] | None = None) -> None:
        self._symbols: dict[str, str] = dict(symbols if symbols is not None else DEFAULT_SYMBOLS)
        self._enabled: dict[str, bool] = {s: True for s in self._symbols}
        self._admin_aliases: dict[str, str] = {}
        self._learned_aliases: dict[str, tuple[str, float]] = {}  # raw -> (canonical, learned_at)

    # ── Admin symbol management ─────────────────────────────────────────────

    def add_symbol(self, canonical: str, asset_class: str = "forex", enabled: bool = True) -> None:
        canonical = canonical.upper()
        self._symbols[canonical] = asset_class
        self._enabled[canonical] = enabled

    def enable_symbol(self, canonical: str) -> None:
        self._enabled[canonical.upper()] = True

    def disable_symbol(self, canonical: str) -> None:
        self._enabled[canonical.upper()] = False

    def is_enabled(self, canonical: str) -> bool:
        return self._enabled.get(canonical.upper(), False)

    def list_symbols(self) -> list[dict[str, Any]]:
        return [
            {"symbol": s, "asset_class": cls, "enabled": self._enabled.get(s, False)}
            for s, cls in sorted(self._symbols.items())
        ]

    # ── Admin alias management ──────────────────────────────────────────────

    def add_alias(self, raw: str, canonical: str) -> None:
        canonical = canonical.upper()
        if canonical not in self._symbols:
            raise UnknownSymbolError(f"Cannot alias to unregistered canonical symbol {canonical!r}")
        self._admin_aliases[raw.upper()] = canonical

    def remove_alias(self, raw: str) -> None:
        self._admin_aliases.pop(raw.upper(), None)

    # ── Runtime learning ─────────────────────────────────────────────────────

    def learn_from_observation(self, raw: str, canonical: str) -> None:
        """Record a provider-supplied spelling the caller has independently
        confirmed resolves to ``canonical`` (e.g. a fill confirmation from a
        broker feed), so future exact-string occurrences resolve without
        an admin needing to hand-register the alias."""
        canonical = canonical.upper()
        if canonical not in self._symbols:
            raise UnknownSymbolError(f"Cannot learn alias to unregistered canonical symbol {canonical!r}")
        self._learned_aliases[raw.upper()] = (canonical, time.time())
        logger.info(f"SymbolNormalizer: learned {raw!r} -> {canonical!r}")

    # ── Normalization ────────────────────────────────────────────────────────

    def _strip_structural_decoration(self, s: str) -> str:
        s = _SEPARATOR_RE.sub("", s)
        s = _SUFFIX_RE.sub("", s)
        return s

    def normalize(self, raw: str) -> NormalizationResult:
        if not raw:
            return NormalizationResult(raw, None, "unresolved", enabled=False)
        upper = raw.upper()

        if upper in self._symbols:
            return NormalizationResult(raw, upper, "exact", self.is_enabled(upper))

        if upper in self._admin_aliases:
            canonical = self._admin_aliases[upper]
            return NormalizationResult(raw, canonical, "admin_alias", self.is_enabled(canonical))

        if upper in BUILTIN_ALIASES:
            canonical = BUILTIN_ALIASES[upper]
            return NormalizationResult(raw, canonical, "builtin_alias", self.is_enabled(canonical))

        if upper in self._learned_aliases:
            canonical, _ = self._learned_aliases[upper]
            return NormalizationResult(raw, canonical, "learned", self.is_enabled(canonical))

        # Structural stripping: separators, dot-suffixes, then a trailing
        # broker-type letter (EURUSDm -> EURUSD), checked against the
        # canonical symbol set only (not aliases — a decorated alias would
        # be unusual and ambiguous to strip).
        stripped = self._strip_structural_decoration(upper)
        if stripped in self._symbols:
            return NormalizationResult(raw, stripped, "structural", self.is_enabled(stripped))
        trimmed = _TRAILING_LETTER_RE.sub("", stripped)
        if trimmed != stripped and trimmed in self._symbols:
            return NormalizationResult(raw, trimmed, "structural", self.is_enabled(trimmed))

        return NormalizationResult(raw, None, "unresolved", enabled=False)

    def resolve(self, raw: str) -> str:
        """Convenience wrapper — returns the canonical symbol or raises."""
        result = self.normalize(raw)
        if result.canonical is None:
            raise UnknownSymbolError(f"Could not normalize symbol {raw!r}")
        if not result.enabled:
            raise UnknownSymbolError(f"Symbol {result.canonical!r} (from {raw!r}) is disabled")
        return result.canonical
