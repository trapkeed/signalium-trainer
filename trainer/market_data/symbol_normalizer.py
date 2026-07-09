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
    "DJ30": "US30", "US30CASH": "US30", "US30USD": "US30", "WALLST30": "US30",
    "NASDAQ100": "NAS100", "NDX": "NAS100",
    "SP500": "SPX500", "SPX": "SPX500",
    "DAX": "GER40", "DAX40": "GER40", "DE40": "GER40", "GER40CASH": "GER40", "GERMANY40": "GER40",
    "FTSE": "UK100", "FTSE100": "UK100",
    "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY",
    "AUD/USD": "AUDUSD", "USD/CHF": "USDCHF", "USD/CAD": "USDCAD", "NZD/USD": "NZDUSD",
}

#: Broker-specific profiles — a broker's own house spellings for canonical
#: symbols, layered on top of BUILTIN_ALIASES/admin/learned aliases. Keyed
#: by broker name (case-insensitive), value maps raw spelling -> canonical.
#: Distinct from admin aliases because the same raw string can legitimately
#: mean different things at different brokers (rare, but the S30/S31 spec
#: calls for per-broker scoping rather than one global flat namespace).
BROKER_PROFILES: dict[str, dict[str, str]] = {
    "oanda": {
        "US30_USD": "US30", "DE30_EUR": "GER40", "SPX500_USD": "SPX500", "NAS100_USD": "NAS100",
    },
    "ig": {
        "WALL STREET": "US30", "GERMANY 40": "GER40", "US 500": "SPX500", "US TECH 100": "NAS100",
    },
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
    def __init__(self, symbols: dict[str, str] | None = None, *, store: Any | None = None) -> None:
        """``store``, if given, is a ``SymbolStore`` (see ``symbol_store.py``)
        used to persist admin aliases/enable-state and normalization events
        across process restarts. Optional so the normalizer stays usable as a
        pure in-memory object (tests, offline tooling) without a database."""
        self._symbols: dict[str, str] = dict(symbols if symbols is not None else DEFAULT_SYMBOLS)
        self._enabled: dict[str, bool] = {s: True for s in self._symbols}
        self._admin_aliases: dict[str, str] = {}
        self._learned_aliases: dict[str, tuple[str, float]] = {}  # raw -> (canonical, learned_at)
        self._store = store
        if self._store is not None:
            self._load_from_store()

    def _load_from_store(self) -> None:
        for alias in self._store.list_aliases():
            if alias.canonical in self._symbols:
                self._admin_aliases[alias.raw.upper()] = alias.canonical
        for canonical, enabled in self._store.get_state().items():
            if canonical in self._symbols:
                self._enabled[canonical] = enabled

    # ── Admin symbol management ─────────────────────────────────────────────

    def add_symbol(self, canonical: str, asset_class: str = "forex", enabled: bool = True) -> None:
        canonical = canonical.upper()
        self._symbols[canonical] = asset_class
        self._enabled[canonical] = enabled

    def enable_symbol(self, canonical: str) -> None:
        canonical = canonical.upper()
        self._enabled[canonical] = True
        if self._store is not None:
            self._store.set_enabled(canonical, True)
        logger.info(f"SymbolNormalizer: enabled {canonical!r}")

    def disable_symbol(self, canonical: str) -> None:
        canonical = canonical.upper()
        self._enabled[canonical] = False
        if self._store is not None:
            self._store.set_enabled(canonical, False)
        logger.info(f"SymbolNormalizer: disabled {canonical!r}")

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
        if self._store is not None:
            self._store.save_alias(raw.upper(), canonical, source="admin")
        logger.info(f"SymbolNormalizer: added alias {raw!r} -> {canonical!r}")

    # ``add_symbol_alias`` is the S33 admin-API name for the same operation
    # as ``add_alias`` — kept as an alias method (not a duplicate
    # implementation) so both the original API and the new admin endpoint
    # naming resolve to one code path.
    add_symbol_alias = add_alias

    def remove_alias(self, raw: str) -> None:
        self._admin_aliases.pop(raw.upper(), None)
        if self._store is not None:
            self._store.remove_alias(raw.upper())
        logger.info(f"SymbolNormalizer: removed alias {raw!r}")

    def list_symbol_aliases(self, canonical: str | None = None) -> list[dict[str, Any]]:
        """All admin-registered aliases (optionally filtered to one canonical
        symbol), for the admin API / diagnostics surface."""
        canonical_u = canonical.upper() if canonical is not None else None
        return [
            {"raw": raw, "canonical": can, "source": "admin"}
            for raw, can in sorted(self._admin_aliases.items())
            if canonical_u is None or can == canonical_u
        ]

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

    def normalize(self, raw: str, *, broker: str | None = None, fuzzy: bool = False) -> NormalizationResult:
        result = self._resolve(raw, broker=broker, fuzzy=fuzzy)
        # Every normalization event is logged — both to the process logger
        # (S33 requirement: "logging of normalization events") and, if a
        # store is attached, persisted for the admin diagnostics surface.
        if result.canonical is None:
            logger.warning(
                f"SymbolNormalizer: could not resolve raw symbol {raw!r} (broker={broker!r})"
            )
        else:
            logger.debug(
                f"SymbolNormalizer: resolved {raw!r} -> {result.canonical!r} "
                f"via {result.matched_via!r} (broker={broker!r}, enabled={result.enabled})"
            )
        if self._store is not None:
            self._store.log_event(raw, result.canonical, result.matched_via, result.enabled, broker=broker)
        return result

    def _resolve(self, raw: str, *, broker: str | None = None, fuzzy: bool = False) -> NormalizationResult:
        if not raw:
            return NormalizationResult(raw, None, "unresolved", enabled=False)
        upper = raw.upper()

        if broker is not None:
            profile = BROKER_PROFILES.get(broker.lower(), {})
            if upper in profile:
                canonical = profile[upper]
                return NormalizationResult(raw, canonical, "broker_profile", self.is_enabled(canonical))

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

        if fuzzy:
            match = self._fuzzy_match(stripped)
            if match is not None:
                return NormalizationResult(raw, match, "fuzzy", self.is_enabled(match))

        return NormalizationResult(raw, None, "unresolved", enabled=False)

    def _fuzzy_match(self, stripped: str, cutoff: float = 0.8) -> str | None:
        """Last-resort approximate match against canonical symbols and all
        known alias spellings, for minor typos/vendor drift the structural
        stripper doesn't cover (e.g. "EURSUD" for "EURUSD"). Deliberately
        applied only when the caller opts in (``fuzzy=True``) since it can
        produce false positives for genuinely unknown instruments."""
        import difflib

        candidates = set(self._symbols) | set(BUILTIN_ALIASES) | set(self._admin_aliases) | set(self._learned_aliases)
        best = difflib.get_close_matches(stripped, candidates, n=1, cutoff=cutoff)
        if not best:
            return None
        matched = best[0]
        if matched in self._symbols:
            return matched
        if matched in BUILTIN_ALIASES:
            return BUILTIN_ALIASES[matched]
        if matched in self._admin_aliases:
            return self._admin_aliases[matched]
        if matched in self._learned_aliases:
            return self._learned_aliases[matched][0]
        return None

    def inverse_lookup(self, canonical: str) -> list[str]:
        """All known raw spellings (builtin/broker/admin/learned aliases,
        plus the canonical symbol itself) that resolve to ``canonical``."""
        canonical = canonical.upper()
        spellings: set[str] = set()
        if canonical in self._symbols:
            spellings.add(canonical)
        spellings.update(k for k, v in BUILTIN_ALIASES.items() if v == canonical)
        spellings.update(k for k, v in self._admin_aliases.items() if v == canonical)
        spellings.update(k for k, (v, _) in self._learned_aliases.items() if v == canonical)
        for profile in BROKER_PROFILES.values():
            spellings.update(k for k, v in profile.items() if v == canonical)
        return sorted(spellings)

    def resolve(self, raw: str, *, broker: str | None = None, fuzzy: bool = False) -> str:
        """Convenience wrapper — returns the canonical symbol or raises."""
        result = self.normalize(raw, broker=broker, fuzzy=fuzzy)
        if result.canonical is None:
            raise UnknownSymbolError(f"Could not normalize symbol {raw!r}")
        if not result.enabled:
            raise UnknownSymbolError(f"Symbol {result.canonical!r} (from {raw!r}) is disabled")
        return result.canonical
