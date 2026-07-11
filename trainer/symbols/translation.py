"""Trainer-wide canonical symbol translation layer.

The trainer **never** consumes broker symbols.  Every instrument string that
enters memory, datasets, labels, decisions, or research records must be a
canonical symbol (e.g. ``EURUSD``, ``XAUUSD``, ``US30``) — never a
broker-decorated variant (``EURUSD.r``, ``GOLD``, ``EUR/USD``, ``EURUSDm``).

This module is the single, import-friendly facade over
``trainer.market_data.symbol_normalizer.SymbolNormalizer`` that all
non-market-data packages use for canonical enforcement.  It deliberately has
no heavy dependencies of its own — ``SymbolNormalizer`` is imported lazily
inside the singleton factory so that lightweight consumers (e.g. test helpers,
label generators) do not pay the full market-data import cost at module load.

Three entry-point operations
-----------------------------
``translate(raw, broker=None) → str``
    Normalise any raw/broker spelling to its canonical form, or raise
    ``SymbolTranslationError`` when the symbol is completely unknown.

``is_canonical(symbol) → bool``
    ``True`` only when the symbol is an *exact* match for a registered
    canonical symbol.  Does **not** accept broker aliases —
    ``"EURUSD.r"`` is NOT canonical even though it translates to ``"EURUSD"``.

``assert_canonical(symbol) → None``
    Raise ``SymbolTranslationError`` when the symbol is not canonical.
    Call at every boundary (memory writes, label creation, research experiment
    creation, protocol commands) that must never receive broker symbols.

``batch_translate(symbols, broker=None) → dict[str, str]``
    Translate a collection of raw symbols; raises on the first failure.

Process-level singleton
-----------------------
A default ``SymbolTranslationLayer`` is constructed lazily the first time any
convenience function is called.  Override with ``configure_translator()``
before consumers run — e.g. in test fixtures or server boot code — to inject
a custom normalizer (additional aliases, restricted symbol set, etc.).

Design constraints
------------------
* No import from the inference, execution, or protocol packages.
* ``SymbolNormalizer`` is imported lazily (inside the singleton factory) to
  keep this module importable even in environments where ``trainer.market_data``
  has not been fully initialised yet.
* Thread-safe singleton via ``threading.Lock``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

if TYPE_CHECKING:
    from trainer.market_data.symbol_normalizer import SymbolNormalizer


# ─────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────


class SymbolTranslationError(ValueError):
    """Raised when a broker symbol cannot be resolved to canonical form,
    or when a non-canonical symbol is presented at a boundary that must
    only accept canonical symbols.

    Inherits from ``ValueError`` so callers that catch ``ValueError``
    (common in dataclass ``__post_init__`` validators) naturally handle it.
    """


# ─────────────────────────────────────────────────────────────────────────
# SymbolTranslationLayer
# ─────────────────────────────────────────────────────────────────────────


class SymbolTranslationLayer:
    """Thin facade over ``SymbolNormalizer`` for cross-package canonical enforcement.

    Construct via ``SymbolTranslationLayer(normalizer=...)`` to inject a
    custom normalizer (e.g. with extra aliases for tests), or call
    ``get_translator()`` to get the shared process-level singleton.
    """

    def __init__(self, normalizer: "SymbolNormalizer | None" = None) -> None:
        if normalizer is None:
            from trainer.market_data.symbol_normalizer import SymbolNormalizer
            normalizer = SymbolNormalizer()
        self._normalizer = normalizer

    # ── Core operations ──────────────────────────────────────────────────────

    def translate(self, raw: str, *, broker: str | None = None) -> str:
        """Translate any raw/broker symbol to its canonical form.

        Raises ``SymbolTranslationError`` if the symbol is completely unknown.
        This is the appropriate call when the input *might* be a broker symbol
        and must be normalised before further use.
        """
        try:
            return self._normalizer.resolve(raw, broker=broker)
        except Exception as exc:
            raise SymbolTranslationError(
                f"Cannot translate {raw!r} to a canonical symbol"
                + (f" (broker={broker!r})" if broker else "")
                + f": {exc}"
            ) from exc

    def is_canonical(self, symbol: str) -> bool:
        """``True`` iff *symbol* is already a known registered canonical symbol.

        Does **not** accept broker aliases — ``"EURUSD.r"`` is NOT canonical
        even though it can be translated to ``"EURUSD"``.
        """
        return self._normalizer.is_canonical_symbol(symbol)

    def assert_canonical(self, symbol: str) -> None:
        """Raise ``SymbolTranslationError`` if *symbol* is not canonical.

        Call at every boundary (memory writes, label creation, research
        experiment creation, protocol commands) that must never accept
        broker-decorated symbols.

        Example::

            from trainer.symbols import assert_canonical
            assert_canonical(instrument)   # raises if e.g. "EURUSD.r"
        """
        if not self.is_canonical(symbol):
            known = self._normalizer.canonical_symbols()[:8]
            raise SymbolTranslationError(
                f"Non-canonical symbol {symbol!r} at a canonical-only boundary. "
                f"Translate first: trainer.symbols.translate({symbol!r}). "
                f"Known canonicals (first 8): {known}"
            )

    def batch_translate(
        self, symbols: list[str], *, broker: str | None = None
    ) -> dict[str, str]:
        """Translate a list of raw symbols, returning ``{raw: canonical}``.

        Raises ``SymbolTranslationError`` on the first symbol that cannot be
        resolved.  Call ``is_canonical(s)`` first to skip already-canonical
        symbols.
        """
        result: dict[str, str] = {}
        for raw in symbols:
            result[raw] = self.translate(raw, broker=broker)
        return result

    def batch_assert_canonical(self, symbols: list[str]) -> None:
        """Assert that every symbol in *symbols* is canonical.

        Raises ``SymbolTranslationError`` listing all non-canonical entries.
        """
        bad = [s for s in symbols if not self.is_canonical(s)]
        if bad:
            raise SymbolTranslationError(
                f"Non-canonical symbol(s) at a canonical-only boundary: {bad}. "
                f"Translate via trainer.symbols.batch_translate() first."
            )

    def register_broker_alias(self, raw: str, canonical: str) -> None:
        """Teach the layer a new broker-specific alias at runtime.

        Equivalent to ``SymbolNormalizer.add_alias()`` but exposed through
        this facade so callers stay decoupled from ``symbol_normalizer``.
        """
        self._normalizer.add_alias(raw, canonical)

    def known_canonicals(self) -> list[str]:
        """Sorted list of all registered canonical symbols."""
        return self._normalizer.canonical_symbols()


# ─────────────────────────────────────────────────────────────────────────
# Process-level singleton
# ─────────────────────────────────────────────────────────────────────────

_lock: threading.Lock = threading.Lock()
_default: SymbolTranslationLayer | None = None


def get_translator() -> SymbolTranslationLayer:
    """Return the process-level singleton ``SymbolTranslationLayer``.

    Constructed lazily on first call.  Thread-safe.
    """
    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = SymbolTranslationLayer()
    return _default


def configure_translator(
    normalizer: "SymbolNormalizer | None" = None,
    *,
    layer: SymbolTranslationLayer | None = None,
) -> None:
    """Override the process-level singleton.

    Call before any consumer touches the translator — e.g. in test fixtures
    or server start-up code — to inject a custom ``SymbolNormalizer`` (extra
    aliases, restricted symbol set) or a fully custom ``SymbolTranslationLayer``
    (e.g. a permissive pass-through for legacy-data migration scripts).

    Supply exactly one of *normalizer* or *layer*.
    """
    global _default
    if layer is not None and normalizer is not None:
        raise ValueError("Supply normalizer OR layer, not both.")
    with _lock:
        if layer is not None:
            _default = layer
        else:
            _default = SymbolTranslationLayer(normalizer)


def reset_translator() -> None:
    """Reset the singleton so the next call rebuilds from defaults.

    Primarily useful in test teardown to avoid cross-test contamination.
    """
    global _default
    with _lock:
        _default = None


# ─────────────────────────────────────────────────────────────────────────
# Module-level convenience wrappers (delegate to the singleton)
# ─────────────────────────────────────────────────────────────────────────


def translate(raw: str, *, broker: str | None = None) -> str:
    """Translate any raw/broker symbol to its canonical form.

    Raises ``SymbolTranslationError`` if the symbol is completely unknown.
    """
    return get_translator().translate(raw, broker=broker)


def is_canonical(symbol: str) -> bool:
    """``True`` iff *symbol* is already a registered canonical symbol."""
    return get_translator().is_canonical(symbol)


def assert_canonical(symbol: str) -> None:
    """Raise ``SymbolTranslationError`` if *symbol* is not canonical.

    This is the primary enforcement call — add it at every boundary that
    must not accept broker symbols.
    """
    get_translator().assert_canonical(symbol)


def batch_translate(symbols: list[str], *, broker: str | None = None) -> dict[str, str]:
    """Translate a list of raw symbols to their canonical forms.

    Returns ``{raw: canonical}``; raises on the first failure.
    """
    return get_translator().batch_translate(symbols, broker=broker)


def batch_assert_canonical(symbols: list[str]) -> None:
    """Assert that every symbol in *symbols* is canonical.

    Raises ``SymbolTranslationError`` listing all non-canonical entries.
    """
    get_translator().batch_assert_canonical(symbols)


__all__ = [
    # Exception
    "SymbolTranslationError",
    # Class
    "SymbolTranslationLayer",
    # Singleton management
    "get_translator",
    "configure_translator",
    "reset_translator",
    # Convenience wrappers
    "translate",
    "is_canonical",
    "assert_canonical",
    "batch_translate",
    "batch_assert_canonical",
]
