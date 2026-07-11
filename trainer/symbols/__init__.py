"""trainer.symbols — canonical symbol translation layer.

The trainer **never** consumes broker symbols.  This package is the
single, importable facade for canonical symbol enforcement across all
trainer packages (memory, datasets, labels, decisions, research, protocol).

Primary API
-----------
``translate(raw, broker=None) → str``
    Normalise any raw/broker spelling to its canonical form.

``is_canonical(symbol) → bool``
    True only when the symbol is an exact registered canonical.

``assert_canonical(symbol) → None``
    Raise ``SymbolTranslationError`` at any boundary that must not
    accept broker-decorated symbols.

``batch_translate(symbols, broker=None) → dict[str, str]``
    Translate a collection; raises on the first unresolvable symbol.

``batch_assert_canonical(symbols) → None``
    Assert every item in the list is canonical.

Singleton management
--------------------
``get_translator() → SymbolTranslationLayer``
    The process-level singleton (lazy, thread-safe).

``configure_translator(normalizer=None, *, layer=None) → None``
    Override the singleton before consumers run.

``reset_translator() → None``
    Reset to defaults (useful in test teardown).

See ``trainer.symbols.translation`` for full implementation details.
"""

from trainer.symbols.translation import (
    SymbolTranslationError,
    SymbolTranslationLayer,
    assert_canonical,
    batch_assert_canonical,
    batch_translate,
    configure_translator,
    get_translator,
    is_canonical,
    reset_translator,
    translate,
)

__all__ = [
    "SymbolTranslationError",
    "SymbolTranslationLayer",
    "translate",
    "is_canonical",
    "assert_canonical",
    "batch_translate",
    "batch_assert_canonical",
    "get_translator",
    "configure_translator",
    "reset_translator",
]
