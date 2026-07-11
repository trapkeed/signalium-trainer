---
name: Symbol Translation Layer
description: Design and boundary enforcement for canonical symbol requirement — trainer never consumes broker symbols.
---

## Rule
The trainer **never** consumes broker symbols. Every instrument string in memory, datasets, labels, decisions, research, and protocol commands must be a registered canonical symbol (e.g. `EURUSD`, `XAUUSD`, `US30`) — never a broker-decorated variant (`EURUSD.r`, `GOLD`, `EUR/USD`).

## What already existed (do not duplicate)
- `trainer/market_data/symbol_normalizer.py` — `SymbolNormalizer` with `normalize()`, `resolve()`, alias management, runtime learning
- `trainer/market_data/symbol_store.py` — SQLite persistence for aliases + state
- `trainer/market_data/market_data_router.py` — enforces canonical for training context, auto-normalizes for inference context

## What was added
- `trainer/symbols/translation.py` + `trainer/symbols/__init__.py` — thin, importable facade; wraps `SymbolNormalizer` as a lazy singleton
- Two new methods on `SymbolNormalizer`: `is_canonical_symbol(symbol)` (exact membership check, no alias resolution) and `canonical_symbols()` (sorted list)
- `SymbolTranslationError(ValueError)` — the boundary exception type

## Primary API (`from trainer.symbols import ...`)
- `translate(raw, broker=None) → str` — normalize broker → canonical, raises on unknown
- `is_canonical(symbol) → bool` — exact canonical check only (aliases return False)
- `assert_canonical(symbol) → None` — raises `SymbolTranslationError` if not canonical; the enforcement call
- `batch_translate(symbols, broker=None) → dict[str, str]`
- `batch_assert_canonical(symbols) → None`
- `configure_translator(normalizer=None, *, layer=None)` — inject custom normalizer (tests, server boot)
- `reset_translator()` — clear singleton (test teardown)

## Boundary guards (where assert_canonical is enforced)
| Boundary | Location |
|---|---|
| Labels / datasets | `TradeOutcomePath.__post_init__` in `trainer/training_signals/outcome_path.py` |
| Memory / decisions | `DecisionRecord.create()` in `trainer/memory/decision_models.py` |
| Protocol commands | `_check_envelope()` in `trainer/protocol/command_validator.py` (adds to `errors` list; soft import guard so protocol doesn't hard-depend on symbols) |
| Research records | `register_backtest()` in `trainer/research_lab/research_lab_api.py` (instruments list) |

## Key design decisions
**Why soft import in protocol validator?** Protocol is the lowest layer; a hard `import trainer.symbols` would create a cross-layer coupling. The `try/except ImportError` allows protocol to work without the full market_data stack.

**Why `is_canonical` ≠ `translate() == symbol`?** An alias like `"GOLD"` translates to `"XAUUSD"` successfully, but `"GOLD"` is NOT canonical. `is_canonical` checks the symbol IS a key in `_symbols`, not that it resolves to something.

**`SymbolTranslationError` inherits `ValueError`** so dataclass `__post_init__` callers that catch `ValueError` handle it naturally.

**Singleton is lazy + thread-safe** via `threading.Lock`; `SymbolNormalizer` (which imports nothing heavy) is constructed on first use only.
