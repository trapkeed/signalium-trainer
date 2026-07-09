---
name: Broker-aware market data routing design
description: How MarketDataRouter/ProviderPriority/SymbolStore compose with pre-existing ProviderManager and SymbolNormalizer — read before touching trainer/market_data/
---

`trainer/market_data/` layers routing on top of two pre-existing, reusable pieces rather than reimplementing them:

- `ProviderManager` (S30) already does health-based failover for a fixed ordered provider list. `MarketDataRouter` builds one internal `ProviderManager` per routing context ("training"/"manual"/"autonomous") and always calls it with `DataContext.LIVE` — `ProviderManager`'s own LIVE/TRAINING split is orthogonal and unused by the router.
- `SymbolNormalizer` (S30) is pure in-memory by design. Persistence (admin aliases, enable/disable state, normalization event log) is bolted on via an optional injected `SymbolStore` (own SQLite DB, `SYMBOL_STORE_DB_PATH` env var) rather than making the normalizer itself DB-aware — keeps it usable standalone in tests/tooling.

**Why:** avoids duplicating failover/matching logic that already had test coverage, and keeps market-data infra independent of both `trainer.qde` and `trainer.research_lab` (neither owns it, both may use it) — same one-way-dependency discipline as the QDE/Research-Lab boundary.

**How to apply:** training must only ever request canonical symbols (`MarketDataRouter.route(..., context="training")` raises `SymbolNotCanonicalError` otherwise — no silent normalization for training data). Live inference always auto-normalizes via `SymbolNormalizer.normalize()`, which unconditionally logs every event (both to the logger and, if a store is attached, persisted) — this is the single choke point for "logging of normalization events", don't add a second one. Provider priority defaults: MANUAL → OANDA, TIINGO; AUTONOMOUS → EA_FEED, OANDA, TIINGO (`trainer/market_data/provider_priority.py`). A missing provider (e.g. no EA feed configured) is skipped when building the priority list, not an error — routing degrades gracefully.
