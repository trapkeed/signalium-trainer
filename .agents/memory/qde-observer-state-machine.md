---
name: QDE always-on observer state machine
description: How the per-symbol market-observation lifecycle (OBSERVING..COOLDOWN) is layered onto QDE without becoming a rule engine
---

The observer state machine (`trainer/qde/qde_observer*.py`) is context/memory
for the existing decision models, not a new decision authority. No code path
anywhere derives an entry/exit order directly from `ObserverState` — it only
ever updates a label plus advisory telemetry that other components may read.

**Why:** Phase S7 explicitly required QDE to "continuously monitor" every
symbol (including ones with no open position) without duplicating or
overriding the real entry/exit/position-management decision logic that
already exists in `qde_inference.py` / `qde_position_manager.py`. Blurring
that line would create two competing sources of truth for whether a trade
should happen.

**How to apply:** When adding new observer-driven behavior, always ask "does
this *place or block* a trade" vs "does this just *label/inform*". Anti-churn
goals (overtrading, revenge trading, late entries, low-quality entries) are
enforced via guard trips (`OVERTRADE_GUARD`, `REVENGE_TRADE_GUARD`) and
telemetry visibility (`ObserverMonitor`/`ObserverDashboard`), not via the
observer vetoing or forcing trader actions. Persistence follows the
`rl_database.py` SQLite-WAL pattern (current-snapshot table + append-only
transitions table) so state survives restarts; API layer rehydrates from DB
on init, same as `ResearchLabAPI`.
