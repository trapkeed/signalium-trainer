---
name: Projection package design
description: trainer/qde/brain/projection/ — conditions→outcome learning, five forward estimates, four integration signals.
---

## Rule
`trainer/qde/brain/projection/` learns `(ProjectionConditions → ProjectionOutcome)` pairs as trades complete and produces five forward estimates per opportunity: expected move (ATR), expected duration (seconds), expected volatility, expected reward (ATR), expected MAE (ATR).

No hardcoded pip targets anywhere. "Sufficient reward" is expressed as a percentile rank within the instrument's own historical distribution.

## Key types
- `ProjectionConditions` — 9-dim fingerprint: `[atr_pct, spread_atr_ratio, volatility, momentum, vol_state_rank_norm, volatility_quality, microstructure_quality, session_quality, trend_valid_f]`. Build via `from_observation(instrument, inp)`.
- `ProjectionOutcome` — what actually happened in ATR multiples: `reward_atr`, `mae_atr`, `mfe_atr`, `duration_seconds`, `realized_vol`. Bridge via `from_outcome_path(path, atr_at_entry)`.
- `ProjectionMemory` — bounded per-symbol rolling store; `to_snapshot`/`load_snapshot` persistence.
- `DurationProjection` — expected duration + volatility from `ExpectedDurationEngine` (stateless).
- `ExpectedProjection` — all five estimates + four integration signals on one frozen dataclass.
- `ExpectedMoveEngine` — top-level orchestrator; holds `ProjectionMemory` + `ExpectedDurationEngine`.

## Four integration points on ExpectedMoveEngine
| Method | Output | Feeds |
|---|---|---|
| `context_for_inference(instrument, conditions, direction)` | 16 `proj_*` keys | brain_context |
| `opportunity_rank_signal(...)` | 0-1 | trade selection / queue ordering |
| `position_size_modifier(...)` | 0.5..2.0 | position sizing |
| `risk_appetite_modifier(...)` | −0.50..+0.25 | risk appetite |

## Learning loop
`record_outcome(instrument, direction, conditions, outcome)` — one call per completed trade.
Conditions should be the snapshot captured **at opportunity time**, not exit conditions.

## Similarity algorithm (same as regime/liquidity packages)
Cosine similarity on 9-dim vector × recency weight `exp(-age_days / τ)`, τ=14 days.
Top-K weighted mean for each metric. Direction weighting: same-direction records get 1.5× weight, opposite get 0.5×.

## Cache key bug (fixed)
`ExpectedMoveEngine._cache` is keyed by `(instrument, direction, tuple(conditions.as_vector()))` NOT just instrument. Caching by instrument alone causes different conditions to return the same cached projection.

## reward_sufficiency_score formula
```
sufficiency = reward_percentile * confidence + 0.5 * (1 - confidence)
```
Regresses toward 0.5 (neutral) when confidence is low. No absolute floor.

## How to apply
Wire `record_outcome` into the post-trade feedback loop (when a trade closes, build `ProjectionOutcome.from_outcome_path()` and call it). Wire `context_for_inference` alongside `regime_*` and `liq_*` keys into `brain_context` per tick.
