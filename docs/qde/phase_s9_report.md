# Phase S9 — Hierarchical Timeframe Understanding

## Goal

Before Phase S9, QDE reasoned about a symbol's own tick-level state (S7),
open positions (S3-S6), and the account (S8) — but every input was
effectively single-timeframe. S9 gives QDE simultaneous context across all
six timeframes it trades on — **D1, H4, H1, M15, M5, M1** (M1 is the
execution timeframe) — so it can understand higher-timeframe trend,
intermediate structure, and lower-timeframe timing *together*, and decide
for itself:

- whether cross-timeframe alignment matters right now,
- whether a countertrend opportunity exists,
- whether a continuation opportunity exists,
- whether reversal is more or less likely than continuation.

**No hard-coded cross-timeframe rules.** There is no "H1 trend filter", no
"trade only with the higher-timeframe trend", no "M15 confirmation" gate
anywhere in this phase. Every cross-timeframe relationship is expressed as
a *trainable parameter* (attention weights, learned per-timeframe identity
embeddings, learned output heads) — the only fixed things are the six
timeframe names/durations and the shape of the feature contract, which are
domain facts, not trading rules.

## Scope of this phase

Exactly the four deliverables requested: timeframe feature contracts, a
synchronization pipeline, a temporal feature store, and a temporal
attention architecture. No wiring into the live QDE decision engine or
FastAPI layer — this phase is the data + architecture layer, mirroring how
S7's `qde_observer.py` and S8's `qde_account_pipeline.py` landed before
their respective API integrations.

## Modules

| Module | Purpose |
|---|---|
| `qde_timeframe_types.py` | `Timeframe` enum (D1, H4, H1, M15, M5, M1), `TIMEFRAME_SECONDS`, `EXECUTION_TIMEFRAME = M1`, coarsest→finest ordering. Purely factual metadata — no trend/filter/priority semantics. |
| `qde_timeframe_models.py` | `TimeframeBar` (OHLCV schema), `TimeframeFeatureVector` (the per-timeframe **feature contract**: returns, vol-normalized momentum, range position, swing distances, candle shape, volume z-score — observational only, no derived trend labels), `MultiTimeframeSnapshot` (six feature vectors aligned to one M1 timestamp — the contract boundary handed to the model), `TemporalAssessment` (model **output**: alignment/countertrend/continuation scores + reversal-vs-continuation probabilities + attention weights for interpretability). |
| `qde_timeframe_sync.py` | `TimeframeSynchronizer` — the **synchronization pipeline**. Pure "as-of" alignment: for any M1 timestamp, selects only bars with `close_time <= as_of` per timeframe (no lookahead) and computes each `TimeframeFeatureVector`. Contains no cross-timeframe conditionals. |
| `qde_timeframe_store.py` | `TemporalFeatureStore` — the **temporal feature store**. SQLite (WAL), mirrors `qde_observer_database.py`/`qde_account_database.py`: `timeframe_bars` (append-only, source of truth for the synchronizer) + `temporal_assessments` (append-only, for replay/telemetry/training export). `build_snapshot()` rehydrates a `MultiTimeframeSnapshot` as-of any timestamp purely from stored bars. |
| `qde_temporal_attention.py` | The **temporal attention architecture** (torch `nn.Module`s): per-timeframe LSTM encoders (independent weights) → learned per-timeframe identity embedding → `nn.MultiheadAttention` self-attention across the 6 timeframes → four learned heads (alignment, countertrend opportunity, continuation opportunity, reversal-vs-continuation 2-way softmax). All relationships are learned; attention weights are exposed only for interpretability/telemetry, never fed back into a rule. |

## Why no field encodes "trend" or "structure" directly

`TimeframeFeatureVector` intentionally stops at *descriptive* quantities
(log return, vol-normalized momentum, position within recent range, swing
distances, candle shape, volume z-score). It never emits a field like
`trend = "BULLISH"` or `structure = "PULLBACK"` — those are exactly the
interpretive judgments the phase says must be *learned*, not computed by a
formula and then handed to the model as a pre-baked label. The line drawn
in this phase: **data plumbing describes what happened; the attention
model decides what it means.**

## Synchronization pipeline: no-lookahead guarantee

`TimeframeSynchronizer.align(instrument, as_of, bars_by_timeframe)` filters
every timeframe's bar history to `close_time <= as_of` before computing any
feature — verified with a functional test asserting that no timeframe's
computed `as_of` ever exceeds the query cutoff, including when the D1/H4/H1
bars close much less frequently than the M1 tick driving the query.

## Temporal attention architecture: design choices

- **Independent per-timeframe encoders** rather than one shared encoder —
  each timeframe's bar dynamics (e.g. D1 daily-close noise vs. M1
  tick-level noise) differ enough to warrant separate weights, consistent
  with the existing `MultiTimeframeStream` pattern in
  `trainer/models/quant_v1/hybrid/streams.py`.
- **Learned timeframe identity embedding** (not positional encoding by
  fixed formula) — lets attention distinguish "this is the D1 slot" from
  "this is the M1 slot" without asserting any priority between them.
- **Full self-attention across all 6 timeframes** (not a fixed
  higher-timeframe-gates-lower-timeframe structure) — every timeframe can
  attend to every other timeframe with a learned weight; the network can
  discover during training that, e.g., M5 momentum shifts matter more
  when D1 is in a strong trend, without that relationship being coded
  anywhere.
- **Four independent linear heads** off the fused latent, matching the
  existing multi-task head convention in `trainer/models/quant_v1/model.py`
  — alignment, countertrend opportunity, and continuation opportunity are
  independent sigmoids (they are not mutually exclusive: a countertrend
  opportunity and a continuation opportunity can both score low or, in
  principle, the model can learn cases where both are elevated), while
  reversal-vs-continuation is a proper 2-way softmax since those are
  complementary probabilities.

## Verification

- `py_compile` clean on all five new modules.
- Functional test (pure Python, no torch needed) built synthetic bar
  histories reproducing the phase's own worked example — D1 bullish drift,
  H1 pullback (negative drift), M5 momentum shift (positive drift) — and
  confirmed the feature contract's `log_return_1` sign matches each
  timeframe's synthetic direction, that no future bar ever leaks into an
  as-of query, and that the feature matrix shape matches
  `(num_timeframes, feature_dim)`.
- Temporal feature store round-trip test: wrote bars, rehydrated a
  `MultiTimeframeSnapshot` from a fresh `TemporalFeatureStore` instance
  after closing/reopening the SQLite file (simulating a process restart),
  and confirmed the rehydrated snapshot's features match the in-memory
  ones bit-for-bit, plus a `TemporalAssessment` log/read round trip.
- **Attention architecture (`qde_temporal_attention.py`) could not be
  executed** in this sandbox: `import torch` fails with `OSError:
  libtorch_global_deps.so: cannot open shared object file`, a pre-existing,
  environment-specific issue (also hit and documented in earlier phases —
  see `.agents/memory`), unrelated to this phase's code. The module was
  verified via `py_compile` and structural/shape review against the
  existing, working `trainer/models/quant_v1/hybrid/streams.py` patterns
  it mirrors (per-timeframe `nn.LSTM` encoders, `nn.MultiheadAttention`,
  linear heads off a shared latent) and is expected to run unmodified in
  an environment with a working torch install.

## What a future phase (S10?) still needs to do

- Wire `TemporalAttentionEncoder` into the live QDE decision path (feed it
  windowed sequences built from `TemporalFeatureStore.build_snapshot()`
  history, not just the latest snapshot) and persist its
  `TemporalAssessment` output via `TemporalFeatureStore.log_assessment()`.
  This phase deliberately stops at "the architecture and its data
  plumbing exist and are individually verified", not full pipeline
  integration.
- Actually train the attention weights on historical data once a training
  loop is wired up — right now every head is randomly initialized; "no
  hard-coded rules" is satisfied by construction, but the *learned*
  relationships don't exist until training happens.
