---
name: QDE behavioral personality vs. coarse SymbolPersonality
description: Two distinct per-symbol "personality" models coexist on BrainState — know which one to extend.
---

`BrainState` carries two separate, non-duplicating personality models:

- `personality` (`brain_models.SymbolPersonality`) — free-form trait bag keyed by
  arbitrary tags + `regime_outcome_quality`, updated by
  `long_term_memory.LongTermMemory.record_experience` from coarse hypothesis/regime
  `ExperienceRecord`s. Answers "has this symbol's hypothesis-tracking been reliable".
- `behavioral_profile` (`personality_profiles.PersonalityProfile`) — fixed, named
  0-1 dimensions (trend_persistence, avg_pullback_depth, volatility_clustering,
  liquidity_sweep_frequency, news_sensitivity, reversal_tendency, session_affinity),
  updated continuously via `behavioral_memory.BehavioralMemory.observe` from raw
  `BehavioralObservation`s (trend runs, pullbacks, vol/liquidity microstructure,
  session outcomes, news reactions, reversals). Answers "how does this symbol
  actually move".

**Why:** the coarse trait bag existed first but only reflects hypothesis-tracking
outcomes, not raw market microstructure behavior — a genuinely different signal
source, so folding it into the same model would conflate two unrelated things.

**How to apply:** new market-behavior learning belongs in `PersonalityProfile`/
`BehavioralMemory`, not `SymbolPersonality`. Both are reconstructed independently
in `brain._state_from_dict` and serialized independently in `BrainState.to_dict`.

`symbol_personality_engine.SymbolPersonalityEngine` is the orchestration facade
(same pattern as `HypothesisEngine`/`SetupQueue`): constructed externally by a
caller holding `QDEBrain`, not auto-wired into `on_market_update`. Its influence
methods (`adjust_confidence`, `risk_appetite_multiplier`, `setup_thresholds_for`,
`expected_holding_duration`) are consult-only seams — a caller must explicitly
call them; profile influence always blends via `profile.trust()` (ramps in with
`sample_count`, neutral/no-op below full trust) so a fresh symbol never biases
decisions before it's actually been observed.
