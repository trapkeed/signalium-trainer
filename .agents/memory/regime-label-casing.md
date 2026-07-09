---
name: Regime label casing inconsistency
description: Live regime telemetry mixes casing conventions; comparisons against a known-regime vocabulary or histograms must normalize case.
---

Regime label strings are not consistently cased across the codebase. Some
producers/tests use uppercase (`"RANGING"`, `"TRENDING_BULL"`, `"CRISIS"`),
while an enum-derived vocabulary (`MarketRegime.value`) uses lowercase
snake_case. Any code that compares a live regime label against a known
vocabulary, or buckets regimes into a histogram for drift/distribution-shift
detection, must lowercase (or otherwise canonicalize) both sides first.

**Why:** a mismatch here doesn't error — it silently produces false
positives (a real regime read as "unseen") or manufactures apparent
distribution shift purely from casing, with no functional signal behind it.
This was caught by a "healthy model, no alerts" regression test firing
unexpectedly.

**How to apply:** whenever you write new regime-label comparison/bucketing
logic, normalize case at the point of comparison — do not assume upstream
telemetry or a shared enum's `.value` will agree on casing.
