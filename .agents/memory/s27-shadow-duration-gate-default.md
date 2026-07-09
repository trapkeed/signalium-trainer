---
name: Shadow test minimum-duration gate defaults to disabled
description: ShadowTestRun.min_duration_seconds (S27 institutional gate) must default to 0, not a real duration, or it silently blocks PROMOTE in every existing volume-only shadow test.
---

`ShadowTestRun.min_duration_seconds` (and the `new()`/`create_shadow_test()` defaults) must default to `0.0`.

**Why:** it was first added defaulting to 7 days "for institutional rigor," which broke ~2 existing PROMOTE-path tests that conclude a shadow test immediately after creating it with high agreement/score. The duration gate is opt-in — callers that want the institutional 7-day minimum must pass `min_duration_seconds` explicitly (e.g. from an institutional-certification workflow), rather than it being a silent global default.

**How to apply:** when adding new mandatory-gate fields to existing lifecycle dataclasses (shadow tests, promotions, etc.), default them to "off"/permissive unless the spec explicitly says the gate must always apply — then confirm against the full existing test suite for that module before treating the change as done.
