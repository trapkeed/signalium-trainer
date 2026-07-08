---
name: ResearchLabAPI public/internal signature drift
description: Public wrapper methods in research_lab_api.py can silently pass kwargs that don't match their underlying rl_models dataclass .new() constructors
---

`ResearchLabAPI` public methods (e.g. `record_feature`, `record_failure`, `create_search`)
are thin wrappers that call a dataclass's `.new(...)` classmethod in `rl_models.py`. These
two layers can drift out of sync — a public method's kwargs (name, types) do not
necessarily match what the dataclass constructor actually accepts, and nothing catches
this until the call raises `TypeError` at runtime (no static check, no test coverage
exercised the mismatched paths).

**Why:** Found three such mismatches while wiring a new consumer (learning-loop actuators)
that called these public methods directly — `record_failure` passed `detected_at`/
`affected_instruments`/`loss_pct` that `FailureEvent.new()` doesn't accept; `create_search`
passed `parameter_space`/`max_trials` instead of the real `search_space`/`n_trials`/
`objective`/`maximize`. These were pre-existing latent bugs, unexercised by prior tests.

**How to apply:** Before relying on any `ResearchLabAPI` method signature (or any similar
public-wrapper/dataclass-`.new()` pattern elsewhere in this codebase), verify the actual
dataclass constructor in `rl_models.py` directly — don't trust the wrapping method's
param names/docstring at face value.
