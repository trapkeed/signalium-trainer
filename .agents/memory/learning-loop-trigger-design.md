---
name: Learning loop trigger design conventions
description: Conventions for adding new autonomous trigger-detection events to trainer/learning_loop/actuators.py
---

New trigger events that have no dedicated observation module (e.g. slippage, scaling failures, capital inefficiency) get a lightweight input dataclass defined directly in `actuators.py`, not a new module — the actuator is the contract, the data collection is left to whatever calls it (or the scheduler's `collector` callable).

**Why:** these three had no existing tracker in the codebase, and inventing full observation modules would have been out of scope / speculative. The dataclasses just pin down the shape the actuator expects.

**How to apply:** when adding a new trigger kind, (1) add an input dataclass near the other report dataclasses, (2) add class-level threshold constants documented inline, (3) add an `act_on_<name>` method following the existing `_apply`/`record_failure`/`create_search` pattern, (4) wire it into `run_cycle` as an optional kwarg, (5) require a minimum sample size before firing to avoid reacting to noise.

`ActionType.SHADOW_CANDIDATE` (`ResearchLabAPI.create_shadow_test`) is reserved for validating a *corrected* policy against a failure finding (e.g. scaling policy misfiring) — distinct from `PARAMETER_SEARCH`, which explores a space without a specific candidate fix in mind. Keep this distinction when adding new failure-driven triggers.

`LearningLoopScheduler` (`trainer/learning_loop/scheduler.py`) is a plain `threading.Thread` loop that calls `run_cycle` on an interval via a caller-supplied `collector()` callable; it never computes reports itself and never gains promotion authority — it only automates *when* `run_cycle` runs, not *what* it's allowed to do.
