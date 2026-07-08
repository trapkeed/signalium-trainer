---
name: icari AllocationRequest pre-existing test bug
description: 21 failures in tests/unit/test_icari.py are pre-existing and unrelated to safety-system/QDE work — do not try to fix them as part of unrelated tasks.
---

`tests/unit/test_icari.py::TestAllocationEngine`, part of `TestPositionSizingEngine`, and `TestIcariAPI` (21 tests total) fail with `AttributeError: 'AllocationRequest' object has no attribute 'ensemble_confidence'`, raised from `trainer/icari/position_sizing_engine.py:146`.

**Why this matters:** confirmed via `git diff --stat` that no safety-system (drawdown/broker/exposure/shadow/rollback) or QDE-output-schema work touches `position_sizing_engine.py`, `icari_models.py`, or `allocation_engine.py` — this is a standing, unrelated bug (likely a missing field on the `AllocationRequest` dataclass vs. what the sizing engine expects).

**How to apply:** when running `tests/unit/test_icari.py` as a regression check for unrelated work, expect these 21 failures and don't block on them. If asked to fix ICARI allocation/sizing specifically, this is the root cause to start from.
