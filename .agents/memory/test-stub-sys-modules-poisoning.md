---
name: Test-file sys.modules stubs poison later tests in the same session
description: Test files that unconditionally replace real packages (trainer, trainer.research, torch) in sys.modules with fakes/stubs corrupt imports for every subsequent test in the same pytest run, causing unrelated collection errors elsewhere.
---

Several test files (`tests/unit/test_cert.py`, `tests/unit/test_learning_loop.py`, and `conftest.py` itself) unconditionally injected fake modules into `sys.modules` (a bare `types.ModuleType("trainer")` with no `__path__`, or a torch stub) assuming an environment condition (torch broken) that was no longer true. Because `sys.modules` is process-global, these stubs leaked into every test that ran afterward in the same session, producing confusing `ImportError`s in seemingly unrelated files (e.g. `test_research.py` failing only when run as part of the full suite, not in isolation).

**Why:** these stubs were written defensively for a genuinely-broken-torch environment, but were never made conditional, so once torch/the interpreter was fixed they became the bug instead of the fix.

**How to apply:** any test-file-level `sys.modules[...] = <fake>` assignment must be gated behind an actual runtime check (e.g. try `torch.tensor([0])` and catch), never assumed unconditionally. When a test only fails in full-suite runs but passes in isolation, suspect global `sys.modules` mutation by an earlier-running test file over an actual logic bug.
