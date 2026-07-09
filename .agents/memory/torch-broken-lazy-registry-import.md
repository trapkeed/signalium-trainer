---
name: torch broken in this env (STALE — see update); trainer.registry must stay lazy about ModelRegistry
description: An earlier session found torch's shared lib missing; a later session installed Python 3.12 fresh and found real torch (2.13) works fine. The lazy-import pattern in trainer/registry is still good practice regardless — it protects isolated test harnesses from accidental torch imports.
---

UPDATE: a later session re-verified this and found `torch` 2.13 works correctly after installing Python 3.12 and rebuilding the `.pythonlibs` venv — the "torch is broken" claim below was specific to a stale/broken venv, not a fundamental environment limitation. `conftest.py` was subsequently fixed to only stub torch when a runtime check (`torch.tensor([0])`) actually fails, rather than unconditionally. Don't assume torch is broken here without checking first.

Original note (may still apply to a differently-broken venv):

`torch` cannot actually load in this environment
(`libtorch_global_deps.so` is missing from the installed package), so
anything that transitively imports it at *module import time* becomes
unimportable, not just slow.

`trainer/registry/registry.py` (`ModelRegistry`) imports
`trainer.models.base.BaseQuantModel`, which imports `torch`. Because Python
always executes a package's `__init__.py` before handing back any of its
submodules, importing *anything* under `trainer.registry.*` — even
`trainer.registry.unified_registry`, which has no torch dependency of its
own — used to force that chain to run.

**Fix in place:** `trainer/registry/__init__.py` exposes `ModelRegistry` via
a module-level `__getattr__` (PEP 562) instead of a top-level import, so
`import trainer.registry.<anything>` no longer eagerly loads torch.
`from trainer.registry import ModelRegistry` still works for real callers,
lazily, the first time it's actually accessed.

**Why:** `tests/unit/test_execution.py` (and similar files) load
`trainer/execution/*.py` via `importlib.util.spec_from_file_location`
specifically to dodge torch-heavy import chains. Any new cross-package
import added to those modules can silently reintroduce a torch dependency
through an unrelated package's `__init__.py`, breaking the whole isolated
test harness even though the new code itself never touches torch.

**How to apply:** Before adding a new `from trainer.<pkg>.<submodule> import
X` at module scope inside `trainer/execution/*.py` (or other
isolation-tested packages), check whether `<pkg>/__init__.py` eagerly pulls
in anything that imports `torch` (grep for `import torch` reachable from
that `__init__.py`'s import graph). If so, either make the import lazy
(inside a function/method, `TYPE_CHECKING` for annotations) or fix the
offending `__init__.py` to expose the heavy symbol lazily, as done here.
Also: to run these isolation-style test files under pytest, use
`pytest <file> --noconftest -o addopts=""` — `tests/conftest.py` itself
imports torch and will fail collection otherwise.
