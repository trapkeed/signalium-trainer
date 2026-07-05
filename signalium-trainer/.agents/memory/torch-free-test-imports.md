---
name: torch-free test imports
description: How to import trainer.research submodules in tests without torch installed.
---

## The Rule

All unit tests for `trainer/research/*.py` modules MUST load them via
`importlib.util.spec_from_file_location`, never via `importlib.import_module`.

## Why

`trainer/research/__init__.py` line 64 does:
```python
from trainer.research.benchmark import (...)
```
`trainer.research.benchmark.runner` does `import torch`.
torch is NOT installed in the Nix test environment.

When `importlib.import_module("trainer.research.anything")` is called, Python
executes `trainer/research/__init__.py` first → fails → all imports in the
test fail with `ModuleNotFoundError: No module named 'torch'`.

## How to Apply

Use the pattern from `tests/unit/test_research_scoring.py` and
`tests/unit/test_hrsa.py`:

```python
_BASE = Path(__file__).resolve().parents[2] / "trainer" / "research"

def _load(mod_key: str, filename: str):
    path = _BASE / filename
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod

# Load in strict dependency order
_raw_metrics = _load("trainer.research.raw_metrics", "raw_metrics.py")
# ... etc.
```

Load order matters: dependencies must be loaded before dependents.
Also stub loguru before loading anything.
