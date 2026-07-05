---
name: MIE test isolation pattern
description: How to isolate trainer.mie modules in tests without triggering __init__.py or loguru import errors
---

## Rule
When testing `trainer/mie/` modules with `spec_from_file_location`, pre-load ALL modules in dependency order and register each in `sys.modules` under its full dotted name BEFORE executing any module. Then redefine `_load(short_name)` to look up from `sys.modules` so per-test helpers work without re-executing.

## Why
`from trainer.mie.mie_types import X` in any module triggers Python to look up `sys.modules["trainer.mie.mie_types"]` first. If already registered there, it returns immediately without loading the parent package `trainer.mie` (which runs `__init__.py` and imports torch). The pre-registration trick is identical to the ICARI test pattern.

## How to apply
```python
def _preload(rel_path):          # used for initial load in dependency order
    full = os.path.join(ROOT, rel_path)
    name = rel_path.replace("/", ".").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, full)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod      # register BEFORE exec
    spec.loader.exec_module(mod)
    return mod

def _load(short_name):           # used inside test helper methods
    key = "trainer.mie." + short_name.replace(".py", "")
    return sys.modules[key]
```

## Loguru
All engine files need `try: from loguru import logger / except ImportError: import logging; logger = logging.getLogger(__name__)`. Loguru is not installed in this environment.
