---
name: Real-server test module isolation
description: research_lab_server.py creates module-level DB/controller singletons; integration tests must force a fresh module import per test class to avoid cross-test DB contamination.
---

## The problem

`research_lab_server.py` creates its `_db`, `_approval_gate`, `_pipeline_ctrl`, and other singletons at import time, reading `RESEARCH_LAB_DB_PATH` from the environment at that moment. Once the module is cached in `sys.modules`, changing the env var and re-importing returns the same stale module — the singletons still point at the first DB path.

When multiple test classes that each set different `RESEARCH_LAB_DB_PATH` values run in the same pytest session, they all end up sharing the DB from the first import.

## The fix

In each `client` fixture that imports `research_lab_server`, purge the relevant modules from `sys.modules` BEFORE importing, then import fresh:

```python
@pytest.fixture()
def client(self, tmp_path):
    import sys, os
    os.environ["RESEARCH_LAB_ADMIN_TOKEN"] = "test-token"
    os.environ["RESEARCH_LAB_DB_PATH"] = str(tmp_path / "test.db")
    for mod_name in list(sys.modules):
        if "research_lab_server" in mod_name or "telemetry_server" in mod_name:
            del sys.modules[mod_name]
    import trainer.research_lab.research_lab_server as srv_mod
    from starlette.testclient import TestClient
    return TestClient(srv_mod.app)
```

## Why

The server module is structured for a long-running process (FastAPI), not per-request re-init. Forcing a module reload is the correct integration-test pattern rather than refactoring the server to lazy-init singletons.

## How to apply

Every integration test class that tests the real `research_lab_server.app` endpoints (not a locally-constructed FastAPI test app) must use this pattern in its `client` fixture.
