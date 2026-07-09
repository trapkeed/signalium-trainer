---
name: Pydantic v2 deferred-annotation bug in tests
description: Pydantic models defined inside methods with `from __future__ import annotations` at module scope silently produce 422 Unprocessable Entity in FastAPI TestClient calls, even though a standalone script works fine.
---

## The bug

When `from __future__ import annotations` is at the top of a test file, ALL annotations in that module are treated as strings (deferred evaluation). If you define a Pydantic `BaseModel` subclass inside a function or method body, Pydantic v2 cannot resolve the forward-reference annotation strings at runtime — the model schema is built with unresolvable types, causing FastAPI to reject every request body with a 422.

The confusing part: the exact same code works fine in a standalone script that doesn't have `from __future__ import annotations`.

## The fix

Define any Pydantic `BaseModel` classes that are used as FastAPI request/response models at **module scope** (top level), never inside `def` or `class` bodies when the module has `from __future__ import annotations`.

## Why

Pydantic v2 builds the JSON schema at class-definition time. At module scope, it can resolve forward references via `sys.modules` and the module's global namespace. Inside a function body, the enclosing scope is a closure — Pydantic cannot look up the string annotations in the right namespace, so type resolution fails silently.

## How to apply

- In test files that use `from __future__ import annotations` and define miniature FastAPI apps for endpoint testing: put the request model class at the top of the file (below the imports).
- Convention adopted: prefix such test-only models with `_` (e.g. `_InvestigateReq`, `_DiscoverReq`).
