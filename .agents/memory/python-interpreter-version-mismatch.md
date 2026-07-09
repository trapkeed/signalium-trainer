---
name: Python 3.11/3.12 interpreter mismatch blocks package installs
description: This environment's pyproject requires Python >=3.12 but the actual runtime is 3.11; both uv add and pip install --break-system-packages fail for new dependencies (e.g. PyJWT).
---

`uv add <pkg>` and `pip install --break-system-packages <pkg>` both fail here because pyproject.toml declares `requires-python >= 3.12` while the actual interpreter running tests/the app is 3.11.14.

**Why:** discovered while trying to add PyJWT for JWT verification — the installer refused to resolve/install against the mismatched interpreter constraint.

**How to apply:** before reaching for a new Python dependency, check `python3 --version` vs `pyproject.toml`'s `requires-python`. If they conflict and you can't change the constraint, prefer implementing the needed functionality with stdlib (`hmac`, `hashlib`, `base64`, `json`) plus whatever's already installed (e.g. `cryptography` was available for RS256) rather than fighting the install. Don't re-attempt the same install a second way expecting a different result.
