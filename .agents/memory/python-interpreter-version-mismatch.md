---
name: Python 3.11/3.12 interpreter mismatch — resolved by installing 3.12
description: pyproject.toml requires Python>=3.12; a stale note claimed installs were blocked, but installing python-3.12 via the language installer fully resolves it (real torch works too).
---

A prior version of this note claimed `uv add`/`pip install` were permanently blocked by a 3.11-vs->=3.12 mismatch and recommended stdlib-only workarounds. That was environment-specific at the time, not a hard constraint: installing Python 3.12 via the language installer (and rebuilding `.pythonlibs`) resolves the mismatch cleanly — `uv run pytest` and the full dependency set (including a genuinely working `torch` 2.13 build) install and run fine afterward.

**Why:** re-verified while fixing the test suite — the "torch is broken" and "3.12 can't be installed" notes were both stale; the real fix was just installing the correct interpreter version.

**How to apply:** if pyproject declares a Python version newer than the active runtime, install that version first via the language installer rather than assuming it's blocked or reaching for stdlib workarounds.
