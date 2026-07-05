---
name: SQLite memory DB persistence
description: HRSADatabase :memory: loses schema across multiple connect() calls.
---

## The Rule

`HRSADatabase(":memory:")` must keep one persistent `sqlite3.Connection`
for the lifetime of the object, reusing it for every `_conn()` call.
Never open a new connection for `:memory:` databases.

## Why

Each `sqlite3.connect(":memory:")` call creates an independent, empty
in-memory database. The schema created in `_init_db()` is on connection C1;
a second `sqlite3.connect(":memory:")` in `save()` gets fresh connection C2
which has no tables → `OperationalError: no such table`.

This affects tests and any in-process usage. File-based databases are fine
because they persist to disk across connections.

## How to Apply

```python
def __init__(self, db_path: str) -> None:
    self._is_memory = db_path == ":memory:"
    self._persistent_conn = None
    if self._is_memory:
        self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._persistent_conn.execute("PRAGMA journal_mode=WAL")
        self._persistent_conn.row_factory = sqlite3.Row
    self._init_db()

@contextmanager
def _conn(self):
    if self._is_memory:
        try:
            yield self._persistent_conn
            self._persistent_conn.commit()
        except Exception:
            self._persistent_conn.rollback()
            raise
    else:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        ...
```
