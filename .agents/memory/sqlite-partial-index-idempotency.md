---
name: SQLite partial index idempotency pattern
description: How to enforce at-most-one PENDING row per compound key using SQLite partial unique index + split INSERT paths, and why multi-clause ON CONFLICT is invalid in SQLite.
---

## The pattern

When you need at most one row in status X per (col_a, col_b), and rows can also transition to other statuses:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_unique
    ON my_table(col_a, col_b) WHERE status = 'PENDING';
```

Then in Python, split the upsert into two paths:

```python
if status == "PENDING":
    # INSERT OR IGNORE: silently drops if the partial unique index fires
    conn.execute("INSERT OR IGNORE INTO ...", params)
else:
    # Status transition: update existing row by primary key
    conn.execute(
        "INSERT INTO ... ON CONFLICT(pk) DO UPDATE SET status = ...", params
    )
```

After a PENDING insert, always re-fetch by (col_a, col_b, status='PENDING') to get the canonical row — the INSERT OR IGNORE may have been silently dropped by a concurrent caller.

## Why NOT multi-clause ON CONFLICT

SQLite does **not** support multiple ON CONFLICT clauses in a single INSERT statement:
```sql
-- INVALID in SQLite:
INSERT ... ON CONFLICT(pk) DO UPDATE SET ... ON CONFLICT(a, b) DO NOTHING
```
This causes a syntax error. Split into two branches instead.

## Why

Single-threaded code can use application-level read-check-then-write. Under concurrent SQLite writes (WAL mode serialises writes but both threads can pass the read check before either writes), two PENDING rows can be inserted. The partial unique index + INSERT OR IGNORE is the correct DB-level enforcement.

## How to apply

Use this pattern whenever a workflow table needs idempotent "create open request" semantics (approval gates, lifecycle authorizations, any queue-style table with a status column).
