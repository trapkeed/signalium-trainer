---
name: SQLite ON CONFLICT scalar sync
description: ON CONFLICT upserts must refresh all indexed/filter scalar columns, not just data_json.
---

# SQLite ON CONFLICT — All Scalar Filter Columns Must Be Updated

When a table has both scalar filter columns AND a data_json blob, the ON CONFLICT
clause must update ALL scalar columns alongside data_json.

**Why:** `list_*` queries filter on scalar columns (e.g. `WHERE severity = ?`). If only
`data_json` is refreshed on conflict, the scalar index columns remain stale after an
update — subsequent filter queries silently return wrong results.

**How to apply:** Every `ON CONFLICT(pk) DO UPDATE SET` block must explicitly refresh
each scalar filter column that list queries depend on:

```sql
ON CONFLICT(alert_id) DO UPDATE SET
    dimension     = excluded.dimension,
    severity      = excluded.severity,
    model_version = excluded.model_version,
    created_at    = excluded.created_at,
    data_json     = excluded.data_json
```

**Pattern:** Write a regression test that upserts the same PK with different scalar values
and then verifies the filter query returns the updated rows (not the old ones). This
catches missing column refreshes in code review.
