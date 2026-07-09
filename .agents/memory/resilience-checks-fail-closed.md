---
name: Safety/rollback gates must fail closed on "unverifiable"
description: Distinguish monitoring signals (soft failure OK) from go/no-go safety gates (soft failure must block) when reusing the same checksum/verification primitive.
---

The same "file missing/unreadable" outcome should be treated differently depending on context:
- In a **monitoring/observability report** (e.g. corrupted-artifact detection dashboard), "unverifiable" is a soft signal — worth surfacing, not worth blocking anything, since the artifact may legitimately live elsewhere.
- In a **binary safety gate** (e.g. "is it safe to roll back to this model version"), the same "unverifiable" outcome must block the action. "Cannot verify" is not "verified safe."

**Why:** a rollback-safety check initially let UNVERIFIABLE artifacts pass silently (treated as non-blocking, same as a monitoring report would), which would have let an operator roll back to a version whose integrity could not actually be confirmed. Caught by code review.

**How to apply:** when a checksum/integrity primitive is shared between a dashboard and a safety gate, the dashboard can pass through the raw three-way status (VERIFIED/CORRUPTED/UNVERIFIABLE), but the safety gate must collapse anything other than VERIFIED into "not safe."
