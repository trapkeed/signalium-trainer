---
name: Duplicated math must stay parity-tested
description: When a module reimplements a shared metric formula instead of importing across a dependency boundary, add a direct parity test.
---

Some modules deliberately duplicate small pure functions (e.g. a
total-variation-distance calculation) rather than importing them from
another module, in order to avoid an unwanted import-time dependency (e.g.
pulling in a torch-dependent module, or creating a circular import). This is
a legitimate design choice, but two independently-named/implemented
functions computing "the same thing" silently drifted apart in practice —
one used a materially different formula (average absolute difference vs.
`0.5 * L1` true TV distance) than the original it was meant to mirror.

**Why:** without a direct test importing both implementations and asserting
numeric equality across representative inputs, this kind of drift is
invisible until it produces inconsistent severity/alerting for the same
underlying telemetry.

**How to apply:** whenever duplicating a formula for dependency-isolation
reasons, add a test that imports both the original and the duplicate and
asserts they agree across several cases — don't rely on "same docstring" as
proof of correctness.
