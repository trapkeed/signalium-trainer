---
name: S26 resilience module reuses S21A signature verification
description: ArtifactIntegrityVerifier/rollback-safety checks must delegate to DeploymentManager.verify_artifact_signatures, not reimplement signature checking.
---

`DeploymentManager.verify_artifact_signatures(model_version)` is the single source of truth for artifact signature verification (Ed25519, `.sig` sidecar files next to each artifact, via `ArtifactSigner`). Any new module that needs to gate an action (e.g. rollback safety) on "is this model version's artifacts trustworthy" must call that method rather than re-deriving signature logic against a different convention (e.g. an `extra.signature` field) — the two will silently diverge and under/over-trust artifacts.

**Why:** first implementation of a resilience/rollback-safety checker reinvented signature checking against a nonexistent field, so it never actually verified anything even though it looked complete; code review caught it.

**How to apply:** when writing safety/integrity gates over model artifacts, take a `deployment_manager` (or equivalent) dependency and call its existing verification method; don't touch `.sig` files directly.
