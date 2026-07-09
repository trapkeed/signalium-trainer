---
name: Artifact signing enforcement wiring
description: Signature verification before promotion only runs when DeploymentManager is constructed with a real ArtifactRegistry — must be wired at every construction site, not just tests.
---

`DeploymentManager` accepts an optional `artifact_registry` param; signature
verification for SHADOW/CANDIDATE/ACTIVE promotions and ACTIVE rollbacks is a
silent no-op whenever it's `None`.

**Why:** kept as opt-in to preserve backward compatibility with tests/call
sites built before artifact signing existed. A code review caught that the
production server (`research_lab_server.py`) originally constructed
`DeploymentManager` without `artifact_registry`, meaning real promotions
skipped verification entirely even though unit tests of `DeploymentManager`
in isolation exercised the enforcement path correctly.

**How to apply:** any new place that constructs `DeploymentManager` for a
real (non-isolated-unit-test) flow must pass the live `ArtifactRegistry`
instance, or signature enforcement is not actually happening. When reviewing
this area again, grep all `DeploymentManager(` call sites and confirm each
one intentionally opts in or out.
