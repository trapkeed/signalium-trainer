---
name: Gateway-mediated cross-service authority enforcement
description: How to make one Python service the sole authority over actions executed by another, sibling Python service, when they must never import each other
---

When Research Lab needed to become the sole authority over QDE version lifecycle mutations (promote/rollback/archive/retire) without violating the `trainer.qde` ↔ `trainer.research_lab` no-import-cycle rule (see qde-research-lab-dependency-direction.md), the authority check could not be a direct Python function call between the two services.

**Pattern used:** the authorizing service (Research Lab) owns a single-use authorization record with a state machine (`PENDING → APPROVED → CONSUMED`/`REJECTED`), issued either automatically as a side effect of its existing approval workflow, or via an explicit authorize endpoint. The Express/Node gateway — the only component with legitimate HTTP access to both Python services — is where enforcement actually lives: a middleware calls the authorizing service to atomically consume the authorization before forwarding the mutating request to the executing service. The executing service (QDE) stays completely unaware the authorizer exists.

**Why:** preserves independent-service process isolation (no import edge, no shared DB) while still getting hard, structural (not conventional) enforcement of a cross-service authority relationship. The "atomic consume" step is the load-bearing detail — without it, two callers could race on the same approval and both succeed.

**How to apply:** any future requirement that Service A must gate/approve mutations performed by sibling Service B (where A and B must not import each other) should reuse this shape: authorization state machine in A, gateway middleware that consumes it, mutation endpoint in B that trusts the gateway.
