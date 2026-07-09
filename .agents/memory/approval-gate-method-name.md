---
name: Approval gate method name
description: ApprovalGate method is create_request(), not request_gate(); exact signature and idempotency contract.
---

# ApprovalGate.create_request() — Correct Method Name and Signature

The `ApprovalGate` class in `trainer/research_lab/approval_gate.py` exposes `create_request()`, NOT `request_gate()`.

**Signature:**
```python
def create_request(
    self,
    gate_type: ApprovalGateType,
    resource_id: str,
    resource_type: str,          # e.g. "shadow_test", "pipeline_job"
    requested_by: str,
    action_summary: str,         # human-readable description
    requestor_context: dict | None = None,
    ttl_seconds: float | None = None,
) -> ApprovalRequest:
```

**Why:** The method was named `create_request` (not `request_gate`) from initial implementation. Using the wrong name silently swallows the error (the tracker logs a warning and continues with an empty approval_id).

**Idempotency:** If an open PENDING request already exists for the same `(gate_type, resource_id)`, `create_request` returns the existing request. Always safe to call multiple times.

**How to apply:** Any code that creates approval gate requests must use `create_request`, not `request_gate`. Check callers by grepping for both names.
