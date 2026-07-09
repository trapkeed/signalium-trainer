---
name: Kill switch vs pause vs freeze
description: Semantic distinction between the three S30 operational-safety master switches
---

`trainer/research_lab/operational_safety.py` (`OperationalSafetySwitches`)
adds three explicit, previously-missing on/off primitives, each with a
distinct blast radius:

- **`emergency_disable`** — halts all trading activity outright (highest
  severity, blocks `can_serve_inference()` unconditionally).
- **`inference_kill_switch`** — severs the inference layer itself,
  regardless of which model would have been used. Distinct from pause:
  pause targets one model version, the kill switch takes down inference
  entirely.
- **`pause_model(model_version)`** — blocks only that specific model
  version from serving; other versions are unaffected.
- **`freeze_deployments`** — blocks all deployment stage transitions
  (SHADOW/CANDIDATE/ACTIVE) *and* `rollback_shortcut`, regardless of
  approval-gate state. Enforced via `assert_not_frozen()` raising
  `DeploymentsFrozenError`.

**Why:** the S30 exploration found rollback/pause were only reachable
procedurally (via withheld approvals or drawdown-triggered lockdown) with
no explicit queryable switch — operators had no single toggle to say "stop
everything now" vs "stop just this model" vs "stop new inference but keep
serving whatever's already running" vs "don't let anything redeploy while
we investigate."

**How to apply:** `rollback_shortcut()` delegates to
`QDERollbackManager.rollback_to_previous()` or
`DeploymentManager.request_rollback()` — it adds no new rollback logic,
only the one-call ergonomics, and it still respects the freeze switch.
