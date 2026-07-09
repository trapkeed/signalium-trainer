---
name: S29 trainer security additions
description: What net-new security modules were added for Phase S29 (encryption, secure loading, inference request security, dataset immutability) and the field-naming gotcha in ArtifactRecord.
---

S29 audited existing security (S20/S21A artifact integrity+signing, sso.py RBAC/JWT, audit_log.py, approval_gate.py, deployment_manager.py signature enforcement) and found these genuinely missing, so added net-new modules rather than duplicating:

- `trainer/research_lab/artifact_encryption.py` — Fernet symmetric encryption at rest (`ARTIFACT_ENCRYPTION_KEY` secret) + `load_artifact_securely()` that decrypts, re-verifies the plaintext SHA-256 against the registered hash, and optionally checks the Ed25519 signature in one call — the "secure loading / anti-corruption" gap (previously verification only ran once at promotion time via `DeploymentManager`).
- `trainer/research_lab/inference_security.py` — HMAC-SHA256 per-caller request signing, nonce+timestamp replay guard, token-bucket rate limiter, sliding-window abuse detector with auto-block. Intentionally decoupled from `sso.py`'s JWT flow since it's meant for machine-to-machine (Signalium -> Trainer) calls authenticated by a shared secret, not a human SSO session.
- `trainer/datasets/protection.py` — write-once dataset version publishing (sentinel-file pattern, mirrors `ArtifactStore.reserve_version`) + `load_dataset_version_securely()` that runs `DatasetValidator` checksum validation and logs every read/write to the mlops `AuditSystem`.

**Gotcha:** `ArtifactRecord.new()` takes `extra: dict`, not `metadata` — despite `TrainingRunRecord.new()` on the same file using `metadata`. Don't assume field names are consistent between sibling dataclasses in `artifact_models.py`; check the actual `.new()` signature before writing kwargs.

**Test-suite note:** running many DB-backed `tests/unit/test_*.py` files together in one pytest invocation can hang/timeout due to pre-existing module-level DB-singleton contention (same root cause as `real-server-test-isolation.md`); this is NOT related to new code — verify by running suspect files individually before concluding a new module broke something.
