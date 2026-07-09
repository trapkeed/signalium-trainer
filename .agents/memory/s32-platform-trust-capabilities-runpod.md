---
name: S32 platform trust, capability endpoints, RunPod job manager
description: Where origin trust, consolidated capability API, RunPod job lifecycle, and symbol/reconciler gap-fills landed after the second multi-domain spec
---

- **Platform origin trust**: `apps/trainer-api/src/middlewares/platform_trust.ts`. `requirePlatformOrigin` only rejects (403) when an `Origin` header is present and not in the allowlist (`PLATFORM_ALLOWED_ORIGINS` env, defaults to signaliumai.com + localhost) — server-to-server calls send no `Origin` and fall through to existing `adminAuth`/token checks untouched. Applied as the first middleware (before `adminAuth`) on every existing admin-gated route in `qde.ts` and `research_lab.ts`. `platformCorsOptions()` replaces the previously fully-permissive `cors()` in `app.ts`.

- **Capability endpoints**: new `apps/trainer-api/src/routes/capabilities.ts`, mounted unauthenticated (read-only discovery surface) at `/api/capabilities/*` plus a consolidated `/api/capabilities` that Promise.all's all nine sub-capabilities. Each capability proxies one upstream call (QDE :8001 / Research Lab :8002) and degrades to `{available:false}` on failure rather than 500ing the whole response — verified live that a downstream-unreachable dev environment still returns 200 with `available:false`.

- **RunPod job lifecycle**: new `trainer/infra/runpod_jobs.py`, distinct from `execution_backend.py`'s `RunPodWorkerBackend` (single-job dispatch) — this adds multi-job tracking (`RunPodJobManager`), GPU selection (`select_gpu`, cheapest-eligible-then-`GPU_PREFERENCE_ORDER`-tiebreak), and artifact/checkpoint/cleanup steps, all via an injectable `RunPodClient` Protocol so it's fully offline-testable. Follows the S31 stub convention: every method raises `RunPodNotConfigured` until a real client is injected.

- **Symbol registry gaps**: added `BROKER_PROFILES` dict (per-broker raw-spelling overrides, e.g. OANDA's `US30_USD`) layered before admin/builtin/learned alias lookup in `symbol_normalizer.py`; added `inverse_lookup(canonical)` (all known spellings across builtin+broker+admin+learned) and opt-in `fuzzy=True` param on `normalize()`/`resolve()` (difflib-based, only applied after every exact/structural path fails, to avoid false positives on genuinely unknown instruments). Also filled in previously-missing aliases named in spec: DJ30/US30Cash/US30USD→US30, DE40/GER40Cash/Germany40→GER40.

- **Reconciler domains**: `ReconciliationDomain` gained `SYMBOLS`, `USER_STATUS`, `AUTONOMOUS_TRADING_PERMISSIONS`, all added to `SIGNALIUM_OWNED` (Signalium is the source of truth for these, Trainer only applies).

**Why not touch `adminAuth` itself**: origin trust and token auth are orthogonal checks (browser-CSRF-style vs. bearer-token authorization) — composing them as two middlewares in sequence, rather than merging into one, keeps `adminAuth`'s existing fail-closed contract and tests untouched.
