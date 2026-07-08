# Research Lab — Final Architecture Report

**Date:** 2026-07-08
**Status:** Confirmed/preserved architecture
**Related documents:**
- `docs/qde/hardcoded_behavior_audit.md`
- `docs/qde/learning_dimensions_architecture.md`
- `docs/qde/unified_qde_architecture.md`

---

## 1. Purpose of This Report

This report confirms and formalizes the boundary between **Research Lab** (experimentation) and **production QDE** (the single learned trading decision-maker defined in `unified_qde_architecture.md`). Research Lab is preserved as-is in principle: administrator-only, fully separated from production inference, and connected to production **exclusively** through the existing promotion/deployment pipeline. Nothing in this report weakens that boundary — it exists to document and lock it in as the system evolves toward the Unified QDE.

---

## 2. Position in the Overall System

```
              ┌───────────────────────────────────────────────┐
              │                 RESEARCH LAB                    │
              │           (administrator-only, isolated)         │
              │                                                   │
              │  Experimental models        Alternative           │
              │  Research datasets          architectures         │
              │  Feature engineering        Experimental risk      │
              │  experiments                 systems               │
              │  Experimental ensembles     Experimental trading    │
              │                              methods                │
              └───────────────────────┬───────────────────────┘
                                       │
                                       │  ONLY path out:
                                       │  Deployment Gates → Promotion Approval
                                       ▼
              ┌───────────────────────────────────────────────┐
              │              QDE Deployment Pipeline              │
              │   (trainer/qde/qde_pipeline.py, qde_promotion.py) │
              │   TRAINING → VALIDATION → BACKTESTING →           │
              │   SHADOW_TESTING → PRODUCTION                     │
              └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
              ┌───────────────────────────────────────────────┐
              │            PRODUCTION QDE (Unified)              │
              │   One model. One intelligence. One trader.        │
              │   Live inference, real capital, real decisions.   │
              └───────────────────────────────────────────────┘
```

There is exactly one door between Research Lab and production: the deployment pipeline. No other code path, shared process, shared database, or shared runtime connects the two.

---

## 3. Isolation Guarantees

### 3.1 Access control (administrator-only)
- **Outer layer:** `apps/trainer-api/src/middlewares/admin_auth.ts` enforces a token check (`RESEARCH_LAB_ADMIN_TOKEN`) via `Authorization: Bearer` or `X-Admin-Token`, using constant-time comparison.
- **Route namespacing:** All Research Lab endpoints live under `/api/admin/research/` (`apps/trainer-api/src/routes/research_lab.ts`) — no retail-facing or production-inference route can reach them.
- **Inner layer:** `trainer/research_lab/research_lab_server.py` runs its own FastAPI server on an internal-only port (8002), separate from the QDE service (8001), and independently re-validates the admin token (`_require_admin`) as defense-in-depth.
- **No public API surface:** `docs_url`, `redoc_url`, `openapi_url` are disabled on the Research Lab FastAPI app to prevent schema/endpoint discovery.

### 3.2 Separation from production inference
- **Separate process:** Research Lab runs as its own service, distinct from the QDE inference service.
- **Separate persistence:** `trainer/research_lab/rl_database.py` uses a dedicated database, isolated from production QDE state, so experimental runs, backtests, and shadow-test decisions never mix with production data.
- **Separate implementation surface:** `ResearchLabAPI` (`trainer/research_lab/research_lab_api.py`) is never imported into or instantiated by production/retail-facing code — it is not on the production dependency graph.

This means a bug, a bad experiment, or an untested idea in Research Lab **cannot** affect a live trading decision, because there is no code path by which it executes against real capital.

---

## 4. What Research Lab May Contain

Research Lab is explicitly the sanctioned space for anything not yet proven — this is where exploration should happen instead of leaking into production logic:

| Category | Examples |
|---|---|
| Experimental models | New architectures, alternative parameterizations, prototypes of the Unified QDE's next iteration |
| Alternative architectures | Structurally different approaches to the same learning problem, compared against the current champion |
| Research datasets | New labeling schemes, new benchmark constructions, synthetic/augmented datasets |
| Feature engineering experiments | Candidate features tracked, ranked by importance, and formally accepted/rejected by an admin |
| Experimental risk systems | New approaches to the Risk Reasoning / Exposure Management evaluation dimensions (see `learning_dimensions_architecture.md`) before they're proven |
| Experimental ensembles | Multi-model or multi-checkpoint combinations being evaluated — note: an ensemble may be *researched* here even though production remains "one model" post-promotion |
| Experimental trading methods | Any idea for how the QDE could behave differently, tested via backtest/shadow test before ever being considered for promotion |

Nothing in this list is production logic. It is all candidate material that must earn its way into production through the pipeline in §5.

---

## 5. The Only Path to Production: Deployment Pipeline

Research Lab artifacts cannot directly affect the production QDE. The only mechanism is the existing promotion/deployment pipeline:

1. **Deployment gates** — an experiment must pass three mandatory gates before promotion can even be considered:
   - Decision Quality Improvement
   - Risk-Adjusted Return Improvement
   - Drawdown Stability
   (documented in `docs/research_lab/deployment_flow.md`)
2. **Two-person promotion approval** — one admin evaluates the gates, a second admin formally approves deployment (`POST /admin/research/experiments/:id/approve`). No single person can move an experiment to production unilaterally.
3. **QDE handover** — on approval, the candidate is registered in `QDERegistry` (`trainer/qde/qde_registry.py`) and advanced through the QDE pipeline (`trainer/qde/qde_pipeline.py`): `TRAINING → VALIDATION → BACKTESTING → SHADOW_TESTING → PRODUCTION`.
4. **Shadow testing before activation** — the candidate ("challenger") runs alongside the current production model ("champion"), recording decisions without executing real trades, so its real-world behavior is verified before it ever touches capital.
5. **Rollback path** — if a promoted version underperforms, a similarly gated multi-admin rollback workflow (`trainer/qde/qde_rollback.py`) reverts production to the prior version.

This pipeline is the same one described in `unified_qde_architecture.md` §4.2 ("Validation," "Deployment") — Research Lab feeds candidates into it, but never bypasses it.

---

## 6. Explicit Boundary Rules

To keep this architecture intact as the system evolves:

1. **No shared runtime.** Research Lab code must never run inside the production QDE service process.
2. **No shared database writes.** Production QDE must never read from or write to `research_lab.db`, and Research Lab must never write to production QDE state except through the registry/pipeline APIs designed for that purpose.
3. **No silent promotion.** There is no code path that marks an experiment "production" without passing through gates + two-admin approval + the QDE pipeline stages.
4. **No production dependency on Research Lab availability.** If the Research Lab service is down, production inference must be unaffected — it is architecturally optional to the running system.
5. **No bypassing shadow testing.** Every promoted candidate must pass through shadow testing against the champion before activation, regardless of how promising its offline backtest results are.
6. **Research Lab experiments remain evaluatable, not authoritative.** Even an approved experiment doesn't "become" the trading logic — it becomes a new trained QDE version, evaluated the same way as any other version (see `learning_dimensions_architecture.md`), consistent with the "one model, one decision maker" principle.

---

## 7. Success Criteria

This architecture is correctly preserved when:
1. Access to Research Lab requires admin authentication at both the gateway and service layer, with no retail/public path in.
2. Research Lab runs as an isolated service and database, with zero shared runtime state with production QDE.
3. Every category in §4 can be freely experimented with inside Research Lab without any risk to live trading.
4. The only way an experimental idea reaches production capital is by passing deployment gates, two-admin approval, and the full `TRAINING → VALIDATION → BACKTESTING → SHADOW_TESTING → PRODUCTION` pipeline.
5. Production inference has no runtime dependency on the Research Lab service.

---

## 8. Non-Goals

This report does not change any existing access control, service topology, or promotion workflow — it documents and formalizes the boundary that already exists in the codebase (`trainer/research_lab/`, `apps/trainer-api/src/middlewares/admin_auth.ts`, `apps/trainer-api/src/routes/research_lab.ts`, `trainer/qde/qde_pipeline.py`, `trainer/qde/qde_promotion.py`, `trainer/qde/qde_rollback.py`) so that it is explicitly preserved as the Unified QDE architecture (`unified_qde_architecture.md`) is implemented.
