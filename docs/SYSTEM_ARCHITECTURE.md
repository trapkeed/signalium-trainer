# Signalium Trainer — Comprehensive System Architecture

*Last updated: 2026-07-09 (post S32: platform origin trust, capability endpoints, RunPod job manager, symbol registry enhancements)*

This document is the top-level map of the whole system: every service, package, trust
boundary, and data flow, in one place. Deep-dive docs for individual subsystems are
linked throughout — this file tells you which one to open next.

## 1. What this system is

Signalium Trainer is the GPU-side training and research platform for a quantitative
trading product. It has three deployable parts living in one pnpm/uv monorepo:

| Part | Path | Role |
|---|---|---|
| **Trainer API** | `apps/trainer-api/` | Node/Express gateway — the only network-facing entrypoint into QDE + Research Lab |
| **Research Lab Admin Dashboard** | `artifacts/research-dashboard/` | React/Vite web UI for researchers/admins |
| **Trainer core** | `trainer/` (Python) | Training, QDE decision engine, Research Lab domain logic, RunPod infra |

Supporting pieces: `config/` (Hydra/dataclass configs + per-environment YAML),
`docker/` (CPU/GPU images), `tests/` (pytest suite mirroring `trainer/`),
`.github/workflows/ci.yml` (CI gate for all of the above).

```
                         ┌─────────────────────────┐
                         │   signaliumai.com (web)  │
                         └────────────┬────────────┘
                                      │ HTTPS (Origin header present)
                                      ▼
                         ┌─────────────────────────┐
                         │   Trainer API (Node)     │  apps/trainer-api
                         │   Express :PORT          │
                         │   - platform_trust        │
                         │   - admin_auth            │
                         │   - capabilities (public) │
                         └──────┬───────────┬───────┘
                                │           │
                    :8001 (QDE) │           │ :8002 (Research Lab)
                                ▼           ▼
                    ┌────────────────┐ ┌──────────────────────┐
                    │ QDE Python svc │ │ Research Lab Python   │
                    │ trainer/qde    │ │ svc trainer/research_lab │
                    └────────────────┘ └──────────────────────┘
                                │
                                ▼
                    ┌────────────────────────────┐
                    │ Trainer core (Python)       │
                    │ training / registry /       │
                    │ validation / infra (RunPod) │
                    └────────────────────────────┘
```

The Research Dashboard talks to the Trainer API the same way any other client does —
over HTTP, through the same origin-trust + auth gates.

## 2. Trainer API (`apps/trainer-api/`)

Express app, entrypoint `src/index.ts` → `src/app.ts`.

```
src/
├── app.ts                         Express app wiring: pino logging, CORS, JSON body, mounts router at /api
├── index.ts                       process bootstrap (listen on PORT)
├── lib/logger.ts                  pino logger instance
├── middlewares/
│   ├── platform_trust.ts          requirePlatformOrigin + platformCorsOptions()
│   ├── admin_auth.ts              bearer-token admin authentication (401/403 contract)
│   └── require_rl_authorization.ts  atomic-consume authorization record for QDE↔Research Lab sole authority
└── routes/
    ├── index.ts                   mounts health, qde, research_lab, capabilities under /api
    ├── health.ts                  /api/healthz liveness probe
    ├── qde.ts                     admin-gated proxy routes to the QDE service (:8001)
    ├── research_lab.ts            admin-gated proxy routes to the Research Lab service (:8002)
    └── capabilities.ts            unauthenticated read-only discovery surface (/api/capabilities/*)
```

### 2.1 Trust boundary layering

Every admin-gated route runs two independent middlewares in sequence, deliberately
kept as two composable checks rather than merged into one:

1. **`requirePlatformOrigin`** (`platform_trust.ts`) — CSRF-style browser check. Rejects
   (403) only when an `Origin` header is present and it is not in the
   `PLATFORM_ALLOWED_ORIGINS` allowlist (defaults to `signaliumai.com` + localhost).
   Server-to-server calls send no `Origin` header and fall through untouched to the
   next check.
2. **`adminAuth`** (`admin_auth.ts`) — bearer-token authorization. Missing token → 401;
   wrong token → 403. This is the same contract documented for the Research Lab admin
   API (see `.agents/memory/research-lab-admin-auth-contract.md`).

`platformCorsOptions()` replaces what used to be a fully-permissive `cors()` call in
`app.ts`, so the CORS layer and the origin-trust middleware enforce the same allowlist.

### 2.2 Capability endpoints (`/api/capabilities/*`)

Unauthenticated, read-only discovery surface — intentionally public because it exposes
no data, only "is this subsystem available" flags. Each sub-capability proxies one
upstream call to QDE (`:8001`) or Research Lab (`:8002`) and **degrades to
`{ "available": false }` on failure instead of 500ing**. The consolidated
`GET /api/capabilities` endpoint `Promise.all`s all nine sub-capabilities so a single
unreachable downstream never breaks the aggregate response — this was verified live
against a dev environment with the downstream services stopped.

### 2.3 QDE / Research Lab sole-authority

`require_rl_authorization.ts` enforces that only the Research Lab may authorize
QDE lifecycle-affecting actions, using an atomic-consume authorization record rather
than a direct Python import — see
`.agents/memory/gateway-mediated-lifecycle-authority.md` for why this is
gateway-mediated instead of an in-process check.

## 3. Research Lab Admin Dashboard (`artifacts/research-dashboard/`)

React + Vite SPA.

```
src/
├── pages/          Route-level views (Symbols, experiment/registry/health dashboards, etc.)
├── components/      Shared presentational components
│   └── ui/           Design-system primitives
├── hooks/           Data-fetching / state hooks
└── lib/             API client, formatting utilities
```

Communicates with the Trainer API exclusively through `/api/...`; visibility for
read-only views is gated separately from mutating-action capability (see
`.agents/memory/visibility-vs-capability-gating.md`) so a role can view dashboards
without being able to trigger actions.

## 4. Trainer core (`trainer/`, Python)

```
trainer/
├── api/              Signalium REST API client (auth, download, upload, retry) — the
│                     only sanctioned path to Signalium's Postgres data; Trainer never
│                     connects to the database directly.
├── checkpoints/      Checkpoint save/load/rotation (latest.pt, best.pt, epoch_NNNN.pt)
├── cli/              Typer CLI, one subcommand module per command
├── config/           Hydra/dataclass config schemas — single source of truth
├── dataloaders/      DataLoaderFactory — reproducible, pinned-memory, prefetching
├── datasets/         BaseDataset, ParquetDataset, StreamingParquetDataset, validation
├── evaluation/       StandardEvaluator, EvaluationReporter
├── exporters/        TorchScriptExporter, ONNXExporter, ManifestBuilder
├── infra/            RunPod job lifecycle (runpod_jobs.py) — see §4.3
├── logging/          Loguru setup — JSON/text, file rotation
├── market_data/      symbol_normalizer.py and related market-data plumbing — see §4.4
├── metrics/          MetricsTracker, EvaluationMetrics
├── models/           Abstract interfaces: BaseQuantModel, BaseTrainer, BaseEvaluator, BaseExporter
├── qde/               Quantitative Decision Engine — signal heads, translation, observer FSM
├── registry/         ModelRegistry — decorator-based model registration; lazily imports
│                     torch so unrelated modules don't force-load it
├── research/          Reproducibility, lineage/knowledge-graph API, benchmarking (IRSE),
│                     experiment records
├── research_lab/      Research Lab domain: persistence (rl_database), approval gates,
│                     champion/challenger tracker, health monitors
├── training/          TrainingEngine, MultiTaskEngine, EarlyStopping, MLflowTracker, seeds
├── training_signals/  Hindsight labelers deriving stop/TP/size/allocation/lifecycle
│                     labels from realized future outcomes (never from live heuristics)
├── utils/             device resolution, GPU info, git info
└── validation/        ValidationEngine, leaderboard, evaluator windowing
```

Full package-level design principles (SOLID, dependency injection, API-only data
access, reproducibility-first) are documented in `docs/architecture.md`; this file
supersedes it as the whole-system view but does not duplicate its content.

### 4.1 QDE (Quantitative Decision Engine)

- Produces raw signal heads; a single pure function translates heads → execution
  vocabulary. New consumers (e.g. position management) read heads directly rather than
  re-deriving through the entry-side translator.
- Runs a per-symbol `OBSERVING → ... → COOLDOWN` state machine that is **advisory
  context/telemetry only** — never a trade-placing or blocking authority.
- Decision engines (Risk/Exposure/Lifecycle/Portfolio) are pure information providers;
  only structural FSM validity and true safety rails may hard-block a decision.
- `trainer.qde` may import Research Lab dataclasses (`rl_models`); the reverse is
  forbidden — `trainer.research_lab` must never import `trainer.qde`. Bridging logic
  lives under `trainer/qde/`.
- See `docs/qde/unified_qde_architecture.md` and
  `docs/qde/learning_dimensions_architecture.md` for the full model.

### 4.2 Research Lab

- Persistence via `ResearchLabDatabase(db_path)`, configurable through
  `RESEARCH_LAB_DB_PATH`; `ResearchLabAPI(db=...)` supports dependency injection for
  tests.
- Champion/challenger promotion, approval gates (idempotent per
  `(gate_type, resource_id)` via a partial unique index), and health monitoring
  (composite score across 13 drift dimensions collapsed into 5 disjoint categories)
  all live here. See `docs/qde/research_lab_final_architecture.md`.
- Immutable artifact store: write-once via `O_CREAT|O_EXCL`, never check-then-write;
  all writers route through the registry's own store.

### 4.3 RunPod infrastructure (`trainer/infra/runpod_jobs.py`)

Added in S32. Distinct from `execution_backend.py`'s `RunPodWorkerBackend` (which does
single-job dispatch): `RunPodJobManager` adds multi-job tracking, GPU selection
(cheapest-eligible, then `GPU_PREFERENCE_ORDER` tiebreak via `select_gpu`), and
artifact/checkpoint/cleanup steps. Fully offline-testable via an injectable
`RunPodClient` Protocol — every method raises `RunPodNotConfigured` until a real
client is injected, following the established stub convention. See
`docs/runpod_deployment.md` for operational details.

### 4.4 Symbol registry (`trainer/market_data/symbol_normalizer.py`)

Enhanced in S32 with:

- `BROKER_PROFILES` — per-broker raw-spelling overrides (e.g. OANDA's `US30_USD`),
  layered before admin/builtin/learned alias lookup.
- `inverse_lookup(canonical)` — all known spellings across builtin + broker + admin +
  learned sources.
- Opt-in `fuzzy=True` on `normalize()`/`resolve()` — difflib-based, applied only after
  every exact/structural path fails, to avoid false positives on genuinely unknown
  instruments.
- Previously-missing aliases: `DJ30`/`US30Cash`/`US30USD` → `US30`;
  `DE40`/`GER40Cash`/`Germany40` → `GER40`.

### 4.5 Reconciliation domains

`ReconciliationDomain` gained `SYMBOLS`, `USER_STATUS`, and
`AUTONOMOUS_TRADING_PERMISSIONS`, all classified `SIGNALIUM_OWNED` — Signalium is the
source of truth for these domains; Trainer only applies changes, never originates them.

## 5. Data & control flow summary

1. **Training**: `trainer.api.SignaliumClient` pulls datasets from Signalium's REST
   API (never Postgres directly) → `dataloaders`/`datasets` → `training.TrainingEngine`
   / `MultiTaskEngine` → `checkpoints` + `MLflowTracker` → `validation.ValidationEngine`
   scores against the leaderboard.
2. **Decisioning**: trained models are wrapped behind `trainer.qde`, which emits signal
   heads; the translator converts heads to execution vocabulary; the observer FSM
   attaches advisory context. Nothing in this path can directly place trades.
3. **Promotion**: `research_lab`'s champion/challenger tracker evaluates candidates
   against the current champion; approval gates and health monitors act as guardrails
   before a model is promoted; the immutable artifact store and signature verification
   (`DeploymentManager.verify_artifact_signatures()`) gate what can actually deploy.
4. **External access**: everything above is reachable only through the Trainer API's
   two-stage trust boundary (origin trust → admin auth), except the explicitly public
   `/api/capabilities/*` discovery surface, and the Research Dashboard which consumes
   the same API surface as any other client.

## 6. CI / quality gates

`.github/workflows/ci.yml` defines the authoritative set of checks that must pass
before merge:

- **Node**: install → typecheck → build (`apps/trainer-api`, dashboard, scripts).
- **Python lint**: black, isort, ruff, mypy (scoped to `trainer/`, not `tests/`).
- **Python test**: pytest with coverage.
- **Docker**: CPU image build smoke test.

Any change to error handling, routing, or domain logic should be validated locally
against these same tool invocations before pushing.

## 7. Related documents

- `docs/architecture.md` — original trainer-core package map and design principles.
- `docs/monorepo_architecture.md` — pnpm workspace layout conventions.
- `docs/qde/unified_qde_architecture.md`, `docs/qde/research_lab_final_architecture.md`
  — deep dives on QDE and Research Lab internals.
- `docs/runpod_deployment.md` — RunPod deployment operations.
- `.agents/memory/s32-platform-trust-capabilities-runpod.md` — implementation notes
  for the S32 changes summarized in §2.1–§4.4.
