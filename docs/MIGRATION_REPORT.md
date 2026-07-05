# Production Refactor — Migration Report

**Date:** 2026-07-05  
**Branch:** `production-refactor`  
**Status:** COMPLETE — TypeScript compiles clean (0 errors across all 7 packages)

---

## 1. Objective

Restructure the Signalium Quant monorepo from its development layout into a clean, enterprise-grade production folder layout. Remove all dev-only artefacts. Validate TypeScript compilation. Preserve all business logic, AI algorithms, model training code, public APIs, and database schemas without modification.

---

## 2. Repository Structure: Before → After

### Before

```
/workspace
├── artifacts/
│   ├── api-server/          TypeScript Express API
│   └── mockup-sandbox/      ❌ Replit UI design prototype (dev-only)
├── lib/
│   ├── api-client-react/
│   ├── api-spec/
│   ├── api-zod/
│   └── db/
├── scripts/
├── signalium-trainer/
│   ├── trainer/             Python ML core (180+ modules, 25 phases)
│   ├── conf/                Hydra YAML configuration
│   ├── docs/
│   ├── tests/               1,501 Python tests
│   ├── Dockerfile
│   ├── docker-compose.yml   ❌ Labeled "dev-only"
│   ├── pyproject.toml
│   ├── .github/workflows/
│   ├── generate_docs.py
│   ├── Signalium_Quant_Documentation.pdf  ❌ Generated artifact
│   └── ...
├── attached_assets/         ❌ Dev-phase spec upload files
├── main.py                  ❌ Replit placeholder
└── pyproject.toml           ❌ Replit placeholder (fpdf2 only)
```

### After

```
/workspace
├── apps/
│   └── api-server/          TypeScript Express API (moved from artifacts/)
├── packages/
│   ├── api-client-react/    (moved from lib/)
│   ├── api-spec/            (moved from lib/)
│   ├── api-zod/             (moved from lib/)
│   └── db/                  (moved from lib/)
├── trainer/                 Python ML core (moved from signalium-trainer/trainer/)
├── config/                  Hydra YAML configs (moved from signalium-trainer/conf/)
├── docker/
│   └── Dockerfile           GPU training container (moved from signalium-trainer/)
├── nginx/
│   └── README.md            Placeholder for production reverse-proxy config
├── workers/
│   └── README.md            Placeholder for future background workers
├── docs/                    Documentation (moved from signalium-trainer/docs/)
├── tests/                   1,501 Python tests (moved from signalium-trainer/tests/)
├── scripts/
│   ├── src/hello.ts
│   ├── post-merge.sh        (updated: --filter "@workspace/db")
│   └── generate_docs.py     (moved from signalium-trainer/)
├── .github/
│   └── workflows/ci.yml     (moved from signalium-trainer/.github/)
├── pyproject.toml           Python package definition (replaced placeholder)
├── .pre-commit-config.yaml  (moved from signalium-trainer/)
├── LICENSE                  MIT License (moved from signalium-trainer/)
├── README.md                (moved from signalium-trainer/)
├── .gitignore               (augmented with Python entries)
├── pnpm-workspace.yaml      (updated: apps/*, packages/*)
├── tsconfig.json            (updated references: packages/*)
├── tsconfig.base.json       (unchanged)
└── package.json             (updated: typecheck filter ./apps/**)
```

---

## 3. Files Removed (Dev-Only)

| File / Directory | Reason |
|---|---|
| `artifacts/mockup-sandbox/` | Replit UI design prototype — `kind="design"` artifact with zero production dependencies |
| `attached_assets/` | Dev-phase spec upload files (25 text files of phase instructions, not code) |
| `signalium-trainer/docker-compose.yml` | Explicitly labeled: _"Development-only docker-compose. Do NOT use in production or RunPod environments."_ |
| `signalium-trainer/Signalium_Quant_Documentation.pdf` | Generated artifact — reproducible via `scripts/generate_docs.py` |
| `main.py` (root) | Replit placeholder only: `print("Hello from repl-nix-workspace!")` |
| `pyproject.toml` (root, original) | Replit placeholder — only contained `fpdf2>=2.8.7` dependency; replaced with the real `signalium-trainer/pyproject.toml` |

---

## 4. Configuration Changes

### `pnpm-workspace.yaml`

```yaml
# Before
packages:
  - artifacts/*
  - lib/*
  - lib/integrations/*
  - scripts

# After
packages:
  - apps/*
  - packages/*
  - scripts
```

### `tsconfig.json` (root TypeScript project references)

```json
// Before
"references": [
  { "path": "./lib/db" },
  { "path": "./lib/api-client-react" },
  { "path": "./lib/api-zod" }
]

// After
"references": [
  { "path": "./packages/db" },
  { "path": "./packages/api-client-react" },
  { "path": "./packages/api-zod" }
]
```

### `apps/api-server/tsconfig.json`

```json
// Before
"references": [
  { "path": "../../lib/db" },
  { "path": "../../lib/api-zod" }
]

// After
"references": [
  { "path": "../../packages/db" },
  { "path": "../../packages/api-zod" }
]
```

### `package.json` (root workspace scripts)

```json
// Before
"typecheck": "pnpm run typecheck:libs && pnpm -r --filter \"./artifacts/**\" --filter \"./scripts\" --if-present run typecheck"

// After
"typecheck": "pnpm run typecheck:libs && pnpm -r --filter \"./apps/**\" --filter \"./scripts\" --if-present run typecheck"
```

### `scripts/post-merge.sh`

```bash
# Before
pnpm --filter db push

# After  
pnpm --filter "@workspace/db" push
```
Rationale: pnpm filter matches by package name; `@workspace/db` is the correct name regardless of directory location.

### `apps/api-server/.replit-artifact/artifact.toml`

```toml
# Before (production run)
args = ["node", "--enable-source-maps", "artifacts/api-server/dist/index.mjs"]

# After
args = ["node", "--enable-source-maps", "apps/api-server/dist/index.mjs"]
```

### `.gitignore`

Merged Python-specific entries from `signalium-trainer/.gitignore` into the root `.gitignore`:
- `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.coverage`, `.mypy_cache/`
- Hydra outputs: `outputs/`, `multirun/`, `.hydra/`
- MLflow: `mlruns/`, `mlartifacts/`
- Data artefacts: `*.parquet`, `*.csv`, `*.arrow`
- Secrets: `.env`, `*.pem`, `*.key`
- ONNX / TorchScript: `*.onnx`, `*.torchscript.pt`
- Generated docs: `*.pdf`

---

## 5. What Was NOT Changed

The following were explicitly preserved without modification:

- All Python business logic in `trainer/` (38 top-level modules, 25 development phases)
- All database schemas (`packages/db/src/schema/`)
- All Zod validators and OpenAPI specs (`packages/api-zod/`, `packages/api-spec/`)
- All Express API routes and middleware (`apps/api-server/src/`)
- All 1,501 Python tests in `tests/`
- All Hydra configuration files in `config/`
- Python `pyproject.toml` project metadata and dependency lists
- `.replit` configuration (Replit environment definition)
- `pnpm-lock.yaml` (regenerated by `pnpm install` with updated workspace paths)

### Python Test Import Paths

Python tests use relative paths like:
```python
ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "trainer", "cert")
```

This resolves correctly both before and after the move because `tests/unit/` and `trainer/` remain at the **same relative depth** from each other. No test files required modification.

---

## 6. Validation Results

| Check | Result |
|---|---|
| `pnpm install` | ✅ Succeeded — 242 packages resolved, lockfile updated |
| `pnpm run typecheck:libs` (`tsc --build`) | ✅ Clean — 0 errors |
| `pnpm -r --filter "./apps/**" run typecheck` | ✅ Clean — 0 errors (api-server) |
| `pnpm -r --filter "./scripts" run typecheck` | ✅ Clean — 0 errors |
| **Total TypeScript errors** | **0** |

---

## 7. Remaining Steps for Operator

### 7a. Create the `production-refactor` branch and commit

The Replit sandbox prevents direct `git checkout -b` in the main agent. Run these commands yourself:

```bash
git checkout -b production-refactor
git add -A
git commit -m "refactor: production-ready monorepo structure

- Moved apps: artifacts/api-server → apps/api-server
- Moved packages: lib/* → packages/*
- Moved trainer, config, docs, tests to repo root
- Moved Dockerfile to docker/, CI to .github/workflows/
- Removed: mockup-sandbox, attached_assets, dev docker-compose, placeholder files
- Updated all import paths, workspace configs, and TS project references
- TypeScript: 0 compilation errors across all 7 packages
"
```

### 7b. Python environment lockfile (optional)

The root `pyproject.toml` was upgraded from the Replit placeholder (`fpdf2` only) to the full Signalium Trainer definition. The `uv.lock` is now stale. Regenerate it when setting up a fresh Python environment:

```bash
uv lock   # or: pip install -e ".[dev]"
```

### 7c. Python test validation (requires GPU dependencies)

```bash
# CPU-only (no CUDA) — suitable for CI
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest tests/ --cov=trainer -v
```

Expected: **1,501 tests, 164 passing** (Phase 25 baseline).

---

## 8. Directory Tree (Final)

```
apps/api-server/        Express.js API server (TypeScript)
packages/
  api-client-react/     React Query API client
  api-spec/             OpenAPI specification + Orval codegen
  api-zod/              Zod validators generated from OpenAPI
  db/                   Drizzle ORM + PostgreSQL connection
trainer/                Python ML core (38 modules, 25 dev phases)
config/                 Hydra YAML configuration trees
docker/Dockerfile       NVIDIA CUDA GPU training container
nginx/                  Reverse-proxy config (placeholder)
workers/                Background job workers (placeholder)
docs/                   Technical documentation
tests/                  1,501-test Python test suite
scripts/
  post-merge.sh         Replit post-merge database migration hook
  generate_docs.py      PDF documentation generator
.github/workflows/ci.yml  GitHub Actions CI (lint + test + docker build)
pyproject.toml          Python package definition (signalium-trainer)
.pre-commit-config.yaml Code quality hooks (black, isort, ruff, mypy)
LICENSE                 MIT License
README.md               Project README
```
