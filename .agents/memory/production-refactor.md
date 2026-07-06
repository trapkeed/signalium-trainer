---
name: Production Refactor Layout
description: Canonical post-refactor monorepo layout — where everything lives after the production-refactor restructuring
---

# Production monorepo layout (after production-refactor)

## TypeScript
- Apps: `apps/trainer-api/` (was `artifacts/api-server/`, renamed via Phase 5)
- Packages: `packages/{api-client-react,api-spec,api-zod,db}/` (was `lib/*`)
- pnpm-workspace.yaml: `apps/*`, `packages/*`, `scripts`
- Root tsconfig.json references: `./packages/db`, `./packages/api-client-react`, `./packages/api-zod`
- api-server tsconfig references: `../../packages/db`, `../../packages/api-zod`

## Python
- Trainer package: `trainer/` (was `signalium-trainer/trainer/`)
- Hydra configs: `config/` (was `signalium-trainer/conf/`)
- Docs: `docs/`; Tests: `tests/`; Docker: `docker/Dockerfile`
- pyproject.toml at root is the real one (signalium-trainer's, replaced the fpdf2 placeholder)
- `[tool.setuptools.packages.find] where = ["."] include = ["trainer*"]` — correct since trainer/ is at root

## Python test imports
- Tests use `os.path.join(__file__, "..", "..", "trainer", ...)` — still correct, depth unchanged

## Removed (dev-only)
- `artifacts/mockup-sandbox/` (Replit design tool, kind="design")
- `attached_assets/` (dev phase spec uploads)
- `signalium-trainer/docker-compose.yml` (explicitly labeled dev-only)
- `main.py` (root Replit placeholder)
- Root `pyproject.toml` placeholder (fpdf2 only) — replaced by real one

**Why:** Enterprise production layout; `artifacts/` and `lib/` are Replit-convention names, not industry-standard.
**How to apply:** Any new TypeScript app goes in `apps/`, any new shared library goes in `packages/`. Python workers go in `workers/`.
