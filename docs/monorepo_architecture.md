# Monorepo Architecture

## Overview

Signalium Quant is a hybrid monorepo managed with **pnpm workspaces** (TypeScript) and **setuptools** (Python).

---

## Repository Layout

```
signalium-trainer/
├── apps/
│   └── trainer-api/          TypeScript Express REST API (@workspace/trainer-api)
├── packages/
│   ├── api-spec/             OpenAPI 3.1 spec + Orval codegen config
│   ├── api-zod/              Zod validators (generated from OpenAPI)
│   ├── api-client-react/     React Query hooks (generated from OpenAPI)
│   └── db/                   Drizzle ORM + PostgreSQL connection pool
├── scripts/
│   ├── post-merge.sh         Replit post-merge DB migration hook
│   └── generate_docs.py      PDF documentation generator
├── trainer/                  Python ML training framework (38 modules)
├── config/                   Hydra YAML configuration trees
├── tests/                    1,501-test pytest suite
├── docs/                     Technical documentation (60 files)
├── docker/
│   └── Dockerfile            NVIDIA CUDA 12.4 GPU training container
├── deploy/
│   ├── pm2/                  PM2 process manager config
│   ├── backup/               DB + checkpoint backup scripts
│   ├── restore/              Point-in-time restore procedures
│   └── monitoring/           Health check and alerting
├── infrastructure/
│   ├── terraform/            Cloud resource provisioning (IaC)
│   ├── ansible/              Server configuration management
│   └── cloud-init/           VM bootstrap scripts
├── nginx/                    Reverse-proxy configuration
├── workers/                  Background job workers (reserved)
├── .github/workflows/ci.yml  GitHub Actions CI (Node + Python + Docker)
├── pyproject.toml            Python package definition
├── pnpm-workspace.yaml       pnpm workspace config
├── tsconfig.json             TypeScript project references
└── tsconfig.base.json        Shared TypeScript compiler options
```

---

## TypeScript Workspace Dependency Graph

```
@workspace/api-spec (packages/api-spec/)
       │
       ├─[codegen]──► @workspace/api-zod          (Zod validators)
       └─[codegen]──► @workspace/api-client-react  (React Query hooks)

@workspace/trainer-api (apps/trainer-api/)
       ├─[depends]──► @workspace/api-zod
       └─[depends]──► @workspace/db

@workspace/db (packages/db/)
       └─[depends]──► drizzle-orm, pg (external catalog)

@workspace/scripts (scripts/)
       └─[no workspace deps]
```

**No circular dependencies.**

---

## Python Package Dependency Graph

```
trainer/                     (root package, installed as `signalium-trainer`)
├── trainer.cli              ──► trainer.config, trainer.training, trainer.models
├── trainer.api              ──► trainer.models, trainer.registry, trainer.checkpoints
├── trainer.training         ──► trainer.models, trainer.losses, trainer.metrics,
│                                trainer.dataloaders, trainer.checkpoints
├── trainer.models           ──► torch, onnx, pydantic
├── trainer.config           ──► hydra-core, omegaconf, pydantic-settings
├── trainer.mlops            ──► mlflow
├── trainer.cert             ──► trainer.evaluation, trainer.registry, trainer.research
└── trainer.ensemble         ──► trainer.models, trainer.evaluation
```

External runtime dependencies: `torch`, `torchmetrics`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `mlflow`, `hydra-core`, `fastapi`, `uvicorn`, `pydantic`, `loguru`, `onnx`, `onnxruntime`, `typer`

---

## API Request Flow

```
Internet
   │
   ▼
NGINX (nginx/)          ← TLS termination, rate limiting, static serving
   │
   ▼
Trainer API             ← apps/trainer-api/ (Node.js / Express.js)
apps/trainer-api/       ← PORT=8080, pino structured logging
   │
   ├──► packages/api-zod     Request/response validation (Zod)
   └──► packages/db          Database queries (Drizzle ORM → PostgreSQL)
```

---

## CI/CD Pipeline

```
git push / pull_request
        │
        ▼
GitHub Actions (.github/workflows/ci.yml)
        │
        ├── node-install     pnpm install --frozen-lockfile
        ├── node-typecheck   pnpm run typecheck (tsc --build)
        ├── node-build       pnpm run build (esbuild bundle)
        ├── lint             black + isort + ruff + mypy
        ├── test             pytest (1,501 tests, CPU torch)
        └── docker-build     docker build --file docker/Dockerfile .
```

---

## Deployment Architectures

### Ubuntu VPS (PM2 + NGINX)
```
VPS
├── NGINX          Reverse proxy + TLS (nginx/)
├── PM2            Process manager (deploy/pm2/)
│   └── trainer-api (apps/trainer-api/dist/index.mjs)
├── PostgreSQL     Database
└── Python trainer docker/Dockerfile (GPU optional)
```

### RunPod GPU
```
RunPod Pod
├── docker/Dockerfile   CUDA 12.4 + Python 3.12
├── trainer cli         ENTRYPOINT ["trainer"]
├── MLflow              Experiment tracking (:5000)
└── volumes             /app/data, /app/checkpoints, /app/exports
```

### Future Kubernetes
```
Namespace: signalium
├── Deployment: trainer-api    (apps/trainer-api/)
├── Deployment: mlflow-server
├── StatefulSet: postgres
├── PersistentVolume: checkpoints
├── ConfigMap: hydra config
└── Secret: DATABASE_URL, SIGNALIUM_API_KEY
```
