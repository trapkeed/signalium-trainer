# =============================================================================
# Signalium Trainer API — Node.js production image
# =============================================================================
# Purpose : Builds and runs the Express / Fastify Trainer API (port $PORT).
# Usage   : Built and started automatically by docker-compose.yml.
#           Can also be run standalone:
#
#   docker build -f apps/trainer-api/Dockerfile -t signalium-trainer-api .
#   docker run -p 3000:3000 \
#     --env-file .env \
#     signalium-trainer-api
# =============================================================================

FROM node:22-slim AS builder

WORKDIR /build

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy workspace manifests first for layer caching
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml ./
COPY apps/trainer-api/package.json ./apps/trainer-api/
# Copy any shared packages referenced by the API (add more as needed)
RUN pnpm install --frozen-lockfile

# Copy source and build
COPY apps/trainer-api/ ./apps/trainer-api/
COPY packages/ ./packages/

WORKDIR /build/apps/trainer-api
RUN pnpm run build

# ---------------------------------------------------------------------------
# Runtime image — lean, no dev dependencies
# ---------------------------------------------------------------------------
FROM node:22-slim AS runtime

WORKDIR /app

RUN corepack enable && corepack prepare pnpm@latest --activate

# Copy workspace files for prod install
COPY --from=builder /build/pnpm-workspace.yaml \
                    /build/package.json \
                    /build/pnpm-lock.yaml ./
COPY --from=builder /build/apps/trainer-api/package.json ./apps/trainer-api/

RUN pnpm install --frozen-lockfile --prod

# Copy compiled output
COPY --from=builder /build/apps/trainer-api/dist ./apps/trainer-api/dist

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV NODE_ENV=production

# PORT is required at runtime — pass via env or docker-compose.
# Default 3000 matches .env.example.
EXPOSE 3000

# node:22-slim has no curl/wget — use Node itself to probe the health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "\
const http = require('http'); \
const port = process.env.PORT || 3000; \
http.get('http://localhost:' + port + '/health', r => process.exit(r.statusCode === 200 ? 0 : 1)) \
    .on('error', () => process.exit(1));"

WORKDIR /app/apps/trainer-api
CMD ["node", "--enable-source-maps", "dist/index.mjs"]
