#!/usr/bin/env bash
# =============================================================================
# Signalium Trainer API — Production Start Script
# Usage: ./deploy/scripts/start.sh [--skip-build] [--skip-migrate]
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable via environment — override before calling this script.
# ---------------------------------------------------------------------------
APP_DIR="${APP_DIR:-/opt/signalium}"
LOG_DIR="${LOG_DIR:-/var/log/signalium-trainer}"
ENV_FILE="${ENV_FILE:-${APP_DIR}/.env}"
PM2_ECOSYSTEM="${PM2_ECOSYSTEM:-${APP_DIR}/deploy/pm2/ecosystem.config.cjs}"
APP_PORT="${APP_PORT:-8080}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${APP_PORT}/api/healthz}"
HEALTH_RETRIES="${HEALTH_RETRIES:-15}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-2}"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
SKIP_BUILD=false
SKIP_MIGRATE=false
for arg in "$@"; do
  case "$arg" in
    --skip-build)   SKIP_BUILD=true ;;
    --skip-migrate) SKIP_MIGRATE=true ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [START] $*"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAIL]  $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
log "Running preflight checks..."

[[ -d "$APP_DIR" ]]  || fail "APP_DIR not found: $APP_DIR"
[[ -f "$ENV_FILE" ]] || fail "ENV file not found: $ENV_FILE (copy .env.example → .env)"

# Verify required env vars are present in the env file
for var in PORT DATABASE_URL API_KEY; do
  grep -q "^${var}=" "$ENV_FILE" || fail "Required variable ${var} missing from ${ENV_FILE}"
done

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Verify Node.js 22+
NODE_VERSION=$(node --version 2>/dev/null | sed 's/v//' | cut -d. -f1) || fail "Node.js not found"
[[ "$NODE_VERSION" -ge 22 ]] || fail "Node.js 22+ required (found v${NODE_VERSION})"

# Verify PM2 is available
command -v pm2 >/dev/null 2>&1 || fail "PM2 not found — run: npm install -g pm2"

log "Preflight checks passed (Node v${NODE_VERSION})."

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# ---------------------------------------------------------------------------
# Install / update Node dependencies
# ---------------------------------------------------------------------------
log "Installing Node.js dependencies..."
cd "$APP_DIR"
pnpm install --frozen-lockfile --prod 2>&1 | tail -5

# ---------------------------------------------------------------------------
# Build TypeScript bundle
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == "false" ]]; then
  log "Building TypeScript bundle..."
  pnpm run build 2>&1 | tail -10
  [[ -f "${APP_DIR}/apps/trainer-api/dist/index.mjs" ]] \
    || fail "Build failed — dist/index.mjs not found"
  log "Build complete."
else
  log "Skipping build (--skip-build)."
  [[ -f "${APP_DIR}/apps/trainer-api/dist/index.mjs" ]] \
    || fail "dist/index.mjs not found and --skip-build was set"
fi

# ---------------------------------------------------------------------------
# Database migrations
# ---------------------------------------------------------------------------
if [[ "$SKIP_MIGRATE" == "false" ]]; then
  log "Running database migrations..."
  pnpm --filter "@workspace/db" run push 2>&1 | tail -5
  log "Migrations complete."
else
  log "Skipping migrations (--skip-migrate)."
fi

# ---------------------------------------------------------------------------
# Start PM2
# ---------------------------------------------------------------------------
log "Starting Trainer API via PM2..."
pm2 start "$PM2_ECOSYSTEM"
pm2 save

# ---------------------------------------------------------------------------
# Health check — wait for the API to start accepting requests
# ---------------------------------------------------------------------------
log "Waiting for health endpoint at ${HEALTH_URL}..."
attempt=0
until curl -sf "$HEALTH_URL" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [[ "$attempt" -ge "$HEALTH_RETRIES" ]]; then
    pm2 logs trainer-api --lines 30 --nostream
    fail "Health check failed after ${HEALTH_RETRIES} attempts. See logs above."
  fi
  log "  Attempt ${attempt}/${HEALTH_RETRIES} — waiting ${HEALTH_INTERVAL}s..."
  sleep "$HEALTH_INTERVAL"
done

log "✅ Trainer API is up and healthy."
log "   Run 'pm2 list' to see worker status."
log "   Run 'pm2 logs trainer-api' to tail logs."
