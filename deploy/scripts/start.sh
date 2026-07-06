#!/usr/bin/env bash
# =============================================================================
# Signalium Trainer — Production Start Script
# Repo: signalium-trainer (NOT the Signalium platform)
#
# Usage: ./deploy/scripts/start.sh [--skip-build] [--skip-migrate]
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable via environment — override before calling this script.
# All variables are prefixed TRAINER_ to avoid collisions with platform vars.
# ---------------------------------------------------------------------------
TRAINER_DIR="${TRAINER_DIR:-/opt/signalium-trainer}"
TRAINER_LOG_DIR="${TRAINER_LOG_DIR:-/var/log/signalium-trainer}"
TRAINER_ENV_FILE="${TRAINER_ENV_FILE:-${TRAINER_DIR}/.env}"
TRAINER_PM2_ECOSYSTEM="${TRAINER_PM2_ECOSYSTEM:-${TRAINER_DIR}/deploy/pm2/ecosystem.config.cjs}"
TRAINER_PORT="${TRAINER_PORT:-8080}"
TRAINER_HEALTH_URL="${TRAINER_HEALTH_URL:-http://127.0.0.1:${TRAINER_PORT}/api/healthz}"
TRAINER_HEALTH_RETRIES="${TRAINER_HEALTH_RETRIES:-15}"
TRAINER_HEALTH_INTERVAL="${TRAINER_HEALTH_INTERVAL:-2}"

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

[[ -d "$TRAINER_DIR" ]]      || fail "TRAINER_DIR not found: $TRAINER_DIR"
[[ -f "$TRAINER_ENV_FILE" ]] || fail "Env file not found: $TRAINER_ENV_FILE (copy .env.example → .env)"

# Verify required env vars are present in the env file
for var in PORT DATABASE_URL API_KEY; do
  grep -q "^${var}=" "$TRAINER_ENV_FILE" || fail "Required variable ${var} missing from ${TRAINER_ENV_FILE}"
done

# Ensure log directory exists
mkdir -p "$TRAINER_LOG_DIR"

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
source "$TRAINER_ENV_FILE"
set +a

# ---------------------------------------------------------------------------
# Install / update Node dependencies
# ---------------------------------------------------------------------------
log "Installing Node.js dependencies..."
cd "$TRAINER_DIR"
pnpm install --frozen-lockfile --prod 2>&1 | tail -5

# ---------------------------------------------------------------------------
# Build TypeScript bundle
# ---------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == "false" ]]; then
  log "Building TypeScript bundle..."
  pnpm run build 2>&1 | tail -10
  [[ -f "${TRAINER_DIR}/apps/trainer-api/dist/index.mjs" ]] \
    || fail "Build failed — dist/index.mjs not found"
  log "Build complete."
else
  log "Skipping build (--skip-build)."
  [[ -f "${TRAINER_DIR}/apps/trainer-api/dist/index.mjs" ]] \
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
log "Starting signalium-trainer via PM2..."
pm2 start "$TRAINER_PM2_ECOSYSTEM"
pm2 save

# ---------------------------------------------------------------------------
# Health check — wait for the API to start accepting requests
# ---------------------------------------------------------------------------
log "Waiting for health endpoint at ${TRAINER_HEALTH_URL}..."
attempt=0
until curl -sf "$TRAINER_HEALTH_URL" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [[ "$attempt" -ge "$TRAINER_HEALTH_RETRIES" ]]; then
    pm2 logs signalium-trainer --lines 30 --nostream
    fail "Health check failed after ${TRAINER_HEALTH_RETRIES} attempts. See logs above."
  fi
  log "  Attempt ${attempt}/${TRAINER_HEALTH_RETRIES} — waiting ${TRAINER_HEALTH_INTERVAL}s..."
  sleep "$TRAINER_HEALTH_INTERVAL"
done

log "✅ signalium-trainer is up and healthy."
log "   Run 'pm2 list' to see worker status."
log "   Run 'pm2 logs signalium-trainer' to tail logs."
