#!/usr/bin/env bash
# =============================================================================
# Signalium Trainer API — Production Restart / Zero-Downtime Reload Script
#
# Default behaviour: zero-downtime rolling reload (pm2 reload).
# Use --hard to do a full stop → build → migrate → start cycle instead.
#
# Usage:
#   ./deploy/scripts/restart.sh            # zero-downtime reload (recommended)
#   ./deploy/scripts/restart.sh --hard     # full redeploy with build + migrate
#   ./deploy/scripts/restart.sh --hard --skip-migrate
# =============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/signalium}"
APP_PORT="${APP_PORT:-8080}"
ENV_FILE="${ENV_FILE:-${APP_DIR}/.env}"
PM2_ECOSYSTEM="${PM2_ECOSYSTEM:-${APP_DIR}/deploy/pm2/ecosystem.config.cjs}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${APP_PORT}/api/healthz}"
HEALTH_RETRIES="${HEALTH_RETRIES:-15}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-2}"

HARD=false
EXTRA_FLAGS=()
for arg in "$@"; do
  case "$arg" in
    --hard)         HARD=true ;;
    --skip-build)   EXTRA_FLAGS+=("--skip-build") ;;
    --skip-migrate) EXTRA_FLAGS+=("--skip-migrate") ;;
  esac
done

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [RESTART] $*"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAIL]    $*" >&2; exit 1; }

command -v pm2 >/dev/null 2>&1 || fail "PM2 not found"
[[ -f "$ENV_FILE" ]] || fail "ENV file not found: $ENV_FILE"

# ---------------------------------------------------------------------------
# Hard redeploy — full stop / build / migrate / start cycle
# ---------------------------------------------------------------------------
if [[ "$HARD" == "true" ]]; then
  log "Hard restart requested — running full redeploy."

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  log "Pulling latest code from origin..."
  cd "$APP_DIR"
  git fetch origin
  git pull origin "$(git rev-parse --abbrev-ref HEAD)"

  log "Stopping current processes..."
  "${SCRIPT_DIR}/stop.sh"

  log "Starting fresh processes..."
  "${SCRIPT_DIR}/start.sh" "${EXTRA_FLAGS[@]}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Zero-downtime rolling reload (default)
# PM2 spawns new workers, waits for them to be ready, then kills old ones.
# In-flight requests on old workers complete before those workers die.
# ---------------------------------------------------------------------------
log "Zero-downtime reload of trainer-api workers..."
pm2 reload trainer-api --update-env

log "Saving PM2 process list..."
pm2 save

# ---------------------------------------------------------------------------
# Post-reload health check
# ---------------------------------------------------------------------------
log "Verifying health endpoint at ${HEALTH_URL}..."
attempt=0
until curl -sf "$HEALTH_URL" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [[ "$attempt" -ge "$HEALTH_RETRIES" ]]; then
    pm2 logs trainer-api --lines 20 --nostream
    fail "Health check failed after reload. See logs above."
  fi
  log "  Attempt ${attempt}/${HEALTH_RETRIES} — waiting ${HEALTH_INTERVAL}s..."
  sleep "$HEALTH_INTERVAL"
done

log "✅ Reload complete — all workers healthy."
log "   PM2 status:"
pm2 list
