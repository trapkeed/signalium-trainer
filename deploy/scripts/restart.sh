#!/usr/bin/env bash
# =============================================================================
# Signalium Trainer — Production Restart / Zero-Downtime Reload Script
# Repo: signalium-trainer (NOT the Signalium platform)
#
# Default behaviour: zero-downtime rolling reload (pm2 reload).
# Use --hard to do a full stop → build → migrate → start cycle instead.
#
# Usage:
#   ./deploy/scripts/restart.sh                        # zero-downtime reload
#   ./deploy/scripts/restart.sh --hard                 # full redeploy
#   ./deploy/scripts/restart.sh --hard --skip-migrate  # redeploy, skip DB step
# =============================================================================
set -euo pipefail

TRAINER_DIR="${TRAINER_DIR:-/opt/signalium-trainer}"
TRAINER_PORT="${TRAINER_PORT:-8080}"
TRAINER_ENV_FILE="${TRAINER_ENV_FILE:-${TRAINER_DIR}/.env}"
TRAINER_PM2_ECOSYSTEM="${TRAINER_PM2_ECOSYSTEM:-${TRAINER_DIR}/deploy/pm2/ecosystem.config.cjs}"
TRAINER_HEALTH_URL="${TRAINER_HEALTH_URL:-http://127.0.0.1:${TRAINER_PORT}/api/healthz}"
TRAINER_HEALTH_RETRIES="${TRAINER_HEALTH_RETRIES:-15}"
TRAINER_HEALTH_INTERVAL="${TRAINER_HEALTH_INTERVAL:-2}"

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

# Executable discovery — no hardcoded paths
PM2_BIN="$(command -v pm2 2>/dev/null)" || fail "pm2 not found in PATH"

[[ -f "$TRAINER_ENV_FILE" ]] || fail "Env file not found: $TRAINER_ENV_FILE"

# ---------------------------------------------------------------------------
# Hard redeploy — full stop / build / migrate / start cycle
# ---------------------------------------------------------------------------
if [[ "$HARD" == "true" ]]; then
  log "Hard restart requested — running full redeploy of signalium-trainer."

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  log "Pulling latest code from origin..."
  cd "$TRAINER_DIR"
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
log "Zero-downtime reload of signalium-trainer workers..."
"$PM2_BIN" reload signalium-trainer --update-env

log "Saving PM2 process list..."
"$PM2_BIN" save

# ---------------------------------------------------------------------------
# Post-reload health check
# ---------------------------------------------------------------------------
log "Verifying health endpoint at ${TRAINER_HEALTH_URL}..."
attempt=0
until curl -sf "$TRAINER_HEALTH_URL" >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [[ "$attempt" -ge "$TRAINER_HEALTH_RETRIES" ]]; then
    "$PM2_BIN" logs signalium-trainer --lines 20 --nostream
    fail "Health check failed after reload. See logs above."
  fi
  log "  Attempt ${attempt}/${TRAINER_HEALTH_RETRIES} — waiting ${TRAINER_HEALTH_INTERVAL}s..."
  sleep "$TRAINER_HEALTH_INTERVAL"
done

log "✅ Reload complete — all signalium-trainer workers healthy."
log "   PM2 status:"
"$PM2_BIN" list
