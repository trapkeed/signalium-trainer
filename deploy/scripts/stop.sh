#!/usr/bin/env bash
# =============================================================================
# Signalium Trainer API — Production Stop Script
# Sends SIGTERM to all PM2 workers and waits for graceful shutdown.
# Usage: ./deploy/scripts/stop.sh
# =============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/signalium}"
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-15}"   # seconds to wait for workers to drain

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STOP]  $*"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAIL]  $*" >&2; exit 1; }

command -v pm2 >/dev/null 2>&1 || fail "PM2 not found"

# Check if the process is actually running before trying to stop it
if ! pm2 list | grep -q "trainer-api"; then
  log "trainer-api is not running — nothing to stop."
  exit 0
fi

log "Sending SIGTERM to all trainer-api workers (graceful drain up to ${DRAIN_TIMEOUT}s)..."
pm2 stop trainer-api --timeout "$DRAIN_TIMEOUT"

log "Saving PM2 process list..."
pm2 save

log "✅ Trainer API stopped."
