#!/usr/bin/env bash
# =============================================================================
# Signalium Trainer — Production Stop Script
# Repo: signalium-trainer (NOT the Signalium platform)
#
# Sends SIGTERM to all PM2 workers and waits for graceful shutdown.
# Usage: ./deploy/scripts/stop.sh
# =============================================================================
set -euo pipefail

TRAINER_DRAIN_TIMEOUT="${TRAINER_DRAIN_TIMEOUT:-15}"  # seconds to wait for workers to drain

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STOP]  $*"; }
fail() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAIL]  $*" >&2; exit 1; }

command -v pm2 >/dev/null 2>&1 || fail "PM2 not found"

# Check if the process is actually running before trying to stop it
if ! pm2 list | grep -q "signalium-trainer"; then
  log "signalium-trainer is not running — nothing to stop."
  exit 0
fi

log "Sending SIGTERM to all signalium-trainer workers (graceful drain up to ${TRAINER_DRAIN_TIMEOUT}s)..."
pm2 stop signalium-trainer --timeout "$TRAINER_DRAIN_TIMEOUT"

log "Saving PM2 process list..."
pm2 save

log "✅ signalium-trainer stopped."
