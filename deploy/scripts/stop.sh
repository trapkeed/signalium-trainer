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

# Executable discovery — no hardcoded paths
PM2_BIN="$(command -v pm2 2>/dev/null)" || fail "pm2 not found in PATH"

# Check if the process is actually running before trying to stop it
if ! "$PM2_BIN" list | grep -q "signalium-trainer"; then
  log "signalium-trainer is not running — nothing to stop."
  exit 0
fi

log "Sending SIGTERM to all signalium-trainer workers (graceful drain up to ${TRAINER_DRAIN_TIMEOUT}s)..."
"$PM2_BIN" stop signalium-trainer --timeout "$TRAINER_DRAIN_TIMEOUT"

log "Saving PM2 process list..."
"$PM2_BIN" save

log "✅ signalium-trainer stopped."
