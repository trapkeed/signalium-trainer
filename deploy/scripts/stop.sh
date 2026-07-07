#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# stop.sh — Gracefully stop the Signalium Trainer API PM2 process
# -----------------------------------------------------------------------------
# Usage (run from repo root):
#   bash deploy/scripts/stop.sh
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

PM2="$(command -v pm2 2>/dev/null || true)"
[[ -z "$PM2" ]] && error "PM2 not found. Is it installed?"

info "Stopping signalium-trainer-api..."
"$PM2" stop signalium-trainer-api

"$PM2" save
info "Process stopped."
"$PM2" list
