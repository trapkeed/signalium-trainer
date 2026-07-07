#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# restart.sh — Zero-downtime reload of the Signalium Trainer API
# -----------------------------------------------------------------------------
# Uses `pm2 reload` (graceful rolling restart).
# Falls back to `pm2 restart` if reload is not applicable (fork mode).
#
# Usage (run from repo root):
#   bash deploy/scripts/restart.sh
# Optional rebuild before restart:
#   REBUILD=1 bash deploy/scripts/restart.sh
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

PM2="$(command -v pm2 2>/dev/null || true)"
PNPM="$(command -v pnpm 2>/dev/null || true)"
[[ -z "$PM2" ]] && error "PM2 not found. Is it installed?"

# ---------------------------------------------------------------------------
# Optional rebuild
# ---------------------------------------------------------------------------
if [[ "${REBUILD:-0}" == "1" ]]; then
  [[ -z "$PNPM" ]] && error "pnpm not found — required for REBUILD=1."
  info "Rebuilding trainer-api before restart..."
  "$PNPM" --filter @workspace/trainer-api run build
fi

# ---------------------------------------------------------------------------
# Reload (graceful) — in fork mode this is equivalent to restart
# ---------------------------------------------------------------------------
info "Reloading signalium-trainer-api..."
"$PM2" reload signalium-trainer-api

"$PM2" save
info "Reload complete."
"$PM2" list
