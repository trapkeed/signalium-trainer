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

  # Restore devDependencies so build tools (esbuild, vite) are available.
  # NODE_ENV= overrides any production env var that would cause pnpm to skip them.
  # --no-frozen-lockfile avoids failures when the lockfile was generated on a
  # different pnpm version than what is installed on the server.
  info "Installing dependencies (including devDeps for build)..."
  NODE_ENV= "$PNPM" install --no-frozen-lockfile

  info "Rebuilding trainer-api..."
  "$PNPM" --filter @workspace/trainer-api run build

  info "Rebuilding research-dashboard..."
  "$PNPM" --filter @workspace/research-dashboard run build

  # Re-prune to production-only modules now that the builds are done.
  info "Pruning devDependencies..."
  "$PNPM" install --prod --ignore-scripts
fi

# ---------------------------------------------------------------------------
# Reload (graceful) — in fork mode this is equivalent to restart
# ---------------------------------------------------------------------------
info "Reloading signalium-trainer-api..."
"$PM2" reload signalium-trainer-api

"$PM2" save
info "Reload complete."
"$PM2" list
