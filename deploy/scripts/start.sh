#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# start.sh — Install, build, and start the Signalium Trainer API via PM2
# -----------------------------------------------------------------------------
# Usage (run from repo root):
#   bash deploy/scripts/start.sh
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || error "'$1' not found. Please install it."
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
require_cmd node
require_cmd pnpm

PNPM="$(command -v pnpm)"
PM2="$(command -v pm2 2>/dev/null || true)"

if [[ -z "$PM2" ]]; then
  info "PM2 not found — installing globally..."
  npm install -g pm2
  PM2="$(command -v pm2)"
fi

NODE_VERSION=$(node --version)
info "Node  : $NODE_VERSION"
info "pnpm  : $($PNPM --version)"
info "PM2   : $($PM2 --version)"
info "Root  : $REPO_ROOT"

# ---------------------------------------------------------------------------
# Ensure .env exists
# ---------------------------------------------------------------------------
if [[ ! -f ".env" ]]; then
  error ".env not found at repo root ($REPO_ROOT). Create it before starting."
fi

# ---------------------------------------------------------------------------
# Install dependencies (production + devDependencies needed for build)
# ---------------------------------------------------------------------------
info "Installing dependencies..."
# NODE_ENV= ensures devDependencies (esbuild, vite) are not skipped even if
# the shell already has NODE_ENV=production set.
# --no-frozen-lockfile avoids failures when pnpm version differs from the
# machine that last regenerated the lockfile.
NODE_ENV= "$PNPM" install --no-frozen-lockfile

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
info "Building trainer-api..."
"$PNPM" --filter @workspace/trainer-api run build

info "Building research-dashboard..."
"$PNPM" --filter @workspace/research-dashboard run build

# ---------------------------------------------------------------------------
# Prune dev dependencies (production-only node_modules)
# ---------------------------------------------------------------------------
info "Pruning dev dependencies..."
"$PNPM" install --prod --ignore-scripts

# ---------------------------------------------------------------------------
# Start via PM2
# ---------------------------------------------------------------------------
info "Starting signalium-trainer-api via PM2..."
"$PM2" start deploy/pm2/ecosystem.config.cjs --env production

"$PM2" save
info "PM2 process list:"
"$PM2" list

info "Done. To persist PM2 across reboots: pm2 startup && pm2 save"
