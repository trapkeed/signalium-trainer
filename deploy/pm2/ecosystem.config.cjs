"use strict";

// ---------------------------------------------------------------------------
// Signalium Trainer — PM2 Ecosystem Configuration
// Target: Ubuntu 24.04 / Node 22 LTS / PM2 5+
//
// This file is for the signalium-trainer monorepo only.
// It is NOT related to the Signalium platform API.
//
// Override any default via environment variable before calling pm2 start:
//   TRAINER_DIR      - repo root on disk          (default: /opt/signalium-trainer)
//   TRAINER_PORT     - port Express binds to      (default: 8080)
//   TRAINER_LOG_DIR  - writable log directory     (default: /var/log/signalium-trainer)
// ---------------------------------------------------------------------------

const TRAINER_DIR     = process.env.TRAINER_DIR     ?? "/opt/signalium-trainer";
const TRAINER_PORT    = process.env.TRAINER_PORT    ?? "8080";
const TRAINER_LOG_DIR = process.env.TRAINER_LOG_DIR ?? "/var/log/signalium-trainer";

module.exports = {
  apps: [
    {
      // ── Identity ────────────────────────────────────────────────────────
      name:             "signalium-trainer",
      script:           `${TRAINER_DIR}/apps/trainer-api/dist/index.mjs`,
      interpreter:      "node",
      interpreter_args: "--enable-source-maps",

      // ── Working directory ────────────────────────────────────────────────
      // Must be set explicitly so source-map resolution is deterministic
      // regardless of where `pm2 start` is invoked from.
      cwd: TRAINER_DIR,

      // ── Cluster mode — one worker per logical CPU core ───────────────────
      instances: "max",
      exec_mode: "cluster",

      // ── Environment ──────────────────────────────────────────────────────
      // env_file is merged first; inline env below overrides individual keys.
      env_file: `${TRAINER_DIR}/.env`,
      env: {
        NODE_ENV:  "production",
        PORT:      TRAINER_PORT,
        LOG_LEVEL: "info",
      },

      // ── Logging ──────────────────────────────────────────────────────────
      error_file:      `${TRAINER_LOG_DIR}/api-error.log`,
      out_file:        `${TRAINER_LOG_DIR}/api-out.log`,
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      merge_logs:      true,      // single file across all cluster workers

      // ── Reliability ──────────────────────────────────────────────────────
      autorestart:        true,
      max_restarts:       10,     // give up after 10 consecutive crashes
      restart_delay:      5000,   // wait 5 s before each restart
      max_memory_restart: "1G",   // restart a worker if RSS exceeds 1 GB
      watch:              false,  // never watch filesystem in production

      // ── Graceful shutdown ─────────────────────────────────────────────────
      // Matches the 10 000 ms timeout in apps/trainer-api/src/index.ts shutdown().
      kill_timeout:   12000,  // ms PM2 waits before SIGKILL after SIGTERM
      listen_timeout: 8000,   // ms PM2 waits for the process to be ready

      // ── Health check (PM2 5+) ─────────────────────────────────────────────
      // Uncomment once pm2-health-check module is installed.
      // health_check_url:      `http://127.0.0.1:${TRAINER_PORT}/api/healthz`,
      // health_check_interval: 30000,
    },
  ],
};
