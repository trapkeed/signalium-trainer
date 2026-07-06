'use strict';

/**
 * PM2 Ecosystem Config — Signalium Trainer API
 *
 * Usage:
 *   pm2 start deploy/pm2/ecosystem.config.cjs --env production
 *   pm2 save
 *   pm2 startup
 *
 * The app must be built before starting:
 *   pnpm --filter @workspace/trainer-api run build
 */

module.exports = {
  apps: [
    {
      name: 'signalium-trainer-api',

      // Entry point — built by esbuild
      script: './dist/index.mjs',

      // Absolute path to the API package inside the repo checkout.
      // Change this to wherever the repo is cloned on the server.
      cwd: '/opt/signalium-trainer/apps/trainer-api',

      // Run as a plain Node.js process (ESM bundle — no cluster needed)
      interpreter: 'node',
      interpreter_args: '--enable-source-maps',
      exec_mode: 'fork',
      instances: 1,

      // dotenv/config loads .env from DOTENV_CONFIG_PATH.
      // ../../.env resolves to the repo root .env when cwd is apps/trainer-api.
      env: {
        NODE_ENV: 'production',
        DOTENV_CONFIG_PATH: '../../.env',
      },
      env_production: {
        NODE_ENV: 'production',
        DOTENV_CONFIG_PATH: '../../.env',
      },

      // Logging
      error_file: '/var/log/signalium-trainer/api-error.log',
      out_file:   '/var/log/signalium-trainer/api-out.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',

      // Stability
      max_memory_restart: '512M',
      restart_delay: 3000,
      max_restarts: 10,
      min_uptime: '5s',

      // Never watch files in production
      watch: false,

      // Graceful shutdown — matches SIGTERM handler in index.ts
      kill_timeout: 10000,
      wait_ready: false,
      listen_timeout: 8000,
    },
  ],
};
