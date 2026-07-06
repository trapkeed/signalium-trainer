# Signalium Trainer — Production Deployment Guide

Target platform: **Ubuntu 24.04 LTS / Node.js 22 / PM2 / NGINX / PostgreSQL 16**

---

## 1. Prerequisites

Install the following on the production server before running any deployment script.

### System packages

```bash
sudo apt update
sudo apt install -y nginx postgresql-client-16 git curl build-essential
```

### Node.js 22 (NodeSource — works alongside nvm/fnm)

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # must print v22.x.x
```

### PM2 and pnpm (global)

```bash
sudo npm install -g pm2@latest
sudo npm install -g pnpm@latest
pm2 --version
pnpm --version
```

> **nvm / fnm users:** If Node.js is installed per-user via nvm or fnm, add the
> node bin directory to `PATH` inside `/opt/signalium-trainer/.env` so systemd
> can resolve `node` and `pm2` at service start time.
>
> ```
> PATH=/home/signalium-trainer/.nvm/versions/node/v22.x.x/bin:/usr/local/bin:/usr/bin:/bin
> ```

---

## 2. Linux User

The API runs as a dedicated, non-login OS user with no shell access.

```bash
sudo useradd \
  --system \
  --shell /usr/sbin/nologin \
  --comment "Signalium Trainer service account" \
  signalium-trainer
```

Verify:

```bash
id signalium-trainer
```

---

## 3. Required Runtime Directories

| Directory | Purpose | Owner |
|---|---|---|
| `/opt/signalium-trainer` | Repository root (cloned here) | `signalium-trainer` |
| `/var/log/signalium-trainer` | PM2 log files | `signalium-trainer` |
| `/var/lib/signalium-trainer` | Future persistent state | `signalium-trainer` |

### Create and set permissions

```bash
sudo mkdir -p \
  /opt/signalium-trainer \
  /var/log/signalium-trainer \
  /var/lib/signalium-trainer

sudo chown -R signalium-trainer:signalium-trainer \
  /opt/signalium-trainer \
  /var/log/signalium-trainer \
  /var/lib/signalium-trainer

sudo chmod 750 \
  /opt/signalium-trainer \
  /var/log/signalium-trainer \
  /var/lib/signalium-trainer
```

> The systemd unit also declares `LogsDirectory=signalium-trainer` and
> `StateDirectory=signalium-trainer` which causes systemd to create and
> own these directories automatically when the service is enabled. The
> manual step above is a belt-and-suspenders measure for provisioning
> scripts that run before systemd activation.

---

## 4. Clone the Repository

```bash
sudo -u signalium-trainer git clone \
  https://github.com/trapkeed/signalium-trainer \
  /opt/signalium-trainer

cd /opt/signalium-trainer
```

---

## 5. Configure Environment Variables

```bash
sudo -u signalium-trainer cp \
  /opt/signalium-trainer/.env.example \
  /opt/signalium-trainer/.env

sudo -u signalium-trainer nano /opt/signalium-trainer/.env
```

**Required variables** (the server refuses to start without all three):

| Variable | Description | How to generate |
|---|---|---|
| `PORT` | Express bind port | Set to `8080` for NGINX reverse proxy |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/dbname` |
| `API_KEY` | Secret key protecting all private endpoints | `openssl rand -hex 32` |

Set file permissions so only the service user can read it:

```bash
sudo chmod 600 /opt/signalium-trainer/.env
sudo chown signalium-trainer:signalium-trainer /opt/signalium-trainer/.env
```

---

## 6. Build Process

The build must run **before** devDependencies are pruned, because `esbuild`
(the TypeScript bundler) is a devDependency.

The `start.sh` script handles this automatically in the correct order:

```
pnpm install --frozen-lockfile   ← installs ALL deps (dev + prod)
pnpm run build                   ← compiles TypeScript → dist/index.mjs
pnpm prune --prod                ← removes devDeps from node_modules
```

**Never** run `pnpm install --prod` before `pnpm run build` — this will
cause the build to fail with `Cannot find module 'esbuild'`.

To build manually:

```bash
cd /opt/signalium-trainer
pnpm install --frozen-lockfile
pnpm run build
pnpm prune --prod
```

Build output: `apps/trainer-api/dist/index.mjs` (ESM bundle, ~1.4 MB)

---

## 7. First Cold Start

```bash
cd /opt/signalium-trainer
sudo -u signalium-trainer ./deploy/scripts/start.sh
```

This script:
1. Verifies executables (`node`, `pm2`, `pnpm`) via `command -v` — no hardcoded paths
2. Creates `/var/log/signalium-trainer` and `/var/lib/signalium-trainer` if missing
3. Validates required env vars in `.env`
4. Installs dependencies → builds → prunes → runs migrations → starts PM2
5. Polls `GET /api/healthz` until the server is accepting requests

Optional flags:

```bash
./deploy/scripts/start.sh --skip-build    # use existing dist/ (already built)
./deploy/scripts/start.sh --skip-migrate  # skip drizzle-kit push
```

---

## 8. PM2 Configuration

**File:** `deploy/pm2/ecosystem.config.cjs`

| Setting | Value | Purpose |
|---|---|---|
| `name` | `signalium-trainer` | PM2 process name — used in all pm2 commands |
| `script` | `${TRAINER_DIR}/apps/trainer-api/dist/index.mjs` | Absolute path to entrypoint |
| `cwd` | `${TRAINER_DIR}` | Working directory for each worker process |
| `interpreter_args` | `--enable-source-maps` | Stack traces reference TypeScript source |
| `instances` | `"max"` | One worker per CPU core |
| `exec_mode` | `"cluster"` | Node.js cluster — all workers share port |
| `kill_timeout` | `12000` | Matches the 10 s graceful shutdown in `index.ts` |

Common PM2 commands:

```bash
pm2 list                              # show all processes
pm2 logs signalium-trainer            # tail live logs
pm2 logs signalium-trainer --lines 50 --nostream  # last 50 lines
pm2 reload signalium-trainer          # zero-downtime rolling reload
pm2 stop signalium-trainer            # graceful stop
pm2 delete signalium-trainer          # remove from PM2 registry
```

---

## 9. systemd Integration (Auto-start on Boot)

```bash
# Install the unit file
sudo cp /opt/signalium-trainer/deploy/systemd/signalium-api.service \
        /etc/systemd/system/signalium-api.service

# Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable signalium-api

# Start the service
sudo systemctl start signalium-api

# Check status
sudo systemctl status signalium-api
journalctl -u signalium-api -f
```

### Why `Type=simple` + `--no-daemon`

PM2 can either daemonize itself (`Type=forking`) or stay in the foreground
(`Type=simple` + `--no-daemon`). Using `Type=forking` with `--no-daemon` is
incompatible — systemd would wait for a fork that never arrives.

This unit uses `Type=simple` + `--no-daemon` so systemd holds a direct
reference to the PM2 process. Automatic restart (`Restart=on-failure`) still
works correctly with this approach.

### Executable discovery

The unit uses `/usr/bin/env pm2` rather than a hardcoded path. `PATH` is set
to `/usr/local/bin:/usr/bin:/bin` in the unit. If Node.js was installed via
nvm or fnm, add the nvm bin directory to `PATH` in `.env`:

```bash
PATH=/home/signalium-trainer/.nvm/versions/node/v22.x.x/bin:/usr/local/bin:/usr/bin:/bin
```

---

## 10. NGINX Setup

**File:** `nginx/signalium.conf`

The domain placeholder is `trainer.example.com`. Replace it with your real
domain before installing the file.

```bash
# Copy and replace the placeholder domain
sudo sed 's/trainer\.example\.com/your.real.domain/g' \
  /opt/signalium-trainer/nginx/signalium.conf \
  > /etc/nginx/sites-available/signalium-trainer

# Issue TLS certificate via Let's Encrypt
sudo certbot --nginx -d your.real.domain

# Enable the site
sudo ln -s /etc/nginx/sites-available/signalium-trainer \
           /etc/nginx/sites-enabled/signalium-trainer

# Test and reload
sudo nginx -t && sudo systemctl reload nginx
```

NGINX listens on ports 80 (redirect) and 443 (HTTPS) and proxies all
`/api/*` traffic to `127.0.0.1:8080` (PM2 cluster).

`GET /api/healthz` is public — no rate limit, no access log, 5 s timeout.
All other `/api/*` routes are private and require a valid `API_KEY`.

---

## 11. Deployment Order (Clean Server)

```
1. Install system packages           (apt)
2. Install Node.js 22                (NodeSource)
3. Install PM2 + pnpm                (npm -g)
4. Create OS user                    (useradd)
5. Create runtime directories        (mkdir + chown)
6. Clone repository                  (git clone)
7. Configure .env                    (cp .env.example + edit)
8. Run start.sh                      (install → build → migrate → pm2 start)
9. Install systemd unit              (systemctl enable + start)
10. Configure NGINX                  (sed domain + certbot + reload)
```

---

## 12. Zero-Downtime Redeploy

```bash
# Rolling reload only (no build, no git pull)
./deploy/scripts/restart.sh

# Full redeploy: git pull → build → migrate → rolling reload
./deploy/scripts/restart.sh --hard

# Full redeploy, skip DB migrations
./deploy/scripts/restart.sh --hard --skip-migrate
```

---

## 13. Log Rotation

PM2 ships with `pm2-logrotate`. Install and configure once:

```bash
pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 50M
pm2 set pm2-logrotate:retain 14
pm2 set pm2-logrotate:compress true
pm2 set pm2-logrotate:dateFormat YYYY-MM-DD
pm2 set pm2-logrotate:rotateInterval "0 0 * * *"   # midnight daily
pm2 save
```

NGINX log rotation is handled by the standard `/etc/logrotate.d/nginx`
configuration installed with the nginx package.
