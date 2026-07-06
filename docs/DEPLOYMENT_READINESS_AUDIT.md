# Deployment Readiness Audit

**Date:** 2026-07-06  
**Target:** Ubuntu 24.04 LTS — Python 3.12, Node 22 LTS, PostgreSQL 16, Redis 7, PM2, NGINX, systemd, Docker  
**Method:** Static analysis only — no code was modified

---

## Executive Summary

| Category | Status | Blockers |
|---|---|---|
| Application build | ✅ Ready | None |
| Runtime configuration | 🟡 Partial | `.env` not deployed, no secrets delivery |
| Database | 🔴 Missing | No migrations exist, schema is empty |
| Process management (PM2) | 🔴 Missing | No `ecosystem.config.cjs` |
| Web server (NGINX) | 🔴 Missing | No site config, no TLS |
| systemd | 🔴 Missing | No unit files |
| Security | 🔴 Critical | Open CORS, no auth, no rate limiting |
| Health checks | 🟡 Partial | `/api/healthz` exists but checks nothing |
| Logging / logrotate | 🟡 Partial | Paths configured, rotation not set up |
| Backup / restore | 🔴 Missing | Placeholder READMEs only |
| Monitoring / alerting | 🔴 Missing | Nothing configured |
| Redis | ⚠️ Listed, unused | No code uses Redis — see note |
| SSL / TLS | 🔴 Missing | No cert provisioning |
| Firewall | 🔴 Missing | No UFW rules |
| Docker (multi-service) | 🔴 Missing | No `docker-compose.prod.yml` |

---

## 1. What Exists vs. What Is Missing

### 1A — What Exists (Ready)

| Item | Location | Notes |
|---|---|---|
| Compiled API bundle | `apps/trainer-api/dist/index.mjs` | 1.4 MB ESM, source maps included |
| API start command | `node --enable-source-maps ./dist/index.mjs` | Reads `PORT` env var, crashes if missing |
| Health endpoint | `GET /api/healthz` → `{"status":"ok"}` | Responds correctly, no DB probe |
| Python trainer CLI | `trainer` entrypoint via `pyproject.toml` | Installed via `pip install -e .` |
| Docker GPU container | `docker/Dockerfile` | CUDA 12.4, Python 3.12, `trainer` entrypoint |
| Post-merge hook | `scripts/post-merge.sh` | Runs `pnpm install` + `drizzle push` |
| Environment template | `.env.example` | Complete variable listing |
| Supply-chain protection | `.npmrc` `minimumReleaseAge: 1440` | 1-day npm package age gate |
| DB layer | `packages/db/src/index.ts` | Pool + Drizzle ORM configured |
| Directory scaffolding | `deploy/`, `infrastructure/` | README placeholders only — **no actual config** |

### 1B — What Is Missing (Blockers and Gaps)

| Item | Severity | Impact if Skipped |
|---|---|---|
| PM2 `ecosystem.config.cjs` | 🔴 Blocker | Cannot start the API as a managed process |
| NGINX site config | 🔴 Blocker | No reverse proxy, no TLS termination |
| systemd unit files | 🔴 Blocker | No guaranteed service restart on reboot |
| Database migrations | 🔴 Blocker | Schema is empty — application has no tables |
| `.env` on server | 🔴 Blocker | API crashes at start (`PORT`, `DATABASE_URL` missing) |
| TLS certificate provisioning | 🔴 Blocker | No HTTPS |
| UFW firewall rules | 🔴 Blocker | All ports exposed |
| CORS origin restriction | 🔴 Security | Any domain can call the API |
| Authentication middleware | 🔴 Security | All endpoints are unauthenticated |
| Rate limiting | 🔴 Security | No abuse protection |
| Graceful shutdown handler | 🟡 Stability | Requests dropped on `SIGTERM` (PM2 reload) |
| Deep health check (DB probe) | 🟡 Stability | Unhealthy pod serves traffic |
| Drizzle migrations folder | 🔴 Blocker | `drizzle-kit generate` has never been run |
| Backup scripts | 🔴 Ops | No checkpoint or DB backup |
| Restore scripts | 🔴 Ops | No recovery path |
| Logrotate config | 🟡 Ops | `/var/log/signalium-trainer` fills disk |
| `.nvmrc` / `.node-version` | 🟡 Config | Node version not pinned on server |
| `.python-version` | 🟡 Config | Python version not pinned on server |
| `signalium` system user | 🔴 Security | App runs as root if not created |
| Redis (code usage) | ⚠️ N/A | Redis 7 listed as target but **zero Redis code exists** |
| `docker-compose.prod.yml` | 🟡 Ops | No multi-service compose for local/staging |
| Monitoring / alerting | 🟡 Ops | No Prometheus metrics, no alerting |
| Secrets delivery to server | 🔴 Blocker | No mechanism to get `.env` values onto VPS |

---

## 2. Security Findings

| # | Finding | Location | Severity | Fix Required |
|---|---|---|---|---|
| S-1 | `app.use(cors())` — all origins allowed | `apps/trainer-api/src/app.ts:14` | 🔴 Critical | Add `CORS_ORIGIN` env var and restrict |
| S-2 | No authentication on any route | `apps/trainer-api/src/routes/` | 🔴 Critical | Add JWT or API key middleware |
| S-3 | No rate limiting | `apps/trainer-api/src/app.ts` | 🔴 Critical | Add `express-rate-limit` |
| S-4 | App will run as root if no system user created | Ubuntu deployment | 🔴 Critical | Create `signalium` user, run PM2 under it |
| S-5 | `SESSION_SECRET` is in secrets but never consumed | Any source file | 🟡 Medium | Wire into session middleware |
| S-6 | No `Helmet.js` HTTP security headers | `apps/trainer-api/src/app.ts` | 🟡 Medium | Add `helmet` middleware |
| S-7 | No input size limit on JSON body parser | `apps/trainer-api/src/app.ts:17` | 🟡 Medium | Add `express.json({ limit: '1mb' })` |
| S-8 | PostgreSQL connection not SSL-enforced | `packages/db/src/index.ts` | 🟡 Medium | Add `?sslmode=require` to `DATABASE_URL` |
| S-9 | No secrets delivery mechanism documented | Server setup | 🔴 Critical | Use `scp .env` or cloud secret manager |
| S-10 | Docker image runs trainer as root | `docker/Dockerfile` | 🟡 Medium | Add `RUN useradd -m trainer && USER trainer` |

---

## 3. Infrastructure Gaps

### Redis
Redis 7 is listed as a deployment dependency but **no application code uses Redis**. The `package.json` has no `ioredis` or `redis` dependency. Python trainer has no redis client. Redis is not needed today but would be required for:
- Session storage (once auth is added)
- Rate limit state (multi-instance PM2 cluster)
- Prediction cache

**Recommendation:** Do not install Redis yet. Add it when auth or caching is implemented.

### Database Migrations
The Drizzle schema file (`packages/db/src/schema/index.ts`) is **empty** — it exports `{}` with no tables defined. This means:
- `drizzle-kit generate` has never been run
- `drizzle-kit migrate` / `push` would create no tables
- The application currently has no persistent storage structure

**This is a deployment blocker if the API is expected to read or write data.**

### Node Version
No `.nvmrc`, `.node-version`, or `engines` field in `package.json`. On a fresh Ubuntu 24.04 server, `apt install nodejs` installs Node 18. Node 22 LTS must be installed manually via NodeSource or NVM.

### Python Version
`pyproject.toml` requires `>=3.12`. Ubuntu 24.04 ships Python 3.12 — this is fine. A `.python-version` file pinning `3.12` is still recommended for pyenv users.

---

## 4. Deployment Checklist

### Phase 0 — Pre-Deployment (Before Touching the Server)

- [ ] **0.1** Generate a `SESSION_SECRET`: `openssl rand -hex 64`
- [ ] **0.2** Generate a JWT signing key (when auth is implemented): `openssl rand -hex 64`  
- [ ] **0.3** Provision a PostgreSQL 16 database and record `DATABASE_URL`
- [ ] **0.4** Obtain a domain name with DNS A record pointing to the server IP
- [ ] **0.5** Confirm `SIGNALIUM_API_KEY` or `SIGNALIUM_TOKEN` is available
- [ ] **0.6** Create `.env` file from `.env.example` with all production values filled in
- [ ] **0.7** Run `drizzle-kit generate` locally against the schema and commit migrations before deploying
- [ ] **0.8** Build the TypeScript bundle locally: `pnpm run build` — verify `apps/trainer-api/dist/index.mjs` exists
- [ ] **0.9** Confirm the repository is accessible from the server (GitHub SSH or HTTPS)

---

### Phase 1 — Server Bootstrap

```bash
# 1.1 — Update system packages
sudo apt update && sudo apt upgrade -y

# 1.2 — Install system dependencies
sudo apt install -y \
  git curl wget ca-certificates gnupg \
  build-essential libssl-dev \
  nginx certbot python3-certbot-nginx \
  ufw fail2ban \
  postgresql-client-16 \
  logrotate

# 1.3 — Install Node 22 LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # must print v22.x.x

# 1.4 — Install pnpm
npm install -g pnpm@10
pnpm --version   # must print 10.x.x

# 1.5 — Install PM2
npm install -g pm2
pm2 --version

# 1.6 — Install Python 3.12 (ships with Ubuntu 24.04, verify)
python3 --version   # must print 3.12.x
pip3 install --upgrade pip

# 1.7 — Create application system user (never run as root)
sudo useradd --system --shell /bin/bash --create-home signalium
sudo usermod -aG sudo signalium   # remove after setup if not needed

# 1.8 — Create application directory
sudo mkdir -p /opt/signalium
sudo chown signalium:signalium /opt/signalium

# 1.9 — Create log and storage directories
sudo mkdir -p /var/log/signalium-trainer
sudo mkdir -p /var/lib/signalium/{checkpoints,exports,data,mlruns}
sudo chown -R signalium:signalium /var/log/signalium-trainer /var/lib/signalium
```

---

### Phase 2 — Firewall

```bash
# 2.1 — Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh          # port 22
sudo ufw allow http         # port 80  (NGINX — redirect to HTTPS)
sudo ufw allow https        # port 443 (NGINX — TLS)
sudo ufw deny 8080          # block direct API access
sudo ufw deny 5432          # block direct DB access (unless remote)
sudo ufw deny 5000          # block MLflow if running locally
sudo ufw --force enable
sudo ufw status verbose

# 2.2 — Enable fail2ban (brute force protection)
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

---

### Phase 3 — Repository Checkout

```bash
# 3.1 — Clone as the signalium user
sudo -u signalium bash -c '
  cd /opt/signalium
  git clone https://github.com/trapkeed/signalium-trainer .
  git checkout production-refactor
'

# 3.2 — Copy .env file to server (do NOT commit this file)
scp .env signalium@<server-ip>:/opt/signalium/.env
sudo chown signalium:signalium /opt/signalium/.env
sudo chmod 600 /opt/signalium/.env   # owner read-only
```

---

### Phase 4 — Node.js / API Installation

```bash
sudo -u signalium bash -c '
  cd /opt/signalium

  # 4.1 — Install Node dependencies
  pnpm install --frozen-lockfile

  # 4.2 — Build TypeScript bundle
  pnpm run build

  # 4.3 — Verify the binary exists
  ls -lh apps/trainer-api/dist/index.mjs   # should be ~1.4 MB
'
```

---

### Phase 5 — Database Setup

```bash
# 5.1 — Verify DATABASE_URL connectivity
sudo -u signalium bash -c '
  cd /opt/signalium
  source .env
  psql "$DATABASE_URL" -c "SELECT version();"
'

# 5.2 — Run Drizzle migrations
# NOTE: Migrations must be generated first (drizzle-kit generate run locally)
sudo -u signalium bash -c '
  cd /opt/signalium
  source .env
  pnpm --filter "@workspace/db" run push
'

# 5.3 — Verify schema was applied
sudo -u signalium bash -c '
  source /opt/signalium/.env
  psql "$DATABASE_URL" -c "\dt"
'
```

**⚠️ BLOCKER:** The current schema (`packages/db/src/schema/index.ts`) exports `{}` — no tables are defined. Step 5.2 will create no tables. Define the schema and run `drizzle-kit generate` before deploying.

---

### Phase 6 — Python Trainer Installation

```bash
sudo -u signalium bash -c '
  cd /opt/signalium

  # 6.1 — Install trainer package (CPU-only on API server, GPU on RunPod)
  pip3 install -e "."

  # 6.2 — Verify CLI entry point
  trainer doctor

  # 6.3 — Verify API key is set
  trainer doctor   # should show SIGNALIUM_API_KEY: ✅
'
```

---

### Phase 7 — PM2 Process Manager

**⚠️ Missing file:** `deploy/pm2/ecosystem.config.cjs` does not exist. Create it:

```javascript
// deploy/pm2/ecosystem.config.cjs
module.exports = {
  apps: [
    {
      name: "trainer-api",
      script: "/opt/signalium/apps/trainer-api/dist/index.mjs",
      interpreter: "node",
      interpreter_args: "--enable-source-maps",
      instances: "max",          // cluster mode — use all CPU cores
      exec_mode: "cluster",
      env_file: "/opt/signalium/.env",
      env: {
        NODE_ENV: "production",
        PORT: 8080,
        LOG_LEVEL: "info",
      },
      error_file: "/var/log/signalium-trainer/api-error.log",
      out_file: "/var/log/signalium-trainer/api-out.log",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      max_memory_restart: "1G",
      restart_delay: 5000,
      max_restarts: 10,
      watch: false,
      autorestart: true,
    },
  ],
};
```

```bash
# 7.1 — Start with PM2
sudo -u signalium bash -c '
  cd /opt/signalium
  pm2 start deploy/pm2/ecosystem.config.cjs
  pm2 save
'

# 7.2 — Configure PM2 to start on boot
pm2 startup systemd -u signalium --hp /home/signalium
# Run the printed sudo command

# 7.3 — Verify running
pm2 list
pm2 logs trainer-api --lines 50

# 7.4 — Smoke test (internal)
curl -s http://127.0.0.1:8080/api/healthz
# Expected: {"status":"ok"}
```

---

### Phase 8 — NGINX Reverse Proxy

**⚠️ Missing file:** `nginx/signalium.conf` does not exist. Create it at `/etc/nginx/sites-available/signalium`:

```nginx
upstream trainer_api {
    least_conn;
    server 127.0.0.1:8080;
    keepalive 64;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_session_cache   shared:SSL:10m;

    # Security headers
    add_header X-Frame-Options         DENY;
    add_header X-Content-Type-Options  nosniff;
    add_header X-XSS-Protection        "1; mode=block";
    add_header Referrer-Policy         strict-origin-when-cross-origin;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Request size limit (match express body parser)
    client_max_body_size 10M;

    location /api/ {
        proxy_pass         http://trainer_api;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Connection        "";
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;

        # Rate limiting zone (define in nginx.conf http block)
        # limit_req zone=api_limit burst=20 nodelay;
    }

    location /health {
        proxy_pass http://trainer_api/api/healthz;
        access_log off;
    }
}
```

```bash
# 8.1 — Enable site
sudo ln -s /etc/nginx/sites-available/signalium /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 8.2 — Issue TLS certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com \
  --email ops@signalium.io --agree-tos --non-interactive

# 8.3 — Verify cert auto-renewal
sudo certbot renew --dry-run

# 8.4 — Confirm HTTPS works
curl -I https://yourdomain.com/api/healthz
# Expected: HTTP/2 200
```

---

### Phase 9 — systemd Unit Files

**⚠️ Missing:** No `.service` files exist. PM2 startup (phase 7.2) creates a systemd unit automatically via `pm2 startup`. For the Python trainer as a long-running service (if needed):

```ini
# /etc/systemd/system/signalium-mlflow.service
[Unit]
Description=Signalium MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=signalium
WorkingDirectory=/opt/signalium
EnvironmentFile=/opt/signalium/.env
ExecStart=mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri /var/lib/signalium/mlruns \
  --default-artifact-root /var/lib/signalium/mlruns
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

```bash
# 9.1 — Install and enable (if running MLflow locally)
sudo systemctl daemon-reload
sudo systemctl enable signalium-mlflow
sudo systemctl start signalium-mlflow
sudo systemctl status signalium-mlflow
```

---

### Phase 10 — Logrotate

**⚠️ Missing:** No logrotate config. Create `/etc/logrotate.d/signalium`:

```
/var/log/signalium-trainer/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        pm2 reloadLogs
    endscript
}
```

```bash
# 10.1 — Install and test
sudo logrotate -d /etc/logrotate.d/signalium   # dry run
sudo logrotate -f /etc/logrotate.d/signalium   # force rotate
```

---

### Phase 11 — Backup Configuration

**⚠️ Missing:** `deploy/backup/` contains only a README. Required scripts:

**Database backup** (`deploy/backup/backup-db.sh`):
```bash
#!/bin/bash
set -euo pipefail
source /opt/signalium/.env
BACKUP_DIR="/var/lib/signalium/backups/db"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"
pg_dump "$DATABASE_URL" | gzip > "$BACKUP_DIR/signalium_${TIMESTAMP}.sql.gz"
# Prune backups older than 30 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +30 -delete
echo "DB backup complete: signalium_${TIMESTAMP}.sql.gz"
```

**Checkpoint backup** (`deploy/backup/backup-checkpoints.sh`):
```bash
#!/bin/bash
set -euo pipefail
BACKUP_DIR="/var/lib/signalium/backups/checkpoints"
SRC="/var/lib/signalium/checkpoints"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/checkpoints_${TIMESTAMP}.tar.gz" -C "$SRC" .
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
echo "Checkpoint backup complete: checkpoints_${TIMESTAMP}.tar.gz"
```

**Cron schedule:**
```bash
# 11.1 — Add to signalium user crontab
sudo -u signalium crontab -e
# Add:
# 0 2 * * * /opt/signalium/deploy/backup/backup-db.sh >> /var/log/signalium-trainer/backup.log 2>&1
# 0 3 * * * /opt/signalium/deploy/backup/backup-checkpoints.sh >> /var/log/signalium-trainer/backup.log 2>&1
```

---

### Phase 12 — Health Checks and Monitoring

**Current state:** `GET /api/healthz` returns `{"status":"ok"}` but does **not** probe the database or disk.

**Missing items:**

- [ ] Deep health check that queries PostgreSQL (`SELECT 1`)
- [ ] External uptime monitor (UptimeRobot, Better Uptime, or Pingdom — free tier)
- [ ] PM2 monitoring: `pm2 monitor` or `pm2 plus`
- [ ] Disk space alert for `/var/lib/signalium/checkpoints`
- [ ] Log error rate alert

**Minimal external health check:**
```bash
# 12.1 — Verify health endpoint is reachable externally
curl -sf https://yourdomain.com/api/healthz || echo "HEALTH CHECK FAILED"

# 12.2 — Add to cron for basic monitoring
# * * * * * curl -sf https://yourdomain.com/api/healthz || \
#   echo "API DOWN $(date)" >> /var/log/signalium-trainer/alerts.log
```

---

### Phase 13 — Deployment Verification

```bash
# 13.1 — All services running
sudo systemctl status nginx
pm2 list
sudo systemctl status postgresql  # if local

# 13.2 — Health check
curl -s https://yourdomain.com/api/healthz
# Expected: {"status":"ok"}

# 13.3 — TLS grade
# Visit https://www.ssllabs.com/ssltest/analyze.html?d=yourdomain.com
# Expected: A or A+

# 13.4 — CORS check (should be restricted)
curl -H "Origin: https://evil.com" -I https://yourdomain.com/api/healthz
# Once CORS middleware is fixed: should NOT return Access-Control-Allow-Origin: *

# 13.5 — PM2 process count (cluster mode)
pm2 list   # should show N instances = CPU core count

# 13.6 — Logs are flowing
pm2 logs trainer-api --lines 20
tail -f /var/log/signalium-trainer/api-out.log

# 13.7 — Python trainer CLI works
sudo -u signalium bash -c 'source /opt/signalium/.env && trainer doctor'

# 13.8 — Logrotate working
sudo logrotate -d /etc/logrotate.d/signalium

# 13.9 — Firewall correct
sudo ufw status numbered
# 22/tcp, 80/tcp, 443/tcp ALLOW — all else DENY
```

---

### Phase 14 — Rollback Procedure

```bash
# 14.1 — Roll back to previous git commit
sudo -u signalium bash -c '
  cd /opt/signalium
  git fetch origin production-refactor
  git log --oneline -10
  git checkout <previous-commit-sha>
  pnpm install --frozen-lockfile
  pnpm run build
  pm2 reload trainer-api
'

# 14.2 — Roll back database (if migration was run)
# Restore from backup:
sudo -u signalium bash -c '
  source /opt/signalium/.env
  gunzip -c /var/lib/signalium/backups/db/signalium_<TIMESTAMP>.sql.gz | psql "$DATABASE_URL"
'
```

---

## 5. Missing Files Summary

The following files must be created before deployment. They do not exist in the repository:

| File | Purpose | Priority |
|---|---|---|
| `deploy/pm2/ecosystem.config.cjs` | PM2 cluster configuration | 🔴 Blocker |
| `nginx/signalium.conf` | NGINX reverse proxy + TLS | 🔴 Blocker |
| `packages/db/src/schema/*.ts` | Drizzle table definitions | 🔴 Blocker |
| `packages/db/migrations/` | Drizzle migration files | 🔴 Blocker |
| `deploy/backup/backup-db.sh` | PostgreSQL backup script | 🔴 Required |
| `deploy/backup/backup-checkpoints.sh` | Checkpoint backup script | 🔴 Required |
| `/etc/logrotate.d/signalium` | Log rotation config | 🟡 Required |
| `/etc/systemd/system/signalium-mlflow.service` | MLflow systemd unit | 🟡 If local MLflow |
| `.nvmrc` | Node 22 version pin | 🟡 Recommended |
| `.python-version` | Python 3.12 version pin | 🟡 Recommended |

---

## 6. Redis Assessment

Redis 7 was listed as a deployment target dependency.

**Finding:** No Redis code exists anywhere in the repository. No `ioredis`, `redis`, or Python `redis` package is imported. The Python trainer has no Redis client. The Node.js API has no session store, cache, or queue using Redis.

**Recommendation:** Do not install or configure Redis at this time. Add it when one of the following is implemented:
- Session storage for authentication
- Rate limit state sharing across PM2 cluster instances  
- Inference prediction cache
- Background job queue (trainer job dispatch)

---

## 7. Deployment Readiness Score

| Area | Score | Notes |
|---|---|---|
| Application code | 10/10 | Builds, runs, health endpoint present |
| Environment config | 6/10 | `.env.example` complete; no delivery mechanism |
| Database | 2/10 | Schema empty, no migrations |
| Process management | 1/10 | No PM2 config |
| Web server | 1/10 | No NGINX config |
| Security | 2/10 | Open CORS, no auth, no rate limiting, no headers |
| Backup / restore | 1/10 | Placeholder READMEs only |
| Monitoring | 1/10 | Nothing configured |
| TLS / firewall | 0/10 | Not configured |
| Documentation | 8/10 | Comprehensive |
| **Total** | **32/100** | Not ready for live traffic |

**To reach deployment-ready (≥80/100):** Complete the Blockers in phases 0–9 above. Estimated effort: **2–3 engineer-days**.
