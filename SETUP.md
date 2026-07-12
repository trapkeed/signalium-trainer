# Signalium Trainer — Fresh VPS Install

This archive contains everything needed to run the Signalium Trainer API and
Research Dashboard on a clean server. Both services are pre-built; **no build
tools are required on the server**.

---

## Prerequisites

- Node.js ≥ 18
- pnpm (`npm install -g pnpm`)
- PM2 (`npm install -g pm2`)
- nginx

---

## Step 1 — Extract the archive

```bash
tar -xzf signalium-trainer-release.tar.gz -C /opt
# Result: /opt/signalium-trainer/
cd /opt/signalium-trainer
```

---

## Step 2 — Create your .env file

```bash
cp .env.example .env
nano .env          # fill in your values (see comments inside)
```

Minimum required variables:

| Variable | Required | Description |
|---|---|---|
| `PORT` | ✅ | Port the Trainer API listens on (e.g. `3000`) |
| `SIGNALIUM_JWKS_URL` | ✅ | Platform JWKS endpoint for verifying inbound admin JWTs |
| `RESEARCH_LAB_ADMIN_TOKEN` | ✅ | Shared secret for internal trainer → Python RL service calls |
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `RESEARCH_LAB_API_URL` | ✅ | Python Research Lab service URL (default `http://localhost:8002`) |
| `QDE_API_URL` | ✅ | QDE service URL (default `http://localhost:8001`) |
| `SIGNALIUM_PLATFORM_API_KEY` | optional | API key for outbound push calls to the platform. Leave blank to disable push (trainer still starts). |

---

## Step 3 — Install production dependencies

```bash
pnpm install --prod --no-frozen-lockfile
```

This installs only runtime packages (no build tools). Takes ~10–20 seconds.

---

## Step 4 — Start the API via PM2

```bash
# Create the log directory
sudo mkdir -p /var/log/signalium-trainer

# Start
pm2 start deploy/pm2/ecosystem.config.cjs --env production
pm2 save

# Persist across reboots
pm2 startup   # follow the printed command
pm2 save
```

Verify it is running:

```bash
pm2 list
pm2 logs signalium-trainer-api --lines 30
```

---

## Step 5 — Configure nginx

The nginx config is at `deploy/nginx/signalium.conf`.

1. **Set the API port** — open the file and change `127.0.0.1:3000` to match
   the `PORT` in your `.env`.

2. **Symlink into sites-enabled**:

```bash
sudo ln -sf /opt/signalium-trainer/deploy/nginx/signalium.conf \
            /etc/nginx/sites-enabled/signalium.conf

# Remove the default site if present
sudo rm -f /etc/nginx/sites-enabled/default

sudo nginx -t          # verify syntax
sudo systemctl reload nginx
```

The dashboard will now be served at `http://<your-server-ip>/` and the API
at `http://<your-server-ip>/api/`.

---

## Updating in the future

Once the server is set up, apply future releases by downloading a new archive
and repeating steps 1–3, then:

```bash
pm2 reload signalium-trainer-api
```

Or, if you switch back to git-based deploys:

```bash
cd /opt/signalium-trainer
bash deploy/scripts/deploy.sh
```

---

## Troubleshooting

| Symptom | Check |
|---|---|
| API returns 502 | `pm2 logs signalium-trainer-api --lines 50` |
| Dashboard loads but API calls fail | Confirm `PORT` in `.env` matches `signalium.conf` upstream |
| White page / 404 on deep links | nginx SPA fallback — check `sudo nginx -t` and reload |
| Static assets 404 | Confirm `artifacts/research-dashboard/dist/public/` exists |
