# nginx Setup — Signalium Trainer

nginx sits in front of both the Research Dashboard (static files) and the
Trainer API (Node.js / PM2), joining them on a single port 80/443 address.

```
Browser ──► nginx :80 ──► /api/*  ──► Trainer API (PM2, :PORT)
                    └──► /*      ──► dist/public/ (static files)
```

---

## 1. Install nginx

```bash
sudo apt update && sudo apt install -y nginx
```

---

## 2. Set the API port in the config

Open `deploy/nginx/signalium.conf` and change the `upstream` block to match
the `PORT` value in your `.env`:

```nginx
upstream trainer_api {
    server 127.0.0.1:3000;   # ← your PORT
}
```

---

## 3. Symlink the config into sites-enabled

```bash
sudo ln -sf /opt/signalium-trainer/deploy/nginx/signalium.conf \
            /etc/nginx/sites-enabled/signalium.conf

# Disable the default site if it's still there
sudo rm -f /etc/nginx/sites-enabled/default

# Verify syntax
sudo nginx -t
```

---

## 4. Reload nginx

```bash
sudo systemctl reload nginx
# or, if nginx isn't running yet:
sudo systemctl enable --now nginx
```

---

## 5. Build and start everything

If this is a fresh server:

```bash
cd /opt/signalium-trainer
cp .env.example .env        # fill in your values
NODE_ENV= pnpm install --no-frozen-lockfile
REBUILD=1 bash deploy/scripts/restart.sh
```

If you're applying an update:

```bash
cd /opt/signalium-trainer
git pull origin main
NODE_ENV= pnpm install --no-frozen-lockfile
REBUILD=1 bash deploy/scripts/restart.sh
```

`REBUILD=1` now builds **both** the Trainer API and the Research Dashboard
and re-prunes devDependencies when done.

---

## 6. SSL / TLS (recommended)

Use Certbot to add HTTPS. nginx's HTTP block stays; Certbot injects the
HTTPS server block automatically:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain.com
```

After Certbot runs, reload nginx:

```bash
sudo systemctl reload nginx
```

---

## Troubleshooting

| Symptom | Check |
|---|---|
| Dashboard loads but API calls fail (network error) | Confirm `PORT` in `.env` matches the upstream in `signalium.conf`; check `pm2 list` |
| White page / 404 on deep links | SPA fallback is in place — make sure the `try_files` line is present and nginx was reloaded |
| `nginx -t` fails | Run `sudo nginx -t` for the exact error; often a stale default site conflict |
| API returns 502 | The PM2 process crashed — check `pm2 logs signalium-trainer-api --lines 50` |
| Static assets return 404 | Confirm the build ran: `ls /opt/signalium-trainer/artifacts/research-dashboard/dist/public/` |
