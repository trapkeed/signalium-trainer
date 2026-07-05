# deploy

Production deployment configurations for Signalium Quant.

## Structure

```
deploy/
  pm2/          PM2 process manager configuration
  backup/       Database and checkpoint backup scripts
  restore/      Restore procedures and scripts
  monitoring/   Health check and alerting configuration
```

## pm2/
Process manager config for the Trainer API Node.js server.
Add `ecosystem.config.cjs` here for PM2 managed deployments.

## backup/
Scheduled backup scripts for:
- PostgreSQL database
- Model checkpoints (`/app/checkpoints/`)
- MLflow artifacts (`/app/mlruns/`)

## restore/
Point-in-time restore procedures.

## monitoring/
Health check endpoints, uptime monitoring, and alert rules.
