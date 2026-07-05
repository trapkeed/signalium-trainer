# deploy/pm2

PM2 process manager configuration for the Trainer API.

Create `ecosystem.config.cjs` with:
```js
module.exports = {
  apps: [{
    name: 'trainer-api',
    script: 'apps/trainer-api/dist/index.mjs',
    instances: 'max',
    exec_mode: 'cluster',
    env_production: {
      NODE_ENV: 'production',
      PORT: 8080
    }
  }]
}
```
