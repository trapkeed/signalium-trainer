# deploy/monitoring

Health check, metrics, and alerting configuration.

Endpoints to monitor:
- Trainer API health: GET /api/healthz
- MLflow UI: http://localhost:5000/health
- Python trainer: `trainer doctor`
