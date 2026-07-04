# Model Lifecycle Guide

## States

```
uploading → uploaded → validating → validated → candidate → challenger → champion → production → retired → archived

Any non-terminal state → rejected
```

| Status | Description | Who Can Trigger |
|---|---|---|
| `uploading` | Version created, awaiting artifacts | system |
| `uploaded` | All artifacts uploaded | system |
| `validating` | Validation job running | admin (POST /validate) |
| `validated` | All checks passed | system |
| `candidate` | Ready for benchmarking | admin |
| `challenger` | Competing with current champion | admin |
| `champion` | Best-performing model | admin |
| `production` | Serving live traffic | admin |
| `retired` | Superseded, kept for audit | admin |
| `archived` | Terminal — read-only | admin |
| `rejected` | Failed validation or manual rejection | admin / system |

## Valid Transitions

```
uploading  → [uploaded]
uploaded   → [validating, rejected]
validating → [validated, rejected]
validated  → [candidate, rejected]
candidate  → [challenger, rejected]
challenger → [champion, rejected]
champion   → [production, retired, rejected]
production → [retired]
retired    → [archived]
archived   → []          ← terminal
rejected   → []          ← terminal
```

## Promoting a Model (Admin API)

```bash
# Step 1: Create model
curl -X POST /api/registry/models \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"quant_v1","displayName":"Quant V1","architecture":"LSTM","createdBy":"trainer"}'

# Step 2: Create version
curl -X POST /api/registry/models/$MODEL_ID/versions \
  -H "X-API-Key: $ADMIN_KEY" \
  -d '{"version":"1.0.0","architecture":"LSTM","datasetVersion":"2.1.0","createdBy":"trainer"}'

# Step 3: Upload artifacts
curl -X POST /api/registry/models/$MODEL_ID/versions/$VERSION_ID/artifacts \
  -H "X-API-Key: $ADMIN_KEY" \
  -H "X-Artifact-Type: torchscript" \
  -H "X-Artifact-Filename: model.pt" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @model.pt

# Step 4: Validate
curl -X POST /api/registry/models/$MODEL_ID/versions/$VERSION_ID/validate \
  -H "X-API-Key: $ADMIN_KEY"

# Step 5: Promote through stages
curl -X POST /api/registry/models/$MODEL_ID/versions/$VERSION_ID/promote \
  -H "X-API-Key: $ADMIN_KEY" \
  -d '{"targetStatus":"candidate"}'

curl -X POST .../promote -d '{"targetStatus":"challenger"}'
curl -X POST .../promote -d '{"targetStatus":"champion"}'
curl -X POST .../promote -d '{"targetStatus":"production"}'
```

## Rollback

```bash
curl -X POST /api/registry/models/$MODEL_ID/rollback \
  -H "X-API-Key: $ADMIN_KEY" \
  -d '{"toVersionId":"$PREV_VERSION_ID","reason":"F1 regression in prod"}'
```

Rollback:
1. Retires the current production version's deployment record
2. Points the `production` alias to the target version
3. Creates a new deployment record
4. Writes to `rollback_history`
5. Writes to `model_audit_log`

## No Automatic Promotions

The registry never automatically promotes models to production. Every transition to
`champion` or `production` requires an admin API call. This is by design — the
platform enforces human review at each production gate.
