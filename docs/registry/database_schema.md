# Database Schema Reference

## models

| Column | Type | Notes |
|---|---|---|
| id | text (UUID) | Primary key |
| name | text | Unique, slug format |
| display_name | text | Human-readable name |
| description | text | Free text |
| architecture | text | e.g. "LSTM", "GRU", "Transformer" |
| framework | text | Default "pytorch" |
| created_by | text | Actor identifier |
| is_deleted | boolean | Soft delete flag |
| deleted_at | timestamptz | Set on soft delete |
| created_at | timestamptz | Auto |
| updated_at | timestamptz | Auto |

## model_versions

| Column | Type | Notes |
|---|---|---|
| id | text (UUID) | Primary key |
| model_id | text | FK → models.id |
| version | text | Semantic version |
| status | enum | Lifecycle status |
| architecture | text | Version-specific arch |
| training_date | timestamptz | When training completed |
| dataset_version | text | e.g. "2.1.0" |
| feature_version | text | Feature pipeline version |
| label_version | text | Label scheme version |
| framework_version | text | PyTorch version |
| trainer_version | text | Trainer release version |
| config_version | text | Config schema version |
| experiment_id | text | Research framework ID |
| git_commit_hash | text | Short or full SHA |
| mlflow_run_id | text | MLflow run identifier |
| input_schema | jsonb | Input tensor specification |
| output_schema | jsonb | Output tensor specification |
| config_snapshot | jsonb | Full training config |
| rejection_reason | text | Set on rejection |
| promoted_by | text | Actor who promoted |
| promoted_at | timestamptz | Last promotion time |
| created_by | text | Actor identifier |

## model_artifacts

| Column | Type | Notes |
|---|---|---|
| id | text (UUID) | Primary key |
| model_version_id | text | FK → model_versions.id |
| artifact_type | enum | torchscript/onnx/manifest/… |
| filename | text | Original filename |
| storage_path | text | Absolute filesystem path |
| mime_type | text | MIME type |
| size_bytes | bigint | File size |
| checksum | text | SHA-256 hex |
| checksum_algorithm | text | Default "sha256" |
| status | enum | pending/uploaded/verified/corrupted |
| uploaded_by | text | Actor identifier |

## model_metrics

| Column | Type | Notes |
|---|---|---|
| id | text (UUID) | Primary key |
| model_version_id | text | FK → model_versions.id |
| category | enum | training/validation/evaluation/inference/operational |
| metric_key | text | e.g. "f1_macro", "accuracy" |
| metric_value | real | Float value |
| unit | text | e.g. "%" or "" |
| recorded_at | timestamptz | When the metric was measured |

## model_validation

Stores the latest validation result per version (upserted).

| Column | Type | Notes |
|---|---|---|
| id | text (UUID) | Primary key |
| model_version_id | text | FK → model_versions.id |
| status | enum | pending/running/passed/failed/skipped |
| checksum_verified | boolean | All artifact checksums pass |
| manifest_valid | boolean | Manifest artifact present and intact |
| metadata_valid | boolean | Metadata artifact present and intact |
| required_artifacts_present | boolean | torchscript+onnx+manifest+metadata present |
| input_schema_valid | boolean | inputSchema field set |
| output_schema_valid | boolean | outputSchema field set |
| framework_version_ok | boolean | frameworkVersion non-empty |
| dataset_version_ok | boolean | datasetVersion non-empty |
| artifact_integrity_ok | boolean | All artifacts verified |
| checklist | jsonb | Full checklist snapshot |
| validation_report | jsonb | Detailed report |
| failure_reasons | jsonb | Array of failure strings |
| validated_by | text | Actor |
| validated_at | timestamptz | Completion time |
| duration_ms | text | Validation wall-clock |

## model_aliases

Named pointers to specific versions. One (model_id, alias_name) pair.

| Column | Notes |
|---|---|
| alias_name | e.g. "champion", "production", "challenger" |
| model_version_id | Current target version |

## model_audit_log (immutable)

| Column | Notes |
|---|---|
| event_type | One of 16 event types |
| model_id | May be null for version-only events |
| model_version_id | May be null for model-only events |
| actor | API key identity |
| from_status / to_status | For status change events |
| details | JSON payload |
| ip_address | Client IP |
| user_agent | Client UA |
