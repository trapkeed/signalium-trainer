import { db, modelValidationTable, modelArtifactsTable, modelVersionsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import { verifyArtifactChecksum } from "./artifactService.js";
import { updateVersionStatus } from "./registryService.js";
import { writeAuditLog } from "./auditService.js";

const REQUIRED_ARTIFACT_TYPES = ["torchscript", "onnx", "manifest", "metadata"] as const;

export async function runValidation(
  modelVersionId: string,
  actor: string,
): Promise<{ passed: boolean; report: Record<string, unknown> }> {
  const start = Date.now();

  // Upsert pending validation record
  const existing = await db
    .select()
    .from(modelValidationTable)
    .where(eq(modelValidationTable.modelVersionId, modelVersionId));

  if (existing.length === 0) {
    await db.insert(modelValidationTable).values({
      modelVersionId,
      status: "running",
      validatedBy: actor,
    });
  } else {
    await db
      .update(modelValidationTable)
      .set({ status: "running", validatedBy: actor, updatedAt: new Date() })
      .where(eq(modelValidationTable.modelVersionId, modelVersionId));
  }

  // Update model version status
  await updateVersionStatus(modelVersionId, "validating");

  const artifacts = await db
    .select()
    .from(modelArtifactsTable)
    .where(eq(modelArtifactsTable.modelVersionId, modelVersionId));

  const version = await db
    .select()
    .from(modelVersionsTable)
    .where(eq(modelVersionsTable.id, modelVersionId))
    .then((r) => r[0] ?? null);

  // Run checks
  const presentTypes = new Set(artifacts.map((a) => a.artifactType));
  const requiredArtifactsPresent = REQUIRED_ARTIFACT_TYPES.every((t) => presentTypes.has(t));

  // Verify checksums for all artifacts
  let checksumVerified = true;
  for (const artifact of artifacts) {
    const ok = await verifyArtifactChecksum(artifact.id);
    if (!ok) checksumVerified = false;
  }

  const manifestArtifact = artifacts.find((a) => a.artifactType === "manifest");
  const metadataArtifact = artifacts.find((a) => a.artifactType === "metadata");

  const manifestValid = !!manifestArtifact && manifestArtifact.status !== "corrupted";
  const metadataValid = !!metadataArtifact && metadataArtifact.status !== "corrupted";
  const frameworkVersionOk = !!(version?.frameworkVersion);
  const datasetVersionOk = !!(version?.datasetVersion);
  const inputSchemaValid = !!(version?.inputSchema);
  const outputSchemaValid = !!(version?.outputSchema);
  const artifactIntegrityOk = checksumVerified;

  const failureReasons: string[] = [];
  if (!requiredArtifactsPresent) {
    const missing = REQUIRED_ARTIFACT_TYPES.filter((t) => !presentTypes.has(t));
    failureReasons.push(`Missing required artifacts: ${missing.join(", ")}`);
  }
  if (!checksumVerified) failureReasons.push("One or more artifact checksums failed.");
  if (!manifestValid) failureReasons.push("Manifest artifact missing or corrupted.");
  if (!metadataValid) failureReasons.push("Metadata artifact missing or corrupted.");

  const passed = failureReasons.length === 0;
  const durationMs = String(Date.now() - start);

  const checklist = {
    checksum_verified: checksumVerified,
    manifest_valid: manifestValid,
    metadata_valid: metadataValid,
    required_artifacts_present: requiredArtifactsPresent,
    input_schema_valid: inputSchemaValid,
    output_schema_valid: outputSchemaValid,
    framework_version_ok: frameworkVersionOk,
    dataset_version_ok: datasetVersionOk,
    artifact_integrity_ok: artifactIntegrityOk,
  };

  const report = {
    passed,
    failure_reasons: failureReasons,
    checklist,
    artifacts_found: presentTypes.size,
    artifacts_verified: artifacts.filter((a) => a.status === "verified").length,
    duration_ms: durationMs,
    validated_at: new Date().toISOString(),
  };

  // Persist validation record
  await db
    .update(modelValidationTable)
    .set({
      status: passed ? "passed" : "failed",
      checksumVerified,
      manifestValid,
      metadataValid,
      requiredArtifactsPresent,
      inputSchemaValid,
      outputSchemaValid,
      frameworkVersionOk,
      datasetVersionOk,
      artifactIntegrityOk,
      checklist,
      validationReport: report,
      failureReasons: failureReasons.length > 0 ? failureReasons : null,
      validatedAt: new Date(),
      durationMs,
      updatedAt: new Date(),
    })
    .where(eq(modelValidationTable.modelVersionId, modelVersionId));

  // Update version status
  const newStatus = passed ? "validated" : "rejected";
  await updateVersionStatus(modelVersionId, newStatus, {
    rejectionReason: passed ? undefined : failureReasons.join("; "),
  });

  await writeAuditLog({
    eventType: passed ? "validation_completed" : "validation_failed",
    modelVersionId,
    actor,
    details: report,
  });

  return { passed, report };
}

export async function getValidationResult(modelVersionId: string) {
  const [record] = await db
    .select()
    .from(modelValidationTable)
    .where(eq(modelValidationTable.modelVersionId, modelVersionId));
  return record ?? null;
}
