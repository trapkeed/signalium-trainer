import { Router } from "express";
import { createReadStream, existsSync } from "fs";
import { requireApiKey, requireAdminApiKey, attachActor } from "../../middlewares/apiKey.js";
import {
  streamArtifactToDisk, createArtifactRecord, listArtifacts, getArtifactById,
  artifactStoragePath,
} from "../../services/artifactService.js";
import { getVersionByModelAndId } from "../../services/registryService.js";
import { writeAuditLog } from "../../services/auditService.js";
import { enqueueChecksumVerification } from "../../workers/checksumWorker.js";
import type { ArtifactType } from "@workspace/db";

const router = Router({ mergeParams: true });

const ARTIFACT_TYPES: string[] = [
  "torchscript", "onnx", "manifest", "metadata",
  "training_report", "evaluation_report", "logs", "config", "other",
];

// GET /api/registry/models/:modelId/versions/:versionId/artifacts
router.get("/", requireApiKey, async (req, res) => {
  const version = await getVersionByModelAndId(String(req.params.modelId), String(req.params.versionId));
  if (!version) { res.status(404).json({ error: "Version not found." }); return; }
  const artifacts = await listArtifacts(version.id);
  res.json({ artifacts });
});

// POST /api/registry/models/:modelId/versions/:versionId/artifacts
// Accepts binary body with metadata in headers
router.post("/", requireAdminApiKey, attachActor, async (req, res) => {
  const version = await getVersionByModelAndId(String(req.params.modelId), String(req.params.versionId));
  if (!version) { res.status(404).json({ error: "Version not found." }); return; }

  const artifactType = String(req.headers["x-artifact-type"] ?? "other");
  const filename = String(req.headers["x-artifact-filename"] ?? "artifact");
  const mimeType = String(req.headers["content-type"] ?? "application/octet-stream");

  if (!ARTIFACT_TYPES.includes(artifactType)) {
    res.status(400).json({ error: `Invalid artifact type. Allowed: ${ARTIFACT_TYPES.join(", ")}` });
    return;
  }

  const storagePath = artifactStoragePath(version.id, filename);
  const { sizeBytes, checksum } = await streamArtifactToDisk(req, storagePath);

  const artifact = await createArtifactRecord({
    modelVersionId: version.id,
    artifactType: artifactType as ArtifactType,
    filename,
    storagePath,
    mimeType: mimeType.split(";")[0].trim(),
    sizeBytes,
    checksum,
    uploadedBy: String(res.locals.actor),
  });

  enqueueChecksumVerification(artifact.id);

  await writeAuditLog(
    { eventType: "artifact_uploaded", modelId: String(req.params.modelId), modelVersionId: version.id, actor: String(res.locals.actor), details: { artifactType, filename, sizeBytes } },
    req,
  );

  res.status(201).json({ artifact });
});

// GET /api/registry/models/:modelId/versions/:versionId/artifacts/download/:artifactId
router.get("/download/:artifactId", requireApiKey, attachActor, async (req, res) => {
  const artifact = await getArtifactById(String(req.params.artifactId));
  if (!artifact) { res.status(404).json({ error: "Artifact not found." }); return; }

  if (!existsSync(artifact.storagePath)) {
    res.status(404).json({ error: "Artifact file not found on storage." });
    return;
  }

  await writeAuditLog(
    { eventType: "artifact_downloaded", modelVersionId: artifact.modelVersionId, actor: String(res.locals.actor), details: { artifactId: artifact.id, filename: artifact.filename } },
    req,
  );

  res.setHeader("Content-Disposition", `attachment; filename="${artifact.filename}"`);
  res.setHeader("Content-Type", artifact.mimeType);
  res.setHeader("X-Artifact-Checksum", artifact.checksum ?? "");
  createReadStream(artifact.storagePath).pipe(res);
});

export default router;
