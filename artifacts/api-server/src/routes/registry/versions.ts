import { Router } from "express";
import { z } from "zod";
import { requireApiKey, requireAdminApiKey, attachActor } from "../../middlewares/apiKey.js";
import {
  createModelVersion, getVersionByModelAndId, listVersionsByModel,
} from "../../services/registryService.js";
import { getMetrics, recordMetrics } from "../../services/metricsService.js";
import { getValidationResult } from "../../services/validationService.js";
import { writeAuditLog } from "../../services/auditService.js";
import { db } from "@workspace/db";
import { modelAuditLogTable } from "@workspace/db";
import { eq } from "drizzle-orm";

const router = Router({ mergeParams: true });

const CreateVersionBody = z.object({
  version: z.string().min(1).max(64),
  architecture: z.string().min(1).max(128),
  datasetVersion: z.string().default(""),
  featureVersion: z.string().default(""),
  labelVersion: z.string().default(""),
  frameworkVersion: z.string().default(""),
  trainerVersion: z.string().default(""),
  configVersion: z.string().default(""),
  experimentId: z.string().optional(),
  gitCommitHash: z.string().optional(),
  mlflowRunId: z.string().optional(),
  inputSchema: z.record(z.unknown()).optional(),
  outputSchema: z.record(z.unknown()).optional(),
  configSnapshot: z.record(z.unknown()).optional(),
  createdBy: z.string().min(1).max(256),
});

// GET /api/registry/models/:modelId/versions
router.get("/", requireApiKey, async (req, res) => {
  const versions = await listVersionsByModel(String(req.params.modelId));
  res.json({ versions });
});

// POST /api/registry/models/:modelId/versions
router.post("/", requireAdminApiKey, attachActor, async (req, res) => {
  const parsed = CreateVersionBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Validation failed", issues: parsed.error.issues });
    return;
  }
  const version = await createModelVersion({ ...parsed.data, modelId: String(req.params.modelId) });
  await writeAuditLog(
    { eventType: "version_created", modelId: String(req.params.modelId), modelVersionId: version.id, actor: String(res.locals.actor), details: { version: version.version } },
    req,
  );
  res.status(201).json({ version });
});

// GET /api/registry/models/:modelId/versions/:versionId
router.get("/:versionId", requireApiKey, async (req, res) => {
  const version = await getVersionByModelAndId(String(req.params.modelId), String(req.params.versionId));
  if (!version) { res.status(404).json({ error: "Version not found." }); return; }
  res.json({ version });
});

// GET /api/registry/models/:modelId/versions/:versionId/metrics
router.get("/:versionId/metrics", requireApiKey, async (req, res) => {
  const version = await getVersionByModelAndId(String(req.params.modelId), String(req.params.versionId));
  if (!version) { res.status(404).json({ error: "Version not found." }); return; }
  const metrics = await getMetrics(version.id);
  res.json(metrics);
});

// POST /api/registry/models/:modelId/versions/:versionId/metrics
router.post("/:versionId/metrics", requireAdminApiKey, attachActor, async (req, res) => {
  const version = await getVersionByModelAndId(String(req.params.modelId), String(req.params.versionId));
  if (!version) { res.status(404).json({ error: "Version not found." }); return; }

  const MetricsBody = z.object({
    metrics: z.array(z.object({
      category: z.enum(["training", "validation", "evaluation", "inference", "operational"]).default("evaluation"),
      metricKey: z.string().min(1),
      metricValue: z.number(),
      unit: z.string().default(""),
    })),
  });

  const parsed = MetricsBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Validation failed", issues: parsed.error.issues });
    return;
  }

  const entries = parsed.data.metrics.map((m) => ({ ...m, modelVersionId: version.id }));
  const saved = await recordMetrics(entries);
  res.status(201).json({ metrics: saved });
});

// GET /api/registry/models/:modelId/versions/:versionId/validation
router.get("/:versionId/validation", requireApiKey, async (req, res) => {
  const version = await getVersionByModelAndId(String(req.params.modelId), String(req.params.versionId));
  if (!version) { res.status(404).json({ error: "Version not found." }); return; }
  const result = await getValidationResult(version.id);
  res.json({ validation: result });
});

// GET /api/registry/models/:modelId/versions/history/audit
router.get("/history/audit", requireApiKey, async (req, res) => {
  const logs = await db
    .select()
    .from(modelAuditLogTable)
    .where(eq(modelAuditLogTable.modelId, String(req.params.modelId)));
  res.json({ history: logs });
});

export default router;
