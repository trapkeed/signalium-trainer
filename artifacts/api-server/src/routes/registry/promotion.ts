import { Router } from "express";
import { z } from "zod";
import { requireAdminApiKey, requireApiKey, attachActor } from "../../middlewares/apiKey.js";
import {
  promoteVersion, rejectVersion, rollbackProduction,
  getAlias, listAliases, getDeploymentHistory, getRollbackHistory,
} from "../../services/promotionService.js";
import { runValidation } from "../../services/validationService.js";
import { getVersionByModelAndId } from "../../services/registryService.js";
import type { ModelLifecycleStatus } from "@workspace/db";

const router = Router({ mergeParams: true });

const PROMOTABLE_STATUSES = [
  "uploaded", "validating", "validated", "candidate", "challenger",
  "champion", "production", "retired", "archived",
] as const;

// POST /api/registry/models/:modelId/versions/:versionId/validate
router.post("/validate", requireAdminApiKey, attachActor, async (req, res) => {
  const version = await getVersionByModelAndId(String(req.params.modelId), String(req.params.versionId));
  if (!version) { res.status(404).json({ error: "Version not found." }); return; }

  const { passed, report } = await runValidation(version.id, String(res.locals.actor));
  res.status(passed ? 200 : 422).json({ passed, report });
});

// POST /api/registry/models/:modelId/versions/:versionId/promote
router.post("/promote", requireAdminApiKey, attachActor, async (req, res) => {
  const parsed = z.object({
    targetStatus: z.enum(PROMOTABLE_STATUSES),
  }).safeParse(req.body);

  if (!parsed.success) {
    res.status(400).json({ error: "Validation failed", issues: parsed.error.issues });
    return;
  }

  const result = await promoteVersion(
    String(req.params.modelId),
    String(req.params.versionId),
    parsed.data.targetStatus as ModelLifecycleStatus,
    String(res.locals.actor),
  );

  if (!result.success) {
    res.status(422).json({ error: result.error });
    return;
  }
  res.json({ success: true, version: result.version });
});

// POST /api/registry/models/:modelId/versions/:versionId/reject
router.post("/reject", requireAdminApiKey, attachActor, async (req, res) => {
  const parsed = z.object({ reason: z.string().min(1) }).safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Validation failed", issues: parsed.error.issues });
    return;
  }
  const result = await rejectVersion(
    String(req.params.modelId),
    String(req.params.versionId),
    parsed.data.reason,
    String(res.locals.actor),
  );
  if (!result.success) {
    res.status(422).json({ error: result.error });
    return;
  }
  res.json({ success: true });
});

// POST /api/registry/models/:modelId/versions/:versionId/archive
router.post("/archive", requireAdminApiKey, attachActor, async (req, res) => {
  const result = await promoteVersion(
    String(req.params.modelId),
    String(req.params.versionId),
    "archived",
    String(res.locals.actor),
  );
  if (!result.success) {
    res.status(422).json({ error: result.error });
    return;
  }
  res.json({ success: true });
});

// POST /api/registry/models/:modelId/rollback
router.post("/rollback", requireAdminApiKey, attachActor, async (req, res) => {
  const parsed = z.object({ toVersionId: z.string(), reason: z.string().min(1) }).safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Validation failed", issues: parsed.error.issues });
    return;
  }
  const result = await rollbackProduction(
    String(req.params.modelId),
    parsed.data.toVersionId,
    parsed.data.reason,
    String(res.locals.actor),
  );
  if (!result.success) {
    res.status(422).json({ error: result.error });
    return;
  }
  res.json({ success: true });
});

// GET /api/registry/models/:modelId/aliases
router.get("/aliases", requireApiKey, async (req, res) => {
  const aliases = await listAliases(String(req.params.modelId));
  res.json({ aliases });
});

// GET /api/registry/models/:modelId/aliases/:aliasName
router.get("/aliases/:aliasName", requireApiKey, async (req, res) => {
  const alias = await getAlias(String(req.params.modelId), String(req.params.aliasName));
  if (!alias) { res.status(404).json({ error: "Alias not found." }); return; }
  res.json({ alias });
});

// GET /api/registry/models/:modelId/deployments
router.get("/deployments", requireApiKey, async (req, res) => {
  const history = await getDeploymentHistory(String(req.params.modelId));
  res.json({ deployments: history });
});

// GET /api/registry/models/:modelId/rollbacks
router.get("/rollbacks", requireApiKey, async (req, res) => {
  const history = await getRollbackHistory(String(req.params.modelId));
  res.json({ rollbacks: history });
});

export default router;
