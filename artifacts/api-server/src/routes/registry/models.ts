import { Router } from "express";
import { z } from "zod";
import { requireApiKey, requireAdminApiKey, attachActor } from "../../middlewares/apiKey.js";
import {
  createModel, getModelById, listModels, softDeleteModel,
  addTag, listTags, removeTag,
} from "../../services/registryService.js";
import { writeAuditLog } from "../../services/auditService.js";

const router = Router();

const CreateModelBody = z.object({
  name: z.string().min(1).max(128).regex(/^[a-z0-9_-]+$/),
  displayName: z.string().min(1).max(256),
  description: z.string().max(2048).default(""),
  architecture: z.string().min(1).max(128),
  framework: z.string().default("pytorch"),
  createdBy: z.string().min(1).max(256),
});

// GET /api/registry/models
router.get("/", requireApiKey, attachActor, async (req, res) => {
  const { architecture, name } = req.query as Record<string, string | undefined>;
  const models = await listModels({ architecture, name });
  res.json({ models });
});

// POST /api/registry/models
router.post("/", requireAdminApiKey, attachActor, async (req, res) => {
  const parsed = CreateModelBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Validation failed", issues: parsed.error.issues });
    return;
  }
  const model = await createModel(parsed.data);
  await writeAuditLog(
    { eventType: "model_created", modelId: model.id, actor: String(res.locals.actor), details: { name: model.name } },
    req,
  );
  res.status(201).json({ model });
});

// GET /api/registry/models/:modelId
router.get("/:modelId", requireApiKey, async (req, res) => {
  const model = await getModelById(String(req.params.modelId));
  if (!model) { res.status(404).json({ error: "Model not found." }); return; }
  res.json({ model });
});

// DELETE /api/registry/models/:modelId (soft delete)
router.delete("/:modelId", requireAdminApiKey, attachActor, async (req, res) => {
  const model = await softDeleteModel(String(req.params.modelId));
  if (!model) { res.status(404).json({ error: "Model not found." }); return; }
  await writeAuditLog(
    { eventType: "model_deleted", modelId: model.id, actor: String(res.locals.actor) },
    req,
  );
  res.json({ success: true });
});

// GET /api/registry/models/:modelId/tags
router.get("/:modelId/tags", requireApiKey, async (req, res) => {
  const tags = await listTags(String(req.params.modelId));
  res.json({ tags });
});

// POST /api/registry/models/:modelId/tags
router.post("/:modelId/tags", requireAdminApiKey, attachActor, async (req, res) => {
  const parsed = z.object({ key: z.string().min(1), value: z.string().default("") }).safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Validation failed", issues: parsed.error.issues });
    return;
  }
  const { key, value } = parsed.data;
  const tag = await addTag(String(req.params.modelId), key, value);
  await writeAuditLog({ eventType: "tag_added", modelId: String(req.params.modelId), actor: String(res.locals.actor), details: { key, value } }, req);
  res.status(201).json({ tag });
});

// DELETE /api/registry/models/:modelId/tags/:tagId
router.delete("/:modelId/tags/:tagId", requireAdminApiKey, attachActor, async (req, res) => {
  const deleted = await removeTag(String(req.params.tagId));
  if (!deleted) { res.status(404).json({ error: "Tag not found." }); return; }
  await writeAuditLog({ eventType: "tag_removed", modelId: String(req.params.modelId), actor: String(res.locals.actor), details: { tagId: req.params.tagId } }, req);
  res.json({ success: true });
});

export default router;
