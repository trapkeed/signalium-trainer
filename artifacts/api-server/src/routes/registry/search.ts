import { Router } from "express";
import { requireApiKey } from "../../middlewares/apiKey.js";
import { searchRegistry } from "../../services/searchService.js";
import type { ModelLifecycleStatus } from "@workspace/db";

const router = Router();

// GET /api/registry/search
router.get("/", requireApiKey, async (req, res) => {
  const q = req.query as Record<string, string | undefined>;
  const results = await searchRegistry({
    architecture: q.architecture,
    status: q.status as ModelLifecycleStatus | undefined,
    datasetVersion: q.datasetVersion,
    featureVersion: q.featureVersion,
    labelVersion: q.labelVersion,
    experimentId: q.experimentId,
    gitCommit: q.gitCommit,
    trainerVersion: q.trainerVersion,
    mlflowRunId: q.mlflowRunId,
    createdAfter: q.createdAfter,
    createdBefore: q.createdBefore,
    page: q.page ? parseInt(q.page, 10) : 1,
    limit: q.limit ? parseInt(q.limit, 10) : 20,
  });
  res.json(results);
});

export default router;
