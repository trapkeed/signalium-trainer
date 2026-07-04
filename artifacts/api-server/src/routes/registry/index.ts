import { Router } from "express";
import { rateLimiter } from "../../middlewares/rateLimit.js";
import { replayProtection } from "../../middlewares/replayProtection.js";
import { requireApiKey } from "../../middlewares/apiKey.js";
import { getRegistryStats } from "../../services/searchService.js";
import modelsRouter from "./models.js";
import versionsRouter from "./versions.js";
import artifactsRouter from "./artifacts.js";
import promotionRouter from "./promotion.js";
import searchRouter from "./search.js";

const registry = Router();

// Apply rate limiting and replay protection to all registry endpoints
registry.use(rateLimiter(200, 60_000));
registry.use(replayProtection);

// Registry statistics — direct route to avoid router mount collision
registry.get("/stats", requireApiKey, async (_req, res) => {
  const stats = await getRegistryStats();
  res.json(stats);
});

// Model CRUD + tags
registry.use("/models", modelsRouter);

// Version CRUD + metrics + validation results
registry.use("/models/:modelId/versions", versionsRouter);

// Artifact upload + download
registry.use("/models/:modelId/versions/:versionId/artifacts", artifactsRouter);

// Lifecycle actions: validate, promote, reject, archive, rollback, aliases, deployments
registry.use("/models/:modelId/versions/:versionId", promotionRouter);
registry.use("/models/:modelId", promotionRouter);

// Search (GET /)
registry.use("/search", searchRouter);

export default registry;
