/**
 * Platform-facing router.
 *
 * All routes registered here are protected by the ES256 JWT platformAuth
 * middleware. This router is mounted at the Express root (no /api prefix)
 * so the Signalium Platform can reach them at https://trainer.signaliumai.com/<path>.
 *
 * Inference endpoints (POST /, POST /qde/inference, …) and Research Lab
 * contract routes (GET /experiments, GET /training/runs, …) are added by
 * subsequent tasks (#11 and #12). This file establishes the middleware
 * boundary so those tasks can import `platformRouter` directly.
 */

import { Router, type IRouter } from "express";
import { platformAuth } from "../middlewares/platform_auth.js";
import inferenceRouter from "./inference.js";
import platformResearchRouter from "./platform_research.js";

const platformRouter: IRouter = Router();

// Apply JWT auth to every route in this router.
platformRouter.use(platformAuth);

// Inference endpoints: POST /, POST /qde/inference,
// POST /qde/review-position, POST /qde/portfolio-review
platformRouter.use(inferenceRouter);

// Research Lab & Dataset contract routes:
// GET|POST /experiments, GET|POST /training/runs,
// POST /shadow-testing/deployments,
// POST /deployment-candidates/:id/approve,
// GET /genealogy/tree, GET /datasets
platformRouter.use(platformResearchRouter);

export default platformRouter;
