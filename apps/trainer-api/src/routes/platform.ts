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

const platformRouter: IRouter = Router();

// Apply JWT auth to every route in this router.
platformRouter.use(platformAuth);

// Routes are added by tasks #11 (inference) and #12 (research lab).
// The router is intentionally empty here — the middleware boundary is
// what this task establishes.

export default platformRouter;
