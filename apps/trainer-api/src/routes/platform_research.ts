/**
 * Signalium Platform — Research Lab & Dataset contract routes.
 *
 * All eight routes are JWT-protected via platformAuth (applied in platform.ts)
 * and mounted at root level (no /api prefix).
 *
 * Contract path                          Internal RL server path
 * ─────────────────────────────────────  ───────────────────────────────────
 * GET  /experiments                   →  GET  /research/experiments
 * POST /experiments                   →  POST /research/experiments
 * GET  /training/runs                 →  GET  /artifacts/runs
 * POST /training/runs                 →  POST /artifacts/runs
 * POST /shadow-testing/deployments    →  POST /research/shadow-tests
 * POST /deployment-candidates/:id/approve → POST /research/experiments/:id/approve
 * GET  /genealogy/tree                →  GET  /genealogy/tree
 * GET  /datasets                      →  GET  /research/datasets
 *
 * The existing /api/admin/research/* routes are left entirely untouched —
 * the admin dashboard continues to use them.
 *
 * Error shape: { "error": "<description>" } — no "success" wrapper.
 */

import { Router, type IRouter, type Request, type Response } from "express";
import { logger } from "../lib/logger.js";

const router: IRouter = Router();

const RL_BASE = process.env["RESEARCH_LAB_API_URL"] ?? "http://localhost:8002";

// ---------------------------------------------------------------------------
// Internal auth headers for upstream RL calls
// ---------------------------------------------------------------------------

/**
 * Build the Authorization header the Research Lab server requires.
 * The platform-facing routes use the internal RESEARCH_LAB_ADMIN_TOKEN —
 * the incoming platform JWT must never be forwarded as the Authorization
 * bearer, but X-Signalium-Token is forwarded for identity resolution.
 *
 * Returns null when the token is not configured; callers should 503.
 */
function rlAuthHeaders(req?: Request): Record<string, string> | null {
  const token = process.env["RESEARCH_LAB_ADMIN_TOKEN"];
  if (!token) return null;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
  if (req) {
    const signaliumToken = req.headers["x-signalium-token"];
    if (signaliumToken) headers["X-Signalium-Token"] = Array.isArray(signaliumToken) ? signaliumToken[0]! : signaliumToken;
  }
  return headers;
}

// ---------------------------------------------------------------------------
// Proxy helpers (error shape: { error: "..." }, no success wrapper)
// ---------------------------------------------------------------------------

async function rlGet(path: string, res: Response, req?: Request): Promise<void> {
  const headers = rlAuthHeaders(req);
  if (!headers) {
    res.status(503).json({ error: "Research Lab is not configured. Set RESEARCH_LAB_ADMIN_TOKEN." });
    return;
  }
  try {
    const upstream = await fetch(`${RL_BASE}${path}`, { headers });
    const data = await upstream.json().catch(() => ({}));
    res.status(upstream.status).json(data);
  } catch (err) {
    logger.error({ err, path }, "platform_research: RL service unreachable");
    res.status(502).json({ error: `Research Lab service unreachable: ${String(err)}` });
  }
}

async function rlPost(path: string, body: unknown, res: Response, req?: Request): Promise<void> {
  const headers = rlAuthHeaders(req);
  if (!headers) {
    res.status(503).json({ error: "Research Lab is not configured. Set RESEARCH_LAB_ADMIN_TOKEN." });
    return;
  }
  try {
    const upstream = await fetch(`${RL_BASE}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    const data = await upstream.json().catch(() => ({}));
    res.status(upstream.status).json(data);
  } catch (err) {
    logger.error({ err, path }, "platform_research: RL service unreachable");
    res.status(502).json({ error: `Research Lab service unreachable: ${String(err)}` });
  }
}

// ---------------------------------------------------------------------------
// GET /experiments  — list ML experiments
// POST /experiments — create a new ML experiment
// ---------------------------------------------------------------------------

router.get("/experiments", async (req: Request, res: Response): Promise<void> => {
  await rlGet("/research/experiments", res, req);
});

router.post("/experiments", async (req: Request, res: Response): Promise<void> => {
  await rlPost("/research/experiments", req.body ?? {}, res, req);
});

// ---------------------------------------------------------------------------
// GET  /training/runs — list training runs (artifact lineage)
// POST /training/runs — register a new training run
// ---------------------------------------------------------------------------

router.get("/training/runs", async (req: Request, res: Response): Promise<void> => {
  const p = new URLSearchParams();
  if (req.query["experiment_id"]) p.set("experiment_id", String(req.query["experiment_id"]));
  if (req.query["limit"]) p.set("limit", String(req.query["limit"]));
  const qs = p.toString() ? `?${p.toString()}` : "";
  await rlGet(`/artifacts/runs${qs}`, res, req);
});

router.post("/training/runs", async (req: Request, res: Response): Promise<void> => {
  await rlPost("/artifacts/runs", req.body ?? {}, res, req);
});

// ---------------------------------------------------------------------------
// POST /shadow-testing/deployments — create a shadow test deployment
// ---------------------------------------------------------------------------

router.post("/shadow-testing/deployments", async (req: Request, res: Response): Promise<void> => {
  await rlPost("/research/shadow-tests", req.body ?? {}, res, req);
});

// ---------------------------------------------------------------------------
// POST /deployment-candidates/:id/approve — approve an experiment for deployment
// ---------------------------------------------------------------------------

router.post(
  "/deployment-candidates/:id/approve",
  async (req: Request, res: Response): Promise<void> => {
    const id = req.params["id"];
    await rlPost(`/research/experiments/${id}/approve`, req.body ?? {}, res, req);
  },
);

// ---------------------------------------------------------------------------
// GET /genealogy/tree — model version lineage tree
// ---------------------------------------------------------------------------

router.get("/genealogy/tree", async (req: Request, res: Response): Promise<void> => {
  const qs = req.query["limit"] ? `?limit=${req.query["limit"]}` : "";
  await rlGet(`/genealogy/tree${qs}`, res, req);
});

// ---------------------------------------------------------------------------
// GET /datasets — list available training datasets
// ---------------------------------------------------------------------------

router.get("/datasets", async (req: Request, res: Response): Promise<void> => {
  await rlGet("/research/datasets", res, req);
});

export default router;
