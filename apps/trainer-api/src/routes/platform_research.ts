/**
 * Signalium Platform — Research Lab & Dataset contract routes.
 *
 * All routes are JWT-protected via platformAuth (applied in platform.ts)
 * and mounted at root level (no /api prefix), so the Signalium Platform
 * reaches them at https://trainer.signaliumai.com/<path>.
 *
 * Every route proxies to the Research Lab Python service at RL_BASE,
 * using the internal RESEARCH_LAB_ADMIN_TOKEN — the incoming platform
 * JWT is never forwarded as the RL bearer.
 *
 * Contract path                                    RL path (same unless noted)
 * ──────────────────────────────────────────────── ──────────────────────────
 * GET  /experiments                             →  /experiments
 * POST /experiments                             →  /research/experiments
 * GET  /experiments/:id                         →  /experiments/:id
 * GET  /experiments/:id/lineage                 →  /experiments/:id/lineage
 * GET  /experiments/:id/history                 →  /experiments/:id/history
 * GET  /training/runs                           →  /training/runs
 * POST /training/runs                           →  /training/runs
 * GET  /training/runs/:id                       →  /training/runs/:id
 * POST /training/runs/:id/stop                  →  /training/runs/:id/stop
 * GET  /training/resource-usage                 →  /training/resource-usage
 * GET  /validation/reports                      →  /validation/reports
 * GET  /validation/reports/:id                  →  /validation/reports/:id
 * GET  /validation/drift                        →  /validation/drift
 * GET  /validation/robustness                   →  /validation/robustness
 * GET  /shadow-testing/deployments              →  /shadow-testing/deployments
 * POST /shadow-testing/deployments              →  /shadow-testing/deployments
 * GET  /shadow-testing/deployments/:id/...      →  /shadow-testing/deployments/:id/...
 * GET  /deployment-candidates                   →  /deployment-candidates
 * POST /deployment-candidates                   →  /deployment-candidates
 * GET  /deployment-candidates/history           →  /deployment-candidates/history
 * GET  /deployment-candidates/:id               →  /deployment-candidates/:id
 * POST /deployment-candidates/:id/approve       →  /deployment-candidates/:id/approve
 * POST /deployment-candidates/:id/rollback      →  /deployment-candidates/:id/rollback
 * GET  /research-intelligence/features          →  /research-intelligence/features
 * GET  /research-intelligence/failure-...       →  /research-intelligence/failure-investigations
 * GET  /research-intelligence/self-evaluations  →  /research-intelligence/self-evaluations
 * GET  /research-intelligence/retraining-...    →  /research-intelligence/retraining-recommendations
 * GET  /genealogy/tree                          →  /genealogy/tree
 * GET  /genealogy/:modelVersion/artifacts       →  /genealogy/:modelVersion/artifacts
 * GET  /genealogy/:modelVersion                 →  /genealogy/:modelVersion
 * GET  /datasets                                →  /datasets
 * GET  /datasets/:id                            →  /datasets/:id
 * GET  /memory                                  →  /memory
 * GET  /reflections                             →  /reflections
 * GET  /research-reports                        →  /research-reports
 * GET  /research-reports/:id                    →  /research-reports/:id
 * GET  /models/active                           →  /models/active
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

function rlAuthHeaders(req?: Request): Record<string, string> | null {
  const token = process.env["RESEARCH_LAB_ADMIN_TOKEN"];
  if (!token) return null;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
  if (req) {
    const signaliumToken = req.headers["x-signalium-token"];
    if (signaliumToken)
      headers["X-Signalium-Token"] = Array.isArray(signaliumToken)
        ? signaliumToken[0]!
        : signaliumToken;
  }
  return headers;
}

// ---------------------------------------------------------------------------
// Proxy helpers
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

/** Forward query string from the incoming request as-is. */
function qs(req: Request): string {
  const raw = req.url.split("?")[1];
  return raw ? `?${raw}` : "";
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

router.get("/experiments", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/experiments${qs(req)}`, res, req);
});

router.post("/experiments", async (req: Request, res: Response): Promise<void> => {
  // POST goes to the canonical /research/experiments on the RL
  await rlPost("/research/experiments", req.body ?? {}, res, req);
});

router.get("/experiments/:id/lineage", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/experiments/${req.params["id"]}/lineage`, res, req);
});

router.get("/experiments/:id/history", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/experiments/${req.params["id"]}/history`, res, req);
});

router.get("/experiments/:id", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/experiments/${req.params["id"]}`, res, req);
});

// ---------------------------------------------------------------------------
// Training runs
// ---------------------------------------------------------------------------

router.get("/training/runs", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/training/runs${qs(req)}`, res, req);
});

router.post("/training/runs", async (req: Request, res: Response): Promise<void> => {
  await rlPost("/training/runs", req.body ?? {}, res, req);
});

router.post("/training/runs/:id/stop", async (req: Request, res: Response): Promise<void> => {
  await rlPost(`/training/runs/${req.params["id"]}/stop`, req.body ?? {}, res, req);
});

router.get("/training/runs/:id", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/training/runs/${req.params["id"]}`, res, req);
});

router.get("/training/resource-usage", async (req: Request, res: Response): Promise<void> => {
  await rlGet("/training/resource-usage", res, req);
});

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

router.get("/validation/reports", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/validation/reports${qs(req)}`, res, req);
});

router.get("/validation/reports/:id", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/validation/reports/${req.params["id"]}`, res, req);
});

router.get("/validation/drift", async (req: Request, res: Response): Promise<void> => {
  await rlGet("/validation/drift", res, req);
});

router.get("/validation/robustness", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/validation/robustness${qs(req)}`, res, req);
});

// ---------------------------------------------------------------------------
// Shadow testing
// NOTE: specific sub-paths must be registered before /:id
// ---------------------------------------------------------------------------

router.get(
  "/shadow-testing/deployments/:id/comparison",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/shadow-testing/deployments/${req.params["id"]}/comparison`, res, req);
  },
);

router.get(
  "/shadow-testing/deployments/:id/agreement",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/shadow-testing/deployments/${req.params["id"]}/agreement`, res, req);
  },
);

router.get(
  "/shadow-testing/deployments/:id/regimes",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/shadow-testing/deployments/${req.params["id"]}/regimes`, res, req);
  },
);

router.get("/shadow-testing/deployments", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/shadow-testing/deployments${qs(req)}`, res, req);
});

router.post("/shadow-testing/deployments", async (req: Request, res: Response): Promise<void> => {
  await rlPost("/shadow-testing/deployments", req.body ?? {}, res, req);
});

// ---------------------------------------------------------------------------
// Deployment candidates
// NOTE: /history must be registered before /:id
// ---------------------------------------------------------------------------

router.get(
  "/deployment-candidates/history",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/deployment-candidates/history${qs(req)}`, res, req);
  },
);

router.get("/deployment-candidates", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/deployment-candidates${qs(req)}`, res, req);
});

router.post("/deployment-candidates", async (req: Request, res: Response): Promise<void> => {
  await rlPost("/deployment-candidates", req.body ?? {}, res, req);
});

router.post(
  "/deployment-candidates/:id/approve",
  async (req: Request, res: Response): Promise<void> => {
    await rlPost(
      `/deployment-candidates/${req.params["id"]}/approve`,
      req.body ?? {},
      res,
      req,
    );
  },
);

router.post(
  "/deployment-candidates/:id/rollback",
  async (req: Request, res: Response): Promise<void> => {
    await rlPost(
      `/deployment-candidates/${req.params["id"]}/rollback`,
      req.body ?? {},
      res,
      req,
    );
  },
);

router.get("/deployment-candidates/:id", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/deployment-candidates/${req.params["id"]}`, res, req);
});

// ---------------------------------------------------------------------------
// Research Intelligence
// ---------------------------------------------------------------------------

router.get(
  "/research-intelligence/features",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/research-intelligence/features${qs(req)}`, res, req);
  },
);

router.get(
  "/research-intelligence/failure-investigations",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/research-intelligence/failure-investigations${qs(req)}`, res, req);
  },
);

router.get(
  "/research-intelligence/self-evaluations",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/research-intelligence/self-evaluations${qs(req)}`, res, req);
  },
);

router.get(
  "/research-intelligence/retraining-recommendations",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/research-intelligence/retraining-recommendations${qs(req)}`, res, req);
  },
);

// ---------------------------------------------------------------------------
// Genealogy
// NOTE: /tree is a fixed path; /:modelVersion/artifacts before /:modelVersion
// ---------------------------------------------------------------------------

router.get("/genealogy/tree", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/genealogy/tree${qs(req)}`, res, req);
});

router.get(
  "/genealogy/:modelVersion/artifacts",
  async (req: Request, res: Response): Promise<void> => {
    await rlGet(`/genealogy/${req.params["modelVersion"]}/artifacts`, res, req);
  },
);

router.get("/genealogy/:modelVersion", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/genealogy/${req.params["modelVersion"]}`, res, req);
});

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

router.get("/datasets", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/datasets${qs(req)}`, res, req);
});

router.get("/datasets/:id", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/datasets/${req.params["id"]}`, res, req);
});

// ---------------------------------------------------------------------------
// Memory, Reflections, Research Reports, Active Models
// ---------------------------------------------------------------------------

router.get("/memory", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/memory${qs(req)}`, res, req);
});

router.get("/reflections", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/reflections${qs(req)}`, res, req);
});

router.get("/research-reports", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/research-reports${qs(req)}`, res, req);
});

router.get("/research-reports/:id", async (req: Request, res: Response): Promise<void> => {
  await rlGet(`/research-reports/${req.params["id"]}`, res, req);
});

router.get("/models/active", async (req: Request, res: Response): Promise<void> => {
  await rlGet("/models/active", res, req);
});

export default router;
