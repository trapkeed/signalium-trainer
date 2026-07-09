import { Router, type IRouter, type Request, type Response } from "express";
import { adminAuth } from "../middlewares/admin_auth";
import { requireResearchLabAuthorization } from "../middlewares/require_rl_authorization";
import { requirePlatformOrigin } from "../middlewares/platform_trust";

/**
 * QDE lifecycle routes.
 *
 * ACCESS MODEL
 * ============
 * Most QDE routes (registry reads, pipeline steps, A/B testing, position
 * management) are internal-service routes used by the training/backtesting
 * pipeline itself and are not lifecycle-authority-gated.
 *
 * However, the four QDE version LIFECYCLE MUTATIONS — promote, rollback,
 * archive, retire — permanently change what model is live in production.
 * Research Lab is the SOLE authority permitted to approve these actions
 * (see docs/research_lab/permissions_matrix.md). Accordingly those routes:
 *   1. Require adminAuth (the same admin token Research Lab uses).
 *   2. Require a matching APPROVED, unconsumed Research Lab
 *      LifecycleAuthorization, verified + atomically consumed via
 *      requireResearchLabAuthorization before the mutation is proxied to
 *      qde_server.py.
 */

const router: IRouter = Router();

const QDE_BASE = process.env["QDE_API_URL"] || "http://localhost:8001";

async function proxyPost(
  path: string,
  body: unknown,
  res: Response,
): Promise<void> {
  try {
    const upstream = await fetch(`${QDE_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ success: false, error: String(err) });
  }
}

async function proxyGet(path: string, res: Response): Promise<void> {
  try {
    const upstream = await fetch(`${QDE_BASE}${path}`);
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ success: false, error: String(err) });
  }
}

// ── Registry ──────────────────────────────────────────────────────────────────

router.post("/qde/versions", async (req: Request, res: Response) => {
  await proxyPost("/qde/versions", req.body, res);
});

router.get("/qde/versions", async (req: Request, res: Response) => {
  const stage = req.query["stage"] ? `?stage=${req.query["stage"]}` : "";
  await proxyGet(`/qde/versions${stage}`, res);
});

router.get("/qde/versions/:versionId", async (req: Request, res: Response) => {
  await proxyGet(`/qde/versions/${req.params["versionId"]}`, res);
});

router.get(
  "/qde/versions/:versionId/lineage",
  async (req: Request, res: Response) => {
    await proxyGet(`/qde/versions/${req.params["versionId"]}/lineage`, res);
  },
);

router.get("/qde/active", async (_req: Request, res: Response) => {
  await proxyGet("/qde/active", res);
});

router.get("/qde/summary", async (_req: Request, res: Response) => {
  await proxyGet("/qde/summary", res);
});

// ── Pipeline ──────────────────────────────────────────────────────────────────

router.post(
  "/qde/versions/:versionId/pipeline/start",
  async (req: Request, res: Response) => {
    await proxyPost(
      `/qde/versions/${req.params["versionId"]}/pipeline/start`,
      req.body,
      res,
    );
  },
);

router.post(
  "/qde/pipeline/:runId/advance",
  async (req: Request, res: Response) => {
    await proxyPost(`/qde/pipeline/${req.params["runId"]}/advance`, req.body, res);
  },
);

router.post(
  "/qde/pipeline/:runId/fail",
  async (req: Request, res: Response) => {
    await proxyPost(`/qde/pipeline/${req.params["runId"]}/fail`, req.body, res);
  },
);

router.post(
  "/qde/pipeline/:runId/skip",
  async (req: Request, res: Response) => {
    await proxyPost(`/qde/pipeline/${req.params["runId"]}/skip`, req.body, res);
  },
);

router.get(
  "/qde/pipeline/:runId",
  async (req: Request, res: Response) => {
    await proxyGet(`/qde/pipeline/${req.params["runId"]}`, res);
  },
);

router.get("/qde/pipeline", async (_req: Request, res: Response) => {
  await proxyGet("/qde/pipeline", res);
});

// ── Promotion ─────────────────────────────────────────────────────────────────

router.post(
  "/qde/versions/:versionId/promotion/evaluate",
  async (req: Request, res: Response) => {
    await proxyPost(
      `/qde/versions/${req.params["versionId"]}/promotion/evaluate`,
      req.body,
      res,
    );
  },
);

router.post(
  "/qde/versions/:versionId/promotion/approve",
  requirePlatformOrigin,
  adminAuth,
  requireResearchLabAuthorization("PROMOTE", (req) => String(req.params["versionId"])),
  async (req: Request, res: Response) => {
    await proxyPost(
      `/qde/versions/${req.params["versionId"]}/promotion/approve`,
      req.body,
      res,
    );
  },
);

router.post(
  "/qde/versions/:versionId/promotion/auto",
  requirePlatformOrigin,
  adminAuth,
  requireResearchLabAuthorization("PROMOTE", (req) => String(req.params["versionId"])),
  async (req: Request, res: Response) => {
    await proxyPost(
      `/qde/versions/${req.params["versionId"]}/promotion/auto`,
      req.body,
      res,
    );
  },
);

router.get("/qde/promotion/history", async (_req: Request, res: Response) => {
  await proxyGet("/qde/promotion/history", res);
});

router.get("/qde/promotion/criteria", async (_req: Request, res: Response) => {
  await proxyGet("/qde/promotion/criteria", res);
});

// ── Rollback ──────────────────────────────────────────────────────────────────

// Rollback authorizations from Research Lab are keyed by the CURRENT active
// version (the one being rolled back FROM), since that is what
// `approve_rollback` records as `version_id`. Resolve it from the registry
// rather than trusting the request body/params.
async function currentActiveVersionId(): Promise<string | undefined> {
  try {
    const upstream = await fetch(`${QDE_BASE}/qde/active`);
    const data = (await upstream.json()) as {
      success?: boolean;
      data?: { version_id?: string };
    };
    return data.success ? data.data?.version_id : undefined;
  } catch {
    return undefined;
  }
}

router.post(
  "/qde/rollback/:targetVersionId",
  requirePlatformOrigin,
  adminAuth,
  requireResearchLabAuthorization("ROLLBACK", () => currentActiveVersionId()),
  async (req: Request, res: Response) => {
    await proxyPost(
      `/qde/rollback/${req.params["targetVersionId"]}`,
      req.body,
      res,
    );
  },
);

router.post(
  "/qde/rollback",
  requirePlatformOrigin,
  adminAuth,
  requireResearchLabAuthorization("ROLLBACK", () => currentActiveVersionId()),
  async (req: Request, res: Response) => {
    await proxyPost("/qde/rollback", req.body, res);
  },
);

router.get("/qde/rollback/history", async (_req: Request, res: Response) => {
  await proxyGet("/qde/rollback/history", res);
});

router.get("/qde/rollback/targets", async (_req: Request, res: Response) => {
  await proxyGet("/qde/rollback/targets", res);
});

// ── Archive / Retire ──────────────────────────────────────────────────────────
//
// Archiving/retiring a version is a terminal, Research-Lab-authorized
// lifecycle mutation (see docs/research_lab/permissions_matrix.md).

router.post(
  "/qde/versions/:versionId/archive",
  requirePlatformOrigin,
  adminAuth,
  requireResearchLabAuthorization("ARCHIVE", (req) => String(req.params["versionId"])),
  async (req: Request, res: Response) => {
    await proxyPost(`/qde/versions/${req.params["versionId"]}/archive`, req.body, res);
  },
);

router.post(
  "/qde/versions/:versionId/retire",
  requirePlatformOrigin,
  adminAuth,
  requireResearchLabAuthorization("RETIRE", (req) => String(req.params["versionId"])),
  async (req: Request, res: Response) => {
    await proxyPost(`/qde/versions/${req.params["versionId"]}/retire`, req.body, res);
  },
);

// ── A/B Testing ───────────────────────────────────────────────────────────────

router.post("/qde/ab/shadow", async (req: Request, res: Response) => {
  await proxyPost("/qde/ab/shadow", req.body, res);
});

router.post("/qde/ab/canary", async (req: Request, res: Response) => {
  await proxyPost("/qde/ab/canary", req.body, res);
});

router.post("/qde/ab/side-by-side", async (req: Request, res: Response) => {
  await proxyPost("/qde/ab/side-by-side", req.body, res);
});

router.post(
  "/qde/ab/:testId/results",
  async (req: Request, res: Response) => {
    await proxyPost(`/qde/ab/${req.params["testId"]}/results`, req.body, res);
  },
);

router.post(
  "/qde/ab/:testId/conclude",
  async (req: Request, res: Response) => {
    await proxyPost(`/qde/ab/${req.params["testId"]}/conclude`, req.body, res);
  },
);

router.get(
  "/qde/ab/:testId/compare",
  async (req: Request, res: Response) => {
    await proxyGet(`/qde/ab/${req.params["testId"]}/compare`, res);
  },
);

router.get("/qde/ab", async (req: Request, res: Response) => {
  const params = new URLSearchParams();
  if (req.query["status"]) params.set("status", String(req.query["status"]));
  if (req.query["test_type"])
    params.set("test_type", String(req.query["test_type"]));
  const qs = params.toString() ? `?${params.toString()}` : "";
  await proxyGet(`/qde/ab${qs}`, res);
});

// ── Position Management (Phase S3) ──────────────────────────────────────────

router.post(
  "/qde/position-management/review",
  async (req: Request, res: Response) => {
    await proxyPost("/qde/position-management/review", req.body, res);
  },
);

router.get(
  "/qde/position-management/interval",
  async (_req: Request, res: Response) => {
    await proxyGet("/qde/position-management/interval", res);
  },
);

// ── Market Observer (Phase S7) ──────────────────────────────────────────────
// Always-running per-symbol monitoring. Internal-service routes, same as
// position management above — not lifecycle-authority-gated.

router.post(
  "/qde/observer/:instrument/observe",
  async (req: Request, res: Response) => {
    await proxyPost(
      `/qde/observer/${req.params["instrument"]}/observe`,
      req.body,
      res,
    );
  },
);

router.get(
  "/qde/observer/:instrument/state",
  async (req: Request, res: Response) => {
    await proxyGet(`/qde/observer/${req.params["instrument"]}/state`, res);
  },
);

router.get("/qde/observer/states", async (req: Request, res: Response) => {
  const state = req.query["state"] ? `?state=${req.query["state"]}` : "";
  await proxyGet(`/qde/observer/states${state}`, res);
});

router.get(
  "/qde/observer/:instrument/history",
  async (req: Request, res: Response) => {
    const limit = req.query["limit"] ? `?limit=${req.query["limit"]}` : "";
    await proxyGet(
      `/qde/observer/${req.params["instrument"]}/history${limit}`,
      res,
    );
  },
);

router.get("/qde/observer/summary", async (_req: Request, res: Response) => {
  await proxyGet("/qde/observer/summary", res);
});

router.get(
  "/qde/observer/:instrument/telemetry",
  async (req: Request, res: Response) => {
    await proxyGet(`/qde/observer/${req.params["instrument"]}/telemetry`, res);
  },
);

router.get("/qde/observer/telemetry", async (_req: Request, res: Response) => {
  await proxyGet("/qde/observer/telemetry", res);
});

router.get(
  "/qde/observer/dashboard/state-timeline",
  async (req: Request, res: Response) => {
    const params = new URLSearchParams();
    if (req.query["granularity"])
      params.set("granularity", String(req.query["granularity"]));
    if (req.query["limit"]) params.set("limit", String(req.query["limit"]));
    const qs = params.toString() ? `?${params.toString()}` : "";
    await proxyGet(`/qde/observer/dashboard/state-timeline${qs}`, res);
  },
);

router.get(
  "/qde/observer/dashboard/dwell-histogram",
  async (_req: Request, res: Response) => {
    await proxyGet("/qde/observer/dashboard/dwell-histogram", res);
  },
);

router.get(
  "/qde/observer/dashboard/transition-funnel",
  async (req: Request, res: Response) => {
    const limit = req.query["limit"] ? `?limit=${req.query["limit"]}` : "";
    await proxyGet(`/qde/observer/dashboard/transition-funnel${limit}`, res);
  },
);

router.get(
  "/qde/observer/dashboard/guard-trips",
  async (req: Request, res: Response) => {
    const params = new URLSearchParams();
    if (req.query["granularity"])
      params.set("granularity", String(req.query["granularity"]));
    if (req.query["limit"]) params.set("limit", String(req.query["limit"]));
    const qs = params.toString() ? `?${params.toString()}` : "";
    await proxyGet(`/qde/observer/dashboard/guard-trips${qs}`, res);
  },
);

export default router;
