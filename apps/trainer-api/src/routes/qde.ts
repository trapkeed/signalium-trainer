import { Router, type IRouter, type Request, type Response } from "express";

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

router.post(
  "/qde/rollback/:targetVersionId",
  async (req: Request, res: Response) => {
    await proxyPost(
      `/qde/rollback/${req.params["targetVersionId"]}`,
      req.body,
      res,
    );
  },
);

router.post("/qde/rollback", async (req: Request, res: Response) => {
  await proxyPost("/qde/rollback", req.body, res);
});

router.get("/qde/rollback/history", async (_req: Request, res: Response) => {
  await proxyGet("/qde/rollback/history", res);
});

router.get("/qde/rollback/targets", async (_req: Request, res: Response) => {
  await proxyGet("/qde/rollback/targets", res);
});

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

export default router;
