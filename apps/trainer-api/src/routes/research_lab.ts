import { Router, type IRouter, type Request, type Response } from "express";
import { adminAuth } from "../middlewares/admin_auth";

/**
 * Research Lab routes — administrator access only.
 *
 * All routes under /api/admin/research/* require a valid admin token.
 * These proxy to the upstream Research Lab Python server (default: :8002).
 *
 * No retail user may access these endpoints.
 */

const router: IRouter = Router();

const RL_BASE = process.env["RESEARCH_LAB_API_URL"] || "http://localhost:8002";

// ── Apply admin auth to ALL research routes ───────────────────────────────────

router.use("/admin/research", adminAuth);

// ── Proxy helpers ─────────────────────────────────────────────────────────────

function forwardToken(req: Request): Record<string, string> {
  const authHeader = req.headers["authorization"] as string | undefined;
  const xAdmin = req.headers["x-admin-token"] as string | undefined;
  if (authHeader) return { Authorization: authHeader, "Content-Type": "application/json" };
  if (xAdmin) return { Authorization: `Bearer ${xAdmin}`, "Content-Type": "application/json" };
  return { "Content-Type": "application/json" };
}

async function proxyGet(
  path: string,
  req: Request,
  res: Response,
): Promise<void> {
  try {
    const upstream = await fetch(`${RL_BASE}${path}`, {
      headers: forwardToken(req),
    });
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ success: false, error: String(err) });
  }
}

async function proxyPost(
  path: string,
  req: Request,
  res: Response,
): Promise<void> {
  try {
    const upstream = await fetch(`${RL_BASE}${path}`, {
      method: "POST",
      headers: forwardToken(req),
      body: JSON.stringify(req.body),
    });
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ success: false, error: String(err) });
  }
}

// ── Health & Summary ──────────────────────────────────────────────────────────

router.get("/admin/research/healthz", async (req, res) => {
  await proxyGet("/research/healthz", req, res);
});

router.get("/admin/research/summary", async (req, res) => {
  await proxyGet("/research/summary", req, res);
});

// ── Datasets ──────────────────────────────────────────────────────────────────

router.post("/admin/research/datasets", async (req, res) => {
  await proxyPost("/research/datasets", req, res);
});

router.get("/admin/research/datasets", async (req, res) => {
  await proxyGet("/research/datasets", req, res);
});

router.get("/admin/research/datasets/:datasetId", async (req, res) => {
  await proxyGet(`/research/datasets/${req.params["datasetId"]}`, req, res);
});

// ── Feature Discovery ─────────────────────────────────────────────────────────

router.post("/admin/research/features", async (req, res) => {
  await proxyPost("/research/features", req, res);
});

router.post("/admin/research/features/:featureId/accept", async (req, res) => {
  await proxyPost(`/research/features/${req.params["featureId"]}/accept`, req, res);
});

router.post("/admin/research/features/:featureId/reject", async (req, res) => {
  await proxyPost(`/research/features/${req.params["featureId"]}/reject`, req, res);
});

router.get("/admin/research/features", async (req, res) => {
  const qs = req.query["status"] ? `?status=${req.query["status"]}` : "";
  await proxyGet(`/research/features${qs}`, req, res);
});

// ── Experiments ───────────────────────────────────────────────────────────────

router.post("/admin/research/experiments", async (req, res) => {
  await proxyPost("/research/experiments", req, res);
});

router.get("/admin/research/experiments/summary", async (req, res) => {
  await proxyGet("/research/experiments/summary", req, res);
});

router.get("/admin/research/experiments", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  if (req.query["qde_version"]) p.set("qde_version", String(req.query["qde_version"]));
  if (req.query["status"]) p.set("status", String(req.query["status"]));
  if (req.query["researcher"]) p.set("researcher", String(req.query["researcher"]));
  await proxyGet(`/research/experiments${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.post("/admin/research/experiments/:experimentId/start", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/start`, req, res);
});

router.post("/admin/research/experiments/:experimentId/complete", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/complete`, req, res);
});

router.post("/admin/research/experiments/:experimentId/fail", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/fail`, req, res);
});

router.get("/admin/research/experiments/:experimentId", async (req, res) => {
  await proxyGet(`/research/experiments/${req.params["experimentId"]}`, req, res);
});

// ── Benchmarking ──────────────────────────────────────────────────────────────

router.post("/admin/research/benchmarks", async (req, res) => {
  await proxyPost("/research/benchmarks", req, res);
});

router.get("/admin/research/benchmarks", async (req, res) => {
  await proxyGet("/research/benchmarks", req, res);
});

router.get("/admin/research/benchmarks/:benchmarkId", async (req, res) => {
  await proxyGet(`/research/benchmarks/${req.params["benchmarkId"]}`, req, res);
});

// ── Backtesting ───────────────────────────────────────────────────────────────

router.post("/admin/research/backtests", async (req, res) => {
  await proxyPost("/research/backtests", req, res);
});

router.get("/admin/research/backtests", async (req: Request, res: Response) => {
  const qs = req.query["experiment_id"] ? `?experiment_id=${req.query["experiment_id"]}` : "";
  await proxyGet(`/research/backtests${qs}`, req, res);
});

// ── Parameter Search ──────────────────────────────────────────────────────────

router.post("/admin/research/searches", async (req, res) => {
  await proxyPost("/research/searches", req, res);
});

router.post("/admin/research/searches/:searchId/trials", async (req, res) => {
  await proxyPost(`/research/searches/${req.params["searchId"]}/trials`, req, res);
});

router.post("/admin/research/searches/:searchId/complete", async (req, res) => {
  await proxyPost(`/research/searches/${req.params["searchId"]}/complete`, req, res);
});

router.get("/admin/research/searches/:searchId", async (req, res) => {
  await proxyGet(`/research/searches/${req.params["searchId"]}`, req, res);
});

router.get("/admin/research/searches", async (req: Request, res: Response) => {
  const qs = req.query["qde_version"] ? `?qde_version=${req.query["qde_version"]}` : "";
  await proxyGet(`/research/searches${qs}`, req, res);
});

// ── Deployment Gates & Validation ─────────────────────────────────────────────

router.post("/admin/research/experiments/:experimentId/gates", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/gates`, req, res);
});

router.post("/admin/research/experiments/:experimentId/validate-retraining", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/validate-retraining`, req, res);
});

// ── Deployment Approval ───────────────────────────────────────────────────────

router.post("/admin/research/experiments/:experimentId/approve", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/approve`, req, res);
});

router.post("/admin/research/experiments/:experimentId/reject", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/reject`, req, res);
});

router.get("/admin/research/deployments", async (req, res) => {
  await proxyGet("/research/deployments", req, res);
});

// ── Failure Analysis ──────────────────────────────────────────────────────────

router.post("/admin/research/failures", async (req, res) => {
  await proxyPost("/research/failures", req, res);
});

router.post("/admin/research/failures/:failureId/investigate", async (req, res) => {
  await proxyPost(`/research/failures/${req.params["failureId"]}/investigate`, req, res);
});

router.post("/admin/research/failures/:failureId/resolve", async (req, res) => {
  await proxyPost(`/research/failures/${req.params["failureId"]}/resolve`, req, res);
});

router.get("/admin/research/failures/summary", async (req, res) => {
  await proxyGet("/research/failures/summary", req, res);
});

router.get("/admin/research/failures", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["category", "severity", "qde_version", "status"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/research/failures${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

// ── Market Regime Analysis ────────────────────────────────────────────────────

router.post("/admin/research/regime", async (req, res) => {
  await proxyPost("/research/regime", req, res);
});

router.get("/admin/research/regime", async (req: Request, res: Response) => {
  const qs = req.query["qde_version"] ? `?qde_version=${req.query["qde_version"]}` : "";
  await proxyGet(`/research/regime${qs}`, req, res);
});

router.get("/admin/research/regime/:reportId", async (req, res) => {
  await proxyGet(`/research/regime/${req.params["reportId"]}`, req, res);
});

// ── Model Comparison ──────────────────────────────────────────────────────────

router.post("/admin/research/comparisons", async (req, res) => {
  await proxyPost("/research/comparisons", req, res);
});

router.get("/admin/research/comparisons", async (req, res) => {
  await proxyGet("/research/comparisons", req, res);
});

router.get("/admin/research/comparisons/:comparisonId", async (req, res) => {
  await proxyGet(`/research/comparisons/${req.params["comparisonId"]}`, req, res);
});

export default router;
