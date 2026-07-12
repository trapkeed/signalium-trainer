import { Router, type IRouter, type Request, type Response } from "express";
import { adminAuth } from "../middlewares/admin_auth";
import { requirePlatformOrigin } from "../middlewares/platform_trust";

/**
 * Research Lab routes — administrator access only.
 *
 * ACCESS MODEL
 * ============
 * - All routes under /api/admin/research/* require a valid admin token.
 * - The adminAuth middleware enforces this on every request.
 * - These routes proxy to the Research Lab Python server (default: :8002).
 * - No retail user may access any endpoint in this router.
 * - No public documentation, no public dashboards, no frontend.
 *
 * RESPONSIBILITIES (8 confirmed)
 * ================================
 *  1. Experiment tracking
 *  2. Backtesting
 *  3. Validation (deployment gates)
 *  4. Dataset generation
 *  5. Failure analysis
 *  6. Version comparisons
 *  7. Promotion approval
 *  8. Rollback approval
 *
 * The Signalium Admin Portal is the sole authorised consumer of these routes.
 */

const router: IRouter = Router();

const RL_BASE = process.env["RESEARCH_LAB_API_URL"] || "http://localhost:8002";

// ── Apply admin auth to ALL research routes ───────────────────────────────────

router.use("/admin/research", requirePlatformOrigin, adminAuth);

// ── Proxy helpers ─────────────────────────────────────────────────────────────

/**
 * Build auth headers for upstream calls to the Python Research Lab service.
 *
 * The incoming request carries a platform JWT (validated by adminAuth), but
 * the Python RL service expects the internal RESEARCH_LAB_ADMIN_TOKEN — the
 * platform JWT must never be forwarded. Use the static internal token instead.
 */
function forwardToken(_req: Request): Record<string, string> {
  const token = process.env["RESEARCH_LAB_ADMIN_TOKEN"];
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

async function proxyGet(path: string, req: Request, res: Response): Promise<void> {
  try {
    const upstream = await fetch(`${RL_BASE}${path}`, { headers: forwardToken(req) });
    const data = await upstream.json();
    res.status(upstream.status).json(data);
  } catch (err) {
    res.status(502).json({ success: false, error: String(err) });
  }
}

async function proxyPost(path: string, req: Request, res: Response): Promise<void> {
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

async function proxyDelete(path: string, req: Request, res: Response): Promise<void> {
  try {
    const upstream = await fetch(`${RL_BASE}${path}`, {
      method: "DELETE",
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

// ── 1. Datasets ───────────────────────────────────────────────────────────────

router.post("/admin/research/datasets", async (req, res) => {
  await proxyPost("/research/datasets", req, res);
});

router.get("/admin/research/datasets", async (req, res) => {
  await proxyGet("/research/datasets", req, res);
});

router.get("/admin/research/datasets/:datasetId", async (req, res) => {
  await proxyGet(`/research/datasets/${req.params["datasetId"]}`, req, res);
});

// ── 4. Dataset Generation Jobs ────────────────────────────────────────────────
//
// A generation job tracks the full pipeline for building a dataset:
//   PENDING → GENERATING → VALIDATING → READY
//                                     ↘ FAILED
//
// POST   /datasets/:id/generate            — create a generation job
// GET    /datasets/:id/generation-jobs     — list jobs for a dataset
// GET    /generation-jobs                  — list all jobs (filterable by status)
// GET    /generation-jobs/:jobId           — get a specific job
// POST   /generation-jobs/:jobId/start     — PENDING → GENERATING
// POST   /generation-jobs/:jobId/validating — GENERATING → VALIDATING
// POST   /generation-jobs/:jobId/complete  — VALIDATING → READY
// POST   /generation-jobs/:jobId/fail      — * → FAILED

router.post("/admin/research/datasets/:datasetId/generate", async (req, res) => {
  await proxyPost(`/research/datasets/${req.params["datasetId"]}/generate`, req, res);
});

router.get("/admin/research/datasets/:datasetId/generation-jobs", async (req, res) => {
  await proxyGet(`/research/datasets/${req.params["datasetId"]}/generation-jobs`, req, res);
});

router.get("/admin/research/generation-jobs", async (req: Request, res: Response) => {
  const qs = req.query["status"] ? `?status=${req.query["status"]}` : "";
  await proxyGet(`/research/generation-jobs${qs}`, req, res);
});

router.get("/admin/research/generation-jobs/:jobId", async (req, res) => {
  await proxyGet(`/research/generation-jobs/${req.params["jobId"]}`, req, res);
});

router.post("/admin/research/generation-jobs/:jobId/start", async (req, res) => {
  await proxyPost(`/research/generation-jobs/${req.params["jobId"]}/start`, req, res);
});

router.post("/admin/research/generation-jobs/:jobId/validating", async (req, res) => {
  await proxyPost(`/research/generation-jobs/${req.params["jobId"]}/validating`, req, res);
});

router.post("/admin/research/generation-jobs/:jobId/complete", async (req, res) => {
  await proxyPost(`/research/generation-jobs/${req.params["jobId"]}/complete`, req, res);
});

router.post("/admin/research/generation-jobs/:jobId/fail", async (req, res) => {
  await proxyPost(`/research/generation-jobs/${req.params["jobId"]}/fail`, req, res);
});

// ── 2. Feature Discovery ──────────────────────────────────────────────────────

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

// ── 3. Experiments / Training Runs ────────────────────────────────────────────

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

// ── Backtesting ───────────────────────────────────────────────────────────────

router.post("/admin/research/benchmarks", async (req, res) => {
  await proxyPost("/research/benchmarks", req, res);
});

router.get("/admin/research/benchmarks", async (req, res) => {
  await proxyGet("/research/benchmarks", req, res);
});

router.get("/admin/research/benchmarks/:benchmarkId", async (req, res) => {
  await proxyGet(`/research/benchmarks/${req.params["benchmarkId"]}`, req, res);
});

router.post("/admin/research/backtests", async (req, res) => {
  await proxyPost("/research/backtests", req, res);
});

router.get("/admin/research/backtests", async (req: Request, res: Response) => {
  const qs = req.query["experiment_id"] ? `?experiment_id=${req.query["experiment_id"]}` : "";
  await proxyGet(`/research/backtests${qs}`, req, res);
});

// ── Shadow Testing ────────────────────────────────────────────────────────────
//
// Shadow tests run a challenger QDE version alongside the champion.
// The challenger's decisions are NEVER executed — only observed.
// Admins use these results to decide whether to promote the challenger.

router.post("/admin/research/shadow-tests", async (req, res) => {
  await proxyPost("/research/shadow-tests", req, res);
});

router.get("/admin/research/shadow-tests", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  if (req.query["challenger_version"])
    p.set("challenger_version", String(req.query["challenger_version"]));
  if (req.query["status"]) p.set("status", String(req.query["status"]));
  await proxyGet(`/research/shadow-tests${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/shadow-tests/:testId", async (req, res) => {
  await proxyGet(`/research/shadow-tests/${req.params["testId"]}`, req, res);
});

router.post("/admin/research/shadow-tests/:testId/decisions", async (req, res) => {
  await proxyPost(`/research/shadow-tests/${req.params["testId"]}/decisions`, req, res);
});

router.post("/admin/research/shadow-tests/:testId/conclude", async (req, res) => {
  await proxyPost(`/research/shadow-tests/${req.params["testId"]}/conclude`, req, res);
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

// ── Validation (Deployment Gates) ─────────────────────────────────────────────

router.post("/admin/research/experiments/:experimentId/gates", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/gates`, req, res);
});

router.post("/admin/research/experiments/:experimentId/validate-retraining", async (req, res) => {
  await proxyPost(
    `/research/experiments/${req.params["experimentId"]}/validate-retraining`,
    req,
    res,
  );
});

// ── Promotion Approval ────────────────────────────────────────────────────────

router.post("/admin/research/experiments/:experimentId/approve", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/approve`, req, res);
});

router.post("/admin/research/experiments/:experimentId/reject", async (req, res) => {
  await proxyPost(`/research/experiments/${req.params["experimentId"]}/reject`, req, res);
});

router.get("/admin/research/deployments", async (req, res) => {
  await proxyGet("/research/deployments", req, res);
});

// ── Rollback Approval ─────────────────────────────────────────────────────────
//
// Rollback approval is a two-administrator workflow:
//  1. Admin A creates a rollback request (current → target version, with reason).
//  2. Admin B approves or rejects it.
//  3. Admin A may withdraw their own pending request at any time.
//
// POST   /rollbacks                        — create a rollback request
// GET    /rollbacks                        — list all requests (filterable by status)
// GET    /rollbacks/:id                    — get a specific request
// POST   /rollbacks/:id/approve            — approve (must be a different admin)
// POST   /rollbacks/:id/reject             — reject
// POST   /rollbacks/:id/withdraw           — requester withdraws their own request

router.post("/admin/research/rollbacks", async (req, res) => {
  await proxyPost("/research/rollbacks", req, res);
});

router.get("/admin/research/rollbacks", async (req: Request, res: Response) => {
  const qs = req.query["status"] ? `?status=${req.query["status"]}` : "";
  await proxyGet(`/research/rollbacks${qs}`, req, res);
});

router.get("/admin/research/rollbacks/:requestId", async (req, res) => {
  await proxyGet(`/research/rollbacks/${req.params["requestId"]}`, req, res);
});

router.post("/admin/research/rollbacks/:requestId/approve", async (req, res) => {
  await proxyPost(`/research/rollbacks/${req.params["requestId"]}/approve`, req, res);
});

router.post("/admin/research/rollbacks/:requestId/reject", async (req, res) => {
  await proxyPost(`/research/rollbacks/${req.params["requestId"]}/reject`, req, res);
});

router.post("/admin/research/rollbacks/:requestId/withdraw", async (req, res) => {
  await proxyPost(`/research/rollbacks/${req.params["requestId"]}/withdraw`, req, res);
});

// ── Lifecycle Authorizations ──────────────────────────────────────────────────
//
// Research Lab is the SOLE authority permitted to approve PROMOTE, ROLLBACK,
// ARCHIVE, and RETIRE actions against QDE versions. PROMOTE and ROLLBACK
// authorizations are issued automatically as a side effect of the
// promotion-approval and rollback-approval endpoints above. ARCHIVE and
// RETIRE have no separate gated workflow, so they are authorized directly
// through /admin/research/lifecycle/authorize.
//
// The QDE gateway (see routes/qde.ts) requires a matching APPROVED,
// unconsumed authorization before it will execute the corresponding
// registry mutation, and consumes it atomically as part of doing so.

router.post("/admin/research/lifecycle/authorize", async (req, res) => {
  await proxyPost("/research/lifecycle/authorize", req, res);
});

router.get("/admin/research/lifecycle/authorizations", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["version_id", "action", "status"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(
    `/research/lifecycle/authorizations${p.toString() ? "?" + p.toString() : ""}`,
    req,
    res,
  );
});

router.get("/admin/research/lifecycle/authorizations/:authorizationId", async (req, res) => {
  await proxyGet(
    `/research/lifecycle/authorizations/${req.params["authorizationId"]}`,
    req,
    res,
  );
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

// ── Version Comparisons ───────────────────────────────────────────────────────

router.post("/admin/research/comparisons", async (req, res) => {
  await proxyPost("/research/comparisons", req, res);
});

router.get("/admin/research/comparisons", async (req, res) => {
  await proxyGet("/research/comparisons", req, res);
});

router.get("/admin/research/comparisons/:comparisonId", async (req, res) => {
  await proxyGet(`/research/comparisons/${req.params["comparisonId"]}`, req, res);
});

// ── Health Monitoring (five drift dimensions) ─────────────────────────────────

router.post("/admin/research/health/check", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  if (req.query["model_version"]) p.set("model_version", String(req.query["model_version"]));
  if (req.query["since"]) p.set("since", String(req.query["since"]));
  await proxyPost(`/health/check${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/health/status", async (req: Request, res: Response) => {
  const qs = req.query["model_version"] ? `?model_version=${req.query["model_version"]}` : "";
  await proxyGet(`/health/status${qs}`, req, res);
});

router.get("/admin/research/health/alerts", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["model_version", "severity", "dimension", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/health/alerts${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

// ── Approval Gates (generic, shared by all gate types) ────────────────────────

router.post("/admin/research/approvals", async (req, res) => {
  await proxyPost("/approvals", req, res);
});

router.get("/admin/research/approvals", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["gate_type", "resource_id", "status", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/approvals${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/approvals/:requestId", async (req, res) => {
  await proxyGet(`/approvals/${req.params["requestId"]}`, req, res);
});

router.post("/admin/research/approvals/:requestId/approve", async (req, res) => {
  await proxyPost(`/approvals/${req.params["requestId"]}/approve`, req, res);
});

router.post("/admin/research/approvals/:requestId/reject", async (req, res) => {
  await proxyPost(`/approvals/${req.params["requestId"]}/reject`, req, res);
});

// ── Deployment / Promotion Lifecycle ───────────────────────────────────────────

router.get("/admin/research/deployment/current", async (req, res) => {
  await proxyGet("/deployment/current", req, res);
});

router.get("/admin/research/deployment/history", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["model_version", "event_type", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/deployment/history${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.post("/admin/research/deployment/promote", async (req, res) => {
  await proxyPost("/deployment/promote", req, res);
});

router.post("/admin/research/deployment/activate", async (req, res) => {
  await proxyPost("/deployment/activate", req, res);
});

router.post("/admin/research/deployment/rollback", async (req, res) => {
  await proxyPost("/deployment/rollback", req, res);
});

router.post("/admin/research/deployment/rollback/activate", async (req, res) => {
  await proxyPost("/deployment/rollback/activate", req, res);
});

// ── Artifact Registry / Training Run Lineage ──────────────────────────────────

router.post("/admin/research/artifacts", async (req, res) => {
  await proxyPost("/artifacts/register", req, res);
});

router.get("/admin/research/artifacts", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of [
    "artifact_type",
    "model_version",
    "training_run_id",
    "experiment_id",
    "limit",
  ] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/artifacts${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/artifacts/types", async (req, res) => {
  await proxyGet("/artifacts/types", req, res);
});

router.post("/admin/research/artifacts/runs", async (req, res) => {
  await proxyPost("/artifacts/runs", req, res);
});

router.get("/admin/research/artifacts/runs", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["experiment_id", "pipeline_job_id", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/artifacts/runs${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/artifacts/runs/:runId", async (req, res) => {
  await proxyGet(`/artifacts/runs/${req.params["runId"]}`, req, res);
});

router.get("/admin/research/artifacts/:artifactId", async (req, res) => {
  await proxyGet(`/artifacts/${req.params["artifactId"]}`, req, res);
});

// ── Champion vs Challenger (Shadow Testing) ────────────────────────────────────

router.get("/admin/research/shadow/status", async (req: Request, res: Response) => {
  const qs = req.query["test_id"] ? `?test_id=${req.query["test_id"]}` : "";
  await proxyGet(`/shadow/status${qs}`, req, res);
});

router.get("/admin/research/shadow/comparison", async (req: Request, res: Response) => {
  const qs = req.query["test_id"] ? `?test_id=${req.query["test_id"]}` : "";
  await proxyGet(`/shadow/comparison${qs}`, req, res);
});

router.post("/admin/research/shadow/evaluate", async (req: Request, res: Response) => {
  const qs = req.query["test_id"] ? `?test_id=${req.query["test_id"]}` : "";
  await proxyPost(`/shadow/evaluate${qs}`, req, res);
});

// ── Retraining Recommendations ────────────────────────────────────────────────

router.post("/admin/research/evaluate", async (req, res) => {
  await proxyPost("/research/evaluate", req, res);
});

router.get("/admin/research/recommendations", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["status", "model_version", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/research/recommendations${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/recommendations/:recommendationId", async (req, res) => {
  await proxyGet(`/research/recommendations/${req.params["recommendationId"]}`, req, res);
});

router.post("/admin/research/recommendations/:recommendationId/approve", async (req, res) => {
  await proxyPost(`/research/recommendations/${req.params["recommendationId"]}/approve`, req, res);
});

router.post("/admin/research/recommendations/:recommendationId/dismiss", async (req, res) => {
  await proxyPost(`/research/recommendations/${req.params["recommendationId"]}/dismiss`, req, res);
});

// ── Admin Dashboard Aggregations ──────────────────────────────────────────────

router.get("/admin/research/symbol-performance", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["model_version", "since", "until", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/research/symbol-performance${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/calibration", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["model_version", "since", "until", "bucket_width", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(`/research/calibration${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.get("/admin/research/reasoning-distribution", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  for (const key of ["model_version", "symbol", "since", "until", "bucket_seconds", "limit"] as const) {
    if (req.query[key]) p.set(key, String(req.query[key]));
  }
  await proxyGet(
    `/research/reasoning-distribution${p.toString() ? "?" + p.toString() : ""}`,
    req,
    res,
  );
});

// ── S33 — Symbol Normalization Admin API ─────────────────────────────────────
//
// Broker-aware market data routing: manage broker symbol aliases and
// enable/disable canonical instruments, plus expose normalization
// diagnostics for the Signalium Admin Portal.

router.post("/admin/research/symbols/aliases", async (req: Request, res: Response) => {
  await proxyPost("/admin/symbols/aliases", req, res);
});

router.delete("/admin/research/symbols/aliases", async (req: Request, res: Response) => {
  await proxyDelete("/admin/symbols/aliases", req, res);
});

router.get("/admin/research/symbols/aliases", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  if (req.query["canonical"]) p.set("canonical", String(req.query["canonical"]));
  await proxyGet(`/admin/symbols/aliases${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

router.post("/admin/research/symbols/disable", async (req: Request, res: Response) => {
  await proxyPost("/admin/symbols/disable", req, res);
});

router.post("/admin/research/symbols/enable", async (req: Request, res: Response) => {
  await proxyPost("/admin/symbols/enable", req, res);
});

router.get("/admin/research/symbols", async (req: Request, res: Response) => {
  await proxyGet("/admin/symbols", req, res);
});

router.get("/admin/research/symbols/diagnostics", async (req: Request, res: Response) => {
  const p = new URLSearchParams();
  if (req.query["limit"]) p.set("limit", String(req.query["limit"]));
  if (req.query["unresolved_only"]) p.set("unresolved_only", String(req.query["unresolved_only"]));
  await proxyGet(`/admin/symbols/diagnostics${p.toString() ? "?" + p.toString() : ""}`, req, res);
});

export default router;
