import { Router, type IRouter, type Request, type Response } from "express";

/**
 * Capability Endpoints.
 *
 * A single, unauthenticated-but-read-only surface the Signalium Platform
 * polls to learn what Trainer currently supports and is doing, without
 * needing admin credentials for read-only discovery: supported symbols,
 * active model version, active datasets, current market providers, health
 * status, training status, shadow deployment status, experiment status,
 * and RunPod status.
 *
 * Each capability aggregates a best-effort call to the relevant upstream
 * service (QDE server on :8001, Research Lab server on :8002) and degrades
 * to a clearly-labeled "unavailable" entry rather than failing the whole
 * response if one upstream is down — the Platform should be able to render
 * a partial capability picture.
 */

const router: IRouter = Router();

const QDE_BASE = process.env["QDE_API_URL"] || "http://localhost:8001";
const RL_BASE = process.env["RESEARCH_LAB_API_URL"] || "http://localhost:8002";

async function safeFetchJson(url: string): Promise<{ ok: boolean; data: unknown }> {
  try {
    const upstream = await fetch(url);
    const data = await upstream.json();
    return { ok: upstream.ok, data };
  } catch (err) {
    return { ok: false, data: { error: String(err) } };
  }
}

router.get("/capabilities/symbols", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${QDE_BASE}/qde/observer/states`);
  res.json({ capability: "supported_symbols", available: result.ok, data: result.data });
});

router.get("/capabilities/model-version", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${QDE_BASE}/qde/active`);
  res.json({ capability: "active_model_version", available: result.ok, data: result.data });
});

router.get("/capabilities/datasets", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${RL_BASE}/research/datasets`);
  res.json({ capability: "active_datasets", available: result.ok, data: result.data });
});

router.get("/capabilities/market-providers", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${RL_BASE}/market-data/providers/status`);
  res.json({ capability: "market_providers", available: result.ok, data: result.data });
});

router.get("/capabilities/health", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${RL_BASE}/health/composite/latest`);
  res.json({ capability: "health_status", available: result.ok, data: result.data });
});

router.get("/capabilities/training", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${RL_BASE}/research/experiments/summary`);
  res.json({ capability: "training_status", available: result.ok, data: result.data });
});

router.get("/capabilities/shadow", async (_req: Request, res: Response) => {
  const p = new URLSearchParams({ status: "RUNNING" });
  const result = await safeFetchJson(`${RL_BASE}/research/shadow-tests?${p.toString()}`);
  res.json({ capability: "shadow_deployment_status", available: result.ok, data: result.data });
});

router.get("/capabilities/experiments", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${RL_BASE}/research/experiments/summary`);
  res.json({ capability: "experiment_status", available: result.ok, data: result.data });
});

router.get("/capabilities/runpod", async (_req: Request, res: Response) => {
  const result = await safeFetchJson(`${RL_BASE}/infra/runpod/status`);
  res.json({ capability: "runpod_status", available: result.ok, data: result.data });
});

// ── Consolidated view ─────────────────────────────────────────────────────────
//
// Single round-trip for a Platform integration that wants everything at
// once rather than nine separate polls.

router.get("/capabilities", async (_req: Request, res: Response) => {
  const [symbols, modelVersion, datasets, providers, health, training, shadow, experiments, runpod] =
    await Promise.all([
      safeFetchJson(`${QDE_BASE}/qde/observer/states`),
      safeFetchJson(`${QDE_BASE}/qde/active`),
      safeFetchJson(`${RL_BASE}/research/datasets`),
      safeFetchJson(`${RL_BASE}/market-data/providers/status`),
      safeFetchJson(`${RL_BASE}/health/composite/latest`),
      safeFetchJson(`${RL_BASE}/research/experiments/summary`),
      safeFetchJson(`${RL_BASE}/research/shadow-tests?status=RUNNING`),
      safeFetchJson(`${RL_BASE}/research/experiments/summary`),
      safeFetchJson(`${RL_BASE}/infra/runpod/status`),
    ]);

  res.json({
    supported_symbols: { available: symbols.ok, data: symbols.data },
    active_model_version: { available: modelVersion.ok, data: modelVersion.data },
    active_datasets: { available: datasets.ok, data: datasets.data },
    market_providers: { available: providers.ok, data: providers.data },
    health_status: { available: health.ok, data: health.data },
    training_status: { available: training.ok, data: training.data },
    shadow_deployment_status: { available: shadow.ok, data: shadow.data },
    experiment_status: { available: experiments.ok, data: experiments.data },
    runpod_status: { available: runpod.ok, data: runpod.data },
  });
});

export default router;
