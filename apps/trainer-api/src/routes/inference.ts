/**
 * Signalium Platform inference endpoints.
 *
 * All four routes are JWT-protected via platformAuth (applied in platform.ts)
 * and mounted at root level (no /api prefix).
 *
 * POST /                  — telemetry snapshot → { action, reasoning }
 * POST /qde/inference     — market direction/signal
 * POST /qde/review-position — single open position assessment
 * POST /qde/portfolio-review — batch portfolio review
 *
 * Error shape on all failures: { "error": "<description>" } with no
 * "success" wrapper, so the platform circuit breaker reads them correctly.
 */

import { Router, type IRouter, type Request, type Response } from "express";
import { ZodError } from "zod";
import { logger } from "../lib/logger.js";
import { qdePost, QdeClientError } from "../lib/qde_client.js";
import {
  TelemetryPayload,
  InferenceRequest,
  PositionReviewRequest,
  PortfolioReviewRequest,
  normalizeAction,
} from "../lib/inference_schemas.js";

const router: IRouter = Router();

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

function sendError(res: Response, status: number, message: string): void {
  res.status(status).json({ error: message });
}

function handleRouteError(res: Response, err: unknown): void {
  if (err instanceof ZodError) {
    sendError(res, 400, err.errors.map((e) => `${e.path.join(".")}: ${e.message}`).join("; "));
    return;
  }
  if (err instanceof QdeClientError) {
    const status = err.status >= 500 ? 502 : err.status;
    sendError(res, status, err.message);
    return;
  }
  logger.error({ err }, "inference: unexpected error");
  sendError(res, 500, "Internal server error");
}

// ---------------------------------------------------------------------------
// POST / — legacy QDE decision
//
// The platform posts this on every EA tick:
//   { symbol, balance, equity, margin, freeMargin, openPositions,
//     floatingPnl, entryPrice, currentPrice, spread, drawdown, tradeAgeSeconds }
// Expected response: { "action": "HOLD|ADD|REDUCE|TRAIL|BREAKEVEN|LOCK_PROFIT|CLOSE",
//                      "reasoning": "<string>" }
// ---------------------------------------------------------------------------

router.post("/", async (req: Request, res: Response): Promise<void> => {
  try {
    const body = TelemetryPayload.parse(req.body ?? {});
    const data = (await qdePost("/qde/platform/decide", body)) as {
      action?: string;
      reasoning?: string;
    };
    const action = normalizeAction(data.action ?? "HOLD");
    const reasoning = data.reasoning ?? "No reasoning provided.";
    res.status(200).json({ action, reasoning });
  } catch (err) {
    handleRouteError(res, err);
  }
});

// ---------------------------------------------------------------------------
// POST /qde/inference — market direction/signal
// ---------------------------------------------------------------------------

router.post("/qde/inference", async (req: Request, res: Response): Promise<void> => {
  try {
    const body = InferenceRequest.parse(req.body ?? {});
    const data = await qdePost("/qde/platform/inference", body);
    res.status(200).json(data);
  } catch (err) {
    handleRouteError(res, err);
  }
});

// ---------------------------------------------------------------------------
// POST /qde/review-position — single open position assessment
//
// Proxies to the existing internal /qde/position-management/review endpoint.
// The platform may send the structured PositionTelemetryRequest shape or a
// flat TelemetryPayload; we accept both.
// ---------------------------------------------------------------------------

router.post("/qde/review-position", async (req: Request, res: Response): Promise<void> => {
  try {
    const body = req.body ?? {};

    // If the body looks like a flat telemetry payload (has "symbol" instead of
    // "instrument"), route to the platform/decide handler which handles the
    // field mapping internally.
    if (typeof body["symbol"] === "string" && typeof body["instrument"] !== "string") {
      const telemetry = TelemetryPayload.parse(body);
      const data = (await qdePost("/qde/platform/decide", telemetry)) as {
        action?: string;
        reasoning?: string;
      };
      const action = normalizeAction(data.action ?? "HOLD");
      const reasoning = data.reasoning ?? "No reasoning provided.";
      res.status(200).json({ action, reasoning });
      return;
    }

    // Structured PositionTelemetryRequest — pass through to position manager.
    const structured = PositionReviewRequest.parse(body);
    const data = (await qdePost("/qde/position-management/review", structured)) as {
      success?: boolean;
      decision?: { action?: string; [key: string]: unknown };
      [key: string]: unknown;
    };

    // Normalize action in the decision object
    if (data.decision?.action) {
      data.decision.action = normalizeAction(data.decision.action as string);
    }
    res.status(200).json(data);
  } catch (err) {
    handleRouteError(res, err);
  }
});

// ---------------------------------------------------------------------------
// POST /qde/portfolio-review — batch portfolio review
// ---------------------------------------------------------------------------

router.post("/qde/portfolio-review", async (req: Request, res: Response): Promise<void> => {
  try {
    const body = PortfolioReviewRequest.parse(req.body ?? {});
    const data = (await qdePost("/qde/platform/portfolio-review", body)) as {
      decisions?: Array<{ action?: string; [key: string]: unknown }>;
      [key: string]: unknown;
    };

    // Normalize action on every decision
    if (Array.isArray(data.decisions)) {
      for (const d of data.decisions) {
        if (d.action) d.action = normalizeAction(d.action as string);
      }
    }
    res.status(200).json(data);
  } catch (err) {
    handleRouteError(res, err);
  }
});

export default router;
