/**
 * Zod schemas for the Signalium Platform inference endpoints.
 *
 * These are the shapes the platform sends and expects back.
 * All four platform-facing POST endpoints use types derived from here.
 */

import { z } from "zod";

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

/** The seven action values the platform contract allows. */
export const PlatformAction = z.enum([
  "HOLD",
  "ADD",
  "REDUCE",
  "TRAIL",
  "BREAKEVEN",
  "LOCK_PROFIT",
  "CLOSE",
]);
export type PlatformActionType = z.infer<typeof PlatformAction>;

/** Map internal QDE action names to platform contract names. */
export function normalizeAction(action: string): string {
  if (action === "MOVE_TO_BREAKEVEN") return "BREAKEVEN";
  return action;
}

// ---------------------------------------------------------------------------
// POST / — legacy QDE decision (telemetry → action + reasoning)
// ---------------------------------------------------------------------------

export const TelemetryPayload = z.object({
  symbol: z.string().min(1),
  balance: z.number().default(0),
  equity: z.number().default(0),
  margin: z.number().default(0),
  freeMargin: z.number().default(0),
  openPositions: z.number().int().default(0),
  floatingPnl: z.number().default(0),
  entryPrice: z.number().default(0),
  currentPrice: z.number().default(0),
  spread: z.number().default(0),
  drawdown: z.number().default(0),
  tradeAgeSeconds: z.number().default(0),
});
export type TelemetryPayloadType = z.infer<typeof TelemetryPayload>;

export const DecisionResponse = z.object({
  action: PlatformAction,
  reasoning: z.string().min(1),
});
export type DecisionResponseType = z.infer<typeof DecisionResponse>;

// ---------------------------------------------------------------------------
// POST /qde/inference — market direction/signal
// ---------------------------------------------------------------------------

export const InferenceRequest = z.object({
  symbol: z.string().min(1),
  currentPrice: z.number().optional().default(0),
  spread: z.number().optional().default(0),
  volatility: z.number().optional().default(0.01),
  momentum: z.number().optional().default(0),
  balance: z.number().optional().default(0),
  equity: z.number().optional().default(0),
  openPositions: z.number().int().optional().default(0),
});
export type InferenceRequestType = z.infer<typeof InferenceRequest>;

// ---------------------------------------------------------------------------
// POST /qde/review-position — single open position assessment
// ---------------------------------------------------------------------------

export const PositionReviewRequest = z.object({
  instrument: z.string().min(1),
  position: z.record(z.number()).default({}),
  market: z.record(z.unknown()).default({}),
  heads: z.record(z.unknown()).default({}),
  free_margin_pct: z.number().default(1.0),
  seconds_since_last_review: z.number().default(30.0),
});
export type PositionReviewRequestType = z.infer<typeof PositionReviewRequest>;

// ---------------------------------------------------------------------------
// POST /qde/portfolio-review — batch portfolio review
// ---------------------------------------------------------------------------

export const PortfolioReviewRequest = z.object({
  positions: z.array(TelemetryPayload).min(1),
});
export type PortfolioReviewRequestType = z.infer<typeof PortfolioReviewRequest>;
