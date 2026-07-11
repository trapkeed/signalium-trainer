/**
 * Outbound Signalium Platform HTTP client.
 *
 * All five push endpoints use X-Api-Key auth sourced from the
 * SIGNALIUM_PLATFORM_API_KEY environment secret. The client is fail-closed:
 * any call made without the secret configured throws PlatformClientError
 * immediately — no network request is made and no value is silently dropped.
 *
 * Configure before deployment:
 *   SIGNALIUM_PLATFORM_API_KEY=<key issued by Admin → API Keys on the platform>
 *
 * Optional override for staging/testing:
 *   SIGNALIUM_PLATFORM_BASE_URL=https://staging.signaliumai.com/api
 */

import { logger } from "./logger.js";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Base URL of the Signalium Platform API. Override via env for staging. */
export const PLATFORM_BASE =
  process.env["SIGNALIUM_PLATFORM_BASE_URL"] ?? "https://signaliumai.com/api";

const REQUEST_TIMEOUT_MS = 10_000;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/** Thrown on any failure: missing key, network error, timeout, or non-2xx. */
export class PlatformClientError extends Error {
  constructor(
    /** HTTP status code, or 0 for network/timeout/config errors. */
    public readonly status: number,
    /** Parsed response body from the platform, or null. */
    public readonly body: unknown,
    message: string,
  ) {
    super(message);
    this.name = "PlatformClientError";
  }
}

// ---------------------------------------------------------------------------
// Payload types — contract §4 plus inferred shapes for the four other endpoints
// ---------------------------------------------------------------------------

/** POST /operational/telemetry — live account/EA snapshot (contract §4). */
export interface TelemetryPayload {
  symbol: string;
  balance: number;
  equity: number;
  margin: number;
  freeMargin: number;
  openPositions: number;
  floatingPnl: number;
  entryPrice: number;
  currentPrice: number;
  spread: number;
  drawdown: number;
  tradeAgeSeconds: number;
  /**
   * Optional risk bucket ("core" | "growth" | "speculative").
   * When present, the platform applies per-bucket risk limits instead of
   * whole-account limits for the QDE gate.
   */
  bucketType?: "core" | "growth" | "speculative";
}

/** POST /operational/ticks — per-symbol tick. */
export interface TickPayload {
  symbol: string;
  bid: number;
  ask: number;
  /** Unix epoch milliseconds. */
  timestamp: number;
  volume?: number;
  spread?: number;
}

/** POST /operational/heartbeat — EA/runtime liveness ping. */
export interface HeartbeatPayload {
  status: "running" | "idle" | "degraded";
  /** Unix epoch milliseconds. */
  timestamp: number;
  /** Trainer instance identifier (EA account / broker server ID). */
  traderInstanceId?: string;
  /** Seconds since last restart. */
  uptime?: number;
}

/** POST /operational/execution-quality — trade fill metrics. */
export interface ExecutionQualityPayload {
  symbol: string;
  /** Platform or broker order reference (optional but recommended). */
  orderId?: string;
  requestedPrice: number;
  executedPrice: number;
  /** Executed minus requested, in price units. Negative = positive slippage. */
  slippage: number;
  /** Time from order submission to fill, in milliseconds. */
  fillTimeMs?: number;
  /** Unix epoch milliseconds of the fill event. */
  timestamp: number;
}

/** POST /symbols/discovery — new symbol/broker mapping for admin review. */
export interface SymbolDiscoveryPayload {
  broker: string;
  /** Raw symbol as seen at the broker (e.g. "XAUUSDc", "GOLD_pro"). */
  brokerSymbol: string;
  /** Canonical Signalium symbol (e.g. "XAUUSD"). */
  canonicalSymbol: string;
  description?: string;
}

/** Generic platform API success envelope. */
export interface PlatformResponse {
  success?: boolean;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Internal fetch helper
// ---------------------------------------------------------------------------

async function platformPost(
  path: string,
  payload: unknown,
): Promise<PlatformResponse> {
  const apiKey = process.env["SIGNALIUM_PLATFORM_API_KEY"];
  if (!apiKey) {
    throw new PlatformClientError(
      0,
      null,
      "SIGNALIUM_PLATFORM_API_KEY is not configured — outbound platform calls are disabled",
    );
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(`${PLATFORM_BASE}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Api-Key": apiKey,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeoutId);
    const isTimeout =
      err instanceof Error &&
      (err.name === "AbortError" || err.name === "TimeoutError");
    const msg = isTimeout
      ? `Platform request timed out after ${REQUEST_TIMEOUT_MS}ms (${path})`
      : `Platform request failed: ${String(err)} (${path})`;
    logger.error({ path, err }, "platform_client: outbound request failed");
    throw new PlatformClientError(0, null, msg);
  }

  clearTimeout(timeoutId);

  let body: unknown;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (!res.ok) {
    const msg = `Platform returned ${res.status} for POST ${path}`;
    logger.error(
      { path, status: res.status, body },
      "platform_client: non-2xx response from platform",
    );
    throw new PlatformClientError(res.status, body, msg);
  }

  return body as PlatformResponse;
}

// ---------------------------------------------------------------------------
// Public client API
// ---------------------------------------------------------------------------

/**
 * POST /operational/telemetry
 * Push a live account/EA telemetry snapshot to the platform.
 * Called by the trainer on every tick or cadence-based update.
 */
export async function pushTelemetry(
  payload: TelemetryPayload,
): Promise<PlatformResponse> {
  return platformPost("/operational/telemetry", payload);
}

/**
 * POST /operational/ticks
 * Push a per-symbol market tick to the platform's continuous tick stream.
 */
export async function pushTick(
  payload: TickPayload,
): Promise<PlatformResponse> {
  return platformPost("/operational/ticks", payload);
}

/**
 * POST /operational/heartbeat
 * Push an EA/runtime liveness ping. Call on a regular cadence (e.g. every 30s).
 */
export async function pushHeartbeat(
  payload: HeartbeatPayload,
): Promise<PlatformResponse> {
  return platformPost("/operational/heartbeat", payload);
}

/**
 * POST /operational/execution-quality
 * Push trade execution quality metrics after a fill is confirmed.
 */
export async function pushExecutionQuality(
  payload: ExecutionQualityPayload,
): Promise<PlatformResponse> {
  return platformPost("/operational/execution-quality", payload);
}

/**
 * POST /symbols/discovery
 * Submit a new broker symbol mapping for admin review on the platform.
 */
export async function pushSymbolDiscovery(
  payload: SymbolDiscoveryPayload,
): Promise<PlatformResponse> {
  return platformPost("/symbols/discovery", payload);
}
