/**
 * Tests for the outbound Signalium Platform HTTP client.
 *
 * fetch is globally stubbed — no real network requests are made.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  pushTelemetry,
  pushTick,
  pushHeartbeat,
  pushExecutionQuality,
  pushSymbolDiscovery,
  PlatformClientError,
  PLATFORM_BASE,
  type TelemetryPayload,
  type TickPayload,
  type HeartbeatPayload,
  type ExecutionQualityPayload,
  type SymbolDiscoveryPayload,
} from "./platform_client.js";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const TEST_API_KEY = "test-platform-api-key";

const telemetry: TelemetryPayload = {
  symbol: "XAUUSD",
  balance: 10000,
  equity: 9850,
  margin: 500,
  freeMargin: 9350,
  openPositions: 2,
  floatingPnl: -150,
  entryPrice: 1925.5,
  currentPrice: 1920.1,
  spread: 0.3,
  drawdown: 1.5,
  tradeAgeSeconds: 3600,
};

const tick: TickPayload = {
  symbol: "XAUUSD",
  bid: 1920.0,
  ask: 1920.3,
  timestamp: 1720000000000,
};

const heartbeat: HeartbeatPayload = {
  status: "running",
  timestamp: 1720000000000,
  uptime: 3600,
};

const execQuality: ExecutionQualityPayload = {
  symbol: "XAUUSD",
  requestedPrice: 1920.0,
  executedPrice: 1920.1,
  slippage: 0.1,
  timestamp: 1720000000000,
};

const discovery: SymbolDiscoveryPayload = {
  broker: "ICMarkets",
  brokerSymbol: "XAUUSDc",
  canonicalSymbol: "XAUUSD",
};

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

beforeEach(() => {
  process.env["SIGNALIUM_PLATFORM_API_KEY"] = TEST_API_KEY;
});

afterEach(() => {
  delete process.env["SIGNALIUM_PLATFORM_API_KEY"];
  vi.restoreAllMocks();
});

function mockSuccess(body: Record<string, unknown> = { success: true }) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve(body),
    }),
  );
}

// ---------------------------------------------------------------------------
// Fail-closed — missing API key
// ---------------------------------------------------------------------------

describe("fail-closed: missing SIGNALIUM_PLATFORM_API_KEY", () => {
  beforeEach(() => {
    delete process.env["SIGNALIUM_PLATFORM_API_KEY"];
    vi.stubGlobal("fetch", vi.fn()); // should never be called
  });

  it("pushTelemetry throws PlatformClientError with status 0", async () => {
    await expect(pushTelemetry(telemetry)).rejects.toBeInstanceOf(PlatformClientError);
    const err = await pushTelemetry(telemetry).catch((e: unknown) => e as PlatformClientError);
    expect(err.status).toBe(0);
    expect(err.message).toMatch(/not configured/i);
  });

  it("no network request is made when key is absent", async () => {
    await pushTelemetry(telemetry).catch(() => {});
    expect(vi.mocked(fetch)).not.toHaveBeenCalled();
  });

  it("pushTick fails closed", async () => {
    await expect(pushTick(tick)).rejects.toBeInstanceOf(PlatformClientError);
  });

  it("pushHeartbeat fails closed", async () => {
    await expect(pushHeartbeat(heartbeat)).rejects.toBeInstanceOf(PlatformClientError);
  });

  it("pushExecutionQuality fails closed", async () => {
    await expect(pushExecutionQuality(execQuality)).rejects.toBeInstanceOf(PlatformClientError);
  });

  it("pushSymbolDiscovery fails closed", async () => {
    await expect(pushSymbolDiscovery(discovery)).rejects.toBeInstanceOf(PlatformClientError);
  });
});

// ---------------------------------------------------------------------------
// X-Api-Key header is sent on every outbound call
// ---------------------------------------------------------------------------

describe("X-Api-Key header forwarding", () => {
  it("pushTelemetry sends X-Api-Key header", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushTelemetry(telemetry);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((opts.headers as Record<string, string>)["X-Api-Key"]).toBe(TEST_API_KEY);
  });

  it("pushTick sends X-Api-Key header", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushTick(tick);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((opts.headers as Record<string, string>)["X-Api-Key"]).toBe(TEST_API_KEY);
  });

  it("pushHeartbeat sends X-Api-Key header", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushHeartbeat(heartbeat);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((opts.headers as Record<string, string>)["X-Api-Key"]).toBe(TEST_API_KEY);
  });

  it("pushExecutionQuality sends X-Api-Key header", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushExecutionQuality(execQuality);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((opts.headers as Record<string, string>)["X-Api-Key"]).toBe(TEST_API_KEY);
  });

  it("pushSymbolDiscovery sends X-Api-Key header", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushSymbolDiscovery(discovery);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect((opts.headers as Record<string, string>)["X-Api-Key"]).toBe(TEST_API_KEY);
  });
});

// ---------------------------------------------------------------------------
// Correct endpoint paths
// ---------------------------------------------------------------------------

describe("endpoint path routing", () => {
  it("pushTelemetry POSTs to /operational/telemetry", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushTelemetry(telemetry);
    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toBe(`${PLATFORM_BASE}/operational/telemetry`);
  });

  it("pushTick POSTs to /operational/ticks", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushTick(tick);
    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toBe(`${PLATFORM_BASE}/operational/ticks`);
  });

  it("pushHeartbeat POSTs to /operational/heartbeat", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushHeartbeat(heartbeat);
    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toBe(`${PLATFORM_BASE}/operational/heartbeat`);
  });

  it("pushExecutionQuality POSTs to /operational/execution-quality", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushExecutionQuality(execQuality);
    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toBe(`${PLATFORM_BASE}/operational/execution-quality`);
  });

  it("pushSymbolDiscovery POSTs to /symbols/discovery", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushSymbolDiscovery(discovery);
    const [url] = mockFetch.mock.calls[0] as [string];
    expect(url).toBe(`${PLATFORM_BASE}/symbols/discovery`);
  });
});

// ---------------------------------------------------------------------------
// Payload serialisation
// ---------------------------------------------------------------------------

describe("payload forwarding", () => {
  it("pushTelemetry sends correct payload including optional bucketType", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    const withBucket: TelemetryPayload = { ...telemetry, bucketType: "speculative" };
    await pushTelemetry(withBucket);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const sent = JSON.parse(opts.body as string) as TelemetryPayload;
    expect(sent.symbol).toBe("XAUUSD");
    expect(sent.bucketType).toBe("speculative");
    expect(sent.balance).toBe(10000);
  });

  it("pushTelemetry omits bucketType when not provided", async () => {
    const mockFetch = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({}) });
    vi.stubGlobal("fetch", mockFetch);
    await pushTelemetry(telemetry);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const sent = JSON.parse(opts.body as string) as Record<string, unknown>;
    expect(sent["bucketType"]).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Non-2xx → PlatformClientError with status + body
// ---------------------------------------------------------------------------

describe("non-2xx error handling", () => {
  it("throws PlatformClientError with correct status on 401", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
      json: () => Promise.resolve({ error: "Unauthorized" }),
    }));
    const err = await pushTelemetry(telemetry).catch((e: unknown) => e as PlatformClientError);
    expect(err).toBeInstanceOf(PlatformClientError);
    expect(err.status).toBe(401);
    expect(err.body).toMatchObject({ error: "Unauthorized" });
  });

  it("throws PlatformClientError with correct status on 500", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ message: "internal error" }),
    }));
    const err = await pushTelemetry(telemetry).catch((e: unknown) => e as PlatformClientError);
    expect(err.status).toBe(500);
  });

  it("body is null when response is not valid JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: () => Promise.reject(new SyntaxError("bad json")),
    }));
    const err = await pushTelemetry(telemetry).catch((e: unknown) => e as PlatformClientError);
    expect(err.status).toBe(503);
    expect(err.body).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Network / timeout errors → PlatformClientError with status 0
// ---------------------------------------------------------------------------

describe("network and timeout errors", () => {
  it("throws PlatformClientError(0) on network failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));
    const err = await pushTelemetry(telemetry).catch((e: unknown) => e as PlatformClientError);
    expect(err).toBeInstanceOf(PlatformClientError);
    expect(err.status).toBe(0);
  });

  it("throws PlatformClientError(0) on AbortError (timeout)", async () => {
    const abortErr = Object.assign(new Error("The operation was aborted"), { name: "AbortError" });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(abortErr));
    const err = await pushTelemetry(telemetry).catch((e: unknown) => e as PlatformClientError);
    expect(err).toBeInstanceOf(PlatformClientError);
    expect(err.status).toBe(0);
    expect(err.message).toMatch(/timed out/i);
  });
});

// ---------------------------------------------------------------------------
// Happy path — returns parsed response body
// ---------------------------------------------------------------------------

describe("happy path", () => {
  it("pushTelemetry returns platform response body", async () => {
    mockSuccess({ success: true, recorded: true });
    const result = await pushTelemetry(telemetry);
    expect(result).toMatchObject({ success: true, recorded: true });
  });

  it("pushTick returns platform response body", async () => {
    mockSuccess({ success: true });
    const result = await pushTick(tick);
    expect(result).toMatchObject({ success: true });
  });

  it("pushHeartbeat returns platform response body", async () => {
    mockSuccess({ ok: true });
    const result = await pushHeartbeat(heartbeat);
    expect(result).toMatchObject({ ok: true });
  });

  it("pushExecutionQuality returns platform response body", async () => {
    mockSuccess({ accepted: true });
    const result = await pushExecutionQuality(execQuality);
    expect(result).toMatchObject({ accepted: true });
  });

  it("pushSymbolDiscovery returns platform response body", async () => {
    mockSuccess({ queued: true });
    const result = await pushSymbolDiscovery(discovery);
    expect(result).toMatchObject({ queued: true });
  });
});
