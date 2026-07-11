/**
 * Tests for the platform inference endpoints.
 *
 * The QDE Python service is mocked at the fetch level so these tests
 * run without a live upstream.
 */

import { describe, it, expect, vi, beforeAll, afterEach, beforeEach } from "vitest";
import express from "express";
import request from "supertest";
import { webcrypto } from "crypto";
import { platformAuth } from "../middlewares/platform_auth.js";
import inferenceRouter from "./inference.js";
import { _injectJwksForTest, clearJwksCache, type JwkPublicKey } from "../lib/jwks_client.js";

// ---------------------------------------------------------------------------
// Test app + JWT helpers
// ---------------------------------------------------------------------------

function makeApp() {
  const app = express();
  app.use(express.json());
  app.use(platformAuth);
  app.use(inferenceRouter);
  return app;
}

async function generateKeyPair() {
  const pair = await webcrypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const exported = await webcrypto.subtle.exportKey("jwk", pair.publicKey);
  const jwkPublic: JwkPublicKey = { ...exported, kid: "inf-kid-1" } as JwkPublicKey;
  return { privateKey: pair.privateKey, jwkPublic };
}

function strToBuffer(s: string): ArrayBuffer {
  return new TextEncoder().encode(s).buffer.slice(0) as ArrayBuffer;
}

function b64url(buf: ArrayBuffer): string {
  return Buffer.from(buf).toString("base64")
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function makeJwt(privateKey: CryptoKey): Promise<string> {
  const header = { alg: "ES256", typ: "JWT", kid: "inf-kid-1" };
  const payload = {
    iss: "signalium", aud: "signalium-trainer",
    sub: "signalium-platform-service",
    exp: Math.floor(Date.now() / 1000) + 300,
  };
  const enc = `${b64url(strToBuffer(JSON.stringify(header)))}.${b64url(strToBuffer(JSON.stringify(payload)))}`;
  const sig = await webcrypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, privateKey, strToBuffer(enc));
  return `${enc}.${b64url(sig)}`;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

let privateKey: CryptoKey;
let jwkPublic: JwkPublicKey;
const app = makeApp();

beforeAll(async () => {
  ({ privateKey, jwkPublic } = await generateKeyPair());
});

afterEach(() => {
  clearJwksCache();
  vi.restoreAllMocks();
});

// Inject auth for authenticated tests
async function authHeader(): Promise<string> {
  _injectJwksForTest([jwkPublic]);
  return `Bearer ${await makeJwt(privateKey)}`;
}

// ---------------------------------------------------------------------------
// Auth guard tests (apply to all four routes)
// ---------------------------------------------------------------------------

describe("auth guard", () => {
  it("POST / returns 401 without token", async () => {
    const res = await request(app).post("/").send({ symbol: "XAUUSD" });
    expect(res.status).toBe(401);
    expect(res.body).toEqual({ error: "Unauthorized" });
  });

  it("POST /qde/inference returns 401 without token", async () => {
    const res = await request(app).post("/qde/inference").send({ symbol: "XAUUSD" });
    expect(res.status).toBe(401);
  });

  it("POST /qde/review-position returns 401 without token", async () => {
    const res = await request(app).post("/qde/review-position").send({ symbol: "XAUUSD" });
    expect(res.status).toBe(401);
  });

  it("POST /qde/portfolio-review returns 401 without token", async () => {
    const res = await request(app).post("/qde/portfolio-review").send({ positions: [] });
    expect(res.status).toBe(401);
  });
});

// ---------------------------------------------------------------------------
// POST / — legacy QDE decision
// ---------------------------------------------------------------------------

describe("POST /", () => {
  it("returns { action, reasoning } on a valid telemetry payload", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ action: "HOLD", reasoning: "No threshold crossed." }),
    }));
    const res = await request(app)
      .post("/")
      .set("Authorization", await authHeader())
      .send({ symbol: "XAUUSD", currentPrice: 1920.1, spread: 0.3 });
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ action: "HOLD", reasoning: expect.any(String) });
  });

  it("normalizes MOVE_TO_BREAKEVEN → BREAKEVEN", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ action: "MOVE_TO_BREAKEVEN", reasoning: "moved" }),
    }));
    const res = await request(app)
      .post("/")
      .set("Authorization", await authHeader())
      .send({ symbol: "XAUUSD" });
    expect(res.status).toBe(200);
    expect(res.body.action).toBe("BREAKEVEN");
  });

  it("returns 400 when symbol is missing", async () => {
    const res = await request(app)
      .post("/")
      .set("Authorization", await authHeader())
      .send({ currentPrice: 1920 });
    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty("error");
  });

  it("returns 502 when QDE service is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));
    const res = await request(app)
      .post("/")
      .set("Authorization", await authHeader())
      .send({ symbol: "XAUUSD" });
    expect(res.status).toBe(502);
    expect(res.body).toHaveProperty("error");
  });

  it("error response has no success field", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
    const res = await request(app)
      .post("/")
      .set("Authorization", await authHeader())
      .send({ symbol: "XAUUSD" });
    expect(res.body).not.toHaveProperty("success");
  });
});

// ---------------------------------------------------------------------------
// POST /qde/inference
// ---------------------------------------------------------------------------

describe("POST /qde/inference", () => {
  it("returns a direction signal on valid request", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ direction: "FLAT", confidence: 0, reasoning: "no model" }),
    }));
    const res = await request(app)
      .post("/qde/inference")
      .set("Authorization", await authHeader())
      .send({ symbol: "XAUUSD", currentPrice: 1920 });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("direction");
  });

  it("returns 400 when symbol is missing", async () => {
    const res = await request(app)
      .post("/qde/inference")
      .set("Authorization", await authHeader())
      .send({ currentPrice: 1920 });
    expect(res.status).toBe(400);
  });
});

// ---------------------------------------------------------------------------
// POST /qde/review-position
// ---------------------------------------------------------------------------

describe("POST /qde/review-position", () => {
  it("handles flat telemetry (symbol field) via platform/decide", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ action: "TRAIL", reasoning: "trailing stop" }),
    }));
    const res = await request(app)
      .post("/qde/review-position")
      .set("Authorization", await authHeader())
      .send({ symbol: "XAUUSD", currentPrice: 1920, entryPrice: 1900 });
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({ action: "TRAIL" });
  });

  it("handles structured PositionTelemetryRequest (instrument field)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        success: true,
        decision: { instrument: "XAUUSD", action: "HOLD", reasoning: "holding" },
      }),
    }));
    const res = await request(app)
      .post("/qde/review-position")
      .set("Authorization", await authHeader())
      .send({ instrument: "XAUUSD", position: {}, market: {}, heads: {} });
    expect(res.status).toBe(200);
    expect(res.body.decision.action).toBe("HOLD");
  });

  it("normalizes MOVE_TO_BREAKEVEN in structured response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        success: true,
        decision: { action: "MOVE_TO_BREAKEVEN", reasoning: "moved" },
      }),
    }));
    const res = await request(app)
      .post("/qde/review-position")
      .set("Authorization", await authHeader())
      .send({ instrument: "XAUUSD", position: {}, market: {}, heads: {} });
    expect(res.status).toBe(200);
    expect(res.body.decision.action).toBe("BREAKEVEN");
  });
});

// ---------------------------------------------------------------------------
// POST /qde/portfolio-review
// ---------------------------------------------------------------------------

describe("POST /qde/portfolio-review", () => {
  it("returns decisions array", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        decisions: [{ symbol: "XAUUSD", action: "HOLD", reasoning: "holding", confidence: 0 }],
      }),
    }));
    const res = await request(app)
      .post("/qde/portfolio-review")
      .set("Authorization", await authHeader())
      .send({ positions: [{ symbol: "XAUUSD" }] });
    expect(res.status).toBe(200);
    expect(Array.isArray(res.body.decisions)).toBe(true);
  });

  it("normalizes MOVE_TO_BREAKEVEN in decisions", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        decisions: [{ symbol: "XAUUSD", action: "MOVE_TO_BREAKEVEN", reasoning: "moved" }],
      }),
    }));
    const res = await request(app)
      .post("/qde/portfolio-review")
      .set("Authorization", await authHeader())
      .send({ positions: [{ symbol: "XAUUSD" }] });
    expect(res.status).toBe(200);
    expect(res.body.decisions[0].action).toBe("BREAKEVEN");
  });

  it("returns 400 when positions array is empty", async () => {
    const res = await request(app)
      .post("/qde/portfolio-review")
      .set("Authorization", await authHeader())
      .send({ positions: [] });
    expect(res.status).toBe(400);
  });

  it("returns 400 when positions field is missing", async () => {
    const res = await request(app)
      .post("/qde/portfolio-review")
      .set("Authorization", await authHeader())
      .send({ symbol: "XAUUSD" });
    expect(res.status).toBe(400);
  });
});
