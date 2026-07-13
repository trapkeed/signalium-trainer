/**
 * Tests for the platform Research Lab & Dataset contract routes.
 *
 * The Research Lab Python service is mocked at the fetch level so these
 * tests run without a live upstream.
 */

import { describe, it, expect, vi, beforeAll, afterEach } from "vitest";
import express from "express";
import request from "supertest";
import { webcrypto } from "crypto";
import { platformAuth } from "../middlewares/platform_auth.js";
import platformResearchRouter from "./platform_research.js";
import { _injectJwksForTest, clearJwksCache, type JwkPublicKey } from "../lib/jwks_client.js";

// ---------------------------------------------------------------------------
// Test app + JWT helpers (mirrors pattern from inference.test.ts)
// ---------------------------------------------------------------------------

function makeApp() {
  const app = express();
  app.use(express.json());
  app.use(platformAuth);
  app.use(platformResearchRouter);
  return app;
}

async function generateKeyPair() {
  const pair = await webcrypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const exported = await webcrypto.subtle.exportKey("jwk", pair.publicKey);
  const jwkPublic: JwkPublicKey = { ...exported, kid: "rl-kid-1" } as JwkPublicKey;
  return { privateKey: pair.privateKey, jwkPublic };
}

function strToBuffer(s: string): ArrayBuffer {
  return new TextEncoder().encode(s).buffer.slice(0) as ArrayBuffer;
}

function b64url(buf: ArrayBuffer): string {
  return Buffer.from(buf)
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

async function makeJwt(privateKey: CryptoKey): Promise<string> {
  const header = { alg: "ES256", typ: "JWT", kid: "rl-kid-1" };
  const payload = {
    iss: "signalium",
    aud: "signalium-trainer",
    sub: "signalium-platform-service",
    exp: Math.floor(Date.now() / 1000) + 300,
  };
  const enc = `${b64url(strToBuffer(JSON.stringify(header)))}.${b64url(strToBuffer(JSON.stringify(payload)))}`;
  const sig = await webcrypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    privateKey,
    strToBuffer(enc),
  );
  return `${enc}.${b64url(sig)}`;
}

let privateKey: CryptoKey;
let jwkPublic: JwkPublicKey;
const app = makeApp();

beforeAll(async () => {
  ({ privateKey, jwkPublic } = await generateKeyPair());
  // Provide a test admin token so rlAuthHeaders() does not return null
  process.env["RESEARCH_LAB_ADMIN_TOKEN"] = "test-admin-token";
});

afterEach(() => {
  clearJwksCache();
  vi.restoreAllMocks();
});

async function authHeader(): Promise<string> {
  _injectJwksForTest([jwkPublic]);
  return `Bearer ${await makeJwt(privateKey)}`;
}

// ---------------------------------------------------------------------------
// Auth guard — all 8 routes reject unauthenticated requests
// ---------------------------------------------------------------------------

describe("auth guard", () => {
  const unauthRoutes: Array<{ method: "get" | "post"; path: string }> = [
    { method: "get", path: "/experiments" },
    { method: "post", path: "/experiments" },
    { method: "get", path: "/training/runs" },
    { method: "post", path: "/training/runs" },
    { method: "post", path: "/shadow-testing/deployments" },
    { method: "post", path: "/deployment-candidates/abc/approve" },
    { method: "get", path: "/genealogy/tree" },
    { method: "get", path: "/datasets" },
  ];

  for (const { method, path } of unauthRoutes) {
    it(`${method.toUpperCase()} ${path} returns 401 without token`, async () => {
      const res = await (request(app) as unknown as Record<string, (p: string) => request.Test>)[method](path);
      expect(res.status).toBe(401);
      expect(res.body).toEqual({ error: "Unauthorized" });
    });
  }
});

// ---------------------------------------------------------------------------
// Auth header forwarding — cross-cutting concern for all proxy routes
// ---------------------------------------------------------------------------

describe("RL upstream auth header forwarding", () => {
  it("includes Authorization: Bearer <RESEARCH_LAB_ADMIN_TOKEN> on outbound GET calls", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: () => Promise.resolve({ success: true, experiments: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    await request(app).get("/experiments").set("Authorization", await authHeader());

    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const sentAuth = (opts?.headers as Record<string, string>)?.["Authorization"] ?? "";
    expect(sentAuth).toBe("Bearer test-admin-token");
  });

  it("includes Authorization header on outbound POST calls", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true, status: 201,
      json: () => Promise.resolve({ success: true }),
    });
    vi.stubGlobal("fetch", mockFetch);

    await request(app).post("/experiments")
      .set("Authorization", await authHeader())
      .send({ name: "exp" });

    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const sentAuth = (opts?.headers as Record<string, string>)?.["Authorization"] ?? "";
    expect(sentAuth).toBe("Bearer test-admin-token");
  });

  it("does NOT forward the incoming platform JWT to the RL server", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: () => Promise.resolve({ success: true, experiments: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const platformJwt = await makeJwt(privateKey);
    _injectJwksForTest([jwkPublic]);
    await request(app).get("/experiments")
      .set("Authorization", `Bearer ${platformJwt}`);

    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const sentAuth = (opts?.headers as Record<string, string>)?.["Authorization"] ?? "";
    // Must be the internal admin token, not the platform JWT
    expect(sentAuth).toBe("Bearer test-admin-token");
    expect(sentAuth).not.toContain(platformJwt);
  });

  it("returns 503 when RESEARCH_LAB_ADMIN_TOKEN is not configured", async () => {
    const original = process.env["RESEARCH_LAB_ADMIN_TOKEN"];
    delete process.env["RESEARCH_LAB_ADMIN_TOKEN"];
    try {
      const res = await request(app)
        .get("/experiments")
        .set("Authorization", await authHeader());
      expect(res.status).toBe(503);
      expect(res.body).toHaveProperty("error");
      expect(res.body).not.toHaveProperty("success");
    } finally {
      process.env["RESEARCH_LAB_ADMIN_TOKEN"] = original;
    }
  });
});

// ---------------------------------------------------------------------------
// GET /experiments
// ---------------------------------------------------------------------------

describe("GET /experiments", () => {
  it("proxies to /research/experiments and returns the response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true, experiments: [] }),
    }));
    const res = await request(app)
      .get("/experiments")
      .set("Authorization", await authHeader());
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("experiments");
  });

  it("returns 502 with { error } when RL service is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));
    const res = await request(app)
      .get("/experiments")
      .set("Authorization", await authHeader());
    expect(res.status).toBe(502);
    expect(res.body).toHaveProperty("error");
    expect(res.body).not.toHaveProperty("success");
  });
});

// ---------------------------------------------------------------------------
// POST /experiments
// ---------------------------------------------------------------------------

describe("POST /experiments", () => {
  it("proxies body to /research/experiments", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: () => Promise.resolve({ success: true, experiment_id: "exp-001" }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const body = { name: "test-experiment", researcher: "alice" };
    const res = await request(app)
      .post("/experiments")
      .set("Authorization", await authHeader())
      .send(body);
    expect(res.status).toBe(201);

    const [calledUrl, calledOpts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(calledUrl).toContain("/research/experiments");
    expect(JSON.parse(calledOpts.body as string)).toMatchObject(body);
  });
});

// ---------------------------------------------------------------------------
// GET /training/runs
// ---------------------------------------------------------------------------

describe("GET /training/runs", () => {
  it("proxies to /training/runs", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true, training_runs: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const res = await request(app)
      .get("/training/runs?experiment_id=exp-1&limit=10")
      .set("Authorization", await authHeader());
    expect(res.status).toBe(200);

    const [calledUrl] = mockFetch.mock.calls[0] as [string];
    expect(calledUrl).toContain("/training/runs");
    expect(calledUrl).toContain("experiment_id=exp-1");
    expect(calledUrl).toContain("limit=10");
  });
});

// ---------------------------------------------------------------------------
// POST /training/runs
// ---------------------------------------------------------------------------

describe("POST /training/runs", () => {
  it("proxies body to /training/runs", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: () => Promise.resolve({ success: true, training_run: { run_id: "run-001" } }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const body = { experiment_id: "exp-1", dataset_version: "v1" };
    const res = await request(app)
      .post("/training/runs")
      .set("Authorization", await authHeader())
      .send(body);
    expect(res.status).toBe(201);

    const [calledUrl] = mockFetch.mock.calls[0] as [string];
    expect(calledUrl).toContain("/training/runs");
  });
});

// ---------------------------------------------------------------------------
// POST /shadow-testing/deployments
// ---------------------------------------------------------------------------

describe("POST /shadow-testing/deployments", () => {
  it("proxies to /shadow-testing/deployments", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: () => Promise.resolve({ success: true, test_id: "st-001" }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const body = { challenger_version_id: "v2", control_version_id: "v1" };
    const res = await request(app)
      .post("/shadow-testing/deployments")
      .set("Authorization", await authHeader())
      .send(body);
    expect(res.status).toBe(201);

    const [calledUrl] = mockFetch.mock.calls[0] as [string];
    expect(calledUrl).toContain("/shadow-testing/deployments");
  });
});

// ---------------------------------------------------------------------------
// POST /deployment-candidates/:id/approve
// ---------------------------------------------------------------------------

describe("POST /deployment-candidates/:id/approve", () => {
  it("proxies to /deployment-candidates/:id/approve with correct id", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const res = await request(app)
      .post("/deployment-candidates/exp-123/approve")
      .set("Authorization", await authHeader())
      .send({ approver: "alice" });
    expect(res.status).toBe(200);

    const [calledUrl] = mockFetch.mock.calls[0] as [string];
    expect(calledUrl).toContain("/deployment-candidates/exp-123/approve");
  });

  it("returns 502 with { error } (no success field) when RL is down", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
    const res = await request(app)
      .post("/deployment-candidates/exp-999/approve")
      .set("Authorization", await authHeader())
      .send({});
    expect(res.status).toBe(502);
    expect(res.body).toHaveProperty("error");
    expect(res.body).not.toHaveProperty("success");
  });
});

// ---------------------------------------------------------------------------
// GET /genealogy/tree
// ---------------------------------------------------------------------------

describe("GET /genealogy/tree", () => {
  it("proxies to /genealogy/tree and returns tree data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true, data: { nodes: [], total_experiments: 0, total_runs: 0 } }),
    }));
    const res = await request(app)
      .get("/genealogy/tree")
      .set("Authorization", await authHeader());
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("data");
  });

  it("passes limit query param through", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true, data: { nodes: [] } }),
    });
    vi.stubGlobal("fetch", mockFetch);

    await request(app)
      .get("/genealogy/tree?limit=50")
      .set("Authorization", await authHeader());

    const [calledUrl] = mockFetch.mock.calls[0] as [string];
    expect(calledUrl).toContain("limit=50");
  });
});

// ---------------------------------------------------------------------------
// GET /datasets
// ---------------------------------------------------------------------------

describe("GET /datasets", () => {
  it("proxies to /datasets", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ success: true, datasets: [] }),
    });
    vi.stubGlobal("fetch", mockFetch);

    const res = await request(app)
      .get("/datasets")
      .set("Authorization", await authHeader());
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("datasets");

    const [calledUrl] = mockFetch.mock.calls[0] as [string];
    expect(calledUrl).toContain("/datasets");
  });

  it("returns 502 with { error } (no success field) when RL is down", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("timeout")));
    const res = await request(app)
      .get("/datasets")
      .set("Authorization", await authHeader());
    expect(res.status).toBe(502);
    expect(res.body).toHaveProperty("error");
    expect(res.body).not.toHaveProperty("success");
  });
});
