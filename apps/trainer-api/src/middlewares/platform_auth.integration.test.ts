/**
 * Integration tests for the platformAuth middleware wired into a real
 * Express app via platformRouter.
 *
 * Tests prove that the JWT middleware is enforced end-to-end at the
 * HTTP request level — not just in unit isolation.
 */

import { describe, it, expect, beforeAll, afterEach } from "vitest";
import express from "express";
import request from "supertest";
import { webcrypto } from "crypto";
import { platformAuth } from "./platform_auth.js";
import { _injectJwksForTest, clearJwksCache, type JwkPublicKey } from "../lib/jwks_client.js";

// ---------------------------------------------------------------------------
// Test app setup
// ---------------------------------------------------------------------------

/** Create a minimal Express app with platformAuth protecting a /protected route. */
function makeApp() {
  const app = express();
  app.use(express.json());

  // Health endpoint — no auth (mimics app.ts)
  app.get("/health", (_req, res) => {
    res.status(200).json({ status: "ok" });
  });

  // Protected platform route — requires valid JWT
  app.get("/protected", platformAuth, (_req, res) => {
    res.status(200).json({ ok: true });
  });

  return app;
}

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

async function generateEcKeyPair(): Promise<{
  privateKey: CryptoKey;
  jwkPublic: JwkPublicKey;
}> {
  const pair = await webcrypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const exported = await webcrypto.subtle.exportKey("jwk", pair.publicKey);
  const jwkPublic: JwkPublicKey = { ...exported, kid: "integ-kid-1" } as JwkPublicKey;
  return { privateKey: pair.privateKey, jwkPublic };
}

function strToBuffer(s: string): ArrayBuffer {
  const u8 = new TextEncoder().encode(s);
  return u8.buffer.slice(0) as ArrayBuffer;
}

function base64urlEncode(buf: ArrayBuffer): string {
  return Buffer.from(buf)
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

async function buildValidJwt(privateKey: CryptoKey, overrideClaims?: Record<string, unknown>): Promise<string> {
  const header = { alg: "ES256", typ: "JWT", kid: "integ-kid-1" };
  const payload = {
    iss: "signalium",
    aud: "signalium-trainer",
    sub: "signalium-platform-service",
    exp: Math.floor(Date.now() / 1000) + 300,
    iat: Math.floor(Date.now() / 1000),
    ...overrideClaims,
  };
  const encHeader = base64urlEncode(strToBuffer(JSON.stringify(header)));
  const encPayload = base64urlEncode(strToBuffer(JSON.stringify(payload)));
  const signingInput = `${encHeader}.${encPayload}`;
  const sig = await webcrypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    privateKey,
    strToBuffer(signingInput),
  );
  return `${signingInput}.${base64urlEncode(sig)}`;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

let privateKey: CryptoKey;
let jwkPublic: JwkPublicKey;
const app = makeApp();

beforeAll(async () => {
  const pair = await generateEcKeyPair();
  privateKey = pair.privateKey;
  jwkPublic = pair.jwkPublic;
});

afterEach(() => {
  clearJwksCache();
});

describe("GET /health (no auth required)", () => {
  it("returns 200 without any Authorization header", async () => {
    const res = await request(app).get("/health");
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ status: "ok" });
  });
});

describe("protected route — platformAuth enforcement", () => {
  it("returns 401 with no Authorization header", async () => {
    const res = await request(app).get("/protected");
    expect(res.status).toBe(401);
    expect(res.body).toEqual({ error: "Unauthorized" });
  });

  it("returns 401 with a malformed (non-JWT) token", async () => {
    const res = await request(app)
      .get("/protected")
      .set("Authorization", "Bearer notajwt");
    expect(res.status).toBe(401);
    expect(res.body).toEqual({ error: "Unauthorized" });
  });

  it("returns 200 with a valid signed JWT", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildValidJwt(privateKey);
    const res = await request(app)
      .get("/protected")
      .set("Authorization", `Bearer ${token}`);
    expect(res.status).toBe(200);
    expect(res.body).toEqual({ ok: true });
  });

  it("returns 401 with an expired token", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildValidJwt(privateKey, {
      exp: Math.floor(Date.now() / 1000) - 10,
    });
    const res = await request(app)
      .get("/protected")
      .set("Authorization", `Bearer ${token}`);
    expect(res.status).toBe(401);
    expect(res.body).toEqual({ error: "Unauthorized" });
  });

  it("returns 401 when exp is missing (fail-closed)", async () => {
    _injectJwksForTest([jwkPublic]);
    // Build JWT without exp claim
    const header = { alg: "ES256", typ: "JWT", kid: "integ-kid-1" };
    const payload = { iss: "signalium", aud: "signalium-trainer", sub: "svc" }; // no exp
    const encHeader = base64urlEncode(strToBuffer(JSON.stringify(header)));
    const encPayload = base64urlEncode(strToBuffer(JSON.stringify(payload)));
    const signingInput = `${encHeader}.${encPayload}`;
    const sig = await webcrypto.subtle.sign(
      { name: "ECDSA", hash: "SHA-256" },
      privateKey,
      strToBuffer(signingInput),
    );
    const token = `${signingInput}.${base64urlEncode(sig)}`;
    const res = await request(app)
      .get("/protected")
      .set("Authorization", `Bearer ${token}`);
    expect(res.status).toBe(401);
  });

  it("returns 401 when exp is a non-numeric string (fail-closed)", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildValidJwt(privateKey, { exp: "never" });
    const res = await request(app)
      .get("/protected")
      .set("Authorization", `Bearer ${token}`);
    expect(res.status).toBe(401);
  });

  it("returns 401 with wrong issuer", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildValidJwt(privateKey, { iss: "attacker.com" });
    const res = await request(app)
      .get("/protected")
      .set("Authorization", `Bearer ${token}`);
    expect(res.status).toBe(401);
  });

  it("returns 401 with wrong audience", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildValidJwt(privateKey, { aud: "other-service" });
    const res = await request(app)
      .get("/protected")
      .set("Authorization", `Bearer ${token}`);
    expect(res.status).toBe(401);
  });
});
