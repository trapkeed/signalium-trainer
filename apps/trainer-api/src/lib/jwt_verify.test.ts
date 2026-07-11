/**
 * Unit tests for verifyPlatformJwt.
 *
 * All tests generate real ES256 key pairs using Node's webcrypto so the
 * signature is genuine — only the claims vary between cases.
 */

import { describe, it, expect, beforeAll, afterEach } from "vitest";
import { webcrypto } from "crypto";
import { verifyPlatformJwt, JwtVerifyError } from "./jwt_verify.js";
import { _injectJwksForTest, clearJwksCache, type JwkPublicKey } from "./jwks_client.js";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

async function generateEcKeyPair(): Promise<{
  privateKey: CryptoKey;
  publicKey: CryptoKey;
  jwkPublic: JwkPublicKey;
}> {
  const pair = await webcrypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const exported = await webcrypto.subtle.exportKey("jwk", pair.publicKey);
  const jwkPublic: JwkPublicKey = { ...exported, kid: "test-kid-1" } as JwkPublicKey;
  return { privateKey: pair.privateKey, publicKey: pair.publicKey, jwkPublic };
}

/** Encode any ArrayBuffer or Uint8Array to base64url. */
function base64urlEncode(buf: ArrayBuffer): string {
  return Buffer.from(buf)
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

function strToBuffer(s: string): ArrayBuffer {
  const u8 = new TextEncoder().encode(s);
  return u8.buffer.slice(0) as ArrayBuffer;
}

async function buildJwt(
  privateKey: CryptoKey,
  payload: Record<string, unknown>,
  overrideHeader?: Record<string, unknown>,
): Promise<string> {
  const header: Record<string, unknown> = {
    alg: "ES256",
    typ: "JWT",
    kid: "test-kid-1",
    ...overrideHeader,
  };
  // Remove undefined keys (e.g. when overrideHeader sets kid: undefined)
  for (const k of Object.keys(header)) {
    if (header[k] === undefined) delete header[k];
  }
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

function validClaims(): Record<string, unknown> {
  return {
    iss: "signalium",
    aud: "signalium-trainer",
    sub: "signalium-platform-service",
    exp: Math.floor(Date.now() / 1000) + 300,
    iat: Math.floor(Date.now() / 1000),
  };
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

let privateKey: CryptoKey;
let jwkPublic: JwkPublicKey;

beforeAll(async () => {
  const pair = await generateEcKeyPair();
  privateKey = pair.privateKey;
  jwkPublic = pair.jwkPublic;
});

afterEach(() => {
  clearJwksCache();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("verifyPlatformJwt", () => {
  it("accepts a valid ES256 token", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, validClaims());
    const claims = await verifyPlatformJwt(token);
    expect(claims.iss).toBe("signalium");
    expect(claims.aud).toBe("signalium-trainer");
  });

  it("rejects a token with an expired exp", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, {
      ...validClaims(),
      exp: Math.floor(Date.now() / 1000) - 1,
    });
    await expect(verifyPlatformJwt(token)).rejects.toMatchObject({ kind: "EXPIRED" });
  });

  it("rejects a token with wrong iss", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, { ...validClaims(), iss: "other-issuer" });
    await expect(verifyPlatformJwt(token)).rejects.toMatchObject({ kind: "WRONG_ISSUER" });
  });

  it("rejects a token with wrong aud", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, { ...validClaims(), aud: "other-service" });
    await expect(verifyPlatformJwt(token)).rejects.toMatchObject({ kind: "WRONG_AUDIENCE" });
  });

  it("accepts token when aud is an array that includes signalium-trainer", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, {
      ...validClaims(),
      aud: ["signalium-trainer", "other"],
    });
    const claims = await verifyPlatformJwt(token);
    expect(claims.sub).toBe("signalium-platform-service");
  });

  it("rejects a token signed by a different key (invalid signature)", async () => {
    _injectJwksForTest([jwkPublic]);
    // Generate a different key pair — token signed by otherKey but JWKS has jwkPublic
    const other = await generateEcKeyPair();
    const token = await buildJwt(other.privateKey, validClaims());
    await expect(verifyPlatformJwt(token)).rejects.toMatchObject({ kind: "INVALID_SIGNATURE" });
  });

  it("rejects a token with an unknown kid when JWKS has no matching key", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, validClaims(), { kid: "nonexistent-kid" });
    await expect(verifyPlatformJwt(token)).rejects.toMatchObject({ kind: "UNKNOWN_KID" });
  });

  it("rejects a malformed token (wrong number of parts)", async () => {
    await expect(verifyPlatformJwt("not.a.valid.jwt.at.all")).rejects.toMatchObject({
      kind: "MALFORMED",
    });
  });

  it("rejects a token with alg !== ES256", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, validClaims(), { alg: "RS256" });
    await expect(verifyPlatformJwt(token)).rejects.toMatchObject({ kind: "UNSUPPORTED_ALG" });
  });

  it("rejects a token with no kid in header", async () => {
    _injectJwksForTest([jwkPublic]);
    const token = await buildJwt(privateKey, validClaims(), { kid: undefined });
    await expect(verifyPlatformJwt(token)).rejects.toMatchObject({ kind: "MALFORMED" });
  });
});
