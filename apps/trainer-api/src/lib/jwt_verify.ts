/**
 * ES256 JWT verification for inbound Signalium Platform requests.
 *
 * Validates:
 *   - Algorithm must be ES256
 *   - Signature via ECDSA P-256 / SHA-256 using the platform's JWKS
 *   - iss === "signalium"
 *   - aud === "signalium-trainer"
 *   - exp is in the future
 *
 * Returns the decoded payload on success; throws a typed JwtVerifyError on any failure.
 */

import { webcrypto } from "crypto";
import { getJwk } from "./jwks_client.js";

// JsonWebKey is a DOM/WebCrypto type that is NOT in lib.es2022.
// Node v20+ supports webcrypto at runtime with this shape.
// We declare a local compatible interface so we can call importKey("jwk", ...).
interface JsonWebKey {
  alg?: string;
  crv?: string;
  d?: string;
  dp?: string;
  dq?: string;
  e?: string;
  ext?: boolean;
  k?: string;
  key_ops?: string[];
  kty: string;
  n?: string;
  p?: string;
  q?: string;
  qi?: string;
  use?: string;
  x?: string;
  y?: string;
}

type JwkInput = JsonWebKey & { kid?: string; [key: string]: unknown };

export type JwtClaims = {
  iss: string;
  aud: string | string[];
  sub: string;
  exp: number;
  iat?: number;
  scope?: string;
  [key: string]: unknown;
};

export type JwtVerifyErrorKind =
  | "MISSING_TOKEN"
  | "MALFORMED"
  | "UNSUPPORTED_ALG"
  | "UNKNOWN_KID"
  | "INVALID_SIGNATURE"
  | "EXPIRED"
  | "WRONG_ISSUER"
  | "WRONG_AUDIENCE";

export class JwtVerifyError extends Error {
  constructor(
    public readonly kind: JwtVerifyErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "JwtVerifyError";
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Decode a base64url string into a plain ArrayBuffer. */
function base64urlToBuffer(s: string): ArrayBuffer {
  const base64 = s.replace(/-/g, "+").replace(/_/g, "/");
  const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");
  const buf = Buffer.from(padded, "base64");
  // Produce a plain ArrayBuffer (not SharedArrayBuffer-backed) required by webcrypto.
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function base64urlDecode(s: string): string {
  return Buffer.from(base64urlToBuffer(s)).toString("utf8");
}

/** Encode a string as a plain ArrayBuffer for signing/verifying. */
function stringToBuffer(s: string): ArrayBuffer {
  const encoded = new TextEncoder().encode(s);
  return encoded.buffer.slice(0) as ArrayBuffer;
}

// ---------------------------------------------------------------------------
// Core verifier
// ---------------------------------------------------------------------------

export async function verifyPlatformJwt(token: string): Promise<JwtClaims> {
  const parts = token.split(".");
  if (parts.length !== 3) {
    throw new JwtVerifyError("MALFORMED", "JWT must have three dot-separated parts");
  }
  const [rawHeader, rawPayload, rawSig] = parts as [string, string, string];

  // --- Decode header ---
  let header: { alg?: string; kid?: string };
  try {
    header = JSON.parse(base64urlDecode(rawHeader)) as { alg?: string; kid?: string };
  } catch {
    throw new JwtVerifyError("MALFORMED", "Failed to parse JWT header");
  }

  if (header.alg !== "ES256") {
    throw new JwtVerifyError(
      "UNSUPPORTED_ALG",
      `Expected ES256, got ${header.alg ?? "none"}`,
    );
  }
  if (!header.kid) {
    throw new JwtVerifyError("MALFORMED", "JWT header missing kid");
  }

  // --- Decode payload ---
  let payload: JwtClaims;
  try {
    payload = JSON.parse(base64urlDecode(rawPayload)) as JwtClaims;
  } catch {
    throw new JwtVerifyError("MALFORMED", "Failed to parse JWT payload");
  }

  // --- Fetch public key ---
  let jwk: JwkInput;
  try {
    jwk = (await getJwk(header.kid)) as JwkInput;
  } catch (err) {
    throw new JwtVerifyError(
      "UNKNOWN_KID",
      `Could not resolve signing key: ${(err as Error).message}`,
    );
  }

  // --- Verify signature ---
  const signingInput = `${rawHeader}.${rawPayload}`;
  const sigBuffer = base64urlToBuffer(rawSig);
  const dataBuffer = stringToBuffer(signingInput);

  let valid: boolean;
  try {
    const cryptoKey = await webcrypto.subtle.importKey(
      "jwk",
      jwk,
      { name: "ECDSA", namedCurve: "P-256" },
      false,
      ["verify"],
    );
    valid = await webcrypto.subtle.verify(
      { name: "ECDSA", hash: "SHA-256" },
      cryptoKey,
      sigBuffer,
      dataBuffer,
    );
  } catch (err) {
    throw new JwtVerifyError(
      "INVALID_SIGNATURE",
      `Signature verification error: ${(err as Error).message}`,
    );
  }

  if (!valid) {
    throw new JwtVerifyError("INVALID_SIGNATURE", "JWT signature is invalid");
  }

  // --- Verify claims ---
  const nowSec = Math.floor(Date.now() / 1000);
  // Fail closed: exp must be present and numeric. A token without exp is rejected.
  if (typeof payload.exp !== "number") {
    throw new JwtVerifyError("MALFORMED", "JWT missing or non-numeric exp claim");
  }
  if (payload.exp < nowSec) {
    throw new JwtVerifyError("EXPIRED", "JWT has expired");
  }

  if (payload.iss !== "signalium") {
    throw new JwtVerifyError(
      "WRONG_ISSUER",
      `Expected iss=signalium, got ${payload.iss}`,
    );
  }

  // aud may be a string or array per RFC 7519
  const audValues = Array.isArray(payload.aud) ? payload.aud : [payload.aud];
  if (!audValues.includes("signalium-trainer")) {
    throw new JwtVerifyError(
      "WRONG_AUDIENCE",
      `Expected aud=signalium-trainer, got ${JSON.stringify(payload.aud)}`,
    );
  }

  return payload;
}
