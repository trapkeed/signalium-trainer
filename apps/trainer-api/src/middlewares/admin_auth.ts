import type { Request, Response, NextFunction } from "express";
import { createRemoteJWKSet, jwtVerify, errors as joseErrors } from "jose";
import { logger } from "../lib/logger";

/**
 * Admin-only authentication middleware for Research Lab and QDE lifecycle endpoints.
 *
 * Verifies the incoming request carries a valid JWT signed by the Signalium
 * platform. The platform mints short-lived JWTs via its own SSO system and
 * presents them as:
 *
 *   Authorization: Bearer <signed-jwt>
 *
 * The public key is fetched once from SIGNALIUM_JWKS_URL and cached
 * indefinitely in-process (jose rotates the key automatically on verification
 * failure, so key roll-overs are handled transparently).
 *
 * Required env var:
 *   SIGNALIUM_JWKS_URL  — JWKS endpoint published by the platform
 *                         e.g. https://signaliumai.com/sso/jwks
 *
 * Error responses:
 *   503  SIGNALIUM_JWKS_URL not configured
 *   401  No Authorization header / empty Bearer token
 *   403  JWT invalid, expired, or signature mismatch
 */

// ---------------------------------------------------------------------------
// JWKS cache — one remote keyset per process lifetime
// ---------------------------------------------------------------------------

let _jwks: ReturnType<typeof createRemoteJWKSet> | null = null;

function getJwks(): ReturnType<typeof createRemoteJWKSet> | null {
  const url = process.env["SIGNALIUM_JWKS_URL"];
  if (!url) return null;
  if (!_jwks) {
    _jwks = createRemoteJWKSet(new URL(url));
  }
  return _jwks;
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

export async function adminAuth(
  req: Request,
  res: Response,
  next: NextFunction,
): Promise<void> {
  const jwks = getJwks();

  if (!jwks) {
    logger.warn(
      { url: req.url },
      "adminAuth: SIGNALIUM_JWKS_URL not configured — denying request",
    );
    res.status(503).json({
      success: false,
      error: "Trainer is not configured. Set SIGNALIUM_JWKS_URL.",
    });
    return;
  }

  // Extract Bearer token
  const authHeader = req.headers["authorization"] ?? "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7).trim() : null;

  if (!token) {
    logger.warn({ url: req.url }, "adminAuth: no Bearer token provided");
    res.status(401).json({
      success: false,
      error: "Unauthorized. Provide Authorization: Bearer <platform-jwt>.",
    });
    return;
  }

  // Verify signature + expiry against the platform JWKS
  try {
    await jwtVerify(token, jwks);
    next();
  } catch (err) {
    if (err instanceof joseErrors.JWTExpired) {
      logger.warn({ url: req.url }, "adminAuth: JWT expired");
      res.status(403).json({ success: false, error: "Forbidden. JWT has expired." });
    } else if (
      err instanceof joseErrors.JWSInvalid ||
      err instanceof joseErrors.JWSSignatureVerificationFailed ||
      err instanceof joseErrors.JWTClaimValidationFailed
    ) {
      logger.warn({ url: req.url, err: String(err) }, "adminAuth: JWT validation failed");
      res.status(403).json({ success: false, error: "Forbidden. Invalid JWT." });
    } else {
      // JWKS fetch failure or unexpected error — treat as misconfiguration
      logger.error({ url: req.url, err: String(err) }, "adminAuth: JWKS verification error");
      res.status(503).json({
        success: false,
        error: "Could not verify JWT. JWKS endpoint may be unreachable.",
      });
    }
  }
}
