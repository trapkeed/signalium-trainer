import type { Request, Response, NextFunction } from "express";
import { verifyPlatformJwt, JwtVerifyError } from "../lib/jwt_verify.js";
import { logger } from "../lib/logger.js";

/**
 * Admin-only authentication middleware for Research Lab and QDE lifecycle endpoints.
 *
 * Verifies the incoming request carries a valid JWT signed by the Signalium
 * platform. The platform mints short-lived ES256 JWTs via its own SSO system and
 * presents them as:
 *
 *   Authorization: Bearer <signed-jwt>
 *
 * Verification enforces:
 *   - Algorithm must be ES256
 *   - Signature valid against the platform JWKS (SIGNALIUM_JWKS_URL, with 10-min cache)
 *   - iss === "signalium"
 *   - aud === "signalium-trainer"
 *   - exp is in the future
 *
 * Error responses:
 *   401  No Authorization header / empty Bearer token / expired token
 *   403  Wrong issuer, wrong audience, or invalid/malformed JWT
 *   503  JWKS endpoint unreachable (propagated from jwks_client)
 */
export async function adminAuth(
  req: Request,
  res: Response,
  next: NextFunction,
): Promise<void> {
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

  try {
    await verifyPlatformJwt(token);
    next();
  } catch (err) {
    if (err instanceof JwtVerifyError) {
      const isUnauthenticated =
        err.kind === "MISSING_TOKEN" || err.kind === "EXPIRED";

      logger.warn(
        { url: req.url, kind: err.kind, message: err.message },
        "adminAuth: JWT verification failed",
      );

      if (isUnauthenticated) {
        res.status(401).json({ success: false, error: err.message });
      } else {
        res.status(403).json({ success: false, error: `Forbidden. ${err.message}` });
      }
      return;
    }

    // Unexpected error (e.g. JWKS fetch failure) — surface as 503
    logger.error({ url: req.url, err: String(err) }, "adminAuth: unexpected JWT error");
    res.status(503).json({
      success: false,
      error: `Could not verify JWT: ${String(err)}`,
    });
  }
}
