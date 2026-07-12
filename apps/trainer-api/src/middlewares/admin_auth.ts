import type { Request, Response, NextFunction } from "express";
import { verifyPlatformJwt, JwtVerifyError } from "../lib/jwt_verify.js";
import { logger } from "../lib/logger.js";

/**
 * Admin-only authentication middleware for Research Lab and QDE lifecycle endpoints.
 *
 * Accepts one of two credential forms:
 *
 *   1. Static admin token (RESEARCH_LAB_ADMIN_TOKEN env var).
 *      Used by the Research Lab admin dashboard — the user enters the token
 *      directly in the login form. No JWT required.
 *
 *   2. Platform JWT signed by Signalium (ES256, SIGNALIUM_JWKS_URL).
 *      Used by the Signalium Platform backend calling Trainer server-to-server.
 *      Verification enforces iss === "signalium", aud === "signalium-trainer",
 *      valid signature, and non-expired exp.
 *
 * The static token is checked first (fast path). If the Bearer value does not
 * match the configured static token, JWT verification is attempted. This means
 * the dashboard always works with the shared secret while platform integrations
 * continue to use short-lived JWTs.
 *
 * Error responses:
 *   401  No Authorization header / empty Bearer / expired JWT
 *   403  Wrong issuer, wrong audience, invalid JWT signature, or wrong static token
 *   503  JWKS endpoint unreachable
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
      error: "Unauthorized. Provide Authorization: Bearer <admin-token-or-platform-jwt>.",
    });
    return;
  }

  // ── Fast path: static admin token ────────────────────────────────────────────
  const staticToken = process.env["RESEARCH_LAB_ADMIN_TOKEN"];
  if (staticToken && token === staticToken) {
    logger.debug({ url: req.url }, "adminAuth: authenticated via static admin token");
    next();
    return;
  }

  // ── Slow path: platform JWT ───────────────────────────────────────────────────
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
