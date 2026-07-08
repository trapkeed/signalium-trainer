import type { Request, Response, NextFunction } from "express";
import { logger } from "../lib/logger";

/**
 * Admin-only authentication middleware for Research Lab endpoints.
 *
 * Requires one of:
 *   Authorization: Bearer <RESEARCH_LAB_ADMIN_TOKEN>
 *   X-Admin-Token: <RESEARCH_LAB_ADMIN_TOKEN>
 *
 * Set RESEARCH_LAB_ADMIN_TOKEN in the environment.
 * If the env var is not set, all admin requests are rejected (fail-closed).
 */
export function adminAuth(req: Request, res: Response, next: NextFunction): void {
  const configuredToken = process.env["RESEARCH_LAB_ADMIN_TOKEN"];

  if (!configuredToken) {
    logger.warn(
      { url: req.url },
      "adminAuth: RESEARCH_LAB_ADMIN_TOKEN not configured — denying request",
    );
    res.status(503).json({
      success: false,
      error: "Research Lab is not configured. Set RESEARCH_LAB_ADMIN_TOKEN.",
    });
    return;
  }

  // Accept token from Authorization header (Bearer) or X-Admin-Token header
  const authHeader = req.headers["authorization"] ?? "";
  const tokenFromBearer = authHeader.startsWith("Bearer ")
    ? authHeader.slice(7).trim()
    : null;
  const tokenFromHeader = req.headers["x-admin-token"] as string | undefined;

  const providedToken = tokenFromBearer ?? tokenFromHeader ?? null;

  if (!providedToken) {
    logger.warn({ url: req.url }, "adminAuth: no token provided");
    res.status(401).json({
      success: false,
      error:
        "Unauthorized. Provide Authorization: Bearer <token> or X-Admin-Token: <token>.",
    });
    return;
  }

  // Constant-time comparison to prevent timing attacks
  if (!timingSafeEqual(providedToken, configuredToken)) {
    logger.warn({ url: req.url }, "adminAuth: invalid token");
    res.status(403).json({ success: false, error: "Forbidden. Invalid admin token." });
    return;
  }

  next();
}

/**
 * Constant-time string comparison that avoids early exit on mismatch.
 * Not cryptographically perfect in JS but prevents trivial timing leaks.
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    // Still iterate to avoid length-timing side channel
    let diff = 0;
    for (let i = 0; i < a.length; i++) {
      diff |= (a.charCodeAt(i) ^ (b.charCodeAt(i % b.length) || 0));
    }
    return false;
  }
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}
