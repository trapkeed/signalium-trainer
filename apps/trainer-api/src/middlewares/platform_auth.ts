/**
 * Platform authentication middleware for Signalium inbound requests.
 *
 * Every request from the Signalium Platform carries:
 *   Authorization: Bearer <ES256-signed JWT>
 *
 * The optional legacy X-Api-Key header is accepted but not required.
 *
 * On success, the decoded JWT claims are attached to res.locals.platformClaims.
 * On failure, returns 401 { "error": "Unauthorized" } for all JWT error kinds.
 */

import type { Request, Response, NextFunction } from "express";
import { logger } from "../lib/logger.js";
import { verifyPlatformJwt, JwtVerifyError, type JwtClaims } from "../lib/jwt_verify.js";

export type { JwtClaims };

export async function platformAuth(
  req: Request,
  res: Response,
  next: NextFunction,
): Promise<void> {
  const authHeader = req.headers["authorization"] ?? "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice(7).trim() : null;

  if (!token) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }

  try {
    const claims = await verifyPlatformJwt(token);
    // Attach parsed claims for downstream handlers
    res.locals["platformClaims"] = claims;
    next();
  } catch (err) {
    if (err instanceof JwtVerifyError) {
      logger.warn(
        { url: req.url, kind: err.kind, message: err.message },
        "platformAuth: JWT verification failed",
      );
      res.status(401).json({ error: "Unauthorized" });
      return;
    }
    // Unexpected errors propagate to Express error handler
    next(err);
  }
}
