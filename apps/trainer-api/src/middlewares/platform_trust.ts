import type { Request, Response, NextFunction } from "express";
import { logger } from "../lib/logger";

/**
 * Platform Registration / Origin Trust.
 *
 * Signalium Trainer must trust management requests originating from the
 * Signalium Platform (https://signaliumai.com) and reject unauthorized
 * management requests from anywhere else.
 *
 * Browser-originated requests carry an `Origin` header — when present, it
 * must match an allowlisted platform origin or the request is rejected
 * outright (403), before adminAuth even runs. Server-to-server requests
 * (the Platform backend calling Trainer directly) do not set `Origin`, so
 * those are allowed through this check and rely on `adminAuth`'s admin
 * token for authorization instead — this middleware only closes the
 * browser-CSRF-style origin gap, it is not a substitute for adminAuth.
 *
 * Configure trusted origins via PLATFORM_ALLOWED_ORIGINS (comma-separated).
 * Defaults to the production Signalium Platform origin plus localhost for
 * local development.
 */

function allowedOrigins(): string[] {
  const configured = process.env["PLATFORM_ALLOWED_ORIGINS"];
  if (configured) {
    return configured.split(",").map((o) => o.trim()).filter(Boolean);
  }
  return [
    "https://signaliumai.com",
    "https://www.signaliumai.com",
    "https://trainer.signaliumai.com",
    "http://localhost:3000",
    "http://localhost:5173",
  ];
}

export function requirePlatformOrigin(req: Request, res: Response, next: NextFunction): void {
  const origin = req.headers["origin"];

  if (!origin) {
    // No Origin header => not a browser cross-origin request (e.g. a direct
    // server-to-server call from the Platform backend). Trust falls through
    // to adminAuth's token check.
    next();
    return;
  }

  const trusted = allowedOrigins();
  if (trusted.includes(origin)) {
    next();
    return;
  }

  logger.warn({ url: req.url, origin }, "requirePlatformOrigin: rejected untrusted origin");
  res.status(403).json({
    success: false,
    error: "Forbidden. This management API only accepts requests from the Signalium Platform.",
  });
}

export function platformCorsOptions(): { origin: (origin: string | undefined, cb: (err: Error | null, allow?: boolean) => void) => void; credentials: boolean } {
  const trusted = allowedOrigins();
  return {
    origin(origin, cb) {
      // Non-browser requests (no Origin header) are always allowed through
      // CORS itself — actual authorization is enforced by adminAuth /
      // requirePlatformOrigin on the specific management routes.
      if (!origin || trusted.includes(origin)) {
        cb(null, true);
        return;
      }
      cb(null, false);
    },
    credentials: true,
  };
}
