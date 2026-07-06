import { type Request, type Response, type NextFunction } from "express";
import { createHash, timingSafeEqual } from "node:crypto";
import { logger } from "../lib/logger";

function safeCompare(a: string, b: string): boolean {
  const ha = createHash("sha256").update(a).digest();
  const hb = createHash("sha256").update(b).digest();
  return timingSafeEqual(ha, hb);
}

export function requireApiKey(
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  const configuredKey = process.env.API_KEY;

  if (!configuredKey) {
    logger.error("API_KEY env var is not set — rejecting all authenticated requests");
    res.status(503).json({
      status: "error",
      code: "SERVICE_UNAVAILABLE",
      message: "Authentication is not configured on this server",
    });
    return;
  }

  const authHeader = req.headers.authorization;
  const apiKeyHeader = req.headers["x-api-key"];

  let incomingKey: string | undefined;

  if (typeof authHeader === "string" && authHeader.startsWith("Bearer ")) {
    incomingKey = authHeader.slice(7).trim();
  } else if (typeof apiKeyHeader === "string" && apiKeyHeader.trim().length > 0) {
    incomingKey = apiKeyHeader.trim();
  }

  if (!incomingKey || !safeCompare(incomingKey, configuredKey)) {
    logger.warn({ ip: req.ip, method: req.method, path: req.path }, "Unauthorized request");
    res.status(401).json({
      status: "error",
      code: "UNAUTHORIZED",
      message:
        "Valid API key required. Supply it via 'Authorization: Bearer <key>' or 'x-api-key: <key>'",
    });
    return;
  }

  next();
}
