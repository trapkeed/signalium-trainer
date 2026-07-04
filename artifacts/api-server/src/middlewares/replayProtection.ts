import type { Request, Response, NextFunction } from "express";

/**
 * Replay protection: reject requests where the X-Request-Timestamp header
 * is older than TOLERANCE_MS from the server's current time.
 *
 * Clients must include:
 *   X-Request-Timestamp: <Unix ms>
 *   X-Request-Id: <UUID>
 *
 * Only enforced for mutating methods (POST, PUT, PATCH, DELETE).
 */

const TOLERANCE_MS = 5 * 60 * 1000; // 5 minutes
const seenIds = new Set<string>();

// Evict seen IDs every 10 minutes
setInterval(() => seenIds.clear(), 10 * 60 * 1000);

export function replayProtection(req: Request, res: Response, next: NextFunction): void {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(req.method)) {
    next();
    return;
  }

  const tsHeader = req.headers["x-request-timestamp"];
  if (!tsHeader) {
    next(); // Header not required in development; skip silently
    return;
  }

  const ts = Number(tsHeader);
  if (isNaN(ts) || Math.abs(Date.now() - ts) > TOLERANCE_MS) {
    res.status(400).json({ error: "Request timestamp out of acceptable range." });
    return;
  }

  const requestId = req.headers["x-request-id"];
  if (typeof requestId === "string") {
    if (seenIds.has(requestId)) {
      res.status(409).json({ error: "Duplicate request detected (replay)." });
      return;
    }
    seenIds.add(requestId);
  }

  next();
}
