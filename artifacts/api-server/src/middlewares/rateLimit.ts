import type { Request, Response, NextFunction } from "express";

interface WindowEntry {
  count: number;
  resetAt: number;
}

const store = new Map<string, WindowEntry>();

function getKey(req: Request): string {
  const forwarded = req.headers["x-forwarded-for"];
  const ip = typeof forwarded === "string" ? forwarded.split(",")[0].trim() : req.ip ?? "unknown";
  return ip;
}

/**
 * Simple in-memory sliding-window rate limiter.
 * @param maxRequests — max requests per windowMs
 * @param windowMs    — window size in milliseconds
 */
export function rateLimiter(maxRequests = 120, windowMs = 60_000) {
  return function limit(req: Request, res: Response, next: NextFunction): void {
    const key = getKey(req);
    const now = Date.now();
    const entry = store.get(key);

    if (!entry || now > entry.resetAt) {
      store.set(key, { count: 1, resetAt: now + windowMs });
      next();
      return;
    }

    entry.count += 1;
    if (entry.count > maxRequests) {
      res.setHeader("Retry-After", Math.ceil((entry.resetAt - now) / 1000).toString());
      res.status(429).json({
        error: "Too many requests.",
        retryAfterSeconds: Math.ceil((entry.resetAt - now) / 1000),
      });
      return;
    }
    next();
  };
}

// Periodically evict expired entries to prevent unbounded memory growth
setInterval(() => {
  const now = Date.now();
  for (const [key, entry] of store) {
    if (now > entry.resetAt) store.delete(key);
  }
}, 60_000);
