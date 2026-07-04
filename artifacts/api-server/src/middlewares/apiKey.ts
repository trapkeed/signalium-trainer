import type { Request, Response, NextFunction } from "express";

const REGISTRY_API_KEY = process.env.REGISTRY_API_KEY ?? "";
const REGISTRY_ADMIN_API_KEY = process.env.REGISTRY_ADMIN_API_KEY ?? "";

function extractKey(req: Request): string | undefined {
  const header = req.headers["x-api-key"];
  if (typeof header === "string" && header.length > 0) return header;
  const auth = req.headers["authorization"];
  if (typeof auth === "string" && auth.startsWith("Bearer ")) {
    return auth.slice(7);
  }
  return undefined;
}

/**
 * Require a valid API key (read or admin).
 * Set REGISTRY_API_KEY or REGISTRY_ADMIN_API_KEY env vars.
 * If both are empty the server runs in open mode (dev only).
 */
export function requireApiKey(req: Request, res: Response, next: NextFunction): void {
  if (!REGISTRY_API_KEY && !REGISTRY_ADMIN_API_KEY) {
    // No keys configured — open mode (development)
    next();
    return;
  }
  const key = extractKey(req);
  if (!key || (key !== REGISTRY_API_KEY && key !== REGISTRY_ADMIN_API_KEY)) {
    res.status(401).json({ error: "Invalid or missing API key." });
    return;
  }
  next();
}

/**
 * Require the admin API key specifically.
 * Used for write/promote/rollback operations.
 */
export function requireAdminApiKey(req: Request, res: Response, next: NextFunction): void {
  if (!REGISTRY_ADMIN_API_KEY) {
    // No admin key configured — open mode (development)
    next();
    return;
  }
  const key = extractKey(req);
  if (!key || key !== REGISTRY_ADMIN_API_KEY) {
    res.status(403).json({ error: "Admin API key required for this operation." });
    return;
  }
  next();
}

/** Attach the actor identity to res.locals for use in services. */
export function attachActor(req: Request, res: Response, next: NextFunction): void {
  const key = extractKey(req);
  if (key && key === REGISTRY_ADMIN_API_KEY) {
    res.locals.actor = "admin";
  } else if (key && key === REGISTRY_API_KEY) {
    res.locals.actor = "api-user";
  } else {
    res.locals.actor = "anonymous";
  }
  next();
}
