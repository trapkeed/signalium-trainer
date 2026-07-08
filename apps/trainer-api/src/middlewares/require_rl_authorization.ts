import type { Request, Response, NextFunction } from "express";
import { logger } from "../lib/logger";

/**
 * Enforces that Research Lab is the SOLE authority for QDE version
 * lifecycle mutations: PROMOTE, ROLLBACK, ARCHIVE, RETIRE.
 *
 * Before this middleware's route handler is allowed to execute a lifecycle
 * mutation against the QDE registry (port 8001), it must first present a
 * matching APPROVED, unconsumed LifecycleAuthorization issued by Research
 * Lab (port 8002). The authorization is atomically marked CONSUMED as part
 * of this check, so it can never be reused for a second mutation.
 *
 * This closes the gap where `apps/trainer-api/src/routes/qde.ts` could
 * previously execute promotion/rollback directly against qde_server.py
 * with no Research Lab sign-off at all.
 *
 * Usage: apply AFTER `adminAuth` and provide a function that extracts the
 * (versionId, action) pair to check from the request.
 */

const RL_BASE = process.env["RESEARCH_LAB_API_URL"] || "http://localhost:8002";

export type LifecycleAction = "PROMOTE" | "ROLLBACK" | "ARCHIVE" | "RETIRE";

export function requireResearchLabAuthorization(
  action: LifecycleAction,
  extractVersionId: (req: Request) => string | undefined | Promise<string | undefined>,
) {
  return async function (req: Request, res: Response, next: NextFunction): Promise<void> {
    const versionId = await extractVersionId(req);
    if (!versionId) {
      res.status(400).json({
        success: false,
        error: `Could not determine target version_id for ${action} authorization check`,
      });
      return;
    }

    const configuredToken = process.env["RESEARCH_LAB_ADMIN_TOKEN"];
    if (!configuredToken) {
      logger.warn(
        { url: req.url, action, versionId },
        "requireResearchLabAuthorization: RESEARCH_LAB_ADMIN_TOKEN not configured — denying",
      );
      res.status(503).json({
        success: false,
        error: "Research Lab is not configured. Set RESEARCH_LAB_ADMIN_TOKEN.",
      });
      return;
    }

    try {
      const upstream = await fetch(`${RL_BASE}/research/lifecycle/consume`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${configuredToken}`,
        },
        body: JSON.stringify({ version_id: versionId, action }),
      });
      const data = (await upstream.json()) as { success?: boolean; error?: string };

      if (!upstream.ok || !data.success) {
        logger.warn(
          { url: req.url, action, versionId, error: data.error },
          "requireResearchLabAuthorization: no approved Research Lab authorization — denying",
        );
        res.status(403).json({
          success: false,
          error:
            data.error ||
            `Research Lab has not authorized ${action} for version ${versionId}. ` +
              "This action must be approved in Research Lab before it can be executed.",
        });
        return;
      }

      logger.info(
        { url: req.url, action, versionId },
        "requireResearchLabAuthorization: authorization consumed, proceeding",
      );
      next();
    } catch (err) {
      logger.error(
        { url: req.url, action, versionId, err: String(err) },
        "requireResearchLabAuthorization: failed to reach Research Lab",
      );
      res.status(502).json({
        success: false,
        error: `Could not verify Research Lab authorization: ${String(err)}`,
      });
    }
  };
}
