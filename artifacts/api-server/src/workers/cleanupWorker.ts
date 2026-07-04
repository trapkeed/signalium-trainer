import { db, modelArtifactsTable } from "@workspace/db";
import { eq, lt } from "drizzle-orm";
import { unlinkSync, existsSync } from "fs";
import { logger } from "../lib/logger.js";

const CLEANUP_INTERVAL_MS = 24 * 60 * 60 * 1000; // once per day
const CORRUPTED_RETENTION_DAYS = 7;

/**
 * Remove corrupted artifacts older than CORRUPTED_RETENTION_DAYS from
 * storage and the database.
 */
async function runCleanup(): Promise<void> {
  const cutoff = new Date(Date.now() - CORRUPTED_RETENTION_DAYS * 86_400_000);
  const stale = await db
    .select()
    .from(modelArtifactsTable)
    .where(eq(modelArtifactsTable.status, "corrupted"));

  let removed = 0;
  for (const artifact of stale) {
    if (artifact.createdAt < cutoff) {
      if (artifact.storagePath && existsSync(artifact.storagePath)) {
        try { unlinkSync(artifact.storagePath); } catch { /* best-effort */ }
      }
      await db.delete(modelArtifactsTable).where(eq(modelArtifactsTable.id, artifact.id));
      removed++;
    }
  }

  if (removed > 0) {
    logger.info({ removed }, "Registry cleanup: removed stale corrupted artifacts");
  }
}

export function startCleanupWorker(): void {
  // Run once on startup, then on interval
  void runCleanup();
  setInterval(() => void runCleanup(), CLEANUP_INTERVAL_MS);
}
