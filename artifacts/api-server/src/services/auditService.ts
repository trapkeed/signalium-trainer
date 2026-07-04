import { db, modelAuditLogTable, type InsertAuditLog } from "@workspace/db";
import type { Request } from "express";

export async function writeAuditLog(
  entry: Omit<InsertAuditLog, "ipAddress" | "userAgent">,
  req?: Request,
): Promise<void> {
  try {
    await db.insert(modelAuditLogTable).values({
      ...entry,
      ipAddress: req
        ? (req.headers["x-forwarded-for"] as string | undefined)?.split(",")[0].trim() ?? req.ip
        : undefined,
      userAgent: req ? (req.headers["user-agent"] ?? undefined) : undefined,
    });
  } catch {
    // Audit log failures must never crash the main request
  }
}
