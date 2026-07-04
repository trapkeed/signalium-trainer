import { pgTable, text, timestamp, index, jsonb, pgEnum } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const auditEventTypeEnum = pgEnum("audit_event_type", [
  "model_created",
  "version_created",
  "artifact_uploaded",
  "artifact_downloaded",
  "validation_started",
  "validation_completed",
  "validation_failed",
  "status_changed",
  "promoted",
  "rejected",
  "archived",
  "rolled_back",
  "alias_set",
  "tag_added",
  "tag_removed",
  "model_deleted",
]);

export const modelAuditLogTable = pgTable(
  "model_audit_log",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    eventType: auditEventTypeEnum("event_type").notNull(),
    modelId: text("model_id"),
    modelVersionId: text("model_version_id"),
    actor: text("actor").notNull(),
    fromStatus: text("from_status"),
    toStatus: text("to_status"),
    details: jsonb("details"),
    ipAddress: text("ip_address"),
    userAgent: text("user_agent"),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("model_audit_log_model_id_idx").on(t.modelId),
    index("model_audit_log_version_id_idx").on(t.modelVersionId),
    index("model_audit_log_event_type_idx").on(t.eventType),
    index("model_audit_log_actor_idx").on(t.actor),
    index("model_audit_log_created_at_idx").on(t.createdAt),
  ],
);

export const insertAuditLogSchema = createInsertSchema(modelAuditLogTable).omit({
  id: true,
  createdAt: true,
});
export type InsertAuditLog = z.infer<typeof insertAuditLogSchema>;
export type AuditLog = typeof modelAuditLogTable.$inferSelect;
