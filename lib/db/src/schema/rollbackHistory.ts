import { pgTable, text, timestamp, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelsTable } from "./models";
import { modelVersionsTable } from "./modelVersions";

export const rollbackHistoryTable = pgTable(
  "rollback_history",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelId: text("model_id")
      .notNull()
      .references(() => modelsTable.id, { onDelete: "cascade" }),
    fromVersionId: text("from_version_id")
      .notNull()
      .references(() => modelVersionsTable.id),
    toVersionId: text("to_version_id")
      .notNull()
      .references(() => modelVersionsTable.id),
    reason: text("reason").notNull(),
    operator: text("operator").notNull(),
    deploymentDurationSeconds: text("deployment_duration_seconds"),
    rolledBackAt: timestamp("rolled_back_at", { withTimezone: true }).notNull().defaultNow(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("rollback_history_model_id_idx").on(t.modelId),
    index("rollback_history_rolled_back_at_idx").on(t.rolledBackAt),
  ],
);

export const insertRollbackHistorySchema = createInsertSchema(rollbackHistoryTable).omit({
  id: true,
  createdAt: true,
});
export type InsertRollbackHistory = z.infer<typeof insertRollbackHistorySchema>;
export type RollbackHistory = typeof rollbackHistoryTable.$inferSelect;
