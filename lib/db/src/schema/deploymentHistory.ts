import { pgTable, text, timestamp, index, pgEnum } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelsTable } from "./models";
import { modelVersionsTable } from "./modelVersions";

export const deploymentEnvironmentEnum = pgEnum("deployment_environment", [
  "staging",
  "production",
  "shadow",
]);

export const deploymentStatusEnum = pgEnum("deployment_status", [
  "active",
  "rolled_back",
  "retired",
]);

export const deploymentHistoryTable = pgTable(
  "deployment_history",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelId: text("model_id")
      .notNull()
      .references(() => modelsTable.id, { onDelete: "cascade" }),
    modelVersionId: text("model_version_id")
      .notNull()
      .references(() => modelVersionsTable.id, { onDelete: "cascade" }),
    environment: deploymentEnvironmentEnum("environment").notNull().default("production"),
    status: deploymentStatusEnum("status").notNull().default("active"),
    deployedBy: text("deployed_by").notNull(),
    deployedAt: timestamp("deployed_at", { withTimezone: true }).notNull().defaultNow(),
    retiredAt: timestamp("retired_at", { withTimezone: true }),
    notes: text("notes").notNull().default(""),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("deployment_history_model_id_idx").on(t.modelId),
    index("deployment_history_environment_idx").on(t.environment),
    index("deployment_history_status_idx").on(t.status),
    index("deployment_history_deployed_at_idx").on(t.deployedAt),
  ],
);

export const insertDeploymentHistorySchema = createInsertSchema(deploymentHistoryTable).omit({
  id: true,
  createdAt: true,
});
export type InsertDeploymentHistory = z.infer<typeof insertDeploymentHistorySchema>;
export type DeploymentHistory = typeof deploymentHistoryTable.$inferSelect;
