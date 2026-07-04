import { pgTable, text, timestamp, pgEnum, index, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelsTable } from "./models";

export const modelLifecycleStatusEnum = pgEnum("model_lifecycle_status", [
  "uploading",
  "uploaded",
  "validating",
  "validated",
  "candidate",
  "challenger",
  "champion",
  "production",
  "retired",
  "archived",
  "rejected",
]);

export const modelVersionsTable = pgTable(
  "model_versions",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelId: text("model_id")
      .notNull()
      .references(() => modelsTable.id, { onDelete: "cascade" }),
    version: text("version").notNull(),
    status: modelLifecycleStatusEnum("status").notNull().default("uploading"),
    architecture: text("architecture").notNull(),
    trainingDate: timestamp("training_date", { withTimezone: true }),
    datasetVersion: text("dataset_version").notNull().default(""),
    featureVersion: text("feature_version").notNull().default(""),
    labelVersion: text("label_version").notNull().default(""),
    frameworkVersion: text("framework_version").notNull().default(""),
    trainerVersion: text("trainer_version").notNull().default(""),
    configVersion: text("config_version").notNull().default(""),
    experimentId: text("experiment_id"),
    gitCommitHash: text("git_commit_hash"),
    mlflowRunId: text("mlflow_run_id"),
    inputSchema: jsonb("input_schema"),
    outputSchema: jsonb("output_schema"),
    configSnapshot: jsonb("config_snapshot"),
    rejectionReason: text("rejection_reason"),
    promotedBy: text("promoted_by"),
    promotedAt: timestamp("promoted_at", { withTimezone: true }),
    createdBy: text("created_by").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("model_versions_model_id_idx").on(t.modelId),
    index("model_versions_status_idx").on(t.status),
    index("model_versions_dataset_version_idx").on(t.datasetVersion),
    index("model_versions_feature_version_idx").on(t.featureVersion),
    index("model_versions_git_commit_idx").on(t.gitCommitHash),
    index("model_versions_experiment_id_idx").on(t.experimentId),
    index("model_versions_created_at_idx").on(t.createdAt),
  ],
);

export const insertModelVersionSchema = createInsertSchema(modelVersionsTable).omit({
  id: true,
  status: true,
  promotedBy: true,
  promotedAt: true,
  rejectionReason: true,
  createdAt: true,
  updatedAt: true,
});
export const selectModelVersionSchema = createSelectSchema(modelVersionsTable);

export type InsertModelVersion = z.infer<typeof insertModelVersionSchema>;
export type ModelVersion = typeof modelVersionsTable.$inferSelect;
export type ModelLifecycleStatus = typeof modelLifecycleStatusEnum.enumValues[number];
