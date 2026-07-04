import { pgTable, text, timestamp, boolean, jsonb, index, pgEnum } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelVersionsTable } from "./modelVersions";

export const validationStatusEnum = pgEnum("validation_status", [
  "pending",
  "running",
  "passed",
  "failed",
  "skipped",
]);

export const modelValidationTable = pgTable(
  "model_validation",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelVersionId: text("model_version_id")
      .notNull()
      .references(() => modelVersionsTable.id, { onDelete: "cascade" }),
    status: validationStatusEnum("status").notNull().default("pending"),
    checksumVerified: boolean("checksum_verified").notNull().default(false),
    manifestValid: boolean("manifest_valid").notNull().default(false),
    metadataValid: boolean("metadata_valid").notNull().default(false),
    requiredArtifactsPresent: boolean("required_artifacts_present").notNull().default(false),
    inputSchemaValid: boolean("input_schema_valid").notNull().default(false),
    outputSchemaValid: boolean("output_schema_valid").notNull().default(false),
    frameworkVersionOk: boolean("framework_version_ok").notNull().default(false),
    datasetVersionOk: boolean("dataset_version_ok").notNull().default(false),
    artifactIntegrityOk: boolean("artifact_integrity_ok").notNull().default(false),
    checklist: jsonb("checklist"),
    validationReport: jsonb("validation_report"),
    failureReasons: jsonb("failure_reasons"),
    validatedBy: text("validated_by"),
    validatedAt: timestamp("validated_at", { withTimezone: true }),
    durationMs: text("duration_ms"),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("model_validation_version_id_idx").on(t.modelVersionId),
    index("model_validation_status_idx").on(t.status),
  ],
);

export const insertModelValidationSchema = createInsertSchema(modelValidationTable).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});
export type InsertModelValidation = z.infer<typeof insertModelValidationSchema>;
export type ModelValidation = typeof modelValidationTable.$inferSelect;
