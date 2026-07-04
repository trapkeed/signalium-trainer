import { pgTable, text, timestamp, pgEnum, index, bigint } from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelVersionsTable } from "./modelVersions";

export const artifactTypeEnum = pgEnum("artifact_type", [
  "torchscript",
  "onnx",
  "manifest",
  "metadata",
  "training_report",
  "evaluation_report",
  "logs",
  "config",
  "other",
]);

export const artifactStatusEnum = pgEnum("artifact_status", [
  "pending",
  "uploaded",
  "verified",
  "corrupted",
]);

export const modelArtifactsTable = pgTable(
  "model_artifacts",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelVersionId: text("model_version_id")
      .notNull()
      .references(() => modelVersionsTable.id, { onDelete: "cascade" }),
    artifactType: artifactTypeEnum("artifact_type").notNull(),
    filename: text("filename").notNull(),
    storagePath: text("storage_path").notNull(),
    mimeType: text("mime_type").notNull().default("application/octet-stream"),
    sizeBytes: bigint("size_bytes", { mode: "number" }).notNull().default(0),
    checksum: text("checksum"),
    checksumAlgorithm: text("checksum_algorithm").notNull().default("sha256"),
    status: artifactStatusEnum("status").notNull().default("pending"),
    uploadedBy: text("uploaded_by").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("model_artifacts_version_id_idx").on(t.modelVersionId),
    index("model_artifacts_type_idx").on(t.artifactType),
    index("model_artifacts_status_idx").on(t.status),
  ],
);

export const insertModelArtifactSchema = createInsertSchema(modelArtifactsTable).omit({
  id: true,
  status: true,
  createdAt: true,
  updatedAt: true,
});
export const selectModelArtifactSchema = createSelectSchema(modelArtifactsTable);

export type InsertModelArtifact = z.infer<typeof insertModelArtifactSchema>;
export type ModelArtifact = typeof modelArtifactsTable.$inferSelect;
export type ArtifactType = typeof artifactTypeEnum.enumValues[number];
