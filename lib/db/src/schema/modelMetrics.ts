import { pgTable, text, timestamp, real, index, pgEnum } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelVersionsTable } from "./modelVersions";

export const metricCategoryEnum = pgEnum("metric_category", [
  "training",
  "validation",
  "evaluation",
  "inference",
  "operational",
]);

export const modelMetricsTable = pgTable(
  "model_metrics",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelVersionId: text("model_version_id")
      .notNull()
      .references(() => modelVersionsTable.id, { onDelete: "cascade" }),
    category: metricCategoryEnum("category").notNull().default("evaluation"),
    metricKey: text("metric_key").notNull(),
    metricValue: real("metric_value").notNull(),
    unit: text("unit").notNull().default(""),
    recordedAt: timestamp("recorded_at", { withTimezone: true }).notNull().defaultNow(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("model_metrics_version_id_idx").on(t.modelVersionId),
    index("model_metrics_category_idx").on(t.category),
    index("model_metrics_key_idx").on(t.metricKey),
  ],
);

export const insertModelMetricSchema = createInsertSchema(modelMetricsTable).omit({
  id: true,
  createdAt: true,
});
export type InsertModelMetric = z.infer<typeof insertModelMetricSchema>;
export type ModelMetric = typeof modelMetricsTable.$inferSelect;
