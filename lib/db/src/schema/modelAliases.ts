import { pgTable, text, timestamp, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelsTable } from "./models";
import { modelVersionsTable } from "./modelVersions";

/**
 * Named pointers to specific model versions.
 * Examples: "champion", "production", "challenger", "staging"
 * One alias per (model, alias_name) pair.
 */
export const modelAliasesTable = pgTable(
  "model_aliases",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelId: text("model_id")
      .notNull()
      .references(() => modelsTable.id, { onDelete: "cascade" }),
    modelVersionId: text("model_version_id")
      .notNull()
      .references(() => modelVersionsTable.id, { onDelete: "cascade" }),
    aliasName: text("alias_name").notNull(),
    setBy: text("set_by").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("model_aliases_model_id_idx").on(t.modelId),
    index("model_aliases_alias_name_idx").on(t.aliasName),
  ],
);

export const insertModelAliasSchema = createInsertSchema(modelAliasesTable).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});
export type InsertModelAlias = z.infer<typeof insertModelAliasSchema>;
export type ModelAlias = typeof modelAliasesTable.$inferSelect;
