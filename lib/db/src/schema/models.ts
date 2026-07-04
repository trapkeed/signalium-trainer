import { pgTable, text, timestamp, boolean, index } from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const modelsTable = pgTable(
  "models",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    name: text("name").notNull().unique(),
    displayName: text("display_name").notNull(),
    description: text("description").notNull().default(""),
    architecture: text("architecture").notNull(),
    framework: text("framework").notNull().default("pytorch"),
    createdBy: text("created_by").notNull(),
    isDeleted: boolean("is_deleted").notNull().default(false),
    deletedAt: timestamp("deleted_at", { withTimezone: true }),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("models_name_idx").on(t.name),
    index("models_architecture_idx").on(t.architecture),
    index("models_created_at_idx").on(t.createdAt),
  ],
);

export const insertModelSchema = createInsertSchema(modelsTable).omit({
  id: true,
  isDeleted: true,
  deletedAt: true,
  createdAt: true,
  updatedAt: true,
});
export const selectModelSchema = createSelectSchema(modelsTable);

export type InsertModel = z.infer<typeof insertModelSchema>;
export type Model = typeof modelsTable.$inferSelect;
