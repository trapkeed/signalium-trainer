import { pgTable, text, timestamp, index } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";
import { modelsTable } from "./models";

export const modelTagsTable = pgTable(
  "model_tags",
  {
    id: text("id").primaryKey().$defaultFn(() => crypto.randomUUID()),
    modelId: text("model_id")
      .notNull()
      .references(() => modelsTable.id, { onDelete: "cascade" }),
    tagKey: text("tag_key").notNull(),
    tagValue: text("tag_value").notNull().default(""),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index("model_tags_model_id_idx").on(t.modelId),
    index("model_tags_key_idx").on(t.tagKey),
  ],
);

export const insertModelTagSchema = createInsertSchema(modelTagsTable).omit({
  id: true,
  createdAt: true,
});
export type InsertModelTag = z.infer<typeof insertModelTagSchema>;
export type ModelTag = typeof modelTagsTable.$inferSelect;
